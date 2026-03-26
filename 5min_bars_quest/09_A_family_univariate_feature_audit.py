# Family A univariate feature audit against frozen retained-episode side labels.
#
# Purpose
# This cell evaluates Family A side-native features one at a time as simple
# ranking signals for retained long and short episode entries.
#
# What it does
# 1. Rebuilds binary side labels on the native df5 / df_feat row index:
#    - y_long  = 1 at retained long-entry bars, else 0
#    - y_short = 1 at retained short-entry bars, else 0
# 2. Creates a fixed train / test split by timestamp.
# 3. Selects exact Family A side-native columns:
#    - long audit uses only Family A up-side columns
#    - short audit uses only Family A down-side columns
# 4. Runs a univariate audit for each feature separately.
# 5. Orients each feature on train only:
#    - keep x if train AUC(x) >= train AUC(-x)
#    - flip to -x otherwise
# 6. Reports train AUC, test AUC, test class-mean separation, and
#    top-decile hit-rate / lift for each feature.
#
# What it does not do
# - It does not train a multivariate model.
# - It does not modify df5, df_feat, or episodes_df.
# - It does not rebalance classes or resample rows.
#
# Main outputs
# - famA_long_audit  : long-side univariate audit table
# - famA_short_audit : short-side univariate audit table
# - A_long_uni       : convenience copy of famA_long_audit
# - A_short_uni      : convenience copy of famA_short_audit

import numpy as np
import pandas as pd

# Require the source bars, research features, and retained episode table.
# The audit depends on the original timestamp index from df5, the feature
# columns in df_feat, and the retained side labels implied by episodes_df.
if "df5" not in globals():
    raise NameError("df5 is not defined")
if "df_feat" not in globals():
    raise NameError("df_feat is not defined")
if "episodes_df" not in globals():
    raise NameError("episodes_df is not defined")

# -----------------------------
# Build frozen side-specific labels on the native df5 / df_feat index.
#
# episodes_df stores retained entry timestamps and side.
# Map those entry timestamps back to df5 row positions, then create:
# - y_long  = 1 only for retained long-entry bars
# - y_short = 1 only for retained short-entry bars
#
# These labels are then attached to a copy of df_feat so the audit can work
# directly in feature space without altering the original research dataframe.
# -----------------------------
ts = pd.to_datetime(df5["timestamp"], utc=True, errors="raise")
ts_index = pd.Index(ts)

entry_t = pd.to_datetime(episodes_df["entry_t"], utc=True, errors="raise")
entry_i = ts_index.get_indexer(entry_t)
if (entry_i < 0).any():
    raise ValueError("Some episode entry_t values could not be mapped to df5")

y_long = np.zeros(len(df5), dtype=np.int8)
y_short = np.zeros(len(df5), dtype=np.int8)

ep_side = episodes_df["side"].astype(str).to_numpy()
y_long[entry_i[ep_side == "long"]] = 1
y_short[entry_i[ep_side == "short"]] = 1

df_audit = df_feat.copy()
df_audit["y_long"] = y_long
df_audit["y_short"] = y_short

# -----------------------------
# Fixed chronological train / test split.
#
# This is a pure time split. Nothing after TRAIN_END is allowed into train.
# The printout makes the class balance visible before any feature scoring.
# That matters because some features may have coverage gaps after validity
# filtering, and class counts can become thin on one side.
# -----------------------------
TRAIN_END = pd.Timestamp("2024-06-30 23:59:59+00:00")
train_mask_all = df_audit["timestamp"] <= TRAIN_END
test_mask_all  = df_audit["timestamp"] > TRAIN_END

print("train rows:", int(train_mask_all.sum()))
print("test rows :", int(test_mask_all.sum()))
print("long positives train/test:", int(df_audit.loc[train_mask_all, "y_long"].sum()), int(df_audit.loc[test_mask_all, "y_long"].sum()))
print("short positives train/test:", int(df_audit.loc[train_mask_all, "y_short"].sum()), int(df_audit.loc[test_mask_all, "y_short"].sum()))

# -----------------------------
# Exact Family A side-native feature selection.
#
# Long audit:
#   include only Family A columns native to the up-side view.
#
# Short audit:
#   include only Family A columns native to the down-side view.
#
# This intentionally excludes cross-side mixing. The purpose here is to audit
# the native Family A signal stack for each side separately.
# -----------------------------
famA_long_cols = sorted([
    c for c in df_audit.columns
    if c.startswith("famA_")
    and (
        c.endswith("_up")
        or "_up_" in c
    )
])

famA_short_cols = sorted([
    c for c in df_audit.columns
    if c.startswith("famA_")
    and (
        c.endswith("_dn")
        or "_dn_" in c
    )
])

print("Family A long-native cols :", len(famA_long_cols))
print("Family A short-native cols:", len(famA_short_cols))

# -----------------------------
# Scoring helpers.
#
# safe_auc
#   Computes rank-based ROC AUC without sklearn.
#   Returns NaN when there are no valid rows, or when only one class is present.
#
# top_decile_stats
#   Sorts by descending score and evaluates the highest-scoring 10 percent.
#   Returns:
#   - top-decile hit rate
#   - top-decile lift versus base rate
#   - top-decile row count
#   - base positive rate
#
# Both helpers first remove non-finite values so feature validity masks are
# respected naturally.
# -----------------------------
def safe_auc(y_true, score):
    y_true = np.asarray(y_true, dtype=np.int8)
    score = np.asarray(score, dtype=float)

    ok = np.isfinite(score) & np.isfinite(y_true)
    y_true = y_true[ok]
    score = score[ok]

    if len(y_true) == 0:
        return np.nan

    n_pos = int((y_true == 1).sum())
    n_neg = int((y_true == 0).sum())
    if n_pos == 0 or n_neg == 0:
        return np.nan

    ranks = pd.Series(score).rank(method="average").to_numpy(dtype=float)
    sum_ranks_pos = ranks[y_true == 1].sum()
    auc = (sum_ranks_pos - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg)
    return float(auc)

def top_decile_stats(y_true, score):
    y_true = np.asarray(y_true, dtype=np.int8)
    score = np.asarray(score, dtype=float)

    ok = np.isfinite(score) & np.isfinite(y_true)
    y_true = y_true[ok]
    score = score[ok]

    n = len(y_true)
    if n == 0:
        return np.nan, np.nan, 0, np.nan

    k = max(1, int(np.ceil(0.10 * n)))
    order = np.argsort(-score, kind="mergesort")
    top_idx = order[:k]

    base_rate = float(y_true.mean()) if n > 0 else np.nan
    top_rate = float(y_true[top_idx].mean()) if k > 0 else np.nan
    lift = np.nan if (not np.isfinite(base_rate) or base_rate <= 0.0) else float(top_rate / base_rate)

    return top_rate, lift, int(k), base_rate

# Core per-side univariate audit.
#
# For each feature:
# 1. keep only rows where that feature is non-null
# 2. split those valid rows into train and test by timestamp mask
# 3. require both train and test to have at least one positive example
# 4. compute train AUC for x and for -x
# 5. choose orientation using train only
# 6. evaluate the oriented feature on test
# 7. collect summary metrics into one output row
#
# Stored metrics include:
# - coverage counts and positive counts
# - raw and oriented train AUC
# - test AUC
# - test positive and negative class means
# - test mean gap
# - top-decile hit rate and lift on test
# - chosen orientation (+1 or -1)
def audit_one_side(feature_cols, y_col, side_name):
    rows = []

    for col in feature_cols:
        x = pd.to_numeric(df_audit[col], errors="coerce")
        y = df_audit[y_col].astype(np.int8)

        valid = x.notna().to_numpy()
        train_mask = valid & train_mask_all.to_numpy()
        test_mask = valid & test_mask_all.to_numpy()

        n_train = int(train_mask.sum())
        n_test = int(test_mask.sum())

        pos_train = int(y.loc[train_mask].sum())
        pos_test = int(y.loc[test_mask].sum())

        # If either split has no usable rows or no positive class, keep the
        # feature in the table but mark scoring outputs as NaN.
        if n_train == 0 or n_test == 0 or pos_train == 0 or pos_test == 0:
            rows.append({
                "side_audit": side_name,
                "feature": col,
                "n_train": n_train,
                "pos_train": pos_train,
                "n_test": n_test,
                "pos_test": pos_test,
                "train_auc_raw": np.nan,
                "train_auc_oriented": np.nan,
                "test_auc": np.nan,
                "test_pos_mean": np.nan,
                "test_neg_mean": np.nan,
                "test_mean_gap": np.nan,
                "test_top_decile_hit_rate": np.nan,
                "test_top_decile_lift": np.nan,
                "test_top_decile_n": 0,
                "test_base_rate": np.nan,
                "orientation": np.nan,
            })
            continue

        x_train = x.loc[train_mask].to_numpy(dtype=float)
        y_train = y.loc[train_mask].to_numpy(dtype=np.int8)
        x_test = x.loc[test_mask].to_numpy(dtype=float)
        y_test = y.loc[test_mask].to_numpy(dtype=np.int8)

        # Train-time orientation:
        # compare the feature as-is versus its negation.
        # Keep whichever direction gives better train AUC.
        auc_train_raw = safe_auc(y_train, x_train)
        auc_train_neg = safe_auc(y_train, -x_train)

        if np.isnan(auc_train_raw) and np.isnan(auc_train_neg):
            orient = np.nan
            x_test_or = x_test.copy()
            auc_train_or = np.nan
        elif np.isnan(auc_train_neg) or (np.isfinite(auc_train_raw) and auc_train_raw >= auc_train_neg):
            orient = 1.0
            x_test_or = x_test
            auc_train_or = auc_train_raw
        else:
            orient = -1.0
            x_test_or = -x_test
            auc_train_or = auc_train_neg

        # Evaluate the chosen orientation on held-out test rows only.
        auc_test = safe_auc(y_test, x_test_or)

        # Class-conditional means on test provide an interpretable directional
        # separation view in addition to AUC.
        pos_mask_test = (y_test == 1)
        neg_mask_test = (y_test == 0)

        pos_mean_test = float(np.mean(x_test_or[pos_mask_test])) if pos_mask_test.any() else np.nan
        neg_mean_test = float(np.mean(x_test_or[neg_mask_test])) if neg_mask_test.any() else np.nan
        mean_gap_test = pos_mean_test - neg_mean_test if np.isfinite(pos_mean_test) and np.isfinite(neg_mean_test) else np.nan

        # Top-decile behavior answers a practical ranking question:
        # if we kept only the top-scoring 10 percent of valid test rows,
        # how concentrated would positives be there?
        top_rate, top_lift, top_n, base_rate = top_decile_stats(y_test, x_test_or)

        rows.append({
            "side_audit": side_name,
            "feature": col,
            "n_train": n_train,
            "pos_train": pos_train,
            "n_test": n_test,
            "pos_test": pos_test,
            "train_auc_raw": auc_train_raw,
            "train_auc_oriented": auc_train_or,
            "test_auc": auc_test,
            "test_pos_mean": pos_mean_test,
            "test_neg_mean": neg_mean_test,
            "test_mean_gap": mean_gap_test,
            "test_top_decile_hit_rate": top_rate,
            "test_top_decile_lift": top_lift,
            "test_top_decile_n": top_n,
            "test_base_rate": base_rate,
            "orientation": orient,
        })

    out = pd.DataFrame(rows)

    # Rank features primarily by held-out test AUC, then by top-decile lift,
    # then by oriented train AUC, with feature name as deterministic tie-breaker.
    out = out.sort_values(
        ["test_auc", "test_top_decile_lift", "train_auc_oriented", "feature"],
        ascending=[False, False, False, True],
        kind="mergesort",
    ).reset_index(drop=True)

    return out

# -----------------------------
# Run the Family A side-separated univariate audit.
#
# This produces one ranked table for long-side native features and one ranked
# table for short-side native features.
# -----------------------------
famA_long_audit = audit_one_side(famA_long_cols, "y_long", "long")
famA_short_audit = audit_one_side(famA_short_cols, "y_short", "short")

print("\nTop Family A long features")
print(
    famA_long_audit[
        [
            "feature",
            "n_train", "pos_train", "n_test", "pos_test",
            "train_auc_oriented", "test_auc",
            "test_top_decile_lift", "test_top_decile_hit_rate",
            "test_pos_mean", "test_neg_mean", "test_mean_gap",
            "orientation",
        ]
    ].head(25).to_string(index=False)
)

print("\nTop Family A short features")
print(
    famA_short_audit[
        [
            "feature",
            "n_train", "pos_train", "n_test", "pos_test",
            "train_auc_oriented", "test_auc",
            "test_top_decile_lift", "test_top_decile_hit_rate",
            "test_pos_mean", "test_neg_mean", "test_mean_gap",
            "orientation",
        ]
    ].head(25).to_string(index=False)
)

# Convenient named outputs kept for downstream use.
A_long_uni = famA_long_audit.copy()
A_short_uni = famA_short_audit.copy()# Family A univariate feature audit against frozen retained-episode side labels.
#
# Purpose
# This cell evaluates Family A side-native features one at a time as simple
# ranking signals for retained long and short episode entries.
#
# What it does
# 1. Rebuilds binary side labels on the native df5 / df_feat row index:
#    - y_long  = 1 at retained long-entry bars, else 0
#    - y_short = 1 at retained short-entry bars, else 0
# 2. Creates a fixed train / test split by timestamp.
# 3. Selects exact Family A side-native columns:
#    - long audit uses only Family A up-side columns
#    - short audit uses only Family A down-side columns
# 4. Runs a univariate audit for each feature separately.
# 5. Orients each feature on train only:
#    - keep x if train AUC(x) >= train AUC(-x)
#    - flip to -x otherwise
# 6. Reports train AUC, test AUC, test class-mean separation, and
#    top-decile hit-rate / lift for each feature.
#
# What it does not do
# - It does not train a multivariate model.
# - It does not modify df5, df_feat, or episodes_df.
# - It does not rebalance classes or resample rows.
#
# Main outputs
# - famA_long_audit  : long-side univariate audit table
# - famA_short_audit : short-side univariate audit table
# - A_long_uni       : convenience copy of famA_long_audit
# - A_short_uni      : convenience copy of famA_short_audit

import numpy as np
import pandas as pd

# Require the source bars, research features, and retained episode table.
# The audit depends on the original timestamp index from df5, the feature
# columns in df_feat, and the retained side labels implied by episodes_df.
if "df5" not in globals():
    raise NameError("df5 is not defined")
if "df_feat" not in globals():
    raise NameError("df_feat is not defined")
if "episodes_df" not in globals():
    raise NameError("episodes_df is not defined")

# -----------------------------
# Build frozen side-specific labels on the native df5 / df_feat index.
#
# episodes_df stores retained entry timestamps and side.
# Map those entry timestamps back to df5 row positions, then create:
# - y_long  = 1 only for retained long-entry bars
# - y_short = 1 only for retained short-entry bars
#
# These labels are then attached to a copy of df_feat so the audit can work
# directly in feature space without altering the original research dataframe.
# -----------------------------
ts = pd.to_datetime(df5["timestamp"], utc=True, errors="raise")
ts_index = pd.Index(ts)

entry_t = pd.to_datetime(episodes_df["entry_t"], utc=True, errors="raise")
entry_i = ts_index.get_indexer(entry_t)
if (entry_i < 0).any():
    raise ValueError("Some episode entry_t values could not be mapped to df5")

y_long = np.zeros(len(df5), dtype=np.int8)
y_short = np.zeros(len(df5), dtype=np.int8)

ep_side = episodes_df["side"].astype(str).to_numpy()
y_long[entry_i[ep_side == "long"]] = 1
y_short[entry_i[ep_side == "short"]] = 1

df_audit = df_feat.copy()
df_audit["y_long"] = y_long
df_audit["y_short"] = y_short

# -----------------------------
# Fixed chronological train / test split.
#
# This is a pure time split. Nothing after TRAIN_END is allowed into train.
# The printout makes the class balance visible before any feature scoring.
# That matters because some features may have coverage gaps after validity
# filtering, and class counts can become thin on one side.
# -----------------------------
TRAIN_END = pd.Timestamp("2024-06-30 23:59:59+00:00")
train_mask_all = df_audit["timestamp"] <= TRAIN_END
test_mask_all  = df_audit["timestamp"] > TRAIN_END

print("train rows:", int(train_mask_all.sum()))
print("test rows :", int(test_mask_all.sum()))
print("long positives train/test:", int(df_audit.loc[train_mask_all, "y_long"].sum()), int(df_audit.loc[test_mask_all, "y_long"].sum()))
print("short positives train/test:", int(df_audit.loc[train_mask_all, "y_short"].sum()), int(df_audit.loc[test_mask_all, "y_short"].sum()))

# -----------------------------
# Exact Family A side-native feature selection.
#
# Long audit:
#   include only Family A columns native to the up-side view.
#
# Short audit:
#   include only Family A columns native to the down-side view.
#
# This intentionally excludes cross-side mixing. The purpose here is to audit
# the native Family A signal stack for each side separately.
# -----------------------------
famA_long_cols = sorted([
    c for c in df_audit.columns
    if c.startswith("famA_")
    and (
        c.endswith("_up")
        or "_up_" in c
    )
])

famA_short_cols = sorted([
    c for c in df_audit.columns
    if c.startswith("famA_")
    and (
        c.endswith("_dn")
        or "_dn_" in c
    )
])

print("Family A long-native cols :", len(famA_long_cols))
print("Family A short-native cols:", len(famA_short_cols))

# -----------------------------
# Scoring helpers.
#
# safe_auc
#   Computes rank-based ROC AUC without sklearn.
#   Returns NaN when there are no valid rows, or when only one class is present.
#
# top_decile_stats
#   Sorts by descending score and evaluates the highest-scoring 10 percent.
#   Returns:
#   - top-decile hit rate
#   - top-decile lift versus base rate
#   - top-decile row count
#   - base positive rate
#
# Both helpers first remove non-finite values so feature validity masks are
# respected naturally.
# -----------------------------
def safe_auc(y_true, score):
    y_true = np.asarray(y_true, dtype=np.int8)
    score = np.asarray(score, dtype=float)

    ok = np.isfinite(score) & np.isfinite(y_true)
    y_true = y_true[ok]
    score = score[ok]

    if len(y_true) == 0:
        return np.nan

    n_pos = int((y_true == 1).sum())
    n_neg = int((y_true == 0).sum())
    if n_pos == 0 or n_neg == 0:
        return np.nan

    ranks = pd.Series(score).rank(method="average").to_numpy(dtype=float)
    sum_ranks_pos = ranks[y_true == 1].sum()
    auc = (sum_ranks_pos - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg)
    return float(auc)

def top_decile_stats(y_true, score):
    y_true = np.asarray(y_true, dtype=np.int8)
    score = np.asarray(score, dtype=float)

    ok = np.isfinite(score) & np.isfinite(y_true)
    y_true = y_true[ok]
    score = score[ok]

    n = len(y_true)
    if n == 0:
        return np.nan, np.nan, 0, np.nan

    k = max(1, int(np.ceil(0.10 * n)))
    order = np.argsort(-score, kind="mergesort")
    top_idx = order[:k]

    base_rate = float(y_true.mean()) if n > 0 else np.nan
    top_rate = float(y_true[top_idx].mean()) if k > 0 else np.nan
    lift = np.nan if (not np.isfinite(base_rate) or base_rate <= 0.0) else float(top_rate / base_rate)

    return top_rate, lift, int(k), base_rate

# Core per-side univariate audit.
#
# For each feature:
# 1. keep only rows where that feature is non-null
# 2. split those valid rows into train and test by timestamp mask
# 3. require both train and test to have at least one positive example
# 4. compute train AUC for x and for -x
# 5. choose orientation using train only
# 6. evaluate the oriented feature on test
# 7. collect summary metrics into one output row
#
# Stored metrics include:
# - coverage counts and positive counts
# - raw and oriented train AUC
# - test AUC
# - test positive and negative class means
# - test mean gap
# - top-decile hit rate and lift on test
# - chosen orientation (+1 or -1)
def audit_one_side(feature_cols, y_col, side_name):
    rows = []

    for col in feature_cols:
        x = pd.to_numeric(df_audit[col], errors="coerce")
        y = df_audit[y_col].astype(np.int8)

        valid = x.notna().to_numpy()
        train_mask = valid & train_mask_all.to_numpy()
        test_mask = valid & test_mask_all.to_numpy()

        n_train = int(train_mask.sum())
        n_test = int(test_mask.sum())

        pos_train = int(y.loc[train_mask].sum())
        pos_test = int(y.loc[test_mask].sum())

        # If either split has no usable rows or no positive class, keep the
        # feature in the table but mark scoring outputs as NaN.
        if n_train == 0 or n_test == 0 or pos_train == 0 or pos_test == 0:
            rows.append({
                "side_audit": side_name,
                "feature": col,
                "n_train": n_train,
                "pos_train": pos_train,
                "n_test": n_test,
                "pos_test": pos_test,
                "train_auc_raw": np.nan,
                "train_auc_oriented": np.nan,
                "test_auc": np.nan,
                "test_pos_mean": np.nan,
                "test_neg_mean": np.nan,
                "test_mean_gap": np.nan,
                "test_top_decile_hit_rate": np.nan,
                "test_top_decile_lift": np.nan,
                "test_top_decile_n": 0,
                "test_base_rate": np.nan,
                "orientation": np.nan,
            })
            continue

        x_train = x.loc[train_mask].to_numpy(dtype=float)
        y_train = y.loc[train_mask].to_numpy(dtype=np.int8)
        x_test = x.loc[test_mask].to_numpy(dtype=float)
        y_test = y.loc[test_mask].to_numpy(dtype=np.int8)

        # Train-time orientation:
        # compare the feature as-is versus its negation.
        # Keep whichever direction gives better train AUC.
        auc_train_raw = safe_auc(y_train, x_train)
        auc_train_neg = safe_auc(y_train, -x_train)

        if np.isnan(auc_train_raw) and np.isnan(auc_train_neg):
            orient = np.nan
            x_test_or = x_test.copy()
            auc_train_or = np.nan
        elif np.isnan(auc_train_neg) or (np.isfinite(auc_train_raw) and auc_train_raw >= auc_train_neg):
            orient = 1.0
            x_test_or = x_test
            auc_train_or = auc_train_raw
        else:
            orient = -1.0
            x_test_or = -x_test
            auc_train_or = auc_train_neg

        # Evaluate the chosen orientation on held-out test rows only.
        auc_test = safe_auc(y_test, x_test_or)

        # Class-conditional means on test provide an interpretable directional
        # separation view in addition to AUC.
        pos_mask_test = (y_test == 1)
        neg_mask_test = (y_test == 0)

        pos_mean_test = float(np.mean(x_test_or[pos_mask_test])) if pos_mask_test.any() else np.nan
        neg_mean_test = float(np.mean(x_test_or[neg_mask_test])) if neg_mask_test.any() else np.nan
        mean_gap_test = pos_mean_test - neg_mean_test if np.isfinite(pos_mean_test) and np.isfinite(neg_mean_test) else np.nan

        # Top-decile behavior answers a practical ranking question:
        # if we kept only the top-scoring 10 percent of valid test rows,
        # how concentrated would positives be there?
        top_rate, top_lift, top_n, base_rate = top_decile_stats(y_test, x_test_or)

        rows.append({
            "side_audit": side_name,
            "feature": col,
            "n_train": n_train,
            "pos_train": pos_train,
            "n_test": n_test,
            "pos_test": pos_test,
            "train_auc_raw": auc_train_raw,
            "train_auc_oriented": auc_train_or,
            "test_auc": auc_test,
            "test_pos_mean": pos_mean_test,
            "test_neg_mean": neg_mean_test,
            "test_mean_gap": mean_gap_test,
            "test_top_decile_hit_rate": top_rate,
            "test_top_decile_lift": top_lift,
            "test_top_decile_n": top_n,
            "test_base_rate": base_rate,
            "orientation": orient,
        })

    out = pd.DataFrame(rows)

    # Rank features primarily by held-out test AUC, then by top-decile lift,
    # then by oriented train AUC, with feature name as deterministic tie-breaker.
    out = out.sort_values(
        ["test_auc", "test_top_decile_lift", "train_auc_oriented", "feature"],
        ascending=[False, False, False, True],
        kind="mergesort",
    ).reset_index(drop=True)

    return out

# -----------------------------
# Run the Family A side-separated univariate audit.
#
# This produces one ranked table for long-side native features and one ranked
# table for short-side native features.
# -----------------------------
famA_long_audit = audit_one_side(famA_long_cols, "y_long", "long")
famA_short_audit = audit_one_side(famA_short_cols, "y_short", "short")

print("\nTop Family A long features")
print(
    famA_long_audit[
        [
            "feature",
            "n_train", "pos_train", "n_test", "pos_test",
            "train_auc_oriented", "test_auc",
            "test_top_decile_lift", "test_top_decile_hit_rate",
            "test_pos_mean", "test_neg_mean", "test_mean_gap",
            "orientation",
        ]
    ].head(25).to_string(index=False)
)

print("\nTop Family A short features")
print(
    famA_short_audit[
        [
            "feature",
            "n_train", "pos_train", "n_test", "pos_test",
            "train_auc_oriented", "test_auc",
            "test_top_decile_lift", "test_top_decile_hit_rate",
            "test_pos_mean", "test_neg_mean", "test_mean_gap",
            "orientation",
        ]
    ].head(25).to_string(index=False)
)

# Convenient named outputs kept for downstream use.
A_long_uni = famA_long_audit.copy()
A_short_uni = famA_short_audit.copy()