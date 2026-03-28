# Family C univariate side audit.
#
# Purpose:
# Evaluate each side-native Family C feature one at a time against frozen
# episode-entry labels aligned to the bar index.
#
# Label definition:
# y_long and y_short are sparse entry-bar labels only.
# They mark where retained long and short episodes begin.
# They are not bar-by-bar trade outcome labels.
#
# Source selection:
# This audit prefers df_feat_reduced_B when it exists, so Family B pruning is
# already reflected in the working feature frame.
# If that frame is unavailable, it falls back to df_feat_reduced, then df_feat.
#
# Side separation:
# Long audit uses only Family C up-side features.
# Short audit uses only Family C down-side features.
# There is no cross-side feature mixing in this cell.
#
# Train/test protocol:
# The split is time-based.
# Train includes rows through 2024-06-30 23:59:59 UTC.
# Test includes all later rows.
#
# Orientation protocol:
# Each feature is scored in raw form and sign-flipped form on train.
# The better train direction is frozen, then applied unchanged on test.
# orientation =  1.0 means the raw feature is kept.
# orientation = -1.0 means the negated feature is used.
#
# Output meaning:
# Each output row is one univariate Family C feature audit.
# Ranking is driven first by out-of-sample test AUC,
# then by out-of-sample top-decile lift,
# then by oriented train AUC.
#
# Saved tables:
# C_long_uni and C_short_uni keep the full long-side and short-side audit
# results for downstream review, selection, and later modeling.

import numpy as np
import pandas as pd

# Choose the feature source for the audit.
# Prefer the Family B pruned frame when it exists.
# Otherwise fall back to the broader reduced frame, then the full frame.
if "df_feat_reduced_B" in globals():
    df_feat_C_src = df_feat_reduced_B.copy()
elif "df_feat_reduced" in globals():
    df_feat_C_src = df_feat_reduced.copy()
elif "df_feat" in globals():
    df_feat_C_src = df_feat.copy()
else:
    raise NameError("No feature frame found")

# Map retained episode entry timestamps back to exact df5 row positions.
# Those row positions become the sparse positive labels for long and short.
ts = pd.to_datetime(df5["timestamp"], utc=True, errors="raise")
ts_index = pd.Index(ts)

entry_t = pd.to_datetime(episodes_df["entry_t"], utc=True, errors="raise")
entry_i = ts_index.get_indexer(entry_t)
if (entry_i < 0).any():
    raise ValueError("Some episode entry_t values could not be mapped to df5")

# Build side-specific sparse entry labels on the df5 index.
# A value of 1 marks an entry bar for that side.
# All other bars remain 0.
y_long = np.zeros(len(df5), dtype=np.int8)
y_short = np.zeros(len(df5), dtype=np.int8)

ep_side = episodes_df["side"].astype(str).to_numpy()
y_long[entry_i[ep_side == "long"]] = 1
y_short[entry_i[ep_side == "short"]] = 1

# Start the audit frame from the selected feature source,
# normalize timestamp parsing,
# and append the frozen long/short entry labels.
df_audit = df_feat_C_src.copy()
df_audit["timestamp"] = pd.to_datetime(df_audit["timestamp"], utc=True, errors="raise")
df_audit["y_long"] = y_long
df_audit["y_short"] = y_short

# Global time-based split used throughout the audit.
# This split is independent of feature availability.
TRAIN_END = pd.Timestamp("2024-06-30 23:59:59+00:00")
train_mask_all = df_audit["timestamp"] <= TRAIN_END
test_mask_all  = df_audit["timestamp"] > TRAIN_END

# Basic split diagnostics for row counts and positive entry counts by side.
print("train rows:", int(train_mask_all.sum()))
print("test rows :", int(test_mask_all.sum()))
print("long positives train/test:", int(df_audit.loc[train_mask_all, "y_long"].sum()), int(df_audit.loc[test_mask_all, "y_long"].sum()))
print("short positives train/test:", int(df_audit.loc[train_mask_all, "y_short"].sum()), int(df_audit.loc[test_mask_all, "y_short"].sum()))

# Long audit uses only up-native Family C columns.
famC_long_cols = sorted([
    c for c in df_audit.columns
    if c.startswith("famC_") and ("_up_" in c or c.endswith("_up"))
])

# Short audit uses only down-native Family C columns.
famC_short_cols = sorted([
    c for c in df_audit.columns
    if c.startswith("famC_") and ("_dn_" in c or c.endswith("_dn"))
])

print("Family C long-native cols :", len(famC_long_cols))
print("Family C short-native cols:", len(famC_short_cols))

# Rank-based AUC for a binary label and numeric score.
# Non-finite values are removed first.
# Returns NaN when the sample is empty or contains only one class.
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

# Concentration metric for the highest-scored 10 percent of rows.
# Returns:
# 1. hit rate inside the top decile
# 2. lift versus the full-sample base rate
# 3. number of rows in the top decile
# 4. full-sample base positive rate
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

# Audit one side at a time.
# For each feature:
# 1. keep only rows where that feature is present
# 2. split those rows into train and test
# 3. choose feature direction on train only
# 4. evaluate the frozen direction on test
# 5. collect discrimination and concentration metrics
def audit_one_side(feature_cols, y_col, side_name):
    rows = []

    for col in feature_cols:
        x = pd.to_numeric(df_audit[col], errors="coerce")
        y = df_audit[y_col].astype(np.int8)

        # Validity is feature-specific.
        # Each feature is evaluated only on rows where it is not missing.
        valid = x.notna().to_numpy()
        train_mask = valid & train_mask_all.to_numpy()
        test_mask = valid & test_mask_all.to_numpy()

        n_train = int(train_mask.sum())
        n_test = int(test_mask.sum())

        pos_train = int(y.loc[train_mask].sum())
        pos_test = int(y.loc[test_mask].sum())

        # If either split has no usable sample or no positives,
        # keep the row but leave evaluation metrics empty.
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

        # Compare raw and sign-flipped versions on train.
        auc_train_raw = safe_auc(y_train, x_train)
        auc_train_neg = safe_auc(y_train, -x_train)

        # Freeze the better direction from train.
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

        # Evaluate the frozen orientation on test.
        auc_test = safe_auc(y_test, x_test_or)

        # Compare oriented feature means for positives and negatives on test.
        pos_mask_test = (y_test == 1)
        neg_mask_test = (y_test == 0)

        pos_mean_test = float(np.mean(x_test_or[pos_mask_test])) if pos_mask_test.any() else np.nan
        neg_mean_test = float(np.mean(x_test_or[neg_mask_test])) if neg_mask_test.any() else np.nan
        mean_gap_test = pos_mean_test - neg_mean_test if np.isfinite(pos_mean_test) and np.isfinite(neg_mean_test) else np.nan

        # Measure how concentrated positives are in the highest-scored test decile.
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

    # Final ranking emphasizes out-of-sample quality first.
    out = out.sort_values(
        ["test_auc", "test_top_decile_lift", "train_auc_oriented", "feature"],
        ascending=[False, False, False, True],
        kind="mergesort",
    ).reset_index(drop=True)

    return out

# Run separate long-side and short-side Family C univariate audits.
famC_long_audit = audit_one_side(famC_long_cols, "y_long", "long")
famC_short_audit = audit_one_side(famC_short_cols, "y_short", "short")

# Show the highest-ranked long-side Family C features.
print("\nTop Family C long features")
print(
    famC_long_audit[
        [
            "feature",
            "n_train", "pos_train", "n_test", "pos_test",
            "train_auc_oriented", "test_auc",
            "test_top_decile_lift", "test_top_decile_hit_rate",
            "test_pos_mean", "test_neg_mean", "test_mean_gap",
            "orientation",
        ]
    ].head(30).to_string(index=False)
)

# Show the highest-ranked short-side Family C features.
print("\nTop Family C short features")
print(
    famC_short_audit[
        [
            "feature",
            "n_train", "pos_train", "n_test", "pos_test",
            "train_auc_oriented", "test_auc",
            "test_top_decile_lift", "test_top_decile_hit_rate",
            "test_pos_mean", "test_neg_mean", "test_mean_gap",
            "orientation",
        ]
    ].head(30).to_string(index=False)
)

# Preserve the full audit tables for downstream review and selection.
C_long_uni = famC_long_audit.copy()
C_short_uni = famC_short_audit.copy()