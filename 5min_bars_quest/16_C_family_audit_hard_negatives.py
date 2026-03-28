# Family C hard-negative univariate side audit.
#
# Purpose:
# Evaluate each side-native Family C feature one at a time against frozen
# episode-entry labels, using a harder sampled negative set rather than all
# non-entry bars.
#
# Label definition:
# y_long and y_short are sparse entry-bar labels only.
# They mark where retained long and short episodes begin on the df5 bar index.
# They are not bar-by-bar trade outcome labels.
#
# Source selection:
# This audit prefers df_feat_reduced_B when it exists, so any prior Family B
# pruning is already reflected in the working feature frame.
# If that frame is unavailable, it falls back to df_feat_reduced, then df_feat.
#
# Side separation:
# Long audit uses only Family C up-side features.
# Short audit uses only Family C down-side features.
# There is no cross-side feature mixing in this cell.
#
# Hard-negative design:
# Negatives are sampled from bars that are outside a blocked buffer around any
# retained entry.
# Sampling is done separately within each calendar year so the negative sample
# roughly tracks the positive year distribution.
# The target negative-to-positive ratio is fixed at 10:1 within year.
#
# Why the buffer exists:
# Bars close to retained entries are removed from the negative pool so the
# audit tests against cleaner non-entry contexts, not bars sitting right next
# to known episode starts.
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
# Each output row is one univariate Family C feature audit under the
# hard-negative sampling design.
# Ranking is driven first by out-of-sample test AUC,
# then by out-of-sample top-decile lift,
# then by oriented train AUC.
#
# Saved tables:
# C_long_uni_hard and C_short_uni_hard keep the full long-side and short-side
# hard-negative audit results for downstream review, comparison, and selection.

import numpy as np
import pandas as pd

# Require the master 5 minute bar frame for timestamp-to-row alignment.
# Require the retained episode table that defines entry timestamps and sides.

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

# -----------------------------
# frozen labels aligned to df5 / feature frame
# -----------------------------
# Map retained episode entry timestamps back to exact df5 row positions.
# Those positions become the sparse positive labels for long and short.
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
# normalize timestamps,
# attach sparse side labels,
# and extract calendar year for year-matched negative sampling.
df_audit = df_feat_C_src.copy()
df_audit["timestamp"] = pd.to_datetime(df_audit["timestamp"], utc=True, errors="raise")
df_audit["y_long"] = y_long
df_audit["y_short"] = y_short
df_audit["year"] = df_audit["timestamp"].dt.year.astype(int)

# -----------------------------
# time split
# -----------------------------
# Global time-based split used throughout the audit.
# This split is independent of feature availability.
TRAIN_END = pd.Timestamp("2024-06-30 23:59:59+00:00")
train_mask_all = (df_audit["timestamp"] <= TRAIN_END).to_numpy()
test_mask_all  = (df_audit["timestamp"] > TRAIN_END).to_numpy()

# -----------------------------
# exact Family C side-native columns
# -----------------------------
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

# -----------------------------
# hard-negative construction
# exclude +/- 24 bars around any retained entry
# match negatives by calendar year to positives
# fixed negative ratio = 10:1
# -----------------------------
# Hard-negative setup:
# BUFFER blocks a symmetric window around every retained entry from the
# negative pool.
# NEG_PER_POS sets the target number of negatives per positive within year.
# RNG_SEED keeps sampling reproducible.
BUFFER = 24
NEG_PER_POS = 10
RNG_SEED = 42

# Build the blocked region around every retained entry.
# Any bar inside these blocked windows cannot be sampled as a negative.
blocked = np.zeros(len(df_audit), dtype=bool)
for idx in entry_i:
    lo = max(0, idx - BUFFER)
    hi = min(len(df_audit), idx + BUFFER + 1)
    blocked[lo:hi] = True

# Build one sampled mask for a given side and split.
# Positives are all labeled entry bars for that side inside the split.
# Negatives come from non-entry bars inside the split and outside blocked zones.
# Negative sampling is done year by year to roughly preserve the positive year mix.
# Within each year, sample up to NEG_PER_POS negatives per positive.
def build_hard_sample_mask(y_col, split_mask, rng_seed=42):
    rng = np.random.default_rng(rng_seed)

    y_arr = df_audit[y_col].to_numpy(dtype=np.int8)
    year_arr = df_audit["year"].to_numpy(dtype=np.int64)

    pos_idx = np.flatnonzero((y_arr == 1) & split_mask)
    neg_pool_idx = np.flatnonzero((y_arr == 0) & split_mask & (~blocked))

    # Partition the negative pool by calendar year.
    neg_by_year = {}
    for yr in np.unique(year_arr[neg_pool_idx]):
        neg_by_year[int(yr)] = neg_pool_idx[year_arr[neg_pool_idx] == yr].copy()

    # Count how many positives exist in each year for this side and split.
    pos_year_counts = pd.Series(year_arr[pos_idx]).value_counts().sort_index()

    chosen_neg = []
    for yr, pos_count in pos_year_counts.items():
        yr = int(yr)
        need = int(pos_count) * NEG_PER_POS
        pool = neg_by_year.get(yr, np.array([], dtype=np.int64))
        if len(pool) == 0:
            continue

        # If the pool is smaller than requested, keep all available negatives.
        # Otherwise sample without replacement.
        if len(pool) <= need:
            picked = pool.copy()
        else:
            picked = np.sort(rng.choice(pool, size=need, replace=False))

        chosen_neg.append(picked)

    if len(chosen_neg) == 0:
        neg_idx = np.array([], dtype=np.int64)
    else:
        neg_idx = np.concatenate(chosen_neg)

    # Final sample mask contains all positives plus sampled negatives.
    sample_mask = np.zeros(len(df_audit), dtype=bool)
    sample_mask[pos_idx] = True
    sample_mask[neg_idx] = True

    return sample_mask, pos_idx, neg_idx

# Build separate hard-negative samples for long and short, on train and test.
# Different seeds keep the samples reproducible while avoiding identical draws.
train_sample_long, train_pos_long, train_neg_long = build_hard_sample_mask("y_long", train_mask_all, rng_seed=RNG_SEED + 51)
test_sample_long,  test_pos_long,  test_neg_long  = build_hard_sample_mask("y_long", test_mask_all,  rng_seed=RNG_SEED + 52)

train_sample_short, train_pos_short, train_neg_short = build_hard_sample_mask("y_short", train_mask_all, rng_seed=RNG_SEED + 61)
test_sample_short,  test_pos_short,  test_neg_short  = build_hard_sample_mask("y_short", test_mask_all,  rng_seed=RNG_SEED + 62)

print("long hard negatives train/test:", len(train_neg_long), len(test_neg_long))
print("short hard negatives train/test:", len(train_neg_short), len(test_neg_short))

# -----------------------------
# scoring helpers
# -----------------------------
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

# Audit one side under the hard-negative sample design.
# For each feature:
# 1. keep only rows where that feature is present
# 2. intersect with the prebuilt hard-negative sample masks
# 3. choose feature direction on train only
# 4. evaluate the frozen direction on test
# 5. collect discrimination and concentration metrics
def audit_one_side_hard(feature_cols, y_col, side_name, train_sample_mask, test_sample_mask):
    rows = []

    for col in feature_cols:
        x = pd.to_numeric(df_audit[col], errors="coerce")
        y = df_audit[y_col].astype(np.int8)

        # Validity is feature-specific.
        # Each feature is evaluated only on sampled rows where it is not missing.
        valid = x.notna().to_numpy()
        train_mask = valid & train_sample_mask
        test_mask = valid & test_sample_mask

        n_train = int(train_mask.sum())
        n_test = int(test_mask.sum())

        pos_train = int(y.loc[train_mask].sum())
        pos_test = int(y.loc[test_mask].sum())
        neg_train = int(n_train - pos_train)
        neg_test = int(n_test - pos_test)

        # If either split has no usable rows, no positives, or no negatives,
        # keep the row but leave evaluation metrics empty.
        if n_train == 0 or n_test == 0 or pos_train == 0 or pos_test == 0 or neg_train == 0 or neg_test == 0:
            rows.append({
                "side_audit": side_name,
                "feature": col,
                "n_train": n_train,
                "pos_train": pos_train,
                "neg_train": neg_train,
                "n_test": n_test,
                "pos_test": pos_test,
                "neg_test": neg_test,
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
            "neg_train": neg_train,
            "n_test": n_test,
            "pos_test": pos_test,
            "neg_test": neg_test,
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

# -----------------------------
# run Family C hard-negative univariate audit
# -----------------------------
# Run separate long-side and short-side audits under the hard-negative design.
famC_long_hard_audit = audit_one_side_hard(
    famC_long_cols,
    "y_long",
    "long",
    train_sample_long,
    test_sample_long,
)

famC_short_hard_audit = audit_one_side_hard(
    famC_short_cols,
    "y_short",
    "short",
    train_sample_short,
    test_sample_short,
)

# Show the highest-ranked long-side Family C features under the hard-negative design.
print("\nTop Family C long features, hard negatives")
print(
    famC_long_hard_audit[
        [
            "feature",
            "n_train", "pos_train", "neg_train",
            "n_test", "pos_test", "neg_test",
            "train_auc_oriented", "test_auc",
            "test_top_decile_lift", "test_top_decile_hit_rate",
            "test_pos_mean", "test_neg_mean", "test_mean_gap",
            "orientation",
        ]
    ].head(30).to_string(index=False)
)

# Show the highest-ranked short-side Family C features under the hard-negative design.
print("\nTop Family C short features, hard negatives")
print(
    famC_short_hard_audit[
        [
            "feature",
            "n_train", "pos_train", "neg_train",
            "n_test", "pos_test", "neg_test",
            "train_auc_oriented", "test_auc",
            "test_top_decile_lift", "test_top_decile_hit_rate",
            "test_pos_mean", "test_neg_mean", "test_mean_gap",
            "orientation",
        ]
    ].head(30).to_string(index=False)
)

# Preserve the full hard-negative audit tables for downstream review and selection.
C_long_uni_hard = famC_long_hard_audit.copy()
C_short_uni_hard = famC_short_hard_audit.copy()