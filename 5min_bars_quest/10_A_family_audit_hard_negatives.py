# Family A univariate feature audit with hard negatives.
#
# Purpose
# This cell evaluates Family A side-native features one at a time against
# retained long and short entry labels, but under a stricter negative-sampling
# scheme than the broad audit.
#
# Core idea
# Positives are retained episode entry bars.
# Negatives are not all other rows. Instead, negatives are sampled from a
# harder pool that:
# 1. excludes a +/- 24 bar buffer around every retained entry
# 2. matches negative availability by calendar year
# 3. uses a fixed 10:1 negative-to-positive ratio within each year
#
# Why this exists
# The broad audit can be dominated by easy negatives. This version asks a
# tighter question: which Family A features still separate positives from a
# deliberately harder and more controlled negative set?
#
# What it does
# 1. rebuilds frozen long and short labels on the native df5 / df_feat index
# 2. creates a fixed chronological train / test split
# 3. selects exact Family A side-native columns
# 4. builds hard-negative train and test samples separately for long and short
# 5. runs a univariate audit for each feature on those sampled sets
# 6. orients each feature on train only, then evaluates on held-out test
#
# What it does not do
# - it does not train a multivariate model
# - it does not modify df5, df_feat, or episodes_df
# - it does not change the retained episode set
#
# Main outputs
# - famA_long_hard_audit
# - famA_short_hard_audit
# - A_long_uni_hard
# - A_short_uni_hard

import numpy as np
import pandas as pd


# -----------------------------
# Rebuild frozen side labels aligned to the native df5 / df_feat index.
#
# episodes_df stores retained entry timestamps and side.
# Map those entry timestamps back to row positions in df5, then create:
# - y_long  = 1 only at retained long-entry bars
# - y_short = 1 only at retained short-entry bars
#
# Attach both labels to a copy of df_feat so the audit works in feature space
# without altering the original research dataframe.
# Also materialize calendar year from timestamp because hard negatives are
# sampled year by year.
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
df_audit["timestamp"] = pd.to_datetime(df_audit["timestamp"], utc=True, errors="raise")
df_audit["y_long"] = y_long
df_audit["y_short"] = y_short
df_audit["year"] = df_audit["timestamp"].dt.year.astype(int)

# -----------------------------
# Fixed chronological train / test split.
#
# Training rows are all timestamps up to and including TRAIN_END.
# Test rows are all timestamps strictly after TRAIN_END.
# These masks define the only time partition used in this cell.
# -----------------------------
TRAIN_END = pd.Timestamp("2024-06-30 23:59:59+00:00")
train_mask_all = (df_audit["timestamp"] <= TRAIN_END).to_numpy()
test_mask_all  = (df_audit["timestamp"] > TRAIN_END).to_numpy()

# -----------------------------
# Exact Family A side-native columns.
#
# Long audit:
#   use only Family A columns native to the up-side view.
#
# Short audit:
#   use only Family A columns native to the down-side view.
#
# This keeps the audit strictly side-native and avoids cross-side mixing.
# -----------------------------
famA_long_cols = sorted([
    c for c in df_audit.columns
    if c.startswith("famA_")
    and (c.endswith("_up") or "_up_" in c)
])

famA_short_cols = sorted([
    c for c in df_audit.columns
    if c.startswith("famA_")
    and (c.endswith("_dn") or "_dn_" in c)
])

print("Family A long-native cols :", len(famA_long_cols))
print("Family A short-native cols:", len(famA_short_cols))

# -----------------------------
# Hard-negative construction.
#
# Rules
# 1. Start from all retained entry bars across both sides.
# 2. Block out a +/- BUFFER neighborhood around every retained entry.
#    This removes easy near-event negatives and reduces label contamination.
# 3. Within each split and each target side:
#    - positives are the retained entries for that side
#    - negatives come only from non-positive, non-blocked rows
# 4. Sample negatives year by year to match the calendar-year distribution
#    of positives.
# 5. Use a fixed NEG_PER_POS negative-to-positive ratio within each year.
#
# Result
# The sampled set is much smaller and harder than the full row universe.
# -----------------------------
BUFFER = 24
NEG_PER_POS = 10
RNG_SEED = 42

# all_entry_mask marks every retained entry bar regardless of side.
# blocked marks the exclusion zone around all retained entries.
all_entry_mask = np.zeros(len(df_audit), dtype=bool)
all_entry_mask[entry_i] = True

blocked = np.zeros(len(df_audit), dtype=bool)
for idx in entry_i:
    lo = max(0, idx - BUFFER)
    hi = min(len(df_audit), idx + BUFFER + 1)
    blocked[lo:hi] = True

# Build the sampled hard-negative mask for one side and one time split.
#
# Inputs
# - y_col: target side label column, either y_long or y_short
# - split_mask: train or test mask
# - rng_seed: deterministic seed so sampling is reproducible
#
# Output
# - sample_mask: final mask containing all positives plus sampled negatives
# - pos_idx: positions of positives included in the sample
# - neg_idx: positions of chosen negatives included in the sample
#
# Sampling details
# - positives are all positive rows in the split
# - negatives come only from rows with y=0 in the split and outside blocked zones
# - negatives are sampled separately by calendar year
# - if a year has fewer negatives than requested, all available negatives are kept
def build_hard_sample_mask(y_col, split_mask, rng_seed=42):
    rng = np.random.default_rng(rng_seed)

    y_arr = df_audit[y_col].to_numpy(dtype=np.int8)
    year_arr = df_audit["year"].to_numpy(dtype=np.int64)

    pos_idx = np.flatnonzero((y_arr == 1) & split_mask)
    neg_pool_idx = np.flatnonzero((y_arr == 0) & split_mask & (~blocked))

    neg_by_year = {}
    for yr in np.unique(year_arr[neg_pool_idx]):
        neg_by_year[int(yr)] = neg_pool_idx[year_arr[neg_pool_idx] == yr].copy()

    pos_year_counts = pd.Series(year_arr[pos_idx]).value_counts().sort_index()

    chosen_neg = []
    for yr, pos_count in pos_year_counts.items():
        yr = int(yr)
        need = int(pos_count) * NEG_PER_POS
        pool = neg_by_year.get(yr, np.array([], dtype=np.int64))
        if len(pool) == 0:
            continue

        if len(pool) <= need:
            picked = pool.copy()
        else:
            picked = np.sort(rng.choice(pool, size=need, replace=False))

        chosen_neg.append(picked)

    if len(chosen_neg) == 0:
        neg_idx = np.array([], dtype=np.int64)
    else:
        neg_idx = np.concatenate(chosen_neg)

    sample_mask = np.zeros(len(df_audit), dtype=bool)
    sample_mask[pos_idx] = True
    sample_mask[neg_idx] = True

    return sample_mask, pos_idx, neg_idx

# Build train and test hard samples independently for long and short.
# Different seeds are used so each sample is reproducible but distinct.
train_sample_long, train_pos_long, train_neg_long = build_hard_sample_mask("y_long", train_mask_all, rng_seed=RNG_SEED + 11)
test_sample_long,  test_pos_long,  test_neg_long  = build_hard_sample_mask("y_long", test_mask_all,  rng_seed=RNG_SEED + 12)

train_sample_short, train_pos_short, train_neg_short = build_hard_sample_mask("y_short", train_mask_all, rng_seed=RNG_SEED + 21)
test_sample_short,  test_pos_short,  test_neg_short  = build_hard_sample_mask("y_short", test_mask_all,  rng_seed=RNG_SEED + 22)

print("long hard negatives train/test:", len(train_neg_long), len(test_neg_long))
print("short hard negatives train/test:", len(train_neg_short), len(test_neg_short))

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

# Core per-side hard-negative univariate audit.
#
# For each feature:
# 1. keep only rows where that feature is non-null
# 2. intersect that validity mask with the prebuilt hard-negative train/test masks
# 3. require both positive and negative classes in both splits
# 4. compute train AUC for x and for -x
# 5. choose orientation on train only
# 6. evaluate the oriented feature on held-out test
# 7. collect coverage, AUC, mean-gap, and top-decile metrics
#
# Compared with the broad audit, the difference here is entirely in the sampled
# row set used for train and test evaluation.
def audit_one_side_hard(feature_cols, y_col, side_name, train_sample_mask, test_sample_mask):
    rows = []

    for col in feature_cols:
        x = pd.to_numeric(df_audit[col], errors="coerce")
        y = df_audit[y_col].astype(np.int8)

        valid = x.notna().to_numpy()
        train_mask = valid & train_sample_mask
        test_mask = valid & test_sample_mask

        n_train = int(train_mask.sum())
        n_test = int(test_mask.sum())

        pos_train = int(y.loc[train_mask].sum())
        pos_test = int(y.loc[test_mask].sum())
        neg_train = int(n_train - pos_train)
        neg_test = int(n_test - pos_test)

        # Keep the feature in the output even when scoring is impossible.
        # In that case, report coverage counts and leave score fields as NaN.
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

        # Evaluate the chosen orientation on held-out hard-negative test rows.
        auc_test = safe_auc(y_test, x_test_or)

        # Test class-conditional means provide an interpretable separation view
        # alongside rank metrics such as AUC.
        pos_mask_test = (y_test == 1)
        neg_mask_test = (y_test == 0)

        pos_mean_test = float(np.mean(x_test_or[pos_mask_test])) if pos_mask_test.any() else np.nan
        neg_mean_test = float(np.mean(x_test_or[neg_mask_test])) if neg_mask_test.any() else np.nan
        mean_gap_test = pos_mean_test - neg_mean_test if np.isfinite(pos_mean_test) and np.isfinite(neg_mean_test) else np.nan

        # Top-decile metrics answer a ranking question:
        # among the highest-scoring 10 percent of valid hard-negative test rows,
        # how concentrated are positives?
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

    # Rank features primarily by held-out hard-negative test AUC, then by
    # top-decile lift, then by oriented train AUC.
    out = out.sort_values(
        ["test_auc", "test_top_decile_lift", "train_auc_oriented", "feature"],
        ascending=[False, False, False, True],
        kind="mergesort",
    ).reset_index(drop=True)

    return out

# -----------------------------
# Run the Family A hard-negative univariate audit.
#
# This produces one ranked table for long-side native features and one ranked
# table for short-side native features, both evaluated under the stricter
# hard-negative sampling scheme defined above.
# -----------------------------
famA_long_hard_audit = audit_one_side_hard(
    famA_long_cols,
    "y_long",
    "long",
    train_sample_long,
    test_sample_long,
)

famA_short_hard_audit = audit_one_side_hard(
    famA_short_cols,
    "y_short",
    "short",
    train_sample_short,
    test_sample_short,
)

print("\nTop Family A long features, hard negatives")
print(
    famA_long_hard_audit[
        [
            "feature",
            "n_train", "pos_train", "neg_train",
            "n_test", "pos_test", "neg_test",
            "train_auc_oriented", "test_auc",
            "test_top_decile_lift", "test_top_decile_hit_rate",
            "test_pos_mean", "test_neg_mean", "test_mean_gap",
            "orientation",
        ]
    ].head(25).to_string(index=False)
)

print("\nTop Family A short features, hard negatives")
print(
    famA_short_hard_audit[
        [
            "feature",
            "n_train", "pos_train", "neg_train",
            "n_test", "pos_test", "neg_test",
            "train_auc_oriented", "test_auc",
            "test_top_decile_lift", "test_top_decile_hit_rate",
            "test_pos_mean", "test_neg_mean", "test_mean_gap",
            "orientation",
        ]
    ].head(25).to_string(index=False)
)

# Convenient named outputs kept for downstream use.
A_long_uni_hard = famA_long_hard_audit.copy()
A_short_uni_hard = famA_short_hard_audit.copy()