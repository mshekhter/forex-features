# Audit primitive validity and basic range sanity for all Family A-D feature columns.
# This cell summarizes non-null counts, first valid index, distribution moments,
# selected quantiles, inf counts, and impossible-range violations for frozen
# bounded features. It does not modify df_feat.
#
# Purpose:
# - Check whether feature columns were populated where expected
# - Detect infinities and unexpected missingness patterns
# - Summarize basic distribution shape for quick inspection
# - Explicitly test bounded features against their frozen allowed ranges
#
# Output:
# - audit_df: one row per Family A-D feature column
# - printed slices for infinities, impossible values, and first-valid-index behavior
# - primitive_validity_audit_df: convenience copy of the final audit dataframe

import numpy as np
import pandas as pd

# This audit is read-only and expects the research dataframe to already exist.
if "df_feat" not in globals():
    raise NameError("df_feat is not defined")

# Restrict the audit to feature columns created by Families A through D.
fam_cols = [c for c in df_feat.columns if c.startswith(("famA_", "famB_", "famC_", "famD_"))]
if not fam_cols:
    raise ValueError("No Family A-D columns found in df_feat")

# Frozen bounded ranges to audit explicitly
# Only features with inherently bounded definitions are checked here.
# Anything not listed below is still summarized statistically, but is not treated
# as range-invalid if it exceeds a particular interval.
range_rules = [
    ("famA_support_ratio_", 0.0, 1.0),
    ("famA_body_support_ratio_", 0.0, 1.0),

    ("famB_dir_eff_", -1.0, 1.0),
    ("famB_sign_change_density_", 0.0, 1.0),
    ("famB_new_ground_frequency_", 0.0, 1.0),
    ("famB_monotone_segment_concentration_", 0.0, 1.0),

    ("famC_build_support_persistence_", 0.0, 1.0),
    ("famC_build_support_clustering_", 0.0, 1.0),
    ("famC_build_pullback_acceptability_", 0.0, 1.0),

    ("famD_clv", 0.0, 1.0),
    ("famD_release_acceptance_quality_", 0.0, 1.0),
    ("famD_release_decisiveness_", 0.0, 1.0),
    ("famD_release_local_takeover_balance_3_", -1.0, 1.0),
]

# Return the frozen allowed range for a column when one exists.
# Match either an exact name or a family-style prefix rule.
def _get_range_rule(col):
    for prefix, lo, hi in range_rules:
        if col == prefix or col.startswith(prefix):
            return lo, hi
    return None

# Accumulate one audit row per feature column, then materialize as a dataframe.
rows = []

for col in fam_cols:
    # Force numeric interpretation so the audit behaves consistently even if a
    # column was stored with object dtype.
    s = pd.to_numeric(df_feat[col], errors="coerce")
    arr = s.to_numpy(dtype=float)

    # Separate three cases:
    # - inf values
    # - finite values
    # - non-null values, which includes inf but excludes NaN
    inf_mask = np.isinf(arr)
    finite_mask = np.isfinite(arr)
    non_null_mask = ~np.isnan(arr)

    non_null_count = int(non_null_mask.sum())
    inf_count = int(inf_mask.sum())

    # first_valid_index records where the column first becomes populated.
    # This is useful for checking whether rolling-window validity starts where expected.
    if non_null_count > 0:
        first_valid_index = int(np.flatnonzero(non_null_mask)[0])
    else:
        first_valid_index = np.nan

    # Distribution summaries are computed only on finite values.
    # NaN and inf are intentionally excluded from quantiles and moments.
    finite_vals = arr[finite_mask]

    if finite_vals.size > 0:
        min_val = float(np.min(finite_vals))
        max_val = float(np.max(finite_vals))
        mean_val = float(np.mean(finite_vals))
        std_val = float(np.std(finite_vals, ddof=1)) if finite_vals.size > 1 else 0.0

        # Quantiles provide a compact view of feature scale and tail behavior.
        q01 = float(np.quantile(finite_vals, 0.01))
        q05 = float(np.quantile(finite_vals, 0.05))
        q25 = float(np.quantile(finite_vals, 0.25))
        q50 = float(np.quantile(finite_vals, 0.50))
        q75 = float(np.quantile(finite_vals, 0.75))
        q95 = float(np.quantile(finite_vals, 0.95))
        q99 = float(np.quantile(finite_vals, 0.99))
    else:
        min_val = np.nan
        max_val = np.nan
        mean_val = np.nan
        std_val = np.nan
        q01 = np.nan
        q05 = np.nan
        q25 = np.nan
        q50 = np.nan
        q75 = np.nan
        q95 = np.nan
        q99 = np.nan

    # Apply bounded-range validation only to columns with explicit frozen rules.
    # Count separately how many finite values fall below or above the allowed interval.
    rule = _get_range_rule(col)
    if rule is not None and finite_vals.size > 0:
        lo, hi = rule
        below_count = int((finite_vals < lo).sum())
        above_count = int((finite_vals > hi).sum())
        impossible_count = below_count + above_count
        range_rule = f"[{lo}, {hi}]"
    else:
        below_count = np.nan
        above_count = np.nan
        impossible_count = np.nan
        range_rule = ""

    # Store one complete audit record for this feature column.
    rows.append({
        "column": col,
        "non_null_count": non_null_count,
        "first_valid_index": first_valid_index,
        "inf_count": inf_count,
        "min": min_val,
        "q01": q01,
        "q05": q05,
        "q25": q25,
        "q50": q50,
        "q75": q75,
        "q95": q95,
        "q99": q99,
        "max": max_val,
        "mean": mean_val,
        "std": std_val,
        "range_rule": range_rule,
        "below_range_count": below_count,
        "above_range_count": above_count,
        "impossible_count": impossible_count,
    })

# Build the final audit table sorted by column name for stable inspection.
audit_df = pd.DataFrame(rows).sort_values(["column"]).reset_index(drop=True)

print("Primitive validity audit rows:", len(audit_df))
print(audit_df.head(20))

# Compact sanity slices
# Surface any infinities explicitly, since they often indicate divide-by-zero
# or missing validity masking upstream.
print("\nColumns with inf values:")
print(audit_df.loc[audit_df["inf_count"] > 0, ["column", "inf_count"]])

# Surface only bounded features that violate their frozen allowed intervals.
print("\nColumns with impossible values under frozen bounded ranges:")
print(audit_df.loc[audit_df["impossible_count"].fillna(0) > 0, [
    "column", "range_rule", "below_range_count", "above_range_count", "impossible_count"
]])

# Describe where features begin to populate.
# This is especially useful for checking rolling-window onset behavior by family.
print("\nFirst valid index summary:")
print(audit_df["first_valid_index"].describe())

# Optional convenience views by family
# These are compact per-family slices showing validity counts, onset index,
# infinities, and range violations without printing the full audit table.
for fam in ["famA_", "famB_", "famC_", "famD_"]:
    sub = audit_df[audit_df["column"].str.startswith(fam)]
    print(f"\n{fam} count:", len(sub))
    print(sub[["column", "non_null_count", "first_valid_index", "inf_count", "impossible_count"]].head(15))

# Keep a named copy available for later inspection or export.
primitive_validity_audit_df = audit_df.copy()