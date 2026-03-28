# Prune Family B to a frozen retained shortlist.
#
# Purpose:
# Start from the current feature frame and remove every Family B column except
# a small fixed set that has already been selected for retention.
#
# Source selection:
# If df_feat_reduced already exists, prune from that frame.
# Otherwise prune from the full df_feat frame.
#
# Retention rule:
# Keep only the exact Family B columns listed in B_keep.
# All non Family B columns remain unchanged.
# All other Family B columns are removed.
#
# Validation:
# The cell first checks that every requested retained Family B column is
# present in the source frame before pruning.
#
# Output:
# df_feat_reduced_B is the new reduced feature frame after Family B pruning.
# The printed summary reports source and reduced shapes, Family B keep/drop
# counts, and the exact retained and dropped Family B column names.

import pandas as pd

# Choose the source feature frame for pruning.
# Prefer the already reduced frame when it exists.
# Otherwise use the full feature frame.
if "df_feat_reduced" in globals():
    df_feat_B_prune_src = df_feat_reduced.copy()
elif "df_feat" in globals():
    df_feat_B_prune_src = df_feat.copy()
else:
    raise NameError("Neither df_feat_reduced nor df_feat is defined")

# Frozen Family B shortlist to retain.
# Every other Family B column will be removed.
B_keep = [
    "famB_final_pullback_from_peak_up_w12",
    "famB_final_pullback_from_peak_dn_w12",

    "famB_max_pullback_from_peak_up_w6",
    "famB_max_pullback_from_peak_dn_w6",

    "famB_slope_3_add_up_w12",
    "famB_slope_3_add_dn_w12",

    "famB_terminal_disp_up_w6",
    "famB_terminal_disp_dn_w6",
]

# Confirm that every requested retained Family B column exists in the source frame.
missing_keep = [c for c in B_keep if c not in df_feat_B_prune_src.columns]
if missing_keep:
    raise KeyError(f"These requested retained Family B columns are missing: {missing_keep}")

# Collect all source columns, identify the full Family B block,
# and determine which Family B columns will be dropped.
all_cols = list(df_feat_B_prune_src.columns)
B_all = [c for c in all_cols if c.startswith("famB_")]
B_drop = [c for c in B_all if c not in B_keep]

# Build the retained column list:
# keep every non Family B column,
# plus only the Family B columns explicitly listed in B_keep.
reduced_cols = [c for c in all_cols if (not c.startswith("famB_")) or (c in B_keep)]
df_feat_reduced_B = df_feat_B_prune_src[reduced_cols].copy()

# Print pruning summary for auditability.
print("source shape                :", df_feat_B_prune_src.shape)
print("reduced_B shape             :", df_feat_reduced_B.shape)
print("Family B original col count :", len(B_all))
print("Family B kept col count     :", len(B_keep))
print("Family B dropped col count  :", len(B_drop))
print()

# Print the retained Family B shortlist in the frozen keep order.
print("Retained Family B columns:")
for c in B_keep:
    print(c)

# Print all dropped Family B columns for full pruning visibility.
print("\nDropped Family B columns:")
for c in sorted(B_drop):
    print(c)