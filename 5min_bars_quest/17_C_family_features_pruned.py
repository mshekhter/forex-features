# Prune Family C to a frozen retained shortlist.
#
# Purpose:
# Start from the current feature frame and remove every Family C column except
# a small fixed set that has already been selected for retention.
#
# Source selection:
# If df_feat_reduced_B already exists, prune from that frame so prior Family B
# pruning is preserved.
# Otherwise fall back to df_feat_reduced, then df_feat.
#
# Retention rule:
# Keep only the exact Family C columns listed in C_keep.
# All non Family C columns remain unchanged.
# All other Family C columns are removed.
#
# Validation:
# The cell first checks that every requested retained Family C column is
# present in the source frame before pruning.
#
# Output:
# df_feat_reduced_C is the new reduced feature frame after Family C pruning.
# The printed summary reports source and reduced shapes, Family C keep/drop
# counts, and the exact retained and dropped Family C column names.

import pandas as pd

# Choose the source feature frame for pruning.
# Prefer the Family B pruned frame when it exists.
# Otherwise fall back to the broader reduced frame, then the full frame.
if "df_feat_reduced_B" in globals():
    df_feat_C_prune_src = df_feat_reduced_B.copy()
elif "df_feat_reduced" in globals():
    df_feat_C_prune_src = df_feat_reduced.copy()
elif "df_feat" in globals():
    df_feat_C_prune_src = df_feat.copy()
else:
    raise NameError("No feature frame found")

# Frozen Family C shortlist to retain.
# Every other Family C column will be removed.
C_keep = [
    "famC_build_pre_release_efficiency_up_w6",
    "famC_build_pre_release_efficiency_dn_w6",

    "famC_build_anchor_separation_up_w12",
    "famC_build_anchor_separation_dn_w12",

    "famC_build_progressive_balance_up_w24",
    "famC_build_progressive_balance_dn_w24",
]

# Confirm that every requested retained Family C column exists in the source frame.
missing_keep = [c for c in C_keep if c not in df_feat_C_prune_src.columns]
if missing_keep:
    raise KeyError(f"These requested retained Family C columns are missing: {missing_keep}")

# Collect all source columns, identify the full Family C block,
# and determine which Family C columns will be dropped.
all_cols = list(df_feat_C_prune_src.columns)
C_all = [c for c in all_cols if c.startswith("famC_")]
C_drop = [c for c in C_all if c not in C_keep]

# Build the retained column list:
# keep every non Family C column,
# plus only the Family C columns explicitly listed in C_keep.
reduced_cols = [c for c in all_cols if (not c.startswith("famC_")) or (c in C_keep)]
df_feat_reduced_C = df_feat_C_prune_src[reduced_cols].copy()

# Print pruning summary for auditability.
print("source shape                :", df_feat_C_prune_src.shape)
print("reduced_C shape             :", df_feat_reduced_C.shape)
print("Family C original col count :", len(C_all))
print("Family C kept col count     :", len(C_keep))
print("Family C dropped col count  :", len(C_drop))
print()

# Print the retained Family C shortlist in the frozen keep order.
print("Retained Family C columns:")
for c in C_keep:
    print(c)

# Print all dropped Family C columns for full pruning visibility.
print("\nDropped Family C columns:")
for c in sorted(C_drop):
    print(c)