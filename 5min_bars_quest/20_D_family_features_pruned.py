# Prune Family D to a frozen retained shortlist.
#
# Purpose:
# Start from the current feature frame and remove every Family D column except
# a small fixed set that has already been selected for retention.
#
# Source selection:
# If df_feat_reduced_C already exists, prune from that frame so prior Family B
# and Family C pruning is preserved.
# Otherwise fall back to df_feat_reduced_B, then df_feat_reduced, then df_feat.
#
# Retention rule:
# Keep only the exact Family D columns listed in D_keep.
# All non Family D columns remain unchanged.
# All other Family D columns are removed.
#
# Validation:
# The cell first checks that every requested retained Family D column is
# present in the source frame before pruning.
#
# Output:
# df_feat_reduced_D is the new reduced feature frame after Family D pruning.
# The printed summary reports source and reduced shapes, Family D keep/drop
# counts, and the exact retained and dropped Family D column names.

import pandas as pd

# Choose the source feature frame for pruning.
# Prefer the Family C pruned frame when it exists.
# Otherwise fall back through earlier reduced frames to the full feature frame.
if "df_feat_reduced_C" in globals():
    df_feat_D_prune_src = df_feat_reduced_C.copy()
elif "df_feat_reduced_B" in globals():
    df_feat_D_prune_src = df_feat_reduced_B.copy()
elif "df_feat_reduced" in globals():
    df_feat_D_prune_src = df_feat_reduced.copy()
elif "df_feat" in globals():
    df_feat_D_prune_src = df_feat.copy()
else:
    raise NameError("No feature frame found")

# Frozen Family D shortlist to retain.
# Every other Family D column will be removed.
D_keep = [
    "famD_release_containment_escape_up",
    "famD_release_containment_escape_dn",
    "famD_release_local_takeover_balance_3_up",
    "famD_release_local_takeover_balance_3_dn",
]

# Confirm that every requested retained Family D column exists in the source frame.
missing_keep = [c for c in D_keep if c not in df_feat_D_prune_src.columns]
if missing_keep:
    raise KeyError(f"These requested retained Family D columns are missing: {missing_keep}")

# Collect all source columns, identify the full Family D block,
# and determine which Family D columns will be dropped.
all_cols = list(df_feat_D_prune_src.columns)
D_all = [c for c in all_cols if c.startswith("famD_")]
D_drop = [c for c in D_all if c not in D_keep]

# Build the retained column list:
# keep every non Family D column,
# plus only the Family D columns explicitly listed in D_keep.
reduced_cols = [c for c in all_cols if (not c.startswith("famD_")) or (c in D_keep)]
df_feat_reduced_D = df_feat_D_prune_src[reduced_cols].copy()

# Print pruning summary for auditability.
print("source shape                :", df_feat_D_prune_src.shape)
print("reduced_D shape             :", df_feat_reduced_D.shape)
print("Family D original col count :", len(D_all))
print("Family D kept col count     :", len(D_keep))
print("Family D dropped col count  :", len(D_drop))
print()

# Print the retained Family D shortlist in the frozen keep order.
print("Retained Family D columns:")
for c in D_keep:
    print(c)

# Print all dropped Family D columns for full pruning visibility.
print("\nDropped Family D columns:")
for c in sorted(D_drop):
    print(c)