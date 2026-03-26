# Build a reduced research feature frame after Family A pruning.
# This cell keeps the full df_feat row set and preserves every non-Family-A
# column automatically. Within Family A, it retains only the frozen shortlist
# of approved columns and drops the rest. The result is written to
# df_feat_reduced as a narrower feature frame for downstream work.

import pandas as pd


# Capture the full column order from df_feat so the reduced frame preserves
# original ordering for all retained columns.
all_cols = list(df_feat.columns)

# Frozen retained Family A shortlist.
# These are the only Family A columns allowed to remain in the reduced frame.
A_keep = [
    "famA_damage_flow_up_w6",
    "famA_damage_flow_dn_w6",
    "famA_net_support_balance_up_w6",
    "famA_net_support_balance_dn_w6",
    "famA_support_efficiency_up_w6",
    "famA_support_efficiency_dn_w6",
    "famA_support_efficiency_up_w12",
    "famA_support_efficiency_dn_w12",
]

# Enumerate all Family A columns currently present, then identify which of them
# will be removed because they are not part of the frozen keep list.
A_all = [c for c in all_cols if c.startswith("famA_")]
A_drop = [c for c in A_all if c not in A_keep]

# Build the reduced column list:
# - keep every non-Family-A column automatically
# - keep only the approved Family A shortlist
# Preserve original column order from df_feat.
reduced_cols = [c for c in all_cols if (not c.startswith("famA_")) or (c in A_keep)]
df_feat_reduced = df_feat[reduced_cols].copy()

# Print compact shape and column-count diagnostics so the pruning result is visible.
print("original df_feat shape       :", df_feat.shape)
print("reduced df_feat shape        :", df_feat_reduced.shape)
print("Family A original col count  :", len(A_all))
print("Family A kept col count      :", len(A_keep))
print("Family A dropped col count   :", len(A_drop))
print()

# Print the retained Family A shortlist explicitly in frozen order.
print("Retained Family A columns:")
for c in A_keep:
    print(c)

# Print the dropped Family A columns alphabetically for easy audit.
print("\nDropped Family A columns:")
for c in sorted(A_drop):
    print(c)