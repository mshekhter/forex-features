# Audit the retained episode set and final binary labels for internal consistency.
# This is a read-only diagnostics cell. It does not rebuild candidates or labels.
# It checks that:
# 1. every retained episode in episodes_df can be traced back to candidates_df
# 2. retained intervals remain strictly non-overlapping in df5 index space
# 3. the final binary label series y marks exactly the retained entry bars
# It is intended as a post-selection integrity check after the episode-retention cell.

import numpy as np
import pandas as pd

# Require all upstream objects produced by the labeling pipeline.
if "candidates_df" not in globals():
    raise NameError("candidates_df is not defined")
if "episodes_df" not in globals():
    raise NameError("episodes_df is not defined")
if "y" not in globals():
    raise NameError("y is not defined")
if "df5" not in globals():
    raise NameError("df5 is not defined")

# Recreate the df5 timestamp index used by the labeling workflow.
# episodes_df stores timestamps, but overlap checks and y consistency checks
# need the retained intervals expressed back in raw bar index space.
ts = pd.to_datetime(df5["timestamp"], utc=True, errors="raise")

# Map retained episode timestamps back to df5 row indices.
# entry_idx and exit_idx should align exactly to original bars.
# Any -1 means an episode timestamp cannot be mapped back, which is a hard failure.
entry_idx = pd.Index(ts).get_indexer(pd.to_datetime(episodes_df["entry_t"], utc=True))
exit_idx  = pd.Index(ts).get_indexer(pd.to_datetime(episodes_df["exit_t"],  utc=True))

if (entry_idx < 0).any() or (exit_idx < 0).any():
    raise ValueError("Some episode timestamps could not be mapped back to df5 indices")

# Build a retained-check dataframe that mirrors episodes_df but adds
# reconstructed entry_i and exit_i for direct comparison and interval auditing.
retained_check = episodes_df.copy()
retained_check["entry_i"] = entry_idx
retained_check["exit_i"] = exit_idx

# 1. Verify that every retained episode is an actual member of candidates_df.
# Use a tuple key of side, entry_i, exit_i, and best_outcome_pips.
# best_outcome_pips is rounded to stabilize float comparison.
cand_key = set(
    zip(
        candidates_df["side"].astype(str),
        candidates_df["entry_i"].astype(int),
        candidates_df["exit_i"].astype(int),
        np.round(candidates_df["best_outcome_pips"].astype(float), 10),
    )
)

ret_key = list(
    zip(
        retained_check["side"].astype(str),
        retained_check["entry_i"].astype(int),
        retained_check["exit_i"].astype(int),
        np.round(retained_check["best_outcome_pips"].astype(float), 10),
    )
)

exists_mask = np.array([k in cand_key for k in ret_key], dtype=bool)

# 2. Verify strict non-overlap in retained intervals.
# Sort by entry_i, then exit_i, then episode_id for deterministic inspection.
# The strict rule is previous exit < next entry.
# Equality would still count as overlap under this rule.
retained_sorted = retained_check.sort_values(["entry_i", "exit_i", "episode_id"], kind="mergesort").reset_index(drop=True)
prev_exit = retained_sorted["exit_i"].shift(1)
strict_nonoverlap_mask = prev_exit.isna() | (prev_exit < retained_sorted["entry_i"])

# 3. Verify y consistency.
# Convert y into a plain integer array on df5 index space, then recover
# all entry locations labeled as positive. These should match retained entry_i exactly.
y_arr = pd.Series(y, index=df5.index).astype(int).to_numpy()
y_entry_idx = np.flatnonzero(y_arr == 1)

# Collect compact diagnostics for the three checks above.
diag = {
    "retained_episodes": int(len(episodes_df)),
    "positive_labels": int((y_arr == 1).sum()),
    "positive_labels_match_episode_count": bool((y_arr == 1).sum() == len(episodes_df)),
    "all_retained_exist_in_candidates": bool(exists_mask.all()),
    "missing_retained_in_candidates": int((~exists_mask).sum()),
    "strict_nonoverlap_pass": bool(strict_nonoverlap_mask.all()),
    "overlap_violations": int((~strict_nonoverlap_mask).sum()),
    "y_entries_match_episode_entries": bool(np.array_equal(np.sort(y_entry_idx), np.sort(retained_sorted["entry_i"].to_numpy()))),
    "retained_total_best_outcome_pips": float(retained_check["best_outcome_pips"].sum()),
}

print(pd.Series(diag))

# Show any retained episodes that cannot be found in candidates_df.
# In a correct pipeline, this section should be empty.
print("\nmissing retained episodes in candidates_df:")
display(retained_check.loc[~exists_mask].head(20))

# Show the retained-interval rows involved in strict non-overlap failures.
# For each violating row, print the previous retained interval and the current one
# so the offending overlap can be inspected directly.
print("\noverlap violations:")
viol_ix = np.where(~strict_nonoverlap_mask.to_numpy())[0]
if len(viol_ix) == 0:
    display(retained_sorted.iloc[0:0])
else:
    rows = []
    for i in viol_ix[:20]:
        rows.append(retained_sorted.iloc[i - 1])
        rows.append(retained_sorted.iloc[i])
    display(pd.DataFrame(rows).drop_duplicates())

# Compare the positive label positions in y with the retained episode entry bars.
# In a correct pipeline, both sets should be identical.
print("\ny vs retained entry mismatch:")
ret_entry_sorted = np.sort(retained_sorted["entry_i"].to_numpy())
if np.array_equal(np.sort(y_entry_idx), ret_entry_sorted):
    print("none")
else:
    print("in y not retained:", np.setdiff1d(y_entry_idx, ret_entry_sorted)[:20])
    print("retained not in y:", np.setdiff1d(ret_entry_sorted, y_entry_idx)[:20])

# Print a small sample of retained intervals after sorting in index space.
# This is a convenience view for quick visual inspection.
print("\nhead of retained intervals:")
display(retained_sorted.head(20))