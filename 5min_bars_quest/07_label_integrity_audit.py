# Wanderings in data. Audit cell, not a research result cell.
# Build non-overlapping binary trade-entry labels from df5 using a fixed-horizon
# TP/SL candidate scan and weighted interval selection.
# For each eligible bar, test both long and short entries, keep only candidates
# that reach TP before SL with no ambiguous same-bar TP/SL touch, then retain
# a maximum-weight set of non-overlapping winning episodes using best favorable
# post-TP outcome as the weight. Output episodes_df, y, and df_trade.

import numpy as np
import pandas as pd
from bisect import bisect_left

PIP = 0.0001
TP_PIPS = 10.0
SL_PIPS = 10.0
H = 24

# Convert timestamps once and build mid-price close, high, and low arrays.
# Mid prices are used throughout candidate generation and episode retention.
ts = pd.to_datetime(df5["timestamp"], utc=True, errors="raise")
m_close = ((df5["close_bid"] + df5["close_ask"]) / 2.0).astype(float).to_numpy()
m_high  = ((df5["high_bid"]  + df5["high_ask"])  / 2.0).astype(float).to_numpy()
m_low   = ((df5["low_bid"]   + df5["low_ask"])   / 2.0).astype(float).to_numpy()

n = len(df5)
tp_px = TP_PIPS * PIP
sl_px = SL_PIPS * PIP

# Build one candidate episode from a single entry bar and side.
# A candidate is kept only if:
# 1. TP is reached before SL,
# 2. there is no ambiguous same-bar TP and SL touch,
# 3. once TP has been reached, the path continues until the first later SL
#    or the horizon ends, and the best favorable outcome after TP is retained.
#
# Returned object:
# - side
# - entry_i
# - exit_i where best favorable outcome was observed after TP
# - best_outcome_pips, measured as best favorable excursion after TP
#
# Return None when the path never becomes a valid winning candidate.
def build_candidate(entry_i: int, side: str):
    entry_px = m_close[entry_i]
    won = False
    best_outcome_pips = np.nan
    best_exit_i = -1

    # Scan only future bars inside the fixed horizon H.
    # The last reachable bar is capped by the dataframe end.
    for j in range(entry_i + 1, min(entry_i + H, n - 1) + 1):
        # For long:
        # - TP if high reaches entry + tp_px
        # - SL if low reaches entry - sl_px
        # - favorable progress is based on bar high versus entry
        if side == "long":
            tp_touch = m_high[j] >= (entry_px + tp_px)
            sl_touch = m_low[j]  <= (entry_px - sl_px)
            fav_now = (m_high[j] - entry_px) / PIP

        # For short:
        # - TP if low reaches entry - tp_px
        # - SL if high reaches entry + sl_px
        # - favorable progress is based on entry versus bar low
        elif side == "short":
            tp_touch = m_low[j]  <= (entry_px - tp_px)
            sl_touch = m_high[j] >= (entry_px + sl_px)
            fav_now = (entry_px - m_low[j]) / PIP
        else:
            raise ValueError(f"bad side: {side}")

        # Ambiguous bar handling:
        # if both TP and SL are touched in the same future bar,
        # reject the candidate entirely.
        if tp_touch and sl_touch:
            return None

        # Before the candidate has won:
        # - any SL kills it immediately
        # - the first TP marks the episode as won and starts post-TP tracking
        if not won:
            if sl_touch:
                return None
            if tp_touch:
                won = True
                best_outcome_pips = fav_now
                best_exit_i = j
            continue

        # After TP has already occurred:
        # - stop scanning when a later SL is touched
        # - otherwise keep updating the best favorable outcome seen so far
        if sl_touch:
            break

        if fav_now > best_outcome_pips:
            best_outcome_pips = fav_now
            best_exit_i = j

    # If TP was never reached before termination, this is not a valid candidate.
    if not won:
        return None

    return {
        "side": side,
        "entry_i": int(entry_i),
        "exit_i": int(best_exit_i),
        "best_outcome_pips": float(best_outcome_pips),
    }

# Enumerate all valid long and short candidates whose full horizon fits in the file.
cand_rows = []
last_entry_i = n - H - 1
for entry_i in range(max(last_entry_i + 1, 0)):
    c_long = build_candidate(entry_i, "long")
    if c_long is not None:
        cand_rows.append(c_long)

    c_short = build_candidate(entry_i, "short")
    if c_short is not None:
        cand_rows.append(c_short)

candidates_df = pd.DataFrame(cand_rows)

# If no valid candidates exist, return empty episode outputs and an all-zero label series.
if len(candidates_df) == 0:
    episodes_df = pd.DataFrame(columns=["episode_id", "side", "entry_t", "exit_t", "best_outcome_pips"])
    y = pd.Series(np.zeros(n, dtype=np.int8), index=df5.index, name="y")
    df_trade = df5.copy()
    df_trade["y"] = y
    print("valid candidates: 0")
    print("retained episodes: 0")
    print("positive labels: 0")
else:
    # Sort candidates for weighted interval scheduling.
    # Primary time axis is exit_i.
    # Ties are broken by earlier entry_i, then higher best_outcome_pips, then side.
    candidates_df = candidates_df.sort_values(
        ["exit_i", "entry_i", "best_outcome_pips", "side"],
        ascending=[True, True, False, True],
        kind="mergesort",
    ).reset_index(drop=True)

    entry_arr = candidates_df["entry_i"].to_numpy(dtype=np.int64)
    exit_arr = candidates_df["exit_i"].to_numpy(dtype=np.int64)
    weight_arr = candidates_df["best_outcome_pips"].to_numpy(dtype=float)

    m = len(candidates_df)

    # p[i] stores the index of the last candidate that ends strictly before
    # candidate i begins. This is the compatibility link used by interval DP.
    p = np.full(m, -1, dtype=np.int64)
    exits = exit_arr.tolist()
    for i in range(m):
        p[i] = bisect_left(exits, entry_arr[i]) - 1

    # Standard weighted interval scheduling DP.
    # dp_weight[i] is the best total weight using the first i sorted candidates.
    # dp_seq[i] stores the retained entry index sequence for deterministic tie breaks.
    # take[i] records whether candidate i-1 was chosen in the forward DP pass.
    dp_weight = np.zeros(m + 1, dtype=float)
    dp_seq = [tuple() for _ in range(m + 1)]
    take = np.zeros(m + 1, dtype=bool)

    for i in range(1, m + 1):
        j = i - 1
        prev_idx = p[j] + 1

        # Include current candidate j and jump back to its compatible predecessor set.
        incl_weight = dp_weight[prev_idx] + weight_arr[j]
        excl_weight = dp_weight[i - 1]

        # Tie breaking uses lexicographic order of retained entry indices.
        # Earlier entry sequences win when total retained weight is equal.
        incl_seq = dp_seq[prev_idx] + (int(entry_arr[j]),)
        excl_seq = dp_seq[i - 1]

        if incl_weight > excl_weight:
            choose_include = True
        elif incl_weight < excl_weight:
            choose_include = False
        else:
            choose_include = incl_seq < excl_seq

        if choose_include:
            dp_weight[i] = incl_weight
            dp_seq[i] = incl_seq
            take[i] = True
        else:
            dp_weight[i] = excl_weight
            dp_seq[i] = excl_seq
            take[i] = False

    # Reconstruct the retained non-overlapping candidate set by walking backward.
    # Reconstruction repeats the same decision logic to remain consistent with the DP.
    kept_idx = []
    i = m
    while i > 0:
        j = i - 1
        prev_idx = p[j] + 1

        incl_weight = dp_weight[prev_idx] + weight_arr[j]
        excl_weight = dp_weight[i - 1]

        incl_seq = dp_seq[prev_idx] + (int(entry_arr[j]),)
        excl_seq = dp_seq[i - 1]

        choose_include = (incl_weight > excl_weight) or (
            np.isclose(incl_weight, excl_weight) and incl_seq < excl_seq
        )

        if choose_include:
            kept_idx.append(j)
            i = prev_idx
        else:
            i -= 1

    kept_idx = kept_idx[::-1]
    kept_df = candidates_df.iloc[kept_idx].copy().reset_index(drop=True)
    kept_df["episode_id"] = np.arange(1, len(kept_df) + 1, dtype=np.int64)

    # Create the retained episode table with timestamps rather than raw indices.
    episodes_df = pd.DataFrame({
        "episode_id": kept_df["episode_id"].to_numpy(),
        "side": kept_df["side"].to_numpy(),
        "entry_t": ts.iloc[kept_df["entry_i"].to_numpy()].to_numpy(),
        "exit_t": ts.iloc[kept_df["exit_i"].to_numpy()].to_numpy(),
        "best_outcome_pips": kept_df["best_outcome_pips"].to_numpy(),
    })

    # Create binary entry labels on the original df5 index.
    # Only retained episode entry bars receive label 1.
    y_arr = np.zeros(n, dtype=np.int8)
    y_arr[kept_df["entry_i"].to_numpy(dtype=np.int64)] = 1
    y = pd.Series(y_arr, index=df5.index, name="y")

    # Copy the source dataframe and append the binary label column.
    df_trade = df5.copy()
    df_trade["y"] = y

    # Print compact retention diagnostics and a small episode preview.
    print("valid candidates:", len(candidates_df))
    print("retained episodes:", len(episodes_df))
    print("positive labels:", int(y.sum()))
    print("total retained best_outcome_pips:", float(episodes_df["best_outcome_pips"].sum()))
    print()
    print(episodes_df.head(20))
    print()
    print(y.value_counts(dropna=False).sort_index())