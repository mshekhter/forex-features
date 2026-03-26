# Append the Family B addon to the research dataframe.
# This cell adds only the deferred ordered-path addon features that were
# intentionally left out of the reduced Family B core.
#
# The addon covers five frozen objects:
# B23  half-window support balance
# B24  quartile support balance
# B25  fixed-third endpoint slopes
# B26  bend metrics derived from those third-slopes
# B27  late-path dominance ratio
#
# The source price dataframe df5 is not touched here.
# New columns are computed from existing Family A side-relative one-bar
# increments and appended to df_feat in one block at the end.

import numpy as np
import pandas as pd

FAM_B_ADD_W_HALF = (6, 12, 24)
FAM_B_ADD_W_QUART = (12, 24)
FAM_B_ADD_W_THIRD = (6, 12, 24)
FAM_B_ADD_W_LATE = (6, 12, 24)
FAM_B_ADD_EPS = 1e-8

# The research dataframe must already exist before this addon can be appended.
if "df_feat" not in globals():
    raise NameError("df_feat is not defined")

# The addon depends only on the frozen Family A scale-validity mask and the
# side-relative one-bar increment streams for up and down.
required_cols = [
    "famA_scale_pips",
    "famA_side_ret_1_up",
    "famA_side_ret_1_dn",
]
missing_cols = [c for c in required_cols if c not in df_feat.columns]
if missing_cols:
    raise KeyError(f"Missing required Family A columns in df_feat: {missing_cols}")

# n is the total number of rows in the research dataframe.
# scale_valid marks rows where the frozen Family A scale exists.
# Every addon feature for a given trailing window requires valid scale on the
# full window, not only on the endpoint row.
n = len(df_feat)
scale_valid = df_feat["famA_scale_pips"].notna().to_numpy()

# Collect every addon output array here first, then build a dataframe once and
# append it once. This keeps the append step clean and avoids piecemeal growth.
famB_add_data = {}

# Build the addon features separately for the up-side and down-side increment streams.
for side in ["up", "dn"]:
    x_src = df_feat[f"famA_side_ret_1_{side}"].to_numpy(dtype=float)

    # B23, B25, B26, B27
    # These are defined for W in {6, 12, 24}.
    # For each valid row t, the ordered window is x[t-W+1 : t+1].
    for W in (6, 12, 24):
        half_balance_first = np.full(n, np.nan, dtype=float)
        half_balance_second = np.full(n, np.nan, dtype=float)
        half_progressive_balance = np.full(n, np.nan, dtype=float)

        slope_1_add = np.full(n, np.nan, dtype=float)
        slope_2_add = np.full(n, np.nan, dtype=float)
        slope_3_add = np.full(n, np.nan, dtype=float)
        bend_early_add = np.full(n, np.nan, dtype=float)
        bend_late_add = np.full(n, np.nan, dtype=float)
        bend_total_add = np.full(n, np.nan, dtype=float)

        late_support_ratio = np.full(n, np.nan, dtype=float)
        window_support_ratio_local = np.full(n, np.nan, dtype=float)
        late_path_dominance_ratio = np.full(n, np.nan, dtype=float)

        # half splits the ordered window into an early half and a late half.
        # third splits the window into three equal fixed segments.
        half = W // 2
        third = W // 3

        # Populate row t only when the full trailing window is valid.
        for t in range(W - 1, n):
            if not scale_valid[t - W + 1 : t + 1].all():
                continue

            # x is the ordered side-relative one-bar increment sequence over the window.
            if np.isnan(x).any():
                continue

            # cp is the ordered cumulative path built from x.
            # cp[0] is anchored at zero, and cp[j] is the cumulative total of
            # the first j increments. It is used for the fixed-third slope features.
            cp = np.empty(W + 1, dtype=float)
            cp[0] = 0.0
            cp[1:] = np.cumsum(x)

            # B23 half-window support balance
            # Compute support minus damage separately for the first half and
            # second half of the ordered increment stream, then compare them.
            x_first = x[:half]
            x_second = x[half:]

            half_support_first = np.maximum(x_first, 0.0).sum()
            half_damage_first = np.maximum(-x_first, 0.0).sum()
            half_support_second = np.maximum(x_second, 0.0).sum()
            half_damage_second = np.maximum(-x_second, 0.0).sum()

            hb_first = half_support_first - half_damage_first
            hb_second = half_support_second - half_damage_second

            half_balance_first[t] = hb_first
            half_balance_second[t] = hb_second
            half_progressive_balance[t] = hb_second - hb_first

            # B25 fixed-third endpoint slopes
            # Use cumulative-path endpoints at 0, W/3, 2W/3, and W.
            # Each slope is the average rise per bar across one third of the window.
            cp_w3 = cp[third]
            cp_2w3 = cp[2 * third]
            cp_w = cp[W]

            s1 = (cp_w3 - cp[0]) / third
            s2 = (cp_2w3 - cp_w3) / third
            s3 = (cp_w - cp_2w3) / third

            slope_1_add[t] = s1
            slope_2_add[t] = s2
            slope_3_add[t] = s3

            # B26 bend metrics
            # These describe how the third-segment slopes change from early
            # to middle, middle to late, and early to late.
            bend_early_add[t] = s2 - s1
            bend_late_add[t] = s3 - s2
            bend_total_add[t] = s3 - s1

            # B27 late-path dominance ratio
            # Compare support ratio in the final third of the window against
            # support ratio across the entire window.
            # Positive values mean the late path is more support-dominant than
            # the window as a whole.
            x_last_third = x[-third:]
            late_support = np.maximum(x_last_third, 0.0).sum()
            late_damage = np.maximum(-x_last_third, 0.0).sum()
            late_sr = late_support / (late_support + late_damage + FAM_B_ADD_EPS)

            window_support = np.maximum(x, 0.0).sum()
            window_damage = np.maximum(-x, 0.0).sum()
            win_sr = window_support / (window_support + window_damage + FAM_B_ADD_EPS)

            late_support_ratio[t] = late_sr
            window_support_ratio_local[t] = win_sr
            late_path_dominance_ratio[t] = late_sr - win_sr

        # Store the completed arrays under their final addon column names.
        famB_add_data[f"famB_half_balance_first_{side}_w{W}"] = half_balance_first
        famB_add_data[f"famB_half_balance_second_{side}_w{W}"] = half_balance_second
        famB_add_data[f"famB_half_progressive_balance_{side}_w{W}"] = half_progressive_balance

        famB_add_data[f"famB_slope_1_add_{side}_w{W}"] = slope_1_add
        famB_add_data[f"famB_slope_2_add_{side}_w{W}"] = slope_2_add
        famB_add_data[f"famB_slope_3_add_{side}_w{W}"] = slope_3_add
        famB_add_data[f"famB_bend_early_add_{side}_w{W}"] = bend_early_add
        famB_add_data[f"famB_bend_late_add_{side}_w{W}"] = bend_late_add
        famB_add_data[f"famB_bend_total_add_{side}_w{W}"] = bend_total_add

        famB_add_data[f"famB_late_support_ratio_{side}_w{W}"] = late_support_ratio
        famB_add_data[f"famB_window_support_ratio_local_{side}_w{W}"] = window_support_ratio_local
        famB_add_data[f"famB_late_path_dominance_ratio_{side}_w{W}"] = late_path_dominance_ratio

    # B24 quartile support balance only for W in {12, 24}
    # These windows divide evenly into four ordered quartiles.
    for W in (12, 24):
        quart_balance_1 = np.full(n, np.nan, dtype=float)
        quart_balance_2 = np.full(n, np.nan, dtype=float)
        quart_balance_3 = np.full(n, np.nan, dtype=float)
        quart_balance_4 = np.full(n, np.nan, dtype=float)

        quart_progressive_balance = np.full(n, np.nan, dtype=float)
        quart_late_vs_early_ratio = np.full(n, np.nan, dtype=float)

        # qlen is the size of each quartile segment.
        qlen = W // 4

        for t in range(W - 1, n):
            if not scale_valid[t - W + 1 : t + 1].all():
                continue

            # x is the ordered increment window, then split into four equal quartiles.
            x = x_src[t - W + 1 : t + 1]
            if np.isnan(x).any():
                continue

            x_q1 = x[0:qlen]
            x_q2 = x[qlen:2 * qlen]
            x_q3 = x[2 * qlen:3 * qlen]
            x_q4 = x[3 * qlen:4 * qlen]

            # Helper for support balance inside one quartile:
            # positive flow minus negative flow magnitude.
            def _quart_balance(arr):
                return np.maximum(arr, 0.0).sum() - np.maximum(-arr, 0.0).sum()

            qb1 = _quart_balance(x_q1)
            qb2 = _quart_balance(x_q2)
            qb3 = _quart_balance(x_q3)
            qb4 = _quart_balance(x_q4)

            # Also retain raw supportive flow per quartile so late-vs-early
            # supportive concentration can be compared as a ratio.
            qs1 = np.maximum(x_q1, 0.0).sum()
            qs2 = np.maximum(x_q2, 0.0).sum()
            qs3 = np.maximum(x_q3, 0.0).sum()
            qs4 = np.maximum(x_q4, 0.0).sum()

            quart_balance_1[t] = qb1
            quart_balance_2[t] = qb2
            quart_balance_3[t] = qb3
            quart_balance_4[t] = qb4

            # Progressive balance compares late half of the window quartiles
            # against early half of the window quartiles.
            quart_progressive_balance[t] = (qb3 + qb4) - (qb1 + qb2)

            # Late-vs-early supportive ratio compares supportive flow in the
            # last two quartiles against the first two quartiles.
            quart_late_vs_early_ratio[t] = (qs3 + qs4 + FAM_B_ADD_EPS) / (qs1 + qs2 + FAM_B_ADD_EPS)

        famB_add_data[f"famB_quart_balance_1_{side}_w{W}"] = quart_balance_1
        famB_add_data[f"famB_quart_balance_2_{side}_w{W}"] = quart_balance_2
        famB_add_data[f"famB_quart_balance_3_{side}_w{W}"] = quart_balance_3
        famB_add_data[f"famB_quart_balance_4_{side}_w{W}"] = quart_balance_4
        famB_add_data[f"famB_quart_progressive_balance_{side}_w{W}"] = quart_progressive_balance
        famB_add_data[f"famB_quart_late_vs_early_ratio_{side}_w{W}"] = quart_late_vs_early_ratio

# Materialize the addon block once and append it once to df_feat.
famB_add_df = pd.DataFrame(famB_add_data, index=df_feat.index)
df_feat = pd.concat([df_feat, famB_add_df], axis=1).copy()

famB_add_cols = list(famB_add_df.columns)

print("Family B addon columns added:", len(famB_add_cols))

# Quick audit slice for spot-checking representative addon outputs.
# The selection mixes short-window up-side examples and longer-window down-side
# examples so the appended block can be visually inspected.
audit_cols = [
    "famB_half_balance_first_up_w6",
    "famB_half_balance_second_up_w6",
    "famB_half_progressive_balance_up_w6",
    "famB_slope_1_add_up_w6",
    "famB_slope_2_add_up_w6",
    "famB_slope_3_add_up_w6",
    "famB_bend_early_add_up_w6",
    "famB_bend_late_add_up_w6",
    "famB_bend_total_add_up_w6",
    "famB_late_support_ratio_up_w6",
    "famB_window_support_ratio_local_up_w6",
    "famB_late_path_dominance_ratio_up_w6",
    "famB_quart_balance_1_up_w12",
    "famB_quart_balance_2_up_w12",
    "famB_quart_balance_3_up_w12",
    "famB_quart_balance_4_up_w12",
    "famB_quart_progressive_balance_up_w12",
    "famB_quart_late_vs_early_ratio_up_w12",
    "famB_bend_total_add_dn_w24",
    "famB_late_path_dominance_ratio_dn_w24",
]
audit_cols = list(dict.fromkeys(audit_cols))

print(df_feat[audit_cols].head(40))