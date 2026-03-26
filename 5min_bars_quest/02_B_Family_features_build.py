# Append reduced Family B ordered path-geometry features to the research dataframe.
# Work only on df_feat, which is the research frame intended to accumulate
# feature families. The source price dataframe remains untouched elsewhere.
#
# Rebuild policy:
# 1. Remove any previously appended Family B columns from df_feat.
# 2. Recompute the entire Family B block off-frame in a separate dictionary.
# 3. Append the finished Family B block once at the end.
#
# This avoids duplicate columns, partial rebuilds, and column fragmentation.
#
# Family B uses the already-frozen Family A primitives:
# - famA_scale_pips for validity
# - famA_side_ret_1_up
# - famA_side_ret_1_dn
#
# For each side and trailing window, it forms an ordered sequence of
# side-relative one-bar increments x over the last W completed bars, then
# constructs a cumulative path cp where:
# - cp[0] = 0
# - cp[j] = cumulative sum of the first j increments
#
# All features in this block are derived from that ordered cumulative path.

import numpy as np
import pandas as pd

FAM_B_W_SET = (6, 12, 24)
FAM_B_EPS = 1e-8
FAM_B_EPS_POS = 1e-8
FAM_B_THETA_INTERRUPT = 0.25
FAM_B_THETA_SIGN = 0.05

# Family B must append onto an existing research dataframe.
if "df_feat" not in globals():
    raise NameError("df_feat is not defined")

# These Family A columns are the frozen inputs required by Family B.
required_cols = [
    "famA_scale_pips",
    "famA_side_ret_1_up",
    "famA_side_ret_1_dn",
]
missing_cols = [c for c in required_cols if c not in df_feat.columns]
if missing_cols:
    raise KeyError(f"Missing required Family A columns in df_feat: {missing_cols}")

# Remove any previous Family B block so the rebuild is clean and non-duplicative.
old_famB_cols = [c for c in df_feat.columns if c.startswith("famB_")]
if old_famB_cols:
    df_feat = df_feat.drop(columns=old_famB_cols).copy()

# Global row count and a convenience mask for rows where the Family A scale exists.
# A Family B window is valid only if every bar in that window has valid scale.
n = len(df_feat)
scale_valid = df_feat["famA_scale_pips"].notna().to_numpy()

# Store all Family B output arrays here, then materialize once as a dataframe.
famB_data = {}

# Small helper for the segment slopes.
# It returns the ordinary least squares slope of y versus x.
# This is used on cumulative-path segments to measure local path inclination.
def _ols_slope(xv, yv):
    xv_mean = xv.mean()
    yv_mean = yv.mean()
    denom = np.sum((xv - xv_mean) ** 2)
    if denom <= 0:
        return np.nan
    return np.sum((xv - xv_mean) * (yv - yv_mean)) / denom

# Build Family B separately for the up-side and down-side ordered increment streams.
for side in ["up", "dn"]:
    x_src = df_feat[f"famA_side_ret_1_{side}"].to_numpy(dtype=float)

    # Build the full reduced Family B feature set for each trailing window.
    for W in FAM_B_W_SET:
        # Preallocate one output vector per feature.
        # Keep NaN default so invalid or unavailable rows remain explicitly missing.
        terminal_disp = np.full(n, np.nan, dtype=float)
        path_length = np.full(n, np.nan, dtype=float)
        dir_eff = np.full(n, np.nan, dtype=float)
        disp_first_half = np.full(n, np.nan, dtype=float)
        disp_second_half = np.full(n, np.nan, dtype=float)
        progress_tilt = np.full(n, np.nan, dtype=float)

        max_pullback_from_peak = np.full(n, np.nan, dtype=float)
        final_pullback_from_peak = np.full(n, np.nan, dtype=float)
        pullback_retention_ratio = np.full(n, np.nan, dtype=float)

        monotone_segment_concentration = np.full(n, np.nan, dtype=float)
        damaging_interruption_count = np.full(n, np.nan, dtype=float)
        sign_change_density = np.full(n, np.nan, dtype=float)

        slope_1 = np.full(n, np.nan, dtype=float)
        slope_2 = np.full(n, np.nan, dtype=float)
        slope_3 = np.full(n, np.nan, dtype=float)
        bend_early = np.full(n, np.nan, dtype=float)
        bend_late = np.full(n, np.nan, dtype=float)
        bend_total = np.full(n, np.nan, dtype=float)

        new_ground_count = np.full(n, np.nan, dtype=float)
        new_ground_frequency = np.full(n, np.nan, dtype=float)
        stalled_tail_length = np.full(n, np.nan, dtype=float)

        recovery_after_deepest_pullback = np.full(n, np.nan, dtype=float)
        recovery_after_deepest_pullback_ratio = np.full(n, np.nan, dtype=float)

        # Derived partition sizes used by this frozen reduced Family B definition.
        # half is used for first-half vs second-half displacement.
        # third is used for the three cumulative-path slope segments.
        half = W // 2
        third = W // 3

        # t is the row being populated.
        # The ordered window is [t-W+1, ..., t], inclusive, using completed bars only.
        for t in range(W - 1, n):
            # Require valid Family A scale over the full trailing window.
            if not scale_valid[t - W + 1 : t + 1].all():
                continue

            # Extract the ordered increment path for this side and window.
            x = x_src[t - W + 1 : t + 1]
            if np.isnan(x).any():
                continue

            # Build cumulative path cp with an explicit zero start:
            # cp[0] = 0
            # cp[1:] = cumulative sum of x
            #
            # cp therefore has length W+1 and represents the ordered path shape.
            cp = np.empty(W + 1, dtype=float)
            cp[0] = 0.0
            cp[1:] = np.cumsum(x)

            # Terminal displacement is the net ordered move across the whole window.
            # Path length is the total realized movement, ignoring sign.
            terminal = cp[-1]
            plen = np.abs(x).sum()

            terminal_disp[t] = terminal
            path_length[t] = plen
            dir_eff[t] = terminal / (plen + FAM_B_EPS)

            # Split total progress into first-half and second-half displacement.
            # progress_tilt is positive when later progress exceeds earlier progress.
            first_half_disp = cp[half]
            second_half_disp = terminal - first_half_disp
            disp_first_half[t] = first_half_disp
            disp_second_half[t] = second_half_disp
            progress_tilt[t] = second_half_disp - first_half_disp

            # Pullback logic is defined relative to the running peak of the cumulative path.
            # dd_from_peak measures how far below the best-so-far level the path is at each point.
            run_peak = np.maximum.accumulate(cp)
            dd_from_peak = run_peak - cp
            max_dd = dd_from_peak.max()

            # peak_level is the best cumulative-path level reached anywhere in the window.
            # final_pb is how far the final point sits below that peak.
            peak_level = cp.max()
            final_pb = peak_level - terminal

            max_pullback_from_peak[t] = max_dd
            final_pullback_from_peak[t] = final_pb

            # Retention ratio keeps only cases with a positive peak level,
            # since the denominator is interpreted as peak progress achieved.
            if peak_level > 0.0:
                pullback_retention_ratio[t] = terminal / (peak_level + FAM_B_EPS_POS)

            # Supportive-run concentration:
            # identify contiguous positive runs in x, sum each run,
            # then measure how dominant the largest supportive run is
            # relative to total supportive flow in the window.
            supportive_runs = []
            j = 0
            while j < W:
                if x[j] > 0.0:
                    run_sum = 0.0
                    while j < W and x[j] > 0.0:
                        run_sum += x[j]
                        j += 1
                    supportive_runs.append(run_sum)
                else:
                    j += 1

            if len(supportive_runs) == 0:
                monotone_segment_concentration[t] = 0.0
            else:
                total_support = np.sum(supportive_runs)
                monotone_segment_concentration[t] = np.max(supportive_runs) / (total_support + FAM_B_EPS)

            # Damaging interruptions:
            # count contiguous negative runs whose absolute size is at least
            # the frozen interruption threshold.
            dmg_count = 0
            j = 0
            while j < W:
                if x[j] < 0.0:
                    run_abs_sum = 0.0
                    while j < W and x[j] < 0.0:
                        run_abs_sum += -x[j]
                        j += 1
                    if run_abs_sum >= FAM_B_THETA_INTERRUPT:
                        dmg_count += 1
                else:
                    j += 1
            damaging_interruption_count[t] = dmg_count

            # Sign-change density:
            # first filter out very small increments near zero using FAM_B_THETA_SIGN,
            # then compute how often the remaining sign sequence flips.
            filt_sign = np.zeros(W, dtype=int)
            filt_sign[x > FAM_B_THETA_SIGN] = 1
            filt_sign[x < -FAM_B_THETA_SIGN] = -1
            nz = filt_sign[filt_sign != 0]

            if len(nz) >= 2:
                valid_comparisons = len(nz) - 1
                sign_changes = np.sum(nz[1:] != nz[:-1])
                sign_change_density[t] = sign_changes / max(valid_comparisons, 1)
            else:
                sign_change_density[t] = 0.0

            # Divide the cumulative path into three fixed ordered segments and
            # fit an OLS slope within each segment.
            #
            # The x-axis is the step index in the cumulative path.
            # The y-axis is the cumulative-path level over that segment.
            j_seg1 = np.arange(1, third + 1, dtype=float)
            y_seg1 = cp[1 : third + 1]

            j_seg2 = np.arange(third + 1, 2 * third + 1, dtype=float)
            y_seg2 = cp[third + 1 : 2 * third + 1]

            j_seg3 = np.arange(2 * third + 1, W + 1, dtype=float)
            y_seg3 = cp[2 * third + 1 : W + 1]

            s1 = _ols_slope(j_seg1, y_seg1)
            s2 = _ols_slope(j_seg2, y_seg2)
            s3 = _ols_slope(j_seg3, y_seg3)

            # Bend metrics describe how the path slope changes through the window.
            slope_1[t] = s1
            slope_2[t] = s2
            slope_3[t] = s3
            bend_early[t] = s2 - s1
            bend_late[t] = s3 - s2
            bend_total[t] = s3 - s1

            # New-ground count:
            # count how many cumulative-path positions exceed all prior positions.
            running_best = cp[0]
            ng_count = 0
            for j in range(1, W + 1):
                if cp[j] > running_best:
                    ng_count += 1
                    running_best = cp[j]
            new_ground_count[t] = ng_count
            new_ground_frequency[t] = ng_count / W

            # Stalled tail length:
            # locate the final occurrence of the peak level,
            # then count how many steps remain after that peak.
            last_peak_idx = np.where(cp == peak_level)[0].max()
            stalled_tail_length[t] = W - last_peak_idx

            # Recovery after deepest pullback:
            # find the first point where drawdown from the running peak is maximal,
            # then measure how much the path has recovered from that low point
            # by the final step.
            j_star = np.where(dd_from_peak == max_dd)[0][0]
            recovery = terminal - cp[j_star]
            recovery_after_deepest_pullback[t] = recovery
            recovery_after_deepest_pullback_ratio[t] = recovery / (max_dd + FAM_B_EPS)

        # Store every completed feature vector under its final Family B column name.
        famB_data[f"famB_terminal_disp_{side}_w{W}"] = terminal_disp
        famB_data[f"famB_path_length_{side}_w{W}"] = path_length
        famB_data[f"famB_dir_eff_{side}_w{W}"] = dir_eff
        famB_data[f"famB_disp_first_half_{side}_w{W}"] = disp_first_half
        famB_data[f"famB_disp_second_half_{side}_w{W}"] = disp_second_half
        famB_data[f"famB_progress_tilt_{side}_w{W}"] = progress_tilt

        famB_data[f"famB_max_pullback_from_peak_{side}_w{W}"] = max_pullback_from_peak
        famB_data[f"famB_final_pullback_from_peak_{side}_w{W}"] = final_pullback_from_peak
        famB_data[f"famB_pullback_retention_ratio_{side}_w{W}"] = pullback_retention_ratio

        famB_data[f"famB_monotone_segment_concentration_{side}_w{W}"] = monotone_segment_concentration
        famB_data[f"famB_damaging_interruption_count_{side}_w{W}"] = damaging_interruption_count
        famB_data[f"famB_sign_change_density_{side}_w{W}"] = sign_change_density

        famB_data[f"famB_slope_1_{side}_w{W}"] = slope_1
        famB_data[f"famB_slope_2_{side}_w{W}"] = slope_2
        famB_data[f"famB_slope_3_{side}_w{W}"] = slope_3
        famB_data[f"famB_bend_early_{side}_w{W}"] = bend_early
        famB_data[f"famB_bend_late_{side}_w{W}"] = bend_late
        famB_data[f"famB_bend_total_{side}_w{W}"] = bend_total

        famB_data[f"famB_new_ground_count_{side}_w{W}"] = new_ground_count
        famB_data[f"famB_new_ground_frequency_{side}_w{W}"] = new_ground_frequency
        famB_data[f"famB_stalled_tail_length_{side}_w{W}"] = stalled_tail_length

        famB_data[f"famB_recovery_after_deepest_pullback_{side}_w{W}"] = recovery_after_deepest_pullback
        famB_data[f"famB_recovery_after_deepest_pullback_ratio_{side}_w{W}"] = recovery_after_deepest_pullback_ratio

# Materialize the full Family B block once and append it to df_feat.
famB_df = pd.DataFrame(famB_data, index=df_feat.index)
df_feat = pd.concat([df_feat, famB_df], axis=1).copy()

famB_cols = list(famB_df.columns)

print("Family B columns added:", len(famB_cols))

# Small audit view for spot-checking representative Family B outputs.
# The selection includes early-window up-side examples and a few long-window
# down-side examples so the appended block can be inspected quickly.
audit_cols = [
    "famB_terminal_disp_up_w6",
    "famB_path_length_up_w6",
    "famB_dir_eff_up_w6",
    "famB_progress_tilt_up_w6",
    "famB_max_pullback_from_peak_up_w6",
    "famB_final_pullback_from_peak_up_w6",
    "famB_pullback_retention_ratio_up_w6",
    "famB_monotone_segment_concentration_up_w6",
    "famB_damaging_interruption_count_up_w6",
    "famB_sign_change_density_up_w6",
    "famB_slope_1_up_w6",
    "famB_slope_2_up_w6",
    "famB_slope_3_up_w6",
    "famB_bend_early_up_w6",
    "famB_bend_late_up_w6",
    "famB_bend_total_up_w6",
    "famB_new_ground_frequency_up_w6",
    "famB_stalled_tail_length_up_w6",
    "famB_recovery_after_deepest_pullback_ratio_up_w6",
    "famB_terminal_disp_dn_w24",
    "famB_dir_eff_dn_w24",
    "famB_bend_total_dn_w24",
]
audit_cols = list(dict.fromkeys(audit_cols))

print(df_feat[audit_cols].head(35))