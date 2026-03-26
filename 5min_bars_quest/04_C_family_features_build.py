# Append Family C build features to the research dataframe.
# This cell adds only the frozen pre-release build objects C1 through C8.
# It uses only approved Family A and Family B inputs already present in df_feat.
# The source 5 minute bar dataframe df5 is not modified.
# New Family C columns are built off-frame, then appended to df_feat in one block.

import numpy as np
import pandas as pd

FAM_C_W_SET = (6, 12, 24)
FAM_C_EPS = 1e-8
FAM_C_THETA_BUILD_SUPPORT = 0.10

# Family C is an append step onto the research dataframe.
if "df_feat" not in globals():
    raise NameError("df_feat is not defined")

# These are the exact frozen upstream inputs allowed for this build.
# The checks below fail fast if any prerequisite Family A or Family B object
# has not been created yet.
required_cols = [
    "famA_scale_pips",
    "m_close",
    "m_high",
    "m_low",
    "famA_side_ret_1_up",
    "famA_side_ret_1_dn",
    "famB_dir_eff_up_w6",
    "famB_dir_eff_up_w12",
    "famB_dir_eff_up_w24",
    "famB_dir_eff_dn_w6",
    "famB_dir_eff_dn_w12",
    "famB_dir_eff_dn_w24",
    "famB_half_progressive_balance_up_w6",
    "famB_half_progressive_balance_dn_w6",
    "famB_quart_progressive_balance_up_w12",
    "famB_quart_progressive_balance_up_w24",
    "famB_quart_progressive_balance_dn_w12",
    "famB_quart_progressive_balance_dn_w24",
    "famB_bend_late_add_up_w6",
    "famB_bend_late_add_up_w12",
    "famB_bend_late_add_up_w24",
    "famB_bend_late_add_dn_w6",
    "famB_bend_late_add_dn_w12",
    "famB_bend_late_add_dn_w24",
    "famB_late_path_dominance_ratio_up_w6",
    "famB_late_path_dominance_ratio_up_w12",
    "famB_late_path_dominance_ratio_up_w24",
    "famB_late_path_dominance_ratio_dn_w6",
    "famB_late_path_dominance_ratio_dn_w12",
    "famB_late_path_dominance_ratio_dn_w24",
]
missing_cols = [c for c in required_cols if c not in df_feat.columns]
if missing_cols:
    raise KeyError(f"Missing required columns in df_feat: {missing_cols}")

# Preload common arrays used across all Family C objects.
# scale_valid is used both at the endpoint row t and across the full trailing window.
n = len(df_feat)
scale_valid = df_feat["famA_scale_pips"].notna().to_numpy()

m_close = df_feat["m_close"].to_numpy(dtype=float)
m_high = df_feat["m_high"].to_numpy(dtype=float)
m_low = df_feat["m_low"].to_numpy(dtype=float)
scale_pips = df_feat["famA_scale_pips"].to_numpy(dtype=float)

# Collect all Family C output arrays here, then materialize one dataframe once.
famC_data = {}

# Build the Family C feature set separately for the up-side and down-side views.
for side in ["up", "dn"]:
    # dir_sign converts anchor separation into a side-relative signed quantity.
    # For up, positive distance from the opposite anchor is positive.
    # For down, the sign is flipped so the same interpretation holds.
    dir_sign = 1.0 if side == "up" else -1.0

    # x_src is the frozen Family A one-bar side-relative increment stream.
    x_src = df_feat[f"famA_side_ret_1_{side}"].to_numpy(dtype=float)

    # Build all C1 through C8 objects for each allowed trailing window.
    for W in FAM_C_W_SET:
        # Preallocate each output vector with NaN so invalid rows remain explicitly missing.
        build_support_persistence = np.full(n, np.nan, dtype=float)
        build_support_clustering = np.full(n, np.nan, dtype=float)
        build_anchor_separation = np.full(n, np.nan, dtype=float)
        build_pullback_acceptability = np.full(n, np.nan, dtype=float)
        build_progressive_balance = np.full(n, np.nan, dtype=float)
        build_late_strengthening = np.full(n, np.nan, dtype=float)
        build_late_dominance = np.full(n, np.nan, dtype=float)
        build_pre_release_efficiency = np.full(n, np.nan, dtype=float)

        # Pull the allowed Family B source objects for this side and window.
        # W=6 uses the half progressive balance object.
        # W in {12, 24} uses the quartile progressive balance object.
        dir_eff_src = df_feat[f"famB_dir_eff_{side}_w{W}"].to_numpy(dtype=float)
        if W == 6:
            progressive_balance_src = df_feat[f"famB_half_progressive_balance_{side}_w6"].to_numpy(dtype=float)
        else:
            progressive_balance_src = df_feat[f"famB_quart_progressive_balance_{side}_w{W}"].to_numpy(dtype=float)

        late_strengthening_src = df_feat[f"famB_bend_late_add_{side}_w{W}"].to_numpy(dtype=float)
        late_dominance_src = df_feat[f"famB_late_path_dominance_ratio_{side}_w{W}"].to_numpy(dtype=float)

        # Populate row t from the trailing window [t-W+1, ..., t].
        # Validity requires a valid scale at the endpoint and across the full window.
        for t in range(W - 1, n):
            if not scale_valid[t]:
                continue
            if not scale_valid[t - W + 1 : t + 1].all():
                continue

            # x is the ordered side-relative increment path over the window.
            # Any NaN inside the path invalidates the entire row for this window.
            x = x_src[t - W + 1 : t + 1]
            if np.isnan(x).any():
                continue

            # C1 build_support_persistence
            # Share of bars in the window whose side-relative increment exceeds
            # the frozen support threshold. This measures how persistently the
            # build advances rather than merely whether it advances at all.
            support_mask = x > FAM_C_THETA_BUILD_SUPPORT
            build_support_persistence[t] = support_mask.sum() / W

            # C2 build_support_clustering
            # Longest contiguous run of support bars, normalized by window length.
            # This captures whether support is concentrated into a cluster rather
            # than scattered across isolated bars.
            max_run_len = 0
            j = 0
            while j < W:
                if x[j] > FAM_C_THETA_BUILD_SUPPORT:
                    run_len = 0
                    while j < W and x[j] > FAM_C_THETA_BUILD_SUPPORT:
                        run_len += 1
                        j += 1
                    if run_len > max_run_len:
                        max_run_len = run_len
                else:
                    j += 1
            build_support_clustering[t] = max_run_len / W if max_run_len > 0 else 0.0

            # C3 build_anchor_separation
            # Measure current close against the opposite-side anchor inside the window.
            # For up, the opposite anchor is the lowest low in the window.
            # For down, the opposite anchor is the highest high in the window.
            # The result is normalized by the frozen scale at row t and signed
            # so that stronger same-side separation is positive for both sides.
            if side == "up":
                opp_anchor = np.min(m_low[t - W + 1 : t + 1])
            else:
                opp_anchor = np.max(m_high[t - W + 1 : t + 1])
            build_anchor_separation[t] = dir_sign * (m_close[t] - opp_anchor) / (0.0001 * scale_pips[t])

            # C4 build_pullback_acceptability
            # Compare average adverse excursion magnitude against average supportive
            # excursion magnitude, but only using bars that meaningfully exceed the
            # same frozen threshold on either side.
            # If no support bars exist, the metric is undefined.
            # If support exists but no meaningful damage exists, acceptability is 1.0.
            damage_mask = x < -FAM_C_THETA_BUILD_SUPPORT

            if support_mask.sum() == 0:
                build_pullback_acceptability[t] = np.nan
            else:
                mean_support_mag = x[support_mask].mean()
                if damage_mask.sum() == 0:
                    build_pullback_acceptability[t] = 1.0
                else:
                    mean_damage_mag = np.abs(x[damage_mask]).mean()
                    ratio = mean_damage_mag / (mean_support_mag + FAM_C_EPS)
                    build_pullback_acceptability[t] = 1.0 - min(ratio, 2.0) / 2.0

            # C5 through C8 are direct uses or simple composites of frozen Family B objects.
            # C5 build_progressive_balance
            # C6 build_late_strengthening
            # C7 build_late_dominance
            # C8 build_pre_release_efficiency = directional efficiency times support persistence
            build_progressive_balance[t] = progressive_balance_src[t]
            build_late_strengthening[t] = late_strengthening_src[t]
            build_late_dominance[t] = late_dominance_src[t]
            build_pre_release_efficiency[t] = dir_eff_src[t] * build_support_persistence[t]

        # Store the completed arrays under their final Family C column names.
        famC_data[f"famC_build_support_persistence_{side}_w{W}"] = build_support_persistence
        famC_data[f"famC_build_support_clustering_{side}_w{W}"] = build_support_clustering
        famC_data[f"famC_build_anchor_separation_{side}_w{W}"] = build_anchor_separation
        famC_data[f"famC_build_pullback_acceptability_{side}_w{W}"] = build_pullback_acceptability
        famC_data[f"famC_build_progressive_balance_{side}_w{W}"] = build_progressive_balance
        famC_data[f"famC_build_late_strengthening_{side}_w{W}"] = build_late_strengthening
        famC_data[f"famC_build_late_dominance_{side}_w{W}"] = build_late_dominance
        famC_data[f"famC_build_pre_release_efficiency_{side}_w{W}"] = build_pre_release_efficiency

# Materialize the Family C block once and append it once to df_feat.
famC_df = pd.DataFrame(famC_data, index=df_feat.index)
df_feat = pd.concat([df_feat, famC_df], axis=1).copy()

famC_cols = list(famC_df.columns)

print("Family C columns added:", len(famC_cols))

# Audit view for quick inspection of representative Family C outputs.
# Mix short-window up-side objects with longer-window down-side objects
# so both ends of the build are easy to spot-check.
audit_cols = [
    "famC_build_support_persistence_up_w6",
    "famC_build_support_clustering_up_w6",
    "famC_build_anchor_separation_up_w6",
    "famC_build_pullback_acceptability_up_w6",
    "famC_build_progressive_balance_up_w6",
    "famC_build_late_strengthening_up_w6",
    "famC_build_late_dominance_up_w6",
    "famC_build_pre_release_efficiency_up_w6",
    "famC_build_support_persistence_dn_w24",
    "famC_build_progressive_balance_dn_w24",
    "famC_build_late_strengthening_dn_w24",
    "famC_build_late_dominance_dn_w24",
    "famC_build_pre_release_efficiency_dn_w24",
]
audit_cols = list(dict.fromkeys(audit_cols))

print(df_feat[audit_cols].head(40))