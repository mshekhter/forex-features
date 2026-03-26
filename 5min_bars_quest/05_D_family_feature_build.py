# Append Family D release-transition features to the research dataframe.
# This block creates the frozen Family D release objects, whose purpose is to
# describe the shift from organized pre-release build into actual outward move.
# The intent is to capture whether the current bar is behaving like a true
# directional release rather than remaining in the build phase.
#
# Upstream dependency rules:
# - Use only already-frozen upstream objects from Families A and C.
# - Do not derive any new inputs outside those approved sources.
# - Preserve the source df5 entirely.
# - Build all new Family D columns off-frame, then append them to df_feat once.
#
# Window conventions used here:
# - FAM_D_W_REL_SHORT = 3 is the very local takeover window.
# - FAM_D_W_REL_CONTEXT = 12 is the recent containment context window.

import numpy as np
import pandas as pd

FAM_D_EPS = 1e-8
FAM_D_W_REL_SHORT = 3
FAM_D_W_REL_CONTEXT = 12

# Family D is an append step onto the existing research dataframe.
if "df_feat" not in globals():
    raise NameError("df_feat is not defined")

# Exact frozen prerequisites for this family.
# This includes raw mid-price geometry, Family A one-bar normalized objects,
# rejection components, and the Family C pre-release efficiency object C8 at W=12.
required_cols = [
    "m_open", "m_high", "m_low", "m_close",
    "famA_scale_pips",
    "famA_upper_wick_norm", "famA_lower_wick_norm",
    "famA_side_ret_1_up", "famA_side_ret_1_dn",
    "famA_side_body_up", "famA_side_body_dn",
    "famA_adverse_rejection_component_up", "famA_adverse_rejection_component_dn",
    "famC_build_pre_release_efficiency_up_w12",
    "famC_build_pre_release_efficiency_dn_w12",
]
missing_cols = [c for c in required_cols if c not in df_feat.columns]
if missing_cols:
    raise KeyError(f"Missing required columns in df_feat: {missing_cols}")

# Pull commonly used arrays once.
# These are the base numeric inputs used throughout the Family D construction.
m_open = df_feat["m_open"].to_numpy(dtype=float)
m_high = df_feat["m_high"].to_numpy(dtype=float)
m_low = df_feat["m_low"].to_numpy(dtype=float)
m_close = df_feat["m_close"].to_numpy(dtype=float)
scale_pips = df_feat["famA_scale_pips"].to_numpy(dtype=float)

side_ret_up = df_feat["famA_side_ret_1_up"].to_numpy(dtype=float)
side_ret_dn = df_feat["famA_side_ret_1_dn"].to_numpy(dtype=float)
side_body_up = df_feat["famA_side_body_up"].to_numpy(dtype=float)
side_body_dn = df_feat["famA_side_body_dn"].to_numpy(dtype=float)

rej_up = df_feat["famA_adverse_rejection_component_up"].to_numpy(dtype=float)
rej_dn = df_feat["famA_adverse_rejection_component_dn"].to_numpy(dtype=float)

c8_up_w12 = df_feat["famC_build_pre_release_efficiency_up_w12"].to_numpy(dtype=float)
c8_dn_w12 = df_feat["famC_build_pre_release_efficiency_dn_w12"].to_numpy(dtype=float)

n = len(df_feat)
famD_data = {}

# Shared primitive: clv
# CLV = close location value inside the bar.
# It measures where the close sits within the high-low range:
# - near 1.0 means close is near the high
# - near 0.0 means close is near the low
# - 0.5 is used for zero-range bars to avoid division by zero
bar_range = m_high - m_low
clv = np.where(bar_range == 0.0, 0.5, (m_close - m_low) / bar_range)
famD_data["famD_clv"] = clv

# Build the same Family D release objects separately for up-side and down-side views.
for side in ["up", "dn"]:
    # dir_sign makes signed distance measures comparable across sides.
    # For up, favorable expansion is positive.
    # For down, the sign is flipped so favorable expansion is also positive.
    dir_sign = 1.0 if side == "up" else -1.0

    # Select the side-relative primitives for the current side.
    # These are already normalized and frozen upstream.
    side_ret = side_ret_up if side == "up" else side_ret_dn
    side_body = side_body_up if side == "up" else side_body_dn
    side_rej = rej_up if side == "up" else rej_dn
    c8_w12 = c8_up_w12 if side == "up" else c8_dn_w12

    # Preallocate each Family D output vector with NaN so invalid rows remain missing.
    release_expansion_shock = np.full(n, np.nan, dtype=float)
    release_body_shock = np.full(n, np.nan, dtype=float)
    release_acceptance_quality = np.full(n, np.nan, dtype=float)
    release_containment_escape = np.full(n, np.nan, dtype=float)
    release_local_takeover_balance_3 = np.full(n, np.nan, dtype=float)
    release_transition_alignment = np.full(n, np.nan, dtype=float)
    release_rejection_burden = np.full(n, np.nan, dtype=float)
    release_decisiveness = np.full(n, np.nan, dtype=float)

    # Family D is row-local except where explicit context windows are required.
    for t in range(n):
        # If the current row lacks core normalized inputs, skip it entirely.
        if np.isnan(scale_pips[t]) or np.isnan(side_ret[t]) or np.isnan(side_body[t]) or np.isnan(side_rej[t]):
            continue

        # D1
        # Positive side-relative return only.
        # This measures immediate expansion shock in the favorable direction.
        d1 = max(side_ret[t], 0.0)
        release_expansion_shock[t] = d1

        # D2
        # Positive side-relative body only.
        # This isolates decisive real-body participation in the release bar.
        d2 = max(side_body[t], 0.0)
        release_body_shock[t] = d2

        # D7
        # Carry forward the side-relative adverse rejection burden directly.
        # Higher values mean more adverse wick-type rejection is present.
        d7 = side_rej[t]
        release_rejection_burden[t] = d7

        # D3
        # Release acceptance quality combines:
        # - where the close sits inside the bar range for the current side
        # - how strong the favorable body component is
        #
        # For up: high close is better.
        # For down: low close is better, implemented as 1 - clv.
        #
        # The body term is capped at 2.0 before scaling into [0, 1].
        side_close_quality = clv[t] if side == "up" else (1.0 - clv[t])
        release_acceptance_quality[t] = (
            0.6 * side_close_quality +
            0.4 * (min(d2, 2.0) / 2.0)
        )

        # D4: requires prior 12-bar containment edge from t-12 to t-1
        # Measure whether the current close has actually escaped the recent
        # containment boundary in the favorable side-relative direction.
        #
        # For up: compare current close against the highest high of the prior context.
        # For down: compare current close against the lowest low of the prior context.
        #
        # The result is normalized by the current frozen scale.
        if t >= FAM_D_W_REL_CONTEXT and np.all(~np.isnan(scale_pips[t - FAM_D_W_REL_CONTEXT : t + 1])):
            if side == "up":
                recent_containment_edge = np.max(m_high[t - FAM_D_W_REL_CONTEXT : t])
            else:
                recent_containment_edge = np.min(m_low[t - FAM_D_W_REL_CONTEXT : t])

            d4 = dir_sign * (m_close[t] - recent_containment_edge) / (0.0001 * scale_pips[t])
            release_containment_escape[t] = d4

            # D8
            # Composite decisiveness score for the release transition.
            # It blends:
            # - favorable expansion shock
            # - favorable body shock
            # - positive containment escape
            # - inverse rejection burden
            #
            # Each bounded input is clipped through min(..., 2.0) and rescaled.
            release_decisiveness[t] = (
                0.30 * (min(d1, 2.0) / 2.0) +
                0.25 * (min(d2, 2.0) / 2.0) +
                0.30 * (min(max(d4, 0.0), 2.0) / 2.0) +
                0.15 * (1.0 - min(d7, 2.0) / 2.0)
            )

        # D5 revised: 3-bar local takeover balance
        # Look only at the last 3 side-relative one-bar returns.
        # Compute normalized support-minus-damage balance over that local window.
        # This measures whether the immediate neighborhood is being taken over
        # by favorable directional flow.
        if t >= FAM_D_W_REL_SHORT - 1 and np.all(~np.isnan(side_ret[t - 2 : t + 1])):
            x3 = side_ret[t - 2 : t + 1]
            local_support_3 = np.maximum(x3, 0.0).sum()
            local_damage_3 = np.maximum(-x3, 0.0).sum()
            release_local_takeover_balance_3[t] = (
                (local_support_3 - local_damage_3) /
                (local_support_3 + local_damage_3 + FAM_D_EPS)
            )

        # D6
        # Align current release shock with upstream pre-release organization.
        # This uses Family C C8 at W=12 and scales it by current favorable expansion.
        # Strong pre-release organization followed by current expansion produces
        # larger positive alignment.
        if not np.isnan(c8_w12[t]):
            release_transition_alignment[t] = c8_w12[t] * min(d1, 2.0)

    # Store all completed Family D vectors under their final output names.
    famD_data[f"famD_release_expansion_shock_{side}"] = release_expansion_shock
    famD_data[f"famD_release_body_shock_{side}"] = release_body_shock
    famD_data[f"famD_release_acceptance_quality_{side}"] = release_acceptance_quality
    famD_data[f"famD_release_containment_escape_{side}"] = release_containment_escape
    famD_data[f"famD_release_local_takeover_balance_3_{side}"] = release_local_takeover_balance_3
    famD_data[f"famD_release_transition_alignment_{side}"] = release_transition_alignment
    famD_data[f"famD_release_rejection_burden_{side}"] = release_rejection_burden
    famD_data[f"famD_release_decisiveness_{side}"] = release_decisiveness

# Materialize the Family D block once, then append it once to df_feat.
famD_df = pd.DataFrame(famD_data, index=df_feat.index)
df_feat = pd.concat([df_feat, famD_df], axis=1).copy()

famD_cols = list(famD_df.columns)

print("Family D columns added:", len(famD_cols))

# Quick audit slice for spot-checking representative outputs.
# Include shared CLV, the full up-side release stack, and a smaller down-side sample.
audit_cols = [
    "famD_clv",
    "famD_release_expansion_shock_up",
    "famD_release_body_shock_up",
    "famD_release_acceptance_quality_up",
    "famD_release_containment_escape_up",
    "famD_release_local_takeover_balance_3_up",
    "famD_release_transition_alignment_up",
    "famD_release_rejection_burden_up",
    "famD_release_decisiveness_up",
    "famD_release_expansion_shock_dn",
    "famD_release_containment_escape_dn",
    "famD_release_local_takeover_balance_3_dn",
    "famD_release_decisiveness_dn",
]
print(df_feat[audit_cols].head(40))