# Build Family A v1 features into df_feat as a separate research dataframe.
# Keep df5 unchanged so later feature families can append to the same output frame.
# Family A captures live-safe, side-relative support and damage geometry
# using fixed windows, a fixed local scale, exact wick-rejection logic,
# and NaN-preserving validity rules.

import numpy as np
import pandas as pd

PIP = 0.0001
FAM_A_W_SET = (3, 6, 12, 24)
FAM_A_W_SCALE = 24
FAM_A_SCALE_FLOOR_PIPS = 0.25
FAM_A_EPS = 1e-8

df_feat = df5.copy()

# Use existing mid-price columns when available; otherwise derive them from bid/ask.
if {"m_open", "m_high", "m_low", "m_close"}.issubset(df_feat.columns):
    m_open = df_feat["m_open"].astype(float)
    m_high = df_feat["m_high"].astype(float)
    m_low = df_feat["m_low"].astype(float)
    m_close = df_feat["m_close"].astype(float)
else:
    m_open = ((df_feat["open_bid"] + df_feat["open_ask"]) / 2.0).astype(float)
    m_high = ((df_feat["high_bid"] + df_feat["high_ask"]) / 2.0).astype(float)
    m_low = ((df_feat["low_bid"] + df_feat["low_ask"]) / 2.0).astype(float)
    m_close = ((df_feat["close_bid"] + df_feat["close_ask"]) / 2.0).astype(float)

df_feat["m_open"] = m_open
df_feat["m_high"] = m_high
df_feat["m_low"] = m_low
df_feat["m_close"] = m_close

# Compute the exact local volatility scale from mid-price true range in pips.
prev_close = m_close.shift(1)

tr_mid_pips = pd.Series(np.nan, index=df_feat.index, dtype=float)
tr_mid_pips.iloc[1:] = np.maximum.reduce([
    ((m_high.iloc[1:] - m_low.iloc[1:]) / PIP).to_numpy(),
    ((m_high.iloc[1:] - prev_close.iloc[1:]).abs() / PIP).to_numpy(),
    ((m_low.iloc[1:] - prev_close.iloc[1:]).abs() / PIP).to_numpy(),
])

scale_raw = tr_mid_pips.rolling(FAM_A_W_SCALE, min_periods=FAM_A_W_SCALE).median()
scale = np.maximum(scale_raw, FAM_A_SCALE_FLOOR_PIPS)

df_feat["famA_tr_mid_pips"] = tr_mid_pips
df_feat["famA_scale_raw_pips"] = scale_raw
df_feat["famA_scale_pips"] = scale

# Build one-bar normalized side-relative return, body, range, and wick primitives.
side_ret_1_up = (m_close - m_close.shift(1)) / (PIP * scale)
side_ret_1_dn = -(m_close - m_close.shift(1)) / (PIP * scale)

side_body_up = (m_close - m_open) / (PIP * scale)
side_body_dn = -(m_close - m_open) / (PIP * scale)

rng_norm = (m_high - m_low) / (PIP * scale)
upper_wick_norm = (m_high - np.maximum(m_open, m_close)) / (PIP * scale)
lower_wick_norm = (np.minimum(m_open, m_close) - m_low) / (PIP * scale)

df_feat["famA_side_ret_1_up"] = side_ret_1_up
df_feat["famA_side_ret_1_dn"] = side_ret_1_dn
df_feat["famA_side_body_up"] = side_body_up
df_feat["famA_side_body_dn"] = side_body_dn
df_feat["famA_rng_norm"] = rng_norm
df_feat["famA_upper_wick_norm"] = upper_wick_norm
df_feat["famA_lower_wick_norm"] = lower_wick_norm

# Compute the exact adverse-rejection wick component for each side.
adverse_rejection_component_up = (
    ((m_close >= m_open).astype(float) * upper_wick_norm) +
    ((m_close < m_open).astype(float) * lower_wick_norm)
)

adverse_rejection_component_dn = (
    ((m_close <= m_open).astype(float) * lower_wick_norm) +
    ((m_close > m_open).astype(float) * upper_wick_norm)
)

df_feat["famA_adverse_rejection_component_up"] = adverse_rejection_component_up
df_feat["famA_adverse_rejection_component_dn"] = adverse_rejection_component_dn

for W in FAM_A_W_SET:
    support_flow_up = side_ret_1_up.clip(lower=0.0).rolling(W, min_periods=W).sum()
    damage_flow_up = (-side_ret_1_up).clip(lower=0.0).rolling(W, min_periods=W).sum()
    support_body_flow_up = side_body_up.clip(lower=0.0).rolling(W, min_periods=W).sum()
    damage_body_flow_up = (-side_body_up).clip(lower=0.0).rolling(W, min_periods=W).sum()
    adverse_rejection_wick_up = adverse_rejection_component_up.rolling(W, min_periods=W).sum()

    net_support_balance_up = support_flow_up - damage_flow_up
    net_side_disp_up = side_ret_1_up.rolling(W, min_periods=W).sum()
    support_ratio_up = support_flow_up / (support_flow_up + damage_flow_up + FAM_A_EPS)
    support_efficiency_up = net_side_disp_up / (support_flow_up + damage_flow_up + FAM_A_EPS)
    body_support_ratio_up = support_body_flow_up / (
        support_body_flow_up + damage_body_flow_up + FAM_A_EPS
    )
    wick_rejection_ratio_up = adverse_rejection_wick_up / (
        support_flow_up + damage_flow_up + FAM_A_EPS
    )

    df_feat[f"famA_support_flow_up_w{W}"] = support_flow_up
    df_feat[f"famA_damage_flow_up_w{W}"] = damage_flow_up
    df_feat[f"famA_support_ratio_up_w{W}"] = support_ratio_up
    df_feat[f"famA_support_body_flow_up_w{W}"] = support_body_flow_up
    df_feat[f"famA_damage_body_flow_up_w{W}"] = damage_body_flow_up
    df_feat[f"famA_adverse_rejection_wick_up_w{W}"] = adverse_rejection_wick_up
    df_feat[f"famA_net_support_balance_up_w{W}"] = net_support_balance_up
    df_feat[f"famA_net_side_disp_up_w{W}"] = net_side_disp_up
    df_feat[f"famA_support_efficiency_up_w{W}"] = support_efficiency_up
    df_feat[f"famA_body_support_ratio_up_w{W}"] = body_support_ratio_up
    df_feat[f"famA_wick_rejection_ratio_up_w{W}"] = wick_rejection_ratio_up

    support_flow_dn = side_ret_1_dn.clip(lower=0.0).rolling(W, min_periods=W).sum()
    damage_flow_dn = (-side_ret_1_dn).clip(lower=0.0).rolling(W, min_periods=W).sum()
    support_body_flow_dn = side_body_dn.clip(lower=0.0).rolling(W, min_periods=W).sum()
    damage_body_flow_dn = (-side_body_dn).clip(lower=0.0).rolling(W, min_periods=W).sum()
    adverse_rejection_wick_dn = adverse_rejection_component_dn.rolling(W, min_periods=W).sum()

    net_support_balance_dn = support_flow_dn - damage_flow_dn
    net_side_disp_dn = side_ret_1_dn.rolling(W, min_periods=W).sum()
    support_ratio_dn = support_flow_dn / (support_flow_dn + damage_flow_dn + FAM_A_EPS)
    support_efficiency_dn = net_side_disp_dn / (support_flow_dn + damage_flow_dn + FAM_A_EPS)
    body_support_ratio_dn = support_body_flow_dn / (
        support_body_flow_dn + damage_body_flow_dn + FAM_A_EPS
    )
    wick_rejection_ratio_dn = adverse_rejection_wick_dn / (
        support_flow_dn + damage_flow_dn + FAM_A_EPS
    )

    df_feat[f"famA_support_flow_dn_w{W}"] = support_flow_dn
    df_feat[f"famA_damage_flow_dn_w{W}"] = damage_flow_dn
    df_feat[f"famA_support_ratio_dn_w{W}"] = support_ratio_dn
    df_feat[f"famA_support_body_flow_dn_w{W}"] = support_body_flow_dn
    df_feat[f"famA_damage_body_flow_dn_w{W}"] = damage_body_flow_dn
    df_feat[f"famA_adverse_rejection_wick_dn_w{W}"] = adverse_rejection_wick_dn
    df_feat[f"famA_net_support_balance_dn_w{W}"] = net_support_balance_dn
    df_feat[f"famA_net_side_disp_dn_w{W}"] = net_side_disp_dn
    df_feat[f"famA_support_efficiency_dn_w{W}"] = support_efficiency_dn
    df_feat[f"famA_body_support_ratio_dn_w{W}"] = body_support_ratio_dn
    df_feat[f"famA_wick_rejection_ratio_dn_w{W}"] = wick_rejection_ratio_dn

# Keep primitive validity intact and blank only scale-dependent outputs where scale is invalid.
scale_dependent_cols = [c for c in df_feat.columns if c.startswith("famA_") and c != "famA_tr_mid_pips"]
valid_scale_mask = df_feat["famA_scale_pips"].notna()

for col in scale_dependent_cols:
    df_feat.loc[~valid_scale_mask, col] = np.nan

print("df_feat created")
print("Family A columns added:", len([c for c in df_feat.columns if c.startswith('famA_')]))