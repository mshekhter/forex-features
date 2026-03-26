import pandas as pd
import numpy as np

PIP = 0.0001
SPREAD_CAP_PIPS = 2.5

def build_5m_bars_from_1m(df_1m: pd.DataFrame) -> pd.DataFrame:
    print("[5M] build_5m_bars_from_1m start (groupby bucket, no empty buckets)")

    df = df_1m.copy()
    print("[5M] input shape:", df.shape)

    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True, errors="raise")
    df = df.sort_values("timestamp").reset_index(drop=True)
    df["i_1m"] = np.arange(len(df), dtype=np.int64)

    print("[5M] time span:", df["timestamp"].iloc[0], "to", df["timestamp"].iloc[-1])

    # Bucket each row into a 5-minute bar (right-labeled to match your prior intent)
    # Example: 22:01..22:05 -> bar timestamp 22:05
    bucket_end = df["timestamp"].dt.floor("5min") + pd.Timedelta(minutes=5)
    df["bar_5m_ts"] = bucket_end

    g = df.groupby("bar_5m_ts", sort=True)

    df5 = g.agg(
        open_bid=("open_bid", "first"),
        high_bid=("high_bid", "max"),
        low_bid=("low_bid", "min"),
        close_bid=("close_bid", "last"),
        volume_bid=("volume_bid", "sum"),
        open_ask=("open_ask", "first"),
        high_ask=("high_ask", "max"),
        low_ask=("low_ask", "min"),
        close_ask=("close_ask", "last"),
        volume_ask=("volume_ask", "sum"),
        i_1m_first=("i_1m", "first"),
        i_1m_last=("i_1m", "last"),
        n_1m=("i_1m", "count"),
    ).reset_index().rename(columns={"bar_5m_ts": "timestamp"})

    print("[5M] 5m bars:", len(df5))
    print("[5M] n_1m stats:")
    print(df5["n_1m"].describe())

    # Spread at 5m close and fixed cap flag
    df5["spread_pips_close"] = (df5["close_ask"] - df5["close_bid"]) / PIP
    df5["spread_ok"] = df5["spread_pips_close"] <= SPREAD_CAP_PIPS

    print("[5M] spread cap pips:", SPREAD_CAP_PIPS)
    print("[5M] spread close stats:")
    print(df5["spread_pips_close"].describe())
    print("[5M] spread_ok %:", round(float(df5["spread_ok"].mean() * 100.0), 2))

    print("[5M] head:")
    print(df5.head(3))

    return df5
    
    FILE = "EURUSD_1MIN_2015_2025_ASK_BID_NO_COVID.csv"

df = pd.read_csv(FILE)
print("[1M] loaded rows:", len(df), "| cols:", df.shape[1])

df5 = build_5m_bars_from_1m(df)

print("[5M] done. rows:", len(df5), "| cols:", df5.shape[1])

print("[5M] monotonic timestamp:", df5["timestamp"].is_monotonic_increasing)
print("[5M] i_1m_first monotonic:", df5["i_1m_first"].is_monotonic_increasing)
print("[5M] i_1m_last monotonic:", df5["i_1m_last"].is_monotonic_increasing)

bad_span = (df5["i_1m_last"] < df5["i_1m_first"]).sum()
print("[5M] bad spans (i_last < i_first):", int(bad_span))

# Gaps check: difference between consecutive i_1m_first
d = df5["i_1m_first"].diff().dropna()
print("[5M] i_1m_first diff stats:")
print(d.describe())

# Ensure typical is 5
print("[5M] pct diff==5:", round(float((d == 5).mean() * 100.0), 2))

import pandas as pd

# Ensure datetime
df5["timestamp"] = pd.to_datetime(df5["timestamp"], utc=True, errors="raise")

start = pd.Timestamp("2022-06-01 00:00:00+00:00")
end   = pd.Timestamp("2025-06-30 23:59:59+00:00")

before = len(df5)

df5 = df5[(df5["timestamp"] >= start) & (df5["timestamp"] <= end)].reset_index(drop=True)

after = len(df5)

print("[5M] rows before:", before)
print("[5M] rows after :", after)
print("[5M] span:", df5["timestamp"].iloc[0], "to", df5["timestamp"].iloc[-1])
print("[5M] rows kept:", after)


# And here is 3 years of clean 5 min EURUSD bars. 230305 of them.