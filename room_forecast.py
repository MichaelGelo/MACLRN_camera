#!/usr/bin/env python3
"""
Room Occupancy Forecast Script
Computes 1-hour predictions based on rolling average.
Runs on SERVER, not Jetson.
"""

from datetime import datetime, timedelta, time, timezone
from supabase import create_client
from dateutil import tz
import pandas as pd
import os

# ---------- CONFIG ----------

SUPABASE_URL = "https://zpezidrqlotoyequnywe.supabase.co"
SUPABASE_SERVICE_KEY = os.environ.get("SUPABASE_SERVICE_KEY")

ROOM_NAME = "GK407"

HISTORY_HOURS = 2
FORECAST_MINUTES = 60
ROLLING_WINDOW_MIN = 10

LOCAL_TZ = tz.gettz("Asia/Manila")

SCHOOL_START = time(7, 0)
SCHOOL_END = time(21, 0)


# ---------- HELPERS ----------

def in_school_hours(t_local):
    t = t_local.time()
    return SCHOOL_START <= t <= SCHOOL_END


def connect_supabase():
    if not SUPABASE_SERVICE_KEY:
        raise Exception("SUPABASE_SERVICE_KEY not found in environment!")
    return create_client(SUPABASE_URL, SUPABASE_SERVICE_KEY)


# ---------- MAIN LOGIC ----------

def load_recent_logs(supabase):
    now = datetime.now(timezone.utc)
    start = now - timedelta(hours=HISTORY_HOURS)

    resp = (
        supabase.table("occupancy_logs")
        .select("timestamp, count")
        .eq("room", ROOM_NAME)
        .gte("timestamp", start.isoformat())
        .order("timestamp", desc=False)
        .execute()
    )

    rows = resp.data or []
    if not rows:
        return pd.DataFrame(columns=["timestamp", "count"])

    df = pd.DataFrame(rows)
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)

    # Round to nearest minute
    df["minute"] = df["timestamp"].dt.floor("T")
    df_min = df.groupby("minute")["count"].mean().reset_index()
    df_min = df_min.set_index("minute")
    return df_min


def generate_forecast(df_min):
    now = datetime.now(timezone.utc).replace(second=0, microsecond=0)

    if df_min.empty:
        future_times = [now + timedelta(minutes=i) for i in range(1, FORECAST_MINUTES + 1)]
        return pd.DataFrame({
            "predicted_for": future_times,
            "predicted_count": [0] * FORECAST_MINUTES,
            "horizon_minutes": list(range(1, FORECAST_MINUTES + 1)),
        })

    # build continuous minute index
    full_index = pd.date_range(
        start=df_min.index.min(),
        end=now,
        freq="T",
        tz="UTC",
    )
    df_full = df_min.reindex(full_index)
    df_full["count"].ffill(inplace=True)
    df_full["count"].fillna(0, inplace=True)

    history = df_full["count"].copy()

    # SAFETY: should never be empty now, but double-check
    if history.empty:
        future_times = [now + timedelta(minutes=i) for i in range(1, FORECAST_MINUTES + 1)]
        return pd.DataFrame({
            "predicted_for": future_times,
            "predicted_count": [0] * FORECAST_MINUTES,
            "horizon_minutes": list(range(1, FORECAST_MINUTES + 1)),
        })

    # generate predictions
    future_times, preds = [], []

    for i in range(1, FORECAST_MINUTES + 1):
        target = now + timedelta(minutes=i)
        target_local = target.astimezone(LOCAL_TZ)

        if True:
            w_end = history.index[-1]
            w_start = w_end - timedelta(minutes=ROLLING_WINDOW_MIN - 1)
            window = history[w_start:w_end]
            pred = float(window.mean()) if len(window) else float(history.iloc[-1])
        else:
            pred = 0.0

        history.loc[target] = pred
        future_times.append(target)
        preds.append(max(0, pred))

    return pd.DataFrame({
        "predicted_for": future_times,
        "predicted_count": preds,
        "horizon_minutes": list(range(1, FORECAST_MINUTES + 1)),
    })


def save_results(supabase, forecast_df):
    payload = []
    now = datetime.now(timezone.utc).isoformat()

    for _, row in forecast_df.iterrows():
        payload.append({
            "room": ROOM_NAME,
            "predicted_for": row["predicted_for"].isoformat(),
            "predicted_count": int(round(row["predicted_count"])),
            "horizon_minutes": int(row["horizon_minutes"]),
            "model_name": "simple_forecast_v1",
            "created_at": now,
        })

    supabase.table("occupancy_predictions").upsert(payload, on_conflict="room,predicted_for").execute()
    print(f"[✓] Saved {len(payload)} predictions.")


# ---------- ENTRY ----------

def main():
    print("[*] Running forecast generator...")

    supabase = connect_supabase()
    df_min = load_recent_logs(supabase)
    forecast = generate_forecast(df_min)
    save_results(supabase, forecast)

    print("[✓] Forecast complete.")


if __name__ == "__main__":
    main()
