import requests
from config import UPSTOX_BASE
import pandas as pd
import requests
import io
import gzip
from datetime import datetime
from config import UPSTOX_ACCESS_TOKEN
from pandas import Timestamp
from config import INDEX_CONFIG

INSTRUMENTS_URL = (
    "https://assets.upstox.com/market-quote/instruments/exchange/NSE.csv.gz"
)

def get_index_spot(underlying: str):
    cfg = INDEX_CONFIG[underlying]
    url = "https://api.upstox.com/v2/market-quote/ltp"
    headers = {
        "Authorization": f"Bearer {UPSTOX_ACCESS_TOKEN}"
    }
    params = {
        "instrument_key": cfg["spot_key"]
    }
    
    r = requests.get(url, headers=headers, params=params)
    r.raise_for_status()

    data = r.json().get("data", {})
    print("LTP REQUEST:", params)
    print("LTP RAW RESPONSE:", r.text)

    if not data:
        raise ValueError("No LTP data returned for NIFTY")

    # ✅ take the FIRST value dynamically
    spot = list(data.values())[0]["last_price"]
    return spot


def get_instruments_df():
    r = requests.get(INSTRUMENTS_URL, timeout=10)
    r.raise_for_status()
    with gzip.open(io.BytesIO(r.content), mode="rt") as f:
        df = pd.read_csv(f)
    df["expiry"] = pd.to_datetime(df["expiry"], errors="coerce").dt.date
    if "strike" in df.columns:
        df["strike"] = pd.to_numeric(df["strike"], errors="coerce")
    if "option_type" in df.columns:
        df["option_type"] = df["option_type"].str.upper()

    df = df.dropna(subset=["expiry"])

    return df


def get_nifty_expiries(underlying: str):
    df = get_instruments_df()

    nifty_opts = df[
        (df["instrument_type"] == "OPTIDX") &
        (df["name"] == underlying)
    ]

    expiries = sorted(
        nifty_opts["expiry"].dropna().unique().tolist()
    )

    return expiries


UPSTOX_QUOTES_URL = "https://api.upstox.com/v2/market-quote/quotes"

def get_live_quotes(instrument_keys):
    headers = {
        "Authorization": f"Bearer {UPSTOX_ACCESS_TOKEN}",
        "Accept": "application/json"
    }
    params = [("instrument_key", k) for k in instrument_keys]

    r = requests.get(UPSTOX_QUOTES_URL, headers=headers, params=params, timeout=10)
    r.raise_for_status()

    data = r.json()["data"]

    # Normalize → key by instrument_token
    quotes = {
        v["instrument_token"]: v
        for v in data.values()
        if "instrument_token" in v
    }

    return quotes


def build_live_option_chain(instruments_df, quotes, expiry):
    rows = []

    for _, row in instruments_df.iterrows():
        key = row["instrument_key"]
        q = quotes.get(key)

        if not q or "last_price" not in q:
            continue

        opt_type = row["option_type"].upper()
        entry_date = Timestamp.now().normalize()
        rows.append({
            "Entry_Date": entry_date, #datetime.today().replace(hour=0, minute=0, second=0),
            "Expiry": pd.to_datetime(expiry),
            "Strike Price": row["strike"],
            "Option type": "CE" if opt_type in ["CE", "CALL"] else "PE",
            "Entry_Premium": float(q["last_price"]),
        })

    if not rows:
        raise ValueError(
            "Live quotes fetched but no tradable options have LTPs"
        )

    return pd.DataFrame(rows)
