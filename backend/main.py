from fastapi import FastAPI, HTTPException
from upstox import build_live_option_chain, get_index_spot, get_instruments_df, get_live_quotes, get_nifty_expiries
from pricing import price_live
from fastapi.middleware.cors import CORSMiddleware
from typing import Optional
from datetime import datetime
from config import INDEX_CONFIG
import numpy as np

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173"],  # frontend origin
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

INSTRUMENTS = {
    "NIFTY": "NSE_INDEX|Nifty 50",
    "BANKNIFTY": "NSE_INDEX|Nifty Bank"
}

# ---- test endpoint ----
@app.get("/ping")
def ping():
    return {"status": "ok"}

@app.get("/expiries")
def expiries(underlying: str):
    try:
        if underlying not in INDEX_CONFIG:
            raise HTTPException(400, f"Unsupported index: {underlying}")

        return get_nifty_expiries(underlying)
    except HTTPException:
        raise
    except Exception as e:
        print("EXPIRY ERROR:", e)
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/price")
def price_options(
    underlying: str,
    expiry: str,
):
    try:
        if underlying not in INDEX_CONFIG:
            raise HTTPException(
                status_code=400,
                detail=f"Unsupported index: {underlying}"
            )

        spot = get_index_spot(underlying)

        expiry_date = datetime.strptime(expiry, "%Y-%m-%d").date()

        instruments = get_instruments_df()
        # print(instruments[instruments["name"] == "NIFTY"]["expiry"].unique()[:5])
        df = instruments[
            (instruments["exchange"].isin(["NSE_FO", "NFO"])) &
            (instruments["instrument_type"] == "OPTIDX") &
            (instruments["name"] == underlying) &
            (instruments["expiry"] == expiry_date)
        ]
        # print(
        #     instruments[
        #         instruments["instrument_type"] == "OPTIDX"
        #     ][["exchange", "name"]].drop_duplicates().head(10)
        # )

        if df.empty:
            raise HTTPException(400, "No instruments found for expiry")
        print("AVAILABLE EXPIRIES:", sorted(instruments[instruments["name"] == underlying]["expiry"].unique())[:10])
        print("REQUESTED EXPIRY:", expiry_date)

        df = df.copy()

        keys = df["instrument_key"].tolist()

        quotes = {}
        for i in range(0, len(keys), 40):
            batch = keys[i:i+40]
            q = get_live_quotes(batch)
            if q:
                quotes.update(q)

        if not quotes:
            return {
                "status": "market_closed",
                "message": (
                    "No live option quotes available. "
                    "Market may be closed or FO market-data permission is missing."
                ),
                "expiry": expiry,
                "underlying": underlying
            }


        # Step 2: keep only instruments that ACTUALLY have quotes
        df = df[df["instrument_key"].isin(quotes.keys())].copy()

        # Step 3: now safely do ATM filtering
        atm_strike = df["strike"].median()
        df.loc[:, "dist"] = (df["strike"] - atm_strike).abs()
        df = df.sort_values("dist").head(40)

        chain_df = build_live_option_chain(df, quotes, expiry)
        chain_df["Spot"] = spot

        results = price_live(chain_df)

        if isinstance(results, dict):
            return results

        results = results.replace([np.inf, -np.inf], None)

        return results[[
            "Strike Price",
            "Option type",
            "Entry_Premium",
            "Market_Vol",
            "SABR_IV",
            "SABR_B76_Price",
            "BS_Price",
            "Mispricing_Pct",
            "SABR_vs_BS",
            "Vega",
            "Delta",
            "Valuation"
        ]].to_dict(orient="records")
    
    except HTTPException:
        # re-raise clean FastAPI errors
        raise

    except Exception as e:
        print("BACKEND ERROR:", e)
        return {
            "status": "error",
            "message": str(e)
        }
