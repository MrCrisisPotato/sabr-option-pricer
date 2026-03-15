import pandas as pd
from datetime import datetime
from sabr_engine import process_day
import numpy as np



def upstox_to_df(chain):
    rows = []
    entry_date = pd.Timestamp.now().normalize()

    for opt in chain:
        row = {
            "Entry_Date": entry_date,                      
            "Expiry": pd.to_datetime(opt["expiry"]),       
            "Strike Price": float(opt["strike_price"]),
            "Option type": "CE" if opt["option_type"] == "CALL" else "PE",
            "Entry_Premium": float(opt["last_price"]),
        }
        rows.append(row)

    df = pd.DataFrame(rows)

    return df

def price_live(chain):
    if chain is None:
        return {
            "status": "market_closed",
            "message": "No live option quotes available."
        }

    if isinstance(chain, list) and len(chain) == 0:
        return {
            "status": "market_closed",
            "message": "No live option quotes available."
        }

    if isinstance(chain, pd.DataFrame) and chain.empty:
        return {
            "status": "market_closed",
            "message": "No live option quotes available."
        }

    df = upstox_to_df(chain) if not isinstance(chain, pd.DataFrame) else chain

    if df.empty:
        return {
            "status": "market_closed",
            "message": "Option chain exists but contains no usable quotes."
        }

    out = df.groupby("Entry_Date", group_keys=False).apply(process_day)
    out = out.replace([np.inf, -np.inf], np.nan)

    return out.reset_index(drop=True)


