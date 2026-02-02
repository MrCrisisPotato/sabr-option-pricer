import pandas as pd
from datetime import datetime
from sabr_engine import process_day
import numpy as np



def upstox_to_df(chain):
    rows = []

    # Use a single entry date for the whole chain
    entry_date = pd.Timestamp.now().normalize()

    for opt in chain:
        row = {
            "Entry_Date": entry_date,                       # ✅ REQUIRED
            "Expiry": pd.to_datetime(opt["expiry"]),        # ✅ REQUIRED
            "Strike Price": float(opt["strike_price"]),
            "Option type": "CE" if opt["option_type"] == "CALL" else "PE",
            "Entry_Premium": float(opt["last_price"]),
        }
        rows.append(row)

    df = pd.DataFrame(rows)

    # 🔥 HARD DEBUG (do not remove yet)
    print("DEBUG upstox_to_df columns:", df.columns.tolist())
    print("DEBUG upstox_to_df head:\n", df.head())

    return df

# def upstox_to_df(chain):
#     rows = []
#     today = date.today()

#     for opt in chain:
#         rows.append({
#             "Entry_Date": today,
#             "Expiry": pd.to_datetime(opt["expiry"]),
#             "Strike Price": opt["strike_price"],
#             "Option type": "CE" if opt["option_type"] == "CALL" else "PE",
#             "Entry_Premium": opt["last_price"]
#         })

#     return pd.DataFrame(rows)

def price_live(chain):
    # Case 1: chain is None
    if chain is None:
        return {
            "status": "market_closed",
            "message": "No live option quotes available."
        }

    # Case 2: chain is an empty list
    if isinstance(chain, list) and len(chain) == 0:
        return {
            "status": "market_closed",
            "message": "No live option quotes available."
        }

    # Case 3: chain is already a DataFrame
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


