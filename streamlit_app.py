# streamlit_app.py
import sys
import os

sys.path.append(os.path.dirname(__file__))

import streamlit as st
import pandas as pd

from backend.upstox import (
    get_instruments_df,
    get_live_quotes,
    build_live_option_chain,
    get_index_spot
)
from pricing import price_live
from backend.config import INDEX_CONFIG

st.set_page_config(page_title="SABR Option Pricer", layout="wide")
st.title("📈 SABR Option Pricer")

underlying = st.selectbox(
    "Select Index",
    list(INDEX_CONFIG.keys())
)

instruments = get_instruments_df()
expiries = sorted(
    instruments[instruments["name"] == underlying]["expiry"].unique()
)

expiry = st.selectbox("Select Expiry", expiries)

if st.button("Load Options"):
    with st.spinner("Fetching market data..."):
        spot = get_index_spot(underlying)
        expiry_date = pd.to_datetime(expiry).date()

        df = instruments[
            (instruments["instrument_type"] == "OPTIDX") &
            (instruments["name"] == underlying) &
            (instruments["expiry"] == expiry_date)
        ].copy()

        if df.empty:
            st.error("No instruments found")
            st.stop()

        keys = df["instrument_key"].tolist()

        quotes = {}
        for i in range(0, len(keys), 40):
            q = get_live_quotes(keys[i:i+40])
            if q:
                quotes.update(q)

        if not quotes:
            st.warning("Market closed / no quotes")
            st.stop()

        df = df[df["instrument_key"].isin(quotes.keys())]

        atm = df["strike"].median()
        df["dist"] = (df["strike"] - atm).abs()
        df = df.sort_values("dist").head(40)

        chain = build_live_option_chain(df, quotes, expiry)
        chain["Spot"] = spot

        result = price_live(chain)

c1, c2, c3, c4 = st.columns(4)

c1.metric("Spot", f"{spot:,.2f}")
c2.metric("Options", len(result))
c3.metric("Avg SABR–BS", f"{(result['SABR_B76_Price'] - result['BS_Price']).mean():.2f}")
c4.metric("Avg Vega", f"{result['Vega'].mean():.1f}")

st.dataframe(
    result[[
        "Strike Price",
        "Option type",
        "Entry_Premium",
        "Market_Vol",
        "SABR_IV",
        "SABR_B76_Price",
        "BS_Price",
        "Vega",
        "Delta",
        "Valuation"
    ]],
    use_container_width=True
)
st.subheader("Volatility Smile")

st.line_chart(
    result.pivot_table(
        index="Strike Price",
        values=["Market_Vol", "SABR_IV"]
    )
)
