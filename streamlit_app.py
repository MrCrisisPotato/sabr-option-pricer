import streamlit as st
import requests
import pandas as pd

# ---------------- CONFIG ----------------
BACKEND_URL = "http://localhost:8000"   # change when deployed
# ---------------------------------------

st.set_page_config(
    page_title="SABR Option Pricer",
    layout="wide"
)

st.title("SABR Option Pricer")

# ---------------- Sidebar ----------------
st.sidebar.header("Inputs")

INDEX_MAP = {
    "NIFTY 50": "NIFTY",
    "BANKNIFTY": "BANKNIFTY",
    "FINNIFTY": "FINNIFTY",
    "MIDCAPNIFTY": "MIDCPNIFTY"
}

index_label = st.sidebar.selectbox(
    "Select Index",
    list(INDEX_MAP.keys())
)

underlying = INDEX_MAP[index_label]

# ---------------- Load expiries ----------------
@st.cache_data
def fetch_expiries(underlying):
    r = requests.get(
        f"{BACKEND_URL}/expiries",
        params={"underlying": underlying},
        timeout=10
    )
    r.raise_for_status()
    return r.json()

try:
    expiries = fetch_expiries(underlying)
    expiry = st.sidebar.selectbox("Expiry Date", expiries)
except Exception as e:
    st.sidebar.error(f"Failed to load expiries: {e}")
    st.stop()

# ---------------- Fetch prices ----------------
if st.sidebar.button("Load Options"):
    with st.spinner("Pricing options using SABR..."):
        r = requests.get(
            f"{BACKEND_URL}/price",
            params={
                "underlying": underlying,
                "expiry": expiry
            },
            timeout=30
        )

    if r.status_code != 200:
        st.error(r.text)
        st.stop()

    data = r.json()

    if isinstance(data, dict) and data.get("status") == "market_closed":
        st.warning(data["message"])
        st.stop()

    df = pd.DataFrame(data)

    # ---------------- Display ----------------
    st.subheader(f"Option Chain — {underlying} {expiry}")

    st.dataframe(
        df.style
        .format({
            "Entry_Premium": "{:.2f}",
            "SABR_B76_Price": "{:.2f}",
            "BS_Price": "{:.2f}",
            "Market_Vol": "{:.3f}",
            "SABR_IV": "{:.3f}",
        })
        .applymap(
            lambda x: "color: green" if "Cheap" in str(x)
            else "color: red" if "Expensive" in str(x)
            else "",
            subset=["Valuation"]
        ),
        use_container_width=True
    )

    def highlight_signal(row):
        if row["Buy_Signal"] == "STRONG BUY":
            return ["background-color: #004d1a"] * len(row)
        elif row["Buy_Signal"] == "BUY":
            return ["background-color: #0b3d91"] * len(row)
        elif row["Buy_Signal"] == "AVOID / SELL":
            return ["background-color: #4d0000"] * len(row)
        return ["" for _ in row]
    
    st.dataframe(
        df.style.apply(highlight_signal, axis=1),
        use_container_width=True
    )

    st.subheader("Buy Recommendations")

    buy_df = df[df["Buy_Signal"].isin(["BUY", "STRONG BUY"])]

    if buy_df.empty:
        st.info("No attractive buy opportunities for this expiry.")
    else:
        st.dataframe(
            buy_df.sort_values("Mispricing_Pct", ascending=False)[
                ["Strike Price", "Option type", "Entry_Premium",
                "SABR_B76_Price", "Mispricing_Pct", "Vega", "Delta", "Buy_Signal"]
            ],
            use_container_width=True
        )



    # ---------------- Charts ----------------
    # st.subheader("SABR vs Black-Scholes Prices")
    
    # chart_df = df.sort_values("Strike Price")

    # st.line_chart(
    #     chart_df.set_index("Strike Price")[
    #         ["Entry_Premium", "SABR_B76_Price", "BS_Price"]
    #     ]
    # )

    # chart_df["SABR_Error"] = chart_df["SABR_B76_Price"] - chart_df["Entry_Premium"]
    # chart_df["BS_Error"] = chart_df["BS_Price"] - chart_df["Entry_Premium"]

    # st.line_chart(
    #     chart_df.set_index("Strike Price")[["SABR_Error", "BS_Error"]]
    # )

