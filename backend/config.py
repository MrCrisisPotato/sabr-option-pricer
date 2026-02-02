import os
UPSTOX_BASE = "https://api.upstox.com/v2"
UPSTOX_ACCESS_TOKEN = os.getenv("UPSTOX_ACCESS_TOKEN")

INDEX_CONFIG = {
    "NIFTY": {
        "label": "NIFTY 50",
        "spot_key": "NSE_INDEX|Nifty 50",
        "instrument_name": "NIFTY",
        "lot_size": 50
    },
    "BANKNIFTY": {
        "label": "BANK NIFTY",
        "spot_key": "NSE_INDEX|Nifty Bank",
        "instrument_name": "BANKNIFTY",
        "lot_size": 15
    },
    "FINNIFTY": {
        "label": "FIN NIFTY",
        "spot_key": "NSE_INDEX|Nifty Fin Service",
        "instrument_name": "FINNIFTY",
        "lot_size": 40
    },
    "MIDCPNIFTY": {
        "label": "MIDCAP NIFTY",
        "spot_key": "NSE_INDEX|NIFTY MID SELECT",
        "instrument_name": "MIDCPNIFTY",
        "lot_size": 75
    }
}

