import os
import json

UPSTOX_BASE = "https://api.upstox.com/v2"
UPSTOX_ACCESS_TOKEN = os.getenv("UPSTOX_ACCESS_TOKEN")

def load_index_config(file_path="indices.json"):
    try:
        with open(file_path, "r") as f:
            return json.load(f)
    except FileNotFoundError:
        print(f"Error: {file_path} not found.")
        return {}
    except json.JSONDecodeError:
        print(f"Error: Failed to decode JSON from {file_path}.")
        return {}


INDEX_CONFIG = load_index_config()
