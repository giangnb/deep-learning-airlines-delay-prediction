import pandas as pd
import numpy as np
import json
import os

# --- CONFIG ---
RAW_DATA_PATH = "data/DelayFlights-raw.csv"
CLEAN_HISTORY_PATH = "data/Flights_report_clean.csv"
ID_MAP_PATH = "data/Flights_report_ids.json"

def process_report_data():
    print(f"⏳ Loading raw data from {RAW_DATA_PATH}...")
    if not os.path.exists(RAW_DATA_PATH):
        print(f"❌ Error: {RAW_DATA_PATH} not found.")
        return

    # Load minimal columns to speed up processing
    cols_to_load = [
        "UniqueCarrier", "Airline Code", 
        "Origin", "Origin Airport Code", 
        "Dest", "Destination Airport Code",
        "Flight Date", "FL_DATE", "Date",
        "CarrierDelay", "Carrier Delay",
        "WeatherDelay", "Weather Delay",
        "NASDelay", "NAS Delay",
        "SecurityDelay", "Security Delay",
        "LateAircraftDelay", "Late Aircraft Delay"
    ]
    
    # Smart load (only columns that exist)
    df_preview = pd.read_csv(RAW_DATA_PATH, nrows=1)
    existing_cols = [c for c in cols_to_load if c in df_preview.columns]
    df = pd.read_csv(RAW_DATA_PATH, usecols=existing_cols)
    
    # 1. STANDARDIZE NAMES
    print("🧹 Standardizing Columns...")
    rename_map = {
        "UniqueCarrier": "Airline", "Airline Code": "Airline",
        "Origin Airport Code": "Origin", "Destination Airport Code": "Dest",
        "Flight Date": "Date", "FL_DATE": "Date",
        "CarrierDelay": "Carrier Delay",
        "WeatherDelay": "Weather Delay",
        "NASDelay": "NAS Delay",
        "SecurityDelay": "Security Delay",
        "LateAircraftDelay": "Late Aircraft Delay"
    }
    df.rename(columns=rename_map, inplace=True)

    # 2. GENERATE ID MAPS (The "Master Index")
    # We must sort alphabetically to match the training logic
    print("🔤 Building Master ID Indexes...")
    
    airlines = sorted(df["Airline"].dropna().astype(str).unique().tolist())
    origins  = sorted(df["Origin"].dropna().astype(str).unique().tolist())
    dests    = sorted(df["Dest"].dropna().astype(str).unique().tolist())
    
    id_maps = {
        "airline": {label: idx for idx, label in enumerate(airlines)},
        "origin":  {label: idx for idx, label in enumerate(origins)},
        "dest":    {label: idx for idx, label in enumerate(dests)}
    }
    
    # Save IDs to JSON
    with open(ID_MAP_PATH, 'w') as f:
        json.dump(id_maps, f)
    print(f"   -> Saved ID Maps to {ID_MAP_PATH}")

    # 3. BUILD HISTORICAL RISK PROFILES
    print("📚 Calculating Risk Profiles (History)...")
    
    # Extract Month
    if "Date" in df.columns:
        df["Date"] = pd.to_datetime(df["Date"], errors='coerce')
        df["Month"] = df["Date"].dt.month.fillna(0).astype(int)
    
    # Clean Targets
    cause_cols = ["Carrier Delay", "Weather Delay", "NAS Delay", "Security Delay", "Late Aircraft Delay"]
    for col in cause_cols:
        if col not in df.columns: df[col] = 0.0
        else: df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0.0)

    # GroupBy [Origin, Month] -> Mean Delays
    history_df = df.groupby(["Origin", "Month"])[cause_cols].mean().reset_index()
    
    # 4. SAVE CLEAN HISTORY
    print(f"💾 Saving History Knowledge Base to {CLEAN_HISTORY_PATH}...")
    history_df.to_csv(CLEAN_HISTORY_PATH, index=False)
    print("✅ Report Data Preparation Complete.")

if __name__ == "__main__":
    process_report_data()