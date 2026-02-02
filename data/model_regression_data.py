import pandas as pd
import numpy as np
import os

# --- CONFIG ---
RAW_DATA_PATH = "data/DelayFlights-raw.csv"
CLEAN_DATA_PATH = "data/Flights_regression_clean.csv"

def get_sin_cos(value, max_val):
    """Helper to convert Time to Cyclic Features"""
    theta = 2 * np.pi * value / max_val
    return np.sin(theta), np.cos(theta)

def process_data():
    print(f"⏳ Loading raw data from {RAW_DATA_PATH}...")
    if not os.path.exists(RAW_DATA_PATH):
        print(f"❌ Error: {RAW_DATA_PATH} not found.")
        return

    df = pd.read_csv(RAW_DATA_PATH)
    
    # 1. STANDARDIZE COLUMN NAMES
    # We rename columns to a standard format to avoid confusion later
    rename_map = {
        "UniqueCarrier": "Airline", "Airline Code": "Airline",
        "Origin Airport Code": "Origin", "Destination Airport Code": "Dest",
        "ArrDelay": "Target", "Arrival Delay": "Target",
        "Distance": "Distance", "Distance Miles": "Distance",
        "CRSElapsedTime": "Duration", "Fly Time Scheduled": "Duration",
        "CRSDepTime": "DepTime", "Departure Time Scheduled": "DepTime",
        "Flight Date": "Date", "FL_DATE": "Date"
    }
    df.rename(columns=rename_map, inplace=True)
    
    # 2. CLEAN TARGET (Robust Logic)
    print("🧹 Cleaning Targets & Clipping Outliers...")
    df['Target'] = pd.to_numeric(df['Target'], errors='coerce').fillna(0.0)
    # Clip at 180 mins (3 hours) to prevent model paranoia (Huber Loss logic)
    df['Target'] = df['Target'].clip(upper=180)

    # 3. FEATURE ENGINEERING (Cyclic Time)
    print("⏳ Engineering Time Features...")
    
    # Extract Month/Day/Hour
    date_col = "Date" if "Date" in df.columns else None
    if date_col:
        df[date_col] = pd.to_datetime(df[date_col], errors='coerce')
        df['Month'] = df[date_col].dt.month.fillna(0)
        df['Day'] = df[date_col].dt.dayofweek.fillna(0)
    
    # Extract Hour from HHMM format (e.g., 1430 -> 14)
    if "DepTime" in df.columns:
        df['Hour'] = (pd.to_numeric(df['DepTime'], errors='coerce').fillna(0) // 100)
    
    # 4. SCALING & CALCULATIONS
    print("📐 Scaling & Calculating Inputs...")
    
    # Fill NaNs
    df['Duration'] = pd.to_numeric(df['Duration'], errors='coerce').fillna(120.0)
    df['Distance'] = pd.to_numeric(df['Distance'], errors='coerce').fillna(800.0)

    # Scale to 0-1 range (Robust Scaling)
    df['f_dur'] = df['Duration'] / 400.0
    df['f_dist'] = df['Distance'] / 3000.0

    # Calculate Cyclic Sin/Cos
    df['f_dep_sin'], df['f_dep_cos'] = get_sin_cos(df['Hour'], 24)
    df['f_day_sin'], df['f_day_cos'] = get_sin_cos(df['Day'], 7)
    df['f_mon_sin'], df['f_mon_cos'] = get_sin_cos(df['Month'], 12)

    # 5. ENCODE CATEGORICALS (Integer Encoding)
    print("🔤 Encoding Categoricals...")
    cat_cols = ["Airline", "Origin", "Dest"]
    for col in cat_cols:
        if col in df.columns:
            df[col] = df[col].astype(str).astype('category').cat.codes.astype(np.int32)
        else:
            print(f"⚠️ Warning: Missing column {col}")
            df[col] = 0

    # 6. SELECT FINAL COLUMNS
    # We only save what we need for training
    final_cols = [
        "Airline", "Origin", "Dest", "Target",
        "f_dur", "f_dist", 
        "f_dep_sin", "f_dep_cos", 
        "f_day_sin", "f_day_cos", 
        "f_mon_sin", "f_mon_cos"
    ]
    
    # Sort chronologically if possible for sequence logic
    if date_col: df = df.sort_values(by=date_col)
    
    df_clean = df[final_cols]
    
    # 7. SAVE
    print(f"💾 Saving cleaned data to {CLEAN_DATA_PATH}...")
    df_clean.to_csv(CLEAN_DATA_PATH, index=False)
    print("✅ Data Preparation Complete.")

if __name__ == "__main__":
    process_data()