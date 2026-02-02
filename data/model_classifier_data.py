import pandas as pd
import numpy as np
import os

# --- CONFIG ---
RAW_DATA_PATH = "data/DelayFlights-raw.csv"
CLEAN_DATA_PATH = "data/Flights_classifier_clean.csv"

def process_data():
    print(f"⏳ Loading raw data from {RAW_DATA_PATH}...")
    if not os.path.exists(RAW_DATA_PATH):
        print(f"❌ Error: {RAW_DATA_PATH} not found.")
        return

    df = pd.read_csv(RAW_DATA_PATH)
    
    # 1. STANDARDIZE COLUMN NAMES
    print("🧹 Cleaning Columns...")
    rename_map = {
        "UniqueCarrier": "Airline", "Airline Code": "Airline",
        "Origin Airport Code": "Origin", "Destination Airport Code": "Dest",
        "ArrDelay": "TargetDelay", "Arrival Delay": "TargetDelay",
        "Distance": "Distance", "Distance Miles": "Distance",
        "CRSElapsedTime": "Duration", "Fly Time Scheduled": "Duration",
        "Flight Date": "Date", "FL_DATE": "Date"
    }
    df.rename(columns=rename_map, inplace=True)
    
    # Standardize Cause Columns
    cause_map = {
        "CarrierDelay": "Carrier Delay",
        "WeatherDelay": "Weather Delay",
        "NASDelay": "NAS Delay",
        "SecurityDelay": "Security Delay",
        "LateAircraftDelay": "Late Aircraft Delay"
    }
    df.rename(columns=cause_map, inplace=True)
    
    cause_cols = ["Carrier Delay", "Weather Delay", "NAS Delay", "Security Delay", "Late Aircraft Delay"]
    
    # Ensure Numeric
    for col in cause_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0.0)
        else:
            df[col] = 0.0

    # 2. FEATURE ENGINEERING (Date/Month)
    print("📅 Extracting Date Features...")
    if "Date" in df.columns:
        df["Date"] = pd.to_numeric(pd.to_datetime(df["Date"], errors='coerce')) # Convert to numeric timestamp for sorting
        df["Month"] = pd.to_datetime(df["Date"]).dt.month.fillna(0).astype(int)
    
    # 3. BUILD HISTORICAL RISK PROFILES (The "Cheat Sheet")
    print("📚 Calculating Historical Averages (Origin + Month)...")
    
    # Group by Origin and Month to get average delay causes
    history_df = df.groupby(["Origin", "Month"])[cause_cols].mean().reset_index()
    
    # Rename columns to avoid collision (e.g., "Hist_Weather Delay")
    new_names = {col: f"Hist_{col}" for col in cause_cols}
    history_df.rename(columns=new_names, inplace=True)
    
    # Merge history back into main dataframe
    print("   -> Merging history into dataset...")
    df = pd.merge(df, history_df, on=["Origin", "Month"], how="left")
    
    # Fill missing history with 0
    hist_cols = list(new_names.values())
    df[hist_cols] = df[hist_cols].fillna(0.0)

    # 4. FILTER: LATE FLIGHTS ONLY
    # We only train the classifier on flights that were actually late (>= 15 mins)
    initial_len = len(df)
    df["TargetDelay"] = pd.to_numeric(df["TargetDelay"], errors='coerce').fillna(0)
    df = df[df["TargetDelay"] >= 15].copy()
    print(f"📉 Filtered Late Flights: {initial_len} -> {len(df)}")

    # 5. CREATE TARGET LABELS (0-4)
    # Determine which cause was the largest for each flight
    print("🏷️ Creating Classification Labels...")
    y_raw = df[cause_cols].values
    df['Label'] = np.argmax(y_raw, axis=1) # 0=Carrier, 1=Weather...

    # 6. SCALING & ENCODING
    print("📐 Scaling Inputs...")
    df['Duration'] = pd.to_numeric(df['Duration'], errors='coerce').fillna(120.0)
    df['Distance'] = pd.to_numeric(df['Distance'], errors='coerce').fillna(800.0)
    
    # Scale to 0-1 range (Same as Regression)
    df['f_dur'] = df['Duration'] / 400.0
    df['f_dist'] = df['Distance'] / 3000.0

    print("🔤 Encoding Categoricals...")
    for col in ["Airline", "Origin", "Dest"]:
        df[col] = df[col].astype(str).astype('category').cat.codes.astype(np.int32)

    # 7. SAVE FINAL DATASET
    # Columns: IDs + Scaled Base Features + History Features + Target Label
    final_cols = ["Airline", "Origin", "Dest", "Label", "f_dur", "f_dist"] + hist_cols
    
    # Sort chronologically (using the numeric date we created)
    df = df.sort_values(by="Date")
    
    df_clean = df[final_cols]
    
    print(f"💾 Saving cleaned data to {CLEAN_DATA_PATH}...")
    df_clean.to_csv(CLEAN_DATA_PATH, index=False)
    print("✅ Classifier Data Preparation Complete.")

if __name__ == "__main__":
    process_data()