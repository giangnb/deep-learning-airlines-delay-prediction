import pandas as pd
import json
import os

# --- CONFIG ---
# We use the Regression Data because it has ALL flights (not just late ones)
# and it has the Distance/Duration columns.
INPUT_CSV = "data/Flights_regression_clean.csv" 
OUTPUT_JSON = "data/Flights_routes.json"

def build_route_map():
    if not os.path.exists(INPUT_CSV):
        print(f"❌ Error: Could not find {INPUT_CSV}")
        return

    print("🗺️  Scanning training data to learn Route Physics...")
    df = pd.read_csv(INPUT_CSV)
    
    # The clean CSV likely has "f_dist" (scaled) and "f_dur" (scaled)
    # We need to un-scale them to get human-readable numbers for the UI.
    # Scaling logic from training: Dist/3000, Dur/400
    
    # Group by IDs (Origin, Dest) -> Get Median Values
    print("   Grouping unique routes...")
    # Note: In the clean CSV, Origin/Dest are likely already IDs (Integers)
    route_stats = df.groupby(["Origin", "Dest"])[["f_dist", "f_dur"]].median().reset_index()
    
    route_map = {}
    count = 0
    
    for _, row in route_stats.iterrows():
        # Key: "OriginID_DestID" (e.g., "55_12")
        key = f"{int(row['Origin'])}_{int(row['Dest'])}"
        
        # Un-scale to get "Real" numbers for display
        real_dist = row["f_dist"] * 3000.0
        real_dur  = row["f_dur"] * 400.0
        
        route_map[key] = {
            "dist": round(real_dist, 1),
            "dur":  round(real_dur, 0)
        }
        count += 1
        
    # Save
    with open(OUTPUT_JSON, 'w') as f:
        json.dump(route_map, f)
        
    print(f"✅ Success! Mapped {count} routes.")
    print(f"💾 Saved to {OUTPUT_JSON}")

if __name__ == "__main__":
    build_route_map()