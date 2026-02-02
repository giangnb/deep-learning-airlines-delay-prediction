import pandas as pd
import numpy as np
import tensorflow as tf
import json
import os
from tensorflow.keras.models import load_model

# --- CONFIG ---
MODEL_PATH = "data/model_classifier.keras"
ID_MAP_PATH = "data/Flights_report_ids.json"
HISTORY_DATA_PATH = "data/Flights_report_clean.csv"
ROUTE_MAP_PATH = "data/Flights_routes.json"  # <--- NEW FILE

CAUSE_LABELS = [
    "Carrier (Airline Fault)", "Weather (Storm/Snow)", "NAS (Traffic/ATC)", 
    "Security (Breach/Check)", "Late Aircraft (Ripple Effect)"
]

class DelayCausePredictor:
    def __init__(self):
        print("🔍 Initializing Tool...")
        self.model = None
        self.maps = {}
        self.history_lookup = {}
        self.route_lookup = {}
        self.load_resources()

    def load_resources(self):
        # 1. Load Model
        try:
            self.model = load_model(MODEL_PATH)
            print("   ✅ Classifier Model Loaded.")
        except: return

        # 2. Load ID Maps
        try:
            with open(ID_MAP_PATH, 'r') as f: self.maps = json.load(f)
            print("   ✅ ID Maps Loaded.")
        except: print("❌ ID Maps missing.")

        # 3. Load Route Physics (NEW)
        try:
            with open(ROUTE_MAP_PATH, 'r') as f: self.route_lookup = json.load(f)
            print("   ✅ Route Physics Loaded.")
        except: print("❌ Route Map missing. Run utils_route_mapper.py!")

        # 4. Load History
        try:
            df_hist = pd.read_csv(HISTORY_DATA_PATH)
            for _, row in df_hist.iterrows():
                key = f"{row['Origin']}_{int(row['Month'])}"
                self.history_lookup[key] = [
                    row["Carrier Delay"], row["Weather Delay"], 
                    row["NAS Delay"], row["Security Delay"], 
                    row["Late Aircraft Delay"]
                ]
            print(f"   ✅ History Base Loaded.")
        except: print("❌ History file missing.")

    def predict_cause(self, airline, origin, dest, month):
        # --- 1. GET IDS ---
        air_id = self.maps["airline"].get(airline, 0)
        org_id = self.maps["origin"].get(origin, 0)
        dst_id = self.maps["dest"].get(dest, 0)

        # --- 2. AUTO-DETECT DISTANCE & DURATION ---
        route_key = f"{org_id}_{dst_id}"
        
        if route_key in self.route_lookup:
            # Found it! Use the auto-detected values
            dist = self.route_lookup[route_key]["dist"]
            dur  = self.route_lookup[route_key]["dur"]
            print(f"   (📍 Auto-Detected: {dist} miles, {dur} mins)")
        else:
            # Fallback if route is unknown (e.g. new route)
            print("   (⚠️ Unknown Route - Using Defaults)")
            dist = 1000.0
            dur = 120.0

        # --- 3. PREPARE VECTORS ---
        SEQ_LEN = 5
        in_air = np.full((1, SEQ_LEN, 1), air_id)
        in_org = np.full((1, SEQ_LEN, 1), org_id)
        in_dst = np.full((1, SEQ_LEN, 1), dst_id)

        scaled_dur = dur / 400.0
        scaled_dist = dist / 3000.0

        hist_key = f"{origin}_{month}"
        hist_vec = self.history_lookup.get(hist_key, [0.0]*5)
        
        num_vec = np.zeros((1, SEQ_LEN, 7))
        num_vec[0, :, 0] = scaled_dur
        num_vec[0, :, 1] = scaled_dist
        num_vec[0, :, 2:] = hist_vec

        # --- 4. PREDICT ---
        probs = self.model.predict([in_air, in_org, in_dst, num_vec], verbose=0)[0]
        return probs, hist_vec

    def run(self):
        print("\n" + "="*50)
        print("🕵️   FLIGHT DELAY TOOL (Auto-Route)")
        print("="*50)
        
        while True:
            print("\n--- Analyze a Route (or 'q' to quit) ---")
            air = input("Airline (AA, UA...): ").strip().upper()
            if air == 'Q': break
            org = input("Origin (JFK, ATL...): ").strip().upper()
            dst = input("Dest (LAX, LHR...): ").strip().upper()
            
            try:
                month = int(input("Month (1-12): "))
            except ValueError: continue

            print("\n🔮 Diagnosing...")
            probabilities, history = self.predict_cause(air, org, dst, month)
            
            # REPORTING
            winner_idx = np.argmax(probabilities)
            print("-" * 50)
            print(f"⚠️  PRIMARY CAUSE: {CAUSE_LABELS[winner_idx]} ({probabilities[winner_idx]*100:.1f}%)")
            print("-" * 50)
            
            for i, label in enumerate(CAUSE_LABELS):
                bar_len = int(probabilities[i] * 20)
                bar = "█" * bar_len + "░" * (20 - bar_len)
                print(f"   {label:<25} {bar} {probabilities[i]*100:5.1f}%  [Hist: {history[i]:4.1f}m]")

if __name__ == "__main__":
    app = DelayCausePredictor()
    app.run()