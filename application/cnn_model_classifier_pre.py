import pandas as pd
import numpy as np
import tensorflow as tf
import json
import os
from keras.models import load_model

class DelayCausePredictor:
    __cause_labels = [
        "Carrier", "Weather", "Airport", 
        "Security", "Late Aircraft"
    ]
    
    def __init__(self, model_path, id_map_path, history_data_path, route_map_path):
        print("🔍 Initializing Tool...")
        self.model = None
        self.maps = {}
        self.history_lookup = {}
        self.route_lookup = {}
        self.model_path = model_path
        self.id_map_path = id_map_path
        self.history_data_path = history_data_path
        self.route_map_path = route_map_path
        self.load_resources()

    def load_resources(self):
        # 1. Load Model
        try:
            self.model = load_model(self.model_path)
        except: 
            print("❌ Classifier Model missing.")

        # 2. Load ID Maps
        try:
            with open(self.id_map_path, 'r') as f: self.maps = json.load(f)
        except: 
            print("❌ ID Maps missing.")

        # 3. Load Route Physics (NEW)
        try:
            with open(self.route_map_path, 'r') as f: self.route_lookup = json.load(f)
        except: 
            print("❌ Route Map missing. Run utils_route_mapper.py!")

        # 4. Load History
        try:
            with open(self.history_data_path, 'r') as f:
                self.history_lookup = json.load(f)
        except Exception as e:
            print(f"❌ History JSON missing. Run utils_history_mapper.py! ({e})")

    def predict_cause(self, airline, origin, dest, month, **kwargs):
        # --- 1. GET IDS ---
        air_id = airline
        if isinstance(airline, str):
            air_id = self.maps["airline"].get(airline, 0)
        org_id = origin
        if isinstance(origin, str):
            org_id = self.maps["origin"].get(origin, 0)
        dst_id = dest
        if isinstance(dest, str):
            dst_id = self.maps["dest"].get(dest, 0)

        # --- 2. AUTO-DETECT DISTANCE & DURATION ---
        route_key = f"{org_id}_{dst_id}"
        
        if route_key in self.route_lookup:
            # Found it! Use the auto-detected values
            dist = self.route_lookup[route_key]["dist"]
            dur  = self.route_lookup[route_key]["dur"]
        else:
            # Fallback if route is unknown (e.g. new route)
            dist = float(kwargs.get("Distance Miles", 1200.0))
            dur = float(kwargs.get("Fly Time Scheduled", 150.0))

        # --- 3. PREPARE VECTORS ---
        SEQ_LEN = 5
        in_air = np.full((1, SEQ_LEN, 1), air_id)
        in_org = np.full((1, SEQ_LEN, 1), org_id)
        in_dst = np.full((1, SEQ_LEN, 1), dst_id)

        scaled_dur = dur / 400.0
        scaled_dist = dist / 3000.0
        
        # UPDATED LOOKUP LOGIC
        hist_key = f"{origin}_{month}"
        hist_vec = self.history_lookup.get(hist_key, [0.0, 0.0, 0.0, 0.0, 0.0])
        
        num_vec = np.zeros((1, SEQ_LEN, 7))
        num_vec[0, :, 0] = scaled_dur
        num_vec[0, :, 1] = scaled_dist
        num_vec[0, :, 2:] = hist_vec

        # --- 4. PREDICT ---
        probs = self.model.predict([in_air, in_org, in_dst, num_vec], verbose=0)[0]
        return probs, hist_vec
    
    def predict(self, airline, origin, dest, month, **kwargs):
        probs, _ = self.predict_cause(airline, origin, dest, month, **kwargs)
        result = {}
        winner_idx = np.argmax(probs)
        result["primary_reason"] = self.__cause_labels[winner_idx]
        result["probability_of_delay"] = float(probs[winner_idx])
        result["predicted_delays_minutes"] = {}
        for i, label in enumerate(self.__cause_labels):
            result["predicted_delays_minutes"][label] = float(probs[i])
        return result

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
            print(f"⚠️  PRIMARY CAUSE: {self.__cause_labels[winner_idx]} ({probabilities[winner_idx]*100:.1f}%)")
            print("-" * 50)
            
            for i, label in enumerate(self.__cause_labels):
                bar_len = int(probabilities[i] * 20)
                bar = "█" * bar_len + "░" * (20 - bar_len)
                print(f"   {label:<25} {bar} {probabilities[i]*100:5.1f}%  [Hist: {history[i]:4.1f}m]")

if __name__ == "__main__":
    app = DelayCausePredictor()
    app.run()