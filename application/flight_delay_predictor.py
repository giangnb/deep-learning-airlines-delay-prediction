import tensorflow as tf
import numpy as np
import pandas as pd
import pickle
from datetime import datetime

class FlightDelayPredictor:
    def __init__(self, model_path, artifacts_path, model_type="ANN"):
        # 1. Load Model
        self.model = tf.keras.models.load_model(model_path)
        
        # 2. Load Artifacts (Statistics from training)
        with open(artifacts_path, "rb") as f:
            artifacts = pickle.load(f)
            
        self.origin_means = artifacts["origin_means"]
        self.dest_means = artifacts["dest_means"]
        self.global_mean = artifacts["global_mean"]
        self.scaler = artifacts["scaler"]
        self.model_type = model_type
        self.reasons = ["Carrier Delay", "Weather Delay", "Airport Delay", "Security Delay", "Late Aircraft Delay"]
        
        # Define the exact column order used in 'num_cols' during training
        self.num_cols_order = [
            'Fly Time Scheduled', 'Distance Miles', 'Distance Group',
            'hr_sin', 'hr_cos', 'dow_sin', 'dow_cos', 'month_sin', 'month_cos',
            'origin_avg_delay', 'dest_avg_delay'
        ]

    def preprocess_single_input(self, user_input):
        # --- A. Target Encoding ---
        origin = user_input.get("Origin Airport Code")
        dest = user_input.get("Destination Airport Code")
        # if the route doesn't appear in the original data
        origin_val = self.origin_means.get(origin, self.global_mean)
        dest_val = self.dest_means.get(dest, self.global_mean)
        
        # --- B. Cyclical Encoding ---
        # Helper to calculate sin/cos
        def get_cyclical(val, max_val):
            return np.sin(2 * np.pi * val / max_val), np.cos(2 * np.pi * val / max_val)

        hr_sin, hr_cos = get_cyclical(user_input['Departure Block Hour'], 24)
        dow_sin, dow_cos = get_cyclical(user_input['Day Of Week'], 7)
        month_sin, month_cos = get_cyclical(user_input['Month'], 12)

        # --- C. Assemble Numerical Vector ---
        # Create a dictionary first to ensure easy mapping
        raw_features = {
            'Origin Airport Code': origin,
            'Destination Airport Code': dest,
            'Fly Time Scheduled': user_input['Fly Time Scheduled'],
            'Distance Miles': user_input['Distance Miles'],
            'Distance Group': user_input['Distance Group'],
            'hr_sin': hr_sin, 'hr_cos': hr_cos,
            'dow_sin': dow_sin, 'dow_cos': dow_cos,
            'month_sin': month_sin, 'month_cos': month_cos,
            'origin_avg_delay': origin_val,
            'dest_avg_delay': dest_val
        }
        
        # Convert to list in the EXACT order of self.num_cols_order
        num_vector = [raw_features[col] for col in self.num_cols_order]
        
        # Reshape for Scaler: (1, N)
        num_vector = np.array(num_vector).reshape(1, -1)
        
        # --- E. Prepare Categorical Input ---
        # Assuming your model takes strings (via StringLookup) or you handle LabelEncoding here
        # For this example, we pass the raw code string in a 2D array
        if self.model_type == "TabTransformer":
            tab_input = {
                "Origin Airport Code": np.array([[origin]]),
                "Destination Airport Code": np.array([[dest]]),
                "numerical": self.scaler.fit_transform(num_vector).reshape(1, -1)
            }
            return tab_input
        else:
            scaled_num_vector = self.scaler.transform(num_vector)
            cat_vector = np.array([[origin, dest]])
            return [cat_vector, scaled_num_vector]
        

    def predict(self, user_input):
        # 1. Preprocess
        processed_inputs = self.preprocess_single_input(user_input)

        # 2. Inference
        predictions = self.model.predict(processed_inputs, verbose=0)
        
        # 3. Parse Multi-Task Outputs
        # Assuming output order: [binary, reason, regression]
        prob_delay, reason_probs, reg_log_vals = None, None, None
        if self.model_type == "TabTransformer":
            reg_log_vals = predictions[0]
        else:
            prob_delay = predictions[0][0][0] # Binary probability
            reason_probs = predictions[1][0]  # Class probabilities
            reg_log_vals = predictions[2][0]  # Regression (Log scale)

        # 4. Inverse Transform Regression (Log -> Real Minutes)
        pred_minutes = np.expm1(reg_log_vals)
        pred_minutes = np.maximum(pred_minutes, 0)

        reason = None
        if reason_probs is not None:
            reason_code = int(np.argmax(reason_probs))
            if reason_code < len(self.reasons):
                reason = self.reasons[reason_code]
        
        return {
            "probability_of_delay": float(prob_delay) if prob_delay is not None else None,
            "primary_reason": reason,
            "predicted_delays_minutes": {
                "Carrier": pred_minutes[0],
                "Weather": pred_minutes[1],
                "Airport": pred_minutes[2],
                "Security": pred_minutes[3],
                "Late Aircraft": pred_minutes[4]
            }
        }
    
if  __name__ == "__main__":
    model = FlightDelayPredictor(model_path="model/Transformer-20260202-084654-NEW DAT-5epochs-huber-scaled targets.h5", 
                                 artifacts_path="model/training_artifacts.pkl",
                                 model_type="TabTransformer")
    pred = model.predict({
        "Airline": 1,
        "Origin Airport Code": 182,
        "Destination Airport Code": 193,
        "Departure Block Hour": 17,
        "Day Of Week": 3,
        "Month": 7,
        "Fly Time Scheduled": 300,
        "Distance Miles": 1800,
        "Distance Group": 3
    })

    print(pred)