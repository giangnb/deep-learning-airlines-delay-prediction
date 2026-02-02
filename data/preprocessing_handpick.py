import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, MinMaxScaler


class DataPreprocessor:
    def __init__(self, raw_file = "data/DelayFlights-raw.csv", output_file = "data/DelayFlights-cleaned.csv", fields=None):
        self.raw_file = raw_file
        self.output_file = output_file
        # Updated fields based on approach.md
        self.fields = fields if fields is not None else [
            "Arrival Delay",            # Target (y)
            "Airline Code",             # Input (x) - Categorical
            "Departure Block Hour",     # Input (x) - Cyclical
            "Day Of Week",              # Input (x) - Cyclical
            "Flight Date",              # Input (x) - Date (for Month)
            "Origin Airport Code",      # Input (x) - Categorical
            "Destination Airport Code", # Input (x) - Categorical
            "Fly Time Scheduled",       # Input (x) - Numerical
            "Distance Miles"            # Input (x) - Numerical
        ]
    
    def process(self):
        """Load CSV, apply transformations (Scaling, Cyclical, Encoding), and save processed data."""
        # Load CSV file
        print(f"Loading data from {self.raw_file}...")
        df = pd.read_csv(self.raw_file)
        
        # Select only the fields specified
        # Check if fields exist to avoid errors
        available_fields = [f for f in self.fields if f in df.columns]
        missing_fields = set(self.fields) - set(available_fields)
        if missing_fields:
            print(f"Warning: Missing fields in raw data: {missing_fields}")
        
        df = df[available_fields]
        
        # Fill missing values (Basic imputation)
        df = df.fillna(0)
        
        # --- Date Handling ---
        df['Flight Date'] = pd.to_datetime(df['Flight Date'])
        df['Month'] = df['Flight Date'].dt.month
        
        # --- Strategy C: Categorical Data (Encoding) ---
        # 1. Airports
        airport_encoder = LabelEncoder()
        # Fit on both Origin and Destination to ensure consistent encoding
        all_airports = pd.concat([df['Origin Airport Code'], df['Destination Airport Code']]).unique()
        airport_encoder.fit(all_airports)
        
        # Save Airport mapping
        self._save_mapping(airport_encoder, 'data/DelayFlights-Airports-mapping.csv', 'Airport ID')
        
        df['Origin Airport Code'] = airport_encoder.transform(df['Origin Airport Code'])
        df['Destination Airport Code'] = airport_encoder.transform(df['Destination Airport Code'])
        
        # 2. Airlines
        airline_encoder = LabelEncoder()
        airline_encoder.fit(df['Airline Code'])
        
        # Save Airline mapping
        self._save_mapping(airline_encoder, 'data/DelayFlights-Airlines-mapping.csv', 'Airline Code')
        
        df['Airline Code'] = airline_encoder.transform(df['Airline Code'])

        # --- Strategy B: Cyclical Data (Time) ---
        # Helper for sin/cos transformation
        def to_cyclical(series, period):
            return np.sin(2 * np.pi * series / period), np.cos(2 * np.pi * series / period)

        # Departure Block Hour (0-23)
        df['DepHou_Sin'], df['DepHou_Cos'] = to_cyclical(df['Departure Block Hour'], 24)
        
        # Day Of Week (1-7)
        df['DayOfWeek_Sin'], df['DayOfWeek_Cos'] = to_cyclical(df['Day Of Week'], 7)
        
        # Month (1-12)
        df['Month_Sin'], df['Month_Cos'] = to_cyclical(df['Month'], 12)
        
        # Remove original cyclical columns (optional, but cleaner for ML input)
        # Keeping them might be useful for debugging, but usually we drop them. 
        # For now, I'll keep them but usually we drop 'Flight Date' at least.
        
        # --- Strategy A: Numerical Data (Scaling) ---
        scaler = MinMaxScaler()
        num_features = ['Fly Time Scheduled', 'Distance Miles']
        df[num_features] = scaler.fit_transform(df[num_features])
        
        # Drop Flight Date as it's no longer needed (we extracted Month)
        df = df.drop(columns=['Flight Date'])

        # Ensure all data is numeric
        for col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
        
        # Save processed data to CSV file
        print(f"Saving processed data to {self.output_file}...")
        df.to_csv(self.output_file, index=False)
        print("Done.")

    def _save_mapping(self, encoder, filepath, category_name):
        mapping = pd.DataFrame({
            category_name: encoder.classes_,
            'Encoded value': range(len(encoder.classes_))
        })
        mapping.to_csv(filepath, index=False)
    
    def load_data(self):
        """Load the output file into DataFrame and return."""
        return pd.read_csv(self.output_file)

if __name__ == "__main__":
    preprocessor = DataPreprocessor()
    preprocessor.process()
    data = preprocessor.load_data()
    print(data.info())
    print(data.head())