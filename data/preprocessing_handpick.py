import pandas as pd
from sklearn.preprocessing import LabelEncoder


class DataPreprocessor:
    def __init__(self, raw_file = "data/DelayFlights-raw.csv", output_file = "data/DelayFlights-cleaned.csv", fields=None):
        self.raw_file = raw_file
        self.output_file = output_file
        self.fields = fields if fields is not None else [
            "Departure Block Hour",
            "Arrival Time Block",
            "Day Of Week",
            "Flight Date",
            "Origin Airport Code",
            "Destination Airport Code",
            "Fly Time Scheduled",
            "Distance Miles",
            "Distance Group",
            "Carrier Delay",
            "Weather Delay",
            "Airport Delay",
            "Security Delay",
            "Late Aircraft Delay"
        ]
    
    def process(self):
        """Load CSV, fill missing values, add date-based columns, and save processed data."""
        # Load CSV file
        df = pd.read_csv(self.raw_file)
        
        # Select only the fields specified
        df = df[self.fields]
        
        # Fill missing values
        df = df.fillna(0)
        
        # Convert Flight Date to datetime
        df['Flight Date'] = pd.to_datetime(df['Flight Date'])
        
        # Add Day of Month and Month columns
        df['Day of Month'] = df['Flight Date'].dt.day
        df['Month'] = df['Flight Date'].dt.month
        
        # Encode airport codes with shared LabelEncoder
        airport_encoder = LabelEncoder()
        all_airports = pd.concat([df['Origin Airport Code'], df['Destination Airport Code']]).unique()
        airport_encoder.fit(all_airports)

        # Save airport encoder mapping to CSV
        airport_mapping = pd.DataFrame({
            'Airport ID': airport_encoder.classes_,
            'Encoded value': range(len(airport_encoder.classes_))
        })
        airport_mapping.to_csv('data/DelayFlights-Airports-mapping.csv', index=False)
        
        # Apply encoding to both columns
        df['Origin Airport Code'] = airport_encoder.transform(df['Origin Airport Code'])
        df['Destination Airport Code'] = airport_encoder.transform(df['Destination Airport Code'])

        # Convert all columns except Flight Date to numerical dtype
        for col in df.columns:
            if col != 'Flight Date':
                df[col] = pd.to_numeric(df[col], errors='coerce')
        
        # Save processed data to CSV file
        df.to_csv(self.output_file, index=False)
    
    def load_data(self):
        """Load the output file into DataFrame and return."""
        return pd.read_csv(self.output_file)

if __name__ == "__main__":
    preprocessor = DataPreprocessor()
    preprocessor.process()
    data = preprocessor.load_data()
    print(data.info())