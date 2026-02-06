from sklearn.preprocessing import LabelEncoder, StandardScaler
import pandas as pd
import keras
import numpy as np
from data.mapping_reader import MappingReader

cat_cols = [
        "Origin Airport Code", 
        "Destination Airport Code",
        "Departure Block Hour", 
        "Arrival Time Block", 
        "Day Of Week", 
        "Distance Group", 
        "Day of Month", 
        "Month"
]
num_cols = [
        "Fly Time Scheduled", 
        "Distance Miles"
]
target_cols = [
        "Carrier Delay", "Weather Delay", "Airport Delay", 
        "Security Delay", "Late Aircraft Delay"
]

file_name = "data/DelayFlights-cleaned.csv" # Preprocessed data file
dataset = pd.read_csv(file_name)
model_output = "models/tab_transformer_model" # Model save path

mapping_reader = MappingReader()

target_scaler = StandardScaler()
target_scaler.fit_transform(dataset[target_cols].values)

print("\n=== Prediction on New Sample ===")
sample_input = {}

# Collect categorical features
for col in cat_cols:
    user_input = input(f"Enter {col}: ")
    if col in ["Origin Airport Code", "Destination Airport Code"]:
        sample_input[col] = np.array([int(user_input)])
    else:
        # For non-airport categorical features, encode the input
        try:
            sample_input[col] = np.array([mapping_reader.get_mapping_value(col, user_input)])
        except ValueError:
            print(f"Warning: '{user_input}' not found in training data for {col}")
            sample_input[col] = np.array([0])

# Collect numerical features
scaler = StandardScaler()
df_sample = pd.read_csv(file_name)
scaler.fit(df_sample[num_cols])

num_values = []
for col in num_cols:
    user_input = float(input(f"Enter {col}: "))
    num_values.append(user_input)

sample_input["numerical"] = scaler.transform([num_values])

# Make prediction
model = keras.models.load_model(model_output)
prediction = model.predict(sample_input, verbose=0)

# Display results
delay_types = ["Carrier Delay", "Weather Delay", "Airport Delay", "Security Delay", "Late Aircraft Delay"]
print("\n=== Prediction Results ===")
for i, delay_type in enumerate(delay_types):
    print(f"{delay_type}: {prediction[0][i]:.2f} minutes")