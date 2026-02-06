import joblib
from tensorflow.keras.models import load_model
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score, mean_absolute_error
import matplotlib.pyplot as plt

# 1. Load the Model and Scalers (No retraining needed!)
model = load_model('delay_prediction_bilstm.keras')
scaler_x = joblib.load('scaler_features.pkl')
scaler_y = joblib.load('scaler_targets.pkl')

# 2. LOAD AND PREPARE DATA (Exactly like your training script)
file_path = 'data/DelayFlights-cleaned.csv'
df = pd.read_csv(file_path)

# Sort and define columns
df['Flight Date'] = pd.to_datetime(df['Flight Date'])
df = df.sort_values(by=['Flight Date', 'Departure Block Hour'])

target_cols = ['Carrier Delay', 'Airport Delay', 'Weather Delay', 'Security Delay', 'Late Aircraft Delay']
feature_cols = [col for col in df.columns if col not in target_cols + ['Flight Date']]

# 3. APPLY THE LOADED SCALERS
# Use .transform() here, NOT .fit_transform()
df_scaled_features = scaler_x.transform(df[feature_cols])
df_scaled_targets = scaler_y.transform(df[target_cols])

# 4. RECREATE SEQUENCES
def create_sequences(features, targets, window_size=5):
    X, y = [], []
    for i in range(len(features) - window_size):
        X.append(features[i : i + window_size])
        y.append(targets[i + window_size])
    return np.array(X), np.array(y)

window_size = 5
X, y = create_sequences(df_scaled_features, df_scaled_targets, window_size)

# 5. RE-APPLY THE SUBSET AND SPLIT
# This ensures X_test is the exact same 20% the model hasn't seen yet
subset_ratio = 0.25  
subset_size = int(len(X) * subset_ratio)

X_subset = X[-subset_size:]
y_subset = y[-subset_size:]

split_index = int(len(X_subset) * 0.8)
X_test = X_subset[split_index:]
y_test = y_subset[split_index:]

# 2. Generate Predictions
# Note: Ensure X_test and y_test are available in your environment
print("Generating predictions on test set...")
predictions_scaled = model.predict(X_test)

# 3. Inverse Transform to Real Minutes
# This converts the [0, 1] scale back to actual delay minutes
y_true_minutes = scaler_y.inverse_transform(y_test)
y_pred_minutes = scaler_y.inverse_transform(predictions_scaled)

# 4. Calculate Individual Accuracy (MAE in Minutes)
target_cols = ['Carrier Delay', 'Airport Delay', 'Weather Delay', 'Security Delay', 'Late Aircraft Delay']

print("\n--- Individual Accuracy Report (Mean Absolute Error) ---")
print(f"{'Delay Category':<25} | {'Error in Minutes':<15}")
print("-" * 45)

for i, col in enumerate(target_cols):
    # Calculate the average difference between actual and predicted minutes
    mae = mean_absolute_error(y_true_minutes[:, i], y_pred_minutes[:, i])
    print(f"{col:<25} | {mae:<15.2f} min")

# 5. Overall Performance
overall_mae = mean_absolute_error(y_true_minutes, y_pred_minutes)
print("-" * 45)
print(f"{'OVERALL AVERAGE ERROR':<25} | {overall_mae:<15.2f} min")


# Indices: 0:Carrier, 1:Airport, 2:Weather, 3:Security, 4:Late Aircraft
target_index = 4 
target_name = target_cols[target_index]

plt.figure(figsize=(12, 6))

# Plotting a sample of 200 to keep the plot clean
plt.scatter(y_true_minutes[:200, target_index], y_pred_minutes[:200, target_index], alpha=0.6, color='orange', label='Predictions')

# Perfect Prediction Line
max_val = max(y_true_minutes[:200, target_index].max(), y_pred_minutes[:200, target_index].max())
plt.plot([0, max_val], [0, max_val], 'r--', label='Perfect Prediction')

plt.title(f'Actual vs. Predicted: {target_name}')
plt.xlabel('Actual Delay (Minutes)')
plt.ylabel('Predicted Delay (Minutes)')
plt.legend()
plt.grid(True, linestyle='--', alpha=0.7)
plt.show()