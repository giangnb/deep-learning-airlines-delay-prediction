import pandas as pd 
import numpy as np 
import tensorflow as tf 
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Bidirectional, Dense, Dropout, Input
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.callbacks import EarlyStopping
import joblib
import os

## Data Loading & Preparation

file_path = 'data/DelayFlights-cleaned.csv'

if not os.path.exists(file_path):
    print(f"Error: {file_path} not found")
    exit()
    
df = pd.read_csv(file_path)

#Sort the data chronologically
df['Flight Date'] = pd.to_datetime(df['Flight Date'])
df = df.sort_values(by=['Flight Date', 'Departure Block Hour'])

#Target Columns 
target_cols = [
    'Carrier Delay',
    'Airport Delay', 
    'Weather Delay',
    'Security Delay',
    'Late Aircraft Delay'
]

feature_cols = [col for col in df.columns if col not in target_cols + ['Flight Date']]

## Scaling - Normalization 
scaler_x = MinMaxScaler()
scaler_y = MinMaxScaler()

df_scaled_features = scaler_x.fit_transform(df[feature_cols])
df_scaled_targets = scaler_y.fit_transform(df[target_cols])

## Create Sliding windows (sequences)

def create_sequences(features, targets, window_size):
    """
    Groups data into windows. 
    Example: Look at flights 1-5 to predict delays for flight 6.
    """
    X, y = [], []
    for i in range(len(features) - window_size):
        # Grab a window of features
        X.append(features[i : i + window_size])
        # Grab the target for the flight immediately following that window
        y.append(targets[i + window_size])
    return np.array(X), np.array(y)

# use a window of 5 (the model looks at the 5 previous flights)
window_size = 10
X, y = create_sequences(df_scaled_features, df_scaled_targets, window_size)

# Let's take a subset of 500,000 samples instead of 6.6 million
subset_ratio = 0.25  
subset_size = int(len(X) * subset_ratio)

#Take the LATEST data (often more relevant)
X_subset = X[-subset_size:]
y_subset = y[-subset_size:]

# Update the split using the subset
split_index = int(len(X_subset) * 0.8)
X_train, X_test = X_subset[:split_index], X_subset[split_index:]
y_train, y_test = y_subset[:split_index], y_subset[split_index:]

print(f"Training shapeL {X_train.shape}")
print(f"Testing shape : {y_train.shape}")


# Building The BI-LSTM Model 

model = Sequential([
    Input(shape=(window_size, len(feature_cols))),
    Bidirectional(LSTM(128, return_sequences=True)),
    Dropout(0.2), # Dropout prevents overfitting, a core DL concept
    Bidirectional(LSTM(32)),
    Dropout(0.2),
    Dense(32, activation='relu'),
    Dense(5, activation='linear') # 5 units for the 5 target delay types
])

# Using Mean Squared Error (MSE) for regression
model.compile(optimizer='RMSprop', loss='mse', metrics=['mae'])

model.summary()
    
# Training and Evaluation

early_stop = EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)

print("\nStarting model training...")
history = model.fit(
    X_train, y_train,
    epochs=20, 
    batch_size=512,
    validation_split=0.1,
    # callbacks=[early_stop],
    verbose=1
)

# Evaluate on the unseen test set
test_loss, test_mae = model.evaluate(X_test, y_test)
print(f"\nFinal Test Mean Absolute Error: {test_mae}")


# 1. Save the Keras model
model.save('delay_prediction_bilstm.keras')
print("Model saved as 'delay_prediction_bilstm.h5'")

# 2. Save the scalers (Essential for un-scaling results later)
joblib.dump(scaler_x, 'scaler_features.pkl')
joblib.dump(scaler_y, 'scaler_targets.pkl')
print("Scalers saved as .pkl files")

