import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, LSTM, Bidirectional, Dense, Dropout, Embedding, Concatenate, Reshape
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import joblib

# --- 1. Load Data ---
file_path = 'data/DelayFlights-cleaned-scaled.csv' # Adjust to your actual path
df = pd.read_csv(file_path)

# Sort chronologically to maintain time-series integrity
df['Flight Date'] = pd.to_datetime(df['Flight Date'])
df = df.sort_values(by=['Flight Date', 'Departure Block Hour'])

# --- 2. Feature & Target Selection ---
# Based on your sample data:
cat_cols = ['Origin Airport Code', 'Destination Airport Code']
num_cols = [
    'Departure Block Hour', 'Day Of Week', 'Fly Time Scheduled', 
    'Distance Miles', 'Distance Group', 'Day of Month', 'Month'
]
delay_cols_log = [
    'Carrier Delay_log', 'Weather Delay_log', 'Airport Delay_log', 
    'Security Delay_log', 'Late Aircraft Delay_log'
]

# --- 3. Scaling ---
scaler = StandardScaler()
df[num_cols] = scaler.fit_transform(df[num_cols])

# --- 4. Sequence Building (Windowing) ---
def create_multitask_sequences(df, window_size=10):
    X_cat, X_num = [], []
    y_binary, y_reason, y_regression = [], [], []
    
    # Extract values
    cat_vals = df[cat_cols].values
    num_vals = df[num_cols].values
    bin_target = df['is_delayed'].values
    class_target = df['primary_reason'].values
    reg_target = df[delay_cols_log].values

    for i in range(len(df) - window_size):
        X_cat.append(cat_vals[i : i + window_size])
        X_num.append(num_vals[i : i + window_size])
        y_binary.append(bin_target[i + window_size])
        y_reason.append(class_target[i + window_size])
        y_regression.append(reg_target[i + window_size])
        
    return ([np.array(X_cat), np.array(X_num)], 
            {"out_binary": np.array(y_binary), 
             "out_reason": np.array(y_reason), 
             "out_regression": np.array(y_regression)})

# Using a subset of latest data for training efficiency
window_size = 10
X, Y = create_multitask_sequences(df.tail(500000), window_size)

# Split into Train/Test
split = int(0.85 * len(X[0]))
X_train = [X[0][:split], X[1][:split]]
X_test = [X[0][split:], X[1][split:]]
Y_train = {k: v[:split] for k, v in Y.items()}
Y_test = {k: v[split:] for k, v in Y.items()}

# --- 5. Model Architecture ---
# Input Layers
input_cat = Input(shape=(window_size, 2), name="input_categorical")
input_num = Input(shape=(window_size, len(num_cols)), name="input_numerical")

# Embedding for Airport IDs (Assuming max ID of 500 based on your 144/217 sample)
emb = Embedding(input_dim=500, output_dim=8)(input_cat)
emb_reshaped = Reshape((window_size, 16))(emb) 

# Combine Inputs
merged = Concatenate(axis=-1)([emb_reshaped, input_num])

# Bi-LSTM Core
x = Bidirectional(LSTM(128, return_sequences=True))(merged)
x = Dropout(0.2)(x)
x = Bidirectional(LSTM(64))(x)
shared_dense = Dense(64, activation='relu')(x)

# Three Task-Specific Output Heads
out_binary = Dense(1, activation='sigmoid', name='out_binary')(shared_dense)
out_reason = Dense(5, activation='softmax', name='out_reason')(shared_dense)
out_regression = Dense(5, activation='linear', name='out_regression')(shared_dense)

model = Model(inputs=[input_cat, input_num], outputs=[out_binary, out_reason, out_regression])

# --- 6. Compile & Train ---
model.compile(
    optimizer='RMSprop',
    loss={
        "out_binary": "binary_crossentropy",
        "out_reason": "sparse_categorical_crossentropy",
        "out_regression": "mse"
    },
    loss_weights={"out_binary": 1.0, "out_reason": 0.8, "out_regression": 1.2}
)

early_stop = EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)

model.fit(
    X_train, Y_train,
    validation_data=(X_test, Y_test),
    epochs=15,
    batch_size=512,
    callbacks=[early_stop]
)

# Save results
model.save('final_multitask_model.keras')
joblib.dump(scaler, 'scaler.pkl')