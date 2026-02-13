import pandas as pd
import numpy as np
import tensorflow as tf
import os
from datetime import date, datetime
from keras import layers, models
from keras.optimizers import Nadam
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from data.airport_code import AirportCode
import tensorflow as tf

gpus = tf.config.list_physical_devices('GPU')
if gpus:
    print(f"GPUs found: {len(gpus)}")
    # Prevent TensorFlow from grabbing all memory at once
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)
else:
    print("No GPU detected. Check your CUDA/cuDNN installation.")
    
# 0. Configuration
NUM_EPOCHS = 5
BATCH_SIZE = 32
LEARNING_RATE = 0.001
DROPOUT_RATE = 0.2
REBALANCE_DATA = True
COMBINE_TARGETS = False
file_name = "data/DelayFlights-cleaned.csv"
model_output = "model/ann_model"
log_dir = model_output + "/logs/" + datetime.now().strftime("%Y%m%d-%H%M%S")

# 1. Load Data
df = pd.read_csv(file_name)

# 2. Preprocessing
if COMBINE_TARGETS:
    df["_target"] = df["Carrier Delay"] + df["Weather Delay"] + df["Airport Delay"] + df["Security Delay"] + df["Late Aircraft Delay"]
    target_cols = ["_target"]
else:
    target_cols = ["Carrier Delay", "Weather Delay", "Airport Delay", "Security Delay", "Late Aircraft Delay"]

y = df[target_cols].values
target_scaler = StandardScaler()
y = target_scaler.fit_transform(y)

# Select 1M samples with delay and 1M without delay for training
if REBALANCE_DATA:
    df_delay = df[df[target_cols].sum(axis=1) > 0].sample(n=1000000, random_state=42)
    df_no_delay = df[df[target_cols].sum(axis=1) == 0].sample(n=1000000, random_state=42)
    df = pd.concat([df_delay, df_no_delay]).reset_index(drop=True)
    y = df[target_cols].values

# Numerical Features -> Scale
num_cols = ["Departure Block Hour", "Fly Time Scheduled", "Distance Miles"]
scaler = StandardScaler()
X = scaler.fit_transform(df[num_cols])
X = pd.DataFrame(X, columns=num_cols)

# Categorical Features -> Label Encode has already done during data cleaning
cat_cols = ["Origin Airport Code", "Destination Airport Code"]
for col in cat_cols:
    X = pd.concat([X, df[col].astype(int)], axis=1)

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.05, random_state=42)
print(f"Training samples: {X_train.shape[0]}, Testing samples: {X_test.shape[0]}")

# Prepare inputs for the model
train_inputs = [
    X_train["Origin Airport Code"].values,   # For input_origin
    X_train["Destination Airport Code"].values, # For input_dest
    X_train[num_cols].values                # For input_numeric
]

# 3. Model Architecture (Functional API)
# Inputs
input_origin = layers.Input(shape=(1,), name="origin_input")
input_dest = layers.Input(shape=(1,), name="dest_input")
input_numeric = layers.Input(shape=(len(num_cols),), name="numeric_input")

# Embeddings
# Size: (Vocab Size) -> (Vector Size, e.g., 10)
airport_encoder = AirportCode()
emb_origin = layers.Embedding(input_dim=airport_encoder.data.shape[0], output_dim=10)(input_origin)
emb_dest = layers.Embedding(input_dim=airport_encoder.data.shape[0], output_dim=10)(input_dest)

# Flatten embeddings to connect to Dense layers
flat_origin = layers.Flatten()(emb_origin)
flat_dest = layers.Flatten()(emb_dest)

# Concatenate all features
concat = layers.Concatenate()([flat_origin, flat_dest, input_numeric])

# Hidden Layers
x = layers.Dense(512, activation="swish")(concat)
x = layers.BatchNormalization()(x)
x = layers.Activation("swish")(x)
x = layers.Dropout(DROPOUT_RATE)(x)
x = layers.Dense(256, activation="swish")(x)
x = layers.BatchNormalization()(x)
x = layers.Dropout(DROPOUT_RATE)(x)
x = layers.Dense(128, activation="swish")(x)
x = layers.BatchNormalization()(x)
x = layers.Dropout(DROPOUT_RATE)(x)
x = layers.Dense(64, activation="swish")(x)
x = layers.Dropout(DROPOUT_RATE)(x)
x = layers.Dense(32, activation="swish")(x)

# Output Layer (5 neurons for 5 targets)
outputs = layers.Dense(len(target_cols), activation="linear", dtype="float32")(x) # Linear for regression

# Compile
model = models.Model(inputs=[input_origin, input_dest, input_numeric], outputs=outputs)
model.compile(optimizer=Nadam(learning_rate=LEARNING_RATE), loss='mse', metrics=['mae'])

# Summary
model.summary()

# 4. Train the Model
tensorboard_callback = tf.keras.callbacks.TensorBoard(
    log_dir=log_dir, 
    histogram_freq=1,    # Log weights/biases distributions every epoch
    embeddings_freq=1,   # Visualize your Airport Code embeddings!
    update_freq='epoch'  # How often to write logs ('batch' or 'epoch')
)
earlystop_callback = tf.keras.callbacks.EarlyStopping(patience=5, restore_best_weights=True)
reduce_lr_callback = tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=3, min_lr=1e-6)
history = model.fit(train_inputs, y_train, 
                    epochs=NUM_EPOCHS, 
                    batch_size=BATCH_SIZE, 
                    validation_split=0.1, 
                    callbacks=[tensorboard_callback, earlystop_callback, reduce_lr_callback],
                    verbose=1)

print("Training Complete.")

print("Model Metrics:")
print(f"Training Loss: {history.history['loss'][-1]:.4f}")
print(f"Training MAE: {history.history['mae'][-1]:.4f}")
print(f"Validation Loss: {history.history['val_loss'][-1]:.4f}")
print(f"Validation MAE: {history.history['val_mae'][-1]:.4f}")

# 5. Test the Model
test_inputs = [
    X_test["Origin Airport Code"].values,
    X_test["Destination Airport Code"].values,
    X_test[num_cols].values
]
test_loss, test_mae = model.evaluate(test_inputs, y_test)
print(f"Test Loss: {test_loss:.4f}")
print(f"Test MAE: {test_mae:.4f}")

# 6. Save the Model
model.save(f"{model_output}/{datetime.now().strftime('%Y%m%d-%H%M%S')}_{NUM_EPOCHS}epochs.keras")
print("Model saved.")
