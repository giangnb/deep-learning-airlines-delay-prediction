from datetime import date
import pandas as pd
import numpy as np
import tensorflow as tf
import keras
from keras import layers
from sklearn.preprocessing import LabelEncoder, StandardScaler
from data.airport_code import AirportCode
import os

# A. Configuration
BATCH_SIZE = 64
LEARNING_RATE = 0.001
NUM_EPOCHS = 5
EMBEDDING_DIM = 32
NUM_TRANSFORMER_BLOCKS = 3
NUM_HEADS = 4
DROPOUT_RATE = 0.1
file_name = "data/DelayFlights-cleaned.csv" # Preprocessed data file
model_output = "models/tab_transformer_model" # Model save path
mappings_output = "mappings" # Mappings directory

# B. Data Loading & Preprocessing
def get_dataset(file_path):
    df = pd.read_csv(file_path)

    # --- Step 1: Define Targets ---
    target_cols = [
        "Carrier Delay", "Weather Delay", "Airport Delay", 
        "Security Delay", "Late Aircraft Delay"
    ]
    y = df[target_cols].values
    
    # --- Step 2: Define Features ---
    # Categorical Features (To be Label Encoded & Embedded)
    # Includes Airports + Time blocks + Date components (excluding full date)
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
    
    # Numerical Features (To be Scaled)
    num_cols = [
        "Fly Time Scheduled", 
        "Distance Miles"
    ]
    
    # --- Step 3: Preprocessing ---
    X_cat = {} # Categorical feature dictionary
    vocab_sizes = {} # Store sizes of distinct values in categorical features
    airport_code = AirportCode()
    
    # 3.1. Process Categorical Features
    for col in cat_cols:
        if col in ["Origin Airport Code", "Destination Airport Code"]:
            X_cat[col] = df[col].astype(int).values 
            vocab_sizes[col] = airport_code.data.shape[0]
        else:
            le = LabelEncoder()
            X_cat[col] = le.fit_transform(df[col].astype(str))
            vocab_sizes[col] = len(le.classes_)
            mapping = pd.DataFrame({
                'Category': le.classes_,
                'Encoded value': range(len(le.classes_))
            })
            if not os.path.exists(mappings_output):
                os.makedirs(mappings_output)
            mapping.to_csv(f'{mappings_output}/{col}.csv', index=False)
        
    # 3.2. Scale Numerical Features
    scaler = StandardScaler()
    X_num = scaler.fit_transform(df[num_cols])
    
    # Prepare input dictionary for Keras
    X = {col: X_cat[col] for col in cat_cols}
    X["numerical"] = X_num
    
    return X, y, vocab_sizes, cat_cols, len(num_cols)

# C. Load the data, split train/test sets
X, y, vocab_sizes, cat_cols, num_dim = get_dataset(file_name)

# D. Create TabTransformer model
def create_tabtransformer(vocab_sizes, cat_cols, num_dim, num_outputs):
    inputs = {}
    embeddings = []
    
    # --- 1. Categorical Embeddings ---
    for col in cat_cols:
        inputs[col] = layers.Input(shape=(1,), dtype=tf.int32, name=col)
        
        # Embedding: Integer -> Dense Vector
        emb = layers.Embedding(
            input_dim=vocab_sizes[col], 
            output_dim=EMBEDDING_DIM
        )(inputs[col])
        embeddings.append(emb)
        
    # Stack embeddings for Transformer (Batch, Num_Cats, Embed_Dim)
    x = layers.Concatenate(axis=1)(embeddings)
    
    # --- 2. Transformer Blocks ---
    for _ in range(NUM_TRANSFORMER_BLOCKS):
        # Attention
        x_att = layers.MultiHeadAttention(
            num_heads=NUM_HEADS, key_dim=EMBEDDING_DIM, dropout=DROPOUT_RATE
        )(x, x)
        x = layers.Add()([x, x_att])
        x = layers.LayerNormalization()(x)
        
        # Feed Forward
        x_ff = keras.Sequential([
            layers.Dense(EMBEDDING_DIM, activation="relu"),
            layers.Dense(EMBEDDING_DIM),
            layers.Dropout(DROPOUT_RATE)
        ])(x)
        x = layers.Add()([x, x_ff])
        x = layers.LayerNormalization()(x)
        
    # Flatten Categorical Features
    x = layers.Flatten()(x)
    
    # --- 3. Numerical Features ---
    inputs["numerical"] = layers.Input(shape=(num_dim,), name="numerical")
    x_num = inputs["numerical"]
    
    # --- 4. Combine & Regress ---
    x = layers.Concatenate()([x, x_num])
    
    # MLP Head
    x = layers.Dense(128, activation="relu")(x)
    x = layers.Dropout(0.1)(x)
    x = layers.Dense(64, activation="relu")(x)
    
    # Output: 5 neurons for the 5 delay types
    outputs = layers.Dense(num_outputs, activation="linear")(x)
    
    model = keras.Model(inputs=inputs, outputs=outputs)
    return model

# E. Create and Compile
model = create_tabtransformer(vocab_sizes, cat_cols, num_dim, num_outputs=5)

model.compile(
    optimizer=keras.optimizers.Adam(learning_rate=LEARNING_RATE),
    loss="mse",        # Mean Squared Error for regression
    metrics=["mae"]    # Mean Absolute Error
)

model.summary()

# F. Training
# Split 5% of data set for validation
history = model.fit(
    X, y,
    epochs=NUM_EPOCHS,
    batch_size=BATCH_SIZE,
    validation_split=0.05,
    verbose=1
)

print("Training Complete.")

print("Model Metrics:")
print(f"Training Loss: {history.history['loss'][-1]:.4f}")
print(f"Training MAE: {history.history['mae'][-1]:.4f}")
print(f"Validation Loss: {history.history['val_loss'][-1]:.4f}")
print(f"Validation MAE: {history.history['val_mae'][-1]:.4f}")

# G. Save the model
model.save(f"{model_output}/{date.today().isoformat()}_tab_transformer_model.h5")
