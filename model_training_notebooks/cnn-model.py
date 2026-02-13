import pandas as pd
import numpy as np
import tensorflow as tf
import time
import json
import os
from tensorflow.keras.models import Model
from tensorflow.keras import layers, callbacks, losses, optimizers
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight

# --- CONFIG ---
# Input Data (Created by data/model_classifier_data.py)
CLEAN_DATA_PATH = "data/Flights_classifier_clean.csv"

# Outputs (Renamed for consistency)
MODEL_SAVE_PATH = "data/model_classifier.keras"
METRICS_SAVE_PATH = "data/results_classifier.json"

SEQUENCE_LENGTH = 5
BATCH_SIZE = 4096
EPOCHS = 15

# --- GPU SETUP ---
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus: tf.config.experimental.set_memory_growth(gpu, True)
        print(f"✅ GPU Ready: {gpus[0].name}")
    except RuntimeError as e: print(e)

class ClassifierTrainer:
    def __init__(self):
        self.model = None
        self.history = None
        self.training_time = 0
        self.weights_dict = {}

    def load_data(self):
        print(f"📂 Loading pre-processed data from {CLEAN_DATA_PATH}...")
        if not os.path.exists(CLEAN_DATA_PATH):
            raise FileNotFoundError(f"Run data/model_classifier_data.py first!")
            
        self.df = pd.read_csv(CLEAN_DATA_PATH)
        
        # Define Columns
        self.cat_cols = ["Airline", "Origin", "Dest"]
        #  Numerical Features: Duration + Distance + 5 History Averages
        self.num_features = [
            "f_dur", "f_dist", 
            "Hist_Carrier Delay", "Hist_Weather Delay", "Hist_NAS Delay", 
            "Hist_Security Delay", "Hist_Late Aircraft Delay"
        ]
        self.target_col = "Label" # 0-4 Integer

    def get_sequences(self):
        # Convert to numpy
        X_cat = self.df[self.cat_cols].values.astype(np.int32)
        X_num = self.df[self.num_features].values.astype(np.float32)
        y = self.df[self.target_col].values.astype(np.int32)

        print(f"🔄 Generating sequences from {len(self.df)} late flights...")
        
        total_len = len(y)
        num_sequences = total_len - SEQUENCE_LENGTH
        
        idx = np.arange(num_sequences)[:, None] + np.arange(SEQUENCE_LENGTH)
        
        cat_seqs = X_cat[idx]
        num_seqs = X_num[idx]
        targets = y[SEQUENCE_LENGTH:]
        
        # Reshape Categoricals
        air_s = cat_seqs[:, :, 0].reshape(-1, SEQUENCE_LENGTH, 1)
        org_s = cat_seqs[:, :, 1].reshape(-1, SEQUENCE_LENGTH, 1)
        dst_s = cat_seqs[:, :, 2].reshape(-1, SEQUENCE_LENGTH, 1)
        
        # Split
        indices = np.arange(len(targets))
        tr_idx, te_idx = train_test_split(indices, test_size=0.2, shuffle=False)
        
        # ⚖️ COMPUTE CLASS WEIGHTS 
        print("⚖️ Calculating Class Weights...")
        y_train = targets[tr_idx]
        class_weights = compute_class_weight(
            class_weight='balanced', 
            classes=np.unique(y_train), 
            y=y_train
        )
        self.weights_dict = dict(enumerate(class_weights))
        print(f"   Weights: {self.weights_dict}")

        return (
            [air_s[tr_idx], org_s[tr_idx], dst_s[tr_idx], num_seqs[tr_idx]], y_train,
            [air_s[te_idx], org_s[te_idx], dst_s[te_idx], num_seqs[te_idx]], targets[te_idx]
        )

    def build_model(self):
        # Inputs
        in_air = layers.Input(shape=(SEQUENCE_LENGTH, 1))
        in_org = layers.Input(shape=(SEQUENCE_LENGTH, 1))
        in_dst = layers.Input(shape=(SEQUENCE_LENGTH, 1))
        in_num = layers.Input(shape=(SEQUENCE_LENGTH, len(self.num_features)))
        
        # Embeddings
        max_air = int(self.df["Airline"].max()) + 1
        max_org = int(self.df["Origin"].max()) + 1
        max_dst = int(self.df["Dest"].max()) + 1
        
        e_air = layers.Embedding(max_air, 5)(layers.Reshape((SEQUENCE_LENGTH,))(in_air))
        e_org = layers.Embedding(max_org, 10)(layers.Reshape((SEQUENCE_LENGTH,))(in_org))
        e_dst = layers.Embedding(max_dst, 10)(layers.Reshape((SEQUENCE_LENGTH,))(in_dst))
        
        # Merge
        merged = layers.Concatenate(axis=-1)([e_air, e_org, e_dst, in_num])
        x = layers.BatchNormalization()(merged) # Helps mix History features with Embeddings
        
        # CNN Architecture
        x = layers.Conv1D(64, 2, activation='relu')(x)
        x = layers.MaxPooling1D(2)(x)
        x = layers.Flatten()(x)
        x = layers.Dense(64, activation='relu')(x)
        x = layers.Dropout(0.3)(x)
        
        # Output: 5 Classes (Softmax)
        output = layers.Dense(5, activation='softmax')(x)
        
        self.model = Model(inputs=[in_air, in_org, in_dst, in_num], outputs=output)
        
        self.model.compile(
            optimizer='adam',
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy']
        )

    def train(self):
        X_train, y_train, X_test, y_test = self.get_sequences()
        
        print(f"🚀 Starting Training (History-Aware)...")
        start_time = time.time()
        
        self.history = self.model.fit(
            X_train, y_train,
            validation_data=(X_test, y_test),
            epochs=EPOCHS,
            batch_size=BATCH_SIZE,
            class_weight=self.weights_dict, # Apply weights
            callbacks=[
                callbacks.ModelCheckpoint(MODEL_SAVE_PATH, save_best_only=True, monitor='val_accuracy'),
                callbacks.EarlyStopping(patience=4, monitor='val_accuracy')
            ]
        )
        
        self.training_time = time.time() - start_time
        print(f"✅ Training Complete in {self.training_time:.2f} seconds.")
        print(f"💾 Model Saved to {MODEL_SAVE_PATH}")

    def save_metrics(self):
        # Extract metrics
        val_acc = max(self.history.history['val_accuracy'])
        val_loss = min(self.history.history['val_loss'])
        final_epoch = len(self.history.history['loss'])
        
        metrics_data = {
            "model_type": "CNN_History_Classifier",
            "class_weights": {k: float(v) for k, v in self.weights_dict.items()},
            "training_time_seconds": round(self.training_time, 2),
            "epochs_trained": final_epoch,
            "final_val_accuracy": round(float(val_acc), 4),
            "final_val_loss": round(float(val_loss), 4),
            "batch_size": BATCH_SIZE,
            "sequence_length": SEQUENCE_LENGTH,
            "history": {
                "loss": [round(x, 4) for x in self.history.history['loss']],
                "accuracy": [round(x, 4) for x in self.history.history['accuracy']],
                "val_loss": [round(x, 4) for x in self.history.history['val_loss']],
                "val_accuracy": [round(x, 4) for x in self.history.history['val_accuracy']]
            }
        }
        
        with open(METRICS_SAVE_PATH, 'w') as f:
            json.dump(metrics_data, f, indent=4)
        print(f"📊 Metrics saved to {METRICS_SAVE_PATH}")

if __name__ == "__main__":
    trainer = ClassifierTrainer()
    trainer.load_data()
    trainer.build_model()
    trainer.train()
    trainer.save_metrics()