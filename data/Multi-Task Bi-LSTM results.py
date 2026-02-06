import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from tensorflow.keras.models import load_model
from sklearn.metrics import classification_report, mean_absolute_error, accuracy_score, confusion_matrix

# --- 1. SETTINGS & LOAD ASSETS ---
WINDOW_SIZE = 10
MODEL_PATH = 'final_multitask_model.keras'
SCALER_PATH = 'scaler.pkl'
DATA_PATH = 'data/DelayFlights-cleaned-scaled.csv'

model = load_model(MODEL_PATH)
scaler = joblib.load(SCALER_PATH)

# --- 2. DATA RECONSTRUCTION (Defining X_test and Y_test) ---
df = pd.read_csv(DATA_PATH)
df['Flight Date'] = pd.to_datetime(df['Flight Date'])
df = df.sort_values(by=['Flight Date', 'Departure Block Hour'])

# Re-run Target Logic
delay_cols = ["Carrier Delay", "Weather Delay", "Airport Delay", "Security Delay", "Late Aircraft Delay"]
df['is_delayed'] = (df[delay_cols].sum(axis=1) > 0).astype(int)
df['primary_reason'] = np.argmax(df[delay_cols].values, axis=1)

# Apply Log Transform (Target for Regression)
for col in delay_cols:
    df[f'{col}_log'] = np.log1p(df[col])

# Apply Scaler to Numerical Features
num_cols = num_cols = [
    'Departure Block Hour', 'Day Of Week', 'Fly Time Scheduled', 
    'Distance Miles', 'Distance Group', 'Day of Month', 'Month'
]
df[num_cols] = scaler.transform(df[num_cols])

# Sequence Building Function
def prepare_test_data(df, window_size):
    X_cat, X_num = [], []
    y_bin, y_reason, y_reg = [], [], []
    
    cat_vals = df[['Origin Airport Code', 'Destination Airport Code']].values
    num_vals = df[num_cols].values
    bin_vals = df['is_delayed'].values
    reason_vals = df['primary_reason'].values
    reg_vals = df[[c + "_log" for c in delay_cols]].values

    # We only take the last 20% of the data to simulate the Test Set
    test_start = int(len(df) * 0.8)
    for i in range(test_start, len(df) - window_size):
        X_cat.append(cat_vals[i : i + window_size])
        X_num.append(num_vals[i : i + window_size])
        y_bin.append(bin_vals[i + window_size])
        y_reason.append(reason_vals[i + window_size])
        y_reg.append(reg_vals[i + window_size])
        
    return ([np.array(X_cat), np.array(X_num)], 
            {"binary": np.array(y_bin), "reason": np.array(y_reason), "regression": np.array(y_reg)})

X_test, Y_test = prepare_test_data(df, WINDOW_SIZE)

# --- 3. GENERATE PREDICTIONS ---
print("Running Multi-Task Inference...")
preds = model.predict(X_test)

# Unpack Heads
# Note: Check model.summary() for correct output order (Binary, Reason, Reg)
p_bin = (preds[0] > 0.5).astype(int)
p_reason = np.argmax(preds[1], axis=1)
p_reg_minutes = np.expm1(preds[2]) # Reverse Log1p
y_true_minutes = np.expm1(Y_test["regression"])

# --- 4. THE REPORT ---
print("\n" + "="*50)
print("             MULTI-TASK MODEL REPORT")
print("="*50)

# Binary Results
print(f"\n[Task 1] Delay Detection Accuracy: {accuracy_score(Y_test['binary'], p_bin):.2%}")

# Classification Results
print("\n[Task 2] Delay Reason Classification:")
reasons = ['Carrier', 'Airport', 'Weather', 'Security', 'Late Aircraft']
print(classification_report(Y_test['reason'], p_reason, target_names=reasons))

# Regression Results
print("\n[Task 3] Regression MAE (Actual Minutes):")
for i, col in enumerate(reasons):
    mae = mean_absolute_error(y_true_minutes[:, i], p_reg_minutes[:, i])
    print(f"{col:<15} | Error: {mae:.2f} min")

# --- 5. VISUALIZATION ---
plt.figure(figsize=(10, 6))
cm = confusion_matrix(Y_test['reason'], p_reason)
sns.heatmap(cm, annot=True, fmt='d', xticklabels=reasons, yticklabels=reasons, cmap='Blues')
plt.title("Confusion Matrix: Primary Delay Reason")
plt.ylabel('Actual Reason')
plt.xlabel('Predicted Reason')
plt.show()