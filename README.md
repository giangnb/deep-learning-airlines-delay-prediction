# Flight Delays — Vanilla ANN (Multi‑Output)

**Goal:** Predict **delay minutes by cause** for each flight:  
**Carrier**, **Weather**, **Airport**, **Security**, **Late Aircraft**.

**Type:** Deep Learning (tabular, multi‑output regression)  
**Stack:** Python · TensorFlow/Keras · scikit‑learn · Google Colab + Drive

---

## 1) Project Overview

This project trains a **multi‑output MLP (ANN)** that, given flight details (time block, day of week, month, airline, origin, destination, distance group, etc.), predicts **five continuous delay values**—one per cause—**in minutes**.

Key design choices:

- **Single ANN, five outputs** (shared representation → 5 neurons).
- **Leakage‑free** historical features (computed on **TRAIN only**, merged into VAL/TEST by key).
- **Robust training** for heavy‑tailed delays: **Huber loss**, **L2**, **Dropout**, **Softplus** outputs (non‑negative).
- **Scientific hygiene**: temporal split (no shuffle), scaler fit on TRAIN only, baselines, and per‑target metrics.

---

## 2) Repository Structure

```
DeepLearning_Project/
├─ data/
│  └─ DelayFlights-cleaned-handpick.csv      # (place your CSV here)
├─ models/
│  ├─ best_val_model.keras                   # best checkpoint (by val loss)
│  ├─ vanilla_ann_model_softplus/            # final Keras SavedModel
│  ├─ standard_scaler.pkl                    # StandardScaler (fit on TRAIN)
│  ├─ feats_airline_month.csv                # TRAIN-only lookups
│  ├─ feats_origin_slot.csv
│  ├─ feats_route_dow.csv
│  └─ metadata.json                          # feature_cols, keys, global_mean
├─ outputs/
│  └─ predictions_test.csv                   # test/new data predictions
└─ notebooks/
   ├─ train_model.ipynb                      # training pipeline
   └─ test_model.ipynb                       # inference/evaluation pipeline
```

> You can also use the **one‑file** notebook:
> - `flight_delay_ann_train_test.ipynb` → Train **and** Test in a single notebook.

---

## 3) Requirements

Run in **Google Colab** (recommended) or locally with:

```bash
python>=3.9
tensorflow>=2.12
scikit-learn>=1.2
pandas>=1.5
numpy>=1.23
matplotlib>=3.7
joblib>=1.3
```

No extra packages are needed in Colab (they come preinstalled).

---

## 4) Data

- **Input:** fully **numeric** and pre‑encoded columns; typical fields:  
  `depart_time_block, day_of_week, month, airline_code, origin_city, destination, distance_group`, …
- **Targets (minutes):**  
  `CarrierDelay, WeatherDelay, AirportDelay, SecurityDelay, LateAircraftDelay`  
  *(Aliases with spaces like “Carrier Delay” are auto‑handled.)*

### Leakage‑free historical features
Computed **only** on TRAIN, then merged into VAL/TEST:

- `feat_airline_month_mean`  ← mean total delay by *(airline_code, month)*
- `feat_origin_slot_mean`    ← mean total delay by *(origin_city, depart_time_block)*
- `feat_route_dow_mean`      ← mean total delay by *(route, day_of_week)*  
  where `route = origin_city + '_' + destination`

Missing lookups in VAL/TEST are filled with the **TRAIN global mean**.

---

## 5) Model

**Architecture (final):**

- Dense(512, ReLU) → BatchNorm → Dropout(0.2)  
- Dense(256, ReLU) → BatchNorm → Dropout(0.2)  
- Dense(128, ReLU)  
- Dense(**5**, **Softplus**)  ← (**non‑negative outputs**)

**Loss:** Huber (`delta ≈ 10.0`)  
**Regularization:** L2 (`5e‑5` to `1e‑4`) + Dropout (`0.2`–`0.3`)  
**Optimizer:** Adam (`lr ≈ 2e‑4`)  
**Callbacks:** EarlyStopping (patience 6, min_delta 0.005), ReduceLROnPlateau (×0.5, patience 3), ModelCheckpoint

---

## 6) Training & Evaluation Protocol

- **Temporal split** (no shuffle): TRAIN / VAL / TEST.  
- **Standardization**: `StandardScaler` **fit on TRAIN only**, applied to VAL/TEST.  
- **Metrics:** MAE & RMSE per target and **Global**.  
- **Baselines:**  
  - **Zeros** (predict 0 for all delays)  
  - **Mean‑by‑airline** (TRAIN‑only mean per airline)  
  → Report **skill**: `1 − MAE_model / MAE_baseline`.

> For sparse columns (e.g., **Security**), also report **MAE only‑positive** (rows where target > 0).

---

## 7) Quick Start (Colab)

### Option A — Two‑notebook flow
1. Create folders in Drive:
   ```
   MyDrive/DeepLearning_Project/{data, models, outputs, notebooks}
   ```
2. Upload your **CSV** to `data/`.
3. Open **`notebooks/train_model.ipynb`** → **Run all**.  
   Artifacts will appear in `models/`.
4. Open **`notebooks/test_model.ipynb`** → **Run all** to predict/evaluate (and write `outputs/predictions_test.csv`).

### Option B — One notebook for both
Open **`flight_delay_ann_train_test.ipynb`** and **Run all**.

---

## 8) Results (example template)

Replace with your latest numbers.

```text
GLOBAL — MAE: 6.80 | RMSE: 34.71

Carrier Delay        — MAE: 12.09 | RMSE: 53.51
Weather Delay        — MAE:  2.24 | RMSE: 22.91
Airport Delay        — MAE:  7.01 | RMSE: 23.73
Security Delay       — MAE:  0.28 | RMSE:  2.30
Late Aircraft Delay  — MAE: 13.90 | RMSE: 45.80
```

**Reading:**  
- Weather/Airport: lower MAE → features capture regular patterns.  
- Carrier/Late: higher error & RMSE → heavy‑tailed events and operational variability.  
- Security: very low MAE due to **zero‑inflation** (also report **only‑positive** MAE).

---

## 9) Reproducibility & Inference

Artifacts written to `models/`:

- `best_val_model.keras` (best checkpoint)  
- `vanilla_ann_model_softplus/` (final SavedModel)  
- `standard_scaler.pkl` (TRAIN‑fit scaler)  
- `feats_*.csv` (historical lookups)  
- `metadata.json` (feature order, target names, keys, global mean)

**Loading example (in Colab):**
```python
from tensorflow import keras
import joblib, json, pandas as pd

# Load model & scaler
model  = keras.models.load_model("/content/drive/MyDrive/DeepLearning_Project/models/best_val_model.keras")
scaler = joblib.load("/content/drive/MyDrive/DeepLearning_Project/models/standard_scaler.pkl")

# Load metadata for feature order and keys
with open("/content/drive/MyDrive/DeepLearning_Project/models/metadata.json") as f:
    meta = json.load(f)
feature_cols = meta["feature_cols"]
```

> Use the **test notebook** to replicate historical features and scale new data automatically.

---

## 10) Troubleshooting

- **FileNotFoundError (CSV):** confirm the CSV is under `data/`. The notebooks auto‑detect the first `.csv` if the default name is missing.  
- **Missing columns:** check `metadata.json["feature_cols"]` and ensure the same columns exist in test data.  
- **Validation rising (overfitting):** increase Dropout (0.25–0.30), L2 (→1e‑4), decrease LR (1e‑4), reduce width (256/128/64).  
- **Validation flat/high (underfitting/noise):** add simple historical features, try more rows, adjust Huber `delta` (7.5–10).  
- **Negative predictions:** not expected (Softplus output). If you use linear outputs, clip at 0 during evaluation.

---

## 11) Roadmap / Future Work

- Add more **historical aggregates** (e.g., destination×hour, carrier×route, route frequency).  
- Integrate **external weather** signals.  
- Explore **sequence models** (LSTM) to capture aircraft rotation chain effects.  
- Try **tabular transformers** for richer cross‑feature interactions.

---

## 12) License & Credits

- **Academic use** (course project).  
- Data: private class dataset, **do not redistribute**.  
- Authors: **Laura Lugo Latorre** & team (George Brown College).  
- Guidance: M365 Copilot.
