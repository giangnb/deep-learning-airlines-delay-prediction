from __future__ import annotations

import os
import csv
import json
from dataclasses import dataclass
from datetime import datetime, date
from collections import defaultdict, deque
from typing import Dict, Iterator, Tuple, Optional, List

import numpy as np
import tensorflow as tf

# Optional (for evaluation). Install: pip install scikit-learn
from sklearn.metrics import classification_report, confusion_matrix, f1_score

# -----------------------------
# Configuration
# -----------------------------

@dataclass
class Config:
    csv_path: str = "flights_sorted.csv"

    # Sequence
    T: int = 12
    stride: int = 5 # >1 reduces dataset size
    min_gap_minutes: Optional[int] = 180  # break sequences if time gap > this; set None to disable

    # Date splits (time-based, avoids leakage)
    # Example: train <= 2023-06-30, val <= 2023-10-31, test <= 2023-12-31
    train_end: str = "2023-06-30"
    val_end: str = "2023-10-31"
    test_end: str = "2023-12-31"

    # Training
    batch_size: int = 512
    shuffle_buffer: int = 20000
    epochs: int = 5
    lr: float = 1e-3

    # Embeddings
    emb_dim_dest: int = 16

    # Output
    out_dir: str = "./artifacts_airport_rnn"
    mapping_path: str = "./artifacts_airport_rnn/airport_mapping.json"

    # Class handling
    use_class_weights: bool = True


# -----------------------------
# Labels
# -----------------------------
# Update to include Navigation if you have a column for it.
LABELS = ["NoDelay", "Weather", "Carrier", "Airport", "Security", "LateAircraft"]
LABEL_TO_ID = {name: i for i, name in enumerate(LABELS)}
ID_TO_LABEL = {i: name for name, i in LABEL_TO_ID.items()}
NUM_CLASSES = len(LABELS)

# Column names (adjust if needed)
COL_FLIGHT_DATE = "Flight Date"
COL_ORIGIN = "Origin Airport Code"
COL_DEST = "Destination Airport Code"
COL_DEP_HOUR = "Departure Block Hour"
COL_ARR_BLOCK = "Arrival Time Block"
COL_DOW = "Day Of Week"
COL_DOM = "Day of Month"
COL_MONTH = "Month"
COL_FLY_SCHED = "Fly Time Scheduled"
COL_DIST_MILES = "Distance Miles"
COL_DIST_GROUP = "Distance Group"

COL_CARRIER_DELAY = "Carrier Delay"
COL_WEATHER_DELAY = "Weather Delay"
COL_AIRPORT_DELAY = "Airport Delay"
COL_SECURITY_DELAY = "Security Delay"
COL_LATE_DELAY = "Late Aircraft Delay"


# -----------------------------
# Utility: parsing & features
# -----------------------------

def parse_date(s: str) -> date:
    return datetime.strptime(s, "%Y-%m-%d").date()

def parse_sort_time(row: Dict[str, str]) -> datetime:
    """
    Timestamp from Flight Date + Departure Block Hour (approx HH:00).
    Used for gap detection. If you have minute-level departure time, use it instead.
    """
    d = datetime.strptime(row[COL_FLIGHT_DATE], "%Y-%m-%d")
    dep_hour = int(row[COL_DEP_HOUR])
    return d.replace(hour=dep_hour, minute=0, second=0, microsecond=0)

def make_delay_class(row: Dict[str, str]) -> int:
    """
    Convert multiple delay-minute columns to a single class id.
    Priority order is important; adjust to your business definition.
    """
    w = float(row.get(COL_WEATHER_DELAY, 0) or 0)
    c = float(row.get(COL_CARRIER_DELAY, 0) or 0)
    a = float(row.get(COL_AIRPORT_DELAY, 0) or 0)
    s = float(row.get(COL_SECURITY_DELAY, 0) or 0)
    l = float(row.get(COL_LATE_DELAY, 0) or 0)

    if w > 0: return LABEL_TO_ID["Weather"]
    if c > 0: return LABEL_TO_ID["Carrier"]
    if a > 0: return LABEL_TO_ID["Airport"]
    if s > 0: return LABEL_TO_ID["Security"]
    if l > 0: return LABEL_TO_ID["LateAircraft"]
    return LABEL_TO_ID["NoDelay"]

def make_numeric_features(row: Dict[str, str]) -> np.ndarray:
    """
    Numeric timestep features: (F_NUM,) float32
    DO NOT include delay-minute columns here (label leakage).
    """
    vals = [
        float(row[COL_DEP_HOUR]),
        float(row[COL_ARR_BLOCK]),
        float(row[COL_DOW]),
        float(row[COL_DOM]),
        float(row[COL_MONTH]),
        float(row[COL_FLY_SCHED]),
        float(row[COL_DIST_MILES]),
        float(row[COL_DIST_GROUP]),
    ]
    return np.asarray(vals, dtype=np.float32)

F_NUM = 8  # must match make_numeric_features length

def safe_int(x: str) -> int:
    return int(float(x))  # handles "144" or "144.0"


# -----------------------------
# Step 1: Build airport ID mapping (0..N-1)
# -----------------------------

def build_airport_mapping(csv_path: str) -> Dict[int, int]:
    """
    Build mapping from original airport codes to compact IDs.
    Reads CSV once (streaming).
    """
    airport_codes = set()
    with open(csv_path, "r", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            airport_codes.add(safe_int(row[COL_ORIGIN]))
            airport_codes.add(safe_int(row[COL_DEST]))
    codes_sorted = sorted(airport_codes)
    return {code: i for i, code in enumerate(codes_sorted)}

def save_mapping(mapping: Dict[int, int], path: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        json.dump(mapping, f)

def load_mapping(path: str) -> Dict[int, int]:
    with open(path, "r") as f:
        d = json.load(f)
    # json keys become strings
    return {int(k): int(v) for k, v in d.items()}


# -----------------------------
# Step 2: Streaming sequence generator with splits
# -----------------------------

def which_split(flight_date: date, train_end: date, val_end: date, test_end: date) -> str:
    if flight_date <= train_end:
        return "train"
    if flight_date <= val_end:
        return "val"
    if flight_date <= test_end:
        return "test"
    return "skip"

def build_sequence_generator(
    csv_path: str,
    mapping: Dict[int, int],
    split_name: str,
    T: int,
    stride: int,
    train_end: date,
    val_end: date,
    test_end: date,
    min_gap_minutes: Optional[int] = None,
) -> Iterator[Tuple[np.ndarray, np.ndarray, np.int32]]:
    """
    Yields (x_num_seq, x_dest_seq, y) for the chosen split.
      x_num_seq: (T, F_NUM) float32
      x_dest_seq: (T,) int32 (mapped IDs)
      y: scalar int32 class id for NEXT flight
    """
    # rolling buffers per origin (mapped origin id)
    num_bufs: Dict[int, deque] = defaultdict(lambda: deque(maxlen=T+1))
    dest_bufs: Dict[int, deque] = defaultdict(lambda: deque(maxlen=T+1))
    time_bufs: Dict[int, deque] = defaultdict(lambda: deque(maxlen=T+1))
    y_bufs: Dict[int, deque] = defaultdict(lambda: deque(maxlen=T+1))
    step_count: Dict[int, int] = defaultdict(int)

    with open(csv_path, "r", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            d = parse_date(row[COL_FLIGHT_DATE])
            sp = which_split(d, train_end, val_end, test_end)
            if sp != split_name:
                # We still must update buffers? For strict time-split, better NOT mix:
                # We avoid cross-split leakage by clearing buffers when split changes per origin.
                # Easiest: just ignore rows not in this split, but sequences might link across boundary.
                # So we hard-reset whenever row isn't in current split.
                # This is safer and recommended.
                origin_code = safe_int(row[COL_ORIGIN])
                origin = mapping.get(origin_code)
                if origin is not None:
                    num_bufs[origin].clear()
                    dest_bufs[origin].clear()
                    time_bufs[origin].clear()
                    y_bufs[origin].clear()
                    step_count[origin] = 0
                continue

            origin = mapping[safe_int(row[COL_ORIGIN])]
            dest = mapping[safe_int(row[COL_DEST])]

            t = parse_sort_time(row)
            y = np.int32(make_delay_class(row))
            x_num = make_numeric_features(row).astype(np.float32)
            x_dest = np.int32(dest)

            # optional gap-based reset (per origin)
            if min_gap_minutes is not None and len(time_bufs[origin]) > 0:
                prev_t = time_bufs[origin][-1]
                gap_min = (t - prev_t).total_seconds() / 60.0
                if gap_min > min_gap_minutes:
                    num_bufs[origin].clear()
                    dest_bufs[origin].clear()
                    time_bufs[origin].clear()
                    y_bufs[origin].clear()
                    step_count[origin] = 0

            num_bufs[origin].append(x_num)
            dest_bufs[origin].append(x_dest)
            time_bufs[origin].append(t)
            y_bufs[origin].append(y)

            # Produce sample when we have T+1 items
            if len(num_bufs[origin]) == T + 1:
                step_count[origin] += 1
                if (step_count[origin] - 1) % stride == 0:
                    x_num_seq = np.stack(list(num_bufs[origin])[:T], axis=0)          # (T, F_NUM)
                    x_dest_seq = np.asarray(list(dest_bufs[origin])[:T], dtype=np.int32)  # (T,)
                    y_target = np.int32(list(y_bufs[origin])[T])                      # label of next flight
                    yield (x_num_seq, x_dest_seq, y_target)

                # slide by 1
                num_bufs[origin].popleft()
                dest_bufs[origin].popleft()
                time_bufs[origin].popleft()
                y_bufs[origin].popleft()


def make_tf_dataset(
    cfg: Config,
    mapping: Dict[int, int],
    split_name: str,
    class_weights_accumulator: Optional[np.ndarray] = None,
) -> tf.data.Dataset:
    """
    Returns a tf.data.Dataset yielding (({"num":..., "dest":...}), y).
    Optionally accumulates class counts if class_weights_accumulator is provided.
    """
    train_end = parse_date(cfg.train_end)
    val_end = parse_date(cfg.val_end)
    test_end = parse_date(cfg.test_end)

    output_signature = (
        tf.TensorSpec(shape=(cfg.T, F_NUM), dtype=tf.float32),
        tf.TensorSpec(shape=(cfg.T,), dtype=tf.int32),
        tf.TensorSpec(shape=(), dtype=tf.int32),
    )

    gen = lambda: build_sequence_generator(
        csv_path=cfg.csv_path,
        mapping=mapping,
        split_name=split_name,
        T=cfg.T,
        stride=cfg.stride,
        train_end=train_end,
        val_end=val_end,
        test_end=test_end,
        min_gap_minutes=cfg.min_gap_minutes,
    )

    ds = tf.data.Dataset.from_generator(gen, output_signature=output_signature)

    # Optionally estimate class counts for weights (one pass). We'll do a lightweight method:
    # If you need exact counts, run a separate pass over the generator.
    if class_weights_accumulator is not None and split_name == "train":
        # This iterates once over ds; if dataset is huge, you may skip exact weights or sample counts.
        for _, _, y in ds.take(20000):  # sample first N to estimate (fast). Increase if you want.
            class_weights_accumulator[int(y.numpy())] += 1

        # Recreate ds after iterating it (because generators are exhausted)
        ds = tf.data.Dataset.from_generator(gen, output_signature=output_signature)

    if split_name == "train":
        ds = ds.shuffle(cfg.shuffle_buffer, reshuffle_each_iteration=True)

    ds = ds.batch(cfg.batch_size, drop_remainder=True).prefetch(tf.data.AUTOTUNE)

    # Map into Keras input dict
    ds = ds.map(lambda x_num, x_dest, y: ({"num": x_num, "dest": x_dest}, y),
                num_parallel_calls=tf.data.AUTOTUNE)
    return ds


# -----------------------------
# Step 3: Build model
# -----------------------------

def build_model(num_airports: int, T: int, emb_dim_dest: int, num_classes: int) -> tf.keras.Model:
    num_in = tf.keras.Input(shape=(T, F_NUM), dtype=tf.float32, name="num")
    dest_in = tf.keras.Input(shape=(T,), dtype=tf.int32, name="dest")

    dest_emb = tf.keras.layers.Embedding(
        input_dim=num_airports,
        output_dim=emb_dim_dest,
        name="dest_emb"
    )(dest_in)  # (B, T, emb_dim)

    x = tf.keras.layers.Concatenate(name="concat")([num_in, dest_emb])  # (B, T, F_NUM+emb_dim)

    x = tf.keras.layers.GRU(128, name="gru")(x)
    x = tf.keras.layers.Dense(64, activation="relu", name="dense1")(x)
    x = tf.keras.layers.Dropout(0.2, name="dropout")(x)
    out = tf.keras.layers.Dense(num_classes, activation="softmax", name="out")(x)

    return tf.keras.Model(inputs={"num": num_in, "dest": dest_in}, outputs=out)


# -----------------------------
# Step 4: Training
# -----------------------------

def compute_class_weights_from_counts(counts: np.ndarray) -> Dict[int, float]:
    """
    Simple inverse-frequency weights: total/(K*count_k)
    """
    counts = counts.astype(np.float64)
    counts[counts == 0] = 1.0
    total = counts.sum()
    K = len(counts)
    weights = total / (K * counts)
    return {i: float(w) for i, w in enumerate(weights)}

def train(cfg: Config) -> tf.keras.Model:
    os.makedirs(cfg.out_dir, exist_ok=True)

    # 1) Mapping
    if os.path.exists(cfg.mapping_path):
        mapping = load_mapping(cfg.mapping_path)
    else:
        mapping = build_airport_mapping(cfg.csv_path)
        save_mapping(mapping, cfg.mapping_path)

    num_airports = max(mapping.values()) + 1
    print(f"Loaded airport mapping: {len(mapping)} codes -> num_airports={num_airports}")

    # 2) Datasets
    class_counts = np.zeros(NUM_CLASSES, dtype=np.int64) if cfg.use_class_weights else None
    train_ds = make_tf_dataset(cfg, mapping, "train", class_weights_accumulator=class_counts)
    val_ds = make_tf_dataset(cfg, mapping, "val")

    # 3) Model
    model = build_model(num_airports=num_airports, T=cfg.T, emb_dim_dest=cfg.emb_dim_dest, num_classes=NUM_CLASSES)
    model.compile(
        optimizer=tf.keras.optimizers.Adam(cfg.lr),
        loss=tf.keras.losses.SparseCategoricalCrossentropy(),
        metrics=["accuracy"],
    )
    model.summary()

    # 4) Class weights (optional)
    class_weight = None
    if cfg.use_class_weights and class_counts is not None:
        # class_counts here is an estimate from a sample of train_ds (see make_tf_dataset)
        class_weight = compute_class_weights_from_counts(class_counts)
        print("Estimated class counts (sample):", class_counts)
        print("Class weights:", class_weight)

    # 5) Callbacks
    ckpt_path = os.path.join(cfg.out_dir, "best_model.keras")
    callbacks = [
        tf.keras.callbacks.ModelCheckpoint(ckpt_path, monitor="val_loss", save_best_only=True),
        tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=2, restore_best_weights=True),
        tf.keras.callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=1),
    ]

    # 6) Fit
    model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=cfg.epochs,
        class_weight=class_weight,
        callbacks=callbacks,
        verbose=1,
    )

    # Save final
    final_path = os.path.join(cfg.out_dir, "final_model.keras")
    model.save(final_path)
    print(f"Saved model to: {final_path}")
    return model


# -----------------------------
# Step 5: Evaluation
# -----------------------------

def evaluate(cfg: Config, model: tf.keras.Model) -> None:
    mapping = load_mapping(cfg.mapping_path)
    test_ds = make_tf_dataset(cfg, mapping, "test")

    y_true = []
    y_pred = []

    for batch in test_ds:
        x, y = batch
        probs = model.predict(x, verbose=0)
        preds = np.argmax(probs, axis=1)
        y_true.extend(y.numpy().tolist())
        y_pred.extend(preds.tolist())

    y_true = np.asarray(y_true, dtype=np.int32)
    y_pred = np.asarray(y_pred, dtype=np.int32)

    print("\nMacro F1:", f1_score(y_true, y_pred, average="macro"))

    print("\nClassification report:")
    print(classification_report(
        y_true, y_pred,
        target_names=[ID_TO_LABEL[i] for i in range(NUM_CLASSES)],
        digits=4
    ))

    print("\nConfusion matrix (rows=true, cols=pred):")
    print(confusion_matrix(y_true, y_pred))


# -----------------------------
# Main
# -----------------------------

if __name__ == "__main__":
    cfg = Config(
        csv_path="./data/DelayFlights-cleaned-handpick/DelayFlights-cleaned-scaled.csv",
        T=24,
        stride=2,
        min_gap_minutes=360,
        train_end="2023-06-30",
        val_end="2023-10-31",
        test_end="2023-12-31",
        batch_size=512,
        shuffle_buffer=100000,
        epochs=8,
        lr=3e-4,
        out_dir="./artifacts_airport_rnn",
        mapping_path="./artifacts_airport_rnn/airport_mapping.json",
        use_class_weights=True,
    )

    model = train(cfg)
    evaluate(cfg, model)
