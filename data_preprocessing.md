# 🧹 Data Preprocessing Pipeline

 Data processing scripts to prepare specific subsets of the raw data (`DelayFlights-raw.csv`) for training and inference.

## 1. Classifier Data (`model_classifier_data.py`)
**Goal:** Create a dataset for cause classification.

* **Filter Logic:** **Late Flights Only**.
    * `df = df[df["TargetDelay"] >= 15]`
    * **Reason:** We cannot classify the cause of a delay if the flight was on time.
* **Historical data (Target Encoding):**
    * Calculates the **Mean Delay** for every cause, grouped by `Origin` and `Month`.
    * These averages are merged back into the row as **Input Features** (`Hist_Carrier`, `Hist_Weather`, etc.).
    * **Benefit:** Gives the model a "Baseline Risk" to work from.
* **Label Creation:**
    * Converts the 5 cause columns into a single Integer Target (0-4) using `np.argmax`.
* **Output:** `data/Flights_classifier_clean.csv`

---

## 2 More datasets were prepared for future research.
 a model that predicts de total amount of delay of a flight (regression model), and a model that uses the the classifier and regression predictions together and predits the total amount of delay and the main cause for a complete report. (model report)

## 2. Regression Data (`model_regression_data.py`)
**Goal:** Create a dataset for time prediction.

* **Filter Logic:** None (Uses all flights).
* **Outlier Handling:**
    * **Target Clipping:** `ArrDelay` is clipped at **180 minutes**. This prevents the model from being confused by "Long Delay" events (e.g., 10-hour delays).
    * **Missing Values:** Filled with `0` (Assumes no delay if missing).
* **Feature Engineering:**
    * **Cyclic Time:** Converts `Month`, `Day`, `Hour` into Sin/Cos pairs.
    * **Scaling:** Pre-calculates scaled `f_dur` and `f_dist` columns.
* **Output:** `data/Flights_regression_clean.csv`

---

## 3. Report Data (`model_report_data.py`)
**Goal:** Build the prediction app.

* **ID Indexing:**
    * Scans the entire raw dataset to find all unique Airlines and Airports.
    * Assigns a unique Integer ID to each, sorted alphabetically.
    * **Output:** `data/Flights_report_ids.json` (Ensures `JFK` always equals ID `55`).
* **Historical data Lookup Table:**
    * Calculates the same risk profiles as the Classifier Data.
    * Saves them as a lightweight CSV lookup table.
    * **Output:** `data/Flights_report_clean.csv` (Used for O(1) lookups during prediction).