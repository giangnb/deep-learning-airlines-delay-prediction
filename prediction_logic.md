# Prediction Logic
 
**File:** `model_classifier_pre.py`

This script is designed to use the "Classifier model", allowing users to see the raw probability distribution and compare it against historical baselines.

### **1. Input Retrieval**

* **Auto-Route:**
    * **Input:** Origin + Destination.
    * **Lookup:** Queries `Flights_routes.json` (created by `routs_map.py`).
    * **Action:** Automatically injects the **Median Distance** and **Median Duration** from the training set.
    * **Why:** Prevents "wrong inputs" (e.g., a user testing a 50-mile flight for JFK-LHR), ensuring the model is debugged on realistic data.

* **Use of historical data:**
    * **Lookup:** Queries `Flights_report_clean.csv`.
    * **Action:** Retrieves the 5 historical delay averages (Carrier, Weather, NAS, Security, Late Aircraft) for the specific `Origin_Month` key.
    * **Usage:** These are fed into the model as features *and* displayed in the output for side-by-side comparison.

### **2. Preprocessing & Scaling**
The inputs are processed to match the training environment of `model_classifier.keras`:

* **Numerical Vector (7 Features):**
    * `[Scaled_Duration, Scaled_Distance, Hist_Carrier, Hist_Weather, Hist_NAS, Hist_Security, Hist_LateAircraft]`
    * **Scaling:** `Duration / 400.0`, `Distance / 3000.0`.
* **Categorical IDs:**
    * Airline, Origin, and Dest are mapped to integers using `Flights_report_ids.json`.

### **3. Output Logic**
* **Raw Probabilities:** Returns the Softmax output (0.0 to 1.0) for all 5 classes.
* **Visual Comparison:** Renders a bar chart comparing the **AI's Confidence** (Future Prediction) vs. the **Historical Average** (Past Data).