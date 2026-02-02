# 🏗️ Project Architecture & Logic Breakdown

This project utilizes a **classifier model** to predict the main reason for delay in a flight.


## 1. The Classifier Model (`model_classifier.py`)
* **Goal:** Predict the specific reason for the delay (Carrier, Weather, Traffic, Security, Late Aircraft).
* **Why:** Causes are categorical and highly imbalanced (e.g., Weather is rare).
* **Key Logic:**
    * Trains **ONLY on Late Flights** (>= 15 minutes).
    * Uses **Class Weights** to penalize the model heavily if it misses rare events like Weather.
    * Incorporates **Historical Risk Profiles** (Prior Knowledge) as inputs to improve accuracy.


---
**Why this Architecture?**
Early experiments attempting to predict the delay of the 5 causes for delay (Carrier, Weather, Traffic, Security, Late Aircraft). We couldn't predict the 5 outputs because 90% of the data is zeros, which causes regression models to predict near-zero for everything. Switching to classification allowed us to focus on Pattern Recognition (The Cause) rather than Quantity Estimation.