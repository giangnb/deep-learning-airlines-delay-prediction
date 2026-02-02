# 📊 Model Training Results

This document contains the performance metrics and configuration details.

---

## 2. Classifier Model 
**Goal:** Correctly categorize the cause of the delay (Carrier, Weather, NAS, Security, Late Aircraft).

| Metric | Value |
| :--- | :--- |
| **Final Validation Accuracy** | **42.01%** |
| **Final Validation Loss** | 1.0672 |
| **Training Time** | 253.28 seconds |
| **Epochs Trained** | 7 |
| **Batch Size** | 4096 |

### **Class Weighting Strategy**
Because delays like "Weather" are rare compared to "Late Aircraft", we applied the following weights to force the model to pay attention to minority classes:

```json
{
    "0 (Carrier)": 0.47,
    "1 (Weather)": 5.81,  // High weight due to rarity
    "2 (NAS/Traffic)": 71.17, // Extremely high weight (Very rare in this subset)
    "3 (Late Aircraft)": 0.59
}