## 2. Classifier Model (`model_classifier.keras`)
**Type:** Embedding + CNN (1D) with Batch Normalization
**Objective:** Predict the root cause of delay (5 Classes).

### **Architecture**
Treats different types of data differently before merging them:
* **Road A (Categorical):** Entity Embeddings for IDs.
* **Road B (Numerical):** Physics + Historical Risk Profiles.
* Physics: physical constraints of the specific flight (Distance & Duration)
* Risk Profiles: statistical averages (average delay minutes for each cause)
### **Visual Input Map**

| CSV Column / Feature | Data Type | Input Branch | Notes |
| :--- | :--- | :--- | :--- |
| **Airline Code** | Integer (ID) | `Input_Cat` | Mapped to Embedding (Dim 5) |
| **Origin Airport Code** | Integer (ID) | `Input_Cat` | Mapped to Embedding (Dim 10) |
| **Dest Airport Code** | Integer (ID) | `Input_Cat` | Mapped to Embedding (Dim 10) |
| **Duration (Scaled)** | Float (0-1) | `Input_Num` | Physics |
| **Distance (Scaled)** | Float (0-1) | `Input_Num` | Physics |
| **Hist_Carrier** | Float (Avg) | `Input_Num` | Historical Knowledge |
| **Hist_Weather** | Float (Avg) | `Input_Num` | Historical Knowledge |
| **Hist_NAS** | Float (Avg) | `Input_Num` | Historical Knowledge |
| **Hist_Security** | Float (Avg) | `Input_Num` | Historical Knowledge |
| **Hist_LateAircraft** | Float (Avg) | `Input_Num` | Historical Knowledge |

### **Model Architecture**

| Layer Name | Type | Key Configuration (Settings) | Activation |
| :--- | :--- | :--- | :--- |
| **Input_Cat** | Input | `shape=(5, 1)` (x3 Inputs) | N/A |
| **Input_Num** | Input | `shape=(5, 7)` (7 Features) | N/A |
| **Embedding** | Embedding | `output_dim=10` (Airports) or `5` (Airline) | None |
| **Reshape** | Reshape | Target shape `(5, dim)` | N/A |
| **Merge** | Concatenate | `axis=-1` | N/A |
| **Batch_Norm** | **BatchNormalization** | **Critical:** Normalizes mixed inputs | N/A |
| **Conv1D** | Conv1D | `filters=64`, `kernel_size=2` | ReLU |
| **MaxPooling** | MaxPooling1D | `pool_size=2` | N/A |
| **Flatten** | Flatten | N/A | N/A |
| **Dense** | Dense | `units=64` | ReLU |
| **Dropout** | Dropout | `rate=0.3` | N/A |
| **Output** | Dense | `units=5` (One per Cause) | **Softmax** |

### **Compiler Configuration**
* **Optimizer:** Adam (Default)
* **Loss:** `sparse_categorical_crossentropy` (For Multi-Class Classification)
* **Metrics:** `accuracy`
* **Class Weights:** Balanced (Computed dynamically)