## Approach preprocessing and feature selection.
-To build a successful CNN regression model that predicts the Minutes of Delay.

1. Feature Selection:
    Separate the raw data into imputs(x), Targets (y), and Leakage (Drop)

    Target(y): Arrival Delay: what we are trying to predict.

    Input(x): Columns for training the model: Departure Block Hour, Day Of Week, Flight Date, Origin Airport Code, Destination Airport Code, Airline Code, Distance Miles, Fly Time Schedule.

    Leakage(Drop): Columns to remove: Carrier Delay, Weather Delay, Airport Delay, Security Delay, Late Aircraft Delay, Delay main reason, Touchdown Time, Take Off Time, Arrival Time Actual, Departure Time Actual, Taxi Out, Taxi In, Air Time, Actual Elapsed Time, Origin Airport ID, Origin City Name, Distance Group (since we have Distance Miles).

2. Encoding Strategy:
    
    Strategy A: Numerical Data (Scaling)
    Features: Distance Miles, Fly Time Scheduled.

    Method: Min-Max Scaling or Standardization (Z-Score).

    Why: Neural networks struggle if one input is "0.5" and another is "3000". Scale everything to be roughly between 0 and 1.


    Strategy B: Cyclical Data (Time)
    Features: Departure Block Hour, Day Of Week, Month (from Date).Method: Sin/Cos Transformation.
    
    Why: If you just label hours 0-23, the model thinks Hour 23 and Hour 0 are far apart. In reality, 11 PM and Midnight are close.
    
    Strategy C: Categorical Data (The CNN approach)
    Features: Origin Airport Code, Destination Airport Code, Airline Code.
    
    Method: Label Encoding + Embedding Layers.
    
    Why: "One-Hot Encoding" creates too many columns (there are 300+ airports).

    The Deep Learning Way:

        Label Encode: Turn "JFK" -> 102, "LGA" -> 103.

        Embedding Layer: Inside the Keras/TensorFlow model, add an Embedding layer. This allows the model to learn a vector representation for each airport (e.g., it will learn that SFO and LAX are similar because they are both West Coast hubs).
    
3. Summary of Model Inputs:

    When building the CNN, your inputs will be these processed columns.

    If using 1D-CNN (Sequence): You will group these rows into windows (e.g., 5 rows per sample).

    Handling Embeddings in CNN: You will need to use the Keras Functional API (not just Sequential) because you need to separate the categorical inputs (for Embeddings) from the numerical inputs (which go straight in).