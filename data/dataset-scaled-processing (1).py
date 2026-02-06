def split_and_prepare_data(df, test_size=0.15):
    """
    Split 8.5M samples into Train/Test while preventing data leakage
    and formatting for Multi-Task Learning.
    """
    # 1. Load Data
    
    # 2. Prepare Multi-Task Targets
    delay_cols = ["Carrier Delay", "Weather Delay", "Airport Delay", "Security Delay", "Late Aircraft Delay"]
    df['is_delayed'] = (df[delay_cols].sum(axis=1) > 0).astype(int)
    df['primary_reason'] = np.argmax(df[delay_cols].values, axis=1)
    
    # Log transform targets for the regression head
    for col in delay_cols:
        df[f'{col}_log'] = np.log1p(df[col])

    # 3. Stratified Split (Maintain delay/no-delay ratio)
    train_df, test_df = train_test_split(
        df, 
        test_size=test_size, 
        random_state=42, 
        stratify=df['is_delayed']
    )

    # 4. Target Encoding (Signal Booster) - TRAIN DATA ONLY
    # Calculate historical average delay per airport ONLY from train set
    origin_means = train_df.groupby('Origin Airport Code')['is_delayed'].mean()
    dest_means = train_df.groupby('Destination Airport Code')['is_delayed'].mean()
    global_mean = train_df['is_delayed'].mean()

    # Map averages to both sets (filling unseen airports with global mean)
    for d in [train_df, test_df]:
        d['origin_avg_delay'] = d['Origin Airport Code'].map(origin_means).fillna(global_mean)
        d['dest_avg_delay'] = d['Destination Airport Code'].map(dest_means).fillna(global_mean)

    # 5. Cyclical Encoding (Hours, Days, Months)
    def encode_cyclical(df_in):
        df_in['hr_sin'] = np.sin(2 * np.pi * df_in['Departure Block Hour'] / 24)
        df_in['hr_cos'] = np.cos(2 * np.pi * df_in['Departure Block Hour'] / 24)
        df_in['dow_sin'] = np.sin(2 * np.pi * df_in['Day Of Week'] / 7)
        df_in['dow_cos'] = np.cos(2 * np.pi * df_in['Day Of Week'] / 7)
        df_in['month_sin'] = np.sin(2 * np.pi * df_in['Month'] / 12)
        df_in['month_cos'] = np.cos(2 * np.pi * df_in['Month'] / 12)
        return df_in

    train_df = encode_cyclical(train_df)
    test_df = encode_cyclical(test_df)

    # 6. Feature Selection & Scaling
    cat_cols = ['Origin Airport Code', 'Destination Airport Code']
    num_cols = [
        'Fly Time Scheduled', 'Distance Miles', 'Distance Group',
        'hr_sin', 'hr_cos', 'dow_sin', 'dow_cos', 'month_sin', 'month_cos',
        'origin_avg_delay', 'dest_avg_delay'
    ]

    scaler = StandardScaler()
    train_df[num_cols] = scaler.fit_transform(train_df[num_cols])
    test_df[num_cols] = scaler.transform(test_df[num_cols])

    # 7. Final Keras Formatting
    def format_for_keras(df_in):
        # Input 1: Categorical (Airports)
        # Input 2: Numerical (Distances, Cycles, Historical Means)
        X = [df_in[cat_cols].values, df_in[num_cols].values]
        
        # Targets: Mapping to the named output layers in our Functional Model
        Y = {
            "out_binary": df_in['is_delayed'].values,
            "out_reason": df_in['primary_reason'].values,
            "out_regression": df_in[[c + "_log" for c in delay_cols]].values
        }
        return X, Y

    X_train, Y_train = format_for_keras(train_df)
    X_test, Y_test = format_for_keras(test_df)

    print(f"Dataset Split Complete. Train: {len(train_df)}, Test: {len(test_df)}")
    return X_train[0], X_train[1], Y_train, X_test[0], X_test[1], Y_test, df
