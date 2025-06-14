#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import os
import numpy as np
import pandas as pd
from keras.models import Model
from keras.layers import LSTM, Dense, Dropout, Input, Concatenate
from keras.optimizers import Adam
from keras.callbacks import EarlyStopping
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error

# Function to prepare dataset
def create_dataset(X, y, time_steps=1):
    Xs, ys = [], []
    for i in range(len(X) - time_steps):
        Xs.append(X[i:(i + time_steps)])
        ys.append(y[i + time_steps])
    return np.array(Xs), np.array(ys)

# Function to decide whether to retrain the model
def should_retrain(counter, interval=252):
    return counter % interval == 0

# Function to build the GARCH-LSTM hybrid model
def create_garch_lstm_model(input_shape_lstm, input_shape_garch):
    # Branch 1: LSTM for time-series features
    lstm_input = Input(shape=input_shape_lstm, name="LSTM_Input")
    lstm_layer = LSTM(128, return_sequences=True, activation='tanh')(lstm_input)
    lstm_layer = Dropout(0.1)(lstm_layer)
    lstm_layer = LSTM(128, activation='tanh')(lstm_layer)
    lstm_layer = Dropout(0.1)(lstm_layer)

    # Branch 2: Dense layers for GARCH input
    garch_input = Input(shape=input_shape_garch, name="GARCH_Input")
    garch_layer = Dense(32, activation='relu')(garch_input)
    garch_layer = Dense(16, activation='relu')(garch_layer)

    # Merge both branches
    merged = Concatenate()([lstm_layer, garch_layer])
    output = Dense(1, activation='relu')(merged)

    model = Model(inputs=[lstm_input, garch_input], outputs=output)
    return model

# Load data
gd_lstm = pd.read_excel('DATA-LSTM-GOLD.xlsx')
gd = pd.read_excel('GOLD-2024.xlsx')

# GARCH Processing
gd_garch = garch_data_prep(gd)
garch_results = train_garch_model(gd_garch, '2021-01-01')

# Merge GARCH predictions with LSTM dataset
gd_garch = pd.merge(gd_lstm, garch_results, on=['Date'], how='left')
gd_garch = gd_garch.rename(columns={'prediction': 'predicted_volatility_garch'})

df = gd_garch
df.to_excel('DATA-LSTM-GARCH-GOLD.xlsx')

# LSTM-GARCH hybrid modeling
feature_columns = [col for col in df.columns if col not in ['volatility', 'Date']]
target_column = 'volatility'

if 'predicted_volatility_garch' not in feature_columns:
    feature_columns.append('predicted_volatility_garch')

time_steps = 22
X, y = create_dataset(df[feature_columns], df[target_column].values.reshape(-1, 1), time_steps)

# Prepare input data
input_shape_lstm = (time_steps, X.shape[2] - 1)
input_shape_garch = (1,)

X_lstm = X[:, :, :-1]
X_garch = X[:, -1, -1].reshape(-1, 1)

model_save_path = 'lstm_garch.weights.h5'
early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
initial_train_size = int(724 * 0.80)
validation_size = int(724 * 0.20)

results = []
counter = 0

# Walk-forward prediction
for i in range(len(df) - initial_train_size - validation_size - 1):
    if (i + initial_train_size + validation_size + time_steps > len(X)):
        print("Not enough data to generate test sequence. Ending forecast.")
        break

    # Scaling data
    scaler_X = MinMaxScaler()
    scaler_y = MinMaxScaler()

    scaler_X.fit(X_lstm[i:i+initial_train_size].reshape(-1, X_lstm.shape[2]))
    scaler_y.fit(y[i:i+initial_train_size].reshape(-1, 1))

    train_X_lstm = scaler_X.transform(X_lstm[i:i+initial_train_size].reshape(-1, X_lstm.shape[2])).reshape(-1, time_steps, X_lstm.shape[2])
    train_y = scaler_y.transform(y[i:i+initial_train_size].reshape(-1, 1)).reshape(-1, 1)
    train_X_garch = X_garch[i:i+initial_train_size]

    val_X_lstm = scaler_X.transform(X_lstm[i+initial_train_size:i+initial_train_size+validation_size].reshape(-1, X_lstm.shape[2])).reshape(-1, time_steps, X_lstm.shape[2])
    val_y = scaler_y.transform(y[i+initial_train_size:i+initial_train_size+validation_size].reshape(-1, 1)).reshape(-1, 1)
    val_X_garch = X_garch[i+initial_train_size:i+initial_train_size+validation_size]

    test_X_lstm = scaler_X.transform(X_lstm[i+initial_train_size+validation_size:i+initial_train_size+validation_size+1].reshape(-1, X_lstm.shape[2])).reshape(-1, time_steps, X_lstm.shape[2])
    test_y = scaler_y.transform(y[i+initial_train_size+validation_size:i+initial_train_size+validation_size+1].reshape(-1, 1)).reshape(-1, 1)
    test_X_garch = X_garch[i+initial_train_size+validation_size:i+initial_train_size+validation_size+1]

    # Training or loading model
    if should_retrain(counter) or not os.path.exists(model_save_path):
        model = create_garch_lstm_model(input_shape_lstm, input_shape_garch)
        model.compile(optimizer=Adam(learning_rate=0.001), loss='mean_squared_error')
        model.fit([train_X_lstm, train_X_garch], train_y, epochs=100, batch_size=64, validation_data=([val_X_lstm, val_X_garch], val_y), verbose=0, callbacks=[early_stopping])
        model.save_weights(model_save_path)
    else:
        model = create_garch_lstm_model(input_shape_lstm, input_shape_garch)
        model.load_weights(model_save_path)
        model.compile(optimizer=Adam(learning_rate=0.001), loss='mean_squared_error')

    # Forecast
    predicted = model.predict([test_X_lstm, test_X_garch])
    predicted = scaler_y.inverse_transform(predicted.reshape(-1, 1))
    actual = scaler_y.inverse_transform(test_y.reshape(-1, 1))
    mae = mean_absolute_error(actual, predicted)

    # Save results
    current_result = {
        'train_start': df['Date'][i+time_steps],
        'train_end': df['Date'][i+initial_train_size+time_steps-1],
        'validation_start': df['Date'][i+initial_train_size+time_steps],
        'validation_end': df['Date'][i+initial_train_size+validation_size+time_steps-1],
        'test_date': df['Date'][i+initial_train_size+validation_size+time_steps],
        'prediction': predicted.flatten()[0],
        'actual': actual.flatten()[0],
        'mae': mae
    }
    print(current_result)

    results.append(current_result)
    counter += 1

# Export results to Excel
pd.DataFrame(results).to_excel('LSTM-GARCH-GOLD-RESULT.xlsx')

