#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import os
import numpy as np
import pandas as pd
from keras.models import Sequential
from keras.layers import LSTM, Dense, Dropout, Input
from keras.optimizers import Adam
from keras.callbacks import EarlyStopping
from sklearn.metrics import mean_absolute_error
from sklearn.preprocessing import MinMaxScaler

# Create dataset for LSTM
def create_dataset(X, y, time_steps=1):
    Xs, ys = [], []
    for i in range(len(X) - time_steps):
        Xs.append(X[i:(i + time_steps)])
        ys.append(y[i + time_steps])
    return np.array(Xs), np.array(ys)
def should_retrain(counter, interval=252):
    return counter % interval == 0

# Create model
def create_model(input_shape):
    model = Sequential([
        Input(shape=input_shape),
        LSTM(128, return_sequences=True, activation='tanh'),
        Dropout(0.1),
        LSTM(128, activation='tanh'),
        Dropout(0.1),
        Dense(1, activation='relu')
    ])
    return model  

# Download data
df = pd.read_excel('DATA-LSTM-GOLD.xlsx')
df['Date'] = pd.to_datetime(df['Date'])
df = df[df.Date >= '2021-01-01']

# Identify the features column and target variable
feature_columns = [col for col in df.columns if col not in ['volatility', 'Date']]
target_column = 'volatility'

# Timestep for LSTM model
time_steps = 22
X, y = create_dataset(df[feature_columns].values, df[target_column].values.reshape(-1, 1), time_steps)

# Training and testing in LSTM model
input_shape = (time_steps, X.shape[2])
model_save_path = 'lstm.weights.h5'
early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
initial_train_size = int(724* 0.80)
validation_size = int(724 * 0.20)


results = []
counter = 0

# Walk-forward prediction
for i in range(len(df) - initial_train_size - validation_size - 1):
    if (i + initial_train_size + validation_size + time_steps > len(X)):
        print("Không đủ dữ liệu để tạo chuỗi kiểm tra. Kết thúc dự báo.")
        break

    # Standardize the data
    scaler_X = MinMaxScaler()
    scaler_y = MinMaxScaler()

    # Fit scaler in training set
    scaler_X.fit(X[i:i+initial_train_size].reshape(-1, X.shape[2]))
    scaler_y.fit(y[i:i+initial_train_size].reshape(-1, 1))
    train_X = scaler_X.transform(X[i:i+initial_train_size].reshape(-1, X.shape[2])).reshape(-1, time_steps, X.shape[2])
    train_y = scaler_y.transform(y[i:i+initial_train_size].reshape(-1, 1)).reshape(-1, 1)
    val_X = scaler_X.transform(X[i+initial_train_size:i+initial_train_size+validation_size].reshape(-1, X.shape[2])).reshape(-1, time_steps, X.shape[2])
    val_y = scaler_y.transform(y[i+initial_train_size:i+initial_train_size+validation_size].reshape(-1, 1)).reshape(-1, 1)
    test_X = scaler_X.transform(X[i+initial_train_size+validation_size:i+initial_train_size+validation_size+1].reshape(-1, X.shape[2])).reshape(-1, time_steps, X.shape[2])
    test_y = scaler_y.transform(y[i+initial_train_size+validation_size:i+initial_train_size+validation_size+1].reshape(-1, 1)).reshape(-1, 1)

    # Training model
    if should_retrain(counter) or not os.path.exists(model_save_path):
        model = create_model(input_shape)
        model.compile(optimizer=Adam(learning_rate=0.001), loss='mean_squared_error') 
        model.fit(train_X, train_y, epochs=100, batch_size=64, validation_data=(val_X, val_y), verbose=0, callbacks=[early_stopping])
        model.save_weights(model_save_path)  
    else:
        model = create_model(input_shape)
        model.load_weights(model_save_path)  
        model.compile(optimizer=Adam(learning_rate=0.001), loss='mean_squared_error')  
        model.fit(train_X[-1].reshape(1, *train_X[-1].shape), train_y[-1].reshape(1, 1), epochs=1, verbose=0)  # Fine-tune

    # Prediction
    predicted = model.predict(test_X)
    predicted = scaler_y.inverse_transform(predicted.reshape(-1, 1))
    actual = scaler_y.inverse_transform(test_y.reshape(-1, 1))
    mae = mean_absolute_error(actual, predicted)

    # Save
    current_result = {
        'train_start': df['Date'].iloc[i+time_steps],
        'train_end': df['Date'].iloc[i+initial_train_size+time_steps-1],
        'validation_start': df['Date'].iloc[i+initial_train_size+time_steps],
        'validation_end': df['Date'].iloc[i+initial_train_size+validation_size+time_steps-1],
        'test_date': df['Date'].iloc[i+initial_train_size+validation_size+time_steps],
        'prediction': predicted.flatten()[0],
        'actual': actual.flatten()[0],
        'mae': mae
    }
    print(current_result)

    results.append(current_result)
    counter += 1

# Export Excel
lstm_results = pd.DataFrame(results)
lstm_results.to_excel('LSTM-GOLD-RESULT.xlsx', index=False)

