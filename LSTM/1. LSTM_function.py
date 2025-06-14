#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import pandas as pd
import numpy as np
gd = pd.read_excel('GOLD-2024.xlsx')
rolling_window_size = 22
# Calculate daily log returns
gd['log_returns'] = np.log(gd['Price'] / gd['Price'].shift(1))
# Calculate volatility as the std of daily log returns
gd['volatility'] = gd['log_returns'].rolling(window=rolling_window_size).std()
# Add lagged volatility
lag_days = 1
for i in range(1, lag_days + 1):
    gd[f'lagged_volatility_{i}'] = gd['volatility'].shift(i)
gd.dropna(inplace=True)
# Clean dataset
gd.Date = pd.to_datetime(gd.Date)
gd.dropna(inplace = True)
gd.drop(columns = ['Price'], inplace = True)
# Filter for dates >= 2021 year
gd_2021 = gd[gd.Date >= '2021-01-01']
# Save
gd_2021.to_excel('DATA-LSTM-GOLD.xlsx')

