#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import pandas as pd
import numpy as np
from arch import arch_model
# Download the dataset
df = pd.read_excel('GOLD-2024.xlsx')
gd = garch_data_prep(df)
predicted_volatility = train_garch_model(gd, '2021-01-01')
# Prepare actual and predicted volatility 
actual_volatility = gd[['Date', 'volatility']].set_index('Date')
actual_volatility = actual_volatility.rename(columns={'volatility': 'actual'})
# Merge actual and predicted data
merged_df = gd.merge(actual_volatility, on='Date', how='outer').merge(predicted_volatility, on='Date', how='outer')
merged_df.dropna(inplace = True)
garch_results = merged_df.drop(columns = ['scaled_log_returns', 'volatility'])

print(garch_results)
# Save results
garch_results.to_excel('GARCH-GOLD-RESULT.xlsx')

