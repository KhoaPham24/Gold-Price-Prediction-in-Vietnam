#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import statsmodels.api as sm
from statsmodels.tsa.stattools import adfuller
from statsmodels.stats.diagnostic import het_arch
from tabulate import tabulate  

# STATIONARY TEST
## ADF test
df = pd.read_excel('GOLD-2024.xlsx')
gd = garch_data_prep(df)
result = adfuller(gd["log_returns"].dropna())

## Result
print("Dickey-Fuller Test of log_returns:")
print(f"ADF Statistic: {result[0]}")
print(f"P-value: {result[1]:.4f}")
print("Critical Values:")
for key, value in result[4].items():
    print(f"\t{key}: {value}")

print("With p-value = 0.000, we can conclude that it's stationary at log returns")
# ACF and PACF plot
## Create figure with ACF & PACF
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

## Draw ACF
sm.graphics.tsa.plot_acf(gd["log_returns"].dropna(), lags=30, ax=axes[0])
axes[0].set_title("Autocorrelation Function (ACF)")

## Draw PACF
sm.graphics.tsa.plot_pacf(gd["log_returns"].dropna(), lags=30, ax=axes[1])
axes[1].set_title("Partial Autocorrelation Function (PACF)")
plt.tight_layout()
plt.show()
print('From the result of 2 pictures, we choose ARIMA(1,0,1) or ARMA(1,1) for log_returns')


# TAKE THE RESIDUALS OF ARIMA(1,0,1)
## Estimate the ARIMA(1,0,1) model
model = sm.tsa.ARIMA(gd["log_returns"], order=(1, 0, 1))
results = model.fit()

## Create the Residuals
re = results.resid


# TEST FOR ARCH EFFECT
## Create list of result
lag_results = []

## Use lag 1 to 10
for lag in range(1, 11):
    arch_test = het_arch(re, maxlag=lag)  # ARCH test with the detailed lags
    p_value_formatted = "{:.4f}".format(arch_test[1]) if arch_test[1] >= 0.0001 else "0.0000"
    lag_results.append([lag, round(arch_test[0], 3), lag, p_value_formatted])


results_df = pd.DataFrame(lag_results, columns=["Lags (p)", "Chi2", "df", "Prob > Chi2"])
print("\nLM test for autoregressive conditional heteroskedasticity (ARCH effect)\n")
print(tabulate(results_df.values.tolist(), headers=results_df.columns, tablefmt="grid"))
print("\nH0: No ARCH effects   vs.   H1: ARCH(p) disturbance")
print('The result demonstrates that we have ARCH effect, so we can use ARCH/GARCH model in time series')

