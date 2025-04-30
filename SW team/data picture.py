import numpy as np
import pandas as pd

csv_data = pd.read_csv('dataset_2021.csv')
print(csv_data.columns)
df_data = pd.DataFrame(csv_data)
print(df_data.columns)

print('---------------------------------------------------')
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.model_selection import train_test_split, ParameterGrid
from sklearn.ensemble import RandomForestRegressor, IsolationForest
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, silhouette_score
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA

from statsmodels.tsa.statespace.sarimax import SARIMAX
import itertools

# 0. Load data
df = pd.read_csv('dataset_2021.csv', parse_dates=['created_at'])
df.set_index('created_at', inplace=True)
variables = ['temperature','turbidity','disolved_oxg','ph','ammonia','nitrate']
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)
print('EDA 시작')

# 1. EDA
for var in variables:
    daily = df[var].resample('D').mean()
    plt.figure()
    plt.plot(daily.index, daily.values)
    plt.title(f'Daily Average of {var}')
    plt.xlabel('Date'); plt.ylabel(var)
    plt.show()

corr = df[variables + ['fish_length','fish_weight']].corr()
plt.figure(figsize=(8,6))
plt.imshow(corr, cmap='coolwarm', vmin=-1, vmax=1)
plt.colorbar()
plt.xticks(range(len(corr)), corr.columns, rotation=90)
plt.yticks(range(len(corr)), corr.columns)
plt.title('Correlation Matrix')
plt.show()

plt.figure(figsize=(10,3))
plt.imshow(df.isnull(), aspect='auto', cmap='gray_r')
plt.title('Missing Value Map'); plt.show()

df[variables].boxplot(figsize=(8,4))
plt.title('Boxplot of Water Quality Variables'); plt.show()


# 2. Feature engineering
print('Feature engineering 시작')
df_fe = df.copy()
df_fe['hour']      = df_fe.index.hour
df_fe['dayofweek'] = df_fe.index.dayofweek
df_fe['month']     = df_fe.index.month

for var in variables:
    df_fe[f'{var}_rm3']  = df_fe[var].rolling(window=3).mean()
    df_fe[f'{var}_std3'] = df_fe[var].rolling(window=3).std()
    df_fe[f'{var}_delta'] = df_fe[var].diff()

df_fe.dropna(inplace=True)
