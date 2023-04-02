from pickle import dump

import numpy as np
import pandas as pd
from scipy.stats import ttest_rel
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

import config

data = pd.read_csv(config.output_csv)

X = data[['h_px']].values.reshape(-1, 1)
Y = data['cm'].values.reshape(-1, 1)

x_train, x_test, y_train, y_test = train_test_split(
    X, Y, test_size=0.5, random_state=42)

reg = LinearRegression().fit(x_train, y_train)
with open(config.model_file, 'wb') as f:
    dump(reg, f)

print(f'     r2: {reg.score(x_test, y_test)}')
print(f'      m: {reg.coef_[0]}')
print(f'      b: {reg.intercept_}')

print(ttest_rel(reg.predict(x_test), y_test))
