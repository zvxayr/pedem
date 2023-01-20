from pickle import dump

import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

import config

data = pd.read_csv(config.output_csv)

px = data['px'].values.reshape(-1, 1)
cm = data['cm'].values

x_train, x_test, y_train, y_test = train_test_split(
    px, cm, test_size=0.2, random_state=21)

reg = LinearRegression().fit(x_train, y_train)
with open(config.model_file, 'wb') as f:
    dump(reg, f)

print(f'     r2: {reg.score(x_test, y_test)}')
print(f'      m: {reg.coef_[0]}')
print(f'      b: {reg.intercept_}')
print()
print(f'f(100) = {reg.predict([[100]])[0]}')
