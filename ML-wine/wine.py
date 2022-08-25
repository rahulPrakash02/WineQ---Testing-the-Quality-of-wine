import pandas as pd
import numpy as np
import pickle
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

df = pd.read_csv('wineQual.csv')

x = np.array(df[['alcohol','fixed acidity', 'residual sugar', 'pH', 'free sulfur dioxide']])
y = np.array(df['quality'])

scaler = StandardScaler()
scaler.fit(x)
x_std = scaler.transform(x)
x = x_std

x_train, x_test, y_train, y_test = train_test_split(x,y,test_size=0.2, random_state=123)
rfc = RandomForestClassifier()
rfc.fit(x_train, y_train)
rfc_pred = rfc.predict(x_test)

pickle.dump(rfc, open('wine_mod.pkl', 'wb'))

