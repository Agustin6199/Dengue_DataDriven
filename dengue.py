#!/usr/bin/python3

import pandas as pd
import numpy as np

from sklearn import preprocessing, MinMaxScaler
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dropout, Flatten, Dense
from tensorflow.keras import models

def upload_files (f):
  uploaded = pd.read_csv(f, sep=',', index_col=0)
  return uploaded

## Upload Files ##

train_feat = upload_files('dengue_features_train.csv')
train_labels = upload_files('dengue_labels_train.csv')
test = upload_files('dengue_features_test.csv')

train = pd.merge(train_feat,train_labels,on=['city', 'year', 'weekofyear'])

## Preprocessing ##

train.drop("week_start_date", axis = 1, inplace = True)
test.drop("week_start_date", axis = 1, inplace = True)

pd.isnull(train).any()
train.fillna(method='ffill', inplace=True)
pd.isnull(train).any()

test.fillna(method='ffill', inplace=True)
pd.isnull(test).any()

lb = preprocessing.LabelBinarizer()
train['city_bin'] = lb.fit_transform(train['city'])
test['city_bin'] = lb.fit_transform(test['city'])

selected_features = ['reanalysis_specific_humidity_g_per_kg', 'reanalysis_dew_point_temp_k', 'station_avg_temp_c', 'station_min_temp_c']

scaler = MinMaxScaler()
X_train = scaler.fit_transform(train[selected_features])
X_test = scaler.transform(test[selected_features])
y_train = train['total_cases']


## Model ##

model = models.Sequential()
model.add(Dense(34, activation='sigmoid', input_shape=[len(selected_features)]))
model.add(Dense(22, activation='sigmoid'))
model.add(Dense(16, activation='relu')) # rectified linear unit 
model.add(Dense(1, activation = "linear"))

model.summary()

model.compile(optimizer='adam',loss='mae',metrics=['mae'])
model.fit(X_train, y_train, epochs=100)

## Output File ##

y_pred = model.predict(X_test)
	
y = np.rint(y_pred)
y = y.astype(int)
res = np.hstack(y)

output = pd.DataFrame({ 'city': test['city'], 'year': test['year'], 'weekofyear': test['weekofyear'], 
                       'total_cases': res})

with open('result.csv', 'w') as f:
  output.to_csv(f,  index = False)
