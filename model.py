#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Sep 12 13:33:20 2020

@author: jagriti
"""

import pickle
import pandas as pd
dataset = pd.read_csv("/Users/jagriti/downloads/parkinsons.csv")

print(dataset.head())
print(dataset.shape)
print(dataset.dtypes)
print(dataset.describe())

dataset.isnull().sum()

import seaborn as sns
corr_map = dataset.corr()
sns.heatmap(corr_map,square=True)

correlation_values=dataset.corr()['status']
correlation_values.abs().sort_values(ascending=False)

sns.catplot(x='status',kind='count',data=dataset)

for i in dataset:
    if i != 'status' and i != 'name':
        sns.catplot(x='status',y=i,kind='box',data=dataset)
        

from xgboost import XGBClassifier
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

features=dataset.drop(['status','name'],axis=1)
labels=dataset['status']

scaler=MinMaxScaler((-1,1))
x=scaler.fit_transform(features)
y=labels

x_train,x_test,y_train,y_test=train_test_split(x, y, test_size=0.2, random_state=5)

modelx = XGBClassifier()
modelx.fit(x_train, y_train)

pickle.dump(modelx, open('model.pkl' , "wb"))

y_predtr=modelx.predict(x_train)
print(accuracy_score(y_train,y_predtr)*100)

y_pred=modelx.predict(x_test)
print(accuracy_score(y_test, y_pred)*100)

from sklearn.metrics import confusion_matrix
print(confusion_matrix(y_test,y_pred))
        
