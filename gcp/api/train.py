#!/usr/bin/env python
# coding: utf-8

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import os
import pickle
import sklearn

# ### Load the data
train_data = pd.read_csv('../data/train.csv')

from sklearn.ensemble import RandomForestClassifier

y = train_data["Survived"]

features = ["Pclass", "Sex", "SibSp", "Parch"]
X = pd.get_dummies(train_data[features])

model = RandomForestClassifier(n_estimators=100, max_depth=5, random_state=1)
model.fit(X, y)


pickle.dump(model, open('model.pkl', 'wb'))

