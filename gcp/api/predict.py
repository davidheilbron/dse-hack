import requests
import pandas as pd
import json


# ### Load data
train_data = pd.read_csv('../data/train.csv')
y = train_data["Survived"]

features = ["Pclass", "Sex", "SibSp", "Parch"]
X = pd.get_dummies(train_data[features])

# ### Convert to json
#json = [X.iloc[i].to_dict() for i in range(X.shape[0])]

# Convert to JSON
data_json = X.iloc[0:10].to_json(orient='records')

# ### Predict

url = 'http://127.0.0.1:8000/predict/'
res = requests.post(url, json=json)
y_pred = res.json()
print(y_pred[0:10])

