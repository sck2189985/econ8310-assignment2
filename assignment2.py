import pandas as pd
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

training = pd.read_csv("https://github.com/dustywhite7/Econ8310/raw/master/AssignmentData/assignment3.csv")

training['DateTime'] = pd.to_datetime(training['DateTime'])
training['hour'] = training['DateTime'].dt.hour

x = training.drop(['meal', 'DateTime', 'id'], axis=1)

y = training['meal']

model = XGBClassifier(n_estimators=200, max_depth=3, learning_rate=0.5, objective='binary:logistic')

modelFit = model.fit(x, y)

test = pd.read_csv("https://github.com/dustywhite7/Econ8310/raw/master/AssignmentData/assignment3test.csv")

test['DateTime'] = pd.to_datetime(test['DateTime'])
test['hour'] = test['DateTime'].dt.hour

yt = test['meal']

xt = test.drop(['meal', 'DateTime', 'id'], axis=1)

pred = model.predict(xt)

pred.astype(float)

print("test")