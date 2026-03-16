import pandas as pd
from xgboost import XGBClassifier


training = pd.read_csv("https://github.com/dustywhite7/Econ8310/raw/master/AssignmentData/assignment3.csv")

x = training.drop(columns=["meal"])
y = training["meal"]

model = XGBClassifier(n_estimators=300, max_depth=4, learning_rate=0.05, random_state=42, use_label_encoder=False, eval_metric="logloss")

modelFit = model.fit(x, y)

testing = pd.read_csv("https://github.com/dustywhite7/Econ8310/raw/master/AssignmentData/assignment3test.csv")

pred = modelFit.predict(testing)
pred = pred.astype(int)

