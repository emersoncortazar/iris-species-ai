import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix

# read data csv
df = pd.read_csv('data/Iris.csv')

# create features / predictors and outcome variables
features = ['SepalLengthCm', 'SepalWidthCm', "PetalLengthCm", 'PetalWidthCm']
X = df.loc[:, features]
y = df.loc[:, ['Species']]

# train test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size = 0.5, random_state=42
)

# create logistic_model variable and complete logistic regression
logistic_model  = LogisticRegression(random_state = 0, solver = 'liblinear')
logistic_model.fit(X_train, y_train)

# predict outcomes in the testing set
y_pred = logistic_model.predict(X_test)

# calculate and print accuracy
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy}")

# generate confusion matrix
cm = confusion_matrix(y_test, y_pred)
print("Confusion Matrix:")
print(cm)





