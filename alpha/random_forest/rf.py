import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import matplotlib.pyplot as plt

data = pd.read_csv("spam.csv", encoding='latin-1')


print(data.head())
print(data.info())  
print(data.describe())
print(data.shape)
print(data.isnull().sum())


# x = data['v2']
# y = data['v1']
# x_train, x_test, y_train, y_test = train_test_split(x,y, test_size=0.2, random_state=42)
# from sklearn.feature_extraction.text import CountVectorizer

# vectorizer = CountVectorizer()

# x_train = vectorizer.fit_transform(x_train)
# x_test = vectorizer.transform(x_test)
# model = RandomForestClassifier(n_estimators=100, random_state=42)
# model.fit(x_train, y_train)

# accuracy = model.score(x_test, y_test)
# print("Accuracy:", accuracy)
# y_pred = model.predict(x_test)

# print("MAE :", mean_absolute_error(y_test, y_pred))
# print("MSE :", mean_squared_error(y_test, y_pred))
# print("R2 Score :", r2_score(y_test, y_pred))
# print("accuracy:", accuracy(y_test, y_pred))

# plt.scatter(y_test, y_pred)
# plt.xlabel("Actual Labels")
# plt.ylabel("Predicted Labels")
# plt.title("Actual vs Predicted Labels")
# plt.show()