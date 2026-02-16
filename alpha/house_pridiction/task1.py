
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.preprocessing import StandardScaler


data = pd.read_csv("Housing.csv")

binary_cols = ['mainroad', 'guestroom', 'basement',
               'hotwaterheating', 'airconditioning', 'prefarea']

for col in binary_cols:
    data[col] = data[col].map({'yes': 1, 'no': 0})

data['furnishingstatus'] = data['furnishingstatus'].map({
    'unfurnished': 0,
    'semi-furnished': 1,
    'furnished': 2
})

X = data.drop('price', axis=1)
y = data['price']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

model = LinearRegression()

model.fit(X_train_scaled, y_train)
y_pred = model.predict(X_test_scaled)

# Evaluation
print("Mean Absolute Error:", mean_absolute_error(y_test, y_pred))
print("R2 Score:", r2_score(y_test, y_pred))

# BAR GRAPH
plt.figure()
index = np.arange(10)
plt.bar(index, y_test.values[:10], label="Actual Price", color="blue")
plt.bar(index, y_pred[:10], label="Predicted Price", color="orange", alpha=0.7)
plt.xlabel("House Index")
plt.ylabel("Price")
plt.title("Actual vs Predicted House Prices")
plt.legend()
plt.show()
