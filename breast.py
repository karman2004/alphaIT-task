import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import BaggingClassifier, RandomForestClassifier
from sklearn.svm import SVC
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB

data = load_breast_cancer()
X = data.data
y = data.target

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)


models = {
    "KNN": KNeighborsClassifier(),

    "Logistic Regression": LogisticRegression(max_iter=2000),

    "Logistic Regression L1": LogisticRegression(
        penalty="elasticnet",
        solver="saga",
        l1_ratio=1,
        max_iter=5000
    ),

    "Logistic Regression L2": LogisticRegression(
        penalty="elasticnet",
        solver="saga",
        l1_ratio=0,
        max_iter=5000
    ),

    "Decision Tree": DecisionTreeClassifier(),
    "Bagging": BaggingClassifier(),
    "Random Forest": RandomForestClassifier(),
    "SVM Linear": SVC(kernel="linear"),
    "SVM RBF": SVC(kernel="rbf"),
    "LDA": LinearDiscriminantAnalysis(),
    "Naive Bayes": GaussianNB()
}

results = []

for name, model in models.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    results.append({
        "Model": name,
        "Accuracy": accuracy_score(y_test, y_pred),
        "Precision": precision_score(y_test, y_pred),
        "Recall": recall_score(y_test, y_pred),
        "F1 Score": f1_score(y_test, y_pred)
    })

results_df = pd.DataFrame(results)

# Find Best Model (Based on F1 Score)
best_model = results_df.loc[results_df["F1 Score"].idxmax()]

print(f"\nModel Comparison:\n")
print(results_df)
print(f"\nBest Model Based on F1 Score:\n")
print(best_model)

# Bar Chart (Accuracy Comparison)
plt.figure()
plt.bar(results_df["Model"], results_df["Accuracy"])
plt.xticks(rotation=75)
plt.title("Model Accuracy Comparison")
plt.xlabel("Models")
plt.ylabel("Accuracy")
plt.tight_layout()
plt.show()
