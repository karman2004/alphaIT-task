import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split, cross_validate
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, make_scorer

from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import BaggingClassifier, RandomForestClassifier
from sklearn.svm import SVC
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB

data = pd.read_csv("data.csv")
data = data.drop_duplicates()

print(f"After Removing Duplicates:", data.shape)

data["Machine_Failure"] = (
    data["TWF"] + data["HDF"] + data["PWF"] + data["OSF"] + data["RNF"]
)

data["Machine_Failure"] = data["Machine_Failure"].apply(lambda x: 1 if x > 0 else 0)


data = data.drop(columns=[
    "UDI",
    "Product ID",
    "TWF",
    "HDF",
    "PWF",
    "OSF",
    "RNF"
], errors="ignore")

le = LabelEncoder()
data["Type"] = le.fit_transform(data["Type"])


numeric_cols = data.select_dtypes(include=[np.number]).columns
data[numeric_cols] = data[numeric_cols].fillna(data[numeric_cols].median())


X = data.drop(columns=['Machine_Failure'])
y = data['Machine_Failure']

# Keep only numeric columns
X = X.select_dtypes(include=[np.number])

models = {
    "KNN": Pipeline([
        ("scaler", StandardScaler()),
        ("model", KNeighborsClassifier())
    ]),
    "Logistic Regression": Pipeline([
        ("scaler", StandardScaler()),
        ("model", LogisticRegression(max_iter=1000, class_weight='balanced'))
    ]),
    "Decision Tree": DecisionTreeClassifier(class_weight='balanced'),
    "Bagging": BaggingClassifier(),
    "Random Forest": RandomForestClassifier(class_weight='balanced', random_state=42),
    "SVM Linear": Pipeline([
        ("scaler", StandardScaler()),
        ("model", SVC(kernel="linear", class_weight='balanced'))
    ]),
    "SVM RBF": Pipeline([
        ("scaler", StandardScaler()),
        ("model", SVC(kernel="rbf", class_weight='balanced'))
    ]),
    "LDA": Pipeline([
        ("scaler", StandardScaler()),
        ("model", LinearDiscriminantAnalysis())
    ]),
    "Naive Bayes": GaussianNB()
}

results = []

scoring = {
    'accuracy': make_scorer(accuracy_score),
    'precision': make_scorer(precision_score, zero_division=0),
    'recall': make_scorer(recall_score, zero_division=0),
    'f1': make_scorer(f1_score, zero_division=0)
}

for name, model in models.items():
    cv_results = cross_validate(model, X, y, cv=5, scoring=scoring)

    results.append({
        "Model": name,
        "Accuracy": cv_results['test_accuracy'].mean(),
        "Precision": cv_results['test_precision'].mean(),
        "Recall": cv_results['test_recall'].mean(),
        "F1 Score": cv_results['test_f1'].mean()
    })

results_df = pd.DataFrame(results)


best_model = results_df.loc[results_df["F1 Score"].idxmax()]

print("\nModel Comparison:\n")
print(results_df)

print("\nBest Model Based on F1 Score:\n")
print(best_model)

plt.figure(figsize=(10,5))
plt.bar(results_df["Model"], results_df["Accuracy"])
plt.xticks(rotation=75)
plt.title("Model Accuracy Comparison")
plt.xlabel("Models")
plt.ylabel("Accuracy")
plt.tight_layout()
plt.show()
