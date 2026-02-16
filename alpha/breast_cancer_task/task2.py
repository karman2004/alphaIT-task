import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
import matplotlib.pyplot as plt
import seaborn as sns

data = pd.read_csv("breast-cancer.csv")
data['diagnosis'] = data['diagnosis'].map({'M': 1, 'B': 0})
data['diagnosis'] = data['diagnosis'].astype(int)
x = data.drop(columns=['id', 'diagnosis'])
y = data['diagnosis']
x_train, x_test, y_train, y_test = train_test_split(x, y , test_size=0.2, random_state=42)

lg = LogisticRegression()
dt = DecisionTreeClassifier()
rf = RandomForestClassifier(n_estimators=100, random_state=42)
knn = KNeighborsClassifier()


lg.fit(x_train, y_train)
dt.fit(x_train, y_train)
rf.fit(x_train, y_train)
knn.fit(x_train, y_train)

y_pred_lg=lg.predict(x_test)
y_pred_dt=dt.predict(x_test)
y_pred_rf=rf.predict(x_test)
y_pred_knn=knn.predict(x_test)

lg_acc=accuracy_score(y_test, y_pred_lg)
dt_acc=accuracy_score(y_test, y_pred_dt)
rf_acc=accuracy_score(y_test, y_pred_rf)
knn_acc=accuracy_score(y_test, y_pred_knn)

lg_cr=classification_report(y_test, y_pred_lg)
dt_cr=classification_report(y_test, y_pred_dt)
rf_cr=classification_report(y_test, y_pred_rf)
knn_cr=classification_report(y_test, y_pred_knn)

lg_cm=confusion_matrix(y_test, y_pred_lg)
dt_cm=confusion_matrix(y_test, y_pred_dt)
rf_cm=confusion_matrix(y_test, y_pred_rf)
knn_cm=confusion_matrix(y_test, y_pred_knn)

print("Logistic Regression Accuracy:", lg_acc)
print("Logistic Regression classification_report", lg_cr)
print("Logistic Regression confusion_matrix ",lg_cm )

print("Decision Tree Accuracy:", dt_acc)
print("Decision Tree classification_report:", dt_cr)
print("Decision Tree confusion_matrix:", dt_cm)

print("Random Forest Accuracy:", rf_acc)
print("Random Forest classification_report:", rf_cr)
print("Random Forest confusion_matrix:", rf_cm)

print("KNeighbors Accuracy:", knn_acc)
print("KNeighbors classification_report:", knn_cr)
print("KNeighbors confusion_matrix:", knn_cm)

models = ['Logistic Regression', 'Decision Tree', 'Random Forest', 'KNeighbors']
accuracy = [lg_acc, dt_acc, rf_acc, knn_acc]

plt.figure(figsize=(10, 6))
sns.barplot(x=models, y=accuracy)
plt.title('Model Accuracy Comparison')
plt.xlabel('Models')
plt.ylabel('Accuracy')
plt.ylim(0, 1)
plt.show()