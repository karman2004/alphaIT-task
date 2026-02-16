
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier,plot_tree
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression

data = pd.read_csv("breast-cancer.csv")




data['diagnosis'] = data['diagnosis'].map({'M': 1, 'B': 0})
data['diagnosis']=data['diagnosis'].astype(int)
# print(data.head())

x = data.drop(columns=['id', 'diagnosis'])
y = data['diagnosis']

x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.2,random_state=42)

# models calling
lg=LogisticRegression()
dt=DecisionTreeClassifier()
rf=RandomForestClassifier(n_estimators=100,random_state=42)
knn=KNeighborsClassifier()

# model ko train kiya
lg.fit(x_train,y_train)
dt.fit(x_train,y_train)
rf.fit(x_train,y_train)
knn.fit(x_train,y_train)

# prediction
y_pred_lg=lg.predict(x_test)
y_pred_dt=dt.predict(x_test)
y_pred_rf=rf.predict(x_test)
y_pred_knn=knn.predict(x_test)

# accuracy score
lg_acc=accuracy_score(y_test,y_pred_lg)
dt_acc=accuracy_score(y_test,y_pred_dt)
rf_acc=accuracy_score(y_test,y_pred_rf)
knn_acc=accuracy_score(y_test,y_pred_knn)

# classification report
lg_cr=classification_report(y_test,y_pred_lg)
dt_cr=classification_report(y_test,y_pred_dt)       
rf_cr=classification_report(y_test,y_pred_rf)
knn_cr=classification_report(y_test,y_pred_knn)

#confusion matrix
lg_cm=confusion_matrix(y_test,y_pred_lg)    
dt_cm=confusion_matrix(y_test,y_pred_dt)
rf_cm=confusion_matrix(y_test,y_pred_rf)
knn_cm=confusion_matrix(y_test,y_pred_knn)

# Logistic regression
print(f"logistic regression prediction: {y_pred_lg}")
print(f"logistic regression accuracy: {lg_acc}")
print(f"logistic regression classification report: {lg_cr}")
print(f"logistic regression confusion matrix: {lg_cm}")

# Deciosion Tree
print(f"decision tree prediction: {y_pred_dt}")
print(f"decision tree accuracy: {dt_acc}")
print(f"decision tree classification report: {dt_cr}")
print(f"decision tree confusion matrix: {dt_cm}")

# Random Forest
print(f"random forest prediction: {y_pred_rf}")
print(f"random forest accuracy: {rf_acc}")
print(f"random forest classification report: {rf_cr}")
print(f"random forest confusion matrix: {rf_cm}")

# KNN
print(f"KNN prediction: {y_pred_knn}")  
print(f"KNN accuracy: {knn_acc}")
print(f"KNN classification report: {knn_cr}")       
print(f"KNN confusion matrix: {knn_cm}")


# visualization confusion matrix
models = ['Logistic Regression', 'Decision Tree', 'Random Forest', 'KNN']
accuracies = [lg_acc, dt_acc, rf_acc, knn_acc]

plt.figure(figsize=(8,6))
sns.heatmap(lg_cm,annot=True)
plt.title('Logistic Regression Confusion Matrix')
plt.show()