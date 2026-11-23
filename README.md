# Implementation-of-Decision-Tree-Classifier-Model-for-Predicting-Employee-Churn

## AIM:
To write a program to implement the Decision Tree Classifier Model for Predicting Employee Churn.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
```
1. Generate a synthetic employee dataset using numpy.

2. Create the target column left based on satisfaction and number of projects.

3. Encode categorical columns into dummy variables.

4. Split the data into training (80%) and testing (20%) sets.

5. Train a Decision Tree classifier and evaluate it using accuracy, confusion matrix, and classification report.

6. Visualize the trained tree.
```
## Program:
```
/*
Program to implement the Decision Tree Classifier Model for Predicting Employee Churn.
Developed by: SOWBARNIKA M P
RegisterNumber:  212225230271

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt


rng = np.random.default_rng(42)
n = 500

df = pd.DataFrame({
    "satisfaction_level":        rng.uniform(0, 1, n).round(2),
    "last_evaluation":           rng.uniform(0, 1, n).round(2),
    "number_project":            rng.integers(2, 8, n),
    "average_montly_hours":      rng.integers(120, 320, n),
    "time_spend_company":        rng.integers(1, 11, n),
    "Work_accident":             rng.integers(0, 2, n),
    "promotion_last_5years":     rng.integers(0, 2, n),
    "Departments ":              rng.choice(
                                     ["sales", "technical", "support", "IT",
                                      "hr", "accounting", "management",
                                      "marketing", "product_mng", "RandD"], n),
    "salary":                    rng.choice(["low", "medium", "high"], n),
})


logit = -3 + 4 * (1 - df["satisfaction_level"]) + 0.3 * (df["number_project"] - 5)
df["left"] = (rng.uniform(0, 1, n) < 1 / (1 + np.exp(-logit))).astype(int)


df.to_csv("Employee.csv", index=False)


df.head()
df.info()
df.isnull().sum()

df.rename(columns={'Departments ': 'Department'}, inplace=True)
df = pd.get_dummies(df, columns=['Department', 'salary'], drop_first=True)
df.head()

X = df.drop('left', axis=1)
y = df['left']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

dt_model = DecisionTreeClassifier(
    criterion='gini',
    max_depth=5,
    random_state=42
)
dt_model.fit(X_train, y_train)

y_pred = dt_model.predict(X_test)

print("Accuracy:", accuracy_score(y_test, y_pred))
print(confusion_matrix(y_test, y_pred))
print(classification_report(y_test, y_pred))

plt.figure(figsize=(20, 10))
plot_tree(
    dt_model,
    feature_names=X.columns,
    class_names=['Stayed', 'Left'],
    filled=True
)
plt.show()
*/
```

## Output:

<img width="612" height="611" alt="image" src="https://github.com/user-attachments/assets/89131b0a-2c55-43f7-bffb-91d8fac62487" />
<img width="1570" height="790" alt="image" src="https://github.com/user-attachments/assets/90705e08-15ce-4275-9377-0634c80df328" />

## Result:
Thus the program to implement the  Decision Tree Classifier Model for Predicting Employee Churn is written and verified using python programming.
