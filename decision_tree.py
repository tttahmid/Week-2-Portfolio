import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn import metrics

# Load dataset and split into features and target
X = reduced_data.drop('Potability', axis=1)
y = reduced_data['Potability']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=1)

# Train Decision Tree model
clf = DecisionTreeClassifier()
clf.fit(X_train, y_train)

# Predict and calculate accuracy
y_pred = clf.predict(X_test)
accuracy = metrics.accuracy_score(y_test, y_pred) * 100
print(f"Model Accuracy: {accuracy:.2f}%")
