# Support Vector Machine on the Breast Cancer dataset

from sklearn.datasets import load_breast_cancer
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Load dataset
data = load_breast_cancer()
X = data.data
y = data.target

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# Train SVM model
model = SVC()
model.fit(X_train, y_train)

# Predict and evaluate
y_pred = model.predict(X_test)
print("SVM Accuracy:", accuracy_score(y_test, y_pred))

