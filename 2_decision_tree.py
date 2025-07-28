# Decision Tree classification on the Titanic passenger dataset

import pandas as pd
import seaborn as sns
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import LabelEncoder

# Load Titanic dataset directly from seaborn
df = sns.load_dataset('titanic')

# Remove rows with missing values in selected columns
df = df.dropna(subset=['age', 'embarked', 'sex'])

# Select features and target
X = df[['pclass', 'sex', 'age', 'fare', 'embarked']].copy()
y = df['survived']

# Encode categorical features (sex, embarked)
le = LabelEncoder()
X['sex'] = le.fit_transform(X['sex'])
X['embarked'] = le.fit_transform(X['embarked'])

# Split the data into train/test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Initialize and train the Decision Tree model
model = DecisionTreeClassifier()
model.fit(X_train, y_train)

# Make predictions
y_pred = model.predict(X_test)

# Evaluate the model
print("Accuracy:", accuracy_score(y_test, y_pred))
print(classification_report(y_test, y_pred))

