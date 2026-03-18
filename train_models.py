import pandas as pd
import pickle
import os

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Algorithms
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB

# Load dataset
data = pd.read_csv("/content/Crop_recommendation.csv")

# Split features & target
X = data.drop("label", axis=1)
y = data["label"]

# Train test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# Models dictionary
models = {
    "Random Forest": RandomForestClassifier(),
    "Decision Tree": DecisionTreeClassifier(),
    "KNN": KNeighborsClassifier(),
    "Logistic Regression": LogisticRegression(max_iter=200),
    "SVM": SVC(),
    "Naive Bayes": GaussianNB()
}

best_model = None
best_accuracy = 0

print("\n📊 Model Performance:\n")

for name, model in models.items():
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)
    acc = accuracy_score(y_test, predictions)

    print(f"{name}: {acc:.2f}")

    if acc > best_accuracy:
        best_accuracy = acc
        best_model = model

# Create 'models' directory if it doesn't exist
os.makedirs('models', exist_ok=True)

# Save best model
pickle.dump(best_model, open("models/best_model.pkl", "wb"))

print(f"\n✅ Best Model Saved with Accuracy: {best_accuracy:.2f}")
