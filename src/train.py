import os
import joblib
import mlflow
import mlflow.sklearn
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# -----------------------------
# Load Dataset
# -----------------------------
df = pd.read_csv("data/heart.csv")

# Features and Target
X = df.iloc[:, :-1]
y = df.iloc[:, -1]

# Split Dataset
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

# -----------------------------
# Models
# -----------------------------
models = {
    "Logistic Regression": LogisticRegression(max_iter=1000),
    "Decision Tree": DecisionTreeClassifier(random_state=42),
    "Random Forest": RandomForestClassifier(random_state=42)
}

best_model = None
best_accuracy = 0

# Create models folder if it doesn't exist
os.makedirs("models", exist_ok=True)

# Create MLflow experiment
mlflow.set_experiment("Heart Disease Prediction")

# -----------------------------
# Train Models
# -----------------------------
for name, model in models.items():

    with mlflow.start_run(run_name=name):

        # Train
        model.fit(X_train, y_train)

        # Predict
        predictions = model.predict(X_test)

        # Accuracy
        accuracy = accuracy_score(y_test, predictions)

        print(f"{name}: {accuracy:.4f}")

        # Log Parameters
        mlflow.log_param("model", name)

        # Log Metrics
        mlflow.log_metric("accuracy", accuracy)

        # Log & Register Model
        mlflow.sklearn.log_model(
            sk_model=model,
            name="model",
            registered_model_name="HeartDiseaseModel"
        )

        # Save Best Model
        if accuracy > best_accuracy:
            best_accuracy = accuracy
            best_model = model

# -----------------------------
# Save Best Model Locally
# -----------------------------
joblib.dump(best_model, "models/best_model.pkl")

print(f"\nBest Accuracy: {best_accuracy:.4f}")
print("Best model saved to models/best_model.pkl")