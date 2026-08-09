import mlflow
import mlflow.sklearn
import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score


# Load dataset
data = pd.read_csv("Project/data/iris.csv")

X = data.drop("target", axis=1)
y = data["target"]


# Split dataset
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)


# MLflow configuration
mlflow.set_tracking_uri("sqlite:///mlflow.db")
mlflow.set_experiment("iris-classification")


# Define 3 models
models = {
    "Logistic Regression": Pipeline([
        ("scaler", StandardScaler()),
        ("model", LogisticRegression(max_iter=200))
    ]),

    "Random Forest": RandomForestClassifier(
        n_estimators=100,
        random_state=42
    ),

    "SVM": Pipeline([
        ("scaler", StandardScaler()),
        ("model", SVC(kernel="rbf"))
    ])
}


results = {}


# Train and track each model
for name, model in models.items():

    with mlflow.start_run(run_name=name):

        # Train
        model.fit(X_train, y_train)

        # Predict
        predictions = model.predict(X_test)

        # Evaluate
        accuracy = accuracy_score(y_test, predictions)

        # Store result
        results[name] = accuracy

        # Log parameters
        mlflow.log_param("model_name", name)

        if name == "Random Forest":
            mlflow.log_param("n_estimators", 100)

        elif name == "Logistic Regression":
            mlflow.log_param("max_iter", 200)

        elif name == "SVM":
            mlflow.log_param("kernel", "rbf")

        # Log metric
        mlflow.log_metric("accuracy", accuracy)

        # Log model
        mlflow.sklearn.log_model(
            model,
            artifact_path="model"
        )

        print(f"{name}: {accuracy:.4f}")


# Find best model
best_model_name = max(results, key=results.get)
best_accuracy = results[best_model_name]

print("\nModel Comparison:")
for name, accuracy in results.items():
    print(f"{name}: {accuracy:.4f}")

print(f"\nBest Model: {best_model_name}")
print(f"Best Accuracy: {best_accuracy:.4f}")


# Train best model again and save it
best_model = models[best_model_name]
best_model.fit(X_train, y_train)

joblib.dump(best_model, "Project/model.pkl")

print("\nBest model saved to Project/model.pkl")