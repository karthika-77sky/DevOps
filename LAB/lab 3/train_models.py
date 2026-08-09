import dagshub
import pandas as pd
import mlflow
import mlflow.sklearn

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score


# -----------------------------
# 1. Load the Boston Housing data
# -----------------------------

DATA_PATH = "LAB/lab 3/data/raw/BostonHousing.csv"

data = pd.read_csv(DATA_PATH)

print("Dataset shape:", data.shape)


# -----------------------------
# 2. Separate features and target
# -----------------------------

X = data.drop("medv", axis=1)
y = data["medv"]


# -----------------------------
# 3. Split data into train/test
# -----------------------------

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)


# -----------------------------
# 4. Scale the features
# -----------------------------

scaler = StandardScaler()

X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)


# -----------------------------
# 5. Define regression models
# -----------------------------

models = {
    "Linear Regression": LinearRegression(),
    "Decision Tree": DecisionTreeRegressor(
        max_depth=6,
        random_state=42
    ),
    "Random Forest": RandomForestRegressor(
        n_estimators=100,
        max_depth=6,
        random_state=42
    )
}


# -----------------------------
# 6. Create MLflow experiment
# -----------------------------
dagshub.init(
    repo_owner="karthika-77sky",
    repo_name="DevOps",
    mlflow=True
)

mlflow.set_experiment("Boston Housing Regression")


# -----------------------------
# 7. Train and evaluate models
# -----------------------------

for model_name, model in models.items():

    with mlflow.start_run(run_name=model_name):

        model.fit(X_train, y_train)

        predictions = model.predict(X_test)

        mse = mean_squared_error(y_test, predictions)
        rmse = mse ** 0.5
        mae = mean_absolute_error(y_test, predictions)
        r2 = r2_score(y_test, predictions)

        # Log model information
        mlflow.log_param("model", model_name)

        # Log evaluation metrics
        mlflow.log_metric("MSE", mse)
        mlflow.log_metric("RMSE", rmse)
        mlflow.log_metric("MAE", mae)
        mlflow.log_metric("R2", r2)

        # Store trained model
        mlflow.sklearn.log_model(
            model,
            name="model"
        )

        print(f"\n{model_name}")
        print(f"MSE  : {mse:.4f}")
        print(f"RMSE : {rmse:.4f}")
        print(f"MAE  : {mae:.4f}")
        print(f"R2   : {r2:.4f}")


print("\nAll models have been trained and logged to MLflow.")