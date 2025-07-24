import os
import shutil
import pandas as pd
import mlflow
from mlflow.tracking import MlflowClient
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# Set the tracking URI to the local directory
mlflow.set_tracking_uri("file:./mlruns")

# 1. Load Data
train_df = pd.read_csv('data/processed/train.csv')
test_df = pd.read_csv('data/processed/test.csv')

X_train = train_df.drop('target', axis=1)
y_train = train_df['target']
X_test = test_df.drop('target', axis=1)
y_test = test_df['target']

# 2. Train and Track Logistic Regression
with mlflow.start_run(run_name='LogisticRegression'):
    print("Training Logistic Regression model...")
    mlflow.log_param("model_type", "Logistic Regression")

    lr = LogisticRegression(random_state=42)
    lr.fit(X_train, y_train)

    y_pred = lr.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)

    mlflow.log_metric("accuracy", accuracy)
    mlflow.sklearn.log_model(lr, "model", input_example=X_train.head(1))
    print(f"Logistic Regression Accuracy: {accuracy:.2f}")

# 3. Train and Track Random Forest
with mlflow.start_run(run_name='RandomForest'):
    print("\nTraining Random Forest model...")
    n_estimators = 100
    max_depth = 5
    mlflow.log_param("model_type", "Random Forest")
    mlflow.log_param("n_estimators", n_estimators)
    mlflow.log_param("max_depth", max_depth)

    rf = RandomForestClassifier(n_estimators=n_estimators, max_depth=max_depth, random_state=42)
    rf.fit(X_train, y_train)

    y_pred = rf.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)

    mlflow.log_metric("accuracy", accuracy)
    mlflow.sklearn.log_model(rf, "model", input_example=X_train.head(1))
    print(f"Random Forest Accuracy: {accuracy:.2f}")

print("\nTraining complete. Finding the best model.")

# --- Find Best Model and Save to a Fixed Path ---
best_run = mlflow.search_runs(order_by=['metrics.accuracy DESC']).iloc[0]
best_run_id = best_run.run_id
best_model_accuracy = best_run["metrics.accuracy"]
best_model_uri = f"runs:/{best_run_id}/model"

print(f"Best model found in run {best_run_id} with accuracy: {best_model_accuracy}")

# Define a fixed output path
output_path = "outputs/model"
if os.path.exists(output_path):
    shutil.rmtree(output_path)

# Download the best model's artifacts
model_source_path = mlflow.artifacts.download_artifacts(best_model_uri)
shutil.copytree(model_source_path, output_path)

print(f"Best model copied to {output_path}")