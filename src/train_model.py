import pandas as pd
import mlflow
import mlflow.sklearn
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

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

    # Log model type
    mlflow.log_param("model_type", "Logistic Regression")

    # Train model
    lr = LogisticRegression(random_state=42)
    lr.fit(X_train, y_train)

    # Evaluate model
    y_pred = lr.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)

    # Log metrics
    mlflow.log_metric("accuracy", accuracy)
    print(f"Logistic Regression Accuracy: {accuracy:.2f}")

    # Log the model
    mlflow.sklearn.log_model(
        sk_model=lr,
        artifact_path="model",
        input_example=X_train.head(1)  # This is the crucial addition
    )

# 3. Train and Track Random Forest
with mlflow.start_run(run_name='RandomForest'):
    print("\nTraining Random Forest model...")

    # Define and log parameters
    n_estimators = 100
    max_depth = 5
    mlflow.log_param("model_type", "Random Forest")
    mlflow.log_param("n_estimators", n_estimators)
    mlflow.log_param("max_depth", max_depth)

    # Train model
    rf = RandomForestClassifier(n_estimators=n_estimators, max_depth=max_depth, random_state=42)
    rf.fit(X_train, y_train)

    # Evaluate model
    y_pred = rf.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)

    # Log metrics
    mlflow.log_metric("accuracy", accuracy)
    print(f"Random Forest Accuracy: {accuracy:.2f}")

    # Log the model
mlflow.sklearn.log_model(
    sk_model=rf,
    artifact_path="model",
    input_example=X_train.head(1)  # This is the crucial addition
)

print("\nTraining and tracking complete.")