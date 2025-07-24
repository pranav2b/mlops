# import mlflow
# import pandas as pd
# from fastapi import FastAPI
# from pydantic import BaseModel
#
# # Define the input data schema using Pydantic
# class IrisFeatures(BaseModel):
#     sepal_length: float
#     sepal_width: float
#     petal_length: float
#     petal_width: float
#
# # Initialize the FastAPI app
# app = FastAPI(title="Iris Species Predictor API", version="1.0")
#
# # Load the production model from MLflow Model Registry
# # URI format: "models:/<model_name>/<stage>"
# # Set the tracking URI to the local mlruns directory inside the container
# mlflow.set_tracking_uri("./mlruns")
# model_uri = "models:/iris-classifier@prod"
# model = mlflow.pyfunc.load_model(model_uri)
# print("Model loaded successfully!")
#
# # Define the prediction endpoint
# @app.post("/predict")
# def predict(features: IrisFeatures):
#     # ... (docstring) ...
#
#     # Convert input to a DataFrame
#     input_df = pd.DataFrame([features.dict()])
#
#     # Get schema from the loaded model and rename columns
#     model_expected_features = model.metadata.get_input_schema().input_names()
#     input_df = input_df.rename(columns={old: new for old, new in zip(input_df.columns, model_expected_features)})
#
#     # Make a prediction
#     prediction = model.predict(input_df)[0]
#
#     # Map prediction index to species name
#     species_map = {0: 'setosa', 1: 'versicolor', 2: 'virginica'}
#     predicted_species = species_map.get(prediction, "unknown")
#
#     return {"predicted_species": predicted_species}
#
# # Define a root endpoint for health checks
# @app.get("/")
# def read_root():
#     return {"status": "ok", "message": "Welcome to the Iris Predictor API!"}

import mlflow
from mlflow.tracking import MlflowClient
import pandas as pd
from fastapi import FastAPI
from pydantic import BaseModel
import logging

# Configure logging to write to a file in a 'logs' directory
# Create a 'logs' directory in your project root for this to work
logging.basicConfig(
    filename='logs/api.log',
    level=logging.INFO,

    format='%(asctime)s - %(levelname)s - %(message)s'
)


# --- MODEL LOADING LOGIC ---
print("--- Attempting to load model ---")  # Add this line
try:
    mlflow.set_tracking_uri("file:./mlruns")
    client = MlflowClient()
    model_name = "iris-classifier"

    latest_versions = client.get_latest_versions(model_name, stages=["None"])
    print(f"Found {len(latest_versions)} versions for model '{model_name}'.")  # Add this line

    prod_version = next((v for v in latest_versions if "prod" in v.aliases), None)

    if prod_version is None:
        print("!!! No model version with alias 'prod' found.")  # Add this line
        raise ValueError("Production model version not found.")

    print(f"Found prod version. Source: {prod_version.source}")  # Add this line
    # The 'source' is an absolute path. We need to find the 'mlruns' part
    # and use everything after it to make it a relative path for the container.
    relative_path = "./" + "/".join(prod_version.source.split("/")[-4:])
    print(f"Corrected relative path: {relative_path}")  # Debug print
    model = mlflow.pyfunc.load_model(model_uri=relative_path)

    print("--- Model loaded successfully ---")  # Add this line

except Exception as e:
    print(f"!!! An error occurred: {e}")  # Add this line
    model = None

# --- END OF MODEL LOADING LOGIC ---


# Define the input data schema using Pydantic for request validation
class IrisFeatures(BaseModel):
    sepal_length: float
    sepal_width: float
    petal_length: float
    petal_width: float

# Initialize the FastAPI app
app = FastAPI(
    title="Iris Species Predictor API",
    description="An API to predict Iris flower species using a trained ML model.",
    version="1.0"
)


# Define the prediction endpoint
@app.post("/predict")
def predict(features: IrisFeatures):
    """
    Predicts the Iris species from input features.

    - **sepal_length**: Sepal length in cm
    - **sepal_width**: Sepal width in cm
    - **petal_length**: Petal length in cm
    - **petal_width**: Petal width in cm
    \f
    :param features: Input features from the request body.
    :return: A JSON object with the predicted species or an error message.
    """
    if model is None:
        logging.error("Model is not loaded; cannot make a prediction.")
        return {"error": "Model is not available. Please check the server logs."}

    logging.info(f"Prediction request received: {features.dict()}")

    try:
        # Create a DataFrame from the input features
        input_df = pd.DataFrame([features.dict()])

        # Get the model's expected feature names from its saved schema
        model_expected_features = model.metadata.get_input_schema().input_names()

        # Rename the DataFrame columns to match what the model expects
        input_df = input_df.rename(columns={old: new for old, new in zip(input_df.columns, model_expected_features)})

        # Make a prediction
        prediction_index = model.predict(input_df)[0]

        # Map the numeric prediction to the species name
        species_map = {0: 'setosa', 1: 'versicolor', 2: 'virginica'}
        predicted_species = species_map.get(prediction_index, "unknown")

        logging.info(f"Prediction successful. Result: {predicted_species}")
        return {"predicted_species": predicted_species}

    except Exception as e:
        logging.error(f"An error occurred during prediction: {e}")
        return {"error": "Failed to make a prediction due to a server error."}


# Define a root endpoint for health checks
@app.get("/")
def read_root():
    """A simple health check endpoint."""
    return {"status": "ok", "message": "Welcome to the Iris Predictor API!"}