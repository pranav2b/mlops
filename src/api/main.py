import mlflow
from mlflow.tracking import MlflowClient
import pandas as pd
from fastapi import FastAPI
from pydantic import BaseModel
import logging

mlflow.set_tracking_uri("file:./mlruns")
# Configure logging to write to a file in a 'logs' directory
# Create a 'logs' directory in your project root for this to work
logging.basicConfig(
    filename='logs/api.log',
    level=logging.INFO,

    format='%(asctime)s - %(levelname)s - %(message)s'
)



# --- DEFINITIVE MODEL LOADING LOGIC ---
# --- SIMPLIFIED MODEL LOADING LOGIC ---
try:
    model = mlflow.pyfunc.load_model("./outputs/model")
    logging.info("Model loaded successfully from fixed path.")
except Exception as e:
    logging.error(f"Error loading model from fixed path: {e}")
    model = None
# --- END OF SIMPLIFIED LOGIC ---

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