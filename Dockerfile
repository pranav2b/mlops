# Use an official Python runtime as a parent image
FROM python:3.11-slim

# Set the working directory in the container
WORKDIR /app

# Create the logs directory
RUN mkdir logs

# Copy the dependencies file to the working directory
COPY requirements.txt .

# Install any needed packages specified in requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Copy the local source code and model artifacts to the container
COPY ./src /app/src
COPY ./mlruns /app/mlruns

# Expose the port the app runs on
EXPOSE 8000

# Command to run the application when the container starts
CMD ["uvicorn", "src.api.main:app", "--host", "0.0.0.0", "--port", "8000"]