import os
import pandas as pd
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split

# Create directories if they don't exist
os.makedirs('data/processed', exist_ok=True)
os.makedirs('data/raw', exist_ok=True)

# Load the Iris dataset
iris = load_iris()
iris_df = pd.DataFrame(data=iris.data, columns=iris.feature_names)
iris_df['target'] = iris.target

# Save the raw data
raw_data_path = 'data/raw/iris.csv'
iris_df.to_csv(raw_data_path, index=False)
print(f"Raw data saved to {raw_data_path}")

# Split the data for training and testing
train, test = train_test_split(iris_df, test_size=0.2, random_state=42, stratify=iris_df['target'])

# Save the processed data
train_path = 'data/processed/train.csv'
test_path = 'data/processed/test.csv'
train.to_csv(train_path, index=False)
test.to_csv(test_path, index=False)

print(f"Train data saved to {train_path}")
print(f"Test data saved to {test_path}")