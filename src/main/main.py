import time
import os
import pickle
import numpy as np 
import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt
from sklearn.datasets import make_classification
from sklearn.linear_model import LogisticRegression as SklearnLogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

from concrete.ml.sklearn import LogisticRegression as ConcreteLogisticRegression
from concrete.ml.common.serialization.loaders import load

# CONSTANTS
FHE_MODEL_FILE = Path("fhe_model.json")
SKL_MODEL_FILE = Path("sklearn_model.pkl")
df = pd.read_csv('data/diabetes.csv')

def load_data():
    """Load inference data from user provided file."""
    X = df.drop("Outcome", axis=1)
    y = df["Outcome"]
    # Split data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.3,random_state=101)
    return X_train, X_test, y_train, y_test

def load_model():
    # load FHE model
    with open(FHE_MODEL_FILE, "r") as fhe_model_file:
        fhe_model = load(fhe_model_file)

    # load Sklearn model
    with open(SKL_MODEL_FILE, "rb") as skl_model_file:
        skl_model = pickle.load(skl_model_file)
    
    return skl_model, fhe_model

def get_input_data(X_train):
    """Prompt user for input data."""
    print("Welcome to the Diabetes Prediction System!")
    print("Please enter the following information:")

    # Ask user for input data
    user_type = input("Would you like to enter the data manually? (y/n): ")
    if user_type.lower() == "n":
        input_file = input("Please enter the name of the input file within ./data/ (please use csv file): ")
        input_df = pd.read_csv("data/" + input_file)
    elif user_type.lower() != "y":
        print("Invalid input. Please try again.")
        return get_input_data()
    else:
        column_headers = df.columns.drop("Outcome")
        all_data = []
        input_data = {}
        for header in column_headers:
            input_data[header] = float(input(f"{header}: "))
        
        # Convert input data to DataFrame for prediction
        input_df = pd.DataFrame(input_data, index=[0])
    
    # Check and preprocess input data
    # Match columns and data types with training data
    expected_columns = X_train.columns
    input_df = input_df.reindex(columns=expected_columns, fill_value=0)  # Fill missing columns with zeros
    input_df = input_df.astype(X_train.dtypes)  # Match data types with training data
    return input_df


def run_prediction(fhe_model, fhe_ciruit, x_test):
    """Run prediction on the given model and test data.

    Args:
        model: The model to use for prediction.
        x_test: The test data to use for prediction.

    Returns:
        The prediction made by the model.
    """

    return fhe_model.predict(x_test, fhe="execute")
    
def main():
    # Load data
    #print("Loading data...")
    X_train, X_test, y_train, y_test = load_data()
    
    # Load model
    #print("Loading model...")
    skl_model, fhe_model = load_model()

    # Compile model on loaded data
    #print("Compiling model...")
    fhe_circuit = fhe_model.compile(X_train)

    # Prompt user for input data
    input_data = get_input_data(X_train)

    # Encrypting input data

    print(f"{input_data}")
    # Run prediction
    # calculate prediction time
    time_begin = time.time()
    y_pred = run_prediction(fhe_model, skl_model, input_data)
    print(f"Prediction time: {time.time() - time_begin:.4f} seconds")
    print(f"Prediction: {y_pred}")
    for idx, y in enumerate(y_pred):
        if y == 1:
            print(f"For index {idx}, The patient is likely to have diabetes.")
        else:
            print(f"For index {idx}, The patient is unlikely to have diabetes.")
    
main()