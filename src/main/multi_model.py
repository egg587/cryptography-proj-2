import time

import pandas as pd
from sklearn.ensemble import RandomForestClassifier as SklearnRandomForestClassifier
from sklearn.linear_model import LogisticRegression as SklearnLogisticRegression
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

# Import models from scikit-learn and XGBoost
from sklearn.tree import DecisionTreeClassifier as SklearnDecisionTreeClassifier
from xgboost import XGBClassifier as SklearnXGBoostClassifier

# Import models from Concrete ML
from concrete.ml.sklearn import DecisionTreeClassifier as ConcreteDecisionTreeClassifier
from concrete.ml.sklearn import LogisticRegression as ConcreteLogisticRegression
from concrete.ml.sklearn import RandomForestClassifier as ConcreteRandomForestClassifier
from concrete.ml.sklearn import XGBClassifier as ConcreteXGBoostClassifier

CONCRETE_ML_MODELS = [
    ConcreteDecisionTreeClassifier,
    ConcreteLogisticRegression,
    ConcreteRandomForestClassifier,
    ConcreteXGBoostClassifier,
]

# Read the dataset
df = pd.read_csv("./data/diabetes.csv")

X = df.drop(columns=["Outcome"])
y = df["Outcome"]

def evaluate(
    model, x, y, test_size=0.33, show_circuit=False, predict_in_fhe=True, fhe_samples=None, run_inf=False, run_train=False, test_data=None
):
    """Evaluate the given model using several metrics.

    The model is evaluated using the following metrics: accuracy, F1 score, precision, recall.
    For Concrete ML models, the inference's execution time is provided when done in FHE.

    Args:
        model: The initialized model to consider.
        x: The input data to consider.
        y: The target data to consider.
        test_size: The proportion to use for the test data. Default to 0.33.
        show_circuit: If the FHE circuit should be printed for Concrete ML models. Default to False.
        predict_in_fhe: If the inference should be executed in FHE for Concrete ML models. Else, it
            will only be simulated.
        fhe_sample: The number of samples to consider for evaluating the inference of Concrete ML
            models if predict_in_fhe is set to True. If None, the complete test set is used. Default
            to None.
    """
    evaluation_result = {}

    is_concrete_ml = model.__class__ in CONCRETE_ML_MODELS

    name = model.__class__.__name__ + (" (Concrete ML)" if is_concrete_ml else " (sklearn)")

    evaluation_result["name"] = name

    print(f"Evaluating model {name}")

    # Split the data into test and train sets. Stratify is used to make sure that the test set
    # contains some representative class distribution for targets
    x_train, x_test, y_train, y_test = train_test_split(
        x, y, stratify=y, test_size=test_size, random_state=1
    )
    test_length = len(x_test)

    evaluation_result["Test samples"] = test_length

    evaluation_result["n_bits"] = model.n_bits if is_concrete_ml else None

    # Normalization pipeline
    model = Pipeline(
        [
            ("preprocessor", StandardScaler()),
            ("model", model),
        ]
    )

    # Train the model
    model.fit(x_train, y_train)

    # Run the prediction and store its execution time
    y_pred = model.predict(x_test)

    # Evaluate the model
    # For Concrete ML models, this will execute the (quantized) inference in the clear
    evaluation_result["Accuracy (clear)"] = accuracy_score(y_test, y_pred)
    evaluation_result["F1 (clear)"] = f1_score(y_test, y_pred, average="macro")
    evaluation_result["Precision (clear)"] = precision_score(y_test, y_pred, average="macro")
    evaluation_result["Recall (clear)"] = recall_score(y_test, y_pred, average="macro")

    # If the model is from Concrete ML
    if is_concrete_ml or run_inf:

        print("Compile the model")

        # Compile the model using the training data
        circuit = model["model"].compile(x_train)  # pylint: disable=no-member

        # Print the FHE circuit if needed
        if show_circuit:
            print(circuit)

        # Retrieve the circuit's max bit-width
        evaluation_result["max bit-width"] = circuit.graph.maximum_integer_bit_width()

        #print("Predict (simulated)")

        # Run the prediction in the clear using FHE simulation, store its execution time and
        # evaluate the accuracy score
        y_pred_simulate = model.predict(x_test, fhe="simulate")

        evaluation_result["Accuracy (simulated)"] = accuracy_score(y_test, y_pred_simulate)

        # Run the prediction in FHE, store its execution time and evaluate the accuracy score
        if predict_in_fhe:
            if fhe_samples is not None:
                x_test = x_test[0:fhe_samples]
                y_test = y_test[0:fhe_samples]
                test_length = fhe_samples

            evaluation_result["FHE samples"] = test_length

            print("Predict (FHE)")

            before_time = time.time()
            if run_inf and test_data is not None:
                y_pred_fhe = model.predict(test_data, fhe="execute")
                exec_time = (time.time() - before_time) / len(test_data)
                return y_pred_fhe, exec_time
            else:
                y_pred_fhe = model.predict(x_test, fhe="execute")
            evaluation_result["FHE execution time (second per sample)"] = (
                time.time() - before_time
            ) / test_length

            evaluation_result["Accuracy (FHE)"] = accuracy_score(y_test, y_pred_fhe)

    print("Done !\n")

    return evaluation_result

def get_input_data(X_train, program_type):
    """Prompt user for input data."""
    print("Welcome to the Multi Model Diabetes Prediction System!")
    print("Please enter the following information:")
    model_selection = input("Enter the model you wish to evaluate (log, dt, rf, xgb, all):")

    if program_type == "2":
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

        # Select the model to evaluate and pass it to the run_eval function
        return input_df, model_selection
    elif program_type == "1":
        return None, model_selection


def run_eval(use_dt, use_rt, use_xgb, predict_in_fhe, test_data=None):
    results = []
    results_multi = {"Logistic Regression": [], "Decision Tree": [], "Random Forest": [], "XGBoost": []}

    # Define the test size proportion
    test_size = 0.2

    # For testing FHE execution locally, define the number of inference to run. If None, the complete
    # test set is used
    fhe_samples = None

    # Logistic regression
    if test_data is not None:
        predictions = evaluate(
            ConcreteLogisticRegression(),
            X,
            y,
            test_size=test_size,
            predict_in_fhe=predict_in_fhe,
            run_inf=True,
            test_data=test_data,
        )
        results_multi["Logistic Regression"] = predictions
    else:
        results.append(evaluate(SklearnLogisticRegression(), X, y, test_size=test_size))
        results.append(evaluate(ConcreteLogisticRegression(), X, y, test_size=test_size))

    # Define the initialization parameters for tree-based models
    init_params_dt = {"max_depth": 10}
    init_params_rf = {"max_depth": 7, "n_estimators": 5}
    init_params_xgb = {"max_depth": 7, "n_estimators": 5}
    init_params_cml = {"n_bits": 3}

    # Decision tree models
    if use_dt:

        if test_data is not None:
            predictions = evaluate(
                ConcreteDecisionTreeClassifier(**init_params_dt, **init_params_cml),
                X,
                y,
                test_size=test_size,
                predict_in_fhe=predict_in_fhe,
                run_inf=True,
                test_data=test_data,
            )
            results_multi["Decision Tree"] = predictions
        else:
            
            # Scikit-Learn model
            results.append(
                evaluate(
                    SklearnDecisionTreeClassifier(**init_params_dt),
                    X,
                    y,
                    test_size=test_size,
                )
            )

            # Concrete ML model
            results.append(
                evaluate(
                    ConcreteDecisionTreeClassifier(**init_params_dt, **init_params_cml),
                    X,
                    y,
                    test_size=test_size,
                    predict_in_fhe=predict_in_fhe,
                    fhe_samples=fhe_samples,
                )
            )

    # Random Forest
    if use_rf:

        if test_data is not None:
            predictions = evaluate(
                ConcreteRandomForestClassifier(**init_params_rf, **init_params_cml),
                X,
                y,
                test_size=test_size,
                predict_in_fhe=predict_in_fhe,
                run_inf=True,
                test_data=test_data,
            )
            results_multi["Random Forest"] = predictions
        else:
            # Scikit-Learn model
            results.append(
                evaluate(
                    SklearnRandomForestClassifier(**init_params_rf),
                    X,
                    y,
                    test_size=test_size,
                )
            )

            # Concrete ML model
            results.append(
                evaluate(
                    ConcreteRandomForestClassifier(**init_params_rf, **init_params_cml),
                    X,
                    y,
                    test_size=test_size,
                    predict_in_fhe=predict_in_fhe,
                    fhe_samples=fhe_samples,
                )
            )

    # XGBoost
    if use_xgb:

        if test_data is not None:
            predictions = evaluate(
                ConcreteXGBoostClassifier(**init_params_xgb, **init_params_cml),
                X,
                y,
                test_size=test_size,
                predict_in_fhe=predict_in_fhe,
                run_inf=True,
                test_data=test_data,
            )
            results_multi["XGBoost"] = predictions
        else:

            # Scikit-Learn model
            results.append(
                evaluate(
                    SklearnXGBoostClassifier(**init_params_xgb),
                    X,
                    y,
                    test_size=test_size,
                )
            )

            # Concrete ML model
            results.append(
                evaluate(
                    ConcreteXGBoostClassifier(**init_params_xgb, **init_params_cml),
                    X,
                    y,
                    test_size=test_size,
                    predict_in_fhe=predict_in_fhe,
                    fhe_samples=fhe_samples,
                )
            )
    
    # Return the multiple model results
    if run_inf and test_data is not None:
        return results_multi
    
    pd.set_option("display.precision", 3)

    results_dataframe = pd.DataFrame(results)
    results_dataframe.fillna("")

    print(results_dataframe)
    return None

def display_results(input_df, results):
    """Display the results of the evaluation."""
    print(f"\nResults for input data: \n{input_df}")
    print("\n***************MODEL RESULTS***************")
    for model, data in results.items():
        if len(data) == 0:
            continue
        print(f"\nModel: {model}")
        for idx, y_pred in enumerate(list(data[0])):
            if y_pred == 1:
                print(f"Prediction {idx}: Diabetic")
            else:
                print(f"Prediction {idx}: Not Diabetic")
        print(f"Execution time: {data[1]:.4f} seconds per sample")


if __name__ == "__main__":
    program_type = input("Enter 1 if you want to test models, enter 2 if you want to predict: ")
    if program_type == "2":
        run_inf = True
    else:
        run_inf = False
    input_df, model_selection = get_input_data(X, program_type)

    match(model_selection):
        case "log":
            use_dt = False
            use_rf = False
            use_xgb = False
            predict_in_fhe = True
            results = run_eval(use_dt, use_rf, use_xgb, predict_in_fhe, test_data=input_df)
        case "dt":
            use_dt = True
            use_rf = False
            use_xgb = False
            predict_in_fhe = True
            results = run_eval(use_dt, use_rf, use_xgb, predict_in_fhe, test_data=input_df)
        case "rf":
            use_dt = False
            use_rf = True
            use_xgb = False
            predict_in_fhe = True
            results = run_eval(use_dt, use_rf, use_xgb, predict_in_fhe, test_data=input_df)
        case "xgb":
            use_dt = False
            use_rf = False
            use_xgb = True
            predict_in_fhe = True
            results = run_eval(use_dt, use_rf, use_xgb, predict_in_fhe, test_data=input_df)
        case "all":
            use_dt = True
            use_rf = True
            use_xgb = True
            predict_in_fhe = True
            results = run_eval(use_dt, use_rf, use_xgb, predict_in_fhe, test_data=input_df)
        
    if results is not None:
        display_results(input_df, results)
