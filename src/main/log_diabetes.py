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

print(df.head(5))

X = df.drop(columns=["Outcome"])
y = df["Outcome"]

def evaluate(
    model, x, y, test_size=0.33, show_circuit=False, predict_in_fhe=True, fhe_samples=None
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
    if is_concrete_ml:

        print("Compile the model")

        # Compile the model using the training data
        circuit = model["model"].compile(x_train)  # pylint: disable=no-member

        # Print the FHE circuit if needed
        if show_circuit:
            print(circuit)

        # Retrieve the circuit's max bit-width
        evaluation_result["max bit-width"] = circuit.graph.maximum_integer_bit_width()

        print("Predict (simulated)")

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
            y_pred_fhe = model.predict(x_test, fhe="execute")
            evaluation_result["FHE execution time (second per sample)"] = (
                time.time() - before_time
            ) / test_length

            evaluation_result["Accuracy (FHE)"] = accuracy_score(y_test, y_pred_fhe)

    print("Done !\n")

    return evaluation_result


def run_eval(use_dt, use_rt, use_xgb, predict_in_fhe):
    results = []

    # Define the test size proportion
    test_size = 0.2

    # For testing FHE execution locally, define the number of inference to run. If None, the complete
    # test set is used
    fhe_samples = None

    # Logistic regression
    results.append(evaluate(SklearnLogisticRegression(), X, y, test_size=test_size))
    results.append(evaluate(ConcreteLogisticRegression(), X, y, test_size=test_size))

    # Define the initialization parameters for tree-based models
    init_params_dt = {"max_depth": 10}
    init_params_rf = {"max_depth": 7, "n_estimators": 5}
    init_params_xgb = {"max_depth": 7, "n_estimators": 5}
    init_params_cml = {"n_bits": 3}

    # Determine the type of models to evaluate
    # use_dt = True
    # use_rf = True
    # use_xgb = True
    # predict_in_fhe = True

    # Decision tree models
    if use_dt:

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
    
    pd.set_option("display.precision", 3)

    results_dataframe = pd.DataFrame(results)
    results_dataframe.fillna("")

    print(results_dataframe)
     


if __name__ == "__main__":
    
    model_selection = input("Enter the model you wish to evaluate (log, dt, rf, xgb, all):")

    match(model_selection):
        case "log":
            use_dt = False
            use_rf = False
            use_xgb = False
            predict_in_fhe = True
            run_eval(use_dt, use_rf, use_xgb, predict_in_fhe)
        case "dt":
            use_dt = True
            use_rf = False
            use_xgb = False
            predict_in_fhe = True
            run_eval(use_dt, use_rf, use_xgb, predict_in_fhe)
        case "rf":
            use_dt = False
            use_rf = True
            use_xgb = False
            predict_in_fhe = True
            run_eval(use_dt, use_rf, use_xgb, predict_in_fhe)
        case "xgb":
            use_dt = False
            use_rf = False
            use_xgb = True
            predict_in_fhe = True
            run_eval(use_dt, use_rf, use_xgb, predict_in_fhe)
        case "all":
            use_dt = True
            use_rf = True
            use_xgb = True
            predict_in_fhe = True
            run_eval(use_dt, use_rf, use_xgb, predict_in_fhe)
