import time
import os
import pickle
import numpy as np 
import pandas as pd
import seaborn as sns
from pathlib import Path
import matplotlib.pyplot as plt
from sklearn.datasets import make_classification
from sklearn.linear_model import LogisticRegression as SklearnLogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

from concrete.ml.sklearn import LogisticRegression as ConcreteLogisticRegression

# Find and load dataset
for dirname, _, filenames in os.walk('data/diabetes.csv'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

df = pd.read_csv('data/diabetes.csv')

# Display information about dataset
df.head()
df.info()
df.describe()

# Create heatmap to visualize missing data
sns.heatmap(df.isnull(),yticklabels=False,cbar=False,cmap='viridis')

# Create count plot of Outcome column
sns.countplot(x='Outcome',data=df)

# Create distribution plot for Age column
sns.distplot(df['Age'].dropna(),kde=True)

# Compute pairwise correlation of columns
df.corr()

# Create heatmap to visualize correlation
sns.heatmap(df.corr())

# Create pairplot to visualize relationships between columns
sns.pairplot(df)

# Create boxplot to visualize relationship between Age and BMI
plt.subplots(figsize=(20,15))
sns.boxplot(x='Age', y='BMI', data=df)

# Split data into features and target
x = df.drop('Outcome',axis=1)
y = df['Outcome']

# Split data into training and testing sets
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.3,random_state=101)

# Create Sklearn Logistic Regression model
sklearn_logr = SklearnLogisticRegression()
sklearn_logr.fit(x_train, y_train)

# Predict on the test set using Sklearn model
y_pred_test = sklearn_logr.predict(x_test)

# Output Sklearn model to file
sklearn_model_path = Path("sklearn_model.pkl")
with sklearn_model_path.open("wb") as f:
    pickle.dump(sklearn_logr, f)

# Create Quantized Concrete-ml Logistic Regression model with 8 bits precision
concrete_logr = ConcreteLogisticRegression(n_bits=8)
concrete_logr.fit(x_train, y_train)

# Predict on the test set using Quantized Concrete-ml model
y_proba_q = concrete_logr.predict_proba(x_test)[:, 1]
y_pred_q = concrete_logr.predict(x_test)

# Compile Concrete-ml model to FHE
fhe_circuit = concrete_logr.compile(x_train)

# Output FHE model to file
fhe_model_path = Path("fhe_model.json")
with fhe_model_path.open("w") as f:
    concrete_logr.dump(f)

print(f"Generating a key for an {fhe_circuit.graph.maximum_integer_bit_width()}-bit circuit")

# Measure time taken for key generation
time_begin = time.time()
fhe_circuit.client.keygen(force=False)
print(f"Key generation time: {time.time() - time_begin:.4f} seconds")

# Measure execution time for Sklearn prediction
time_begin = time.time()
y_pred_test = sklearn_logr.predict(x_test)
print(f"Sklearn execution time: {(time.time() - time_begin) / len(x_test):.4f} seconds per sample")

# Measure execution time for Quantized Concrete-ml prediction
time_begin = time.time()
y_pred_q = concrete_logr.predict(x_test)
print(f"Quantized execution time: {(time.time() - time_begin) / len(x_test):.4f} seconds per sample")

# Measure execution time for Concrete-ml FHE prediction
time_begin = time.time()
y_pred_fhe = concrete_logr.predict(x_test, fhe="execute")
print(f"FHE execution time: {(time.time() - time_begin) / len(x_test):.4f} seconds per sample")

# Calculate accuracy of the models
sklearn_accuracy = accuracy_score(y_test, y_pred_test)
quantized_accuracy = accuracy_score(y_test, y_pred_q)
fhe_accuracy = accuracy_score(y_test, y_pred_fhe)

print(f"Sklearn accuracy: {sklearn_accuracy:.4f}")
print(f"Quantized accuracy: {quantized_accuracy:.4f}")
print(f"FHE accuracy: {fhe_accuracy:.4f}")

# Measure the error of the FHE quantized model with respect to the clear quantized model
concrete_score_difference = abs(fhe_accuracy - quantized_accuracy)
print(
    "\nRelative difference between Concrete-ml (quantized clear) and Concrete-ml (FHE) scores:",
    f"{concrete_score_difference:.2f}%",
    )

# Measure the error of the FHE quantized model with respect to the clear scikit-learn float model
score_difference = abs(fhe_accuracy - sklearn_accuracy)
print(
    "Relative difference between scikit-learn (clear) and Concrete-ml (FHE) scores:",
    f"{score_difference:.2f}%",
)

# Create file to store execution times and accuracy
with open("results.txt", "w") as f:
    f.write(f"Key generation time: {time.time() - time_begin:.4f} seconds\n")
    f.write(f"Sklearn execution time: {(time.time() - time_begin) / len(x_test):.4f} seconds per sample\n")
    f.write(f"Quantized execution time: {(time.time() - time_begin) / len(x_test):.4f} seconds per sample\n")
    f.write(f"FHE execution time: {(time.time() - time_begin) / len(x_test):.4f} seconds per sample\n")
    f.write(f"Sklearn accuracy: {sklearn_accuracy:.4f}\n")
    f.write(f"Quantized accuracy: {quantized_accuracy:.4f}\n")
    f.write(f"FHE accuracy: {fhe_accuracy:.4f}\n")
    f.write(
        f"\nRelative difference between Concrete-ml (quantized clear) and Concrete-ml (FHE) scores: {concrete_score_difference:.2f}%\n"
    )
    f.write(
        f"Relative difference between scikit-learn (clear) and Concrete-ml (FHE) scores: {score_difference:.2f}%\n"
    )