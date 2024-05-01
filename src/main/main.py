import time
import os
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from sklearn.datasets import make_classification
from sklearn.linear_model import LogisticRegression as SklearnLogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, StandardScaler

from concrete.ml.sklearn import LogisticRegression as ConcreteLogisticRegression

import matplotlib.pyplot as plt
# from IPython.display import display

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

sns.heatmap(df.corr())
sns.pairplot(df)

plt.subplots(figsize=(20,15))
sns.boxplot(x='Age', y='BMI', data=df)

# Split data into features and target
x = df.drop('Outcome',axis=1)
y = df['Outcome']

# Split data into training and testing sets
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.3,random_state=101)

sklearn_logr = SklearnLogisticRegression()
sklearn_logr.fit(x_train, y_train)
y_pred_test = sklearn_logr.predict(x_test)

# Create concrete logistic regression model with 8 bits precision
concrete_logr = ConcreteLogisticRegression(n_bits=8)
concrete_logr.fit(x_train, y_train)

# Predict on the test set
y_proba_q = concrete_logr.predict_proba(x_test)[:, 1]
y_pred_q = concrete_logr.predict(x_test)

fhe_circuit = concrete_logr.compile(x_train)

print(f"Generating a key for an {fhe_circuit.graph.maximum_integer_bit_width()}-bit circuit")

time_begin = time.time()
fhe_circuit.client.keygen(force=False)
print(f"Key generation time: {time.time() - time_begin:.4f} seconds")

time_begin = time.time()
y_pred_fhe = concrete_logr.predict(x_test, fhe="execute")
print(f"Execution time: {(time.time() - time_begin) / len(x_test):.4f} seconds per sample")

sklearn_accuracy = accuracy_score(y_test, y_pred_test)
quantized_accuracy = accuracy_score(y_test, y_pred_q)
fhe_accuracy = accuracy_score(y_test, y_pred_fhe)

print(f"Sklearn accuracy: {sklearn_accuracy:.4f}")
print(f"Quantized Clear Accuracy: {quantized_accuracy:.4f}")
print(f"FHE Accuracy: {fhe_accuracy:.4f}")

# Measure the error of the FHE quantized model with respect to the clear quantized model
concrete_score_difference = abs(fhe_accuracy - quantized_accuracy)
print(
    "\nRelative difference between Concrete-ml (quantized clear) and Concrete-ml (FHE) scores:",
    f"{concrete_score_difference:.2f}%",
)

# Create Sklearn logistic regression model
logmodel = LogisticRegression()
logmodel.fit(x_train,y_train)

# Generate predictions using Sklearn model
predictions = logmodel.predict(x_test)
print(classification_report(y_test,predictions))