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

df.head()
df.info()
df.describe()

sns.heatmap(df.isnull(),yticklabels=False,cbar=False,cmap='viridis')
sns.countplot(x='Outcome',data=df)
sns.distplot(df['Age'].dropna(),kde=True)

df.corr()

sns.heatmap(df.corr())
sns.pairplot(df)

plt.subplots(figsize=(20,15))
sns.boxplot(x='Age', y='BMI', data=df)

x = df.drop('Outcome',axis=1)
y = df['Outcome']

x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.3,random_state=101)
logmodel = LogisticRegression()
logmodel.fit(x_train,y_train)