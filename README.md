# Applied Cryptography - Project 2

## Project Title: Predicting Diabetes Based on Encrypted Health Data Using Machine Learning (ML) Classifiers

**Project Type 4:** Designing a cryptography solution to allow computation of an outsourced algorithm or machine learning classifier over your encrypted input (test) data via homomorphic encryption scheme<br /><br />

**Team Name: Hard Core Bit-ches**<br /><br />

**Team Members:**<br />
 - Ariana Sutanto<br />
 - Mahmoud Shabana<br />
 - Reagan Bachman<br /><br />

## Setup

#### DISCLAIMER!!!
This program does not work on Windows due to the incompatibility of <code>concrete-ml</code>, please use macOS, Linux, or Windows WSL
<hr>

For creating the project virtual environment:

```
sudo apt update && apt install python3.10.12
python3.10.12 -m venv <your-env>
```

To launch your virtual environment in Linux:
```
source ./<your-env>/bin/activate
```

To install project dependencies:
```
pip install -r requirements.txt
```

## Using Diabetes prediction program

To run the application interface for inference (inside your virtual environment):
```
python3 main.py
```

Once running you will be prompted to enter the following input:
```
Would you like to enter the data manually? (y/n): 
```

If you want to submit multiple datapoints for prediction, please provide the name of your csv file that follows the same header format in <code>example.csv</code>:
```
Please enter the name of the input file within ./data/ (please use csv file): 
```

If you want enter a single datapoint for prediction, provide values for the following prompt that the program will request:
```
Please enter the following information:
Would you like to enter the data manually? (y/n): y
Pregnancies: 
Glucose:  
BloodPressure:
SkinThickness:
Insulin:
BMI: 
DiabetesPedigreeFunction: 
Age:
```

Once input is complete, the following output will be displayed to the user:

```
# Single input
   Pregnancies  Glucose  BloodPressure  SkinThickness  Insulin   BMI  DiabetesPedigreeFunction   Age
0         10.0     10.0           10.0           10.0     10.0  40.0                       0.0  55.0
Prediction time: 0.0098 seconds
Prediction: [0]
The patient is unlikely to have diabetes.
```

```
# Multiple input
   Pregnancies  Glucose  BloodPressure  SkinThickness  Insulin   BMI  DiabetesPedigreeFunction  Age
0            6      200             72             35        0  40.0                     0.627   50
1            6      148             72             35        0  33.6                     0.627   50
2            5      166             72             19      175  25.8                     0.587   51
3            1      103             30             38       83  43.3                     0.183   33
4            1      115             70             30       96  34.6                     0.529   32
5            0      200            100              0        0  50.0                     0.000   50
Prediction time: 0.0186 seconds
Prediction: [1 0 0 0 0 1]
For index 0, The patient is likely to have diabetes.
For index 1, The patient is unlikely to have diabetes.
For index 2, The patient is unlikely to have diabetes.
For index 3, The patient is unlikely to have diabetes.
For index 4, The patient is unlikely to have diabetes.
For index 5, The patient is likely to have diabetes.
```

## EXTRA CREDIT: Running Multiple Model Diabetes prediction 

For this program, we implemented FHE prediction on diabetes datapoints using not only Logistical regression, but also Decision Tree, Random Forest, and XGBoost Classifiers.

To run the following program, please reference the <a href="#setup">setup</a> for creating the program environment.

Once the virtual environment is created, run the <code>multi_model.py</code> file:
```
python3 multi_model.py
```

Once the program is running, you will be prompted to select the type of operation you wish to run in the program. You can select the program to run training and testing against the <code>diabetes.csv</code> dataset, or you can run the program in *inference mode* to predict against user inputs:
```
Enter 1 if you want to test models, enter 2 if you want to predict: 
```

If you select inference mode (option 2) you will be prompted to select the model of choice for inference:
```
Enter the model you wish to evaluate (log, dt, rf, xgb, all):
```

From there you will follow a similar input workflow to the previous <code>main.py</code>, where you can input a file for multiple datapoint predictions or one single datapoint prediction. The results of the program will look something like this when complete:

```
# Running multiple inputs on all models
Results for input data:
   Pregnancies  Glucose  BloodPressure  SkinThickness  Insulin   BMI  DiabetesPedigreeFunction  Age
0            6      200             72             35        0  40.0                     0.627   50
1            6      148             72             35        0  33.6                     0.627   50
2            5      166             72             19      175  25.8                     0.587   51
3            1      103             30             38       83  43.3                     0.183   33
4            1      115             70             30       96  34.6                     0.529   32
5            0      200            100              0        0  50.0                     0.000   50

***************MODEL RESULTS***************

Model: Logistic Regression
Prediction 0: Diabetic
Prediction 1: Diabetic
Prediction 2: Diabetic
Prediction 3: Not Diabetic
Prediction 4: Not Diabetic
Prediction 5: Diabetic
Execution time: 0.0028 seconds per sample

Model: Decision Tree
Prediction 0: Diabetic
Prediction 1: Diabetic
Prediction 2: Diabetic
Prediction 3: Not Diabetic
Prediction 4: Diabetic
Prediction 5: Diabetic
Execution time: 1.0374 seconds per sample

Model: Random Forest
Prediction 0: Diabetic
Prediction 1: Diabetic
Prediction 2: Diabetic
Prediction 3: Not Diabetic
Prediction 4: Not Diabetic
Prediction 5: Diabetic
Execution time: 1.4409 seconds per sample

Model: XGBoost
Prediction 0: Diabetic
Prediction 1: Diabetic
Prediction 2: Diabetic
Prediction 3: Not Diabetic
Prediction 4: Diabetic
Prediction 5: Diabetic
Execution time: 0.9194 seconds per sample
```

