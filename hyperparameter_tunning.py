import pandas as pd
from sklearn.model_selection import train_test_split,GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, recall_score, precision_score, confusion_matrix
import seaborn as sns
import mlflow
import matplotlib.pyplot as plt
import mlflow


# Autolog is used to log everything automatically by them self



# Set MLflow tracking URI to DAGsHub (not GitHub!)
mlflow.set_tracking_uri('http://ec2-13-53-87-54.eu-north-1.compute.amazonaws.com:5000')

# Set MLflow experiment
mlflow.set_experiment('2nd_Experiment')

# Load the dataset
df = pd.read_csv(r'./data/classification_data_5000_records.csv')

# Split the dataset into features and target variable
X = df.drop("output", axis=1)
y = df["output"]

# Split into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.23, random_state=42)

# Initialize hyperparameters
n_estimators = 45
max_depth = 11

# Initialize and train the Random Forest classifier
clf = RandomForestClassifier(random_state=42, n_estimators=n_estimators, max_depth=max_depth)
clf.fit(X_train, y_train)

param_id={
    'n_estimators':[20,25,30,35,40,45],
    'max_depth':[10,12,13,1]
    
}
# Make predictions
gcv=GridSearchCV(estimator=clf,param_grid=param_id,cv=5,n_jobs=-1,verbose=1)

gcv.fit(X_train,y_train)

best_param=gcv.best_params_
best_score=gcv.best_score_


# Log metrics, parameters, and artifacts with MLflow
with mlflow.start_run(run_name='444'):
    mlflow.log_param('best_paramaters',best_param)
    mlflow.log_param('Accuracy',best_score)
    mlflow.log_artifact(__file__) 
    mlflow.set_tag('author','Shashank')
    mlflow.set_tag('model','Random Forest')
