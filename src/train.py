import pandas as pd
from mlflow import log_metric, log_params
import mlflow
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, precision_score
import yaml
with open("configs/config.yaml", "r") as f:
    config = yaml.safe_load(f)

x_train = pd.read_csv("data/processed/x_train.csv")
y_train = pd.read_csv("data/processed/y_train.csv")
x_test = pd.read_csv("data/processed/x_test.csv")
y_test = pd.read_csv("data/processed/y_test.csv")

y_train = y_train.values.ravel()
y_test = y_test.values.ravel()

rf = RandomForestClassifier(n_estimators=config["model"]["n_estimators"], 
                            random_state=config["model"]["random_state"],
                            min_samples_split=config["model"]["min_samples_split"],
                            min_samples_leaf=config["model"]["min_samples_leaf"],
                            max_features=config["model"]["max_features"]
                            )

mlflow.set_experiment("churn-prediction")
#mlfow da başlatıyoruz traini
with mlflow.start_run():
    rf.fit(x_train, y_train)
    y_pred = rf.predict(x_test)
    accuracy = accuracy_score(y_test, y_pred)
    f1_score = classification_report(y_test, y_pred, output_dict=True)['1']['f1-score']
    precision = precision_score(y_test, y_pred)
    log_metric("accuracy", accuracy)
    log_metric("f1_score", f1_score)
    log_metric("precision", precision)
    mlflow.log_params(config["model"])
    mlflow.sklearn.log_model(rf, "random_forest_model")