import pandas as pd
import mlflow
from sklearn.metrics import f1_score
import yaml
config  = yaml.safe_load(open("configs/config.yaml", "r"))

x_test = pd.read_csv("data/processed/x_test.csv")
y_test = pd.read_csv("data/processed/y_test.csv")
y_test = y_test.values.ravel()
mlflow.set_experiment("churn-prediction") 
runs = mlflow.search_runs(order_by=["start_time DESC"])
last_run_id = runs.iloc[0]["run_id"]
model = mlflow.sklearn.load_model(f"runs:/{last_run_id}/random_forest_model")
y_pred = model.predict(x_test)

def test_mlflow_model_accuracy():

    accuracy = (y_pred == y_test).mean()
    assert accuracy > config["thresholds"]["min_accuracy"], f"Model doğruluğu {accuracy} istenilen seviyede değil"

def test_mlflow_model_f1():

    f1 = f1_score(y_test, y_pred, average='weighted')
    assert f1 > config["thresholds"]["min_f1"], f"Model F1 skoru {f1} istenilen seviyede değil"