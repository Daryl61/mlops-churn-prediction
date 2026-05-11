import mlflow
import pandas as pd 
from sklearn.metrics import accuracy_score, classification_report, precision_score
import yaml

with open("configs/config.yaml", "r") as f:
    config = yaml.safe_load(f)


x_test = pd.read_csv("data/processed/x_test.csv")  
y_test = pd.read_csv("data/processed/y_test.csv")

y_test = y_test.values.ravel()

#mlflow ıcın bır ısımle acıyoruz
mlflow.set_experiment("churn-prediction")
 
#mlflow da son modeli yüklüyoruz
runs = mlflow.search_runs(order_by=["start_time DESC"])
last_run_id = runs.iloc[0]["run_id"]
model = mlflow.sklearn.load_model(f"runs:/{last_run_id}/random_forest_model")


y_pred = model.predict(x_test)
accuracy = accuracy_score(y_test, y_pred)
f1_score = classification_report(y_test, y_pred, output_dict=True)['1']['f1-score']
precision = precision_score(y_test, y_pred)
print(f"Accuracy: {accuracy}")
print(f"F1 Score: {f1_score}")  
print(f"Precision: {precision}")

if accuracy > config["thresholds"]["min_accuracy"]:
    print("Model dogrulugu istenilen seviyede.")
else:
    print("Model dogrulugu istenilen seviyede degil.")

if f1_score > config["thresholds"]["min_f1"]:
    print("Model f1 skoru istenilen seviyede.")
else:
    print("Model f1 skoru istenilen seviyede degil.")