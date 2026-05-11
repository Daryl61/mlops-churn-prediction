from fastapi import FastAPI
import mlflow
import pandas as pd

mlflow.set_experiment("churn-prediction")
runs = mlflow.search_runs(order_by=["start_time DESC"])
last_run_id = runs.iloc[0]["run_id"]
model = mlflow.sklearn.load_model(f"runs:/{last_run_id}/random_forest_model")

from pydantic import BaseModel, Field
#burda giriş verilerini kullanıcdan almak için bir pydantic model oluşturuyoruz
class customerdata(BaseModel):
    SeniorCitizen: int
    tenure: int
    MonthlyCharges: float
    TotalCharges: float
    gender_Male: int
    Partner_Yes: int
    Dependents_Yes: int
    PhoneService_Yes: int
    MultipleLines_No_phone_service: int = Field(alias="MultipleLines_No phone service")
    MultipleLines_Yes: int
    InternetService_Fiber_optic: int = Field(alias="InternetService_Fiber optic")
    InternetService_No: int
    OnlineSecurity_No_internet_service: int = Field(alias="OnlineSecurity_No internet service")
    OnlineSecurity_Yes: int
    OnlineBackup_No_internet_service: int = Field(alias="OnlineBackup_No internet service")
    OnlineBackup_Yes: int
    DeviceProtection_No_internet_service: int = Field(alias="DeviceProtection_No internet service")
    DeviceProtection_Yes: int
    TechSupport_No_internet_service: int = Field(alias="TechSupport_No internet service")
    TechSupport_Yes: int
    Contract_One_year: int = Field(alias="Contract_One year")
    Contract_Two_year: int = Field(alias="Contract_Two year")
    PaperlessBilling_Yes: int
    PaymentMethod_Credit_card: int = Field(alias="PaymentMethod_Credit card (automatic)")
    PaymentMethod_Electronic_check: int = Field(alias="PaymentMethod_Electronic check")
    PaymentMethod_Mailed_check: int = Field(alias="PaymentMethod_Mailed check")

    class Config:
        populate_by_name = True

    
app = FastAPI()

@app.get("/health")
def health():
    return {"status": "ok"}

#predict bu kullanıcdan alacagımız veriyi modele verip tahmini geri çeviricek 
@app.post("/predict")
def predict(data: customerdata):
    customer=pd.DataFrame([data.model_dump(by_alias=True)])
    y_pred = model.predict(customer)
    return {"prediction": int(y_pred[0])}


#kullandıgımız parametre ve model bılgılerını kullancıyı gösterıyoruz
@app.get("/model-info")
def model_info():
    model_info = mlflow.get_run(last_run_id).data.params
    return {"model_info": model_info}