import yaml
from fastapi.testclient import TestClient
config  = yaml.safe_load(open("configs/config.yaml", "r"))
import sys
sys.path.insert(0, ".")



def test_api_healt():
    from api.app import app
    client = TestClient(app)

    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}

def test_api_model_info():
    from api.app import app
    client = TestClient(app)
    response = client.get("/model-info")
    assert response.status_code == 200
    assert "model_info" in response.json() or "error" in response.json(), "Model infosu yanlış"

    
def test_api_predict():
    from api.app import app
    client = TestClient(app)
    sample_data = {
        "SeniorCitizen": 0,
    "tenure": 12,
    "MonthlyCharges": 50.0,
    "TotalCharges": 600.0,
    "gender_Male": 1,
    "Partner_Yes": 0,
    "Dependents_Yes": 0,
    "PhoneService_Yes": 1,
    "MultipleLines_No phone service": 0,
    "MultipleLines_Yes": 0,
    "InternetService_Fiber optic": 1,
    "InternetService_No": 0,
    "OnlineSecurity_No internet service": 0,
    "OnlineSecurity_Yes": 0,
    "OnlineBackup_No internet service": 0,
    "OnlineBackup_Yes": 1,
    "DeviceProtection_No internet service": 0,
    "DeviceProtection_Yes": 0,
    "TechSupport_No internet service": 0,
    "TechSupport_Yes": 0,
    "Contract_One year": 1,
    "Contract_Two year": 0,
    "PaperlessBilling_Yes": 1,
    "PaymentMethod_Credit card (automatic)": 0,
    "PaymentMethod_Electronic check": 1,
    "PaymentMethod_Mailed check": 0 

    }
    response = client.post("/predict", json=sample_data)
    assert response.status_code == 200, "predict hatalı sonuç  döndürüyor"   
