import os
import pandas as pd
from sklearn.model_selection import train_test_split
import yaml

os.makedirs("data/processed", exist_ok=True)

with open("configs/config.yaml", "r") as f:
    config = yaml.safe_load(f)

df = pd.read_csv(config["data"]["raw_path"] + "/customer.csv")

fd = pd.to_numeric(df['TotalCharges'], errors='coerce')
df['TotalCharges'] = fd
df.dropna(inplace=True)

df = df.drop(columns=['customerID', 'StreamingMovies', 'StreamingTV'])
# hedef degiskeni 1 ve 0 yapıyoruz
df['Churn'] = df['Churn'].map({'Yes': 1, 'No': 0})

df = pd.get_dummies(df, columns=['gender', 'Partner', 'Dependents', 'PhoneService',
                                  'MultipleLines', 'InternetService', 'OnlineSecurity',
                                  'OnlineBackup',
                                  'DeviceProtection', 'TechSupport', 'Contract',
                                  'PaperlessBilling', 'PaymentMethod'], drop_first=True)

x = df.drop(columns=['Churn'])
y = df['Churn']

x_train, x_test, y_train, y_test = train_test_split(
    x, y,
    test_size=config["data"]["test_size"],
    random_state=config["model"]["random_state"]
)

#bunları artık sureklı kullnacagımız için bölerek kaydediyoruz
x_train.to_csv("data/processed/x_train.csv", index=False)
x_test.to_csv("data/processed/x_test.csv", index=False)
y_train.to_csv("data/processed/y_train.csv", index=False)
y_test.to_csv("data/processed/y_test.csv", index=False)