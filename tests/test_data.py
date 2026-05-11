import pandas as pd
import yaml
config  = yaml.safe_load(open("configs/config.yaml", "r"))

x_train = pd.read_csv("data/processed/x_train.csv")
y_train = pd.read_csv("data/processed/y_train.csv")
x_test = pd.read_csv("data/processed/x_test.csv")
y_test = pd.read_csv("data/processed/y_test.csv")

def test_data():
    df=pd.read_csv(config["data"]["raw_path"] + "/customer.csv")
    assert df.shape[0] > 0, "Dataframe boş"
    assert 'Churn' in df.columns, "Churn bulanamıyor"

def test_no_nulls_in_processed():
   
    assert x_train.isnull().sum().sum() == 0, "İşlenmiş veride null var"

def test_target_binary():
 
    unique_vals = y_train.iloc[:, 0].unique()
    assert set(unique_vals) == {0, 1}, "Target sadece 0 ve 1 olmalı"

def test_train_test_shape():
   
    assert x_train.shape[1] == x_test.shape[1], "Train ve test sütun sayısı farklı"    



    