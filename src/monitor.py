import pandas as pd
from evidently.report import Report
from evidently.metric_preset import DataDriftPreset

x_train = pd.read_csv("data/processed/x_train.csv")
x_test = pd.read_csv("data/processed/x_test.csv")

report = Report(metrics=[DataDriftPreset()])
report.run(reference_data=x_train, current_data=x_test)
report.save_html("reports/drift_report.html")
print("Drift raporu oluşturuldu: reports/drift_report.html")