import pandas as pd
from evidently.report import Report
from evidently.metric_preset import DataDriftPreset


# Load Datasets

reference_data = pd.read_csv(
    "data/reference.csv"
)

current_data = pd.read_csv(
    "data/current.csv"
)

print("Datasets Loaded Successfully")

# Create Report

report = Report(
    metrics=[
        DataDriftPreset()
    ]
)

print("Report Object Created")

# Compare Datasets


report.run(
    reference_data=reference_data,
    current_data=current_data
)

print("Comparison Completed")

# Save HTML Report

report.save_html(
    "reports/model_drift_report.html"
)

print("HTML Report Saved Successfully")