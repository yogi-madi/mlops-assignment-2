import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"  # Disable GPU usage in TensorFlow

print("test")

import pandas as pd
import time
import joblib
import mlflow
import mlflow.sklearn
import json
from evidently.report import Report
from evidently.metrics import DatasetDriftMetric
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

print("Loading datasets...")
# Load dataset
df_train = pd.read_csv("feature_engineering/fashion_mnist_train.csv")
df_test = pd.read_csv("feature_engineering/fashion_mnist_test.csv")

print("Loading best model...")
# Load best model
best_model = joblib.load("model_selection/tpot_model.pkl") 

# Split features and labels
X_train, y_train = df_train.iloc[:, :-1], df_train['label']
X_test, y_test = df_test.iloc[:, :-1], df_test['label']

# Create directories
os.makedirs("model_monitoring", exist_ok=True)

# ============================
# Model Performance Tracking
# ============================
print("Setting up MLflow experiment...")
# mlflow.set_tracking_uri("file:/home/vsdevops/mlruns")
# # mlflow.create_experiment("Fashion_MNIST_Tracking")

# with mlflow.start_run():

# Log model
print("Starting model inference...")
start_time = time.time()
y_pred = best_model.predict(X_test)
end_time = time.time()

accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred, average='weighted')
recall = recall_score(y_test, y_pred, average='weighted')
f1 = f1_score(y_test, y_pred, average='weighted')
inference_time = (end_time - start_time) / len(X_test)

# Save performance metrics
performance_metrics = {
"accuracy": accuracy,
"precision": precision,
"recall": recall,
"f1_score": f1,
"inference_time": inference_time
}

os.makedirs("model_monitoring", exist_ok=True)
os.chmod("model_monitoring", 0o755)
with open("model_monitoring/performance_metrics.json", "w") as f:
    json.dump(performance_metrics, f, indent=4)

print("Saved performance metrics to model_monitoring/performance_metrics.json")

mlflow.set_experiment("Fashion_MNIST_Tracking")
experiment = mlflow.get_experiment_by_name("Fashion_MNIST_Tracking")

with mlflow.start_run(experiment_id=experiment.experiment_id):
    # Log metrics
    mlflow.log_metric("accuracy", accuracy)
    mlflow.log_metric("precision", precision)
    mlflow.log_metric("recall", recall)
    mlflow.log_metric("f1_score", f1)
    mlflow.log_metric("inference_time", inference_time)
    mlflow.end_run()
        
print("Model performance logged successfully!")

# =======================
# Drift Detection
# =======================
print("Running Data Drift Detection...")

df_train_small = df_train.head(1000)
df_test_small = df_test.head(1000)

data_drift_report = Report(metrics=[DatasetDriftMetric()])
# data_drift_report.run(reference_data=df_train, current_data=df_test)
data_drift_report.run(reference_data=df_train_small, current_data=df_test_small)

# Save drift report
data_drift_report.save_html("model_monitoring/drift_report.html")

# Extract drift results
drift_results = data_drift_report.as_dict()

# Check if drift detected
if drift_results["metrics"][0]["result"]["dataset_drift"]:
    drift_status = "Drift detected! Consider retraining the model."
else:
    drift_status = "No significant drift detected."

# Save drift status
with open("model_monitoring/drift_status.txt", "w") as f:
    f.write(drift_status)

print(drift_status)
print("Drift detection completed. Check 'model_monitoring/drift_report.html' for details.")
