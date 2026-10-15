import mlflow

mlflow.set_tracking_uri("http://13.221.201.200:5000")
mlflow.set_experiment("windows-local-test")

with mlflow.start_run():
    mlflow.log_param("model", "RandomForest")
    mlflow.log_metric("accuracy", 0.92)

print("Experiment logged successfully!")