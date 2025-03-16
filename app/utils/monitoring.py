import mlflow
import numpy as np
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix
import pandas as pd
import os
import json
from datetime import datetime

class ModelMonitor:
    def __init__(self, model_name="sentiment-roberta", tracking_uri=None):
        self.model_name = model_name
        if tracking_uri:
            mlflow.set_tracking_uri(tracking_uri)
        self.metrics_dir = "data/metrics"
        os.makedirs(self.metrics_dir, exist_ok=True)
        
    def start_run(self, run_name=None):
        """Start an MLflow run"""
        if not run_name:
            run_name = f"{self.model_name}-{datetime.now().strftime('%Y%m%d-%H%M%S')}"
        mlflow.start_run(run_name=run_name)
        return self
        
    def log_params(self, params):
        """Log parameters to MLflow"""
        for key, value in params.items():
            mlflow.log_param(key, value)
        return self
    
    def evaluate(self, y_true, y_pred, labels=None):
        """
        Evaluate model performance and log metrics
        
        Args:
            y_true (list): True labels
            y_pred (list): Predicted labels
            labels (list): List of label names
            
        Returns:
            dict: Dictionary containing evaluation metrics
        """
        # Calculate metrics
        accuracy = accuracy_score(y_true, y_pred)
        precision, recall, f1, _ = precision_recall_fscore_support(y_true, y_pred, average='weighted')
        
        # Log metrics to MLflow
        mlflow.log_metric("accuracy", accuracy)
        mlflow.log_metric("precision", precision)
        mlflow.log_metric("recall", recall)
        mlflow.log_metric("f1", f1)
        
        # Create confusion matrix
        cm = confusion_matrix(y_true, y_pred)
        
        # Save metrics locally
        metrics = {
            "accuracy": float(accuracy),
            "precision": float(precision),
            "recall": float(recall),
            "f1": float(f1),
            "confusion_matrix": cm.tolist(),
            "timestamp": datetime.now().isoformat()
        }
        
        self._save_metrics(metrics)
        
        return metrics
    
    def _save_metrics(self, metrics):
        """Save metrics to a JSON file"""
        timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        metrics_file = os.path.join(self.metrics_dir, f"metrics-{timestamp}.json")
        
        with open(metrics_file, "w") as f:
            json.dump(metrics, f, indent=2)
        
        # Also log to MLflow
        mlflow.log_artifact(metrics_file)
        
        return metrics_file
    
    def log_model(self, model):
        """Log model to MLflow"""
        mlflow.pytorch.log_model(model, "model")
        return self
    
    def end_run(self):
        """End the MLflow run"""
        mlflow.end_run()
        return self
    
    def get_latest_metrics(self):
        """Get the latest metrics"""
        metrics_files = [os.path.join(self.metrics_dir, f) for f in os.listdir(self.metrics_dir) 
                        if f.startswith("metrics-") and f.endswith(".json")]
        
        if not metrics_files:
            return None
        
        latest_file = max(metrics_files, key=os.path.getctime)
        
        with open(latest_file, "r") as f:
            metrics = json.load(f)
        
        return metrics
    
    def track_prediction_distribution(self, predictions, output_file=None):
        """
        Track the distribution of predictions
        
        Args:
            predictions (list): List of prediction dictionaries
            output_file (str): Path to save the distribution
            
        Returns:
            dict: Dictionary containing prediction distribution
        """
        labels = [pred["label"] for pred in predictions]
        unique_labels, counts = np.unique(labels, return_counts=True)
        
        distribution = {label: int(count) for label, count in zip(unique_labels, counts)}
        
        # Log to MLflow
        for label, count in distribution.items():
            mlflow.log_metric(f"prediction_count_{label}", count)
        
        # Save distribution
        if output_file:
            os.makedirs(os.path.dirname(output_file), exist_ok=True)
            with open(output_file, "w") as f:
                json.dump(distribution, f, indent=2)
        
        return distribution 