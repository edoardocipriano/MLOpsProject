import os
import argparse
import pandas as pd
import numpy as np
from app.models.sentiment_model import SentimentModel
from app.utils.monitoring import ModelMonitor
from app.utils.data_utils import load_dataset_from_csv, preprocess_text

def evaluate_model(data_dir="data", split="test"):
    """
    Evaluate the sentiment model on a dataset
    
    Args:
        data_dir (str): Directory containing the dataset
        split (str): Dataset split to evaluate on
        
    Returns:
        dict: Dictionary containing evaluation metrics
    """
    # Load dataset
    print(f"Loading {split} dataset from {data_dir}...")
    datasets = load_dataset_from_csv(data_dir)
    
    if split not in datasets:
        print(f"Split {split} not found in dataset")
        return None
    
    dataset = datasets[split]
    print(f"Dataset loaded successfully with {len(dataset)} examples")
    
    # Initialize model and monitor
    print("Loading model...")
    model = SentimentModel().load()
    monitor = ModelMonitor()
    
    # Start MLflow run
    monitor.start_run(run_name=f"evaluation-{split}")
    
    # Log parameters
    monitor.log_params({
        "model_name": model.model_name,
        "dataset_split": split,
        "dataset_size": len(dataset)
    })
    
    # Preprocess texts
    texts = dataset["text"].apply(preprocess_text).tolist()
    
    # Get true labels
    y_true = dataset["label"].tolist()
    
    # Make predictions
    print("Making predictions...")
    predictions = model.batch_predict(texts)
    y_pred = [model.labels.index(pred["label"]) for pred in predictions]
    
    # Evaluate model
    print("Evaluating model...")
    metrics = monitor.evaluate(y_true, y_pred, labels=list(model.labels.values()))
    
    # Track prediction distribution
    monitor.track_prediction_distribution(
        predictions, 
        output_file=f"data/distributions/{split}_distribution.json"
    )
    
    # End MLflow run
    monitor.end_run()
    
    print("Evaluation complete:")
    for metric, value in metrics.items():
        if metric not in ["confusion_matrix", "timestamp"]:
            print(f"  {metric}: {value:.4f}")
    
    return metrics

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate sentiment model")
    parser.add_argument("--data-dir", type=str, default="data", help="Data directory")
    parser.add_argument("--split", type=str, default="test", help="Dataset split to evaluate on")
    
    args = parser.parse_args()
    
    evaluate_model(args.data_dir, args.split) 