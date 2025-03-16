import pandas as pd
from datasets import load_dataset
import os
import json

def load_sample_dataset(dataset_name="tweet_eval", subset="sentiment"):
    """
    Load a sample dataset from HuggingFace datasets
    
    Args:
        dataset_name (str): Name of the dataset
        subset (str): Subset of the dataset
        
    Returns:
        dict: Dictionary containing train, validation, and test datasets
    """
    try:
        dataset = load_dataset(dataset_name, subset)
        return dataset
    except Exception as e:
        print(f"Error loading dataset: {e}")
        return None

def save_dataset_to_csv(dataset, output_dir="data"):
    """
    Save a HuggingFace dataset to CSV files
    
    Args:
        dataset (Dataset): HuggingFace dataset
        output_dir (str): Directory to save the CSV files
        
    Returns:
        dict: Dictionary containing paths to the saved files
    """
    os.makedirs(output_dir, exist_ok=True)
    
    paths = {}
    for split in dataset.keys():
        df = pd.DataFrame(dataset[split])
        output_path = os.path.join(output_dir, f"{split}.csv")
        df.to_csv(output_path, index=False)
        paths[split] = output_path
    
    return paths

def load_dataset_from_csv(data_dir="data"):
    """
    Load dataset from CSV files
    
    Args:
        data_dir (str): Directory containing the CSV files
        
    Returns:
        dict: Dictionary containing train, validation, and test datasets
    """
    dataset = {}
    for split in ["train", "validation", "test"]:
        file_path = os.path.join(data_dir, f"{split}.csv")
        if os.path.exists(file_path):
            dataset[split] = pd.read_csv(file_path)
    
    return dataset

def preprocess_text(text):
    """
    Preprocess text for sentiment analysis
    
    Args:
        text (str): Input text
        
    Returns:
        str: Preprocessed text
    """
    if not text:
        return ""
    
    # Convert to lowercase
    text = text.lower()
    
    # Remove extra whitespace
    text = " ".join(text.split())
    
    return text

def save_predictions(texts, predictions, output_file="data/predictions.json"):
    """
    Save predictions to a JSON file
    
    Args:
        texts (list): List of input texts
        predictions (list): List of prediction dictionaries
        output_file (str): Path to the output file
    """
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    
    results = []
    for text, pred in zip(texts, predictions):
        results.append({
            "text": text,
            "prediction": pred
        })
    
    with open(output_file, "w") as f:
        json.dump(results, f, indent=2)
    
    return output_file 