import os
import argparse
from app.utils.data_utils import load_sample_dataset, save_dataset_to_csv

def prepare_dataset(dataset_name="tweet_eval", subset="sentiment", output_dir="data"):
    """
    Download and prepare a dataset for sentiment analysis
    
    Args:
        dataset_name (str): Name of the dataset
        subset (str): Subset of the dataset
        output_dir (str): Directory to save the dataset
        
    Returns:
        dict: Dictionary containing paths to the saved files
    """
    print(f"Loading dataset {dataset_name}/{subset}...")
    dataset = load_sample_dataset(dataset_name, subset)
    
    if not dataset:
        print("Failed to load dataset")
        return None
    
    print(f"Dataset loaded successfully with splits: {', '.join(dataset.keys())}")
    
    # Print dataset statistics
    for split, data in dataset.items():
        print(f"  {split}: {len(data)} examples")
    
    # Save dataset to CSV
    print(f"Saving dataset to {output_dir}...")
    paths = save_dataset_to_csv(dataset, output_dir)
    
    print("Dataset saved successfully:")
    for split, path in paths.items():
        print(f"  {split}: {path}")
    
    return paths

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Prepare dataset for sentiment analysis")
    parser.add_argument("--dataset", type=str, default="tweet_eval", help="Dataset name")
    parser.add_argument("--subset", type=str, default="sentiment", help="Dataset subset")
    parser.add_argument("--output-dir", type=str, default="data", help="Output directory")
    
    args = parser.parse_args()
    
    prepare_dataset(args.dataset, args.subset, args.output_dir) 