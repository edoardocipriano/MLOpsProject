import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer
import numpy as np

class SentimentModel:
    def __init__(self, model_name="cardiffnlp/twitter-roberta-base-sentiment-latest"):
        self.model_name = model_name
        self.tokenizer = None
        self.model = None
        self.labels = {0: "negative", 1: "neutral", 2: "positive"}
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
    def load(self):
        """Load the model and tokenizer"""
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(self.model_name)
        self.model.to(self.device)
        self.model.eval()
        return self
    
    def predict(self, text):
        """
        Predict sentiment for a given text
        
        Args:
            text (str): Input text for sentiment analysis
            
        Returns:
            dict: Dictionary containing sentiment label and scores
        """
        if not self.model or not self.tokenizer:
            self.load()
            
        # Tokenize input
        inputs = self.tokenizer(text, return_tensors="pt", truncation=True, max_length=512)
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        # Make prediction
        with torch.no_grad():
            outputs = self.model(**inputs)
            scores = outputs.logits.squeeze().cpu().numpy()
            scores = torch.nn.functional.softmax(torch.tensor(scores), dim=0).numpy()
            
        # Get prediction
        prediction_id = np.argmax(scores)
        label = self.labels[prediction_id]
        
        # Format results
        results = {
            "label": label,
            "score": float(scores[prediction_id]),
            "scores": {self.labels[i]: float(score) for i, score in enumerate(scores)}
        }
        
        return results
    
    def batch_predict(self, texts):
        """
        Predict sentiment for a batch of texts
        
        Args:
            texts (list): List of input texts
            
        Returns:
            list: List of dictionaries containing sentiment labels and scores
        """
        return [self.predict(text) for text in texts] 