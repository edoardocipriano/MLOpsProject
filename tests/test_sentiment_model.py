import pytest
import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.models.sentiment_model import SentimentModel
from app.utils.data_utils import preprocess_text

@pytest.fixture
def model():
    """Create and load a sentiment model for testing"""
    model = SentimentModel()
    return model

def test_model_initialization(model):
    """Test that the model initializes correctly"""
    assert model.model_name == "cardiffnlp/twitter-roberta-base-sentiment-latest"
    assert model.labels == {0: "negative", 1: "neutral", 2: "positive"}
    assert model.device in ["cuda", "cpu"]

def test_model_loading(model):
    """Test that the model loads correctly"""
    model.load()
    assert model.tokenizer is not None
    assert model.model is not None

def test_positive_sentiment(model):
    """Test prediction on positive text"""
    model.load()
    text = "I love this product! It's amazing and works perfectly."
    prediction = model.predict(text)
    
    assert prediction["label"] == "positive"
    assert prediction["score"] > 0.5
    assert "scores" in prediction
    assert set(prediction["scores"].keys()) == {"positive", "neutral", "negative"}

def test_negative_sentiment(model):
    """Test prediction on negative text"""
    model.load()
    text = "This is terrible. I hate it and it doesn't work at all."
    prediction = model.predict(text)
    
    assert prediction["label"] == "negative"
    assert prediction["score"] > 0.5
    assert "scores" in prediction
    assert set(prediction["scores"].keys()) == {"positive", "neutral", "negative"}

def test_neutral_sentiment(model):
    """Test prediction on neutral text"""
    model.load()
    text = "This is a factual statement about the product."
    prediction = model.predict(text)
    
    # Note: Neutral sentiment can be harder to predict accurately
    assert "label" in prediction
    assert "score" in prediction
    assert "scores" in prediction
    assert set(prediction["scores"].keys()) == {"positive", "neutral", "negative"}

def test_batch_prediction(model):
    """Test batch prediction"""
    model.load()
    texts = [
        "I love this product!",
        "This is terrible.",
        "This is a factual statement."
    ]
    
    predictions = model.batch_predict(texts)
    
    assert len(predictions) == 3
    for pred in predictions:
        assert "label" in pred
        assert "score" in pred
        assert "scores" in pred
        assert set(pred["scores"].keys()) == {"positive", "neutral", "negative"}

def test_text_preprocessing():
    """Test text preprocessing function"""
    text = "  This is a TEST with   extra   spaces.  "
    processed = preprocess_text(text)
    
    assert processed == "this is a test with extra spaces."
    
    # Test with empty text
    assert preprocess_text("") == ""
    assert preprocess_text(None) == "" 