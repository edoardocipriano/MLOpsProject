import pytest
from fastapi.testclient import TestClient
import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.main import app

client = TestClient(app)

def test_root_endpoint():
    """Test the root endpoint"""
    response = client.get("/api")
    assert response.status_code == 200
    assert response.json() == {"message": "Sentiment Analysis API is running"}

def test_predict_endpoint_positive():
    """Test the predict endpoint with positive text"""
    response = client.post(
        "/predict",
        json={"text": "I love this product! It's amazing."}
    )
    assert response.status_code == 200
    data = response.json()
    assert data["text"] == "I love this product! It's amazing."
    assert "label" in data
    assert "score" in data
    assert "scores" in data
    assert set(data["scores"].keys()) == {"positive", "neutral", "negative"}

def test_predict_endpoint_negative():
    """Test the predict endpoint with negative text"""
    response = client.post(
        "/predict",
        json={"text": "This is terrible. I hate it."}
    )
    assert response.status_code == 200
    data = response.json()
    assert data["text"] == "This is terrible. I hate it."
    assert "label" in data
    assert "score" in data
    assert "scores" in data
    assert set(data["scores"].keys()) == {"positive", "neutral", "negative"}

def test_batch_predict_endpoint():
    """Test the batch predict endpoint"""
    response = client.post(
        "/batch-predict",
        json={
            "texts": [
                "I love this product!",
                "This is terrible.",
                "This is a factual statement."
            ]
        }
    )
    assert response.status_code == 200
    data = response.json()
    assert "results" in data
    assert len(data["results"]) == 3
    
    for result in data["results"]:
        assert "text" in result
        assert "label" in result
        assert "score" in result
        assert "scores" in result
        assert set(result["scores"].keys()) == {"positive", "neutral", "negative"}

def test_metrics_endpoint():
    """Test the metrics endpoint"""
    response = client.get("/metrics")
    assert response.status_code == 200
    
    # Note: This might return {"message": "No metrics available"} if no metrics have been logged yet
    data = response.json()
    assert data is not None 