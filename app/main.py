import os
from fastapi import FastAPI, HTTPException, BackgroundTasks
from pydantic import BaseModel
from typing import List, Optional
import uvicorn
from app.models.sentiment_model import SentimentModel
from app.utils.data_utils import preprocess_text, save_predictions
from app.utils.monitoring import ModelMonitor
from app.components import static

# Initialize FastAPI app
app = FastAPI(
    title="Sentiment Analysis API",
    description="API for sentiment analysis using RoBERTa model",
    version="1.0.0"
)

# Include routers
app.include_router(static.router)

# Initialize model
model = SentimentModel()
monitor = ModelMonitor()

# Define request and response models
class TextRequest(BaseModel):
    text: str

class BatchRequest(BaseModel):
    texts: List[str]

class SentimentResponse(BaseModel):
    text: str
    label: str
    score: float
    scores: dict

class BatchResponse(BaseModel):
    results: List[SentimentResponse]

# Load model on startup
@app.on_event("startup")
async def startup_event():
    model.load()
    print("Model loaded successfully")

# Define routes
@app.get("/api")
async def root():
    return {"message": "Sentiment Analysis API is running"}

@app.post("/predict", response_model=SentimentResponse)
async def predict(request: TextRequest, background_tasks: BackgroundTasks):
    try:
        # Preprocess text
        processed_text = preprocess_text(request.text)
        
        # Make prediction
        prediction = model.predict(processed_text)
        
        # Track prediction in background
        background_tasks.add_task(
            monitor.track_prediction_distribution, 
            [prediction], 
            "data/distributions/single_prediction.json"
        )
        
        # Return response
        return SentimentResponse(
            text=request.text,
            label=prediction["label"],
            score=prediction["score"],
            scores=prediction["scores"]
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/batch-predict", response_model=BatchResponse)
async def batch_predict(request: BatchRequest, background_tasks: BackgroundTasks):
    try:
        # Preprocess texts
        processed_texts = [preprocess_text(text) for text in request.texts]
        
        # Make predictions
        predictions = model.batch_predict(processed_texts)
        
        # Track predictions in background
        background_tasks.add_task(
            monitor.track_prediction_distribution, 
            predictions, 
            "data/distributions/batch_prediction.json"
        )
        
        # Save predictions
        background_tasks.add_task(
            save_predictions,
            request.texts,
            predictions
        )
        
        # Return response
        results = [
            SentimentResponse(
                text=text,
                label=pred["label"],
                score=pred["score"],
                scores=pred["scores"]
            )
            for text, pred in zip(request.texts, predictions)
        ]
        
        return BatchResponse(results=results)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/metrics")
async def get_metrics():
    try:
        metrics = monitor.get_latest_metrics()
        if not metrics:
            return {"message": "No metrics available"}
        return metrics
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    uvicorn.run("app.main:app", host="0.0.0.0", port=8000, reload=True) 