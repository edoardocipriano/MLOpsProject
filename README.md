# Sentiment Analysis with RoBERTa

This project implements a sentiment analysis model using the CardiffNLP RoBERTa model for classifying text into positive, neutral, or negative sentiment.

## Project Structure

- `app/`: Main application code
  - `components/`: UI components
  - `models/`: Model implementation
  - `utils/`: Utility functions
- `data/`: Dataset storage
- `tests/`: Test files
- `.github/workflows/`: CI/CD pipeline configuration

## Setup

1. Clone the repository
2. Install dependencies:
   ```
   pip install -r requirements.txt
   ```
3. Run the application:
   ```
   python app/main.py
   ```

## Features

- Sentiment analysis using pre-trained RoBERTa model
- FastAPI web interface
- Continuous integration and deployment
- Model performance monitoring

## Model

This project uses the [twitter-roberta-base-sentiment-latest](https://huggingface.co/cardiffnlp/twitter-roberta-base-sentiment-latest) model from CardiffNLP on HuggingFace. 