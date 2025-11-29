# ChatBot2 — GPT‑2 FastAPI Chatbot

A complete, end‑to‑end chatbot project that fine‑tunes GPT‑2 on the Persona‑Chat dataset and serves an interactive web UI via FastAPI. The repository includes data preprocessing, model training, evaluation, and an optional continuous‑learning loop that incorporates user feedback.

## 1. Project Description

This project provides:
- Data preprocessing for Persona‑Chat into dialogue pairs suitable for causal language modeling
- Fine‑tuning of `gpt2` using Hugging Face `transformers`
- Evaluation with BLEU, ROUGE, BERTScore, and perplexity, plus a metrics plot
- A FastAPI backend exposing a `/chat` endpoint
- A simple, responsive web frontend (login + chat interface)
- An optional continuous‑learning script to incrementally improve the model from rated interactions

Key components:
- `Preprocessing.py`: cleans, tokenizes, and writes datasets to disk
- `Train.py`: trains GPT‑2 on the processed dataset and saves the model
- `Evaluate.py`: computes metrics and exports results/plots
- `Continous_Learning.py`: CLI chatbot with feedback‑driven fine‑tuning
- `app/`: FastAPI app, model loading, and static frontend

## 2. Installation

Prerequisites:
- Python 3.10+ (recommended)
- `pip` and a virtual environment tool
- Optional: CUDA‑enabled GPU and recent PyTorch build for faster training/inference

Steps (Windows):

```bash
# Install dependencies
pip install -r requirements.txt
```

Dataset preprocessing:

```bash
python Preprocessing.py
# Outputs tokenized datasets under ./processed_dataset
```

Model training:

```bash
# Ensure Train.py points to the processed dataset directory
# DATA_PATH should be ./processed_dataset
python Train.py
# Saves the trained model to ./gpt2_chatbot_model
```

Evaluation (optional):

```bash
python Evaluate.py
# Writes results to ./results and saves a metrics plot
```

## 3. Usage

Run the web app:

```bash
# From the project root, start the FastAPI app from the app directory
cd app
uvicorn main:app --reload --port 8000
```

Then open `http://localhost:8000` in your browser. You’ll see an Arabic login page; enter a name to proceed to the chat. Type messages in the chat box; the bot responds via the fine‑tuned model.

API example (POST `/chat`):

```bash
curl -X POST http://localhost:8000/chat \
  -H "Content-Type: application/json" \
  -d "{\"message\": \"Hello!\"}"
```

Response format:

```json
{ "response": "Hi! How can I help you today?" }
```

Direct Python usage:

```python
from app.model import generate_response

print(generate_response("How are you?"))
```

## 4. Configuration

Training and data:
- `Preprocessing.py`: `MODEL_NAME`, `OUTPUT_DIR`, `MAX_LENGTH`
- `Train.py`: `DATA_PATH` (set to `./processed_dataset`), `MODEL_OUTPUT_DIR`, `BASE_MODEL`, `EPOCHS`, `BATCH_SIZE`

Evaluation:
- `Evaluate.py`: `MODEL_PATH` (default `./gpt2_chatbot_model`), `DATA_PATH`, `RESULTS_DIR`

Continuous learning:
- `Continous_Learning.py`: `MODEL_PATH`, `FEEDBACK_FILE`, `DEVICE`

Web backend model location:
- `app/model.py` loads the model from `MODEL_PATH = Path(__file__).parent / "gpt-chatbot"`
- If you trained to `./gpt2_chatbot_model`, either:
  - Copy the trained folder to `app/gpt-chatbot`, or
  - Update `MODEL_PATH` in `app/model.py` to point to the trained directory

Server:
- Start with `uvicorn main:app --port 8000` from the `app` directory to ensure static files resolve