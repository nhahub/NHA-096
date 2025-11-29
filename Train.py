import torch
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    TrainingArguments,
    Trainer
)
from datasets import load_from_disk

# --- Configuration ---
DATA_PATH = "./data/processed_dataset"
MODEL_OUTPUT_DIR = "./gpt2_chatbot_model"
BASE_MODEL = "gpt2"
EPOCHS = 6
BATCH_SIZE = 4

def main():
    print("--- STEP 2: TRAINING ---")

    # 1. Load Processed Data
    print(f"Loading data from {DATA_PATH}...")
    try:
        datasets = load_from_disk(DATA_PATH)
    except FileNotFoundError:
        print("❌ Data not found. Please run 01_preprocess.py first.")
        return

    # 2. Load Model & Tokenizer
    print(f"Loading base model: {BASE_MODEL}...")
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
    tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(BASE_MODEL)

    # 3. Setup Training
    training_args = TrainingArguments(
        output_dir=MODEL_OUTPUT_DIR,
        num_train_epochs=EPOCHS,
        per_device_train_batch_size=BATCH_SIZE,
        per_device_eval_batch_size=BATCH_SIZE,
        warmup_steps=500,
        weight_decay=0.01,
        logging_dir='./logs',
        logging_steps=100,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        fp16=torch.cuda.is_available()  # Use GPU if available
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=datasets["train"],
        eval_dataset=datasets["validation"],
    )

    # 4. Train
    print("Starting training loop...")
    trainer.train()

    # 5. Save Final Model
    print(f"Saving model to {MODEL_OUTPUT_DIR}...")
    trainer.save_model(MODEL_OUTPUT_DIR)
    tokenizer.save_pretrained(MODEL_OUTPUT_DIR)

    print("✅ Training complete.")

if __name__ == "__main__":
    main()
