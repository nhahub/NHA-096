import os
import re
import shutil
from datasets import load_dataset, DatasetDict, Dataset
from transformers import AutoTokenizer

# --- Configuration ---
MODEL_NAME = "gpt2"
OUTPUT_DIR = "./processed_dataset"
MAX_LENGTH = 128

# Special Tokens
EOS = "<|endoftext|>"
USR_TOKEN = "<usr>"
BOT_TOKEN = "<bot>"

def clean_text(text):
    """Basic text cleaning."""
    if not isinstance(text, str): return ""
    text = text.lower()
    text = re.sub(r"http\S+|www\S+", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text

def add_special_tokens(text):
    """Replaces Persona tags with special tokens."""
    if not isinstance(text, str): return text
    text = re.sub(r'Persona\s*[a]\s*', f'{USR_TOKEN} ', text, flags=re.IGNORECASE)
    text = re.sub(r'Persona\s*[b]\s*', f'{BOT_TOKEN} ', text, flags=re.IGNORECASE)
    return text

def dialog_to_pairs(example):
    """Converts dialogue history into training pairs."""
    dialog = example.get("dialogue", [])
    cleaned_turns = []

    for turn in dialog:
        t = clean_text(turn)
        t = add_special_tokens(t)
        if t:
            cleaned_turns.append(t)

    pairs = []
    for i in range(0, len(cleaned_turns) - 1, 2):
        # Format: <usr> msg <bot> msg <|endoftext|>
        pair = f"{cleaned_turns[i]} {cleaned_turns[i+1]} {EOS}"
        pairs.append(pair)

    return {"text": pairs}

def main():
    print("--- STEP 1: PREPROCESSING ---")

    # 1. Load Data
    print("Downloading Persona-Chat dataset...")
    dataset = load_dataset("Cynaptics/persona-chat")

    # 2. Split Data
    print("Splitting dataset...")
    temp_split = dataset['train'].train_test_split(test_size=0.1, seed=42)
    val_test_split = temp_split['test'].train_test_split(test_size=0.5, seed=42)

    datasets = DatasetDict({
        'train': temp_split['train'],
        'validation': val_test_split['train'],
        'test': val_test_split['test']
    })

    # 3. Text Processing
    print("Cleaning and formatting text...")
    processed_datasets = datasets.map(
        dialog_to_pairs,
        remove_columns=datasets['train'].column_names,
        batched=False
    )

    # 4. Flattening (Unnesting list of texts)
    def flatten_data(batch):
        return {"text": [item for sublist in batch["text"] for item in sublist]}

    # We treat the dataset as a list of texts now
    flat_datasets = DatasetDict()
    for split in processed_datasets.keys():
        all_texts = [t for sublist in processed_datasets[split]["text"] for t in sublist]
        flat_datasets[split] = Dataset.from_dict({"text": all_texts})

    # 5. Tokenization
    print(f"Tokenizing using {MODEL_NAME} tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    tokenizer.pad_token = tokenizer.eos_token

    def tokenize_function(examples):
        return tokenizer(
            examples["text"],
            truncation=True,
            padding="max_length",
            max_length=MAX_LENGTH
        )

    tokenized_datasets = flat_datasets.map(tokenize_function, batched=True)
    tokenized_datasets = tokenized_datasets.remove_columns(["text"])

    # Set labels (for Causal LM, labels = input_ids)
    tokenized_datasets = tokenized_datasets.map(lambda x: {"labels": x["input_ids"]}, batched=True)
    tokenized_datasets.set_format("torch")

    # 6. Save to Disk
    if os.path.exists(OUTPUT_DIR):
        shutil.rmtree(OUTPUT_DIR)

    tokenized_datasets.save_to_disk(OUTPUT_DIR)
    print(f"✅ Preprocessing complete. Data saved to '{OUTPUT_DIR}'")

if __name__ == "__main__":
    main()
