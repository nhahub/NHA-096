import os
import json
import torch
import numpy as np
import pandas as pd
import evaluate
import matplotlib.pyplot as plt
from tqdm import tqdm
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, AutoModelForCausalLM
from datasets import load_from_disk

# --- Configuration ---
MODEL_PATH = "./gpt2_chatbot_model"
DATA_PATH = "./processed_dataset"
RESULTS_DIR = "./results"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Special Tokens
BOT_TOKEN = "<bot>"
USR_TOKEN = "<usr>"
EOS = "<|endoftext|>"

def calculate_perplexity(model, dataset):
    """
    Calculates the Cross Entropy Loss and Perplexity.
    Returns: (mean_loss, perplexity)
    """
    model.eval()
    dataloader = DataLoader(dataset, batch_size=4)
    losses = []

    print("Calculating Perplexity...")
    with torch.no_grad():
        for batch in tqdm(dataloader):
            input_ids = batch['input_ids'].to(DEVICE)
            attention_mask = batch['attention_mask'].to(DEVICE)
            labels = batch['labels'].to(DEVICE)

            outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
            losses.append(outputs.loss.item())

    mean_loss = np.mean(losses)
    return mean_loss, np.exp(mean_loss)

def generate_predictions(model, tokenizer, dataset, num_samples=100):
    """
    Generates model responses for evaluation.
    Returns: (prompts_list, predictions_list, references_list)
    """
    print(f"Generating predictions for {num_samples} samples...")
    model.eval()

    prompts = []
    references = []
    predictions = []

    # 1. Extract Prompts and References from encoded dataset
    count = 0
    # Limit dataset scan to find enough valid samples
    for i in range(len(dataset)):
        if count >= num_samples: break

        ids = dataset[i]['input_ids']
        text = tokenizer.decode(ids, skip_special_tokens=False)

        if BOT_TOKEN in text:
            parts = text.split(BOT_TOKEN)
            # Prompt is before <bot>, strip <usr>
            prompt_clean = parts[0].replace(USR_TOKEN, "").strip()
            # Reference is after <bot>, strip <|endoftext|>
            ref_clean = parts[1].replace(EOS, "").strip()

            if prompt_clean and ref_clean:
                prompts.append(prompt_clean)
                references.append(ref_clean)
                count += 1

    # 2. Inference Loop
    for prompt in tqdm(prompts, desc="Inference"):
        input_text = f"{USR_TOKEN} {prompt} {BOT_TOKEN}"
        inputs = tokenizer(input_text, return_tensors="pt").to(DEVICE)

        with torch.no_grad():
            output_ids = model.generate(
                **inputs,
                max_new_tokens=50,
                pad_token_id=tokenizer.eos_token_id,
                do_sample=True,
                top_p=0.9
            )

        full_output = tokenizer.decode(output_ids[0], skip_special_tokens=True)

        # Extract only the bot's response
        try:
            bot_reply = full_output.split(BOT_TOKEN)[1].strip()
        except IndexError:
            bot_reply = full_output # Fallback if tokens malformed

        predictions.append(bot_reply)

    return prompts, predictions, references

def main():
    print("--- STEP 3: EVALUATION ---")

    if not os.path.exists(MODEL_PATH):
        print(f"❌ Model not found at {MODEL_PATH}. Run 02_train.py first.")
        return

    # Ensure results directory exists
    os.makedirs(RESULTS_DIR, exist_ok=True)

    # Load resources
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
    model = AutoModelForCausalLM.from_pretrained(MODEL_PATH).to(DEVICE)
    dataset = load_from_disk(DATA_PATH)["test"]

    # 1. Calculate Perplexity
    eval_loss, ppl = calculate_perplexity(model, dataset)
    print(f"\n📊 Eval Loss: {eval_loss:.4f} | Perplexity: {ppl:.4f}\n")

    # 2. Generate Predictions
    prompts, preds, refs = generate_predictions(model, tokenizer, dataset, num_samples=100)

    # 3. Load Metrics
    print("Loading metrics...")
    sacrebleu = evaluate.load("sacrebleu")
    rouge = evaluate.load("rouge")
    bertscore = evaluate.load("bertscore")

    # 4. Compute Metrics
    print("Computing scores...")
    bleu_res = sacrebleu.compute(predictions=preds, references=[[r] for r in refs])
    rouge_res = rouge.compute(predictions=preds, references=refs)
    bertscore_res = bertscore.compute(predictions=preds, references=refs, lang="en")

    # 5. Extract values
    bleu_score = bleu_res["score"]
    rouge1 = float(rouge_res.get("rouge1", 0.0))
    rouge2 = float(rouge_res.get("rouge2", 0.0))
    rougeL = float(rouge_res.get("rougeL", 0.0))

    # Bertscore returns a list for every sample, take mean
    bert_precision = float(np.mean(bertscore_res["precision"]))
    bert_recall = float(np.mean(bertscore_res["recall"]))
    bert_f1 = float(np.mean(bertscore_res["f1"]))

    # 6. Print Results
    print("\n" + "="*30)
    print("📈 FINAL EVALUATION METRICS")
    print("="*30)
    print(f"BLEU:        {bleu_score:.2f}")
    print(f"ROUGE-1:     {rouge1:.4f}")
    print(f"ROUGE-2:     {rouge2:.4f}")
    print(f"ROUGE-L:     {rougeL:.4f}")
    print(f"BERT Pre:    {bert_precision:.4f}")
    print(f"BERT Rec:    {bert_recall:.4f}")
    print(f"BERT F1:     {bert_f1:.4f}")
    print(f"Perplexity:  {ppl:.4f}")
    print("="*30)

    # 7. Save Results to JSON
    results_path = os.path.join(RESULTS_DIR, "evaluation_results.json")
    results_data = {
        "eval_loss": eval_loss,
        "perplexity": ppl,
        "BLEU": bleu_score,
        "ROUGE": {"rouge1": rouge1, "rouge2": rouge2, "rougeL": rougeL},
        "BERTScore": {"precision": bert_precision, "recall": bert_recall, "f1": bert_f1},
    }
    with open(results_path, "w", encoding="utf-8") as f:
        json.dump(results_data, f, indent=4, ensure_ascii=False)
    print(f"📁 Saved evaluation results → {results_path}")

    # 8. Save Predictions to CSV
    pred_output_path = os.path.join(RESULTS_DIR, "predictions.csv")
    df = pd.DataFrame({
        "prompt": prompts,
        "reference": refs,
        "prediction": preds
    })
    df.to_csv(pred_output_path, index=False, encoding="utf-8")
    print(f"📄 Saved predictions → {pred_output_path}")

    # 9. Generate and Save Plot
    plot_path = os.path.join(RESULTS_DIR, "eval_metrics_plot.png")

    metrics_labels = ["BLEU (0-100)", "ROUGE-1", "ROUGE-2", "ROUGE-L", "BERTScore-F1", "Perplexity(inv)"]

    # Scale scores for visualization consistency (0-100 scale)
    # Note: Perplexity is inverted (1/ppl * 100) just for the bar chart to make 'higher better' visually
    values_display = [
        bleu_score,
        rouge1 * 100,
        rouge2 * 100,
        rougeL * 100,
        bert_f1 * 100,
        (1.0 / ppl) * 100 if ppl > 0 else 0
    ]

    plt.figure(figsize=(12, 6))
    bars = plt.bar(metrics_labels, values_display,
                    color=["#2b8cbe", "#7bccc4", "#bae4bc", "#edf8b1", "#fdae61", "#d73027"])

    plt.ylim(0, max(max(values_display) * 1.2, 10)) # Ensure some height

    for bar, label, real_val in zip(bars, metrics_labels,
                                    [bleu_score, rouge1, rouge2, rougeL, bert_f1, ppl]):
        height = bar.get_height()
        if label == "Perplexity(inv)":
            text = f"ppl={real_val:.2f}"
        elif label == "BLEU (0-100)":
            text = f"{real_val:.2f}"
        else:
            text = f"{real_val:.4f}"

        plt.text(bar.get_x() + bar.get_width() / 2, height + 1, text,
                 ha="center", fontsize=10)

    plt.title("Chatbot Evaluation Metrics")
    plt.savefig(plot_path, dpi=300, bbox_inches="tight")
    print(f"📊 Saved plot → {plot_path}")

if __name__ == "__main__":
    main()
