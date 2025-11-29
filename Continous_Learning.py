import os
import json
import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling
)
from datasets import Dataset

# --- Configuration ---
MODEL_PATH = "./gpt2_chatbot_model"
FEEDBACK_FILE = "feedback.json"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

class PersonaBot:
    def __init__(self):
        if not os.path.exists(MODEL_PATH):
            raise FileNotFoundError(f"Model not found at {MODEL_PATH}. Run training first.")

        print(f"Loading model on {DEVICE}...")
        self.tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
        self.model = AutoModelForCausalLM.from_pretrained(MODEL_PATH).to(DEVICE)

    def generate(self, user_input):
        # Format input exactly like training data
        input_text = f"<usr> {user_input} <bot>"
        inputs = self.tokenizer(input_text, return_tensors="pt").to(DEVICE)

        with torch.no_grad():
            output_ids = self.model.generate(
                **inputs,
                max_new_tokens=60,
                pad_token_id=self.tokenizer.eos_token_id,
                do_sample=True,
                top_p=0.9,
                temperature=0.7
            )

        full_text = self.tokenizer.decode(output_ids[0], skip_special_tokens=True)

        # Extract the bot's part
        if "<bot>" in full_text:
            return full_text.split("<bot>")[-1].strip()
        return full_text

    def fine_tune_on_feedback(self, examples):
        """Continuous learning: fine-tune on good user interactions"""
        print("\n[Continuous Learning] Retraining on positive feedback...")
        self.model.train()

        # Create dataset from feedback
        dataset = Dataset.from_dict({"text": examples})

        def tokenize(ex):
            return self.tokenizer(ex["text"], truncation=True, padding="max_length", max_length=128)

        tokenized = dataset.map(tokenize, batched=True)
        tokenized.set_format("torch")

        # Lightweight training args
        args = TrainingArguments(
            output_dir=MODEL_PATH,
            num_train_epochs=3,
            per_device_train_batch_size=2,
            learning_rate=5e-5,
            save_steps=50,
            overwrite_output_dir=True
        )

        trainer = Trainer(
            model=self.model,
            args=args,
            train_dataset=tokenized,
            data_collator=DataCollatorForLanguageModeling(self.tokenizer, mlm=False)
        )

        trainer.train()
        trainer.save_model(MODEL_PATH)
        self.model.eval() # Switch back to eval mode
        print("[Continuous Learning] Model updated!")

def main():
    print("--- STEP 4: INTERACTIVE CHAT ---")

    try:
        bot = PersonaBot()
    except Exception as e:
        print(e)
        return

    print("\n💬 Chatbot is ready! Type 'exit' to quit.")

    while True:
        user_input = input("\nYou: ").strip()
        if user_input.lower() in ["exit", "quit"]:
            break

        response = bot.generate(user_input)
        print(f"Bot: {response}")

        # --- Continuous Learning Loop ---
        try:
            rating = input("(Optional) Rate reply 1-5 [Enter to skip]: ")
            if rating.isdigit():
                score = int(rating)

                # Save interaction
                interaction = {
                    "user": user_input,
                    "bot": response,
                    "score": score
                }

                history = []
                if os.path.exists(FEEDBACK_FILE):
                    with open(FEEDBACK_FILE, 'r') as f:
                        try: history = json.load(f)
                        except: pass

                history.append(interaction)
                with open(FEEDBACK_FILE, 'w') as f:
                    json.dump(history, f, indent=2)

                # Check triggers for retraining (e.g., if > 5 high quality samples)
                good_samples = [
                    f"<usr> {h['user']} <bot> {h['bot']} <|endoftext|>"
                    for h in history if h['score'] >= 4
                ]

                # Threshold set low (5) for demonstration
                if len(good_samples) >= 5:
                    bot.fine_tune_on_feedback(good_samples)
                    # Clear history after training (optional)
                    os.remove(FEEDBACK_FILE)

        except Exception as e:
            print(f"Error saving feedback: {e}")

if __name__ == "__main__":
    main()
