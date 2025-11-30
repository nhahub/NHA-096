from pathlib import Path
from transformers import GPT2LMHeadModel, GPT2Tokenizer

MODEL_PATH = Path(__file__).parent / "gpt2_chatbot_model"

tokenizer = GPT2Tokenizer.from_pretrained(MODEL_PATH, local_files_only=True)
model = GPT2LMHeadModel.from_pretrained(MODEL_PATH, local_files_only=True)

if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

def generate_response(user_input: str) -> str:
    prompt = f"User: {user_input}\nBot:"
    encoded = tokenizer(prompt, return_tensors='pt', padding=True)
    inputs = encoded['input_ids']
    attention_mask = encoded['attention_mask']

    reply = model.generate(
        inputs,
        attention_mask=attention_mask,
        max_length=inputs.shape[1] + 25,
        pad_token_id=tokenizer.eos_token_id,
        do_sample=True,
        top_p=0.8,
        temperature=0.3  
    )

    response = tokenizer.decode(reply[0], skip_special_tokens=True)
    if "Bot:" in response:
        response = response.split("Bot:")[-1].strip()
    response = response.split("User:")[0].strip()
    return response or "i don't understand your question."