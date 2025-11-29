from transformers import GPT2LMHeadModel, GPT2Tokenizer

MODEL_PATH = "gpt2_chatbot_model"  

print("Loading tokenizer...")
tokenizer = GPT2Tokenizer.from_pretrained(MODEL_PATH)

print("Loading model...")
model = GPT2LMHeadModel.from_pretrained(MODEL_PATH)

if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

user_input = "Hello, how are you?"
print("You:", user_input)

encoded = tokenizer(user_input, return_tensors='pt', padding=True)
inputs = encoded['input_ids']
attention_mask = encoded['attention_mask']
reply = model.generate(
    inputs,
    attention_mask=attention_mask,
    max_length=50,
    pad_token_id=tokenizer.eos_token_id,
    do_sample=True,
    top_p=0.95,
    temperature=0.7
)
response = tokenizer.decode(reply[:, inputs.shape[-1]:][0], skip_special_tokens=True)
print("Bot:", response)