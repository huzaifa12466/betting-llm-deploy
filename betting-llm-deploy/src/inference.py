from peft import PeftModel
from transformers import AutoTokenizer, AutoConfig, AutoModelForCausalLM
import torch

# LoRA adapter folder
LORA_WEIGHTS = "./model"

# Load tokenizer (LoRA adapter ke saath same tokenizer use karo)
tokenizer = AutoTokenizer.from_pretrained(LORA_WEIGHTS)

# Load model config directly from adapter
config = AutoConfig.from_pretrained(LORA_WEIGHTS)

# Initialize model from config (random weights)
model = AutoModelForCausalLM.from_config(config)

# Load LoRA adapter weights
model = PeftModel.from_pretrained(model, LORA_WEIGHTS)
model.eval()

# Inference function
def generate_answer(question: str, max_new_tokens: int = 200):
    prompt = f"Question: {question}\nAnswer:"
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    outputs = model.generate(**inputs, max_new_tokens=max_new_tokens)
    answer = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return answer.replace(prompt, "").strip()

