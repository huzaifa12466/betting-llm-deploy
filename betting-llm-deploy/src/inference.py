from peft import PeftModel
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
import torch

# Base model (HuggingFace repo)
BASE_MODEL = "mistralai/Mistral-7B-Instruct-v0.1"

# LoRA adapter folder
LORA_WEIGHTS = "./model"

# Load tokenizer (LoRA ke saath)
tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)

# 4-bit quantization config
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16
)

# Load base model (quantized, memory-efficient)
model = AutoModelForCausalLM.from_pretrained(
    BASE_MODEL,
    device_map="auto",   # GPU ho to GPU, CPU ho to CPU
    quantization_config=bnb_config
)

# Load LoRA adapter
model = PeftModel.from_pretrained(model, LORA_WEIGHTS)
model.eval()

# Inference
def generate_answer(question: str, max_new_tokens: int = 200):
    prompt = f"Question: {question}\nAnswer:"
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    outputs = model.generate(**inputs, max_new_tokens=max_new_tokens)
    answer = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return answer.replace(prompt, "").strip()
