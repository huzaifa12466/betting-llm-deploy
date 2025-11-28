from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import PeftModel
import torch

# Base model (HuggingFace repo)
BASE_MODEL = "mistralai/Mistral-7B-Instruct-v0.1"

# LoRA adapter folder (container ke andar)
LORA_WEIGHTS = "./model"

# Load tokenizer from base model
tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)

# Load base model with 4-bit quantization
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16
)

model = AutoModelForCausalLM.from_pretrained(
    BASE_MODEL,
    quantization_config=bnb_config,
    device_map="auto"
)

# Load LoRA adapter
model = PeftModel.from_pretrained(model, LORA_WEIGHTS)
model.eval()

# Function to generate answer
def generate_answer(question: str, max_new_tokens: int = 200):
    prompt = f"Question: {question}\nAnswer:"
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    outputs = model.generate(**inputs, max_new_tokens=max_new_tokens)
    answer = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return answer.replace(prompt, "").strip()

