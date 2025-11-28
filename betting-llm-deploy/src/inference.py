import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig

# Model path (yahan fine-tuned model copy karo)
MODEL_PATH = "/app/model"


# Load tokenizer
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)

# Load model with 4-bit quantization
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16
)

model = AutoModelForCausalLM.from_pretrained(
    MODEL_PATH,
    quantization_config=bnb_config,
    device_map="auto"
)

def generate_answer(question: str, max_new_tokens: int = 200):
    """Generate answer from fine-tuned model"""
    prompt = f"Question: {question}\nAnswer:"
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    outputs = model.generate(**inputs, max_new_tokens=max_new_tokens)
    answer = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return answer.replace(prompt, "").strip()
