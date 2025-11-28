from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
import torch
from transformers import BitsAndBytesConfig

BASE_MODEL = "mistralai/Mistral-7B-Instruct-v0.1"  # HuggingFace repo
LORA_WEIGHTS = "/app/model"  # Adapter folder

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

# Load LoRA weights on top of base model
model = PeftModel.from_pretrained(model, LORA_WEIGHTS)
model.eval()  # Set model to evaluation mode
