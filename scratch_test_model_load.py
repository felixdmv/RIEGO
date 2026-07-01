from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import PeftModel
import torch
import sys

print("Python version:", sys.version)
print("CUDA available:", torch.cuda.is_available())

MODEL_ID = "meta-llama/Llama-3.1-8B-Instruct"
ADAPTER_PATH = "./agro_llama_8b_contrafactual"

try:
    print("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, local_files_only=True)
    print("Tokenizer loaded successfully.")
    
    print("Loading model in 4-bit...")
    bnb_config = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype=torch.bfloat16)
    base_model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        quantization_config=bnb_config,
        device_map="auto",
        local_files_only=True
    )
    print("Base model loaded successfully.")
    
    print("Loading Peft adapter...")
    model = PeftModel.from_pretrained(base_model, ADAPTER_PATH)
    print("Adapter loaded successfully!")
    
except Exception as e:
    print("Error occurred:")
    import traceback
    traceback.print_exc()
