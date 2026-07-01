from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
import sys

MODEL_ID = "meta-llama/Llama-3.2-1B-Instruct"

try:
    print("Loading 1B tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, local_files_only=True)
    print("Tokenizer loaded successfully.")
    
    print("Loading 1B model on CUDA...")
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        torch_dtype=torch.float16,
        device_map="auto",
        local_files_only=True
    )
    print("1B model loaded successfully on CUDA!")
except Exception as e:
    print("Error occurred:")
    import traceback
    traceback.print_exc()
