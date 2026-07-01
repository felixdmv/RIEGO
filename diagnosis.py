import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import PeftModel
import os

# Fuerza a usar la carpeta correcta del último entrenamiento
ADAPTER_PATH = "./agro_llama_8b_contrafactual" 
MODEL_ID = "meta-llama/Llama-3.1-8B-Instruct"

def test_inference_final():
    bnb_config = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype=torch.bfloat16)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, local_files_only=True)
    
    base_model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID, quantization_config=bnb_config, device_map={"": 0}, local_files_only=True
    )
    
    # CARGA MANUAL PARA ASEGURARNOS
    model = PeftModel.from_pretrained(base_model, ADAPTER_PATH)
    model.eval()

    # Escenario de test: Estrés hídrico real
    # Usamos exactamente el mismo formato que en el entrenamiento
    prompt = (
        "<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\nExpert Agro AI<|eot_id|>"
        "<|start_header_id|>user<|end_header_id|>\n\n"
        "Simula el impacto del riego y prescribe la dosis óptima: NDVI 0.20, ET0 7.5, Lluvia 0.0<|eot_id|>"
        "<|start_header_id|>assistant<|end_header_id|>\n\n"
    )

    inputs = tokenizer(prompt, return_tensors="pt").to("cuda")
    
    with torch.no_grad():
        outputs = model.generate(
            **inputs, 
            max_new_tokens=250,
            temperature=0.01, # Casi determinista para evitar divagaciones
            top_p=1.0,
            do_sample=False,
            repetition_penalty=1.2 # Evita que se buclee
        )
    
    print("--- RESPUESTA DEL MODELO ---")
    print(tokenizer.decode(outputs[0], skip_special_tokens=False))

test_inference_final()