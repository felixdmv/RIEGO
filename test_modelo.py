import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import PeftModel

# --- CONFIGURACIÓN ---
base_model_id = "meta-llama/Llama-3.1-8B-Instruct"
adapter_path = "./agro_llama_8b_final"
token = "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"

def load_models():
    print("Cargando modelos para comparativa...")
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
    )

    tokenizer = AutoTokenizer.from_pretrained(base_model_id, token=token)
    tokenizer.pad_token = tokenizer.eos_token

    # 1. Cargar Modelo Base
    base_model = AutoModelForCausalLM.from_pretrained(
        base_model_id, 
        quantization_config=bnb_config, 
        device_map="auto", 
        token=token
    )

    # 2. Cargar Modelo Fine-Tuned (Base + Adaptadores LoRA)
    ft_model = PeftModel.from_pretrained(base_model, adapter_path)
    ft_model.eval()
    
    return tokenizer, base_model, ft_model

def generate_response(model, tokenizer, prompt):
    inputs = tokenizer(prompt, return_tensors="pt").to("cuda")
    with torch.no_grad():
        outputs = model.generate(
            **inputs, 
            max_new_tokens=128, 
            temperature=0.1, # Temperatura baja para evaluación técnica
            do_sample=False
        )
    return tokenizer.decode(outputs[0], skip_special_tokens=True)

# --- EJECUCIÓN DE PRUEBA ---
tokenizer, base_model, ft_model = load_models()

# Test case: Un escenario de estrés hídrico que el modelo no vio en el entrenamiento
test_prompt = (
    "<|begin_of_text|>system\nExpert Agro AI\nuser\n"
    "Calcula la prescripción de riego. NDVI: 0.35, ET0: 6.2mm, Lluvia: 0.0mm.\n"
    "assistant\n"
)

print("\n" + "="*50)
print("RESULTADO MODELO BASE (LLAMA 3.1):")
# Para el base, quitamos los adaptadores temporalmente
with ft_model.disable_adapter():
    print(generate_response(ft_model, tokenizer, test_prompt))

print("\n" + "="*50)
print("RESULTADO MODELO FINE-TUNED (AGRO-INTELLIGENCE):")
print(generate_response(ft_model, tokenizer, test_prompt))
print("="*50)