import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

model_id = "meta-llama/Llama-3.2-1B-Instruct"
adapter_id = "./agro_llm_final"

print(">>> Cargando modelo base + tu conocimiento especializado...")
tokenizer = AutoTokenizer.from_pretrained(model_id)
base_model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=torch.float16, device_map="auto")

# Unimos el modelo base con tu entrenamiento
model = PeftModel.from_pretrained(base_model, adapter_id)

def preguntar_a_la_ia(ndvi, temp, lluvia_prev):
    prompt = f"I: Eres el Orquestador Agrónomo. Decide el riego.\nInput: Satelite: NDVI {ndvi}. Clima: Temp {temp}, Lluvia prevista {lluvia_prev}mm.\nAns:"
    
    inputs = tokenizer(prompt, return_tensors="pt").to("cuda")
    outputs = model.generate(**inputs, max_new_tokens=100)
    print("\n--- RESPUESTA DE TU IA ---")
    print(tokenizer.decode(outputs[0], skip_special_tokens=True).split("Ans:")[-1])

# PRUEBA DE FUEGO
preguntar_a_la_ia(0.25, "35.5 (std)", 0.0)
preguntar_a_la_ia(0.75, "22.0 (std)", 5.0)
preguntar_a_la_ia(0.50, "28.0 (std)", 2.0)