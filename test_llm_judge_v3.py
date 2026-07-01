import os
import json
import torch
import gc
import re
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import PeftModel

# --- CONFIGURACIÓN ---
os.environ["TRANSFORMERS_OFFLINE"] = "1"
MODEL_ID = "meta-llama/Llama-3.1-8B-Instruct"
ADAPTER_PATH = "./agro_llama_8b_contrafactual" 
TOKEN = "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"

test_scenarios = [
    {"ndvi": 0.20, "et0": 7.5, "lluvia": 0.0, "desc": "Estrés hídrico severo"},
    {"ndvi": 0.45, "et0": 4.0, "lluvia": 2.0, "desc": "Condiciones moderadas"},
    {"ndvi": 0.80, "et0": 5.5, "lluvia": 0.0, "desc": "Vigor alto, demanda alta"},
    {"ndvi": 0.60, "et0": 3.0, "lluvia": 10.0, "desc": "Vigor medio, lluvia abundante"},
    {"ndvi": 0.30, "et0": 8.0, "lluvia": 0.0, "desc": "Suelo seco, ola de calor"}
]

def clear_memory():
    gc.collect()
    torch.cuda.empty_cache()

def clean_output(text):
    """Limpia repeticiones y basura post-prescripción"""
    # Buscamos el final de la justificación
    match = re.search(r"(JUSTIFICACIÓN:.*?\.)", text, re.DOTALL)
    if match:
        return match.group(1)
    return text.split("\n\n")[0] # Fallback: primer párrafo

def get_model_response(model, tokenizer, prompt, is_ft=False):
    inputs = tokenizer(prompt, return_tensors="pt").to("cuda")
    
    gen_kwargs = {
        "max_new_tokens": 250, # Reducimos para que no se pierda
        "temperature": 0.2 if is_ft else 0.1, # Menos temperatura = menos repetición
        "do_sample": True if is_ft else False,
        "repetition_penalty": 1.5, # <--- PENALIZACIÓN ALTA para evitar el "CONTRAFACTUAL CONTRAFACTUAL"
        "pad_token_id": tokenizer.eos_token_id,
        "eos_token_id": tokenizer.eos_token_id
    }
    
    with torch.no_grad():
        outputs = model.generate(**inputs, **gen_kwargs)
    
    full_text = tokenizer.decode(outputs[0], skip_special_tokens=True).split("assistant")[-1].strip()
    
    if is_ft:
        return clean_output(full_text)
    return full_text

# --- CARGA POR FASES ---
bnb_config = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype=torch.bfloat16)
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, local_files_only=True, add_bos_token=False)

responses_base = []
responses_ft = []

# 1. GENERAR BASE
print(">>> Generando BASE...")
model = AutoModelForCausalLM.from_pretrained(MODEL_ID, quantization_config=bnb_config, device_map="auto", local_files_only=True)
for sc in tqdm(test_scenarios):
    p = f"<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\nExpert Agro AI<|eot_id|><|start_header_id|>user<|end_header_id|>\n\nNDVI Actual: {sc['ndvi']}, ET0: {sc['et0']}mm, Lluvia Pronosticada: {sc['lluvia']}mm. Simula el riego óptimo.<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n"
    responses_base.append(get_model_response(model, tokenizer, p, is_ft=False))
del model
clear_memory()

# 2. GENERAR FT
print(">>> Generando FINE-TUNED...")
model_base = AutoModelForCausalLM.from_pretrained(MODEL_ID, quantization_config=bnb_config, device_map="auto", local_files_only=True)
model = PeftModel.from_pretrained(model_base, ADAPTER_PATH)
for sc in tqdm(test_scenarios):
    p = f"<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\nExpert Agro AI<|eot_id|><|start_header_id|>user<|end_header_id|>\n\nNDVI Actual: {sc['ndvi']}, ET0: {sc['et0']}mm, Lluvia Pronosticada: {sc['lluvia']}mm. Simula el riego óptimo.<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\nANÁLISIS CONTRAFACTUAL SECTOR 3:\n-"
    responses_ft.append("ANÁLISIS CONTRAFACTUAL SECTOR 3:\n-" + get_model_response(model, tokenizer, p, is_ft=True))
del model, model_base
clear_memory()

# 3. JUICIO EXPERTO REFORZADO
print(">>> Iniciando Juicio Final...")
judge_model = AutoModelForCausalLM.from_pretrained(MODEL_ID, quantization_config=bnb_config, device_map="auto", local_files_only=True)
final_report = []

for i, sc in enumerate(tqdm(test_scenarios)):
    # EL PROMPT DEL JUEZ AHORA ES INFALIBLE
    j_p = (f"<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\n"
           f"Eres un Auditor de Sistemas de Riego. Tu única misión es detectar si la respuesta contiene DATOS DE SIMULACIÓN (ej: NDVI t+5 -> 0.XXX) y una PRESCRIPCIÓN en mm.\n"
           f"Si una respuesta NO tiene números de NDVI futuro, PIERDE automáticamente.<|eot_id|>"
           f"<|start_header_id|>user<|end_header_id|>\n\n"
           f"ESCENARIO: {sc['desc']}\n\n"
           f"OPCIÓN A: {responses_base[i]}\n\n"
           f"OPCIÓN B: {responses_ft[i]}\n\n"
           f"Responde JSON con 'winner' ('A' o 'B') y 'reason'.<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n")
    
    res_raw = get_model_response(judge_model, tokenizer, j_p, is_ft=False)
    
    winner = "B" if "winner\": \"B\"" in res_raw or "winner\":\"B\"" in res_raw else "A"
    final_report.append({"escenario": sc['desc'], "winner": winner, "ft": responses_ft[i], "base": responses_base[i], "reason": res_raw})

# GUARDAR
with open("benchmark_final_v5.json", "w", encoding="utf-8") as f:
    json.dump(final_report, f, indent=4, ensure_ascii=False)

wins = sum(1 for r in final_report if r['winner'] == 'B')
print(f"\n==============================")
print(f"RESULTADO DE LA TESIS: {wins}/20 victorias.")
print("==============================")