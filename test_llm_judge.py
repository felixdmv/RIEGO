import os
import json
import torch
import re
import subprocess
import numpy as np
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import PeftModel

os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.environ["HF_DATASETS_OFFLINE"] = "1"

# --- 1. CONFIGURACIÓN DE ENTORNO PARA ESTABILIDAD EN CLÚSTER ---
os.environ["NCCL_P2P_DISABLE"] = "1"
os.environ["NCCL_IB_DISABLE"] = "1"
os.environ["NCCL_SHM_DISABLE"] = "1"
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

# --- 2. VARIABLES GLOBALES ---
MODEL_ID = "meta-llama/Llama-3.1-8B-Instruct"
ADAPTER_PATH = "./agro_llama_8b_final_cot"  # Ruta del modelo CoT que acabas de entrenar
TOKEN = "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"

# Escenarios de prueba (Los 20 casos de tesis)
test_scenarios = [
    {"ndvi": 0.20, "et0": 7.5, "lluvia": 0.0, "desc": "Estrés hídrico severo"},
    {"ndvi": 0.45, "et0": 4.0, "lluvia": 2.0, "desc": "Condiciones moderadas con algo de lluvia"},
    {"ndvi": 0.80, "et0": 5.5, "lluvia": 0.0, "desc": "Vigor alto, demanda alta"},
    {"ndvi": 0.60, "et0": 3.0, "lluvia": 10.0, "desc": "Vigor medio, lluvia abundante"},
    {"ndvi": 0.30, "et0": 8.0, "lluvia": 0.0, "desc": "Suelo seco, ola de calor"},
    {"ndvi": 0.50, "et0": 6.0, "lluvia": 5.0, "desc": "Condiciones mixtas con lluvia moderada"},
    {"ndvi": 0.10, "et0": 9.0, "lluvia": 0.0, "desc": "Estrés hídrico extremo"},
    {"ndvi": 0.70, "et0": 4.5, "lluvia": 0.0, "desc": "Vigor alto, sin lluvia"},
    {"ndvi": 0.55, "et0": 5.0, "lluvia": 3.0, "desc": "Vigor medio-alto con algo de lluvia"},
    {"ndvi": 0.25, "et0": 7.0, "lluvia": 1.0, "desc": "Estrés hídrico moderado con poca lluvia"},
    {"ndvi": 0.65, "et0": 4.0, "lluvia": 0.0, "desc": "Vigor alto, demanda moderada"},
    {"ndvi": 0.40, "et0": 6.5, "lluvia": 0.0, "desc": "Vigor medio, alta demanda"},
    {"ndvi": 0.75, "et0": 3.5, "lluvia": 8.0, "desc": "Vigor alto con lluvia significativa"},
    {"ndvi": 0.35, "et0": 7.0, "lluvia": 2.0, "desc": "Estrés hídrico moderado con algo de lluvia"},
    {"ndvi": 0.85, "et0": 4.5, "lluvia": 0.0, "desc": "Vigor muy alto sin lluvia"},
    {"ndvi": 0.50, "et0": 5.5, "lluvia": 4.0, "desc": "Vigor medio con lluvia moderada"},
    {"ndvi": 0.15, "et0": 8.5, "lluvia": 0.0, "desc": "Estrés hídrico severo con ola de calor"},
    {"ndvi": 0.60, "et0": 4.0, "lluvia": 0.0, "desc": "Vigor medio-alto sin lluvia"},
    {"ndvi": 0.45, "et0": 6.0, "lluvia": 1.0, "desc": "Vigor medio con algo de lluvia"},
    {"ndvi": 0.80, "et0": 5.0, "lluvia": 0.0, "desc": "Vigor alto sin lluvia"}
]

# --- 3. CARGA DE MODELOS ---
def load_evaluation_system():
    print("Iniciando carga de modelos para evaluación...")
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True
    )

    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, token=TOKEN, local_files_only=True)
    tokenizer.pad_token = tokenizer.eos_token

    # Cargamos Generador (FT + Base) en GPU 0
    print("Cargando Generadores en GPU 0...")
    base_m = AutoModelForCausalLM.from_pretrained(
        MODEL_ID, 
        quantization_config=bnb_config, 
        device_map={"": 0}, 
        token=TOKEN,
        local_files_only=True # <--- IMPORTANTE
    )
    ft_m = PeftModel.from_pretrained(base_m, ADAPTER_PATH)
    
    # Cargamos Juez en GPU 1
    print("Cargando Juez en GPU 1...")
    judge_m = AutoModelForCausalLM.from_pretrained(
        MODEL_ID, quantization_config=bnb_config, device_map={"": 1}, token=TOKEN
    )

    return tokenizer, ft_m, judge_m

def get_response(model, tokenizer, prompt, dev_id):
    inputs = tokenizer(prompt, return_tensors="pt").to(f"cuda:{dev_id}")
    with torch.no_grad():
        outputs = model.generate(
            **inputs, 
            max_new_tokens=256, # Aumentado para permitir el razonamiento CoT
            temperature=0.1, 
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id
        )
    # Extraer solo la parte del asistente
    decoded = tokenizer.decode(outputs[0], skip_special_tokens=True)
    if "assistant" in decoded:
        return decoded.split("assistant")[-1].strip()
    return decoded.strip()

# --- 4. LÓGICA DEL JUEZ EXPERTO ---
def judge_logic(judge_model, tokenizer, scenario, resp_base, resp_ft):
    prompt_judge = f"""<|begin_of_text|>system
Eres un Auditor Senior de Inteligencia Artificial Agrícola. Tu misión es evaluar qué respuesta es más útil para un agricultor.

ESCENARIO TÉCNICO:
- Contexto: {scenario['desc']}
- Datos: NDVI {scenario['ndvi']}, ET0 {scenario['et0']}mm, Lluvia {scenario['lluvia']}mm.

RESPUESTA A (Modelo Base):
"{resp_base}"

RESPUESTA B (Modelo Fine-Tuned):
"{resp_ft}"

REGLAS DE EVALUACIÓN:
1. PRESCRIPCIÓN: ¿Da una cifra clara de riego (ej: "X mm")? Si no hay número, la respuesta es nula.
2. RAZONAMIENTO: ¿Explica el cálculo de Kc y la demanda hídrica?
3. VERBOSIDAD: Penaliza explicaciones teóricas largas que no lleguen a una conclusión numérica.

Responde ÚNICAMENTE en formato JSON:
{{
    "winner": "A" o "B",
    "score_A": 1-10,
    "score_B": 1-10,
    "reason": "Explicación técnica de la decisión."
}}
<|start_header_id|>assistant<|end_header_id|>"""
    
    return get_response(judge_model, tokenizer, prompt_judge, dev_id=1)

# --- 5. EJECUCIÓN DEL BENCHMARK ---
if __name__ == "__main__":
    # Limpiar procesos previos
    subprocess.run(["rm", "-rf", os.path.expanduser("~/.triton/cache")])
    
    tok, ft_model, judge_model = load_evaluation_system()
    
    results = []
    print(f"\nEvaluando {len(test_scenarios)} escenarios...")

    for i, sc in enumerate(tqdm(test_scenarios)):
        # Prompt de entrada idéntico al entrenamiento
        prompt = (
            f"<|begin_of_text|>system\nExpert Agro AI\n"
            f"user\nCalcula la prescripción de riego. NDVI: {sc['ndvi']}, ET0: {sc['et0']}mm, Lluvia: {sc['lluvia']}mm.\n"
            f"assistant\n"
        )
        
        # 1. Obtener respuesta del Modelo Base (Desactivando adaptadores LoRA)
        with ft_model.disable_adapter():
            r_base = get_response(ft_model, tok, prompt, dev_id=0)
            
        # 2. Obtener respuesta del Modelo Fine-Tuned (Con CoT)
        r_ft = get_response(ft_model, tok, prompt, dev_id=0)
        
        # 3. Obtener juicio del LLM
        eval_json_raw = judge_logic(judge_model, tok, sc, r_base, r_ft)
        
        # Intentar parsear el JSON de la evaluación
        try:
            # Limpieza básica para asegurar que solo tenemos el JSON
            json_str = eval_json_raw[eval_json_raw.find('{'):eval_json_raw.rfind('}')+1]
            evaluation = json.loads(json_str)
        except:
            evaluation = {"winner": "Error", "reason": eval_json_raw}

        results.append({
            "id": i,
            "escenario": sc['desc'],
            "input": f"NDVI: {sc['ndvi']}, ET0: {sc['et0']}, Lluvia: {sc['lluvia']}",
            "respuesta_base": r_base,
            "respuesta_ft": r_ft,
            "evaluacion": evaluation
        })

    # Guardar resultados finales
    output_file = "evaluacion_final_cot.json"
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=4, ensure_ascii=False)

    # Mostrar resumen
    victorias_ft = sum(1 for r in results if r['evaluacion'].get('winner') == 'B')
    print(f"\n" + "="*30)
    print(f"RESUMEN DE EVALUACIÓN")
    print(f"="*30)
    print(f"Total Escenarios: {len(test_scenarios)}")
    print(f"Victorias Fine-Tuned (B): {victorias_ft}")
    print(f"Victorias Base (A): {len(test_scenarios) - victorias_ft}")
    print(f"Archivo guardado: {output_file}")
    print("="*30)