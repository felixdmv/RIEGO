import os
import json
import torch
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import PeftModel

# --- CONFIGURACIÓN OFFLINE ---
os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.environ["HF_DATASETS_OFFLINE"] = "1"
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1" # Usaremos dos GPUs para ir rápido

MODEL_ID = "meta-llama/Llama-3.1-8B-Instruct"
ADAPTER_PATH = "./agro_llama_8b_contrafactual"
TOKEN = "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"

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

def load_models():
    print("Cargando modelos para evaluación contrafactual...")
    bnb_config = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype=torch.bfloat16)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, token=TOKEN, local_files_only=True)
    tokenizer.pad_token = tokenizer.eos_token
    
    # Generador en GPU 0
    base_m = AutoModelForCausalLM.from_pretrained(MODEL_ID, quantization_config=bnb_config, device_map={"": 0}, local_files_only=True)
    ft_model = PeftModel.from_pretrained(base_m, ADAPTER_PATH)
    
    # Juez en GPU 1
    judge_m = AutoModelForCausalLM.from_pretrained(MODEL_ID, quantization_config=bnb_config, device_map={"": 1}, local_files_only=True)
    
    return tokenizer, ft_model, judge_m

def get_response(model, tokenizer, prompt, dev_id):
    inputs = tokenizer(prompt, return_tensors="pt").to(f"cuda:{dev_id}")
    with torch.no_grad():
        outputs = model.generate(**inputs, max_new_tokens=256, temperature=0.1, do_sample=False)
    return tokenizer.decode(outputs[0], skip_special_tokens=True).split("assistant")[-1].strip()

def judge_contrafactual(judge_model, tokenizer, scenario, resp_base, resp_ft):
    prompt_judge = f"""<|begin_of_text|>system
Eres un Revisor Científico de Agricultura de Precisión. Evalúa cuál de estas dos respuestas ofrece un mejor SOPORTE A LA DECISIÓN basado en simulaciones.

DATOS: NDVI {scenario['ndvi']}, ET0 {scenario['et0']}mm, Lluvia {scenario['lluvia']}mm.

Respuesta A (Base): "{resp_base}"
Respuesta B (IA Contrafactual): "{resp_ft}"

RÚBRICA DE EVALUACIÓN:
1. ¿Incluye un Análisis Contrafactual (qué pasa si no riego vs si riego)?
2. ¿Cuantifica el impacto en el vigor futuro (NDVI t+5)?
3. ¿La prescripción final es clara y está justificada numéricamente?

Responde solo JSON:
{{"winner": "A" o "B", "reason": "breve explicación técnica"}}
<|start_header_id|>assistant<|end_header_id|>"""
    return get_response(judge_model, tokenizer, prompt_judge, dev_id=1)

if __name__ == "__main__":
    tok, ft_m, judge_m = load_models()
    results = []
    
    print("\nEvaluando escenarios contrafactuales...")
    for sc in tqdm(test_scenarios):
        p = f"<|begin_of_text|>system\nExpert Agro AI\nuser\nSimula el impacto del riego y prescribe la dosis óptima: NDVI {sc['ndvi']}, ET0 {sc['et0']}, Lluvia {sc['lluvia']}\nassistant\n"
        
        # 1. Base
        with ft_m.disable_adapter():
            r_base = get_response(ft_m, tok, p, dev_id=0)
        
        # 2. Fine-Tuned (Contrafactual)
        r_ft = get_response(ft_m, tok, p, dev_id=0)
        
        # 3. Juicio
        eval_json = judge_contrafactual(judge_m, tok, sc, r_base, r_ft)
        
        try:
            evaluation = json.loads(eval_json[eval_json.find('{'):eval_json.rfind('}')+1])
        except:
            evaluation = {"winner": "Error", "reason": eval_json}

        results.append({
            "escenario": sc['desc'],
            "base": r_base,
            "ft": r_ft,
            "evaluacion": evaluation
        })

    with open("evaluacion_contrafactual_final.json", "w", encoding="utf-8") as f:
        json.dump(results, f, indent=4, ensure_ascii=False)
    
    print("\n--- RESUMEN FINAL ---")
    wins_ft = sum(1 for r in results if r['evaluacion'].get('winner') == 'B')
    print(f"Victorias Agente Contrafactual: {wins_ft}/20")
    print("Resultados guardados en evaluacion_contrafactual_final.json")