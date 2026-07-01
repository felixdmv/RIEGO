import os
import shutil
import glob

base_path = r"C:\Users\fdemiguel\OneDrive - Universidad de Burgos\Documentos\RIEGO"

# Define directories
notebooks_dir = os.path.join(base_path, "notebooks")
models_dir = os.path.join(base_path, "models")

# Create directories if they do not exist
os.makedirs(notebooks_dir, exist_ok=True)
os.makedirs(models_dir, exist_ok=True)

# 1. Move Notebooks
notebooks = glob.glob(os.path.join(base_path, "*.ipynb"))
print("Moving notebooks...")
for nb in notebooks:
    dst = os.path.join(notebooks_dir, os.path.basename(nb))
    try:
        shutil.move(nb, dst)
        print(f"  Moved: {os.path.basename(nb)} -> notebooks/")
    except Exception as e:
        print(f"  Error moving {os.path.basename(nb)}: {e}")

# 2. Move Models and Adapters
model_files = [
    "modelo_biologico.pt",
    "modelo_percepcion_final.pt",
    "modelo_percepcion_s3.pt"
]
print("Moving models...")
for mf in model_files:
    src = os.path.join(base_path, mf)
    if os.path.exists(src):
        dst = os.path.join(models_dir, mf)
        try:
            shutil.move(src, dst)
            print(f"  Moved: {mf} -> models/")
        except Exception as e:
            print(f"  Error moving {mf}: {e}")

model_dirs = [
    "agro_llama_8b_contrafactual",
    "agro_llama_8b_final",
    "agro_llama_8b_final_cot",
    "agro_llama_torch",
    "agro_llama_torch_cot"
]
print("Moving LoRA adapters...")
for md in model_dirs:
    src = os.path.join(base_path, md)
    if os.path.exists(src):
        dst = os.path.join(models_dir, md)
        try:
            # If destination already exists, remove it first to avoid nesting
            if os.path.exists(dst):
                if os.path.isdir(dst):
                    shutil.rmtree(dst)
                else:
                    os.remove(dst)
            shutil.move(src, dst)
            print(f"  Moved: {md}/ -> models/")
        except Exception as e:
            print(f"  Error moving {md}: {e}")

# 3. Clean old scripts
old_scripts = [
    "descargar_qwen3.py",
    "diagnosis.py",
    "inferencia_experta.py",
    "pdf_report.py",
    "pipeline_32B.py",
    "pipeline_completo.py",
    "pipeline_completo_v8b.py",
    "pipeline_ruido_contrafactual.py",
    "quick_test_riotinto.py",
    "test_contrafactual.py",
    "test_evaluacion_contrafactual.py",
    "test_llm_judge.py",
    "test_llm_judge_v3.py",
    "test_modelo.py",
    "mapa_tesis_final.html",
    "resultado_cientifico_tesis.csv"
]
print("Deleting old scripts and files...")
for osf in old_scripts:
    src = os.path.join(base_path, osf)
    if os.path.exists(src):
        try:
            if os.path.isdir(src):
                shutil.rmtree(src)
            else:
                os.remove(src)
            print(f"  Deleted: {osf}")
        except Exception as e:
            print(f"  Error deleting {osf}: {e}")

print("Clean-up complete!")
