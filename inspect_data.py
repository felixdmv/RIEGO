import os
import re
import json

base_path = r"C:\Users\fdemiguel\OneDrive - Universidad de Burgos\Documentos\RIEGO"

# 1. Search in notebooks
notebooks_dir = os.path.join(base_path, "notebooks")
keywords = ["canduela", "caudal", "superficie", "sector 2", "valvula2", "v2", "m3/h", "m3/ha", "hectarea", "area"]

print("--- SEARCH IN NOTEBOOKS ---")
for f in os.listdir(notebooks_dir):
    if f.endswith(".ipynb"):
        path = os.path.join(notebooks_dir, f)
        try:
            with open(path, "r", encoding="utf-8") as file:
                content = json.load(file)
            # Search text in cells
            for idx, cell in enumerate(content.get("cells", [])):
                cell_text = "".join(cell.get("source", []))
                for kw in keywords:
                    if re.search(r'\b' + re.escape(kw) + r'\b', cell_text, re.IGNORECASE):
                        print(f"Found '{kw}' in {f} cell {idx}: {cell_text[:200]}...")
        except Exception as e:
            print(f"Error reading {f}: {e}")

# 2. Print all files in datos/canduela and subdirectories with no filter
print("\n--- ALL FILES IN datos/canduela ---")
canduela_dir = os.path.join(base_path, "datos", "canduela")
for root, dirs, files in os.walk(canduela_dir):
    for f in files:
        print(os.path.relpath(os.path.join(root, f), base_path))

# 3. Print all files in root directory with no filter (except some folders)
print("\n--- ALL FILES IN root ---")
for root, dirs, files in os.walk(base_path):
    if any(p in root for p in [".git", ".vscode", ".sixth", ".deepeval", "runs", "__pycache__", "datos", "notebooks", "models", "deps", "ds_final_deps", "ds_offline", "offload_qwen3_32b"]):
        continue
    for f in files:
        print(os.path.relpath(os.path.join(root, f), base_path))
