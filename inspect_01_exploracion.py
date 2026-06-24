import json
import os

base_path = r"C:\Users\fdemiguel\OneDrive - Universidad de Burgos\Documentos\RIEGO"
notebook_path = os.path.join(base_path, "notebooks", "01_exploracion.ipynb")

with open(notebook_path, "r", encoding="utf-8") as f:
    notebook = json.load(f)

print("=== Cell 0 source ===")
print("".join(notebook['cells'][0]['source']))
