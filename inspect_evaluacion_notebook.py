import json
import os
import sys

# Safely print unicode
sys.stdout.reconfigure(encoding='utf-8')

base_path = r"C:\Users\fdemiguel\OneDrive - Universidad de Burgos\Documentos\RIEGO"
notebook_path = os.path.join(base_path, "notebooks", "05_evaluacion.ipynb")

print("Checking notebook:", notebook_path)
with open(notebook_path, "r", encoding="utf-8") as f:
    notebook = json.load(f)

keywords = ["sector 2", "sector2", "v2", "valvula2", "caudal", "superficie", "area", "41.814", "-4.577"]

for idx, cell in enumerate(notebook.get("cells", [])):
    cell_text = "".join(cell.get("source", []))
    for kw in keywords:
        if kw in cell_text.lower():
            print(f"Cell {idx} has keyword '{kw}':")
            for line in cell_text.split('\n'):
                if kw in line.lower():
                    print("  ", line.strip()[:150])
