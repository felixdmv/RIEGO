import json
import os
import sys

sys.stdout.reconfigure(encoding='utf-8')

base_path = r"C:\Users\fdemiguel\OneDrive - Universidad de Burgos\Documentos\RIEGO"
notebook_path = os.path.join(base_path, "notebooks", "02_prediccion.ipynb")

print("Checking notebook:", notebook_path)
with open(notebook_path, "r", encoding="utf-8") as f:
    notebook = json.load(f)

for idx, cell in enumerate(notebook.get("cells", [])):
    cell_text = "".join(cell.get("source", []))
    if any(k in cell_text.lower() for k in ["sector 2", "sector2", "v2", "valvula2", "caudal", "superficie", "area"]):
        print(f"Cell {idx} matches:")
        for line in cell_text.split('\n'):
            if any(k in line.lower() for k in ["sector", "v2", "valvula", "caudal", "superficie", "area"]):
                print("  ", line.strip()[:150])
