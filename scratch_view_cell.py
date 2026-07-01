import os
import json

path = r"C:\Users\fdemiguel\OneDrive - Universidad de Burgos\Documentos\RIEGO\notebooks\01_exploracion.ipynb"

with open(path, "r", encoding="utf-8") as f:
    nb = json.load(f)

for cell_idx, cell in enumerate(nb.get("cells", [])):
    source = "".join(cell.get("source", []))
    if "fetch_aemet_data" in source:
        print(f"CELL {cell_idx}:")
        print(source)
        print("-" * 50)
