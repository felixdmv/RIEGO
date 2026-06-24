import json
import os
import sys

# Ensure stdout handles unicode characters
sys.stdout.reconfigure(encoding='utf-8')

base_path = r"C:\Users\fdemiguel\OneDrive - Universidad de Burgos\Documentos\RIEGO"
notebooks_dir = os.path.join(base_path, "notebooks")

print("--- DETAILED SEARCH IN NOTEBOOKS ---")
for f in sorted(os.listdir(notebooks_dir)):
    if f.endswith(".ipynb"):
        path = os.path.join(notebooks_dir, f)
        try:
            with open(path, "r", encoding="utf-8") as file:
                content = json.load(file)
            for idx, cell in enumerate(content.get("cells", [])):
                cell_text = "".join(cell.get("source", []))
                # Search for coordinate components or keywords
                if "41." in cell_text or "42." in cell_text or "-4." in cell_text or "caudal" in cell_text.lower() or "valvula" in cell_text.lower() or "sector" in cell_text.lower():
                    # Check if there is anything about sector 2, or v2, or area, or canduela
                    if any(x in cell_text.lower() for x in ["canduela", "sector", "riegos", "v2", "valvula"]):
                        print(f"File: {f}, Cell: {idx}")
                        for line in cell_text.split("\n"):
                            if any(k in line.lower() for k in ["coor", "lat", "lon", "poligono", "polygon", "area", "superficie", "caudal", "flow", "valvula", "v2", "sector"]):
                                print(f"  Line: {line.strip()[:180]}")
        except Exception as e:
            print(f"Error reading {f}: {e}")
