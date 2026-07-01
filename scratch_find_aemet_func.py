import os
import json

notebooks_dir = r"C:\Users\fdemiguel\OneDrive - Universidad de Burgos\Documentos\RIEGO\notebooks"

found = False
for file in os.listdir(notebooks_dir):
    if file.endswith(".ipynb"):
        path = os.path.join(notebooks_dir, file)
        try:
            with open(path, "r", encoding="utf-8") as f:
                nb = json.load(f)
                for cell_idx, cell in enumerate(nb.get("cells", [])):
                    source = "".join(cell.get("source", []))
                    if "fetch_aemet" in source:
                        print(f"=== Found in {file} (Cell {cell_idx}) ===")
                        print(source)
                        print("=" * 60)
                        found = True
        except Exception as e:
            print(f"Error reading {file}: {e}")

if not found:
    print("Not found in any notebook!")
