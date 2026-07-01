import os
import json

notebooks_dir = r"C:\Users\fdemiguel\OneDrive - Universidad de Burgos\Documentos\RIEGO\notebooks"

for file in os.listdir(notebooks_dir):
    if file.endswith(".ipynb"):
        path = os.path.join(notebooks_dir, file)
        try:
            with open(path, "r", encoding="utf-8") as f:
                nb = json.load(f)
                for cell_idx, cell in enumerate(nb.get("cells", [])):
                    source = "".join(cell.get("source", []))
                    if any(x in source.lower() for x in ["carralancha", "v4", "pozaldez"]):
                        print(f"=== Found in {file} (Cell {cell_idx}) ===")
                        lines = source.split("\n")
                        for line_no, line in enumerate(lines, 1):
                            if any(x in line.lower() for x in ["carralancha", "v4", "pozaldez", "area", "superficie"]):
                                print(f"  Line {line_no}: {line.strip()}")
                        print("=" * 60)
        except Exception as e:
            print(f"Error reading {file}: {e}")
