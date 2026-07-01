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
                    if "aemet" in source.lower():
                        print(f"{file} (Cell {cell_idx}):")
                        lines = source.split("\n")
                        for line_no, line in enumerate(lines, 1):
                            if "aemet" in line.lower():
                                print(f"  Line {line_no}: {line.strip()}")
        except Exception as e:
            print(f"Error reading {file}: {e}")
