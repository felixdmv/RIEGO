import json
import os
import re

base_path = r"C:\Users\fdemiguel\OneDrive - Universidad de Burgos\Documentos\RIEGO"
notebooks_dir = os.path.join(base_path, "notebooks")

keywords = ["caudal", "superficie", "canduela", "sector 2", "valvula2", "v2", "m3", "hectarea", "area", "m3/h"]

print("--- SEARCH IN NOTEBOOKS ---")
for f in sorted(os.listdir(notebooks_dir)):
    if f.endswith(".ipynb"):
        path = os.path.join(notebooks_dir, f)
        try:
            with open(path, "r", encoding="utf-8") as file:
                content = json.load(file)
            for idx, cell in enumerate(content.get("cells", [])):
                cell_text = "".join(cell.get("source", []))
                for kw in keywords:
                    matches = list(re.finditer(r'\b' + re.escape(kw) + r'\b', cell_text, re.IGNORECASE))
                    if matches:
                        print(f"File: {f}, Cell: {idx}, Keyword: {kw}")
                        # Print the line containing the match
                        lines = cell_text.split('\n')
                        for line in lines:
                            if kw.lower() in line.lower():
                                print(f"  Line: {line.strip()[:150]}")
        except Exception as e:
            print(f"Error reading {f}: {e}")

print("\n--- SEARCH IN PYTHON FILES ---")
for f in sorted(os.listdir(base_path)):
    if f.endswith(".py"):
        path = os.path.join(base_path, f)
        try:
            with open(path, "r", encoding="utf-8") as file:
                content = file.read()
            for kw in keywords:
                if kw.lower() in content.lower():
                    print(f"File: {f}, Keyword: {kw}")
                    lines = content.split('\n')
                    for idx, line in enumerate(lines):
                        if kw.lower() in line.lower():
                            print(f"  Line {idx+1}: {line.strip()[:150]}")
        except Exception as e:
            print(f"Error reading {f}: {e}")
