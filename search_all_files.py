import os

base_path = r"C:\Users\fdemiguel\OneDrive - Universidad de Burgos\Documentos\RIEGO"

print("--- SEARCH FOR 'MAP' IN FILES ---")
for root, dirs, files in os.walk(base_path):
    if any(p in root for p in [".git", ".vscode", ".sixth", ".deepeval", "runs", "__pycache__", "datos", "models", "deps"]):
        continue
    for f in files:
        if f.endswith((".py", ".txt", ".json", ".md", ".jsonl")):
            path = os.path.join(root, f)
            try:
                with open(path, "r", encoding="utf-8", errors="ignore") as file:
                    for line_num, line in enumerate(file, 1):
                        if "mapa" in line.lower() or "map" in line.lower():
                            print(f"{os.path.relpath(path, base_path)}:{line_num}: {line.strip()[:100]}")
            except Exception as e:
                print(f"Error reading {f}: {e}")
