import os
import json

base_path = r"C:\Users\fdemiguel\OneDrive - Universidad de Burgos\Documentos\RIEGO"
keywords = ["sector 2", "sector2", "41.814", "-4.577"]

print("--- SCANNING JSON/JSONL FILES ---")
for root, dirs, files in os.walk(base_path):
    for f in files:
        if f.endswith((".json", ".jsonl")):
            path = os.path.join(root, f)
            try:
                with open(path, "r", encoding="utf-8", errors="ignore") as file:
                    for line_num, line in enumerate(file, 1):
                        if any(k in line.lower() for k in keywords):
                            print(f"{os.path.relpath(path, base_path)}:{line_num}: {line.strip()[:150]}")
            except Exception as e:
                print(f"Error reading {f}: {e}")
