import re

path = r"C:\Users\fdemiguel\OneDrive - Universidad de Burgos\Documentos\RIEGO\run_pipeline.py"
with open(path, "r", encoding="utf-8") as f:
    for line_no, line in enumerate(f, 1):
        if "dosis" in line.lower() or "dosis_mm" in line.lower() or "generate_recommendation" in line.lower():
            print(f"Line {line_no}: {line.strip()}")
