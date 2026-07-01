path = r"C:\Users\fdemiguel\OneDrive - Universidad de Burgos\Documentos\RIEGO\run_pipeline.py"
with open(path, "r", encoding="utf-8") as f:
    lines = f.readlines()

in_report = False
for line_no, line in enumerate(lines, 1):
    if "class AgroProductionReport" in line:
        in_report = True
    elif in_report and line.startswith("class "):
        in_report = False
    
    if in_report:
        if "aemet" in line.lower() or "sardon" in line.lower() or "s3" in line.lower():
            print(f"Line {line_no}: {line.strip()}")
