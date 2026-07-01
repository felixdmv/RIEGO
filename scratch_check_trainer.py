path = r"C:\Users\fdemiguel\OneDrive - Universidad de Burgos\Documentos\RIEGO\run_pipeline.py"
with open(path, "r", encoding="utf-8") as f:
    lines = f.readlines()

in_class = False
for line_no, line in enumerate(lines, 1):
    if "class AgroModelTrainer" in line:
        in_class = True
    elif in_class and line.startswith("class "):
        in_class = False
    
    if in_class:
        if "s3" in line.lower() or "canduela" in line.lower() or "dataset_final" in line.lower():
            print(f"Line {line_no}: {line.strip()}")
        if "percepcion" in line.lower():
            print(f"Line {line_no}: {line.strip()}")
