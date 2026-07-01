path = r"C:\Users\fdemiguel\OneDrive - Universidad de Burgos\Documentos\RIEGO\run_pipeline.py"
with open(path, "r", encoding="utf-8") as f:
    lines = f.readlines()

in_class = False
for line_no, line in enumerate(lines, 1):
    if "class AgroLLMOptimizer" in line:
        in_class = True
    elif in_class and line.startswith("class "):
        in_class = False
    
    if in_class:
        if "dataset_final" in line.lower() or "s3" in line.lower() or "sardon" in line.lower() or "canduela" in line.lower():
            print(f"Line {line_no}: {line.strip()}")
