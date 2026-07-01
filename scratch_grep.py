import os
import glob

root_dir = r"C:\Users\fdemiguel\OneDrive - Universidad de Burgos\Documentos\RIEGO"

for root, dirs, files in os.walk(root_dir):
    # Exclude directories we don't care about
    for d in [".git", "__pycache__", ".vscode", ".deepeval", ".sixth"]:
        if d in dirs:
            dirs.remove(d)
    for file in files:
        if file.endswith((".py", ".json", ".txt", ".md", ".jsonl")):
            path = os.path.join(root, file)
            try:
                with open(path, "r", encoding="utf-8", errors="ignore") as f:
                    for line_no, line in enumerate(f, 1):
                        if "v4" in line.lower():
                            print(f"{os.path.relpath(path, root_dir)}:{line_no}: {line.strip()}")
            except Exception as e:
                pass
