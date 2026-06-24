import os

parent_dir = r"C:\Users\fdemiguel\OneDrive - Universidad de Burgos\Documentos"
keywords = ["mapa", "riego", "canduela", "sector", "valvula"]

print("--- SCANNING DOCUMENTOS RECURSIVELY ---")
count = 0
for root, dirs, files in os.walk(parent_dir):
    # Skip some massive folders if they are obviously irrelevant
    if any(p in root for p in ["Mendeley", "WindowsPowerShell", "Copilot"]):
        continue
    for f in files:
        low_f = f.lower()
        if any(k in low_f for k in keywords):
            path = os.path.join(root, f)
            print(f"Match: {os.path.relpath(path, parent_dir)} ({os.path.getsize(path)} bytes)")
            count += 1
            if count > 50:
                print("Too many matches, stopping...")
                break
    if count > 50:
        break
