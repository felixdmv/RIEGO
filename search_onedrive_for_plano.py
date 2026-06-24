import os

onedrive_dir = r"C:\Users\fdemiguel\OneDrive - Universidad de Burgos"
print("Scanning OneDrive folder for 'plano', 'finca', 'parcela':", onedrive_dir)

keywords = ["plano", "finca", "parcela"]
matches = []

checked = 0
for root, dirs, files in os.walk(onedrive_dir):
    if any(p in root for p in [".git", ".vscode", "node_modules", "anaconda3", "AppData", "Mendeley", "Grabaciones"]):
        continue
    for f in files:
        checked += 1
        low_f = f.lower()
        if any(k in low_f for k in keywords):
            path = os.path.join(root, f)
            # Skip GEE chips
            if "GEE_Tesis" in path:
                continue
            matches.append(path)

print(f"Checked {checked} files. Found {len(matches)} matches:")
for m in matches[:100]:
    print(f"  {os.path.relpath(m, onedrive_dir)} ({os.path.getsize(m)} bytes)")
