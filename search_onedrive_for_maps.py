import os

onedrive_dir = r"C:\Users\fdemiguel\OneDrive - Universidad de Burgos"
print("Scanning OneDrive folder:", onedrive_dir)

keywords = ["mapa", "riego", "canduela", "sector", "valvula"]
matches = []

# To avoid walking forever, let's limit the number of files we check
limit = 100000
checked = 0

for root, dirs, files in os.walk(onedrive_dir):
    # Skip some massive folders
    if any(p in root for p in [".git", ".vscode", "node_modules", "anaconda3", "AppData", "Mendeley", "Grabaciones"]):
        continue
    for f in files:
        checked += 1
        if checked > limit:
            break
        low_f = f.lower()
        if any(k in low_f for k in keywords):
            path = os.path.join(root, f)
            matches.append(path)
    if checked > limit:
        print("Checked limit reached.")
        break

print(f"Checked {checked} files. Found {len(matches)} matches:")
for m in matches[:100]:
    print(f"  {os.path.relpath(m, onedrive_dir)} ({os.path.getsize(m)} bytes)")
