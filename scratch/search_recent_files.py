import os
import glob
from datetime import datetime, timedelta

user_dir = r"C:\Users\felix"
now = datetime.now()
limit = now - timedelta(days=7)

recent_files = []

# Directories to search
search_paths = [
    os.path.join(user_dir, "Downloads"),
    os.path.join(user_dir, "Desktop"),
    os.path.join(user_dir, "OneDrive - Universidad de Burgos", "Escritorio"),
    os.path.join(user_dir, "OneDrive - Universidad de Burgos", "Documentos", "RIEGO")
]

print("Scanning for files modified since:", limit)
for base in search_paths:
    if not os.path.exists(base):
        print(f"Path does not exist: {base}")
        continue
    print(f"Scanning {base}...")
    for root, dirs, files in os.walk(base):
        # Skip some big system dirs or hidden dirs to be fast
        if any(p in root for p in [".git", "__pycache__", ".vscode", "venv", ".agents", ".deepeval"]):
            continue
        for f in files:
            p = os.path.join(root, f)
            try:
                mtime = datetime.fromtimestamp(os.path.getmtime(p))
                if mtime >= limit:
                    recent_files.append((p, mtime, os.path.getsize(p)))
            except Exception:
                pass

print(f"\nFound {len(recent_files)} files modified in the last 7 days:")
# Sort by mtime descending
recent_files.sort(key=lambda x: x[1], reverse=True)
for p, mtime, size in recent_files[:50]:
    print(f"{p} | {mtime} | {size} bytes")
