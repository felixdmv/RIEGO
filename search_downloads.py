import os

downloads_dir = r"C:\Users\fdemiguel\Downloads"
print("Scanning Downloads folder:", downloads_dir)
keywords = ["mapa", "riego", "canduela", "sector", "valvula"]

if os.path.exists(downloads_dir):
    try:
        for root, dirs, files in os.walk(downloads_dir):
            for f in files:
                low_f = f.lower()
                if any(k in low_f for k in keywords):
                    path = os.path.join(root, f)
                    print(f"Match: {os.path.relpath(path, downloads_dir)} ({os.path.getsize(path)} bytes)")
    except Exception as e:
        print("Error walking Downloads:", e)
else:
    print("Downloads folder does not exist")
