import os

downloads_dir = r"C:\Users\fdemiguel\Downloads"
print("Scanning all files in Downloads folder:", downloads_dir)

if os.path.exists(downloads_dir):
    try:
        for root, dirs, files in os.walk(downloads_dir):
            for f in files:
                path = os.path.join(root, f)
                print(f"File: {os.path.relpath(path, downloads_dir)} ({os.path.getsize(path)} bytes)")
    except Exception as e:
        print("Error walking Downloads:", e)
else:
    print("Downloads folder does not exist")
