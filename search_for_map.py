import os

parent_dir = r"C:\Users\fdemiguel\OneDrive - Universidad de Burgos\Documentos"
print("Scanning:", parent_dir)

try:
    for item in os.listdir(parent_dir):
        path = os.path.join(parent_dir, item)
        if os.path.isdir(path):
            print(f"[Dir]  {item}")
        else:
            print(f"[File] {item} ({os.path.getsize(path)} bytes)")
except Exception as e:
    print("Error listing parent_dir:", e)

# Also check one level up if needed
grandparent_dir = r"C:\Users\fdemiguel\OneDrive - Universidad de Burgos"
print("\nScanning:", grandparent_dir)
try:
    for item in os.listdir(grandparent_dir):
        path = os.path.join(grandparent_dir, item)
        if os.path.isdir(path):
            print(f"[Dir]  {item}")
        else:
            print(f"[File] {item} ({os.path.getsize(path)} bytes)")
except Exception as e:
    print("Error listing grandparent_dir:", e)
