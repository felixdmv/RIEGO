import os

possible_paths = [
    os.path.expanduser("~/.config/earthengine/credentials"),
    os.path.expanduser("~/AppData/Roaming/gcloud/application_default_credentials.json"),
    os.environ.get("GOOGLE_APPLICATION_CREDENTIALS", "")
]

for p in possible_paths:
    if p:
        print(f"Path: {p}")
        print(f"Exists: {os.path.exists(p)}")
        if os.path.exists(p):
            print(f"Size: {os.path.getsize(p)}")
