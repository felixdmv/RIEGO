import zipfile
import os

base_path = r"C:\Users\fdemiguel\OneDrive - Universidad de Burgos\Documentos\RIEGO"
excel_path = os.path.join(base_path, "datos", "canduela", "CANDUELA - 080221084930-077_ (1).xlsx")

print("Checking Excel file as zip:", excel_path)
try:
    with zipfile.ZipFile(excel_path, 'r') as z:
        # List all files inside the zip
        namelist = z.namelist()
        print("Total files in zip:", len(namelist))
        
        # Look for media files (images)
        media_files = [f for f in namelist if 'media' in f]
        print("Media files found:", len(media_files))
        for f in media_files:
            print(f"  {f}")
            
            # Extract to the workspace
            out_name = os.path.basename(f)
            out_path = os.path.join(base_path, out_name)
            with open(out_path, 'wb') as out_f:
                out_f.write(z.read(f))
            print(f"  Extracted to {out_path} ({os.path.getsize(out_path)} bytes)")
except Exception as e:
    print("Error unzipping Excel:", e)
