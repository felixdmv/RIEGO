import os
import re
from datetime import datetime

dir_old = r"datos/canduela/GEE_Tesis_ImageChips-20260217T075250Z-1-001/GEE_Tesis_ImageChips"
dir_new = r"datos/canduela/GEE_Tesis_NuevaParcela"

print("Old Directory:")
if os.path.exists(dir_old):
    files_old = os.listdir(dir_old)
    print(f"Total files: {len(files_old)}")
    print(f"Sample files: {files_old[:5]}")
else:
    print("Old dir does not exist.")

print("\nNew Directory:")
if os.path.exists(dir_new):
    files_new = os.listdir(dir_new)
    print(f"Total files: {len(files_new)}")
    print(f"Sample files: {files_new[:5]}")
else:
    print("New dir does not exist.")
