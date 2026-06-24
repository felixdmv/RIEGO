import pandas as pd
import os

base_path = r"C:\Users\fdemiguel\OneDrive - Universidad de Burgos\Documentos\RIEGO"
path = os.path.join(base_path, "datos", "canduela", "Dataset_Multimodal_S3_Tesis.csv")

df = pd.read_csv(path)
print("Columns:")
print(list(df.columns))
print("First 3 rows:")
print(df.head(3))
