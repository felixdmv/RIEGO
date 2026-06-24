import pandas as pd
import os

base_path = r"C:\Users\fdemiguel\OneDrive - Universidad de Burgos\Documentos\RIEGO"
csv_path = os.path.join(base_path, "datos_multimodales_completos.csv")

print("File path:", csv_path)
print("File exists:", os.path.exists(csv_path))

df = pd.read_csv(csv_path)
print("Shape:", df.shape)
print("Columns:")
print(list(df.columns))
print("First 3 rows:")
print(df.head(3))
