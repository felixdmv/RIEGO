import pandas as pd
import os

base_path = r"C:\Users\fdemiguel\OneDrive - Universidad de Burgos\Documentos\RIEGO"
excel_path = os.path.join(base_path, "datos", "canduela", "CANDUELA - 080221084930-077_ (1).xlsx")

df = pd.read_excel(excel_path)
print("Raw rows 277 to 285:")
print(df.iloc[277:286][['Equipo', 'Programa', 'Comienzo', 'Finalización', 'Duración', 'Consumo real (m3)', 'Válvulas', 'Fase']])
