import pandas as pd
import os

base_path = r"C:\Users\fdemiguel\OneDrive - Universidad de Burgos\Documentos\RIEGO"
df = pd.read_csv(os.path.join(base_path, 'datos', 'Dataset_Final_S3.csv'), parse_dates=['Fecha'])

print("Total rows:", len(df))
print("Rows with potential < 0.49:", len(df[df['FP_Potencial_Min'] < 0.49]))
print("Rows with potential < 0.0:", len(df[df['FP_Potencial_Min'] < 0.0]))

if len(df[df['FP_Potencial_Min'] < 0.49]) > 0:
    print("\nSample rows with negative potential:")
    print(df[df['FP_Potencial_Min'] < 0.49][['Fecha', 'FP_Potencial_Min', 'FP_Humedad_10cm', 'Riego_mm_Real']].head(10))
    print("\nYearly counts of negative potentials:")
    print(df[df['FP_Potencial_Min'] < 0.49]['Fecha'].dt.year.value_counts())
else:
    print("\nAll potentials are 0.5 or higher. Let's check a sample of potentials:")
    print(df[['Fecha', 'FP_Potencial_Min']].sample(10))
