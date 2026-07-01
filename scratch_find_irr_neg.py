import pandas as pd
import os

base_path = r"C:\Users\fdemiguel\OneDrive - Universidad de Burgos\Documentos\RIEGO"
df = pd.read_csv(os.path.join(base_path, 'datos', 'Dataset_Final_S3.csv'), parse_dates=['Fecha'])

df = df.sort_values('Fecha').reset_index(drop=True)
# Keep only irrigation season months (June to October)
df_irr = df[df['Fecha'].dt.month.isin([6, 7, 8, 9, 10])].copy()

# Print counts of negative potentials in irrigation seasons
print("Total irrigation season rows:", len(df_irr))
print("Negative potential rows in irrigation season:", len(df_irr[df_irr['FP_Potencial_Min'] < 0.49]))

# Show dates with negative potentials during the irrigation season, sorted by year and date
neg_irr = df_irr[df_irr['FP_Potencial_Min'] < 0.49].sort_values('Fecha')
print("\nFirst 30 irrigation season dates with negative potentials:")
print(neg_irr[['Fecha', 'FP_Potencial_Min', 'FP_Humedad_10cm', 'Riego_mm_Real']].head(30))
