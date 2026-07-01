import os
import pandas as pd
import numpy as np

base_path = r"C:\Users\fdemiguel\OneDrive - Universidad de Burgos\Documentos\RIEGO"
df = pd.read_csv(os.path.join(base_path, 'datos', 'Dataset_Final_S3.csv'), parse_dates=['Fecha'])

# Filter summer 2024 (June 1st to September 30th 2024)
df_2024 = df[(df['Fecha'] >= '2024-06-01') & (df['Fecha'] <= '2024-09-30')].copy()

print("Available rows in Summer 2024:", len(df_2024))
print("\nDescriptive stats for FloraPulse Potential:")
print(df_2024['FP_Potencial_Min'].describe())

# Find some interesting dates:
# 1. Hot day with moderate stress (Low potential hídrico, e.g. < -12 bar)
opt1 = df_2024[df_2024['FP_Potencial_Min'] < -12.0].sort_values('FP_Potencial_Min')
if not opt1.empty:
    date_stress = opt1.iloc[0]['Fecha'].strftime('%Y-%m-%d')
    print("Stress Date:", date_stress, "Potential:", opt1.iloc[0]['FP_Potencial_Min'])
else:
    date_stress = "2024-08-10" # fallback

# 2. Day with low stress / high humidity (e.g. potential close to 0)
opt2 = df_2024[df_2024['FP_Potencial_Min'] > -6.0].sort_values('FP_Potencial_Min', ascending=False)
if not opt2.empty:
    date_well_watered = opt2.iloc[0]['Fecha'].strftime('%Y-%m-%d')
    print("Well-watered Date:", date_well_watered, "Potential:", opt2.iloc[0]['FP_Potencial_Min'])
else:
    date_well_watered = "2024-06-15"

# 3. A day in mid-season with average conditions (potential around -9 bar)
opt3 = df_2024[(df_2024['FP_Potencial_Min'] <= -8.0) & (df_2024['FP_Potencial_Min'] >= -10.0)]
if not opt3.empty:
    date_mid = opt3.iloc[len(opt3)//2]['Fecha'].strftime('%Y-%m-%d')
    print("Average Day Date:", date_mid, "Potential:", opt3.iloc[len(opt3)//2]['FP_Potencial_Min'])
else:
    date_mid = "2024-07-20"

# 4. A day with high rain forecast
opt4 = df_2024[df_2024['Lluvia_Prevista_3d'] > 2.0].sort_values('Lluvia_Prevista_3d', ascending=False)
if not opt4.empty:
    date_rainy = opt4.iloc[0]['Fecha'].strftime('%Y-%m-%d')
    print("Rainy Day Date:", date_rainy, "Rain Forecast:", opt4.iloc[0]['Lluvia_Prevista_3d'])
else:
    date_rainy = "2024-09-02"

# Let's generate reports for these dates
dates_to_generate = [date_stress, date_well_watered, date_mid, date_rainy]
print("\nDates to generate reports for:", list(set(dates_to_generate)))
