import pandas as pd
import os

base_path = r"C:\Users\fdemiguel\OneDrive - Universidad de Burgos\Documentos\RIEGO"
df = pd.read_csv(os.path.join(base_path, 'datos', 'Dataset_Final_S3.csv'), parse_dates=['Fecha'])

# Filter summer 2025 (June 1st to September 30th 2025)
df_2025 = df[(df['Fecha'] >= '2025-06-01') & (df['Fecha'] <= '2025-09-30')].copy()
print("Summer 2025 rows:", len(df_2025))

if len(df_2025) > 0:
    print("\nDescriptive stats for FloraPulse Potential in Summer 2025:")
    print(df_2025['FP_Potencial_Min'].describe())
    
    # 1. High stress day (< -15 bar)
    opt1 = df_2025[df_2025['FP_Potencial_Min'] < -15.0].sort_values('FP_Potencial_Min')
    if not opt1.empty:
        print("\nHigh Stress Date:", opt1.iloc[0]['Fecha'].strftime('%Y-%m-%d'), "Potential:", opt1.iloc[0]['FP_Potencial_Min'])
        
    # 2. Moderate stress day (-11 to -14 bar)
    opt2 = df_2025[(df_2025['FP_Potencial_Min'] >= -14.0) & (df_2025['FP_Potencial_Min'] <= -11.0)].sort_values('FP_Potencial_Min')
    if not opt2.empty:
        print("Moderate Stress Date:", opt2.iloc[0]['Fecha'].strftime('%Y-%m-%d'), "Potential:", opt2.iloc[0]['FP_Potencial_Min'])
        
    # 3. Mild stress day (-7 to -10 bar)
    opt3 = df_2025[(df_2025['FP_Potencial_Min'] >= -10.0) & (df_2025['FP_Potencial_Min'] <= -7.0)].sort_values('FP_Potencial_Min')
    if not opt3.empty:
        print("Mild Stress Date:", opt3.iloc[0]['Fecha'].strftime('%Y-%m-%d'), "Potential:", opt3.iloc[0]['FP_Potencial_Min'])
        
    # 4. Low stress / well-watered day (> -6.0 bar)
    opt4 = df_2025[df_2025['FP_Potencial_Min'] > -6.0].sort_values('FP_Potencial_Min', ascending=False)
    if not opt4.empty:
        print("Low Stress Date:", opt4.iloc[0]['Fecha'].strftime('%Y-%m-%d'), "Potential:", opt4.iloc[0]['FP_Potencial_Min'])
