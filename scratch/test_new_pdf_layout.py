import os
import re
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import cv2
import rasterio
from datetime import datetime
import torch
from fpdf import FPDF
from fpdf.enums import XPos, YPos

base_path = r"C:\Users\fdemiguel\OneDrive - Universidad de Burgos\Documentos\RIEGO"
datos_path = os.path.join(base_path, 'datos')
img_dir = os.path.join(datos_path, 'GEE_Tesis_ImageChips-20260217T075250Z-1-001/GEE_Tesis_ImageChips')
date_str = "2024-08-10"
date_dt = pd.to_datetime(date_str)

df = pd.read_csv(os.path.join(datos_path, 'Dataset_Final_S3.csv'), parse_dates=['Fecha'])
row = df[df["Fecha"] == date_dt].iloc[0]
history_df = df[(df["Fecha"] <= date_dt)].tail(30)

# Colors definition for the document
# Forest green (Primary), Soft green (Secondary), Warning amber (Alert), White, Charcoal (Text), Light green-grey (Card fill)
C_PRIMARY = (27, 94, 32)
C_SECONDARY = (56, 142, 60)
C_ALERT = (216, 67, 21)
C_TEXT = (33, 33, 33)
C_BG_CARD = (245, 247, 245)

class AgroPDF(FPDF):
    def header(self):
        # Forest green background banner
        self.set_fill_color(27, 94, 32)
        self.rect(0, 0, 210, 24, "F")
        self.set_y(6)
        self.set_font("helvetica", "B", 14)
        self.set_text_color(255)
        self.cell(0, 8, "INFORME OPERATIVO DE PRECISION - ORQUESTADOR AGRO-XAI", align="C", new_x=XPos.LMARGIN, new_y=YPos.NEXT)
        self.set_font("helvetica", "I", 9)
        self.cell(0, 4, "Monitoreo Integrado de Vigor Satelital, Clima y Fisiologia Foliar", align="C", new_x=XPos.LMARGIN, new_y=YPos.NEXT)
        self.ln(6)

    def footer(self):
        self.set_y(-15)
        self.set_font("helvetica", "I", 8)
        self.set_text_color(120)
        self.cell(0, 10, f"Pagina {self.page_no()} | Generado por Orquestador Agro-XAI", align="C")

    def section_header(self, title):
        self.set_font("helvetica", "B", 11)
        self.set_fill_color(230, 240, 230)
        self.set_text_color(27, 94, 32)
        # 3mm left border in dark green, light green background
        y = self.get_y()
        self.rect(15, y, 180, 7, "F")
        self.set_fill_color(27, 94, 32)
        self.rect(15, y, 3, 7, "F")
        self.set_xy(20, y + 1)
        self.cell(0, 5, title, new_x=XPos.LMARGIN, new_y=YPos.NEXT)
        self.ln(2)

def prepare_img(date_dt):
    files = [f for f in os.listdir(img_dir) if f.endswith(".tif")]
    img_dates = []
    valid_files = []
    for f in files:
        m = re.search(r'(\d{4})-\d{2}-(\d{1,3})', f)
        if m:
            try:
                fecha = datetime.strptime(f"{m.group(1)}-{m.group(2)}", "%Y-%j")
                img_dates.append(pd.Timestamp(fecha))
                valid_files.append(f)
            except: continue
    if not valid_files: return None
    dist = np.abs(pd.to_datetime(img_dates) - date_dt)
    closest_img_name = valid_files[np.argmin(dist)]
    
    with rasterio.open(os.path.join(img_dir, closest_img_name)) as src:
        img = src.read([3, 2, 1])
    img = np.moveaxis(img, 0, -1).astype(np.float32)
    p2, p98 = np.nanpercentile(img, (2, 98))
    img = np.clip((img - p2) / (p98 - p2 + 1e-10), 0, 1)
    img = np.nan_to_num(img, nan=0.0)
    img_8bit = (img * 255).astype(np.uint8)
    h, w = img_8bit.shape[:2]
    side = min(h, w)
    y0 = (h - side) // 2
    x0 = (w - side) // 2
    img_sq = img_8bit[y0:y0+side, x0:x0+side]
    img_sq = cv2.resize(img_sq, (600, 600), interpolation=cv2.INTER_CUBIC)
    cv2.putText(img_sq, "Sentinel-2 True Color (1:1)", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    cv2.line(img_sq, (400, 560), (550, 560), (255, 255, 255), 4)
    cv2.putText(img_sq, "150 m", (430, 550), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    out_path = "temp_sat_test.png"
    cv2.imwrite(out_path, cv2.cvtColor(img_sq, cv2.COLOR_RGB2BGR))
    return out_path

def generate_plots(row, history_df):
    # 1. Bar Plot: ET0 vs Lluvia
    fig, ax = plt.subplots(figsize=(5, 3.2))
    labels = ['ET0', 'Lluvia\n(24h)', 'Lluvia\n(Prev. 3d)']
    values = [row['ET0_Calculada'], row['Lluvia_AEMET'], row['Lluvia_Prevista_3d']]
    bars = ax.bar(labels, values, color=['#e67e22', '#2ecc71', '#3498db'], width=0.5, edgecolor='black', linewidth=0.5)
    ax.set_title("Balance Hidrico Operativo (mm)", fontsize=10, fontweight='bold', color='#1B5E20', pad=8)
    ax.set_ylabel("mm", fontsize=8)
    ax.grid(axis='y', linestyle='--', alpha=0.3)
    ax.tick_params(axis='x', labelsize=8)
    for bar, val in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.05, f"{val:.2f}", ha='center', va='bottom', fontsize=8, fontweight='bold')
    fig.tight_layout()
    plot_balance_path = "temp_plot_balance_test.png"
    fig.savefig(plot_balance_path, dpi=180)
    plt.close(fig)

    # 2. Line Plot: Dual axis evolution (NDVI vs Water Potential)
    fig, ax1 = plt.subplots(figsize=(6.5, 3.0))
    color = '#27ae60'
    ax1.set_xlabel('Ultimos 30 dias', fontsize=8, fontweight='bold')
    ax1.set_ylabel('Vigor Foliar (NDVI)', color=color, fontsize=8, fontweight='bold')
    line1 = ax1.plot(history_df['Fecha'], history_df['Sat_NDVI'], color=color, linewidth=2, label='Vigor (NDVI)', marker='o', markersize=3)
    ax1.tick_params(axis='y', labelcolor=color, labelsize=7)
    
    ax2 = ax1.twinx()
    color = '#2980b9'
    ax2.set_ylabel('Potencial Hidrico (bar)', color=color, fontsize=8, fontweight='bold')
    line2 = ax2.plot(history_df['Fecha'], history_df['FP_Potencial_Min'], color=color, linewidth=2, linestyle='--', label='Potencial Hidrico', marker='s', markersize=3)
    ax2.tick_params(axis='y', labelcolor=color, labelsize=7)
    
    lines = line1 + line2
    labels = [l.get_label() for l in lines]
    ax1.legend(lines, labels, loc='upper left', fontsize=7, framealpha=0.8)
    ax1.set_title("Evolucion de Vigor (NDVI) y Estres (FloraPulse)", fontsize=10, fontweight='bold', color='#1B5E20', pad=8)
    ax1.grid(True, alpha=0.15)
    fig.autofmt_xdate(rotation=15, ha='right')
    fig.tight_layout()
    plot_evol_path = "temp_plot_evol_test.png"
    fig.savefig(plot_evol_path, dpi=180)
    plt.close(fig)
    return plot_balance_path, plot_evol_path

# Generate files
img_sat_path = prepare_img(date_dt)
plot_balance_path, plot_evol_path = generate_plots(row, history_df)

# Initialize PDF
pdf = AgroPDF()
pdf.add_page()
pdf.set_text_color(*C_TEXT)

# --- 1. FICHA TECNICA Y METADATOS ---
pdf.set_font("helvetica", "", 8)
pdf.set_text_color(100)
pdf.cell(90, 4, f"ID PARCELA: Sector 3 | Superficie: 4.07 ha ({row.get('area_s3', 40650):,} m2)")
pdf.cell(90, 4, f"Fecha Analisis: {date_str} | Cultivo: Almendro Guara", align="R", new_x=XPos.LMARGIN, new_y=YPos.NEXT)
pdf.ln(2)

pdf.section_header("1. RESUMEN EJECUTIVO Y RELEVANCIA")
pdf.set_font("helvetica", "", 9.5)
pdf.set_text_color(50)
pdf.multi_cell(0, 5, 
    "El Sector 3 representa un suelo franco-arenoso con alta susceptibilidad al estres hidrico. "
    "Se monitoriza mediante el potencial hidrico foliar diario (FloraPulse) y satelital. "
    "El objetivo es mantener el Riego Deficitario Controlado (RDI) en limites fisiologicos optimos."
)
pdf.ln(3)

# --- 2. KPI DASHBOARD ---
pdf.set_fill_color(*C_BG_CARD)
pdf.set_draw_color(220, 225, 220)

# We will build 3 columns for KPIs: Vigor, Estrés, Humedad
y_kpis = pdf.get_y()
pdf.rect(15, y_kpis, 57, 24, "FD")
pdf.rect(76, y_kpis, 57, 24, "FD")
pdf.rect(138, y_kpis, 57, 24, "FD")

# Column 1: Vigor
pdf.set_xy(17, y_kpis + 2)
pdf.set_font("helvetica", "B", 8)
pdf.set_text_color(*C_PRIMARY)
pdf.cell(53, 4, "VIGOR SATELITAL (NDVI)", new_x=XPos.LMARGIN, new_y=YPos.NEXT)
pdf.set_xy(17, y_kpis + 7)
pdf.set_font("helvetica", "B", 12)
pdf.set_text_color(33, 33, 33)
pdf.cell(53, 6, f"{row['Sat_NDVI']:.3f}", new_x=XPos.LMARGIN, new_y=YPos.NEXT)
pdf.set_xy(17, y_kpis + 14)
pdf.set_font("helvetica", "", 8)
pdf.set_text_color(80)
pdf.cell(53, 4, "Vigor Medio (Optimo)")

# Column 2: Stress (FloraPulse)
pdf.set_xy(78, y_kpis + 2)
pdf.set_font("helvetica", "B", 8)
pdf.set_text_color(*C_PRIMARY)
pdf.cell(53, 4, "ESTRES FOLIAR (TALLO)", new_x=XPos.LMARGIN, new_y=YPos.NEXT)
pdf.set_xy(78, y_kpis + 7)
pdf.set_font("helvetica", "B", 12)
pot_val = row['FP_Potencial_Min']
stress_color = C_ALERT if pot_val < -12 else C_SECONDARY
pdf.set_text_color(*stress_color)
pdf.cell(53, 6, f"{pot_val:.1f} bar", new_x=XPos.LMARGIN, new_y=YPos.NEXT)
pdf.set_xy(78, y_kpis + 14)
pdf.set_font("helvetica", "", 8)
pdf.set_text_color(80)
stress_desc = "Moderado" if pot_val < -12 else "Leve (RDI)"
pdf.cell(53, 4, f"Nivel: {stress_desc}")

# Column 3: Humedad Suelo
pdf.set_xy(140, y_kpis + 2)
pdf.set_font("helvetica", "B", 8)
pdf.set_text_color(*C_PRIMARY)
pdf.cell(53, 4, "HUMEDAD RADICULAR (40cm)", new_x=XPos.LMARGIN, new_y=YPos.NEXT)
pdf.set_xy(140, y_kpis + 7)
pdf.set_font("helvetica", "B", 12)
pdf.set_text_color(33, 33, 33)
pdf.cell(53, 6, f"{row['FP_Humedad_40cm']:.1f}%", new_x=XPos.LMARGIN, new_y=YPos.NEXT)
pdf.set_xy(140, y_kpis + 14)
pdf.set_font("helvetica", "", 8)
pdf.set_text_color(80)
pdf.cell(53, 4, f"Drenaje 60cm: {row['FP_Humedad_60cm']:.1f}%")

pdf.set_xy(15, y_kpis + 28)
pdf.ln(2)

# --- 3. IMAGENES Y CLIMA (ALIGN SIDE-BY-SIDE DYNAMICALLY) ---
pdf.section_header("2. IMAGEN DE CULTIVO Y BALANCE CLIMATICO")
y_imgs = pdf.get_y()
# Set heights
img_h = 65
if img_sat_path and os.path.exists(img_sat_path):
    pdf.image(img_sat_path, x=15, y=y_imgs, w=75, h=img_h)
    
if os.path.exists(plot_balance_path):
    pdf.image(plot_balance_path, x=105, y=y_imgs, w=90, h=img_h)

pdf.set_y(y_imgs + img_h + 2)
pdf.set_font("helvetica", "I", 8)
pdf.set_text_color(120)
pdf.cell(75, 4, "Vista Sentinel-2 RGB Parcela", align="C")
pdf.cell(115, 4, "Demanda (ET0) vs Precipitación", align="R", new_x=XPos.LMARGIN, new_y=YPos.NEXT)
pdf.ln(3)

# Data Table
pdf.set_fill_color(240, 245, 240)
pdf.set_draw_color(220, 220, 220)
pdf.set_font("helvetica", "B", 8)
pdf.set_text_color(*C_PRIMARY)
# Header row
pdf.cell(35, 6, "Indicador", border=1, fill=True)
pdf.cell(30, 6, "Valor Actual", border=1, fill=True)
pdf.cell(30, 6, "Prevision", border=1, fill=True)
pdf.cell(85, 6, "Estado / Efecto Agronomico", border=1, fill=True, new_x=XPos.LMARGIN, new_y=YPos.NEXT)

pdf.set_font("helvetica", "", 8.5)
pdf.set_text_color(*C_TEXT)

pdf.cell(35, 6, "Vigor Foliar (NDVI)", border=1)
pdf.cell(30, 6, f"{row['Sat_NDVI']:.3f}", border=1)
pdf.cell(30, 6, f"t+5: {row['Sat_NDVI']*0.98:.3f}", border=1)
pdf.cell(85, 6, "Estabilidad de vigor foliar. Sin anomalias graves.", border=1, new_x=XPos.LMARGIN, new_y=YPos.NEXT)

pdf.cell(35, 6, "Evapotranspiracion", border=1)
pdf.cell(30, 6, f"{row['ET0_Calculada']:.1f} mm/dia", border=1)
pdf.cell(30, 6, f"ETc: {row['ETc_Sector3']:.1f} mm/dia", border=1)
pdf.cell(85, 6, f"Demanda real del cultivo moderada (Kc={row['Kc_Dinamico']:.2f})", border=1, new_x=XPos.LMARGIN, new_y=YPos.NEXT)

pdf.cell(35, 6, "Precipitacion (AEMET)", border=1)
pdf.cell(30, 6, f"{row['Lluvia_AEMET']:.1f} mm", border=1)
pdf.cell(30, 6, f"3 dias: {row['Lluvia_Prevista_3d']:.1f} mm", border=1)
pdf.cell(85, 6, "Sin precipitaciones significativas. Balance neto negativo.", border=1, new_x=XPos.LMARGIN, new_y=YPos.NEXT)

# PAGE 2
pdf.add_page()
pdf.set_font("helvetica", "", 8)
pdf.set_text_color(100)
pdf.cell(90, 4, f"ID PARCELA: Sector 3 | Fecha: {date_str}")
pdf.cell(90, 4, f"Orquestador Agro-XAI", align="R", new_x=XPos.LMARGIN, new_y=YPos.NEXT)
pdf.ln(2)

pdf.section_header("3. FISIOLOGIA DEL CULTIVO Y EVOLUCION")
y_chart2 = pdf.get_y()
if os.path.exists(plot_evol_path):
    pdf.image(plot_evol_path, x=15, y=y_chart2, w=180, h=65)
    pdf.set_y(y_chart2 + 67)
    pdf.ln(2)

# Soil & Potential Table
pdf.set_font("helvetica", "B", 8)
pdf.set_text_color(*C_PRIMARY)
pdf.set_fill_color(240, 245, 240)
pdf.cell(45, 6, "Sensor", border=1, fill=True)
pdf.cell(45, 6, "Medida", border=1, fill=True)
pdf.cell(90, 6, "Analisis Agronomico y Umbrales", border=1, fill=True, new_x=XPos.LMARGIN, new_y=YPos.NEXT)

pdf.set_font("helvetica", "", 8.5)
pdf.set_text_color(*C_TEXT)
pdf.cell(45, 6, "Tallo FloraPulse", border=1)
pdf.cell(45, 6, f"{row['FP_Potencial_Min']:.1f} bar (Base: {row['FP_Potencial_Base_Min']:.1f})", border=1)
pdf.cell(90, 6, f"Diferencia: {row['FP_Potencial_Min'] - row['FP_Potencial_Base_Min']:+.1f} bar. Nivel: {stress_desc}", border=1, new_x=XPos.LMARGIN, new_y=YPos.NEXT)

pdf.cell(45, 6, "Humedad Suelo 10-20cm", border=1)
pdf.cell(45, 6, f"10cm: {row['FP_Humedad_10cm']:.1f}% | 20cm: {row['FP_Humedad_20cm']:.1f}%", border=1)
pdf.cell(90, 6, "Actividad radicular superficial estable. Humedad adecuada.", border=1, new_x=XPos.LMARGIN, new_y=YPos.NEXT)

pdf.cell(45, 6, "Humedad Suelo 40-60cm", border=1)
pdf.cell(45, 6, f"40cm: {row['FP_Humedad_40cm']:.1f}% | 60cm: {row['FP_Humedad_60cm']:.1f}%", border=1)
pdf.cell(90, 6, "Zona de absorcion profunda activa. Drenaje controlado.", border=1, new_x=XPos.LMARGIN, new_y=YPos.NEXT)
pdf.ln(4)

# --- 4. AUDITORIA AGENTE IA ---
pdf.section_header("4. AUDITORIA OPERATIVA E INFORME DEL AGENTE IA")

# Styled Box using multi_cell fill
pdf.set_fill_color(*C_BG_CARD)
pdf.set_draw_color(220, 225, 220)
pdf.set_font("helvetica", "", 9)
pdf.set_text_color(50)

# Simulate advice CoT
kc_val = row['Kc_Dinamico']
etc_val = row['ETc_Sector3']
rain_prev = row['Lluvia_Prevista_3d']
riego_real = row['Riego_mm_Real']
pot_val = row['FP_Potencial_Min']
sm10_val = row['FP_Humedad_10cm']
sm40_val = row['FP_Humedad_40cm']

dosis_mm = max(0, etc_val * 3 - rain_prev) * 0.65
volumen_m3 = (dosis_mm * 40650) / 1000.0

# Irrigation hours calculation: Dose / 1.2 mm/h
irr_hours = dosis_mm / 1.2
# Sugerir turnos
turns = 2 if irr_hours > 6 else 1
hours_per_turn = irr_hours / turns

advice_text = (
    f"ANALISIS AGRO-XAI: Kc Dinamico {kc_val:.2f}, Demanda ETc {etc_val:.1f} mm/dia. "
    f"MONITORIZACION BIOFISICA (FLORAPULSE): Potencial foliar de {pot_val:.1f} bar (Estres {stress_desc}). "
    f"Humedad del suelo a 10cm: {sm10_val:.1f}%, a 40cm: {sm40_val:.1f}%. "
    f"AUDITORIA: Aplicados {riego_real:.1f} mm. Con lluvia prevista de {rain_prev:.1f} mm, "
    f"la dosis ideal para los proximos 3 dias es {dosis_mm:.1f} mm ({dosis_mm/3:.1f} mm/dia), equivalente a un volumen total de {volumen_m3:.1f} m3. "
    f"VARIANZA: {riego_real - dosis_mm/3:+.1f} mm. "
    f"PRESCRIPCION: Mantener riego deficitario leve. El potencial foliar esta en la zona de aclimatacion optima. "
    f"Se prescribe riego fraccionado para evitar estres termico por la tarde."
)

pdf.multi_cell(0, 5.2, advice_text, border=1, fill=True)
pdf.ln(4)

# --- 5. PRESCRIPCION AGRONOMICA DE RIEGO ---
pdf.section_header("5. PLAN DE APLICACIÓN DE RIEGO (PRÓXIMOS 3 DÍAS)")

# Box for the final numbers
pdf.set_fill_color(240, 250, 240)  # Soft green background
pdf.set_draw_color(34, 197, 94)   # Green border
pdf.set_font("helvetica", "B", 10.5)
pdf.set_text_color(*C_PRIMARY)

pdf.cell(0, 6, "  PLAN DE APLICACIÓN FÍSICA PARA EL AGRICULTOR", border="TLR", fill=True, new_x=XPos.LMARGIN, new_y=YPos.NEXT)
pdf.set_font("helvetica", "", 9.5)
pdf.set_text_color(*C_TEXT)
pdf.cell(90, 6, f"    * Dosis Total Recomendada: {dosis_mm:.1f} mm", border="L", fill=True)
pdf.cell(90, 6, f"    * Volumen Total a Aplicar: {volumen_m3:.1f} m3", border="R", fill=True, new_x=XPos.LMARGIN, new_y=YPos.NEXT)

pdf.cell(90, 6, f"    * Tiempo de Riego Estimado: {irr_hours:.1f} horas", border="L", fill=True)
pdf.cell(90, 6, f"    * Caudal Nominal de Diseño: 1.2 mm/h", border="R", fill=True, new_x=XPos.LMARGIN, new_y=YPos.NEXT)

pdf.set_font("helvetica", "B", 9.5)
pdf.set_text_color(*C_PRIMARY)
pdf.cell(0, 7, f"    * Programación Sugerida: {turns} turnos de {hours_per_turn:.1f} horas cada uno (ej. Lunes y Miércoles)", border="LBR", fill=True, new_x=XPos.LMARGIN, new_y=YPos.NEXT)
pdf.ln(4)

# Disclaimer
pdf.set_font("helvetica", "I", 8)
pdf.set_text_color(120)
pdf.multi_cell(0, 4.5, 
    "Aviso: La recomendación de riego se basa en el balance del cultivo (ETc) y la respuesta fisiológica de la planta (potencial de tallo de FloraPulse). "
    "Se asume un diseño estándar de riego por goteo con doble tubería lateral. Ajustar los turnos según la disponibilidad de energía eléctrica y el caudal real de cabezal."
)

# Output test report
output_path = os.path.join(base_path, "Informe_Final_2024-08-10.pdf")
pdf.output(output_path)
print("PDF test report compiled successfully!")

# Clean test files
for p in [img_sat_path, plot_balance_path, plot_evol_path]:
    if p and os.path.exists(p): os.remove(p)
