import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import os
from datetime import datetime, timedelta

# Create date list
start_date = datetime(2026, 6, 26)
dates = [start_date + timedelta(days=i) for i in range(10)]
date_strs = [d.strftime("%Y-%m-%d") for d in dates]

# Simulation parameters
initial_sm = 22.4  # % on June 26
kc = 0.69  # Crop coefficient for new parcel

# Daily weather data (mild warming up, then hot summer days)
# ET0 in mm/day
et0_values = [2.65, 4.35, 5.07, 5.80, 5.80, 6.09, 6.09, 6.09, 5.80, 5.80]
rain_values = [0.0] * 10

# Irrigation events (Tuesday June 30)
# We apply 19.6 mm on Tuesday June 30
irr_values = [0.0] * 10
irr_values[4] = 19.6  # Index 4 is June 30 (Friday=0, Sat=1, Sun=2, Mon=3, Tue=4)

# Calculate ETc
etc_values = [et0 * kc for et0 in et0_values]

# Simulate Soil Moisture depletion and recharge
# alpha = 0.15% depletion per mm of ETc
# beta = 0.15% recharge per mm of irrigation
alpha = 0.15
beta = 0.15

sm_values = []
current_sm = initial_sm
for i in range(10):
    # Today's starting moisture is current_sm
    # Subtract crop water uptake (ETc)
    depletion = etc_values[i] * alpha
    # Add irrigation
    recharge = irr_values[i] * beta
    current_sm = current_sm - depletion + recharge
    sm_values.append(current_sm)

df_sim = pd.DataFrame({
    "Fecha": date_strs,
    "ET0": et0_values,
    "ETc": etc_values,
    "Lluvia": rain_values,
    "Riego": irr_values,
    "Humedad_Suelo": sm_values
})

print("Simulation Results:")
print(df_sim.to_string(index=False))

# Plot the results
plt.style.use('seaborn-v0_8-whitegrid')
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8), sharex=True)

# Plot Soil Moisture
ax1.plot(df_sim["Fecha"], df_sim["Humedad_Suelo"], marker='o', color='#2E7D32', linewidth=2.5, label="Humedad del Suelo (40cm)")
ax1.axhline(20.0, color='#D84315', linestyle='--', linewidth=1.2, label="Punto de Estrés Crítico (20%)")
ax1.axvline("2026-06-30", color='#1565C0', linestyle=':', linewidth=1.5, label="Aplicación de Riego (Martes)")
ax1.set_ylabel("Humedad del Suelo (%)", fontsize=11, fontweight='bold')
ax1.set_title("Evolución Simulada de la Humedad del Suelo - Sector 2", fontsize=13, fontweight='bold', pad=15)
ax1.legend(loc="upper right", frameon=True, facecolor='white', framealpha=0.9)
ax1.set_ylim(18, 25)

# Plot ETc and Irrigation
width = 0.4
x = np.arange(len(dates))
ax2.bar(x - width/2, df_sim["ETc"], width, label="ETc (Demanda Cultivo)", color='#EF5350')
ax2.bar(x + width/2, df_sim["Riego"], width, label="Riego Aplicado", color='#42A5F5')
ax2.set_ylabel("Lámina de Agua (mm)", fontsize=11, fontweight='bold')
ax2.set_xlabel("Fecha", fontsize=11, fontweight='bold')
ax2.set_xticks(x)
ax2.set_xticklabels(df_sim["Fecha"], rotation=30, ha="right")
ax2.legend(loc="upper right", frameon=True, facecolor='white', framealpha=0.9)
ax2.set_title("Balances Hídricos Diarios (ETc vs Riego)", fontsize=12, fontweight='bold', pad=10)

plt.tight_layout()

# Save plot directly to artifacts folder
artifact_dir = r"C:\Users\felix\.gemini\antigravity-cli\brain\18053a23-fd95-4e96-9c25-992c7ba13f95"
plot_path = os.path.join(artifact_dir, "simulation_plot.png")
fig.savefig(plot_path, dpi=150)
plt.close(fig)

print(f"\nSaved simulation plot to {plot_path}")
