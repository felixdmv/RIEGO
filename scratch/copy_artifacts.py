import os
import shutil

dest_dir = r"C:\Users\felix\.gemini\antigravity-cli\brain\c0d8d9dd-380f-4d08-953e-e61a6b50d3be"
os.makedirs(dest_dir, exist_ok=True)

# Copy PDF
pdf_src = r"recomendaciones/Prescripcion_S2_7d_2026-07-06.pdf"
if os.path.exists(pdf_src):
    pdf_dst = os.path.join(dest_dir, "Prescripcion_S2_7d_2026-07-06.pdf")
    shutil.copy(pdf_src, pdf_dst)
    print(f"Copied PDF to: {pdf_dst}")
else:
    print(f"PDF source {pdf_src} not found!")

# Copy Temp Sat Image
sat_src = "temp_sat.png"
if os.path.exists(sat_src):
    sat_dst = os.path.join(dest_dir, "temp_sat_2026-07-06.png")
    shutil.copy(sat_src, sat_dst)
    print(f"Copied Sat Image to: {sat_dst}")
else:
    print(f"Sat Image source {sat_src} not found!")
