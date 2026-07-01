import os
import re
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from fpdf import FPDF
from fpdf.enums import XPos, YPos
import rasterio
import cv2
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import PeftModel

# ==========================================
# 1. CONFIGURACION Y LIMPIEZA
# ==========================================
os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

MODEL_ID = "meta-llama/Llama-3.1-8B-Instruct"
ADAPTER_PATH = "./agro_llama_8b_contrafactual"
CSV_PATH = "datos/Dataset_Multimodal_S3_Tesis.csv"
IMG_DIR = "datos/GEE_Tesis_ImageChips-20260217T075250Z-1-001/GEE_Tesis_ImageChips"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def clean_llm_output(text):
    """Limpia salida y elimina repeticiones espurias."""
    if not text:
        return ""

    text = text.replace("\r", "").strip()
    lines = [ln.strip() for ln in text.split("\n") if ln.strip()]

    kept = []
    seen = set()

    for line in lines:
        upper = line.upper()
        is_relevant = (
            line.startswith("-")
            or "NDVI" in upper
            or "T+5" in upper
            or "PRESCRIPCIÓN:" in upper
            or "PRESCRIPCION:" in upper
            or "CONCLUSIÓN:" in upper
            or "CONCLUSION:" in upper
            or "JUSTIFICACIÓN:" in upper
            or "JUSTIFICACION:" in upper
            or "ANÁLISIS CONTRAFACTUAL" in upper
            or "ANALISIS CONTRAFACTUAL" in upper
        )

        if not is_relevant:
            continue

        # Desduplica líneas idénticas para evitar bucles tipo "CONTRAFACTUAL".
        key = re.sub(r"\s+", " ", upper)
        if key in seen:
            continue
        seen.add(key)
        kept.append(line)

    if not kept:
        return text.split("<|eot_id|>")[0].strip()

    # Mantén solo un encabezado contrafactual al inicio.
    out = []
    heading_added = False
    for line in kept:
        upper = line.upper()
        if "ANÁLISIS CONTRAFACTUAL" in upper or "ANALISIS CONTRAFACTUAL" in upper:
            if heading_added:
                continue
            heading_added = True
        out.append(line)

    return "\n".join(out)


def sanitize_for_pdf(text):
    """Normaliza texto del LLM para evitar desbordes en FPDF."""
    if not text:
        return ""
    text = text.replace("\t", " ").replace("\r", "")
    text = re.sub(r"[ ]{2,}", " ", text)
    text = re.sub(r"(\S{45})(?=\S)", r"\1 ", text)
    return text.strip()


# ==========================================
# 2. CLASE PDF
# ==========================================
class AgroPDF(FPDF):
    def header(self):
        self.set_fill_color(46, 125, 50)
        self.rect(0, 0, 210, 25, "F")
        self.set_font("helvetica", "B", 18)
        self.set_text_color(255)
        self.set_y(8)
        self.cell(0, 10, "INFORME DE PRESCRIPCION AGRO-XAI", align="C", new_x=XPos.LMARGIN, new_y=YPos.NEXT)
        self.ln(10)

    def chapter_title(self, label):
        self.set_font("helvetica", "B", 12)
        self.set_fill_color(240, 240, 240)
        self.set_text_color(46, 125, 50)
        self.cell(0, 8, f"  {label}", fill=True, new_x=XPos.LMARGIN, new_y=YPos.NEXT)
        self.ln(3)


# ==========================================
# 3. GENERADOR DE INFORMES
# ==========================================
class AgroProductionReport:
    def __init__(self):
        print(">>> Cargando modelos y datos...")
        self.df = pd.read_csv(CSV_PATH, parse_dates=["Fecha"])

        if "Lluvia_Forecast_Robust" not in self.df.columns:
            if "Lluvia_Prevista_3d" in self.df.columns:
                self.df["Lluvia_Forecast_Robust"] = self.df["Lluvia_Prevista_3d"].apply(
                    lambda x: max(0.0, x + np.random.normal(0, x * 0.2)) if x > 0 else max(0.0, np.random.normal(0, 0.2))
                )
            else:
                self.df["Lluvia_Forecast_Robust"] = self.df["Lluvia_AEMET"]

        self.tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, local_files_only=True, add_bos_token=False)
        self.tokenizer.pad_token = self.tokenizer.eos_token

        bnb_config = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype=torch.bfloat16)
        base_model = AutoModelForCausalLM.from_pretrained(
            MODEL_ID,
            quantization_config=bnb_config,
            device_map="auto",
            local_files_only=True,
        )
        self.llm = PeftModel.from_pretrained(base_model, ADAPTER_PATH)
        self.llm.eval()

    @staticmethod
    def _crop_status(ndvi):
        if ndvi >= 0.65:
            return "Alto vigor vegetativo"
        if ndvi >= 0.45:
            return "Vigor medio (seguimiento activo)"
        return "Vigor bajo (riesgo de estres hidrico)"

    @staticmethod
    def _safe_multicell(pdf, text, line_h=7):
        usable_w = pdf.w - pdf.l_margin - pdf.r_margin
        pdf.set_x(pdf.l_margin)
        pdf.multi_cell(usable_w, line_h, sanitize_for_pdf(text))

    def _prepare_img(self, date):
        files = [f for f in os.listdir(IMG_DIR) if f.endswith(".tif")]
        for f in files:
            if date.strftime("%Y") in f:
                with rasterio.open(os.path.join(IMG_DIR, f)) as src:
                    img = src.read([3, 2, 1])

                img = np.moveaxis(img, 0, -1).astype(np.float32)
                p2, p98 = np.percentile(img, (2, 98))
                img = np.clip((img - p2) / (p98 - p2 + 1e-10), 0, 1)
                img_8bit = (img * 255).astype(np.uint8)

                # Recorte centrado 1:1 para evitar deformacion en el PDF.
                h, w = img_8bit.shape[:2]
                side = min(h, w)
                y0 = (h - side) // 2
                x0 = (w - side) // 2
                img_sq = img_8bit[y0:y0 + side, x0:x0 + side]
                img_sq = cv2.resize(img_sq, (900, 900), interpolation=cv2.INTER_CUBIC)

                cv2.imwrite("temp_sat.png", cv2.cvtColor(img_sq, cv2.COLOR_RGB2BGR))
                return "temp_sat.png"
        return None

    def generate(self, date_str):
        date_dt = pd.to_datetime(date_str)
        row = self.df[self.df["Fecha"] == date_dt].iloc[0]
        lluvia = float(row.get("Lluvia_Forecast_Robust", row["Lluvia_Prevista_3d"]))

        balance = float(row["ET0_Calculada"] - lluvia)
        estado = self._crop_status(float(row["Sat_NDVI"]))

        # 1. Grafico limpio sin solape de etiquetas.
        fig, ax = plt.subplots(figsize=(4.8, 3.2))
        labels = ["ET0", "Lluvia"]
        values = [row["ET0_Calculada"], lluvia]
        bars = ax.bar(labels, values, color=["#e67e22", "#3498db"], width=0.55)
        ax.set_title("Balance Hidrico (mm)", fontsize=14, pad=8)
        ax.set_ylabel("mm")
        ax.grid(axis="y", alpha=0.3)
        ax.tick_params(axis="x", labelsize=11)

        for bar, value in zip(bars, values):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.04, f"{value:.2f}", ha="center", va="bottom", fontsize=9)

        fig.tight_layout(pad=1.0)
        fig.savefig("temp_plot.png", dpi=160)
        plt.close(fig)

        img_path = self._prepare_img(date_dt)

        # 2. LLM Prompt
        prompt = (
            f"<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\n"
            f"Eres un agronomo experto en riego deficitario controlado y XAI. Responde en espanol tecnico, breve y sin repetir frases.<|eot_id|>"
            f"<|start_header_id|>user<|end_header_id|>\n\n"
            f"Datos del Sector 3:\n"
            f"- NDVI Actual: {row['Sat_NDVI']:.2f}\n"
            f"- ET0: {row['ET0_Calculada']:.1f} mm\n"
            f"- Lluvia prevista: {lluvia:.1f} mm\n\n"
            f"Devuelve exactamente este formato:\n"
            f"ANALISIS CONTRAFACTUAL SECTOR 3:\n"
            f"- Escenario 0 mm: ...\n"
            f"- Escenario recomendado: ...\n"
            f"- Escenario exceso: ...\n"
            f"PRESCRIPCION: ...\n"
            f"CONCLUSION: ...<|eot_id|>"
            f"<|start_header_id|>assistant<|end_header_id|>\n\n"
        )

        inputs = self.tokenizer(prompt, return_tensors="pt").to(device)

        with torch.no_grad():
            outputs = self.llm.generate(
                **inputs,
                max_new_tokens=220,
                do_sample=False,
                repetition_penalty=1.15,
                no_repeat_ngram_size=4,
                eos_token_id=self.tokenizer.eos_token_id,
                pad_token_id=self.tokenizer.eos_token_id,
            )

        # Decodifica solo tokens generados para evitar eco del prompt.
        gen_tokens = outputs[0][inputs["input_ids"].shape[1]:]
        raw_advice = self.tokenizer.decode(gen_tokens, skip_special_tokens=True).strip()

        if "ANALISIS CONTRAFACTUAL" not in raw_advice.upper() and "ANÁLISIS CONTRAFACTUAL" not in raw_advice.upper():
            raw_advice = "ANALISIS CONTRAFACTUAL SECTOR 3:\n" + raw_advice

        clean_advice = clean_llm_output(raw_advice)

        # 3. Crear PDF
        pdf = AgroPDF()
        pdf.add_page()

        pdf.set_font("helvetica", "", 8)
        pdf.set_text_color(120)
        pdf.cell(0, 8, f"ID: Sector 3 | Fecha: {date_str} | Area: 40.650 m2", align="R", new_x=XPos.LMARGIN, new_y=YPos.NEXT)
        pdf.ln(2)

        # SECCION 1: IMAGENES
        y_imgs = pdf.get_y()

        if img_path:
            pdf.image(img_path, x=15, y=y_imgs, w=70, h=70)
            pdf.set_xy(15, y_imgs + 71)
            pdf.set_font("helvetica", "I", 9)
            pdf.set_text_color(90)
            pdf.cell(70, 5, "Imagen satelital (1:1)", align="C")

        pdf.image("temp_plot.png", x=100, y=y_imgs, w=90, h=70)
        pdf.set_xy(100, y_imgs + 71)
        pdf.set_font("helvetica", "I", 9)
        pdf.set_text_color(90)
        pdf.cell(90, 5, "Balance ET0 vs lluvia", align="C")

        pdf.set_y(y_imgs + 80)
        pdf.ln(5)

        # SECCION RESUMEN
        pdf.chapter_title("RESUMEN OPERATIVO")
        pdf.set_font("helvetica", "", 10)
        pdf.set_text_color(0)
        pdf.cell(95, 8, f"Estado del cultivo: {estado}")
        pdf.cell(95, 8, f"Lluvia prevista (robusta): {lluvia:.2f} mm", new_x=XPos.LMARGIN, new_y=YPos.NEXT)
        pdf.cell(95, 8, f"NDVI actual: {row['Sat_NDVI']:.3f}")
        pdf.cell(95, 8, f"Balance hidrico (ET0-lluvia): {balance:+.2f} mm", new_x=XPos.LMARGIN, new_y=YPos.NEXT)
        pdf.ln(2)

        # SECCION 2: RAZONAMIENTO
        pdf.chapter_title("AUDITORIA DEL AGENTE IA")
        pdf.set_font("helvetica", "", 11)
        pdf.set_text_color(0)

        for line in clean_advice.split("\n"):
            line = sanitize_for_pdf(line)
            if not line:
                pdf.ln(2)
                continue
            if "PRESCRIPCIÓN:" in line.upper() or "PRESCRIPCION:" in line.upper():
                pdf.set_font("helvetica", "B", 12)
                pdf.set_text_color(46, 125, 50)
                self._safe_multicell(pdf, line)
                pdf.set_text_color(0)
            elif "CONCLUSIÓN:" in line.upper() or "CONCLUSION:" in line.upper():
                pdf.set_font("helvetica", "I", 11)
                self._safe_multicell(pdf, line)
            else:
                pdf.set_font("helvetica", "", 11)
                self._safe_multicell(pdf, line)

        pdf.ln(5)

        # SECCION 3: DATOS
        pdf.chapter_title("DATOS DE CAMPO")
        pdf.set_font("helvetica", "", 10)
        pdf.cell(95, 8, f"Vigor Actual (NDVI): {row['Sat_NDVI']:.3f}")
        pdf.cell(95, 8, f"Evapotranspiracion: {row['ET0_Calculada']:.2f} mm", new_x=XPos.LMARGIN, new_y=YPos.NEXT)
        pdf.cell(95, 8, f"Lluvia Prevista: {lluvia:.2f} mm")
        pdf.cell(95, 8, f"Riego Historico: {row['Riego_mm_Real']:.2f} mm", new_x=XPos.LMARGIN, new_y=YPos.NEXT)

        out_name = f"Informe_Final_{date_str}.pdf"
        pdf.output(out_name)
        print(f"Informe generado: {out_name}")


if __name__ == "__main__":
    gen = AgroProductionReport()
    gen.generate("2024-08-10")
