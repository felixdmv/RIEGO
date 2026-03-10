import os
import re
import torch
import numpy as np
import pandas as pd
import tensorflow as tf
import rasterio
import cv2
from datetime import datetime
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
from sklearn.preprocessing import StandardScaler

# ==========================================
# 1. CONFIGURACIÓN (NOMBRES REALES DEL CSV)
# ==========================================
BASE_PATH = 'datos'
IMG_DIR = os.path.join(BASE_PATH, 'GEE_Tesis_ImageChips-20260217T075250Z-1-001/GEE_Tesis_ImageChips')
ADAPTER_ID = "./agro_llm_final"
MODEL_ID = "meta-llama/Llama-3.2-1B-Instruct"

# Nombres exactos detectados en tu CSV anterior
COLS_TAB = ['Temp_Sensor_mean', 'Rad_Sensor', 'Lluvia_Sensor', 'Riego_mm']

# ==========================================
# 2. CARGA DE DATOS (SINCRONIZADA)
# ==========================================
print(">>> Cargando base de datos consolidada...")

def preparar_datos_directos():
    ruta_maestra = os.path.join(BASE_PATH, 'datos_multimodales_completos.csv')
    if not os.path.exists(ruta_maestra):
        raise FileNotFoundError(f"No se encuentra {ruta_maestra}. Ejecuta el pipeline primero.")
    
    df = pd.read_csv(ruta_maestra, parse_dates=['Fecha'])
    
    # Verificamos si las columnas existen antes de limpiar
    for col in COLS_TAB + ['Sat_NDVI']:
        if col not in df.columns:
            raise KeyError(f"Falta la columna '{col}' en el CSV. Columnas actuales: {df.columns.tolist()}")

    df = df.dropna(subset=COLS_TAB + ['Sat_NDVI']).reset_index(drop=True)
    
    scaler = StandardScaler()
    df[COLS_TAB] = scaler.fit_transform(df[COLS_TAB])
    return df

df_test = preparar_datos_directos()

class SimpleDataGen:
    def __init__(self, df, img_dir):
        self.df = df
        self.img_dir = img_dir
        self.img_files = sorted([f for f in os.listdir(img_dir) if f.endswith('.tif')])
        parsed = []
        valid_files = []
        for f in self.img_files:
            m = re.search(r'(\d{4})-\d{2}-(\d{1,3})', f)
            if m:
                try:
                    fecha = datetime.strptime(f"{m.group(1)}-{m.group(2)}", "%Y-%j")
                    parsed.append(pd.Timestamp(fecha))
                    valid_files.append(f)
                except: continue
        self.img_dates = pd.to_datetime(parsed)
        self.valid_files = valid_files
        print(f">>> Generador listo con {len(self.valid_files)} imágenes vinculadas.")

    def get_sample(self, idx):
        date = self.df.iloc[idx]['Fecha']
        img_idx = np.argmin(np.abs(self.img_dates - date))
        with rasterio.open(os.path.join(self.img_dir, self.valid_files[img_idx])) as src:
            img = src.read([3,2,1])
            img = np.moveaxis(img, 0, -1).astype(np.float32)
            p2, p98 = np.percentile(img, (2, 98))
            img = np.clip((img - p2) / (p98 - p2 + 1e-10), 0, 1)
            img = cv2.resize(img, (64,64))
        return self.df.iloc[idx][COLS_TAB].values.astype(np.float32), img

gen_val = SimpleDataGen(df_test, IMG_DIR)

# ==========================================
# 3. CARGA DE MODELOS
# ==========================================
print(">>> Cargando modelos...")
model_dl = tf.keras.models.load_model('modelo_biologico_multimodal.h5', compile=False)

tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
base_model = AutoModelForCausalLM.from_pretrained(MODEL_ID, torch_dtype=torch.float16, device_map="auto")
model_llm = PeftModel.from_pretrained(base_model, ADAPTER_ID)
model_llm.eval()

# ==========================================
# 4. INFERENCIA Y RECOMENDACIÓN
# ==========================================
def ejecutar_analisis_final(idx_dia=None):
    if idx_dia is None: idx_dia = len(df_test) - 1
    
    print(f">>> Analizando día: {df_test.iloc[idx_dia]['Fecha'].date()}")

    x_tab, x_img = gen_val.get_sample(idx_dia)
    ndvi_actual = df_test.iloc[idx_dia]['Sat_NDVI']
    
    # 1. Predicción biológica
    pred_ndvi_t5 = model_dl.predict([x_tab.reshape(1,-1), x_img.reshape(1,64,64,3)], verbose=0)[0][0]
    
    # 2. Construcción de narrativa
    dif = ((pred_ndvi_t5 - ndvi_actual) / (ndvi_actual + 1e-10)) * 100
    narrativa = (
        f"ESTADO SATELITAL: NDVI {ndvi_actual:.3f}. Proyección 5 días: {pred_ndvi_t5:.3f} ({dif:+.1f}%). "
        f"SENSORES: Temp Sensor std {x_tab[0]:.2f}. "
        f"METEO: Datos de Sardegna de Duero (AEMET)."
    )
    
    # 3. Respuesta del LLM
    prompt = f"I: Eres el Orquestador Agrónomo. Decide el riego.\nInput: {narrativa}\nAns:"
    inputs = tokenizer(prompt, return_tensors="pt").to("cuda")
    
    with torch.no_grad():
        outputs = model_llm.generate(**inputs, max_new_tokens=100, temperature=0.1, repetition_penalty=1.2)
    
    respuesta = tokenizer.decode(outputs[0], skip_special_tokens=True).split("Ans:")[-1].strip()
    
    print("\n" + "="*60)
    print(f"  RECOMENDACIÓN FINAL IA")
    print("="*60)
    print(f"EVEDENCIA: {narrativa}")
    print(f"\nDECISIÓN: {respuesta}")
    print("="*60)

ejecutar_analisis_final()