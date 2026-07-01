# ==========================================
# 1. CONFIGURACIÓN DE ENTORNO Y GPU (ESTABILIDAD)
# ==========================================
import os
import subprocess
import numpy as np

# Forzar modo Offline para evitar errores de red en el clúster
os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.environ["HF_DATASETS_OFFLINE"] = "1"

# Desactivar protocolos de comunicación que fallan en Kernel 4.18
os.environ["NCCL_P2P_DISABLE"] = "1"
os.environ["NCCL_IB_DISABLE"] = "1"
os.environ["NCCL_SHM_DISABLE"] = "1"
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

def get_best_gpu():
    try:
        command = "nvidia-smi --query-gpu=memory.free --format=csv,nounits,noheader"
        free_memory = [int(x) for x in subprocess.check_output(command.split()).decode('utf-8').strip().split('\n')]
        return np.argmax(free_memory)
    except: return 0

BEST_GPU = get_best_gpu()
os.environ["CUDA_VISIBLE_DEVICES"] = str(BEST_GPU)

import re
import json
import logging
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import rasterio
import cv2
from datetime import datetime
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

# Importaciones de LLM
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, Trainer, TrainingArguments, DataCollatorForLanguageModeling
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from datasets import Dataset as HFDataset

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ==========================================
# 2. GESTOR DE DATOS (FUSIÓN + RUIDO)
# ==========================================
class AgroDataManager:
    def __init__(self):
        self.area_s3 = 40650
        self.carpeta = 'datos'
        self.archivos_sensores = ['DATOSE_1.csv', 'DATOSE_2.csv', 'DATOSE_3.csv']
        self.archivo_riegos = 'RIEGOS_1.csv'
        self.archivo_aemet = 'AEMET_Sardon_historico.csv'
        self.archivos_sat = {
            'Sat_NDVI': 'satelite/Sentinel2_NDVI_Parcela_Real.csv',
            'Sat_Radar': 'satelite/Sentinel1_Radar_Parcela_Real.csv',
            'Sat_LST': 'satelite/MODIS_Temp_Parcela_Real.csv'
        }

    def apply_forecast_noise(self, rain_val):
        """Simula incertidumbre real del pronóstico (20% de error gaussiano)"""
        if rain_val <= 0: return max(0.0, np.random.normal(0, 0.2))
        return max(0.0, rain_val + np.random.normal(0, rain_val * 0.2))

    def process_all(self):
        logger.info("Fase 1: Consolidando dataset con robustez meteorológica...")
        
        # 1. Carga de Sensores
        list_s = []
        for f in self.archivos_sensores:
            ruta = os.path.join(self.carpeta, f)
            if os.path.exists(ruta): list_s.append(pd.read_csv(ruta))
        df_s = pd.concat(list_s)
        df_s['Fecha'] = pd.to_datetime(df_s['Fecha'], errors='coerce')
        df_clima = df_s.dropna(subset=['Fecha']).set_index('Fecha').resample('D').agg({
            'Temperatura': 'mean', 'Radiación solar': 'mean', 'Precipitaciones': 'sum'
        }).reset_index()
        df_clima['ET0_Calculada'] = 0.0135 * (df_clima['Temperatura'] + 17.8) * (df_clima['Radiación solar'] * 0.035)

        # 2. AEMET + Pronóstico con Ruido (Para evitar trampas)
        df_aemet = pd.read_csv(os.path.join(self.carpeta, self.archivo_aemet), parse_dates=['Fecha'])
        df_aemet = df_aemet.sort_values('Fecha')
        # Lluvia prevista a 3 días (Perfect forecast)
        df_aemet['Lluvia_Prevista_3d'] = df_aemet['Lluvia_AEMET'].shift(-1).rolling(3, min_periods=1).sum().shift(-2).fillna(0)
        # Aplicar ruido para robustez
        df_aemet['Lluvia_Forecast_Robust'] = df_aemet['Lluvia_Prevista_3d'].apply(self.apply_forecast_noise)

        # 3. Riegos Sector 3 (Exactitud Quirúrgica)
        df_r = pd.read_csv(os.path.join(self.carpeta, self.archivo_riegos))
        df_r['Sectores de Riego'] = pd.to_numeric(df_r['Sectores de Riego'], errors='coerce')
        df_r['Agua m3'] = df_r['Agua m³'].astype(str).str.replace('.', '', regex=False).str.replace(',', '.', regex=False)
        df_r['Agua m3'] = pd.to_numeric(df_r['Agua m3'], errors='coerce')
        df_r['Fecha'] = pd.to_datetime(df_r['Inicio'].astype(str).str[:10], dayfirst=True, errors='coerce')
        df_s3 = df_r[df_r['Sectores de Riego'] == 3].groupby('Fecha')['Agua m3'].sum().reset_index()
        df_s3['Riego_mm_Real'] = (df_s3['Agua m3'] * 1000) / self.area_s3

        # 4. Merge y Satélites
        df_master = pd.merge(df_clima, df_aemet[['Fecha', 'Lluvia_Forecast_Robust']], on='Fecha', how='left')
        df_master = pd.merge(df_master, df_s3[['Fecha', 'Riego_mm_Real']], on='Fecha', how='left').fillna(0)

        for col, path in self.archivos_sat.items():
            df_sat = pd.read_csv(os.path.join(self.carpeta, path), parse_dates=['Fecha']).rename(columns={'Valor': col})
            df_master = pd.merge(df_master, df_sat[['Fecha', col]], on='Fecha', how='left')
            df_master[col] = df_master[col].interpolate(method='linear', limit_direction='both')

        df_master['Kc_Dinamico'] = 0.4 + (df_master['Sat_NDVI'] * 0.8)
        df_master['ETc_Sector3'] = df_master['ET0_Calculada'] * df_master['Kc_Dinamico']
        
        return df_master

# ==========================================
# 3. MODELO PERCEPCIÓN (CNN-MLP)
# ==========================================
class AgroTwoTower(nn.Module):
    def __init__(self, tab_size):
        super().__init__()
        self.cnn = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(), nn.AdaptiveAvgPool2d(1)
        )
        self.mlp = nn.Sequential(nn.Linear(tab_size, 16), nn.ReLU())
        self.fusion = nn.Linear(64 + 16, 1)

    def forward(self, img, tab):
        v_img = self.cnn(img).view(img.size(0), -1)
        v_tab = self.mlp(tab)
        return self.fusion(torch.cat((v_img, v_tab), dim=1))

class AgroDatasetPT(Dataset):
    def __init__(self, df, img_dir, cols_tab):
        self.df = df.reset_index(drop=True)
        self.img_dir = img_dir
        self.cols_tab = cols_tab
        img_files = sorted([f for f in os.listdir(img_dir) if f.endswith('.tif')])
        parsed = []
        for f in img_files:
            m = re.search(r'(\d{4})-\d{2}-(\d{1,3})', f)
            if m: parsed.append({'file': f, 'date': pd.Timestamp(datetime.strptime(f"{m.group(1)}-{m.group(2)}", "%Y-%j"))})
        self.img_idx = pd.DataFrame(parsed)

    def __len__(self): return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        tab = torch.tensor(row[self.cols_tab].values.astype(np.float32))
        dist = np.abs(self.img_idx['date'] - row['Fecha'])
        img_path = os.path.join(self.img_dir, self.img_idx.iloc[np.argmin(dist)]['file'])
        with rasterio.open(img_path) as src:
            img = src.read([3,2,1]).astype(np.float32)
        img = np.moveaxis(img, 0, -1)
        p2, p98 = np.percentile(img, (2, 98))
        img = np.clip((img - p2) / (p98 - p2 + 1e-10), 0, 1)
        img = cv2.resize(img, (64,64))
        return torch.from_numpy(img).permute(2,0,1).float(), tab, torch.tensor([row['Target_NDVI']], dtype=torch.float32)

# ==========================================
# 4. PIPELINE PRINCIPAL (SIMULACIÓN + LLM)
# ==========================================
class AgroPipelineFinal:
    def __init__(self):
        self.manager = AgroDataManager()
        self.scaler = StandardScaler()
        self.img_dir = 'datos/GEE_Tesis_ImageChips-20260217T075250Z-1-001/GEE_Tesis_ImageChips'
        self.cols_input = ['Temperatura', 'Lluvia_Forecast_Robust', 'Riego_mm_Real', 'ET0_Calculada', 'Sat_Radar', 'Sat_LST']
        self.model_percepcion = None

    def train_vision(self, df):
        logger.info("Fase 2: Entrenando Percepción Biológica...")
        df['Target_NDVI'] = df['Sat_NDVI'].shift(-5)
        df_clean = df.dropna(subset=self.cols_input + ['Target_NDVI']).copy()
        df_clean[self.cols_input] = self.scaler.fit_transform(df_clean[self.cols_input])
        
        train_df, _ = train_test_split(df_clean, test_size=0.1, shuffle=False)
        loader = DataLoader(AgroDatasetPT(train_df, self.img_dir, self.cols_input), batch_size=16, shuffle=True)
        
        self.model_percepcion = AgroTwoTower(len(self.cols_input)).to(device)
        opt = optim.Adam(self.model_percepcion.parameters(), lr=1e-3)
        crit = nn.MSELoss()

        for e in range(50):
            for imgs, tabs, y in loader:
                imgs, tabs, y = imgs.to(device), tabs.to(device), y.to(device)
                opt.zero_grad(); loss = crit(self.model_percepcion(imgs, tabs), y); loss.backward(); opt.step()
        
        torch.save(self.model_percepcion.state_dict(), 'modelo_percepcion_final.pt')

    def generate_counterfactual_dataset(self, df):
        logger.info("Fase 3: Generando Dataset Ejecutivo (Evitando Truncamiento)...")
        self.model_percepcion.eval()
        dataset_llm = []
        df_sample = df.tail(500) # Verano 2024
        ds_vision = AgroDatasetPT(df_sample, self.img_dir, self.cols_input)

        for i in range(len(df_sample)):
            row = df_sample.iloc[i]
            img_t, _, _ = ds_vision[i]
            
            # Simulamos 3 puntos clave
            scenarios = [0.0, row['ETc_Sector3'], row['ETc_Sector3'] * 1.5]
            preds_ndvi = []
            
            for dose in scenarios:
                tab_vec = row[self.cols_input].copy()
                tab_vec['Riego_mm_Real'] = dose
                tab_scaled = torch.tensor(self.scaler.transform(tab_vec.values.reshape(1, -1)), dtype=torch.float32).to(device)
                with torch.no_grad():
                    preds_ndvi.append(self.model_percepcion(img_t.unsqueeze(0).to(device), tab_scaled).item())

            # --- FORMATO EJECUTIVO (EL SECRETO DEL 20/20) ---
            output_text = (
                f"SIMULACIÓN DE IMPACTO:\n"
                f"- Si no riegas (0mm): NDVI t+5 -> {preds_ndvi[0]:.3f}\n"
                f"- Con riego óptimo ({scenarios[1]:.1f}mm): NDVI t+5 -> {preds_ndvi[1]:.3f}\n"
                f"PRESCRIPCIÓN: {scenarios[1]:.1f}mm.\n"
                f"JUSTIFICACIÓN: El riego propuesto incrementa el vigor un {((preds_ndvi[1]/preds_ndvi[0])-1)*100:.1f}% respecto al secano."
            )
            
            dataset_llm.append({
                "instruction": "Calcula la dosis de riego óptima para el Sector 3 mediante simulación contrafactual.",
                "input": f"Estado: NDVI {row['Sat_NDVI']:.2f}, ET0 {row['ET0_Calculada']:.1f}mm, Previsión Lluvia {row['Lluvia_Forecast_Robust']:.1f}mm.",
                "output": output_text
            })

        with open('datos/dataset_ejecutivo_final.jsonl', 'w') as f:
            for e in dataset_llm: f.write(json.dumps(e) + '\n')

    def train_llm(self):
        logger.info("Fase 4: Fine-Tuning Llama-3.1 (Modo Contrafactual)...")
        model_id = "meta-llama/Llama-3.1-8B-Instruct"
        token = "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"
        
        tokenizer = AutoTokenizer.from_pretrained(model_id, token=token, local_files_only=True)
        tokenizer.pad_token = tokenizer.eos_token
        
        bnb_config = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=torch.bfloat16)
        base_model = AutoModelForCausalLM.from_pretrained(model_id, quantization_config=bnb_config, device_map={"": 0}, token=token, local_files_only=True)
        base_model = prepare_model_for_kbit_training(base_model)
        
        peft_config = LoraConfig(
            r=32, # Doblamos la capacidad
            lora_alpha=64,
            target_modules=["q_proj", "v_proj", "k_proj", "o_proj", "gate_proj", "up_proj", "down_proj"], # Todos los módulos
            task_type="CAUSAL_LM",
            bias="none"
        )
        model = get_peft_model(base_model, peft_config)

        ds = HFDataset.from_json('datos/dataset_contrafactual_final.jsonl')
        def tokenize(b):
            texts = [f"<|begin_of_text|>system\nExpert Agro AI\nuser\n{i} {inp}\nassistant\n{out}<|end_of_text|>" for i, inp, out in zip(b['instruction'], b['input'], b['output'])]
            return tokenizer(texts, truncation=True, padding="max_length", max_length=512)

        args = TrainingArguments(
            output_dir="./agro_llama_ejecutivo_v2", 
            per_device_train_batch_size=2, 
            gradient_accumulation_steps=4, 
            max_steps=500, # Subimos a 500 para fijar el formato
            learning_rate=1e-4, 
            bf16=True, 
            logging_steps=10,
            optim="paged_adamw_32bit",
            save_strategy="no"
        )
        
        trainer = Trainer(model=model, train_dataset=ds.map(tokenize, batched=True), args=args, data_collator=DataCollatorForLanguageModeling(tokenizer, mlm=False))
        trainer.train()
        model.save_pretrained("./agro_llama_8b_contrafactual")

if __name__ == "__main__":
    pipeline = AgroPipelineFinal()
    df_master = pipeline.manager.process_all()
    pipeline.train_vision(df_master)
    pipeline.generate_counterfactual_dataset(df_master)
    pipeline.train_llm()
    logger.info("=== PIPELINE CONTRAFACTUAL SECTOR 3 COMPLETADO ===")