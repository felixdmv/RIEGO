# --- 1. CONFIGURACIÓN DE ENTORNO Y GPU ---
import os

# FORZAR MODO OFFLINE (Solución al error Network unreachable)
os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.environ["HF_DATASETS_OFFLINE"] = "1"

import subprocess
import re
import json
import logging
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
import rasterio
import cv2
from datetime import datetime
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

# ==========================================
# 1. CONFIGURACIÓN DE ENTORNO Y GPU
# ==========================================
def get_best_gpu():
    try:
        command = "nvidia-smi --query-gpu=memory.free --format=csv,nounits,noheader"
        free_memory = [int(x) for x in subprocess.check_output(command.split()).decode('utf-8').strip().split('\n')]
        return np.argmax(free_memory)
    except: return 0

BEST_GPU = get_best_gpu()
os.environ["CUDA_VISIBLE_DEVICES"] = str(BEST_GPU)
# Estabilidad para Kernel 4.18
os.environ["NCCL_P2P_DISABLE"] = "1"
os.environ["NCCL_IB_DISABLE"] = "1"
os.environ["NCCL_SHM_DISABLE"] = "1"
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, Trainer, TrainingArguments, DataCollatorForLanguageModeling
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from datasets import Dataset as HFDataset

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# ==========================================
# 2. PROCESAMIENTO DE DATOS (SECTOR 3 + SATÉLITE)
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

    def process_all(self):
        logger.info("Fase 1: Consolidando dataset multimodal Sector 3...")
        
        # 1. Sensores
        list_s = [pd.read_csv(os.path.join(self.carpeta, f)) for f in self.archivos_sensores]
        df_s = pd.concat(list_s)
        df_s['Fecha'] = pd.to_datetime(df_s['Fecha'], errors='coerce')
        df_clima = df_s.dropna(subset=['Fecha']).set_index('Fecha').resample('D').agg({
            'Temperatura': 'mean', 'Radiación solar': 'mean', 'Precipitaciones': 'sum'
        }).reset_index()
        df_clima['ET0_Calculada'] = 0.0135 * (df_clima['Temperatura'] + 17.8) * (df_clima['Radiación solar'] * 0.035)

        # 2. AEMET + Forecast simulado (3 días vista)
        df_aemet = pd.read_csv(os.path.join(self.carpeta, self.archivo_aemet), parse_dates=['Fecha'])
        df_aemet = df_aemet.sort_values('Fecha')
        df_aemet['Lluvia_Prevista_3d'] = df_aemet['Lluvia_AEMET'].shift(-1).rolling(3, min_periods=1).sum().shift(-2).fillna(0)

        # 3. Riegos Sector 3
        df_r = pd.read_csv(os.path.join(self.carpeta, self.archivo_riegos))
        df_r['Sectores de Riego'] = pd.to_numeric(df_r['Sectores de Riego'], errors='coerce')
        df_r['Agua m3'] = df_r['Agua m³'].astype(str).str.replace('.', '', regex=False).str.replace(',', '.', regex=False)
        df_r['Agua m3'] = pd.to_numeric(df_r['Agua m3'], errors='coerce')
        df_r['Fecha'] = pd.to_datetime(df_r['Inicio'].astype(str).str[:10], dayfirst=True, errors='coerce')
        df_s3 = df_r[df_r['Sectores de Riego'] == 3].groupby('Fecha')['Agua m3'].sum().reset_index()
        df_s3['Riego_mm_Real'] = (df_s3['Agua m3'] * 1000) / self.area_s3

        # 4. Merge y Satélites
        df_master = pd.merge(df_clima, df_aemet[['Fecha', 'Lluvia_Prevista_3d']], on='Fecha', how='left')
        df_master = pd.merge(df_master, df_s3[['Fecha', 'Riego_mm_Real']], on='Fecha', how='left').fillna(0)

        for col, path in self.archivos_sat.items():
            df_sat = pd.read_csv(os.path.join(self.carpeta, path), parse_dates=['Fecha']).rename(columns={'Valor': col})
            df_master = pd.merge(df_master, df_sat[['Fecha', col]], on='Fecha', how='left')
            df_master[col] = df_master[col].interpolate(method='linear', limit_direction='both')

        df_master['Kc_Dinamico'] = 0.4 + (df_master['Sat_NDVI'] * 0.8)
        df_master['ETc_Sector3'] = df_master['ET0_Calculada'] * df_master['Kc_Dinamico']
        
        df_master.to_csv(os.path.join(self.carpeta, 'Dataset_Final_S3.csv'), index=False)
        return df_master

# ==========================================
# 3. MODELO DE PERCEPCIÓN (PYTORCH)
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
        self.img_files = sorted([f for f in os.listdir(img_dir) if f.endswith('.tif')])
        parsed = []
        for f in self.img_files:
            m = re.search(r'(\d{4})-\d{2}-(\d{1,3})', f)
            if m: parsed.append(pd.Timestamp(datetime.strptime(f"{m.group(1)}-{m.group(2)}", "%Y-%j")))
        self.img_dates = pd.to_datetime(parsed)

    def __len__(self): return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        tab = torch.tensor(row[self.cols_tab].values.astype(np.float32))
        dist = np.abs(self.img_dates - row['Fecha'])
        with rasterio.open(os.path.join(self.img_dir, self.img_files[np.argmin(dist)])) as src:
            img = src.read([3,2,1]).astype(np.float32)
        img = np.moveaxis(img, 0, -1)
        p2, p98 = np.percentile(img, (2, 98))
        img = np.clip((img - p2) / (p98 - p2 + 1e-10), 0, 1)
        img = cv2.resize(img, (64,64))
        img_t = torch.from_numpy(img).permute(2,0,1).float()
        target = torch.tensor([row['Target_NDVI']], dtype=torch.float32)
        return img_t, tab, target

# ==========================================
# 4. AGENTE LLM (FINE-TUNING COT)
# ==========================================
class AgroPipelineMaster:
    def __init__(self):
        self.data_manager = AgroDataManager()
        self.img_dir = 'datos/GEE_Tesis_ImageChips-20260217T075250Z-1-001/GEE_Tesis_ImageChips'
        self.cols_input = ['Temperatura', 'Lluvia_Prevista_3d', 'Riego_mm_Real', 'ET0_Calculada', 'Sat_Radar', 'Sat_LST']

    def run_perception(self, df):
        logger.info("Fase 2: Entrenando Percepción...")
        df['Target_NDVI'] = df['Sat_NDVI'].shift(-5)
        df = df.dropna(subset=self.cols_input + ['Target_NDVI']).copy()
        
        scaler = StandardScaler()
        df[self.cols_input] = scaler.fit_transform(df[self.cols_input])
        
        train_df, val_df = train_test_split(df, test_size=0.2, shuffle=False)
        train_loader = DataLoader(AgroDatasetPT(train_df, self.img_dir, self.cols_input), batch_size=16, shuffle=True)
        
        model = AgroTwoTower(len(self.cols_input)).to(device)
        opt = optim.Adam(model.parameters(), lr=1e-3)
        crit = nn.MSELoss()

        model.train()
        for e in range(50):
            l_total = 0
            for imgs, tabs, y in train_loader:
                imgs, tabs, y = imgs.to(device), tabs.to(device), y.to(device)
                opt.zero_grad(); out = model(imgs, tabs); loss = crit(out, y); loss.backward(); opt.step()
                l_total += loss.item()
            if (e+1) % 10 == 0: logger.info(f"Epoch {e+1} Loss: {l_total/len(train_loader):.4f}")
        
        torch.save(model.state_dict(), 'modelo_percepcion_s3.pt')
        return model

    def run_llm_training(self, df):
        logger.info("Fase 3: Generando dataset CoT y Fine-Tuning...")
        dataset_llm = []
        for _, row in df.iterrows():
            kc, demanda = row['Kc_Dinamico'], row['ETc_Sector3']
            lluvia_p, riego_h = row['Lluvia_Prevista_3d'], row['Riego_mm_Real']
            dosis_ideal = max(0, (demanda * 0.7 * 3) - lluvia_p)
            varianza = riego_h - (dosis_ideal / 3)

            output_text = (
                f"ANÁLISIS AGRO-XAI: Kc {kc:.2f}, Demanda {demanda:.1f}mm/día. "
                f"AUDITORÍA: Aplicados {riego_h:.1f}mm. Con lluvia prevista de {lluvia_p:.1f}mm, "
                f"la dosis ideal era {dosis_ideal/3:.1f}mm. VARIANZA: {varianza:+.1f}mm. "
                f"PRESCRIPCIÓN: {'Mantener' if abs(varianza)<2 else 'Reducir' if varianza>0 else 'Aumentar'} riego."
            )
            dataset_llm.append({
                "instruction": "Audita la eficiencia del riego en el Sector 3.",
                "input": f"NDVI: {row['Sat_NDVI']:.2f}, ET0: {row['ET0_Calculada']:.1f}mm, Lluvia_Prevista: {lluvia_p:.1f}mm, Riego: {riego_h:.1f}mm",
                "output": output_text
            })

        with open('datos/dataset_cot_s3.jsonl', 'w') as f:
            for e in dataset_llm: f.write(json.dumps(e) + '\n')

        # Configuración LLM
        model_id = "meta-llama/Llama-3.1-8B-Instruct"
        token = "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"
        bnb_config = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=torch.bfloat16)
        
        tokenizer = AutoTokenizer.from_pretrained(model_id, token=token)
        tokenizer.pad_token = tokenizer.eos_token
        
        base_model = AutoModelForCausalLM.from_pretrained(model_id, quantization_config=bnb_config, device_map={"": 0}, token=token)
        base_model = prepare_model_for_kbit_training(base_model)
        
        peft_config = LoraConfig(r=16, lora_alpha=32, target_modules=["q_proj", "v_proj"], task_type="CAUSAL_LM")
        model = get_peft_model(base_model, peft_config)

        ds = HFDataset.from_json('datos/dataset_cot_s3.jsonl')
        def tokenize(b):
            texts = [f"<|begin_of_text|>system\nExpert AI\nuser\n{i} {inp}\nassistant\n{out}<|end_of_text|>" for i, inp, out in zip(b['instruction'], b['input'], b['output'])]
            return tokenizer(texts, truncation=True, padding="max_length", max_length=512)
        
        args = TrainingArguments(output_dir="./agro_llama_final", per_device_train_batch_size=1, gradient_accumulation_steps=8, max_steps=150, learning_rate=1e-4, bf16=True, logging_steps=10)
        trainer = Trainer(model=model, train_dataset=ds.map(tokenize, batched=True), args=args, data_collator=DataCollatorForLanguageModeling(tokenizer, mlm=False))
        
        trainer.train()
        model.save_pretrained("./agro_llama_8b_s3_final")

if __name__ == "__main__":
    pipeline = AgroPipelineMaster()
    df_master = pipeline.data_manager.process_all()
    pipeline.run_perception(df_master)
    pipeline.run_llm_training(df_master)
    logger.info("=== PIPELINE SECTOR 3 FINALIZADO ===")