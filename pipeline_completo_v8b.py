import os
import re
import json
import logging
import torch
import numpy as np
import pandas as pd
import rasterio
import cv2
import tensorflow as tf
from datetime import datetime
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from tensorflow.keras import layers, models, Input
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, Trainer, TrainingArguments, DataCollatorForLanguageModeling
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from datasets import Dataset

# Configuración de Logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class AgroPipeline8B:
    def __init__(self, base_path='datos'):
        self.base_path = base_path
        self.img_dir = os.path.join(base_path, 'GEE_Tesis_ImageChips-20260217T075250Z-1-001/GEE_Tesis_ImageChips')
        self.cols_tab = ['Temp_Sensor_mean', 'Rad_Sensor', 'Lluvia_Sensor', 'Riego_mm']
        self.model_dl = None
        self.df_master = None
        self.tokenizer = None
        self.model_llm = None

    # ==========================================
    # 1. CONSOLIDACIÓN DE DATOS (Física Real)
    # ==========================================
    def process_data(self):
        logger.info("Fase 1: Consolidando datos y calculando balance hídrico...")
        df = pd.read_csv('datos/datos_multimodales_completos.csv', parse_dates=['Fecha'])
        
        # Limpieza y escalado para la CNN
        df = df.dropna(subset=['Sat_NDVI', 'ET0_Calibrada']).sort_values('Fecha')
        df['Target_NDVI'] = df['Sat_NDVI'].shift(-5)
        df = df.dropna(subset=['Target_NDVI'])
        
        # Guardamos una copia sin escalar para el razonamiento del LLM
        self.df_raw = df.copy()
        
        scaler = StandardScaler()
        df[self.cols_tab] = scaler.fit_transform(df[self.cols_tab]).astype(np.float32)
        self.df_master = df
        logger.info(f"Dataset consolidado: {len(self.df_master)} días.")

    # ==========================================
    # 2. ENTRENAMIENTO CNN (Percepción)
    # ==========================================
    def train_perception_model(self):
        logger.info("Fase 2: Entrenando modelo de percepción (CNN)...")
        
        class DataGen(tf.keras.utils.Sequence):
            def __init__(self, df, img_dir, cols_tab):
                self.df = df.reset_index(drop=True)
                self.img_dir = img_dir
                self.cols_tab = cols_tab
                self.img_files = sorted([f for f in os.listdir(img_dir) if f.endswith('.tif')])
                parsed = []
                for f in self.img_files:
                    m = re.search(r'(\d{4})-\d{2}-(\d{1,3})', f)
                    if m: 
                        d = datetime.strptime(f"{m.group(1)}-{m.group(2)}", "%Y-%j")
                        parsed.append(pd.Timestamp(d))
                self.img_dates = pd.to_datetime(parsed)

            def __len__(self): return int(np.ceil(len(self.df) / 16))
            
            def load_img(self, date):
                idx = np.argmin(np.abs(self.img_dates - date))
                with rasterio.open(os.path.join(self.img_dir, self.img_files[idx])) as src:
                    img = src.read([3,2,1])
                    img = np.moveaxis(img, 0, -1).astype(np.float32)
                    p2, p98 = np.percentile(img, (2, 98))
                    img = np.clip((img - p2) / (p98 - p2 + 1e-10), 0, 1)
                    return cv2.resize(img, (64,64))

            def __getitem__(self, idx):
                batch = self.df.iloc[idx*16:(idx+1)*16]
                x_tab = np.array([row[self.cols_tab].values for _, row in batch.iterrows()], dtype=np.float32)
                x_img = np.array([self.load_img(row['Fecha']) for _, row in batch.iterrows()], dtype=np.float32)
                y = batch['Target_NDVI'].values.astype(np.float32)
                return {"Tabular_Input": x_tab, "Image_Input": x_img}, y

        train_df, val_df = train_test_split(self.df_master, test_size=0.2, shuffle=False)
        gen_train = DataGen(train_df, self.img_dir, self.cols_tab)
        gen_val = DataGen(val_df, self.img_dir, self.cols_tab)

        it = Input(shape=(len(self.cols_tab),), name="Tabular_Input")
        ii = Input(shape=(64, 64, 3), name="Image_Input")
        xt = layers.Dense(16, activation='relu')(it)
        xi = layers.Conv2D(32, (3,3), activation='relu')(ii)
        xi = layers.MaxPooling2D()(xi)
        xi = layers.Conv2D(64, (3,3), activation='relu', name="final_conv")(xi)
        xi = layers.GlobalAveragePooling2D()(xi)
        merged = layers.Concatenate()([xt, xi])
        out = layers.Dense(1)(merged)
        
        self.model_dl = models.Model(inputs=[it, ii], outputs=out)
        self.model_dl.compile(optimizer='adam', loss='mse')
        self.model_dl.fit(gen_train, validation_data=gen_val, epochs=30, verbose=1)
        self.model_dl.save('modelo_biologico_multimodal.h5')

    # ==========================================
    # 3. GENERACIÓN DATASET LLM DINÁMICO (V4 PRO)
    # ==========================================
    def generate_dynamic_llm_dataset(self):
        logger.info("Fase 3: Generando dataset de instrucciones con dosis dinámicas...")
        dataset_llm = []
        
        for i in range(len(self.df_raw)):
            row = self.df_raw.iloc[i]
            
            # --- CÁLCULO DE DOSIS DINÁMICA (Lógica de Tesis) ---
            # Kc estimado por NDVI (0.4 a 1.2)
            kc = 0.4 + (row['Sat_NDVI'] * 0.8)
            # Dosis necesaria para cubrir 3 días
            dosis_teorica = (row['ET0_Calibrada'] * kc * 3) - row['Lluvia_AEMET']
            dosis_final = max(0, round(dosis_teorica))
            
            # --- NARRATIVA VARIABLE ---
            status_vigor = "estable" if row['Sat_NDVI'] > 0.5 else "bajo"
            tendencia = "positiva" if row['Target_NDVI'] > row['Sat_NDVI'] else "de estrés hídrico"
            
            input_text = (
                f"Satelite: NDVI {row['Sat_NDVI']:.2f}. "
                f"DL Predictor: Se prevé una tendencia {tendencia} ({row['Target_NDVI']:.2f} t+5). "
                f"Meteorología: ET0 de {row['ET0_Calibrada']:.1f}mm y lluvia de {row['Lluvia_AEMET']:.1f}mm."
            )
            
            output_text = (
                f"ANÁLISIS: Con un vigor {status_vigor} y una demanda acumulada de {dosis_teorica:.1f}mm, "
                f"el sistema requiere compensación. PRESCRIPCIÓN: Riego de {dosis_final}mm."
            )
            
            dataset_llm.append({
                "instruction": "Calcula la prescripción de riego basándote en la tendencia biológica y el balance hídrico.",
                "input": input_text,
                "output": output_text
            })
        
        with open("dataset_v4_dinamico.jsonl", "w") as f:
            for entry in dataset_llm:
                f.write(json.dumps(entry) + "\n")
        logger.info(f"Dataset V4 generado con {len(dataset_llm)} dosis variables.")

    # ==========================================
    # 4. FINE-TUNING LLAMA 3.1 8B
    # ==========================================
    def train_llm_8b(self):
        logger.info("Fase 4: Fine-Tuning Llama-3.1-8B en el Clúster...")
        model_id = "meta-llama/Llama-3.1-8B-Instruct"
        
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16, # Ideal para A100/H100
            bnb_4bit_use_double_quant=True,
        )
        
        self.tokenizer = AutoTokenizer.from_pretrained(model_id)
        self.tokenizer.pad_token = self.tokenizer.eos_token
        
        model = AutoModelForCausalLM.from_pretrained(
            model_id, quantization_config=bnb_config, device_map="auto"
        )
        model = prepare_model_for_kbit_training(model)
        
        config = LoraConfig(
            r=16, lora_alpha=32, 
            target_modules=["q_proj", "v_proj", "k_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
            task_type="CAUSAL_LM"
        )
        self.model_llm = get_peft_model(model, config)
        
        ds = Dataset.from_json("dataset_v4_dinamico.jsonl")
        def tokenize(batch):
            texts = [f"<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\n{i}<|eot_id|>"
                     f"<|start_header_id|>user<|end_header_id|>\n\n{inp}<|eot_id|>"
                     f"<|start_header_id|>assistant<|end_header_id|>\n\n{out}<|eot_id|>" 
                     for i, inp, out in zip(batch['instruction'], batch['input'], batch['output'])]
            return self.tokenizer(texts, truncation=True, padding="max_length", max_length=512)
        
        ds = ds.map(tokenize, batched=True)
        
        args = TrainingArguments(
            output_dir="./agro_llama_8b_results",
            per_device_train_batch_size=2, # Ajustar según VRAM
            gradient_accumulation_steps=4,
            max_steps=150,
            learning_rate=1e-4,
            bf16=True,
            logging_steps=5,
            optim="paged_adamw_32bit",
            save_total_limit=1
        )
        
        trainer = Trainer(
            model=self.model_llm, train_dataset=ds, args=args,
            data_collator=DataCollatorForLanguageModeling(self.tokenizer, mlm=False)
        )
        trainer.train()
        self.model_llm.save_pretrained("./agro_llama_8b_final")

if __name__ == "__main__":
    pipeline = AgroPipeline8B()
    pipeline.process_data()
    pipeline.train_perception_model()
    pipeline.generate_dynamic_llm_dataset()
    pipeline.train_llm_8b()
    logger.info("Pipeline Clúster Finalizado.")