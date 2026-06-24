import os
import re
import argparse
import logging
import json
import glob
from datetime import datetime

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import cv2
import rasterio

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

from fpdf import FPDF
from fpdf.enums import XPos, YPos

# Force offline mode for Hugging Face to avoid network issues
os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.environ["HF_DATASETS_OFFLINE"] = "1"
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
logger.info(f"Using device: {device}")

# ==========================================
# 1. DATA MANAGER (INGESTION & CLEANING)
# ==========================================
class AgroDataManager:
    def __init__(self, base_path='datos/canduela', sector=3, days=3):
        self.base_path = base_path
        self.sector = sector
        self.days = days
        if 'carralancha' in base_path:
            self.area_s3 = 22357.87  # m2 for Sector V4
            self.archivo_aemet = 'AEMET_Rueda_historico.csv'
            self.archivos_sensores = ['CLIMA_CARRALANCHA_sensor_unificado.csv']
            self.archivos_sat = {
                'Sat_NDVI': 'satelite/Sentinel2_NDVI_Parcela_Real.csv',
                'Sat_Radar': 'satelite/Sentinel1_Radar_Parcela_Real.csv',
                'Sat_LST': 'satelite/MODIS_Temp_Parcela_Real.csv'
            }
        else:
            if sector == 2:
                self.area_s3 = 16698.54  # m2 for Sector 2
                self.archivo_riegos = 'CANDUELA - 080221084930-077_ (1).xlsx'
            else:
                self.area_s3 = 40650  # m2 for Sector S3
                self.archivo_riegos = 'RIEGOS_1.csv'
            self.archivos_sensores = ['DATOSE_1.csv', 'DATOSE_2.csv', 'DATOSE_3.csv']
            self.archivo_aemet = 'AEMET_Sardon_historico.csv'
            self.archivos_sat = {
                'Sat_NDVI': 'satelite/Sentinel2_NDVI_Parcela_Real.csv',
                'Sat_Radar': 'satelite/Sentinel1_Radar_Parcela_Real.csv',
                'Sat_LST': 'satelite/MODIS_Temp_Parcela_Real.csv'
            }

    def process_all(self):
        if 'carralancha' in self.base_path:
            logger.info("Fase 1 (Carralancha): Cargando y consolidando datos...")
            # Clima
            clima_path = os.path.join(self.base_path, "CLIMA_CARRALANCHA_sensor_unificado.csv")
            df_sensor = pd.read_csv(clima_path)
            df_sensor['Fecha'] = pd.to_datetime(df_sensor['Fecha'])
            df_clima = df_sensor.set_index('Fecha').resample('D').agg({
                'Temperatura': 'mean',
                'Radiación solar': 'mean',
                'Precipitaciones': 'sum'
            }).reset_index()
            df_clima['ET0_Calculada'] = 0.0135 * (df_clima['Temperatura'] + 17.8) * (df_clima['Radiación solar'] * 0.035)

            # Crop / Soil / Irrigation
            crop_path = os.path.join(self.base_path, "CARRALANCHA_ALMENDRO_V4_unificado.csv")
            df_crop_raw = pd.read_csv(crop_path)
            df_crop_raw['Fecha'] = pd.to_datetime(df_crop_raw['Fecha'])
            df_crop_daily = df_crop_raw.set_index('Fecha').resample('D').agg({
                'Contenido volumétrico (10 cm)': 'mean',
                'Contenido volumétrico (20 cm)': 'mean',
                'Contenido volumétrico (40 cm)': 'mean',
                'Contenido volumétrico (60 cm)': 'mean',
                'Potencial hídrico (almendra)': 'min',
                'Baseline potencial hídrico (almendra)': 'min',
                'Riego (cantidad total)': 'sum'
            }).reset_index()
            df_crop_daily = df_crop_daily.rename(columns={
                'Contenido volumétrico (10 cm)': 'FP_Humedad_10cm',
                'Contenido volumétrico (20 cm)': 'FP_Humedad_20cm',
                'Contenido volumétrico (40 cm)': 'FP_Humedad_40cm',
                'Contenido volumétrico (60 cm)': 'FP_Humedad_60cm',
                'Potencial hídrico (almendra)': 'FP_Potencial_Min',
                'Baseline potencial hídrico (almendra)': 'FP_Potencial_Base_Min',
                'Riego (cantidad total)': 'Riego_m3_Real'
            })
            df_crop_daily['Riego_mm_Real'] = (df_crop_daily['Riego_m3_Real'] * 1000) / self.area_s3
            if df_crop_daily['FP_Potencial_Min'].isna().all():
                df_crop_daily['FP_Potencial_Min'] = -1.5
            if df_crop_daily['FP_Potencial_Base_Min'].isna().all():
                df_crop_daily['FP_Potencial_Base_Min'] = -0.8

            # AEMET
            path_aemet = os.path.join(self.base_path, self.archivo_aemet)
            df_aemet = pd.read_csv(path_aemet, parse_dates=['Fecha']).sort_values('Fecha')
            df_aemet['Lluvia_Prevista_3d'] = df_aemet['Lluvia_AEMET'].shift(-1).rolling(3, min_periods=1).sum().shift(-2).fillna(0)

            # Merge
            df_master = pd.merge(df_clima, df_aemet[['Fecha', 'Lluvia_Prevista_3d', 'Lluvia_AEMET']], on='Fecha', how='outer')
            df_master = pd.merge(df_master, df_crop_daily, on='Fecha', how='outer')
            df_master = df_master.sort_values('Fecha').reset_index(drop=True)

            # Ingest Sat
            for col, rel_path in self.archivos_sat.items():
                path_sat = os.path.join(self.base_path, rel_path)
                if os.path.exists(path_sat):
                    df_sat = pd.read_csv(path_sat, parse_dates=['Fecha']).rename(columns={'Valor': col})
                    df_master = pd.merge(df_master, df_sat[['Fecha', col]], on='Fecha', how='left')
                    df_master[col] = pd.to_numeric(df_master[col], errors='coerce')
                else:
                    df_master[col] = np.nan

            df_master = df_master[(df_master['Fecha'] >= '2022-01-01') & (df_master['Fecha'] <= '2025-12-31')].reset_index(drop=True)
            df_master['Riego_mm_Real'] = df_master['Riego_mm_Real'].fillna(0)
            df_master['Lluvia_AEMET'] = df_master['Lluvia_AEMET'].fillna(0)
            df_master['Lluvia_Prevista_3d'] = df_master['Lluvia_Prevista_3d'].fillna(0)
            df_master['Precipitaciones'] = df_master['Precipitaciones'].fillna(0)

            cols_to_interpolate = ['Temperatura', 'Radiación solar', 'ET0_Calculada',
                                   'FP_Humedad_10cm', 'FP_Humedad_20cm', 'FP_Humedad_40cm', 'FP_Humedad_60cm',
                                   'FP_Potencial_Min', 'FP_Potencial_Base_Min', 'Sat_NDVI', 'Sat_Radar', 'Sat_LST']
            for col in cols_to_interpolate:
                df_master[col] = df_master[col].interpolate(method='linear', limit_direction='both')
                df_master[col] = df_master[col].bfill().ffill()

            df_master['Kc_Dinamico'] = 0.4 + (df_master['Sat_NDVI'] * 0.8)
            df_master['ETc_Sector3'] = df_master['ET0_Calculada'] * df_master['Kc_Dinamico']

            final_path = os.path.join(self.base_path, 'Dataset_Final_V4.csv')
            df_master.to_csv(final_path, index=False)
            logger.info(f"Dataset consolidado final Carralancha V4 guardado en {final_path}. Filas: {len(df_master)}")
            return df_master

        logger.info("Fase 1: Cargando y consolidando datos (Cesens Excels, AEMET, Riegos, Satélites, FloraPulse)...")

        # 1. Sensors Data (Cesens Excel files)
        excel_files = glob.glob(os.path.join(self.base_path, "Datos estaciones Cesens *.xlsx"))
        if not excel_files:
            raise FileNotFoundError("No se encontraron archivos de Cesens (Datos estaciones Cesens *.xlsx).")
        
        list_s = []
        for f in sorted(excel_files):
            try:
                xl = pd.ExcelFile(f)
                sheet = xl.sheet_names[0]
                df_s = xl.parse(sheet)
                df_s['Fecha'] = pd.to_datetime(df_s['Fecha'], errors='coerce')
                df_s = df_s.dropna(subset=['Fecha'])
                # Select only relevant columns to save memory
                df_s = df_s[['Fecha', 'Temperatura', 'Radiación solar', 'Precipitaciones']]
                list_s.append(df_s)
            except Exception as e:
                logger.error(f"Error cargando Cesens excel {f}: {e}")
        
        if not list_s:
            raise FileNotFoundError("No se pudieron cargar datos de los excels de Cesens.")
            
        df_sensor = pd.concat(list_s, ignore_index=True)
        df_clima = df_sensor.set_index('Fecha').resample('D').agg({
            'Temperatura': 'mean',
            'Radiación solar': 'mean',
            'Precipitaciones': 'sum'
        }).reset_index()

        # Reference Evapotranspiration (ET0) using Hargreaves formula
        df_clima['ET0_Calculada'] = 0.0135 * (df_clima['Temperatura'] + 17.8) * (df_clima['Radiación solar'] * 0.035)

        # 2. AEMET weather forecast
        path_aemet = os.path.join(self.base_path, self.archivo_aemet)
        if os.path.exists(path_aemet):
            df_aemet = pd.read_csv(path_aemet, parse_dates=['Fecha'])
            df_aemet = df_aemet.sort_values('Fecha')
            # 3-day and 5-day accumulated rolling rain forecast
            df_aemet['Lluvia_Prevista_3d'] = df_aemet['Lluvia_AEMET'].shift(-1).rolling(3, min_periods=1).sum().shift(-2).fillna(0)
            df_aemet['Lluvia_Prevista_5d'] = df_aemet['Lluvia_AEMET'].shift(-1).rolling(5, min_periods=1).sum().shift(-4).fillna(0)
        else:
            raise FileNotFoundError(f"No se encontró archivo de AEMET: {path_aemet}")

        # 3. Irrigation logs
        if self.sector == 2:
            excel_path = os.path.join(self.base_path, self.archivo_riegos)
            if os.path.exists(excel_path):
                df_xl = pd.read_excel(excel_path)
                df_xl['event_id'] = df_xl['Equipo'].notna().cumsum()
                comienzo_col = df_xl.columns[4]
                consumo_real_col = df_xl.columns[10]
                valvulas_col = df_xl.columns[27]

                header_cols = [df_xl.columns[i] for i in [0, 2, 4, 5, 6, 9, 10]]
                for col in header_cols:
                    df_xl[col] = df_xl[col].ffill()

                df_v2 = df_xl[df_xl[valvulas_col] == 'V2'].copy()
                df_v2['Fecha'] = pd.to_datetime(df_v2[comienzo_col], dayfirst=True, errors='coerce').dt.normalize()
                df_v2_events = df_v2.drop_duplicates(subset=['event_id']).copy()
                df_s2_grouped = df_v2_events.groupby('Fecha')[consumo_real_col].sum().reset_index()
                df_s2_grouped = df_s2_grouped.rename(columns={consumo_real_col: 'Riego_m3_Real'})
                df_s2_grouped['Riego_mm_Real'] = (df_s2_grouped['Riego_m3_Real'] * 1000) / self.area_s3
                df_riego = df_s2_grouped[['Fecha', 'Riego_mm_Real']]
            else:
                logger.warning(f"Archivo Excel de riegos no encontrado en {excel_path}. Asumiendo riego real 0.")
                df_riego = pd.DataFrame(columns=['Fecha', 'Riego_mm_Real'])
        else:
            path_riegos = os.path.join(self.base_path, self.archivo_riegos)
            if os.path.exists(path_riegos):
                df_r = pd.read_csv(path_riegos)
                df_r['Sectores de Riego'] = pd.to_numeric(df_r['Sectores de Riego'], errors='coerce')
                df_r['Agua m3'] = df_r['Agua m³'].astype(str).str.replace('.', '', regex=False).str.replace(',', '.', regex=False)
                df_r['Agua m3'] = pd.to_numeric(df_r['Agua m3'], errors='coerce')
                df_r['Fecha'] = pd.to_datetime(df_r['Inicio'].astype(str).str[:10], dayfirst=True, errors='coerce')
                # Sector 3 specific irrigation
                df_s3 = df_r[df_r['Sectores de Riego'] == 3].groupby('Fecha')['Agua m3'].sum().reset_index()
                df_s3['Riego_mm_Real'] = (df_s3['Agua m3'] * 1000) / self.area_s3
                df_riego = df_s3[['Fecha', 'Riego_mm_Real']]
            else:
                logger.warning(f"Archivo de riegos no encontrado en {path_riegos}. Asumiendo riego real 0.")
                df_riego = pd.DataFrame(columns=['Fecha', 'Riego_mm_Real'])

        # 4. Merge base climate, AEMET, Riegos
        df_master = pd.merge(df_clima, df_aemet[['Fecha', 'Lluvia_Prevista_3d', 'Lluvia_Prevista_5d', 'Lluvia_AEMET']], on='Fecha', how='outer')
        df_master = pd.merge(df_master, df_riego[['Fecha', 'Riego_mm_Real']], on='Fecha', how='left').fillna({'Riego_mm_Real': 0})
        df_master = df_master.sort_values('Fecha').reset_index(drop=True)

        # 5. Satellite Data Ingest
        for col, rel_path in self.archivos_sat.items():
            path_sat = os.path.join(self.base_path, rel_path)
            if os.path.exists(path_sat):
                df_sat = pd.read_csv(path_sat, parse_dates=['Fecha']).rename(columns={'Valor': col})
                df_sat = df_sat.groupby('Fecha')[col].mean().reset_index()
                df_master = pd.merge(df_master, df_sat[['Fecha', col]], on='Fecha', how='left')
                df_master[col] = pd.to_numeric(df_master[col], errors='coerce')
                # Interpolate satellite data linearly
                df_master[col] = df_master[col].interpolate(method='linear', limit_direction='both')
            else:
                logger.warning(f"Satélite {col} no encontrado en {path_sat}.")
                df_master[col] = np.nan

        # Kc Dinámico and Crop Evapotranspiration (ETc)
        df_master['Kc_Dinamico'] = 0.4 + (df_master['Sat_NDVI'] * 0.8)
        etc_col = 'ETc_Sector2' if (not 'carralancha' in self.base_path and self.sector == 2) else 'ETc_Sector3'
        df_master[etc_col] = df_master['ET0_Calculada'] * df_master['Kc_Dinamico']

        # 6. FloraPulse Data Ingestion
        excel_files = glob.glob(os.path.join(self.base_path, 'florapulse', "*.xlsx"))
        dfs_fp = []
        for f in excel_files:
            try:
                xl = pd.ExcelFile(f)
                df_f = xl.parse(xl.sheet_names[0])
                fecha_col = [c for c in df_f.columns if 'fecha' in str(c).lower()][0]
                df_f = df_f.rename(columns={fecha_col: 'Fecha'})
                df_f['Fecha'] = pd.to_datetime(df_f['Fecha'], errors='coerce')
                df_f = df_f.dropna(subset=['Fecha'])

                # Convert numeric columns
                cols_to_convert = [c for c in df_f.columns if c != 'Fecha']
                for c in cols_to_convert:
                    df_f[c] = pd.to_numeric(df_f[c], errors='coerce')
                dfs_fp.append(df_f)
            except Exception as e:
                logger.error(f"Error procesando florapulse excel {f}: {e}")

        if dfs_fp:
            df_fp_all = pd.concat(dfs_fp, ignore_index=True).sort_values('Fecha')

            # Identify target columns using robust keyword matching
            col_pot = [c for c in df_fp_all.columns if 'potencial' in c.lower() and 'almendra' in c.lower() and 'baseline' not in c.lower()][0]
            col_base = [c for c in df_fp_all.columns if 'baseline' in c.lower() and 'almendra' in c.lower()][0]
            col_sm10 = [c for c in df_fp_all.columns if 'volum' in c.lower() and '10 cm' in c.lower() and 'sustrato' not in c.lower()][0]
            col_sm20 = [c for c in df_fp_all.columns if 'volum' in c.lower() and '20 cm' in c.lower() and 'sustrato' not in c.lower()][0]
            col_sm40 = [c for c in df_fp_all.columns if 'volum' in c.lower() and '40 cm' in c.lower() and 'sustrato' not in c.lower()][0]
            col_sm60 = [c for c in df_fp_all.columns if 'volum' in c.lower() and '60 cm' in c.lower() and 'sustrato' not in c.lower()][0]

            agg_dict = {
                col_pot: 'min',         # Minimum daily potential = maximum daily water stress
                col_base: 'min',
                col_sm10: 'mean',
                col_sm20: 'mean',
                col_sm40: 'mean',
                col_sm60: 'mean'
            }

            df_fp_daily = df_fp_all.set_index('Fecha').resample('D').agg(agg_dict).reset_index()

            rename_dict = {
                col_pot: 'FP_Potencial_Min',
                col_base: 'FP_Potencial_Base_Min',
                col_sm10: 'FP_Humedad_10cm',
                col_sm20: 'FP_Humedad_20cm',
                col_sm40: 'FP_Humedad_40cm',
                col_sm60: 'FP_Humedad_60cm'
            }
            df_fp_daily = df_fp_daily.rename(columns=rename_dict)

            # Merge with master
            df_master = pd.merge(df_master, df_fp_daily, on='Fecha', how='left')

            # Interpolate FloraPulse columns
            fp_cols = ['FP_Potencial_Min', 'FP_Potencial_Base_Min', 'FP_Humedad_10cm', 'FP_Humedad_20cm', 'FP_Humedad_40cm', 'FP_Humedad_60cm']
            for col in fp_cols:
                df_master[col] = df_master[col].interpolate(method='linear', limit_direction='both')
        else:
            logger.warning("No se pudieron cargar datos de FloraPulse. Creando columnas vacías.")
            for col in ['FP_Potencial_Min', 'FP_Potencial_Base_Min', 'FP_Humedad_10cm', 'FP_Humedad_20cm', 'FP_Humedad_40cm', 'FP_Humedad_60cm']:
                df_master[col] = 0.5 if 'Potencial' in col else 20.0

        # Final Cleaning & Interpolation of basic climate data gaps
        df_master['Temperatura'] = df_master['Temperatura'].interpolate(method='linear', limit_direction='both')
        df_master['Radiación solar'] = df_master['Radiación solar'].interpolate(method='linear', limit_direction='both')
        df_master['Precipitaciones'] = df_master['Precipitaciones'].fillna(0)
        df_master['Lluvia_AEMET'] = df_master['Lluvia_AEMET'].fillna(0)
        df_master['Lluvia_Prevista_3d'] = df_master['Lluvia_Prevista_3d'].fillna(0)
        df_master['Lluvia_Prevista_5d'] = df_master['Lluvia_Prevista_5d'].fillna(0)
        df_master['ET0_Calculada'] = df_master['ET0_Calculada'].interpolate(method='linear', limit_direction='both')
        df_master[etc_col] = df_master[etc_col].interpolate(method='linear', limit_direction='both')

        # Crop NDVI, LST, and Radar
        df_master['Sat_NDVI'] = df_master['Sat_NDVI'].interpolate(method='linear', limit_direction='both')
        df_master['Sat_Radar'] = df_master['Sat_Radar'].interpolate(method='linear', limit_direction='both')
        df_master['Sat_LST'] = df_master['Sat_LST'].interpolate(method='linear', limit_direction='both')

        # Drop any remaining NaNs in columns essential for Two-Tower training
        df_master = df_master.dropna(subset=['Sat_NDVI', 'Sat_Radar', 'Sat_LST', 'FP_Potencial_Min']).reset_index(drop=True)

        if 'carralancha' in self.base_path:
            final_filename = 'Dataset_Final_V4.csv'
        else:
            final_filename = f'Dataset_Final_S{self.sector}.csv'
        final_path = os.path.join(self.base_path, final_filename)
        df_master.to_csv(final_path, index=False)
        logger.info(f"Dataset consolidado final guardado en {final_path}. Filas: {len(df_master)}")
        return df_master


# ==========================================
# 2. PYTORCH DEEP LEARNING (TWO-TOWER MODEL)
# ==========================================
class AgroTwoTower(nn.Module):
    def __init__(self, tab_size):
        super().__init__()
        # Image (Sentinel-2 RGB 64x64) Tower
        self.cnn = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2), # 32x32
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d(1) # 64 features
        )
        # Tabular Tower
        self.mlp = nn.Sequential(
            nn.Linear(tab_size, 32),
            nn.ReLU(),
            nn.Linear(32, 16),
            nn.ReLU()
        )
        # Fusion Layer
        self.fusion = nn.Linear(64 + 16, 1)

    def forward(self, img, tab):
        v_img = self.cnn(img).view(img.size(0), -1)
        v_tab = self.mlp(tab)
        x = torch.cat((v_img, v_tab), dim=1)
        return self.fusion(x)


class AgroDatasetPT(Dataset):
    def __init__(self, df, img_dir, cols_tab):
        self.df = df.reset_index(drop=True)
        self.img_dir = img_dir
        self.cols_tab = cols_tab
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
                except:
                    continue
        self.img_dates = pd.to_datetime(parsed)
        self.valid_files = valid_files

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        tab = torch.tensor(row[self.cols_tab].values.astype(np.float32), dtype=torch.float32)
        
        # Link temporal: find the closest satellite image in date
        dist = np.abs(self.img_dates - row['Fecha'])
        closest_img_name = self.valid_files[np.argmin(dist)]
        
        try:
            with rasterio.open(os.path.join(self.img_dir, closest_img_name)) as src:
                # Load Red, Green, Blue bands (3, 2, 1)
                img = src.read([3, 2, 1]).astype(np.float32)
            img = np.moveaxis(img, 0, -1)
            p2, p98 = np.nanpercentile(img, (2, 98))
            img = np.clip((img - p2) / (p98 - p2 + 1e-10), 0, 1)
            img = np.nan_to_num(img, nan=0.0)
            img = cv2.resize(img, (64, 64))
            img_t = torch.from_numpy(img).permute(2, 0, 1).float()
        except Exception as e:
            img_t = torch.zeros((3, 64, 64), dtype=torch.float32)
            
        target = torch.tensor([row['Target_NDVI']], dtype=torch.float32)
        return img_t, tab, target


class AgroModelTrainer:
    def __init__(self, img_dir, cols_tab, model_save_path='models/modelo_percepcion_s3.pt'):
        self.img_dir = img_dir
        self.cols_tab = cols_tab
        self.model_save_path = model_save_path

    def train_model(self, df):
        logger.info("Fase 2: Entrenando modelo de percepción biológica (Two-Tower PyTorch)...")
        # Target: NDVI in 5 days
        df['Target_NDVI'] = df['Sat_NDVI'].shift(-5)
        # Drop rows where target is NaN (the last 5 days)
        df_train_clean = df.dropna(subset=['Target_NDVI'] + self.cols_tab).copy()

        # Scale features
        scaler = StandardScaler()
        df_scaled = df_train_clean.copy()
        df_scaled[self.cols_tab] = scaler.fit_transform(df_train_clean[self.cols_tab])

        # Chronological split (no shuffle to prevent data leakage in time series)
        train_df, val_df = train_test_split(df_scaled, test_size=0.2, shuffle=False)

        train_dataset = AgroDatasetPT(train_df, self.img_dir, self.cols_tab)
        val_dataset = AgroDatasetPT(val_df, self.img_dir, self.cols_tab)
        
        train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False)

        model = AgroTwoTower(len(self.cols_tab)).to(device)
        optimizer = optim.Adam(model.parameters(), lr=1e-3)
        criterion = nn.MSELoss()

        best_val_loss = float('inf')
        
        for epoch in range(40):
            model.train()
            train_loss = 0
            for imgs, tabs, targets in train_loader:
                imgs, tabs, targets = imgs.to(device), tabs.to(device), targets.to(device)
                optimizer.zero_grad()
                outputs = model(imgs, tabs)
                loss = criterion(outputs, targets)
                loss.backward()
                optimizer.step()
                train_loss += loss.item() * imgs.size(0)
            
            # Validation
            model.eval()
            val_loss = 0
            with torch.no_grad():
                for imgs, tabs, targets in val_loader:
                    imgs, tabs, targets = imgs.to(device), tabs.to(device), targets.to(device)
                    outputs = model(imgs, tabs)
                    loss = criterion(outputs, targets)
                    val_loss += loss.item() * imgs.size(0)

            train_loss /= len(train_loader.dataset)
            val_loss /= len(val_loader.dataset)

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                torch.save({
                    'model_state_dict': model.state_dict(),
                    'scaler': scaler,
                    'cols_tab': self.cols_tab
                }, self.model_save_path)

            if (epoch + 1) % 10 == 0 or epoch == 0:
                logger.info(f"  Epoch {epoch+1:02d}/40 | Loss Train: {train_loss:.6f} | Loss Val: {val_loss:.6f}")

        logger.info(f"Modelo de deep learning guardado con éxito en '{self.model_save_path}' (Val Loss: {best_val_loss:.6f})")
        return best_val_loss


# ==========================================
# 3. LLM FINE-TUNING (QLORA COT PRESCRIPTOR)
# ==========================================
class AgroLLMOptimizer:
    def __init__(self, base_model_id="meta-llama/Llama-3.2-1B-Instruct", save_adapter_path="models/agro_llama_1b_s3_final", base_path="datos/canduela", sector=3, days=3):
        self.base_model_id = base_model_id
        self.save_adapter_path = save_adapter_path
        self.base_path = base_path
        self.is_carralancha = 'carralancha' in base_path
        self.sector = sector
        self.days = days
        if self.is_carralancha:
            self.sector_name = "Sector V4"
            self.dataset_cot_name = 'dataset_cot_v4.jsonl'
        else:
            self.sector_name = f"Sector {self.sector}"
            self.dataset_cot_name = f'dataset_cot_s{self.sector}.jsonl'

    @staticmethod
    def _classify_stress(pot):
        # Stress rules based on Midday Stem Water Potential (bars)
        if pot >= -8.0:
            return "Nulo (Suelo hidratado y condiciones óptimas)"
        elif -12.0 <= pot < -8.0:
            return "Leve (Defensa del cultivo activa, régimen deficitario óptimo)"
        elif -16.0 <= pot < -12.0:
            return "Moderado (Estrés hídrico detectado, requiere corrección)"
        else:
            return "Severo (Riesgo grave de defoliación y aborto de frutos)"

    def generate_cot_dataset(self, df):
        logger.info("Generando dataset para Fine-Tuning de orquestación CoT con FloraPulse...")
        dataset_llm = []
        
        etc_col = 'ETc_Sector2' if (not self.is_carralancha and self.sector == 2) else 'ETc_Sector3'
        lluvia_p_col = f'Lluvia_Prevista_{self.days}d'
        for _, row in df.iterrows():
            kc = row['Kc_Dinamico']
            etc = row[etc_col]
            lluvia_p = row[lluvia_p_col]
            riego_h = row['Riego_mm_Real']
            pot = row['FP_Potencial_Min']
            sm10 = row['FP_Humedad_10cm']
            sm40 = row['FP_Humedad_40cm']
            stress = self._classify_stress(pot)
            
            # Agronomic regulation logic for Dose Calculation
            dosis_ideal_base = etc * self.days - lluvia_p  # Next days target
            if pot >= -8.0:
                deficit_coef = 0.70  # Standard regulated deficit
                dosis_rec = max(0, dosis_ideal_base * deficit_coef)
                prescripcion = "Suelo húmedo y potencial óptimo. Se prescribe riego de mantenimiento para mantener el nivel actual de vigor."
            elif -12.0 <= pot < -8.0:
                deficit_coef = 0.60  # Regulated Deficit Irrigation (RDI) optimization
                dosis_rec = max(0, dosis_ideal_base * deficit_coef)
                prescripcion = "Estrés leve favorable para el cultivo en esta etapa. Mantener estrategia de RDI con dosis controlada."
            elif -16.0 <= pot < -12.0:
                deficit_coef = 0.85  # Recovery irrigation
                dosis_rec = max(0, dosis_ideal_base * deficit_coef + 3.0)
                prescripcion = "Se detecta estrés moderado en FloraPulse. Incrementar dosis de riego para recuperar el potencial hídrico foliar."
            else: # < -16.0
                deficit_coef = 1.00  # Extreme recovery
                dosis_rec = max(0, dosis_ideal_base * deficit_coef + 8.0)
                prescripcion = "ALERTA CRÍTICA: Potencial hídrico en zona de peligro. Aplicar riego de choque de forma inmediata para restaurar turgencia celular."

            # Calculate variance
            varianza = riego_h - (dosis_rec / self.days)

            output_text = (
                f"ANÁLISIS AGRO-XAI: Kc Dinámico {kc:.2f}, Demanda ETc {etc:.1f} mm/día. "
                f"MONITORIZACIÓN BIOFÍSICA (FLORAPULSE): Potencial hídrico foliar de {pot:.1f} bar (Estrés: {stress}). "
                f"Humedad del suelo a 10cm: {sm10:.1f}%, a 40cm: {sm40:.1f}%. "
                f"AUDITORÍA: Aplicados {riego_h:.1f} mm en últimas 24h. Con previsión de lluvia de {lluvia_p:.1f} mm en {self.days} días, "
                f"la dosis ideal prescrita para los próximos {self.days} días es {dosis_rec:.1f} mm ({dosis_rec/self.days:.1f} mm/día). "
                f"VARIANZA: {varianza:+.1f} mm. "
                f"PRESCRIPCIÓN: {prescripcion}"
            )
            
            input_text = (
                f"NDVI: {row['Sat_NDVI']:.2f}, ET0: {row['ET0_Calculada']:.1f} mm, {lluvia_p_col}: {lluvia_p:.1f} mm, "
                f"Potencial_Almendra: {pot:.1f} bar, Humedad_10cm: {sm10:.1f}%, Riego_Histórico: {riego_h:.1f} mm"
            )
            
            dataset_llm.append({
                "instruction": f"Audita la eficiencia del riego e indica prescripción agronómica detallada para el {self.sector_name} (Almendro).",
                "input": input_text,
                "output": output_text
            })

        jsonl_path = os.path.join(self.base_path, self.dataset_cot_name)
        with open(jsonl_path, 'w', encoding='utf-8') as f:
            for entry in dataset_llm:
                f.write(json.dumps(entry, ensure_ascii=False) + '\n')
        logger.info(f"Dataset CoT generado con {len(dataset_llm)} muestras en {jsonl_path}")

    def train_llm(self):
        logger.info(f"Fase 3: Cargando {self.base_model_id} e iniciando Fine-Tuning QLoRA (Llama 1B)...")
        from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig, Trainer, TrainingArguments, DataCollatorForLanguageModeling
        from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
        from datasets import Dataset as HFDataset

        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16
        )

        tokenizer = AutoTokenizer.from_pretrained(self.base_model_id, local_files_only=True)
        tokenizer.pad_token = tokenizer.eos_token

        base_model = AutoModelForCausalLM.from_pretrained(
            self.base_model_id,
            quantization_config=bnb_config,
            device_map="auto",
            local_files_only=True
        )
        base_model = prepare_model_for_kbit_training(base_model)

        peft_config = LoraConfig(
            r=8,
            lora_alpha=16,
            target_modules=["q_proj", "v_proj", "k_proj", "o_proj"],
            task_type="CAUSAL_LM",
            lora_dropout=0.05
        )
        model = get_peft_model(base_model, peft_config)

        # Load dataset
        ds = HFDataset.from_json(os.path.join(self.base_path, self.dataset_cot_name))
        
        def tokenize(batch):
            texts = [
                f"<|begin_of_text|>system\nEres un Orquestador Agrónomo de precisión experto en Riego Deficitario Controlado.<|eot_id|>"
                f"user\n{inst}\nContexto: {inp}<|eot_id|>"
                f"assistant\n{out}<|eot_id|>"
                for inst, inp, out in zip(batch['instruction'], batch['input'], batch['output'])
            ]
            return tokenizer(texts, truncation=True, padding="max_length", max_length=256)

        tokenized_ds = ds.map(tokenize, batched=True, remove_columns=ds.column_names)

        args = TrainingArguments(
            output_dir="./agro_llama_1b_results",
            per_device_train_batch_size=1,
            gradient_accumulation_steps=4,
            max_steps=50,
            learning_rate=2e-4,
            bf16=True,
            logging_steps=5,
            optim="paged_adamw_8bit"
        )

        trainer = Trainer(
            model=model,
            train_dataset=tokenized_ds,
            args=args,
            data_collator=DataCollatorForLanguageModeling(tokenizer, mlm=False)
        )

        trainer.train()
        model.save_pretrained(self.save_adapter_path)
        logger.info(f"Fine-Tuning de LLM finalizado. Adaptador guardado en '{self.save_adapter_path}'")


# ==========================================
# 4. REPORT GENERATOR (PDF & VISUALIZATION)
# ==========================================
C_PRIMARY = (27, 94, 32)
C_SECONDARY = (56, 142, 60)
C_ALERT = (216, 67, 21)
C_TEXT = (33, 33, 33)
C_BG_CARD = (245, 247, 245)

class AgroPDF(FPDF):
    def __init__(self, nombre_sector="Sector 3"):
        super().__init__()
        self.nombre_sector = nombre_sector

    def header(self):
        # Forest green background banner
        self.set_fill_color(27, 94, 32)
        self.rect(0, 0, 210, 24, "F")
        self.set_y(6)
        self.set_font("helvetica", "B", 14)
        self.set_text_color(255)
        self.cell(0, 8, "INFORME OPERATIVO DE PRECISION - ORQUESTADOR AGRO-XAI", align="C", new_x=XPos.LMARGIN, new_y=YPos.NEXT)
        self.set_font("helvetica", "I", 9)
        self.cell(0, 4, f"Monitoreo Integrado de Vigor Satelital, Clima y Humedad del Suelo ({self.nombre_sector})", align="C", new_x=XPos.LMARGIN, new_y=YPos.NEXT)
        self.ln(6)

    def footer(self):
        self.set_y(-15)
        self.set_font("helvetica", "I", 8)
        self.set_text_color(120)
        self.cell(0, 10, f"Página {self.page_no()} | Generado por Orquestador Agro-XAI", align="C")

    def section_header(self, title):
        self.set_font("helvetica", "B", 11)
        self.set_fill_color(230, 240, 230)
        self.set_text_color(27, 94, 32)
        y = self.get_y()
        self.rect(15, y, 180, 7, "F")
        self.set_fill_color(27, 94, 32)
        self.rect(15, y, 3, 7, "F")
        self.set_xy(20, y + 1)
        self.cell(0, 5, title, new_x=XPos.LMARGIN, new_y=YPos.NEXT)
        self.ln(2)


class AgroProductionReport:
    def __init__(self, model_dl_path='models/modelo_percepcion_s3.pt', base_model_id='meta-llama/Llama-3.2-1B-Instruct', adapter_path='models/agro_llama_1b_s3_final', base_path='datos/canduela', sector=3, days=3):
        self.base_path = base_path
        self.sector = sector
        self.days = days
        self.is_carralancha = 'carralancha' in base_path
        if self.is_carralancha:
            self.dataset_final_name = 'Dataset_Final_V4.csv'
        else:
            self.dataset_final_name = f'Dataset_Final_S{self.sector}.csv'
        self.df = pd.read_csv(os.path.join(self.base_path, self.dataset_final_name), parse_dates=['Fecha'])
        
        if self.is_carralancha:
            self.img_dir = os.path.join(self.base_path, 'satelite')
            self.area_s3 = 22357.87
            self.nombre_finca = "Carralancha"
            self.nombre_sector = "Sector V4"
            self.nombre_cultivo = "Almendro V4"
            self.precipitation_rate = 1.2
        else:
            self.img_dir = os.path.join(self.base_path, 'GEE_Tesis_ImageChips-20260217T075250Z-1-001/GEE_Tesis_ImageChips')
            if self.sector == 2:
                self.area_s3 = 16698.54
                self.nombre_finca = "Canduela"
                self.nombre_sector = "Sector 2"
                self.nombre_cultivo = "Almendro"
                self.precipitation_rate = 1.09
            else:
                self.area_s3 = 40650
                self.nombre_finca = "Canduela"
                self.nombre_sector = "Sector 3"
                self.nombre_cultivo = "Almendro Guara"
                self.precipitation_rate = 1.2
            
        self.model_dl_path = model_dl_path
        self.base_model_id = base_model_id
        self.adapter_path = adapter_path
        
        # Load DL perception model info
        if os.path.exists(self.model_dl_path):
            checkpoint = torch.load(self.model_dl_path, map_location=device)
            self.cols_tab = checkpoint['cols_tab']
            self.scaler = checkpoint['scaler']
            self.model_dl = AgroTwoTower(len(self.cols_tab)).to(device)
            self.model_dl.load_state_dict(checkpoint['model_state_dict'])
            self.model_dl.eval()
            logger.info("Modelo de deep learning de percepción biológica cargado correctamente.")
        else:
            logger.warning("No se encontró el modelo de deep learning entrenado. Se simularán predicciones.")
            self.model_dl = None
            self.cols_tab = ['Temperatura', 'Lluvia_Prevista_3d', 'Riego_mm_Real', 'ET0_Calculada', 'Sat_Radar', 'Sat_LST', 'FP_Potencial_Min', 'FP_Humedad_10cm']

        # Load LLM
        self.llm = None
        self.tokenizer = None
        if os.path.exists(self.adapter_path):
            try:
                from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
                from peft import PeftModel
                
                logger.info("Cargando tokenizer del LLM...")
                self.tokenizer = AutoTokenizer.from_pretrained(self.base_model_id, local_files_only=True)
                
                logger.info("Cargando base LLM en 4-bit para inferencia...")
                bnb_config = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype=torch.bfloat16)
                base_model = AutoModelForCausalLM.from_pretrained(
                    self.base_model_id,
                    quantization_config=bnb_config,
                    device_map="auto",
                    local_files_only=True
                )
                self.llm = PeftModel.from_pretrained(base_model, self.adapter_path)
                self.llm.eval()
                logger.info("Prescriptor LLM cargado correctamente.")
            except Exception as e:
                logger.error(f"Error al cargar el LLM. Usando motor experto alternativo de reglas: {e}")
                self.llm = None
        else:
            logger.warning("Adaptador LoRA del LLM no encontrado. Se usará el motor experto alternativo de reglas.")

    @staticmethod
    def _crop_status(ndvi):
        if ndvi >= 0.65: return "Alto Vigor (Desarrollo óptimo)"
        if ndvi >= 0.45: return "Vigor Medio (Seguimiento activo)"
        return "Bajo Vigor (Alerta de estrés vegetativo)"
    @staticmethod
    def _classify_stress(pot):
        if pot >= -8.0: return "Nulo (Suelo hidratado)"
        elif -12.0 <= pot < -8.0: return "Leve (RDC óptimo)"
        elif -16.0 <= pot < -12.0: return "Moderado (Déficit activo)"
        else: return "Severo (Riesgo de defoliación)"

    def _prepare_img(self, date):
        if not self.is_carralancha and self.sector == 2:
            try:
                import matplotlib.pyplot as plt
                from rasterio.warp import transform
                
                # Coordenadas del Sector 2
                coords = [
                    [-4.577624452593563, 41.814040723556154],
                    [-4.577847075941799, 41.813829819817485],
                    [-4.578032148363826, 41.813547947385764],
                    [-4.578068697003681, 41.81318036452114],
                    [-4.578187960151398, 41.81304849397068],
                    [-4.5787659761941075, 41.81318443366328],
                    [-4.5789922247252175, 41.8128168929197],
                    [-4.57900161245677, 41.812716936717905],
                    [-4.578906394036741, 41.81257571549979],
                    [-4.579008317979307, 41.81245576758731],
                    [-4.579288608821363, 41.81226179164462],
                    [-4.579359687360258, 41.81213584571337],
                    [-4.579383827241392, 41.81202789186088],
                    [-4.579473681243391, 41.811804986571325],
                    [-4.579570240767927, 41.81157808218388],
                    [-4.579650509117174, 41.81137531452619],
                    [-4.579696106670427, 41.811228375548794],
                    [-4.579764503000307, 41.810963484704025],
                    [-4.579886543510485, 41.81064661383948],
                    [-4.580094414709139, 41.81020878968733],
                    [-4.580137330053377, 41.810058849221306],
                    [-4.580166834352541, 41.80985093119436],
                    [-4.579960304258394, 41.80978995510655],
                    [-4.579854357002306, 41.80994689344449],
                    [-4.579501646516848, 41.80984693276414],
                    [-4.57937424158864, 41.81008084051163],
                    [-4.5798677680473805, 41.81022478331631],
                    [-4.579641121385622, 41.81058863729716],
                    [-4.579350874669361, 41.811062291391984],
                    [-4.578583863489332, 41.81230090431014],
                    [-4.5780997247621436, 41.813084561839695],
                    [-4.577533597319558, 41.813997981766136],
                    [-4.577624452593563, 41.814040723556154]
                ]
                lons = [c[0] for c in coords]
                lats = [c[1] for c in coords]
                
                # Buscar imagen más cercana
                files = [f for f in os.listdir(self.img_dir) if f.endswith(".tif")]
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
                
                if not valid_files:
                    logger.warning("No se encontraron imágenes satelitales en TIF. Usando mapa básico.")
                    return None
                
                dist = np.abs(pd.to_datetime(img_dates) - date)
                closest_img_name = valid_files[np.argmin(dist)]
                
                with rasterio.open(os.path.join(self.img_dir, closest_img_name)) as src:
                    # Leer RGB
                    rgb = src.read([3, 2, 1]).astype(np.float32)
                    rgb = np.moveaxis(rgb, 0, -1)
                    p2, p98 = np.nanpercentile(rgb, (2, 98))
                    rgb = np.clip((rgb - p2) / (p98 - p2 + 1e-10), 0, 1)
                    rgb = np.nan_to_num(rgb)
                    
                    # Leer B4 y B8 para NDVI
                    b4 = src.read(3).astype(np.float32)
                    b8 = src.read(7).astype(np.float32)
                    ndvi = (b8 - b4) / (b8 + b4 + 1e-10)
                    
                    left, bottom, right, top = src.bounds
                    width, height = src.width, src.height
                    
                    # Proyectar y desplazar coordenadas de Sector 2 para alinearlas
                    xs, ys = transform('EPSG:4326', 'EPSG:32630', lons, lats)
                    tif_cx = (left + right) / 2.0
                    tif_cy = (bottom + top) / 2.0
                    s2_cx = (min(xs) + max(xs)) / 2.0
                    s2_cy = (min(ys) + max(ys)) / 2.0
                    
                    dx = tif_cx - s2_cx
                    dy = tif_cy - s2_cy
                    
                    xs_shifted = [x + dx for x in xs]
                    ys_shifted = [y + dy for y in ys]
                    
                    pixel_w = (right - left) / width
                    pixel_h = (top - bottom) / height
                    
                    cols_pixel = [(x - left) / pixel_w for x in xs_shifted]
                    rows_pixel = [(top - y) / pixel_h for y in ys_shifted]
                    
                    # Graficar
                    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8, 4))
                    
                    # Imagen RGB Real
                    ax1.imshow(rgb, extent=[0, width, height, 0])
                    ax1.plot(cols_pixel, rows_pixel, color='cyan', linewidth=2)
                    ax1.fill(cols_pixel, rows_pixel, color='cyan', alpha=0.15)
                    ax1.set_title("Imagen Satelital Real", fontsize=10, fontweight='bold', color='#1B5E20')
                    ax1.axis('off')
                    
                    # Radiografía de Salud NDVI (RdYlBu)
                    im = ax2.imshow(ndvi, cmap='RdYlBu', vmin=0.1, vmax=0.6, extent=[0, width, height, 0])
                    ax2.plot(cols_pixel, rows_pixel, color='black', linewidth=1.5)
                    ax2.fill(cols_pixel, rows_pixel, color='blue', alpha=0.1)
                    ax2.set_title("Radiografía de Salud (NDVI)", fontsize=10, fontweight='bold', color='#1B5E20')
                    ax2.axis('off')
                    
                    cbar = fig.colorbar(im, ax=ax2, fraction=0.046, pad=0.04)
                    cbar.ax.tick_params(labelsize=8)
                    cbar.set_label('Vigor (NDVI)', fontsize=8)
                    
                    plt.tight_layout()
                    out_path = "temp_sat.png"
                    fig.savefig(out_path, dpi=180)
                    plt.close(fig)
                    return out_path
            except Exception as e:
                logger.error(f"Error generando mapa de coordenadas en TIF: {e}")
                return None

        # Look for images matching the year
        files = [f for f in os.listdir(self.img_dir) if f.endswith(".tif")]
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

        if not valid_files:
            if self.is_carralancha:
                try:
                    import matplotlib.pyplot as plt
                    fig, ax = plt.subplots(figsize=(4, 4))
                    coords = [
                        [-4.811755842895575, 41.37409504652195],
                        [-4.812917239399024, 41.37351537008459],
                        [-4.811444593553215, 41.37225123589853],
                        [-4.810167967851344, 41.37273024205082],
                        [-4.811755842895575, 41.37409504652195]
                    ]
                    lons = [c[0] for c in coords]
                    lats = [c[1] for c in coords]
                    ax.fill(lons, lats, color='#2ecc71', alpha=0.4, edgecolor='#27ae60', linewidth=2)
                    ax.plot(lons, lats, 'o', color='#27ae60', markersize=4)
                    ax.set_title("Ubicacion Sector V4 - Pozaldez", fontsize=9, fontweight='bold', color='#1B5E20')
                    ax.axis('off')
                    fig.tight_layout()
                    out_path = "temp_sat.png"
                    fig.savefig(out_path, dpi=180)
                    plt.close(fig)
                    return out_path
                except Exception as e:
                    logger.error(f"Error generando mapa de coordenadas: {e}")
                    return None
            return None

        # Find closest date
        dist = np.abs(pd.to_datetime(img_dates) - date)
        closest_img_name = valid_files[np.argmin(dist)]
        
        try:
            with rasterio.open(os.path.join(self.img_dir, closest_img_name)) as src:
                img = src.read([3, 2, 1])  # Red, Green, Blue
            img = np.moveaxis(img, 0, -1).astype(np.float32)
            p2, p98 = np.nanpercentile(img, (2, 98))
            img = np.clip((img - p2) / (p98 - p2 + 1e-10), 0, 1)
            img = np.nan_to_num(img, nan=0.0)
            img_8bit = (img * 255).astype(np.uint8)

            # Center square crop to avoid distortion
            h, w = img_8bit.shape[:2]
            side = min(h, w)
            y0 = (h - side) // 2
            x0 = (w - side) // 2
            img_sq = img_8bit[y0:y0+side, x0:x0+side]
            img_sq = cv2.resize(img_sq, (600, 600), interpolation=cv2.INTER_CUBIC)

            # Add scale bar and label
            cv2.putText(img_sq, "Sentinel-2 True Color (1:1)", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            cv2.line(img_sq, (400, 560), (550, 560), (255, 255, 255), 4)
            cv2.putText(img_sq, "150 m", (430, 550), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

            out_path = "temp_sat.png"
            cv2.imwrite(out_path, cv2.cvtColor(img_sq, cv2.COLOR_RGB2BGR))
            return out_path
        except Exception as e:
            logger.error(f"Error procesando imagen de satélite: {e}")
            return None

    def _generate_plots(self, current_row, history_df):
        # 1. Bar Plot: ET0 vs Lluvia
        fig, ax = plt.subplots(figsize=(5, 3.2))
        labels = ['ET0', 'Lluvia\n(24h)', 'Lluvia\n(Prev. 3d)']
        values = [current_row['ET0_Calculada'], current_row['Lluvia_AEMET'], current_row['Lluvia_Prevista_3d']]
        bars = ax.bar(labels, values, color=['#e67e22', '#2ecc71', '#3498db'], width=0.5, edgecolor='black', linewidth=0.5)
        ax.set_title("Balance Hídrico Operativo (mm)", fontsize=10, fontweight='bold', color='#1B5E20', pad=8)
        ax.set_ylabel("mm", fontsize=8)
        ax.grid(axis='y', linestyle='--', alpha=0.3)
        ax.tick_params(axis='x', labelsize=8)
        for bar, val in zip(bars, values):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.05, f"{val:.2f}", ha='center', va='bottom', fontsize=8, fontweight='bold')
        fig.tight_layout()
        plot_balance_path = "temp_plot_balance.png"
        fig.savefig(plot_balance_path, dpi=180)
        plt.close(fig)

        # 2. Line Plot: Dual axis evolution (NDVI vs Soil Moisture)
        fig, ax1 = plt.subplots(figsize=(6.5, 3.0))
        color = '#27ae60'
        ax1.set_xlabel('Últimos 30 días', fontsize=8, fontweight='bold')
        ax1.set_ylabel('Vigor Foliar (NDVI)', color=color, fontsize=8, fontweight='bold')
        line1 = ax1.plot(history_df['Fecha'], history_df['Sat_NDVI'], color=color, linewidth=2, label='Vigor (NDVI)', marker='o', markersize=3)
        ax1.tick_params(axis='y', labelcolor=color, labelsize=7)
        
        # Dual axis
        ax2 = ax1.twinx()
        color = '#2980b9'
        ax2.set_ylabel('Humedad Suelo 40cm (%)', color=color, fontsize=8, fontweight='bold')
        line2 = ax2.plot(history_df['Fecha'], history_df['FP_Humedad_40cm'], color=color, linewidth=2, linestyle='--', label='Humedad (40cm)', marker='s', markersize=3)
        ax2.tick_params(axis='y', labelcolor=color, labelsize=7)
        
        lines = line1 + line2
        labels = [l.get_label() for l in lines]
        ax1.legend(lines, labels, loc='upper left', fontsize=7, framealpha=0.8)
        ax1.set_title("Evolución de Vigor (NDVI) y Humedad del Suelo (40cm)", fontsize=10, fontweight='bold', color='#1B5E20', pad=8)
        ax1.grid(True, alpha=0.15)
        fig.autofmt_xdate(rotation=15, ha='right')
        fig.tight_layout()
        plot_evol_path = "temp_plot_evol.png"
        fig.savefig(plot_evol_path)
        plt.close(fig)

        return plot_balance_path, plot_evol_path

    def _generate_rule_based_presc(self, row, etc_val):
        kc = row['Kc_Dinamico']
        etc_col = 'ETc_Sector2' if (not self.is_carralancha and self.sector == 2) else 'ETc_Sector3'
        etc = etc_val
        lluvia_p_col = f'Lluvia_Prevista_{self.days}d'
        lluvia_p = row[lluvia_p_col]
        sm10 = row['FP_Humedad_10cm']
        sm40 = row['FP_Humedad_40cm']
        
        dosis_ideal_base = etc * self.days - lluvia_p
        
        # Estrategia de Riego Deficitario Controlado (RDC) para Almendro en post-cuajado / crecimiento vegetativo (finales de junio):
        # Aportar el 70% de la ETc para mantener el vigor sin provocar excesivo crecimiento vegetativo ni desperdicio de agua.
        dosis_rec = max(0, dosis_ideal_base * 0.70)
        presc = "Estrategia de Riego Deficitario Controlado (RDC) al 70%. Se prescribe riego de mantenimiento para sostener el vigor y el desarrollo del fruto."
        
        advice = (
            f"- <b><u>ANÁLISIS DE DEMANDA:</u></b> Coeficiente Kc de <b>{kc:.2f}</b> con una demanda hídrica diaria ETc de <b>{etc:.1f} mm/día</b>.<br/>"
            f"- <b><u>PERFIL DE HUMEDAD DE SUELO:</u></b> Humedad radicular del <b>{sm10:.1f}%</b> a 10cm y del <b>{sm40:.1f}%</b> a 40cm de profundidad.<br/>"
            f"- <b><u>PLAN DE RIEGO RECOMENDADO:</u></b> Con una previsión de lluvia de <b>{lluvia_p:.1f} mm</b> en los próximos {self.days} días, la dosis recomendada es de <b>{dosis_rec:.1f} mm</b> (<b>{dosis_rec/self.days:.1f} mm/día</b>), lo que equivale a un volumen total de <b>{(dosis_rec*self.area_s3/1000.0):.1f} m³</b>.<br/>"
            f"- <b><u>ESTRATEGIA APLICADA:</u></b> {presc}"
        )
        return advice, dosis_rec

    def generate_report(self, date_str, output_pdf_name=None):
        logger.info(f"Fase 5: Generando reporte profesional para la fecha {date_str}...")
        date_dt = pd.to_datetime(date_str)
        
        # Filter target day row
        matching_rows = self.df[self.df["Fecha"] == date_dt]
        if len(matching_rows) == 0:
            raise KeyError(f"La fecha {date_str} no existe en el Dataset consolidado.")
        
        row = matching_rows.iloc[0]
        
        # Get history of last 30 days for plot evolution
        history_df = self.df[(self.df["Fecha"] <= date_dt)].tail(30)
        
        # Predict Future NDVI with Two-Tower model
        if self.model_dl is not None:
            try:
                # Find closest satellite image
                files = [f for f in os.listdir(self.img_dir) if f.endswith(".tif")]
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
                closest_idx = np.argmin(np.abs(pd.to_datetime(img_dates) - date_dt))
                closest_img_name = valid_files[closest_idx]
                
                with rasterio.open(os.path.join(self.img_dir, closest_img_name)) as src:
                    img_data = src.read([3, 2, 1]).astype(np.float32)
                img_data = np.moveaxis(img_data, 0, -1)
                p2, p98 = np.nanpercentile(img_data, (2, 98))
                img_data = np.clip((img_data - p2) / (p98 - p2 + 1e-10), 0, 1)
                img_data = np.nan_to_num(img_data, nan=0.0)
                img_data = cv2.resize(img_data, (64, 64))
                img_t = torch.from_numpy(img_data).permute(2, 0, 1).float().unsqueeze(0).to(device)

                # Tabular features scaling and prepare
                scaled_row_df = pd.DataFrame([row])
                scaled_row_df[self.cols_tab] = self.scaler.transform(scaled_row_df[self.cols_tab])
                tab_t = torch.tensor(scaled_row_df[self.cols_tab].values.astype(np.float32)).to(device)
                
                with torch.no_grad():
                    pred_ndvi_val = self.model_dl(img_t, tab_t).item()
            except Exception as e:
                logger.error(f"Error prediciendo con Two-Tower: {e}. Usando simulación de predicción.")
                pred_ndvi_val = float(row['Sat_NDVI'] * 0.98 + np.random.normal(0, 0.01))
        else:
            pred_ndvi_val = float(row['Sat_NDVI'] * 0.98 + np.random.normal(0, 0.01))

        # Determine etc_col
        etc_col = 'ETc_Sector2' if (not self.is_carralancha and self.sector == 2) else 'ETc_Sector3'
        
        # Calcular ETc representativo (usando media móvil de 3 días si el último día está incompleto)
        fecha_reporte = row['Fecha']
        df_temp = self.df[self.df['Fecha'] <= fecha_reporte].copy()
        
        es_hoy = (fecha_reporte.date() == datetime.today().date())
        mes = fecha_reporte.month
        es_epoca_riego = 4 <= mes <= 10
        es_incompleto = (es_hoy and datetime.now().hour < 22) or (es_epoca_riego and row['ET0_Calculada'] < 1.5)
        
        if es_incompleto and len(df_temp) > 1:
            etc_rep = df_temp.iloc[:-1][etc_col].tail(3).mean()
            if pd.isna(etc_rep):
                etc_rep = row[etc_col]
        else:
            etc_rep = df_temp[etc_col].tail(3).mean()
            if pd.isna(etc_rep):
                etc_rep = row[etc_col]

        lluvia_p_col = f'Lluvia_Prevista_{self.days}d'
        balance = float(row['ET0_Calculada'] - row[lluvia_p_col])
        estado_vigor = self._crop_status(float(row['Sat_NDVI']))
        stress_desc = self._classify_stress(float(row['FP_Potencial_Min']))
        
        # Inferencia LLM
        dosis_mm = 0.0
        advice = ""
        if self.llm is not None:
            try:
                input_prompt = (
                    f"NDVI: {row['Sat_NDVI']:.2f}, ET0: {row['ET0_Calculada']:.1f} mm, {lluvia_p_col}: {row[lluvia_p_col]:.1f} mm, "
                    f"Potencial_Almendra: {row['FP_Potencial_Min']:.1f} bar, Humedad_10cm: {row['FP_Humedad_10cm']:.1f}%, Riego_Histórico: {row['Riego_mm_Real']:.1f} mm"
                )
                prompt = (
                    f"<|begin_of_text|>system\nEres un Orquestador Agrónomo de precisión experto en Riego Deficitario Controlado.<|eot_id|>"
                    f"user\nAudita la eficiencia del riego e indica prescripción agronómica detallada para el {self.nombre_sector} (Almendro).\nContexto: {input_prompt}<|eot_id|>"
                    f"assistant\n"
                )
                inputs = self.tokenizer(prompt, return_tensors="pt").to(device)
                with torch.no_grad():
                    outputs = self.llm.generate(
                        **inputs,
                        max_new_tokens=150,
                        do_sample=False,
                        repetition_penalty=1.15
                    )
                gen_tokens = outputs[0][inputs["input_ids"].shape[1]:]
                advice = self.tokenizer.decode(gen_tokens, skip_special_tokens=True).strip()
                
                # Extract dose from LLM output if possible, else fall back
                match = re.search(r'dosis.*?\b(\d+(?:\.\d+)?)\s*mm', advice, re.IGNORECASE)
                if match:
                    dosis_mm = float(match.group(1))
                else:
                    _, dosis_mm = self._generate_rule_based_presc(row, etc_rep)
            except Exception as e:
                logger.error(f"Inferencia del LLM fallida. Usando motor experto alternativo de reglas. Error: {e}")
                advice, dosis_mm = self._generate_rule_based_presc(row, etc_rep)
        else:
            advice, dosis_mm = self._generate_rule_based_presc(row, etc_rep)

        # Total water volume & scheduling
        volumen_m3 = (dosis_mm * self.area_s3) / 1000.0
        # Irrigation hours
        irr_hours = dosis_mm / self.precipitation_rate
        turns = max(1, int(np.ceil(irr_hours / 6.0)))
        hours_per_turn = irr_hours / turns

        # Create plots
        img_sat_path = self._prepare_img(date_dt)
        plot_balance_path, plot_evol_path = self._generate_plots(row, history_df)

        # PDF Compilation
        pdf = AgroPDF(nombre_sector=self.nombre_sector)
        pdf.add_page()
        pdf.set_text_color(*C_TEXT)
        
        # Metadata Header
        pdf.set_font("helvetica", "", 8)
        pdf.set_text_color(100)
        pdf.cell(90, 4, f"ID PARCELA: {self.nombre_sector} | Superficie: {self.area_s3/10000:.2f} ha ({self.area_s3:,.1f} m2)")
        pdf.cell(90, 4, f"Fecha Análisis: {date_str} | Cultivo: {self.nombre_cultivo}", align="R", new_x=XPos.LMARGIN, new_y=YPos.NEXT)
        pdf.ln(2)

        # SECTION 1: RECOMENDACIÓN / PLAN DE APLICACIÓN DE RIEGO (PRÓXIMOS {self.days} DÍAS)
        pdf.section_header(f"1. PLAN DE APLICACIÓN DE RIEGO (PRÓXIMOS {self.days} DÍAS)")
        if dosis_mm > 15:
            # Warning orange box
            pdf.set_fill_color(254, 243, 199)
            pdf.set_draw_color(245, 158, 11)
            text_color = (180, 83, 9)
        else:
            # OK green box
            pdf.set_fill_color(240, 250, 240)
            pdf.set_draw_color(34, 197, 94)
            text_color = C_PRIMARY
            
        pdf.set_font("helvetica", "B", 10.5)
        pdf.set_text_color(*text_color)
        pdf.cell(0, 6, "  PLAN DE APLICACIÓN FÍSICA PARA EL AGRICULTOR", border="TLR", fill=True, new_x=XPos.LMARGIN, new_y=YPos.NEXT)
        
        pdf.set_font("helvetica", "", 9.5)
        pdf.set_text_color(*C_TEXT)
        pdf.cell(90, 6, f"    * Dosis Total Recomendada: {dosis_mm:.1f} mm", border="L", fill=True)
        pdf.cell(90, 6, f"    * Volumen Total a Aplicar: {volumen_m3:.1f} m³", border="R", fill=True, new_x=XPos.LMARGIN, new_y=YPos.NEXT)

        pdf.cell(90, 6, f"    * Tiempo de Riego Estimado: {irr_hours:.1f} horas", border="L", fill=True)
        pdf.cell(90, 6, f"    * Caudal Nominal de Diseño: {self.precipitation_rate:.2f} mm/h", border="R", fill=True, new_x=XPos.LMARGIN, new_y=YPos.NEXT)

        pdf.set_font("helvetica", "B", 9.5)
        pdf.set_text_color(*C_PRIMARY)
        pdf.cell(0, 7, f"    * Programación Sugerida: {turns} turnos de {hours_per_turn:.1f} horas cada uno", border="LBR", fill=True, new_x=XPos.LMARGIN, new_y=YPos.NEXT)
        pdf.ln(3)

        # SECTION 2: INFORME DEL AGENTE IA XAI (ESTRATEGIA Y RECOMENDACIÓN)
        pdf.section_header("2. INFORME DEL AGENTE IA XAI (ESTRATEGIA Y RECOMENDACIÓN)")
        y_start = pdf.get_y()
        # Draw background card
        pdf.set_fill_color(*C_BG_CARD)
        pdf.set_draw_color(220, 225, 220)
        pdf.rect(15, y_start, 180, 32, "FD")
        pdf.set_xy(17, y_start + 2)
        pdf.set_font("helvetica", "", 9)
        pdf.set_text_color(50)
        pdf.write_html(advice.strip())
        pdf.set_y(y_start + 32 + 3)

        # SECTION 3: RESUMEN EJECUTIVO Y RELEVANCIA
        pdf.section_header("3. RESUMEN EJECUTIVO Y RELEVANCIA")
        pdf.set_font("helvetica", "", 9.5)
        pdf.set_text_color(50)
        pdf.write_html(
            f"El <b>{self.nombre_sector}</b> de la finca <b>{self.nombre_finca}</b> está clasificado como área de alta prioridad operativa debido a su suelo <b>franco-arenoso</b>, "
            "el cual presenta un <b><u>drenaje rápido</u></b> y una <b><u>baja capacidad de retención de agua</u></b>. "
            "El monitoreo continuo a nivel radicular es indispensable para coordinar el "
            "<b>Riego Deficitario Controlado (RDC)</b>, maximizando la eficiencia de uso del agua en esta fase de desarrollo."
        )
        pdf.ln(4)

        # SECTION 4: INDICADORES CLAVE DE RENDIMIENTO (KPI)
        pdf.section_header("4. INDICADORES CLAVE DE RENDIMIENTO (KPI)")
        pdf.set_fill_color(*C_BG_CARD)
        pdf.set_draw_color(220, 225, 220)
        y_kpis = pdf.get_y()
        pdf.rect(15, y_kpis, 57, 24, "FD")
        pdf.rect(76, y_kpis, 57, 24, "FD")
        pdf.rect(138, y_kpis, 57, 24, "FD")

        # Col 1: Vigor
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
        pdf.cell(53, 4, estado_vigor)

        # Col 2: Soil moisture at 40cm
        pdf.set_xy(78, y_kpis + 2)
        pdf.set_font("helvetica", "B", 8)
        pdf.set_text_color(*C_PRIMARY)
        pdf.cell(53, 4, "HUMEDAD SUELO (40cm)", new_x=XPos.LMARGIN, new_y=YPos.NEXT)
        pdf.set_xy(78, y_kpis + 7)
        pdf.set_font("helvetica", "B", 12)
        pdf.set_text_color(41, 128, 185)
        pdf.cell(53, 6, f"{row['FP_Humedad_40cm']:.1f}%", new_x=XPos.LMARGIN, new_y=YPos.NEXT)
        pdf.set_xy(78, y_kpis + 14)
        pdf.set_font("helvetica", "", 8)
        pdf.set_text_color(80)
        pdf.cell(53, 4, "Zona Radicular Activa")

        # Col 3: Soil moisture at 10cm
        pdf.set_xy(140, y_kpis + 2)
        pdf.set_font("helvetica", "B", 8)
        pdf.set_text_color(*C_PRIMARY)
        pdf.cell(53, 4, "HUMEDAD SUELO (10cm)", new_x=XPos.LMARGIN, new_y=YPos.NEXT)
        pdf.set_xy(140, y_kpis + 7)
        pdf.set_font("helvetica", "B", 12)
        pdf.set_text_color(33, 33, 33)
        pdf.cell(53, 6, f"{row['FP_Humedad_10cm']:.1f}%", new_x=XPos.LMARGIN, new_y=YPos.NEXT)
        pdf.set_xy(140, y_kpis + 14)
        pdf.set_font("helvetica", "", 8)
        pdf.set_text_color(80)
        pdf.cell(53, 4, f"Humedad 20cm: {row['FP_Humedad_20cm']:.1f}%")

        # PAGE 2: Crop details
        pdf.add_page()
        pdf.set_font("helvetica", "", 8)
        pdf.set_text_color(100)
        pdf.cell(90, 4, f"ID PARCELA: {self.nombre_sector} | Fecha Análisis: {date_str}")
        pdf.cell(90, 4, f"Orquestador Agro-XAI", align="R", new_x=XPos.LMARGIN, new_y=YPos.NEXT)
        pdf.ln(2)

        # SECTION 5: IMAGEN DE CULTIVO Y BALANCE CLIMÁTICO
        pdf.section_header("5. IMAGEN DE CULTIVO Y BALANCE CLIMÁTICO")
        y_imgs = pdf.get_y()
        if img_sat_path and os.path.exists(img_sat_path):
            pdf.image(img_sat_path, x=15, y=y_imgs, w=90, h=45)
            
        if os.path.exists(plot_balance_path):
            pdf.image(plot_balance_path, x=110, y=y_imgs, w=85, h=50)

        pdf.set_y(y_imgs + 52)
        pdf.set_font("helvetica", "I", 8)
        pdf.set_text_color(120)
        pdf.cell(90, 4, "Radiografía del Cultivo (Satélite Real / NDVI)")
        pdf.cell(90, 4, "Balance Hídrico Operativo (ET0 vs Precipitaciones)", align="R", new_x=XPos.LMARGIN, new_y=YPos.NEXT)
        pdf.ln(3)

        # SECTION 6: DETALLE DE PARÁMETROS OPERATIVOS
        pdf.section_header("6. DETALLE DE PARÁMETROS OPERATIVOS")
        pdf.set_fill_color(240, 245, 240)
        pdf.set_draw_color(220, 220, 220)
        pdf.set_font("helvetica", "B", 8)
        pdf.set_text_color(*C_PRIMARY)
        pdf.cell(35, 6, "Indicador", border=1, fill=True)
        pdf.cell(30, 6, "Valor Actual", border=1, fill=True)
        pdf.cell(30, 6, "Previsión", border=1, fill=True)
        pdf.cell(85, 6, "Estado / Efecto Agronómico", border=1, fill=True, new_x=XPos.LMARGIN, new_y=YPos.NEXT)

        pdf.set_font("helvetica", "", 8.5)
        pdf.set_text_color(*C_TEXT)

        pdf.cell(35, 6, "Vigor Foliar (NDVI)", border=1)
        pdf.cell(30, 6, f"{row['Sat_NDVI']:.3f}", border=1)
        pdf.cell(30, 6, f"t+5: {pred_ndvi_val:.3f}", border=1)
        tendencia = ((pred_ndvi_val - row['Sat_NDVI']) / (row['Sat_NDVI'] + 1e-10)) * 100
        pdf.cell(85, 6, f"Tendencia proyectada de vigor foliar: {tendencia:+.2f}%", border=1, new_x=XPos.LMARGIN, new_y=YPos.NEXT)

        pdf.cell(35, 6, "Evapotranspiración", border=1)
        pdf.cell(30, 6, f"{row['ET0_Calculada']:.1f} mm/día", border=1)
        pdf.cell(30, 6, f"ETc: {etc_rep:.1f} mm/día", border=1)
        pdf.cell(85, 6, f"Demanda real del cultivo moderada (Kc={row['Kc_Dinamico']:.2f})", border=1, new_x=XPos.LMARGIN, new_y=YPos.NEXT)

        pdf.cell(35, 6, "Precipitación (AEMET)", border=1)
        pdf.cell(30, 6, f"{row['Lluvia_AEMET']:.1f} mm", border=1)
        pdf.cell(30, 6, f"{self.days} días: {row[lluvia_p_col]:.1f} mm", border=1)
        pdf.cell(85, 6, "Sin precipitaciones significativas. Balance neto negativo.", border=1, new_x=XPos.LMARGIN, new_y=YPos.NEXT)
        pdf.ln(3)

        # SECTION 7: HUMEDAD DEL SUELO Y EVOLUCIÓN
        pdf.section_header("7. HUMEDAD DEL SUELO Y EVOLUCIÓN")
        y_chart2 = pdf.get_y()
        if os.path.exists(plot_evol_path):
            pdf.image(plot_evol_path, x=15, y=y_chart2, w=180, h=65)
            pdf.set_y(y_chart2 + 67)
            pdf.ln(2)

        # Soil & Moisture Table
        pdf.set_font("helvetica", "B", 8)
        pdf.set_text_color(*C_PRIMARY)
        pdf.set_fill_color(240, 245, 240)
        pdf.cell(45, 6, "Sensor / Canal", border=1, fill=True)
        pdf.cell(45, 6, "Medida (%)", border=1, fill=True)
        pdf.cell(90, 6, "Análisis Agronómico y Recomendaciones de Humedad", border=1, fill=True, new_x=XPos.LMARGIN, new_y=YPos.NEXT)

        pdf.set_font("helvetica", "", 8.5)
        pdf.set_text_color(*C_TEXT)

        pdf.cell(45, 6, "Humedad Suelo 10-20cm", border=1)
        pdf.cell(45, 6, f"10cm: {row['FP_Humedad_10cm']:.1f}% | 20cm: {row['FP_Humedad_20cm']:.1f}%", border=1)
        pdf.cell(90, 6, "Humedad adecuada en la zona radicular superficial para el crecimiento.", border=1, new_x=XPos.LMARGIN, new_y=YPos.NEXT)

        pdf.cell(45, 6, "Humedad Suelo 40-60cm", border=1)
        pdf.cell(45, 6, f"40cm: {row['FP_Humedad_40cm']:.1f}% | 60cm: {row['FP_Humedad_60cm']:.1f}%", border=1)
        pdf.cell(90, 6, "Zona de absorción profunda activa. Drenaje controlado sin pérdidas.", border=1, new_x=XPos.LMARGIN, new_y=YPos.NEXT)
        pdf.ln(4)

        # Disclaimer
        pdf.set_font("helvetica", "I", 8.5)
        pdf.set_text_color(120)
        pdf.multi_cell(0, 4.5, 
            "Aviso: La recomendación de riego se basa en el balance del cultivo (ETc), el vigor foliar satelital (NDVI) y la humedad medida en el suelo. "
            "Se asume un diseño estándar de riego por goteo con doble tubería lateral. Ajustar los turnos según la disponibilidad de energía eléctrica y el caudal real de cabezal."
        )

        if output_pdf_name is None:
            if self.is_carralancha:
                suffix = f"V4_{self.days}d"
            else:
                suffix = f"S{self.sector}_{self.days}d"
            output_pdf_name = f"Prescripcion_{suffix}_{date_str}.pdf"
            
        pdf.output(output_pdf_name)
        logger.info(f"Reporte PDF guardado exitosamente como '{output_pdf_name}'")

        # Clean up temporary plots
        for p in [img_sat_path, plot_balance_path, plot_evol_path]:
            if p and os.path.exists(p) and p != "temp_sat.png":
                os.remove(p)


# ==========================================
# 5. MAIN ENTRY POINT & ARGUMENT PARSING
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Orquestador de Riego Agro-XAI con FloraPulse")
    parser.add_argument("--finca", type=str, default="canduela", choices=["canduela", "carralancha"],
                        help="Finca a procesar: canduela o carralancha.")
    parser.add_argument("--sector", type=int, default=3, choices=[2, 3],
                        help="Sector a procesar (solo aplicable para canduela: 2 o 3).")
    parser.add_argument("--days", type=int, default=3, choices=[3, 5],
                        help="Número de días para la recomendación de riego: 3 o 5.")
    parser.add_argument("--mode", type=str, default="all", choices=["all", "preprocess", "train_dl", "train_llm", "report"],
                        help="Modo de ejecución: preprocess (preprocesar datos), train_dl (entrenar modelo Two-Tower), train_llm (fine-tuning Llama), report (generar PDF) o all (completo).")
    parser.add_argument("--date", type=str, default=None, help="Fecha para el informe PDF (YYYY-MM-DD).")
    parser.add_argument("--model-dl-path", type=str, default=None, help="Ruta de guardado/lectura del modelo PyTorch.")
    parser.add_argument("--base-model-id", type=str, default="meta-llama/Llama-3.2-1B-Instruct", help="ID del modelo base de HF.")
    parser.add_argument("--adapter-path", type=str, default=None, help="Ruta de guardado/lectura del adaptador LoRA.")
    parser.add_argument("--skip-llm-train", action="store_true", help="Saltarse la fase de entrenamiento de LLM si ya hay un adaptador guardado.")
    args = parser.parse_args()

    # Create directories if they do not exist
    os.makedirs('models', exist_ok=True)
    
    # Configure dynamic paths and defaults depending on finca
    if args.finca == 'carralancha':
        base_path = 'datos/carralancha'
        if args.model_dl_path is None:
            args.model_dl_path = 'models/modelo_percepcion_v4.pt'
        if args.adapter_path is None:
            args.adapter_path = 'models/agro_llama_1b_v4_final'
        if args.date is None:
            args.date = '2025-07-20'
    else:
        base_path = 'datos/canduela'
        if args.model_dl_path is None:
            args.model_dl_path = f'models/modelo_percepcion_s{args.sector}.pt'
        if args.adapter_path is None:
            args.adapter_path = f'models/agro_llama_1b_s{args.sector}_final'
        if args.date is None:
            args.date = '2024-08-10'
            
    os.makedirs(base_path, exist_ok=True)
    os.makedirs('datos/carralancha', exist_ok=True)

    data_manager = AgroDataManager(base_path=base_path, sector=args.sector, days=args.days)
    
    # Run Preprocess (Required for all modes)
    df_master = data_manager.process_all()

    # Column definition for the neural network
    cols_tab = [
        'Temperatura', 'Lluvia_Prevista_3d', 'Riego_mm_Real', 'ET0_Calculada', 
        'Sat_Radar', 'Sat_LST', 'FP_Potencial_Min', 
        'FP_Humedad_10cm', 'FP_Humedad_20cm', 'FP_Humedad_40cm', 'FP_Humedad_60cm'
    ]

    # Two-Tower training
    if args.mode in ["all", "train_dl"]:
        img_dir = os.path.join(base_path, 'GEE_Tesis_ImageChips-20260217T075250Z-1-001/GEE_Tesis_ImageChips')
        if not os.path.exists(img_dir):
            logger.warning(f"Directorio de chips de satelite {img_dir} no existe. Saltando entrenamiento Two-Tower.")
        else:
            trainer = AgroModelTrainer(img_dir=img_dir, cols_tab=cols_tab, model_save_path=args.model_dl_path)
            trainer.train_model(df_master)

    # LLM Dataset generation & training
    if args.mode in ["all", "train_llm"] or (args.mode == "all" and not args.skip_llm_train):
        llm_optimizer = AgroLLMOptimizer(base_model_id=args.base_model_id, save_adapter_path=args.adapter_path, base_path=base_path, sector=args.sector, days=args.days)
        llm_optimizer.generate_cot_dataset(df_master)
        
        if args.mode in ["all", "train_llm"] and not args.skip_llm_train:
            try:
                llm_optimizer.train_llm()
            except Exception as e:
                logger.error(f"Error entrenando el LLM: {e}. VRAM superada o error de entorno. Continuando...")

    # PDF Report generation
    if args.mode in ["all", "report"]:
        report_generator = AgroProductionReport(
            model_dl_path=args.model_dl_path,
            base_model_id=args.base_model_id,
            adapter_path=args.adapter_path,
            base_path=base_path,
            sector=args.sector,
            days=args.days
        )
        report_generator.generate_report(args.date)

    logger.info("=== ORQUESTADOR DE RIEGO AGRO-XAI FINALIZADO CON ÉXITO ===")
