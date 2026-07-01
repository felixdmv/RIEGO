#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Fine-tuning QLoRA para Qwen3-VL-32B en cluster 3xA100 (40GB)
con DeepSpeed ZeRO-2/ZeRO-3 o FSDP y dataset multimodal agricola.

Uso tipico:
  conda run -n agro_ft python pipeline_32B.py \
    --train_jsonl dataset_auditoria_sector3.jsonl \
    --image_dir datos/GEE_Tesis_ImageChips-20260217T075250Z-1-001/GEE_Tesis_ImageChips \
    --output_dir runs/qwen3_vl_32b_qlora \
    --distributed_backend auto \
    --deepspeed_stage zero3 \
    --offline

Lanzamiento distribuido recomendado:
  accelerate launch --config_file runs/qwen3_vl_32b_qlora/configs/accelerate_fsdp.yaml pipeline_32B.py \
    --train_jsonl dataset_auditoria_sector3.jsonl \
    --image_dir datos/GEE_Tesis_ImageChips-20260217T075250Z-1-001/GEE_Tesis_ImageChips \
    --output_dir runs/qwen3_vl_32b_qlora \
    --distributed_backend auto \
    --offline
"""

import os
import re
import gc
import json
import argparse
import logging
import importlib.metadata
import subprocess
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import torch
import numpy as np
import rasterio
from PIL import Image
from torch.utils.data import Dataset

from transformers import (
    AutoProcessor,
    Qwen3VLForConditionalGeneration,
    BitsAndBytesConfig,
    Trainer,
    TrainingArguments,
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training


logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger("agri_qwen3_vl_qlora")

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True,max_split_size_mb:256")
os.environ.setdefault("HF_ENABLE_PARALLEL_LOADING", "false")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

DATE_RE = re.compile(r"Fecha\s*:\s*(\d{4}-\d{2}-\d{2})", re.IGNORECASE)
IMG_KEY_CANDIDATES = [
    "image",
    "image_path",
    "img",
    "img_path",
    "tif",
    "tif_path",
    "chip",
    "chip_path",
    "ruta_imagen",
]


def has_package(pkg_name: str) -> bool:
    try:
        importlib.metadata.version(pkg_name)
        return True
    except importlib.metadata.PackageNotFoundError:
        return False


def set_offline_mode(enabled: bool) -> None:
    if enabled:
        os.environ["TRANSFORMERS_OFFLINE"] = "1"
        os.environ["HF_DATASETS_OFFLINE"] = "1"
        os.environ["HF_HUB_OFFLINE"] = "1"
    else:
        os.environ.pop("TRANSFORMERS_OFFLINE", None)
        os.environ.pop("HF_DATASETS_OFFLINE", None)
        os.environ.pop("HF_HUB_OFFLINE", None)


def resolve_backend(requested: str) -> str:
    if requested == "auto":
        if has_package("deepspeed"):
            return "deepspeed"
        return "fsdp"
    return requested


def query_gpu_free_mb() -> Dict[int, int]:
    if not torch.cuda.is_available():
        return {}
    cmd = [
        "nvidia-smi",
        "--query-gpu=memory.free",
        "--format=csv,nounits,noheader",
    ]
    output = subprocess.check_output(cmd).decode("utf-8").strip().splitlines()
    return {idx: int(value.strip()) for idx, value in enumerate(output)}


def resolve_model_id_for_runtime(
    requested_model_id: str,
    fallback_model_id: str,
    backend: str,
    local_rank: int,
    force_fsdp_32b: bool,
    min_free_mb_32b_fsdp: int,
) -> str:
    """
    Con FSDP, cada rank intenta materializar el modelo cuantizado antes del sharding.
    Para 32B eso suele reventar si la GPU local no esta practicamente vacia.
    """
    if "32B" not in requested_model_id.upper():
        return requested_model_id

    free_map = query_gpu_free_mb()
    local_free = free_map.get(local_rank)
    if local_free is not None:
        logger.info("GPU local rank %s tiene %s MiB libres antes de cargar el modelo.", local_rank, local_free)

    if backend == "fsdp" and not force_fsdp_32b:
        if not fallback_model_id:
            raise RuntimeError(
                "FSDP con Qwen3-VL-32B en 4-bit no es viable en este nodo con la carga actual. "
                "Pasa --force_fsdp_32b para intentarlo igualmente o define --fallback_model_id."
            )
        logger.warning(
            "Backend FSDP detectado: cambiando de %s a fallback %s para evitar OOM durante la carga.",
            requested_model_id,
            fallback_model_id,
        )
        return fallback_model_id

    if local_free is not None and local_free < min_free_mb_32b_fsdp and not force_fsdp_32b:
        if not fallback_model_id:
            raise RuntimeError(
                f"Solo hay {local_free} MiB libres en la GPU local; 32B no va a cargar con margen seguro. "
                "Define --fallback_model_id o usa --force_fsdp_32b bajo tu responsabilidad."
            )
        logger.warning(
            "VRAM libre insuficiente para %s en GPU %s (%s MiB). Uso fallback %s.",
            requested_model_id,
            local_rank,
            local_free,
            fallback_model_id,
        )
        return fallback_model_id

    return requested_model_id


def build_deepspeed_config(stage: str, gradient_accumulation_steps: int, micro_batch_size: int) -> Dict:
    if stage not in {"zero2", "zero3"}:
        raise ValueError("deepspeed stage debe ser 'zero2' o 'zero3'")

    if stage == "zero2":
        zero_cfg = {
            "stage": 2,
            "contiguous_gradients": True,
            "overlap_comm": True,
            "reduce_scatter": True,
            "allgather_bucket_size": 200000000,
            "reduce_bucket_size": 200000000,
        }
    else:
        zero_cfg = {
            "stage": 3,
            "contiguous_gradients": True,
            "overlap_comm": True,
            "reduce_scatter": True,
            "allgather_partitions": True,
            "allgather_bucket_size": 200000000,
            "reduce_bucket_size": 200000000,
            "sub_group_size": 1000000000,
            "stage3_prefetch_bucket_size": 50000000,
            "stage3_param_persistence_threshold": 100000,
            "stage3_max_live_parameters": 1000000000,
            "stage3_max_reuse_distance": 1000000000,
            "stage3_gather_16bit_weights_on_model_save": True,
        }

    return {
        "bf16": {"enabled": True},
        "train_micro_batch_size_per_gpu": micro_batch_size,
        "gradient_accumulation_steps": gradient_accumulation_steps,
        "gradient_clipping": 1.0,
        "zero_optimization": zero_cfg,
        "steps_per_print": 50,
        "wall_clock_breakdown": False,
    }


def build_accelerate_deepspeed_template(ds_cfg_path: str) -> str:
    return f"""compute_environment: LOCAL_MACHINE
debug: false
deepspeed_config:
  deepspeed_config_file: {ds_cfg_path}
  zero3_init_flag: true
distributed_type: DEEPSPEED
downcast_bf16: 'no'
mixed_precision: bf16
num_machines: 1
num_processes: 3
machine_rank: 0
main_training_function: main
same_network: true
use_cpu: false
"""


def build_accelerate_fsdp_template() -> str:
    return """compute_environment: LOCAL_MACHINE
debug: false
distributed_type: FSDP
downcast_bf16: 'no'
fsdp_config:
  fsdp_auto_wrap_policy: TRANSFORMER_BASED_WRAP
  fsdp_backward_prefetch: BACKWARD_PRE
  fsdp_cpu_ram_efficient_loading: true
  fsdp_forward_prefetch: false
  fsdp_offload_params: false
  fsdp_sharding_strategy: FULL_SHARD
  fsdp_state_dict_type: SHARDED_STATE_DICT
  fsdp_sync_module_states: true
mixed_precision: bf16
num_machines: 1
num_processes: 3
machine_rank: 0
main_training_function: main
same_network: true
use_cpu: false
"""


def write_configs(output_dir: Path, stage: str, gas: int, micro_bs: int) -> Tuple[Path, Path, Path]:
    cfg_dir = output_dir / "configs"
    cfg_dir.mkdir(parents=True, exist_ok=True)

    ds_path = cfg_dir / f"deepspeed_{stage}.json"
    acc_ds_path = cfg_dir / f"accelerate_{stage}.yaml"
    acc_fsdp_path = cfg_dir / "accelerate_fsdp.yaml"

    with ds_path.open("w", encoding="utf-8") as f:
        json.dump(build_deepspeed_config(stage, gas, micro_bs), f, ensure_ascii=False, indent=2)

    with acc_ds_path.open("w", encoding="utf-8") as f:
        f.write(build_accelerate_deepspeed_template(str(ds_path)))

    with acc_fsdp_path.open("w", encoding="utf-8") as f:
        f.write(build_accelerate_fsdp_template())

    logger.info("Config DeepSpeed escrita en: %s", ds_path)
    logger.info("Template Accelerate DeepSpeed escrita en: %s", acc_ds_path)
    logger.info("Template Accelerate FSDP escrita en: %s", acc_fsdp_path)
    return ds_path, acc_ds_path, acc_fsdp_path


@dataclass
class Sample:
    instruction: str
    input_text: str
    output_text: str
    image_path: Path


class AgriVLMTrainerDataset(Dataset):
    def __init__(self, jsonl_path: Path, image_dir: Path):
        self.jsonl_path = jsonl_path
        self.image_dir = image_dir
        self.rows = self._load_jsonl(jsonl_path)
        self.chips = self._index_tif_chips(image_dir)

        if not self.rows:
            raise ValueError(f"Dataset vacio: {jsonl_path}")
        if not self.chips:
            raise ValueError(f"No se encontraron chips .tif en: {image_dir}")

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, idx: int) -> Sample:
        row = self.rows[idx]
        instruction = str(row.get("instruction", "")).strip() or "Realiza auditoria agricola y simulacion contrafactual de riego."
        input_text = str(row.get("input", "")).strip()
        output_text = str(row.get("output", "")).strip() or "Sin salida etiquetada."
        image_path = self._resolve_image_path(row, input_text)
        return Sample(instruction=instruction, input_text=input_text, output_text=output_text, image_path=image_path)

    @staticmethod
    def _load_jsonl(path: Path) -> List[Dict]:
        rows: List[Dict] = []
        with path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    rows.append(json.loads(line))
        return rows

    @staticmethod
    def _extract_year_doy_from_chip_name(name: str) -> Optional[Tuple[int, int]]:
        match = re.search(r"(\d{4})-(\d{2})-(\d{2,3})", name)
        if not match:
            return None
        year = int(match.group(1))
        tail = int(match.group(3))
        if tail > 31:
            return year, tail
        month = int(match.group(2))
        dt = datetime(year, month, max(1, min(28, tail)))
        return year, dt.timetuple().tm_yday

    def _index_tif_chips(self, image_dir: Path) -> List[Tuple[int, int, Path]]:
        chips: List[Tuple[int, int, Path]] = []
        for path in sorted(image_dir.glob("*.tif")):
            yd = self._extract_year_doy_from_chip_name(path.name)
            if yd is not None:
                chips.append((yd[0], yd[1], path))
        return chips

    def _resolve_image_path(self, row: Dict, input_text: str) -> Path:
        for key in IMG_KEY_CANDIDATES:
            if key in row and row[key]:
                candidate = Path(str(row[key]))
                if not candidate.is_absolute():
                    candidate = self.image_dir / candidate
                if candidate.exists():
                    return candidate

        match = DATE_RE.search(input_text or "")
        if match:
            dt = datetime.strptime(match.group(1), "%Y-%m-%d")
            year = dt.year
            doy = dt.timetuple().tm_yday
            year_candidates = [(abs(chip_doy - doy), path) for chip_year, chip_doy, path in self.chips if chip_year == year]
            if year_candidates:
                year_candidates.sort(key=lambda item: item[0])
                return year_candidates[0][1]

        return self.chips[0][2]


def load_tif_as_pil_rgb(img_path: Path, out_size: int = 448) -> Image.Image:
    with rasterio.open(img_path) as src:
        arr = src.read([3, 2, 1])
    arr = np.moveaxis(arr, 0, -1).astype(np.float32)
    p2, p98 = np.nanpercentile(arr, (2, 98))
    arr = np.clip((arr - p2) / (p98 - p2 + 1e-10), 0, 1)
    img_8 = (arr * 255).astype(np.uint8)
    img_8 = cv2.resize(img_8, (out_size, out_size), interpolation=cv2.INTER_CUBIC)
    return Image.fromarray(img_8)


class AgriVLMDataCollator:
    def __init__(self, processor: AutoProcessor, image_size: int = 448):
        self.processor = processor
        self.image_size = image_size

    def __call__(self, batch: List[Sample]) -> Dict[str, torch.Tensor]:
        texts: List[str] = []
        images: List[Image.Image] = []

        for sample in batch:
            user_text = (
                f"{sample.instruction}\n\n"
                f"DATOS TELEMETRIA:\n{sample.input_text}\n\n"
                f"Genera ANALISIS CONTRAFACTUAL (0mm vs optimo), justificacion y prescripcion final."
            )
            messages = [
                {
                    "role": "user",
                    "content": [
                        {"type": "image", "image": "<image>"},
                        {"type": "text", "text": user_text},
                    ],
                },
                {
                    "role": "assistant",
                    "content": [{"type": "text", "text": sample.output_text}],
                },
            ]
            texts.append(self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=False))
            images.append(load_tif_as_pil_rgb(sample.image_path, out_size=self.image_size))

        model_inputs = self.processor(
            text=texts,
            images=images,
            padding=True,
            truncation=True,
            max_length=4096,
            return_tensors="pt",
        )
        labels = model_inputs["input_ids"].clone()
        labels[labels == self.processor.tokenizer.pad_token_id] = -100
        model_inputs["labels"] = labels
        return model_inputs


def pick_qwen_vl_target_modules(model: torch.nn.Module) -> List[str]:
    preferred = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
    found = set()
    for name, module in model.named_modules():
        if isinstance(module, torch.nn.Linear):
            suffix = name.split(".")[-1]
            if suffix in preferred:
                found.add(suffix)
    return [item for item in preferred if item in found] or preferred


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="QLoRA Fine-Tuning Qwen3-VL-32B (Agri DSS)")
    parser.add_argument("--model_id", type=str, default="Qwen/Qwen3-VL-32B-Instruct")
    parser.add_argument("--train_jsonl", type=str, default="dataset_auditoria_sector3.jsonl")
    parser.add_argument("--image_dir", type=str, default="datos/GEE_Tesis_ImageChips-20260217T075250Z-1-001/GEE_Tesis_ImageChips")
    parser.add_argument("--output_dir", type=str, default="runs/qwen3_vl_32b_qlora")
    parser.add_argument("--fallback_model_id", type=str, default="")

    parser.add_argument("--offline", action="store_true", default=False)
    parser.add_argument("--local_files_only", action="store_true", default=False)
    parser.add_argument("--write_configs_only", action="store_true", default=False)
    parser.add_argument("--deepspeed_stage", type=str, default="zero3", choices=["zero2", "zero3"])
    parser.add_argument("--distributed_backend", type=str, default="auto", choices=["auto", "deepspeed", "fsdp", "none"])
    parser.add_argument("--force_fsdp_32b", action="store_true", default=False)
    parser.add_argument("--min_free_mb_32b_fsdp", type=int, default=30000)

    parser.add_argument("--num_train_epochs", type=float, default=1.0)
    parser.add_argument("--learning_rate", type=float, default=2e-4)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--warmup_ratio", type=float, default=0.03)
    parser.add_argument("--per_device_train_batch_size", type=int, default=1)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=8)
    parser.add_argument("--logging_steps", type=int, default=10)
    parser.add_argument("--save_steps", type=int, default=100)

    parser.add_argument("--lora_r", type=int, default=32)
    parser.add_argument("--lora_alpha", type=int, default=64)
    parser.add_argument("--lora_dropout", type=float, default=0.05)
    parser.add_argument("--image_size", type=int, default=448)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    set_offline_mode(args.offline)
    ds_cfg_path, acc_ds_path, acc_fsdp_path = write_configs(
        output_dir=output_dir,
        stage=args.deepspeed_stage,
        gas=args.gradient_accumulation_steps,
        micro_bs=args.per_device_train_batch_size,
    )

    backend = resolve_backend(args.distributed_backend)
    if backend == "deepspeed" and not has_package("deepspeed"):
        raise RuntimeError(
            "Has solicitado entrenamiento 32B con DeepSpeed, pero el entorno no tiene instalado el paquete 'deepspeed'. "
            "Para Qwen3-VL-32B en este nodo, DeepSpeed ZeRO-3 es la ruta recomendada."
        )

    if args.write_configs_only:
        logger.info("Solo configuracion generada. Backend resuelto: %s", backend)
        return

    local_files_only = args.local_files_only or args.offline
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    device_map = {"": local_rank} if torch.cuda.is_available() else None
    if "32B" in args.model_id.upper() and backend == "fsdp" and not args.force_fsdp_32b:
        raise RuntimeError(
            "Qwen3-VL-32B con FSDP no es viable en este nodo compartido porque cada rank materializa el modelo antes del sharding. "
            "Para 32B real, instala DeepSpeed y lanza con --distributed_backend deepspeed. "
            "Si aun asi quieres forzar la prueba con FSDP, usa --force_fsdp_32b."
        )

    runtime_model_id = resolve_model_id_for_runtime(
        requested_model_id=args.model_id,
        fallback_model_id=args.fallback_model_id,
        backend=backend,
        local_rank=local_rank,
        force_fsdp_32b=args.force_fsdp_32b,
        min_free_mb_32b_fsdp=args.min_free_mb_32b_fsdp,
    )

    logger.info("Backend distribuido resuelto: %s", backend)
    logger.info("Modelo solicitado: %s", args.model_id)
    logger.info("Modelo activo en runtime: %s", runtime_model_id)
    logger.info("Cargando processor: %s", runtime_model_id)
    processor = AutoProcessor.from_pretrained(
        runtime_model_id,
        local_files_only=local_files_only,
        trust_remote_code=True,
    )
    if processor.tokenizer.pad_token is None:
        processor.tokenizer.pad_token = processor.tokenizer.eos_token

    logger.info("Cargando modelo 4-bit: %s en local_rank=%s", runtime_model_id, local_rank)
    quant_cfg = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    )
    model = Qwen3VLForConditionalGeneration.from_pretrained(
        runtime_model_id,
        quantization_config=quant_cfg,
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
        local_files_only=local_files_only,
        device_map=device_map,
        low_cpu_mem_usage=True,
    )

    model.config.use_cache = False
    model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=True)
    model.gradient_checkpointing_enable()

    target_modules = pick_qwen_vl_target_modules(model)
    logger.info("LoRA target_modules: %s", target_modules)
    lora_cfg = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        target_modules=target_modules,
        bias="none",
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, lora_cfg)
    model.print_trainable_parameters()

    train_dataset = AgriVLMTrainerDataset(Path(args.train_jsonl), Path(args.image_dir))
    collator = AgriVLMDataCollator(processor=processor, image_size=args.image_size)

    effective_batch = max(1, args.per_device_train_batch_size * max(1, args.gradient_accumulation_steps) * max(1, torch.cuda.device_count()))
    steps_per_epoch = max(1, len(train_dataset) // effective_batch)
    warmup_steps = max(1, int(steps_per_epoch * args.num_train_epochs * args.warmup_ratio))

    training_kwargs = dict(
        output_dir=str(output_dir),
        bf16=True,
        fp16=False,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        warmup_steps=warmup_steps,
        lr_scheduler_type="cosine",
        num_train_epochs=args.num_train_epochs,
        per_device_train_batch_size=args.per_device_train_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        gradient_checkpointing=True,
        max_grad_norm=1.0,
        logging_steps=args.logging_steps,
        save_steps=args.save_steps,
        save_total_limit=3,
        report_to="none",
        remove_unused_columns=False,
        dataloader_num_workers=2,
        dataloader_pin_memory=True,
        ddp_find_unused_parameters=False,
    )

    if backend == "deepspeed":
        training_kwargs["deepspeed"] = str(ds_cfg_path)
    elif backend == "fsdp":
        training_kwargs["fsdp"] = "full_shard auto_wrap"
        training_kwargs["fsdp_config"] = {
            "backward_prefetch": "backward_pre",
            "forward_prefetch": False,
            "limit_all_gathers": True,
            "sync_module_states": True,
            "use_orig_params": False,
            "cpu_ram_efficient_loading": True,
            "activation_checkpointing": True,
            "transformer_layer_cls_to_wrap": [
                "Qwen3VLDecoderLayer",
                "Qwen2VLDecoderLayer",
            ],
        }

    training_args = TrainingArguments(**training_kwargs)

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        data_collator=collator,
    )

    logger.info("Iniciando entrenamiento...")
    train_result = trainer.train()

    logger.info("Guardando adaptadores LoRA...")
    trainer.model.save_pretrained(output_dir / "adapter")
    processor.save_pretrained(output_dir / "processor")

    metrics = train_result.metrics
    with (output_dir / "train_metrics.json").open("w", encoding="utf-8") as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2)

    gc.collect()
    torch.cuda.empty_cache()
    logger.info("Entrenamiento completado. Metricas: %s", metrics)
    logger.info("DeepSpeed config: %s", ds_cfg_path)
    logger.info("Accelerate DeepSpeed template: %s", acc_ds_path)
    logger.info("Accelerate FSDP template: %s", acc_fsdp_path)


if __name__ == "__main__":
    main()
