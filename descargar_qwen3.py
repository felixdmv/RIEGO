# Ejecuta esto en el nodo con internet
from huggingface_hub import snapshot_download

model_id = "Qwen/Qwen3-VL-32B-Instruct"
token = "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"

print("Iniciando descarga masiva del modelo...")
snapshot_download(
    repo_id=model_id, 
    token=token,
    ignore_patterns=["*.msgpack", "*.h5", "*.ot"] # Ahorramos espacio
)
print("¡Descarga completada! Ya puedes volver al nodo de GPU.")