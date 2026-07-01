import os
import glob

root_dir = r"C:\Users\fdemiguel\OneDrive - Universidad de Burgos\Documentos\RIEGO"

# Deleting scratch files
scratch_files = glob.glob(os.path.join(root_dir, "scratch_*.py"))
for f in scratch_files:
    try:
        os.remove(f)
        print(f"Eliminado: {os.path.basename(f)}")
    except Exception as e:
        print(f"Error al eliminar {f}: {e}")

# Also delete actualizar_datos_diarios.py since its logic is fully integrated in run_pipeline.py and ejecutar_recomendador_diario.py
path_act = os.path.join(root_dir, "actualizar_datos_diarios.py")
if os.path.exists(path_act):
    try:
        os.remove(path_act)
        print("Eliminado actualizar_datos_diarios.py (integrado)")
    except Exception as e:
        pass
