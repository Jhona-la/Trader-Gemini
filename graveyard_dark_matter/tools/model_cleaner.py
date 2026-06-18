import os
import sys
import glob

# Ensure root
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def clean_legacy_models():
    print("🧹 [MODEL CLEANER] Iniciando purga de modelos JSON legacy...")
    
    models_dir = "models"
    if not os.path.exists(models_dir):
        print("❌ Directorio 'models/' no encontrado.")
        return
        
    json_files = glob.glob(os.path.join(models_dir, "*_xgb.json"))
    
    deleted_count = 0
    bytes_freed = 0
    
    for file in json_files:
        try:
            size_b = os.path.getsize(file)
            os.remove(file)
            deleted_count += 1
            bytes_freed += size_b
            print(f"🗑️ Eliminado: {file} ({(size_b / 1024):.2f} KB)")
        except Exception as e:
            print(f"❌ Error al eliminar {file}: {e}")
            
    print("="*50)
    print(f"✅ OPERACIÓN COMPLETADA")
    print(f"   Modelos purgados: {deleted_count}")
    print(f"   Espacio liberado: {(bytes_freed / 1024 / 1024):.2f} MB")
    print("="*50)
    print("🚀 El sistema ahora opera exclusivamente con modelos de Alta Velocidad (UBJSON).")
    print("⚠️  IMPORTANTE: Recuerde ejecutar 'python analysis/train_supreme.py' para regenerar la inteligencia binaria.")

if __name__ == "__main__":
    clean_legacy_models()
