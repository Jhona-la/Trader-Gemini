"""
Script para limpiar modelos corruptos de ML
Ejecutar: python fix_corrupted_models.py
"""
import os
import shutil
from pathlib import Path

def clean_corrupted_models():
    """Eliminar todos los modelos guardados para forzar reentrenamiento limpio"""
    
    models_dir = Path(".models")
    
    if not models_dir.exists():
        print("✅ No hay directorio de modelos. Nada que limpiar.")
        return
    
    print(f"🔍 Buscando modelos en: {models_dir.absolute()}")
    
    model_files = list(models_dir.glob("*.joblib"))
    
    if not model_files:
        print("✅ No hay archivos de modelos. Nada que limpiar.")
        return
    
    print(f"\n📁 Encontrados {len(model_files)} archivos de modelos:")
    for f in model_files:
        print(f"   - {f.name}")
    
    # Crear backup antes de eliminar
    backup_dir = Path(".models_backup")
    if backup_dir.exists():
        shutil.rmtree(backup_dir)
    
    print(f"\n💾 Creando backup en: {backup_dir.absolute()}")
    shutil.copytree(models_dir, backup_dir)
    print("✅ Backup completado")
    
    # Eliminar modelos corruptos
    print(f"\n🗑️  Eliminando modelos corruptos...")
    for model_file in model_files:
        try:
            model_file.unlink()
            print(f"   ✅ Eliminado: {model_file.name}")
        except Exception as e:
            print(f"   ❌ Error eliminando {model_file.name}: {e}")
    
    print("\n" + "="*60)
    print("✨ LIMPIEZA COMPLETADA")
    print("="*60)
    print("📌 SIGUIENTE PASO:")
    print("   1. Los modelos se reentrenarán automáticamente al iniciar")
    print("   2. Si necesitas restaurar: copiar .models_backup → .models")
    print("="*60)

if __name__ == "__main__":
    print("="*60)
    print("🧹 LIMPIEZA DE MODELOS ML CORRUPTOS")
    print("="*60)
    
    try:
        clean_corrupted_models()
    except Exception as e:
        print(f"\n❌ ERROR: {e}")
        import traceback
        traceback.print_exc()