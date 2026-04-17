"""
🧹 CLEANUP DE MODELOS ML - Trader Gemini
=========================================
QUÉ: Limpia modelos obsoletos, backups y versiones zombie del directorio .models/
POR QUÉ: MLGovernance acumula versiones sin límite, causando 198 archivos (29MB)
PARA QUÉ: Mantener solo los 3 modelos más recientes por símbolo y liberar disco
CÓMO: 1) Borrar .old/.v2 backups, 2) Podar versiones >3, 3) Borrar stale >7 días
CUÁNDO: Antes de cada backtest o deployment
DÓNDE: scripts/cleanup_models.py
QUIÉN: QA Engineer + SRE
"""

import os
import re
import shutil
import time
from pathlib import Path
from datetime import datetime, timedelta
from collections import defaultdict

MODELS_DIR = Path(os.path.join(os.path.dirname(os.path.dirname(__file__)), ".models"))
MAX_VERSIONS_PER_SYMBOL = 3
MAX_AGE_DAYS = 7


def cleanup_backup_files(models_dir: Path) -> dict:
    """Elimina archivos .old y .v2 que son backups redundantes."""
    stats = {"deleted": 0, "bytes_freed": 0, "files": []}
    
    for f in models_dir.iterdir():
        if f.is_file() and (f.name.endswith(".old") or f.name.endswith(".v2")):
            size = f.stat().st_size
            stats["files"].append(f.name)
            stats["bytes_freed"] += size
            stats["deleted"] += 1
            f.unlink()
    
    return stats


def cleanup_versioned_dirs(models_dir: Path, keep_n: int = MAX_VERSIONS_PER_SYMBOL) -> dict:
    """Podar directorios de versiones, manteniendo solo las keep_n más recientes por símbolo."""
    stats = {"deleted": 0, "bytes_freed": 0, "dirs": []}
    
    # Agrupar por símbolo base (e.g., BTC_USDT)
    pattern = re.compile(r"^(.+)_v(\d+)_(\d{8})$")
    symbol_versions = defaultdict(list)
    
    for d in models_dir.iterdir():
        if d.is_dir():
            match = pattern.match(d.name)
            if match:
                symbol = match.group(1)
                version = int(match.group(2))
                date_str = match.group(3)
                symbol_versions[symbol].append((version, date_str, d))
    
    for symbol, versions in symbol_versions.items():
        # Ordenar por versión descendente (más reciente primero)
        versions.sort(key=lambda x: x[0], reverse=True)
        
        # Borrar las versiones más allá del límite
        to_delete = versions[keep_n:]
        for version, date_str, dir_path in to_delete:
            try:
                dir_size = sum(f.stat().st_size for f in dir_path.rglob("*") if f.is_file())
                stats["dirs"].append(dir_path.name)
                stats["bytes_freed"] += dir_size
                stats["deleted"] += 1
                shutil.rmtree(dir_path)
            except Exception as e:
                print(f"  ⚠️ Error borrando {dir_path.name}: {e}")
    
    return stats


def cleanup_stale_models(models_dir: Path, max_age_days: int = MAX_AGE_DAYS) -> dict:
    """Borrar modelos joblib con timestamp > max_age_days."""
    stats = {"deleted": 0, "bytes_freed": 0, "files": []}
    cutoff = time.time() - (max_age_days * 86400)
    
    for f in models_dir.iterdir():
        if f.is_file() and f.suffix == ".joblib":
            if f.stat().st_mtime < cutoff:
                stats["files"].append(f.name)
                stats["bytes_freed"] += f.stat().st_size
                stats["deleted"] += 1
                f.unlink()
    
    return stats


def main():
    print("=" * 60)
    print("🧹 TRADER GEMINI - ML MODEL CLEANUP")
    print("=" * 60)
    
    if not MODELS_DIR.exists():
        print("❌ Directorio .models/ no encontrado.")
        return
    
    # Pre-conteo
    pre_files = sum(1 for _ in MODELS_DIR.rglob("*") if _.is_file())
    pre_size = sum(f.stat().st_size for f in MODELS_DIR.rglob("*") if f.is_file())
    print(f"\n📊 Estado Inicial: {pre_files} archivos, {pre_size / 1024 / 1024:.2f} MB")
    
    # Paso 1: Backups
    print(f"\n🔹 Paso 1: Limpiando backups (.old, .v2)...")
    backup_stats = cleanup_backup_files(MODELS_DIR)
    print(f"   Borrados: {backup_stats['deleted']} archivos ({backup_stats['bytes_freed'] / 1024:.1f} KB)")
    
    # Paso 2: Versiones
    print(f"\n🔹 Paso 2: Podando versiones (max {MAX_VERSIONS_PER_SYMBOL} por símbolo)...")
    version_stats = cleanup_versioned_dirs(MODELS_DIR, MAX_VERSIONS_PER_SYMBOL)
    print(f"   Borrados: {version_stats['deleted']} directorios ({version_stats['bytes_freed'] / 1024:.1f} KB)")
    
    # Paso 3: Stale
    print(f"\n🔹 Paso 3: Eliminando modelos obsoletos (>{MAX_AGE_DAYS} días)...")
    stale_stats = cleanup_stale_models(MODELS_DIR, MAX_AGE_DAYS)
    print(f"   Borrados: {stale_stats['deleted']} archivos ({stale_stats['bytes_freed'] / 1024:.1f} KB)")
    
    # Post-conteo
    post_files = sum(1 for _ in MODELS_DIR.rglob("*") if _.is_file())
    post_size = sum(f.stat().st_size for f in MODELS_DIR.rglob("*") if f.is_file())
    
    total_freed = pre_size - post_size
    print(f"\n{'='*60}")
    print(f"📊 Estado Final:  {post_files} archivos, {post_size / 1024 / 1024:.2f} MB")
    print(f"🗑️  Liberados:     {total_freed / 1024 / 1024:.2f} MB ({pre_files - post_files} archivos)")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
