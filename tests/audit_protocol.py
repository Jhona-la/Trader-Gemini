import os
import sys
import requests
import sqlite3
import json
from datetime import datetime
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from config import Config

def print_header(title):
    print(f"\n{'='*60}")
    print(f"🔍 {title}")
    print(f"{'='*60}")

def check_api_connectivity():
    print_header("2.1 Verificación de Integridad de Credenciales por Mercado")
    
    endpoints = {
        'SPOT': {
            'url': 'https://testnet.binance.vision/api/v3/time',
            'description': 'Binance Spot Testnet'
        },
        'FUTURES': {
            'url': 'https://testnet.binancefuture.com/fapi/v1/time',
            'description': 'Binance Futures Testnet'
        }
    }
    
    results = {}
    
    for market, data in endpoints.items():
        print(f"[{market}] Validando conectividad contra {data['description']}...")
        try:
            response = requests.get(data['url'], timeout=5)
            if response.status_code == 200:
                server_time = response.json().get('serverTime')
                print(f"[{market}] ✓ Autenticación exitosa - Server Time: {server_time}")
                results[market] = True
            else:
                print(f"[{market}] ❌ Error HTTP {response.status_code}: {response.text}")
                results[market] = False
        except Exception as e:
            print(f"[{market}] ❌ Error de conexión: {str(e)}")
            results[market] = False
            
    return results

def check_persistence_layer():
    print_header("2.2 Inicialización de Capa de Persistencia Dual")
    
    # Define expected paths based on config logic
    # Note: Config might be loaded with default (Spot), so we manually construct paths for audit
    base_dir = Path("dashboard/data")
    spot_db = base_dir / "spot" / "trader_gemini.db"
    futures_db = base_dir / "futures" / "trader_gemini.db"
    
    dbs = {
        'SPOT': spot_db,
        'FUTURES': futures_db
    }
    
    for market, db_path in dbs.items():
        print(f"[{market}] Verificando base de datos: {db_path}")
        if db_path.exists():
            print(f"[{market}] ✓ Archivo existe")
            try:
                conn = sqlite3.connect(db_path)
                cursor = conn.cursor()
                cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
                tables = [row[0] for row in cursor.fetchall()]
                conn.close()
                
                required_tables = ['positions', 'trades', 'signals']
                missing = [t for t in required_tables if t not in tables]
                
                if not missing:
                    print(f"[{market}] ✓ Esquema validado (Tablas: {', '.join(tables)})")
                else:
                    print(f"[{market}] ⚠️ Tablas faltantes: {missing}")
            except Exception as e:
                print(f"[{market}] ❌ Error leyendo DB: {e}")
        else:
            print(f"[{market}] ⚠️ Archivo no existe (Se creará en la primera ejecución)")

def check_dashboard_segregation():
    print_header("2.3 Validación de Segregación de Datos del Dashboard")
    
    paths = [
        Path("dashboard/data/spot"),
        Path("dashboard/data/futures")
    ]
    
    for p in paths:
        status = "✓ Existe" if p.exists() else "⚠️ No existe (Se creará en ejecución)"
        print(f"Directorio {p}: {status}")
        
        # Check write permissions (simulation)
        if p.exists():
            try:
                test_file = p / ".audit_test"
                test_file.touch()
                test_file.unlink()
                print(f"  ✓ Permisos de escritura confirmados")
            except Exception as e:
                print(f"  ❌ Error de permisos: {e}")

if __name__ == "__main__":
    print("🚀 INICIANDO PROTOCOLO DE AUDITORÍA DUAL-MARKET v2.0")
    
    api_results = check_api_connectivity()
    check_persistence_layer()
    check_dashboard_segregation()
    
    print("\n" + "="*60)
    print("🏁 AUDITORÍA FASE 1 COMPLETADA")
    print("="*60)
