# Trader Gemini - Comandos de Referencia Rápida

## Activar Entorno Virtual
```powershell
.venv\Scripts\Activate.ps1
```

## Instalar Dependencias
```powershell
pip install -r requirements.txt
```

## Ejecutar Tests
```powershell
# Test de Integridad
python tests/test_integrity.py

# Test de Validación
python tests/test_validation.py

# Test de Conectividad (WebSocket)
python tests/test_connectivity_v2.py
```

## Ejecutar Bot
```powershell
# Modo Spot
python main.py --mode spot

# Modo Futures
python main.py --mode futures
```

## Desactivar Entorno Virtual
```powershell
deactivate
```
