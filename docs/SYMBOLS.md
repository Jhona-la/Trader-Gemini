# 💱 TRADER GEMINI: GESTIÓN DE SÍMBOLOS DINÁMICA

Este sistema ha sido diseñado para operar con una cesta institucional de **26 Activos**. La arquitectura permite modificar, añadir o eliminar símbolos desde un único punto de configuración, propagando los cambios automáticamente a todo el sistema.

---

## 📍 SINGLE SOURCE OF TRUTH (Fuente Única de Verdad)

El archivo maestro es `config.py`. 

```python
# config.py

class Config:
    # ...
    CRYPTO_FUTURES_PAIRS = [
        "BTC/USDT", "ETH/USDT", "SOL/USDT", 
        # ... hasta 26 pares ...
    ]
```

Cualquier cambio aquí afectará a:
1. **Data Loader**: Suscripciones a Websockets de Binance.
2. **Strategy Engine**: Instanciación automática de `MLStrategy` para cada par.
3. **Portfolio**: Seguimiento de PnL y estado.
4. **Dashboard**: Selectores y tablas de monitoreo.

---

## 🔄 CÓMO REEMPLAZAR UN SÍMBOLO (Hot-Swap Process)

Si deseas cambiar, por ejemplo, `LTC/USDT` por `APT/USDT`:

1. **Detener el Bot**: `Ctrl + C` en la terminal.
2. **Editar `config.py`**:
   ```diff
   - "LTC/USDT",
   + "APT/USDT",
   ```
3. **Reiniciar**:
   ```bash
   python main.py --mode futures
   ```

**¡Eso es todo!** El sistema:
- Se desuscribirá del stream de LTC.
- Se suscribirá al stream de APT.
- Creará una nueva instancia de estrategia para APT.
- Empezará a descargar el historial (1500 velas) para APT automáticamente.
- El Dashboard mostrará APT en la lista.

---

## ⚠️ REGLAS INSTITUCIONALES

1. **Formato**: Siempre usar `XXX/USDT` (con barra). El sistema maneja internamente la conversión a `XXXUSDT` para la API de Binance.
2. **Disponibilidad**: Asegurarse de que el par existe en Binance Futures antes de añadirlo, o el bot lanzará un warning y lo ignorará.
3. **Exclusiones**: Si un par tiene problemas de datos, añadirlo a `EXCLUDED_SYMBOLS_GLOBAL` en `config.py` en lugar de borrarlo de la lista, para mantener historial.
