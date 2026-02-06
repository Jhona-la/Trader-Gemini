# 📜 GUÍA DE COMANDOS - TRADER GEMINI

Esta guía define el **orden de ejecución** y la **función** de cada comando del sistema para garantizar una operación institucional segura.

---

## 🚀 FLUJO DE EJECUCIÓN (PASO A PASO)

### **PASO 1: Validación Pre-Vuelo (Laboratorio)**
Antes de encender el bot real, debes validar que la estrategia es robusta en el mercado actual para los símbolos elegidos.
- **Comando**: `.venv\Scripts\python.exe tools/walk_forward_tester.py`
- **Función (QUÉ)**: Simula ciclos de entrenamiento y trading real en el pasado reciente.
- **Resultado**: Si el Sharpe Ratio es > 1.5, el símbolo es apto para producción.

### **PASO 2: Prueba de Estrés Estadístico (Resiliencia)**
Una vez validada la robustez, probamos si el capital es suficiente para soportar rachas de mala suerte.
- **Comando**: `.venv\Scripts\python.exe tools/monte_carlo_sim.py`
- **Función (QUÉ)**: Ejecuta 5,000 "universos paralelos" reordenando los trades.
- **Resultado**: Nos da el **Risk of Ruin**. Si es < 1%, el capital de $15 es seguro.

### **PASO 3: Monitoreo en Paralelo (Vigilancia)**
Mientras el bot opera (o incluso antes), debes tener estas terminales abiertas para ver qué está pensando la IA.
- **Terminal A (Dashboard)**: `.\DASHBOARD_FUTURES.bat`
  - **Función**: Interfaz visual (Streamlit) para ver balance, equity y trades activos.
- **Terminal B (Oracle)**: `.venv\Scripts\python.exe check_oracle.py`
  - **Función**: Muestra en tiempo real las probabilidades y el "árbol de decisión" de la IA para cada moneda.

### **PASO 4: Ejecución Principal (El Motor)**
Una vez que el laboratorio dio "Verde", el estrés es bajo y el monitoreo está activo, enciende el bot.
- **Comando**: `.\START_FUTURES.bat` (o `.venv\Scripts\python.exe main.py --mode futures`)
- **Función**: Ejecución de órdenes reales en Binance.

---

## 📊 RESUMEN DE COMANDOS ÚTILES

| Comando | Función | Cuándo Ejecutar | Paralelo |
| :--- | :--- | :--- | :--- |
| `tools/walk_forward_tester.py` | Auditoría de robustez (WFV) | Antes del bot | No |
| `tools/monte_carlo_sim.py` | Prueba de Supervivencia (Monte Carlo) | Después del WFV | No |
| `check_oracle.py` | Visión cerebral de la IA | Siempre | **SÍ** |
| `DASHBOARD_FUTURES.bat` | Monitoreo visual (Web) | Siempre | **SÍ** |
| `main.py` | Trading real | Después de validar | **SÍ** |
| `health_check.py` | Diagnóstico de latencia y API | Si el bot se siente lento | No |

---

## 👨‍🏫 MODO PROFESOR: ¿Por qué este orden?
- **QUÉ**: Una jerarquía de ejecución segregada.
- **POR QUÉ**: Separamos el **Laboratorio** (tester) de la **Vigilancia** (oracle) y la **Operación** (main). Esto evita que un error de trading detenga tu capacidad de ver qué está pasando.
- **PARA QUÉ**: Para maximizar el Uptime y reducir el riesgo de "volar a ciegas".
- **CUÁNDO**: Sigue este órden cada vez que reinicies el sistema tras una actualización.
