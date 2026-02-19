# 📜 TRADER GEMINI: MANUAL DE OPERACIONES INSTITUCIONAL (SOP)
**Protocolo**: Sincro-Omega | **Nivel de Acceso**: Admin/Quant

Este manual define el **Standard Operating Procedure (SOP)** para el despliegue del "Organismo Supremo" en entornos de producción HFT.

---

## 🏗️ FASE 0: PREPARACIÓN DEL HARDWARE
Para garantizar latencias de microsegundos, el sistema requiere:
1.  **Aislamiento de Cores**: El bot intenta anclarse a cores de alto rendimiento automáticamente.
2.  **Sincronización NTP**: El error de tiempo debe ser < 5ms.
3.  **Power Plan**: Esquema de Energía "Alto Rendimiento" en Windows.

---

## 🚀 FASE 1: DESPEGUE INSTITUCIONAL (GOD MODE)
El despegue debe realizarse exclusivamente a través de los lanzadores optimizados que habilitan banderas de CPU de alta prioridad.

1.  **Lanzamiento Principal**:
    ```powershell
    .\LAUNCH_GOD_MODE.bat
    ```
    *Este comando ejecuta el motor con prioridad `High`, deshabilita asserts de Python (`-O`) y activa el orbe de auditoría `God-Mode`.*

2.  **Lanzamiento de Futuros (Rápido)**:
    ```powershell
    .\START_FUTURES.bat
    ```

---

## 📊 FASE 2: MONITOREO DE SISTEMAS (COCKPIT)
El sistema HFT no debe operarse "a ciegas". Mantén siempre visibles estas tres consolas:

1.  **Terminal de Ejecución**: Muestra el flujo de señales y fills.
2.  **Dashboard de Métricas**:
    ```bash
    streamlit run dashboard/app.py
    ```
    *Verifica el Sharpe Ratio en vivo y la Utilización de la Cola de Eventos.*
3.  **Oráculo de Inferencia**:
    ```bash
    python check_oracle.py
    ```
    *Visualiza las predicciones de la Trinidad (Genético + RL + OL) antes de que lleguen al exchange.*

---

## 🛠️ COMANDOS DE AUDITORÍA Y BENCHMARK
Herramientas para garantizar la perfección operativa antes de escalar el capital.

| Comando | Función | Objetivo |
| :--- | :--- | :--- |
| `python tests/certification_of_perfection.py` | Certificación Omega | Validar latencia < 500μs |
| `python tests/test_extreme_load.py` | Stress Test | Simular flash-crash y ráfaga |
| `python utils/health_check.py` | Diagnostic | Verificar API, Tiempo y Red |

---

## 📚 GLOSARIO DE NANO-LATENCIA HFT
- **Zero-Copy**: Metodología donde los datos no se copian entre CPU/Memoria, sino que se pasan referencias (`Structured Arrays`) para evitar el recolector de basura (GC).
- **Jitter**: Variación en el tiempo de procesamiento. Un jitter alto (ms) rompe la estrategia de scalping.
- **Kernel Fusion**: Consolidación de múltiples funciones lógicas en una sola unidad compilada por LLVM (`Numba`) para maximizar la localidad de cache L1.
- **Trinidad Omega**: El enjambre de 3 IAs (Genética, Refuerzo y Online) que gobierna cada símbolo.

---
**Certificado**: Omega Grade Architecture | **Fecha**: 2026-02-10
