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
| `python tests/certification_of_perfection.py` | Certificación Omega | Validar latencia interna **< 20μs** |
| `python tests/benchmark_total_latency.py` | Benchmark HFT | Medir P99 Tick-to-Action |
| `python tests/test_extreme_load.py` | Stress Test | Simular ráfagas de 10,000 msg/s |
| `python utils/health_check.py` | Diagnostic | Verificar JIT, Red y Deriva NTP |

---

## 📚 GLOSARIO DE NANO-LATENCIA HFT
- **Zero-Copy**: Metodología donde los datos no se copian entre CPU/Memoria, usando `MemoryViews` para evitar el Garbage Collector.
- **Micro-Selectivity**: Capacidad de descartar señales mediocres en **< 5μs**.
- **JIT Warming**: Proceso de pre-compilación de LLVM necesario para evitar picos de latencia en el primer trade.
- **Kernel Fusion**: Consolidación de lógica en una sola unidad compilada (`Numba`) para maximizar hits de Caché L1/L2.

---
**Certificado**: Omega Grade Architecture | **Fecha**: 2026-04-02
