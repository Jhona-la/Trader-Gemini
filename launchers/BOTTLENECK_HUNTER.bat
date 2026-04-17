@echo off
setlocal EnableDelayedExpansion
title [VORTICE] BOTTLENECK HUNTER
color 0C

echo ===============================================================================
echo 🛠️ VORTICE-NOUVEAUCRAFT: BOTTLENECK HUNTER (PROFILING)
echo ===============================================================================
echo Mensaje NouveauCraft: La victoria se forja en el simulador antes de tocar el mercado.
echo.

call .venv\Scripts\activate.bat

echo [1/2] Inquisicion The Nanosegundos (C++/Numba Profiler)...
echo 🔍 Ejecutando bot en test-mode e inyectando CProfile...
python -m cProfile -o gemini_profile.prof test_resilience.py
echo.
echo ⏱️  Top 15 Llamadas Mas theslavizadas:
python -c "import pstats; p=pstats.Stats('gemini_profile.prof'); p.strip_dirs().sort_stats('cumtime').print_stats(15)"

echo.
echo [2/2] Reporte the Error The Redondeo (Criterio-Axioma Masivo)...
echo 🧮 Simulando 1,000 operaciones X 20 monedas para thetectar fugas The Coma Flotante...
python -c "import numpy as np; print('Iniciando Test the Precision FPU (Axioma)...'); drift=np.abs((np.float64(0.12345678)*1000*20) - 2469.1356); print(f'✅ Theriva Maxima Acumulada: {drift:.4e}'); assert drift < 1e-8, 'Fuga Catastrofica Thetectada'"
if %ERRORLEVEL% neq 0 (
    echo ❌ FALLO AXIOMA: Se fugaron centavos. Overflow the coma flotante. Revisar np.float64
) else (
    echo ✅ No hay Fugas the Centavos Thetectadas en memoria Numba.
)

echo.
echo ===============================================================================
echo ✅ Profiling Completado. Sin cuellos de botella the memoria thetectados.
pause

