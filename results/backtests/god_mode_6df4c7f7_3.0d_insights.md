# 📊 Reporte de Insights Estructurados: Backtest 6df4c7f7
**Generado:** 2026-06-17T05:29:32.916468+00:00

## 1. 🎯 Rendimiento General
- **Capital:** $13.00 ➔ $12.78
- **Retorno:** -1.72%
- **Win Rate:** 40.0% (10W / 15L)
- **Drawdown Máximo:** 2.05%
- **Ratio de Sharpe:** -19.30

## 2. 🚰 Análisis de Fuga de Alpha (Alpha Leak)
Este análisis detalla cómo se redujo el PnL bruto ideal debido a fricciones reales de mercado.
- **Alpha Bruto (Ideal):** $-0.1968
- **Pérdida por Slippage:** -$0.0000
- **Comisiones (Fees):** -$0.0000
- **Exits Prematuros (Costos de Oportunidad):** -$0.0000
- **Alpha Neto (Real):** $-0.1968

### 💡 Diagnóstico del Profesor
> ✅ **Salud Óptima:** Las fricciones de mercado están bajo control y el modelo de retención de valor es saludable.

## 3. 🛡️ Confiabilidad del Sistema y Auditoría
- **Trades Rechazados por el Oráculo/BFT:** 1054
- **Kill Switch Activado:** No ✅
- **Métricas Exportadas a JSON:** God Mode ha registrado cada tic. El comportamiento es determinístico.

## 4. 🧠 Recomendación Evolutiva (EAI)
Basado en el sistema adaptativo e integral, enfocar recursos en optimizar los factores responsables de mayor fuga (Fees vs Slippage) o recalibrar los umbrales del RiskManager si el Drawdown excedió expectativas.