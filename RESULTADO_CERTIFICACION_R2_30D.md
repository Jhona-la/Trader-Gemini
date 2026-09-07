# 🏁 RESULTADO DEL BACKTEST DE CERTIFICACIÓN POST-R2 (30 días)
### Primera medición honesta del sistema — 2026-09-07

**Comando:** `continuous_evolution_backtest 30` (release, TG_GENOME_ENV=backtest)
**Datos:** `data/BTCUSDT_ticks.bin` — 1.048.320 ticks (4 sub-ticks/min × ~182 días), formato legacy advertido por R2.4.
**Aislamiento verificado:** la promoción final fue a `config_dir/genomes/backtest/active.json` (generación 1) — el genoma de demo/producción quedó INTACTO. La separación de entornos funciona.

---

## Resultado

| Métrica | Valor |
|---|---|
| Capital inicial | $13.00 |
| Capital final | $13.0000 |
| PnL global | $0.0000 (0.00%) |
| **Trades (30 días)** | **0** |
| Trades de los 250 mutantes | 0 (toda la población) |
| Días con actividad | 0/30 |

## Interpretación (la conclusión que importa)

**El "edge" que mostraban todos los backtests anteriores era el lookahead.** Con el OBI sintético que codificaba la dirección de la vela futura (R2.1), el sistema "predijo" el mercado durante meses y las métricas ganadoras eran una medición del sesgo, no de estrategia. Retirado el sesgo, la pila de señales **no dispara ni una vez en 172.800 ticks reales**.

Esto explica de un golpe los misterios históricos del operador:
- "Los backtests largos ganan lo mismo que 1-2 días": el PnL era la fuga, no la estrategia.
- "El genoma funciona en backtest pero no impacta en producción": el backtest optimizaba su acceso a la fuga; en producción (OBI real) no había nada que optimizar.
- La cuenta estancada bajo $40: sin edge real, fees y drawdowns consumen; los picos de backtest nunca existieron.

## Matiz obligatorio (honestidad estadística)

0 trades en 30 días **no prueba ausencia de edge — prueba que las compuertas nunca se abren**. El stack de señales está siendo bloqueado antes de producir la primera intención (candidatos: `cutoff_floor` derivado de `min_confidence_btc`, `tech_threshold`, la convicción post-recalibración sin el inflado de volatilidad, o features del tensor mayormente vacías en este harness — el CEB solo llena 8 de 54 slots). La distinción es crítica: si es un gate mal calibrado, hay señal muriendo en el pipeline; si es ausencia de señal, hay que construirla. **El diagnóstico de DÓNDE mueren las señales (instrumentación por etapa: estrategia→voto→convicción→cutoff→intención→orden) es el siguiente paso obligatorio.**

## Estado de las reparaciones (verificación de infraestructura)

- ✅ SIM_DAYS obligatorio: 30 días corridos completos (antes se truncaba a 3).
- ✅ Sin congelamiento de Kelly: el capital quedó plano por falta de trades, no por el viejo bug (habría mostrado trades el día 1 y plano después — aquí 0 desde el minuto 0).
- ✅ Aislamiento de genoma por entorno: demo/prod intactos.
- ✅ Warning legacy del formato BinTick: emitido correctamente.
- ✅ Estabilidad del genoma base: protegida (población sin mejora).

## Regla de oro que nace de esta medición

**Ningún cambio futuro al stack de señales se acepta sin este backtest de certificación antes/después.** Este resultado ($13 → $13, 0 trades) es el denominador honesto contra el que se mide todo lo que venga.

*Se agrega a la serie forense sin sustraer contenido.*
