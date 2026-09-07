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

---

## 📌 ADENDA: DIAGNÓSTICO DEL SIGNAL PATH (misma fecha, cierre de la pregunta "¿dónde mueren las señales?")

Instrumentación por etapa (contadores de rechazo por compuerta + conteo scalp/swing + vetos del Consejo). Cadena causal completa de los 0 trades:

1. **Las señales EXISTEN**: ~1.300 intenciones/día no-flat, todas SWING (scalp: 0 — el stack scalp no dispara con este tensor), confianza máx 0.78.
2. **El risk-engine NO rechaza** (tras dos fixes de esta sesión, ver abajo): "sin rechazos".
3. **El Consejo de Seniors veta el 100%** (`swing_vetoes = 1323/1323`): **Senior Causal**, porque `vpin_risk = 1.000 > causal_veto_threshold = 0.60`. VPIN = 1.0 constante es un valor DEGENERADO (100% toxicidad de flujo sostenida es imposible en datos reales): es un artefacto de los 4 sub-ticks sintéticos por minuto con qty concentrada en un solo lado del book — el CVPIN ve buckets de volumen sin mezcla y satura.

### Fixes aplicados en esta sesión (encontrados por la instrumentación)
- **Bug matemático en `candidate_leverage`** (risk-engine): calculaba el leverage para alcanzar el min-notional sobre el capital TOTAL en vez del MARGEN del trade → leverage < 1 en micro-cuenta → rechazo garantizado (~500-640/día).
- **Error de punto flotante en la frontera del min-notional**: `lev × (min_notional/lev) = 5.0999… < 5.1` rechazaba la orden en el borde EXACTO — épsilon relativo de 5 bps en el bump del margen.

### Próximo paso único y acotado
Reparar el VPIN degenerado con ticks sintéticos: agregar los sub-ticks en buckets volumétricos reales (como hace `historical.rs` para aggTrades) antes de alimentar el CVPIN, o marcar el VPIN como no-confiable cuando los buckets no tienen mezcla bid/ask. Con eso, el Consejo deja de vetar sobre un fantasma y la certificación puede volver a correrse — esa será la PRIMERA medición del sistema operando.
