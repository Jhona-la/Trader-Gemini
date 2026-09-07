# 🔬 TERCER INFORME DE ASEGURAMIENTO Y CERTIFICACIÓN PROFUNDA
### Post r1-r3 + n-fixes + e3/e4 + r4.3 + vpin-fix + forense-pnl — con foco en el MOTOR TEMPORAL UNIFICADO

**Fecha:** 2026-09-07 (cierre) · **Método:** 6 auditores paralelos (regresión de 6 commits, motor swing/scalp unificado, genoma/evolución, realismo fill-model, módulos 2/7+1/6, módulos 3/4/8). Solo documentación.
**Se agrega a:** la serie forense (maestro §13-14, aseguramiento F0F3 y R1R3, resultado de certificación).

---

## 🗺️ 0. Topología del grafo — diagnóstico central de esta ronda

El sistema ya opera (post-VPIN-fix), pero el grafo revela su deformidad estructural: **el supuesto "motor dual scalp/swing" es en realidad un motor swing con un motor scalp muerto adjunto**. La mitad temporal del espectro (microestructura) no está desconectada por calibración — está **descableada a nivel de código** (`ScalpEngine::evaluate_microstructure` jamás se invoca). Y la porción operante (swing) gobierna con literales label-bound (2h/72h, 8s/60s, RR 2.0/2.5, trailing swing derivado de genes scalp×1.5). La unificación temporal que el operador exige no es un ajuste: es la fusión de dos islas discretas que nunca fueron un continuo.

---

## 🚦 1. Resumen de resolución acumulada

| Frente | Estado |
|---|---|
| ✅ Certificado en esta ronda | N-08 (triple-barrier con SL y flags de precedencia), N-09 (aggTrades OBI con depth base), N-11 (activaciones fit/predict unificadas ReLU+sigmoid + `transform()` con estadísticos congelados), N-12 (Consejo con tracker adaptativo cableado de verdad), R3.4 (lifecycle_rank + dedup fees con tests), reconciliación periódica programada (60s) con comparación de cantidades, herencia del daemon desde envelope, cache de generación protegiendo hot-swaps, embudo promote→apply en todos los promotores activos, env separation funcionando en bins, R2.1/R2.2 verificados (test formal anti-fuga OBI) |
| ⚠️ Reparado con regresión | R4.3 conformal (C-01: muerto por orden de lectura), VPIN fix (H-01: bucket ratchet que solo crece), E3 (H-03: fallback legacy bypass + cold-start prod huérfano), E4a (M-04: features cross-coin del registry global) |
| 🔴 Crítico nuevo/persistente | Ver matriz §2 |

**Conteo de esta ronda:** ~60 hallazgos (12 críticos, ~18 altos, ~20 medios, ~10 bajos) + 20+ certificaciones.

---

## 📊 2. Matriz de CRÍTICOS de esta ronda

| ID | Hallazgo | Ubicación | Categoría |
|---|---|---|---|
| **T-01** | **El calibrador conformal está MUERTO**: `close_with_fee()` pone `ml_prediction=0.0` ANTES de que ambos sitios de calibración lo lean; el guard `>0.0` impide toda alimentación → `p_value()=1.0` permanente — R4.3 reprodujo la tautología que decía reparar (constante 0.95 → constante 1.0), y el gen `conformal_alpha` sigue anulado | `position.rs:197` vs `lib.rs:834→879, 1110→1158` | Orden de operaciones |
| **T-02** | **El motor scalp canónico es código inalcanzable**: `ScalpEngine::evaluate_microstructure` solo lo llaman un benchmark y tests; `scalp_engines` (30 instancias) se instancian y jamás se evalúan. El "scalp" real es solo composite+tensor con 4 gates label-specific (spread dinámico insuperable con book sintético, composite degenerado por tensor omni 8/54, cutoff tensor 0.65 vs 0.60 del swing, vetos CVD/muro solo-scalp) — explicación completa de scalp=0 intents | `strategy-core/src/scalp.rs:25`; `lib.rs:31,82,178,1325-1445` | Bloqueo de inteligencia / arquitectura muerta |
| **T-03** | **Cosecha del shadow forest: bypass del embudo en producción** — aplica `apply_to_arena` sin promote; con el cache de generación ahora SOBREVIVE a refresh_models (antes se pisaba): el arena diverge del disco indefinidamente, sin linaje, sin rollback; un reinicio lo pierde silenciosamente | `god_engine.rs:1758-1761` | Bypass del gate |
| **T-04** | **No-determinismo — causas raíz identificadas**: (a) `freeze_normalizers` JAMÁS se invoca en las rutas de carga (god-engine-core:110-118, multi_coin_simulator:278) — todo modelo pre-N-11 deserializa descongelado y muta en predict; (b) un TEST reescribe el modelo de producción `models/DarkAlpha_BTCUSDT.json` (y le borra el flag de congelamiento); (c) el campeón de genoma es producto de evolución sin semilla; (d) refresh_models hace hot-swap desde disco MID-BACKTEST (optimización in-test) | `dark-alpha lib.rs:463-465,1063-1076,569-573`; `online_daemon:346`; `lib.rs:403-414` | Determinismo |
| **T-05** | **Geometría de leverage sin tope de equity**: margin 0.85×free × leverage ≤50 → notional hasta ~42× equity por posición y ~85× agregado; `max_pos=50000` es un dólar absoluto (no relativo); las pérdidas flotantes NUNCA reducen `free_cap` (una cuenta bajo el agua sigue abriendo posiciones 0.85×free); sin maintenance margin ni liquidación modelada en la ruta nativa | `lib.rs:1511-1523, 1636` | Solvencia |
| **T-06** | **TP lleno con certeza a precio exacto, trigger por MID y fee MAKER**: un TP limit que se dispara al toque del mid (media spread antes que el bid real) y siempre llena al precio exacto — la fantasía por-trade más grande del fill model | `lib.rs:782-795, 1049-1063` | Fill realism |
| **T-07** | **Funding NO modelado**: `funding_rate=0.0` hardcodeado en ambos sitios de features; swing retiene hasta 72h a 50× sin costo de carry — drag material ausente | `lib.rs:630, 479-484` | Costo de carry |
| **T-08** | **Bug N-04 del espejo INVERTIDO en swing**: scalp-close no cierra el espejo si `horizon != 2`, pero Swing se guarda como **1** → el espejo de un swing vivo SÍ se cierra desde la rama scalp; swing-close excluye `horizon != 1` → cierra el espejo de un scalp vivo y DEJA ABIERTO el espejo de un swing ya cerrado (ghost leak de margen) | `lib.rs:835-841, 1111-1117`; `position.rs:131-135` | Contabilidad de posiciones |
| **T-09** | **Cold-start de producción pierde el linaje acumulado**: con TG_GENOME_ENV=prod y store vacío, el legacy mirror NO se lee (por diseño E3) y el fallback `quantum_champion.json` es cross-env — la primera corrida prod arranca en baseline, el campeón de la era compartida queda huérfano; ADEMÁS `load_or_baseline` (genome.rs) lee el mirror legacy INCONDICIONALMENTE como fallback — bypass directo de la separación | `genome_store.rs:87-111`; `genome.rs:436-439` | Aislamiento / migración |
| **T-10** | **Fitness evolutivo inválido estadísticamente**: walk-forward del daemon entrena sobre `returns_history` GLOBAL (retornos de TODAS las monedas mezclados — el momentum de BTC decide entradas sobre retornos de ETH); y 18 de los 24 genes mutados NO tienen señal en el fitness (deriva neutra arrastrada por ruido en 5 genes) | `online_daemon.rs:163-220, 409-460` | Validez estadística |
| **T-11** | **Retry del OCO reutiliza firma+timestamp+clientOrderId idénticos**: Binance lo rechaza por -4015/recvWindow — el "fix F1.9 de OCO parcial" está muerto por diseño | `executor.rs:1866-1875` | Idempotencia |
| **T-12** | **Diagnósticos contaminados**: `REJECT_COUNTERS` es estático global acumulativo compartido por control + TODOS los mutantes shadow → el reporte "rechazos del día" mezcla N motores y todos los días previos — invalida conclusiones cuantitativas del signal-path diag (las cualitativas —qué gate rechaza— siguen válidas) | `risk-engine lib.rs:53-57`; CEB:430 | Instrumentación |

---

## 🔬 3. MOTOR TEMPORAL UNIFICADO — el inventario completo (foco del operador)

### 3.1 Por qué scalp produce 0 señales (las compuertas exactas, en serie)
1. Motor desconectado (T-02).
2. `spread_ok`: `(atr_pct*0.25).clamp(0.0006,0.0025)` — insuperable con book sintético.
3. Régimen Hurst con 3 sub-ramas que exigen composite sobre tensor omni mayormente vacío (8/54 slots en CEB).
4. Fallback tensor: `net_confidence > 0.65` (vs 0.60 swing) con mayoría de estrategias emitiendo 0.
5. Vetos CVD + muro L2 que SOLO aplican al scalp.
El swing evade 1-5 porque su fuente primaria es `evaluate_trend` directo. **No es que el mercado no tenga microestructura: es que la única ruta hacia ella tiene 5 compuertas label-specific más.**

### 3.2 Inventario de dualidades (F-01..F-21)
| Dualidad | Estado |
|---|---|
| Motores separados scalp/swing (30+30 instancias) | Scalp muerto; swing vivo |
| Consenso tensor dual (11 estrategias Scalp / 3 Swing) | El continuo `TradeHorizon::Continuous` existe SOLO en una rama sin callers |
| **23 genes duplicados por horizonte** (tp/sl/kelly/trail×6/hurst/obi/accel/split) | Mapeables a UN eje temporal continuo |
| Split de capital VIRTUAL: la ruta cuántica pasa 100% del capital a ambos horizontos; todo el aparato bayesiano √n+Robbins-Monro es código muerto en producción | |
| Triple posición + espejo con encoding colisionado (Continuous=1=Swing; `horizon()` jamás devuelve Continuous; state_db sin Continuous; `is_scalp: bool` binario) | K-17 intacto |
| Buckets de métricas triplicados (metrics/scalp/swing) + margen duplicado | |
| Consejo bifurcado por label (0.55/0.42 vs 0.65/0.35; DD 0.95/0.85; slip 25/75) | |
| Online learning dual (buffers 2×30×64), registry con sufijos `_swing`, ledgers "scalp"/"swing", 5 rutas de rollback por label | |
| Gestión temporal label-bound: timeouts 2h/72h, trailing activation 8s/60s, RR 2.0/2.5, trailing swing = genes scalp ×1.5 (los genes `swing_trail_*` EXISTEN y no se usan) | |
| TP/SL de gestión recalculados cada tick IGNORANDO los tp_price/sl_price que el risk-engine derivó y almacenó al abrir — dos verdades por posición | |

### 3.3 Diseño de camino mínimo hacia el continuo (propuesta senior)
**Conservar:** StatefulEngine (agnóstico), trailing.rs con fases (el único bloque genuinamente temporal), el interpolador Continuous del orquestador como ÚNICA ruta de consenso, evaluate_single_intent (genérico salvo `if is_scalp`).
**Fusionar:** 23 genes duplicados → un eje `temporal_scale s∈[0,1]` con `tp(t)=tp_min·(tp_max/tp_min)^s` gobernando TP/SL/trailing/lifetime/cooldown (elimina 22 genes y todos los literales 2h/72h/8s/60s/2.0/2.5); TP/SL de gestión = TP/SL almacenados al abrir.
**Eliminar:** scalp.rs/swing.rs (muerto/label-bound), evaluate_dual_consensus, scalp_position/swing_position → UNA posición (el bug T-08 del espejo desaparece por construcción), buckets duales, enums paralelos (TradingHorizon/TradeHorizon/HorizonIntent), claves con sufijo, smoothed_split (o cablearlo de verdad).
**Coste actual de la estructura dual:** 14 evaluaciones de estrategia + hashing registry por tick ×2 consensos, 3 stacks de 30 motores, triple contabilidad — multiplicado por N mutantes en backtest.

---

## 🛡️ 4. Genoma bt/prod — tabla de las 11 causas + residuales

| Causa | Estado |
|---|---|
| 1 semilla default | ✅ daemon; ⬜ `start_evolution_loop` aún default (posible código muerto) |
| 2 promote→apply | ✅ todos los promotores activos |
| 3 refresh_models clobber | ✅ cache; PERO T-03: la cosecha sin promote ahora sobrevive y diverge |
| 4 harvest sin promote | 🔴 T-03 |
| 5 contaminación bt↔prod | ✅ bins; ⬜ evolver/polars sin TG_GENOME_ENV; 🔴 T-09 cold-start + bypass legacy de genome.rs |
| 6 clamps disjuntos | ✅ fuente única |
| 7 RR cuádruple | ✅ unificado (frágil si los bounds de SL suben: repair clamped puede fallar el gate) |
| 8 baseline inestable | ✅ certificado por test |
| 9 subespacio 10 genes | ⚠️ 24 genes pero 18 sin fitness (T-10) |
| 10 genome_weights divergente | Dormido (ledger sin uso) |
| 11 fitness cross-coin | 🔴 T-10 sin cambiar |
| + herencia features forest E4a | ⚠️ 4/6 features son cross-coin (registry global, last-writer-wins) |

---

## ⚡ 5. Fill-model / realismo financiero (el frente del PnL fantasía)

**Certificado correcto:** fees contabilizados completos sin doble conteo (el spread SÍ se paga a entrada; SL con gap es conservador; zombie/timeout a mid con taker).
**Óptimos que inflan el PnL (en orden de impacto):** (1) no-determinismo T-04 (sin él, nada es medible); (2) TP mid-trigger maker-certain T-06; (3) leverage geometry T-05 + escape hatches micro (min_notional fuerza 39% exposure, Kelly decorativo en bootstrap); (4) slippage no-op (5bps/$1M = cero en todo el rango operado); (5) sin liquidación ni drag de flotante; (6) funding=0 (T-07); (7) compounding geométrico correcto amplificando todo lo anterior ×4201 trades/día.
**Bonus encontrados:** `wf_capital=13.0` hardcodeado en el fitness; macro del snapshot del Consejo con constantes (graph_corr/vpin ya reales, slippage 1.5/3.0 fijos); RecursiveHurst cambiado silenciosamente a R/S sobre retornos (¿correcto? sí — pero invalida todos los umbrales hurst calibrados contra la versión anterior).

---

## 🔒 6. Módulos 1/6/2/7/4/8 — pendientes y nuevos

- **Módulo 1/6**: SegQueue ilimitada (el "FIX #1447" sigue siendo comentario); mmap_bus tearing sin seqlock; lakehouse offset corrupto; forensic Lagged NO re-localizado (posiblemente eliminado — sin certificar); NO_MUTEX latente.
- **Módulo 2/7**: OBI colinealidad AMPLIFICADA (≥15 archivos); API scoped huérfana; ppo/lead_lag/spectral computed-discarded; graph-4d huérfano; `|ml_threshold−0.5|` pierde la dirección del gen; cutoff_floor aún pisable por min_confidence_btc alto.
- **Módulo 4**: T-11 retry OCO; A1 sibling-cancel con reloj local; A2 terminal→terminal overwrite (Rejected pisa Filled); A3 WS replay sin dedup por trade_id; A4 adopción con margen 10x hardcodeado; A5 adopted_{} sin dedup real (inflación por ciclo); A6 dual toggling; M1 rate-limit reactivo; M2 simulator sin leverage; M6 RMW no atómico en margen; M7 drift con .abs() que rompe hedge.
- **Módulo 8**: AUTOEVOLUCIÓN FORZADA + activity bonus persisten; doble increment_tick; tensor 54D divergente CEB vs live; fees 4 fuentes; REJECT_COUNTERS contaminado (T-12); PhaseExecutor/NetworkJitter decorativos.

---

## 🎯 7. Hoja de ruta priorizada (actualización)

1. **Determinismo (T-04)** — llamar `freeze()` en TODAS las rutas de carga; eliminar el test que escribe el modelo; sembrar evolvers; congelar refresh_models durante backtest. Sin esto ningún número futuro es medible.
2. **Conformal vivo (T-01)** — capturar ml_prediction ANTES de close_with_fee.
3. **T-03 + T-09** — cosecha por promote; migración explícita shared→prod/demo + cerrar el bypass de genome.rs.
4. **Fill-model (T-05/T-06/T-07)** — TP por bid/ask con taker; tope notional/equity con flotante; funding y maintenance margin.
5. **T-08 espejo** — mientras llega la unificación (§3.3), corregir la codificación del guard (Scalping=0/Swing=1 real vs los `!=1/!=2` invertidos).
6. **T-10 fitness** — retornos por moneda (o multi-slice) + congelar los 18 genes sin señal.
7. **Unificación temporal (§3.3)** — el proyecto estructural de fondo: eje `temporal_scale`, una posición, un consenso.

*Se agrega sin sustraer contenido a la serie forense. Referencias al árbol en `7172f949` + working tree concurrente.*
