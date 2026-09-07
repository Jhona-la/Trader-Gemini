# 🎯 PLAN MAESTRO DE REPARACIÓN, EVOLUCIÓN Y MEJORA SENIOR
### Trader Gemini — Sistema Universal Continuo Temporal, Autoevolutivo y Omnisciente

**Fecha:** 2026-09-07 · **Base:** INFORME_ASEGURAMIENTO_F0F3.md (~180 hallazgos; 17 críticos K-01..K-17)
**Doctrina:** toda constante debe tener derivación (teórica, estadística o genómica); toda inteligencia calculada debe consumirse; backtest, demo y producción deben ejecutar la MISMA física; ningún fallo es silencioso.

---

## 0. Principios rectores (constitución del plan)

1. **Una sola verdad por concepto**: una fuente para bounds del genoma, una para capital, una para horizonte, una para el modo scalp/swing/continuo.
2. **Gate primero, efecto después**: ninguna mutación toca el arena sin sanción del almacén versionado; ningún estado se publica antes que el dato.
3. **Paridad de física**: si producción paga fee/latencia/slippage/reduceOnly, el backtest los simula con el mismo modelo. Si backtest no lo simula, la métrica no certifica nada.
4. **Fail-loud**: eliminar `let _ =` en rutas de dinero; todo error de ejecución/persistencia tiene telemetría, contador y consecuencia.
5. **Evolutividad real**: cada umbral que gobierne comportamiento vive en el genoma o se deriva de estadística observada — nunca un literal del que no se conozca la derivación.
6. **Cambio atómico y verificable**: cada ítem del plan se entrega con test que falla antes y pasa después, y commit propio.

---

## FASE R — REPARACIÓN ESTRUCTURAL (bloqueante, ~1–2 semanas de trabajo enfocado)

Objetivo: que el sistema que existe hoy haga lo que dice que hace. Sin esto, cualquier mejora evolutiva optimiza ruido.

### R1. Desbloqueo evolutivo (corregir nuestras propias regresiones)
| # | Trabajo | Ref | Criterio de aceptación |
|---|---|---|---|
| R1.1 | Fuente única de bounds: macro/función que genera clamps de `from_vector` Y `get_lower/upper_bounds` Y el gate de `validate()` | K-01 | Test extremo-a-extremo: `mutate_cmaes → to_vector → from_vector → promote` debe tener éxito ≥99% sobre 1000 muestras aleatorias |
| R1.2 | Invariante RR unificada (hoy 1.5×/1.8×/2.2×/3.5× cuádruple): una constante `MIN_RR` derivada de fees reales (RR_break-even = (1+fee_rt)/(1−fee_rt)) usada por mutate, validate y reparación | A1 auditoría | Mutación, validación y ejecución acuerdan en el mismo RR mínimo; test de coherencia |
| R1.3 | Split bayesiano recalibrado: prior Beta real `Beta(α=genome_split·k, β=(1−genome_split)·k)` con k derivado del tamaño muestral objetivo; split con suavizado exponencial y comprometido por época (no por tick); picos de drawdown medidos bajo el split de su época | C2 auditoría | Test: n→∞ no satura al clamp; cambio de split ≤Δ por época; hard-stops estables |
| R1.4 | Kelly: separar la rama de exploración por criterio (PF≤1 usa rampa; WR<0.05 con PF>1 usa Kelly legítimo); anclar `0.25` al shrinkage n/(n+k) del EdgePosterior; `clamp(min,max)` con guard `min>max` | M1/M2 auditoría | Tests de frontera PF∈{0.4,0.5,1.0,1.01}×WR∈{0.01,0.06,0.5} |
| R1.5 | `ws_executor`: decidir honestamente — o se implementa firma HMAC-SHA256 con orden canónico + canal acotado sin secreto en el mensaje, o se degrada a stub explícito `is_connected()=false` con log único | C3 auditoría | Sin secreto en structs de mensaje; sin ruta que prometa WS sin firma |
| R1.6 | Router: re-derivar `expected_volatility` como volatilidad real (EWMA de retornos del feature engine) y reparar el gate degenerado; eliminar el fallback 0.85/0.015 reparando el hot-swap (R3.2) | C6 auditoría | Test: gate de vol dispara solo con volatilidad de precio > umbral |

### R2. Verdad de datos (matar el lookahead y el garbage-in)
| # | Trabajo | Ref | Criterio |
|---|---|---|---|
| R2.1 | Reconstruir los 4 generadores de ticks: OBI neutral o derivado de la vela PREVIA (información disponible en t=0 del sub-tick); trayectoria intra-bar sin coreografía por dirección (path aleatorio consistente con O/H/L/C, no open→low→high→close condicionado a is_bullish); features omni con rezago de vela | K-02, C-1..C-4 mod 3/8 | Test de leakage: correlación OBI(t) vs retorno futuro(t) ≈ 0 en datos generados |
| R2.2 | Descargar aggTrades reales como fuente primaria de backtest HFT (ya existe `binance_vision_sync`); los sintéticos quedan solo para smoke-tests etiquetados `SYNTHETIC` en el propio bin (header/magic) | A-10 mod 1 | Backtest de certificación corre 100% sobre aggTrades reales |
| R2.3 | Indicador World Bank correcto para M2 + macro sintético eliminado o marcado con columna `synthetic=true` que el feature engine rechaza | K-14 | Ninguna feature macro puede ingerir filas sintéticas sin flag |
| R2.4 | Formatos binarios con header magic+versión (mmap_bus, telemetry_mmap, flight record, BinTick); rechazo ruidoso ante mismatch | M-4 mod 1 | Test: archivo de otra versión es rechazado con error, no leído como basura |

### R3. Integridad de ejecución (lo que toca dinero)
| # | Trabajo | Ref | Criterio |
|---|---|---|---|
| R3.1 | `reduceOnly=true` en `execute_reduce_only_market` y ambas piernas del OCO; motor de cancelación de pierna hermana suscrito a ORDER_TRADE_UPDATE | K-03, K-04 | Test de integración testnet: TP lleno → SL cancelado en <1s; cierre nunca puede abrir posición |
| R3.2 | Hot-swap mainnet: re-inyectar arena, OrderRegistry y NTP al nuevo executor antes de `exec.store()`; bloquear la transición si no se puede | K-06 | Test: tras swap, `executor.arena` y registry son los vivos; router lee genes, no fallback |
| R3.3 | `flatten_all_positions` fail-safe: intentar one-way Y hedge ante fallo de lectura de modo; retry con backoff en cuenta insolvente; nunca asumir | K-16 | Test con mock que falla el modo → ambas variantes intentadas |
| R3.4 | Guard monótono de status en el registry; fee contable REST-vs-WS deduplicado por orderId+fillId | H3/H4 | Test: ack tardío NEW no revierte FILLED; fee total = real |
| R3.5 | Reconciliación periódica (cada N minutos) programada en el lazo de producción: diff de cantidades, no solo existencia; resolución ambigua (query+cancel por clientOrderId) en piernas OCO; dedup de adoptados; `prune_terminated` en el loop | H2/H5/H6 | Fuzzer de divergencias sintéticas: toda divergencia se cierra ≤1 ciclo |
| R3.6 | Cierre con protocolo: el close REST fallido reintenta y, si persiste, re-abre el estado local desde el exchange (nunca queda arena flat con posición real); rollback del PnL contable | H7 | Test de fallo inyectado en close → estado converge al exchange |
| R3.7 | Rate-limit con contabilidad local en vuelo (peso acumulado antes de la respuesta) + reset atómico; conectar o eliminar QuantumMultiplexer | H10 | Test de ráfaga: nunca se excede el 80% del presupuesto real |

### R4. Conexión de inteligencias (des-bloquear lo calculado y no consumido)
| # | Trabajo | Ref | Criterio |
|---|---|---|---|
| R4.1 | Cablear `ConsejoDeliberacion` en `process_tick_dual` como gate final pre-orden: los 4 vetos (Causal/Riesgo/Ejecución/Teleonomía) operan en producción; `SeniorPerformanceTracker` alimenta los pesos con ventana deslizante | K-07 | Test: payload con do_calculus_risk alto → orden vetada en producción |
| R4.2 | Reemplazar los `evaluate()` hardcoded de estrategias por las APIs parametrizadas ya existentes (turbo_scalper, conformal, micro_scalp, coaxial) — una sola lógica por concepto | A4/A5 mod 2 | grep: cero umbrales dobles para el mismo filtro |
| R4.3 | Conformal real: set de calibración deslizable de residuos de predicción (el motor ya emite predicción vs realidad); p-valor calculado, no constante 0.95 — o eliminación honesta del filtro | K-08 | Cobertura empírica del intervalo ≈ 1−α medida en test |
| R4.4 | Des-colinear el ensamble: mapear cada módulo a una fuente primaria distinta (OBI, CVD de trades, funding, spread dinámico, momentum multi-escala); `ensemble_boost = 1/√(1+ρ̄(n−1))` con ρ̄ medido online; los módulos que no puedan diversificarse se fusionan o eliminan | C3 mod 2 | Matriz de correlación de outputs de estrategias: |ρ| media < 0.5 |
| R4.5 | Reconectar el `ValidatedOrder` completo en producción: sizing/leverage/TP/SL/maker_only del risk-engine son la verdad; eliminar el recalculo con literales; `kelly_fraction.clamp(0.05,1.0)` → clamp genómico sin piso forzado | K-05 | Test: cambiar el gen de sizing cambia el tamaño de la orden en vivo-path |
| R4.6 | TP/SL por horizonte en `evaluate_single_intent` (bases swing para swing) | H9 | Test de invariante RR por horizonte |
| R4.7 | DarkAlpha: una geometría (54D) en entrenamiento E inferencia; `fit`/`predict` con la misma activación; normalizadores congelados en inferencia (solo actualizan en fase de entrenamiento explícita); limpiar features constantes del tensor | C4/C5/A8/A9 mod 2 | Test de determinismo: mismo input → misma predicción, siempre |
| R4.8 | Auditoría viva: eliminar decorados (`forensics` fuera de cfg(test) como gates reales en CI, PhaseRunner con gates accionables, cablear DriftAuditor/AuditorInterno o borrarlos); un solo DriftAuditor | A6-A8 mod 3/8 | Ningún archivo de "auditoría" sin efecto en runtime o en CI |

---

## FASE E — EVOLUCIÓN REAL (el genoma que el operador exige)

Objetivo: cerrar las 11 causas de la divergencia backtest↔producción (§6 del informe). Orden por dependencia:

### E1. Telemetría viva para el Shadow Forest
- Implementar el escritor del bus mmap en el engine (frames subsystem 12 / frame 30 con predicción vs realidad) con protocolo seqlock corregido (commit del índice DESPUÉS del payload) — corrige K-11 de paso.
- El forest entrena con features reales (obi, spread_bps, atr_pct, hurst vivos), no constantes.
- **Criterio:** el daemon reporta "observaciones reales ingeridas" > 0 en demo; correlación feature→label ≠ 0.

### E2. Herencia correcta
- Eliminar `champion_path` como fuente: el daemon SIEMPRE parte del envelope activo (`GenomeEnvelope::load_active().genome`).
- Darwin: reordenar a gate→apply (idéntico al fix del daemon).
- CMA-ES: semilla desde el envelope; purgar el híbrido PSO (CMA puro o documentar el híbrido con su teoría).

### E3. Gobernanza de promoción unificada
- Un único trait `GenomePromoter` con: lock de archivo (inter-proceso), promote atómico (rename sin remove previo), generación monotónica bajo lock, gate de validación (R1.1/R1.2), y reloj de conservadurismo (cooldown de promoción por linaje para evitar oscilación promote/rollback).
- `refresh_models` no pisa ciegamente: aplica el sobre activo solo si su generación > la última aplicada (cache de generación aplicada).
- Separación de entornos: `config_dir/genomes/{env}/` con env ∈ {backtest, demo, prod}; promoción cross-entorno SOLO explícita y firmada (comando `promote-to-prod` con evidencia adjunta).
- **Criterio:** test de concurrencia (4 promotores simultáneos) → generaciones únicas, sin sobrescritura de historia.

### E4. Espacio de mutación unificado
- Un solo operador de mutación (nichos + CMA) compartido por backtest y daemon: el daemon muta el MISMO espacio de 139 genes, con presupuesto adaptativo (CPU disponible).
- Fitness del daemon: replay de ticks reales en shadow engines (ya existen en CEB) sobre la última ventana, con fees completos — no el modelo clampado actual.
- Eliminar la tarea fantasma de 6h (`cargo run --bin evolution`) o convertirla en promotor explícito con evidencia.

### E5. Certificación estadística de promociones
- Sustituir "AUTOEVOLUCIÓN FORZADA" y el activity_bonus por: promoción solo si el candidato supera al control en ventana OOS con significancia (t-stat ≥ 2 o test de permutación) y ≥ N trades mínimos; empate → conservador (gana el incumbente).
- Watchdog de rollback con muestras DES-autocorrelacionadas (agregar por trade cerrado, no por delta 500ms) y cooldown de re-promoción del linaje revertido.

### E6. Epigenética y consejo senior evolutivo (mejora estructural solicitada)
- Los umbrales del consejo (Hurst, slippage, drawdown) pasan a genes o se derivan de percentiles empíricos; los pesos de los seniors evolucionan vía `SeniorPerformanceTracker` (calibración metacognitiva real: peso_i ∝ Brier score inverso del senior i).
- Exosoma: k del prior del split, cooldowns y tamaños de ventana como genes reguladores (epi-marca) que el propio sistema muta a otra tasa que los genes estructurales — mutación en dos capas con teoría (tasa mayor para reguladores, menor para estructura).

---

## FASE M — MEJORA CUANT AVANZADA (edge real, no narrativa)

1. **Limpieza científica**: cada módulo "cuántico/físico" (oscilador, solitón, shockwave, resonancia) se somete a protocolo: (a) formular la hipótesis matemática real, (b) validar que el código la implementa, (c) medir AUC/IC contra el target en datos reales post-R2.2; los que fallen se eliminan. Menos módulos, más verdad.
2. **Calibración de confianza global**: mapear score→probabilidad con calibración isotónica/Platt online por régimen; la confianza reportada debe igualar la frecuencia empírica de acierto (gráfico de confiabilidad como métrica de CI).
3. **Features que faltan**: CVD real de aggTrades, funding rate como feature viva (ya se consulta), micro-price/imbalance ponderado, cross-exchange basis real (los slots existen pero constantes).
4. **Meta-labeling**: el modelo secundario decide tamaño/no-tomar sobre las señales primarias (estructura estándar de triples-barrier de López de Prado) — reemplaza los gates ad-hoc.
5. **Riesgo de cola**: sustituir tolerancias fijas (0.75 DD micro, 0.95 scalp) por CVaR dinámico estimado de la distribución de retornos observada, con límite de ruina derivado (SURVIVAL_FLOOR ya existe como axioma — completar TRADE_HORIZON desde trades/día reales).

---

## FASE C — SISTEMA UNIVERSAL CONTINUO (migración completa)

1. Codificación de horizonte 3-bucket sin colisión (`Continuous=2`); `horizon()` fiel; round-trip preservado.
2. Eliminar el espejo `position`+`scalp_position`: UNA posición por coin con horizonte como atributo; migrar lectores (telemetría, backtest flotante, validador de paridad, ledger).
3. Cerrar y reportar el ciclo de vida continuo por su propio bucket (no contaminar métricas scalp); `process_event` devuelve intención única + horizonte.
4. Propagar `Continuous` a: state_db (schema v2 con migración), consejo_seniors, online_learning, trajectory_auditor, estrategias (gradual: primero el gate, luego consolidación).
5. Lifetime continuo ya interpolado por confianza (orchestrator) — conectarlo al trailing/timeout real vía el horizonte leído (hoy colapsa a swing por R4/M2).

---

## SECUENCIA Y DEPENDENCIAS

```
R1 (desbloqueo) ──► R2 (datos) ──► R4 (conexión) ──► E (evolución) ──► M (cuant) ──► C (continuo)
        └──────────► R3 (ejecución) ──────────────────┘
```
- R1 y R3 son paralelizables entre sí.
- E1–E3 dependen de R1 (gate unificado) y R2 (datos sin lookahead: de lo contrario la evolución optimiza el sesgo).
- M y C dependen de E (sin evolución real no hay con qué certificar mejoras).
- **Regla de certificación:** cada fase cierra con (a) suite en verde, (b) backtest sobre aggTrades reales post-R2.2, (c) demo ≥48h con telemetría de los contadores nuevos en cero.

## MÉTRICAS DE ÉXITO DEL PROGRAMA

| Métrica | Hoy | Objetivo |
|---|---|---|
| Promociones genómicas en demo/48h | ~0 (gate inalcanzable) | ≥3 con significancia OOS |
| Correlación media entre estrategias del ensamble | ~0.9 (OBI ×7) | <0.5 |
| Confiabilidad de confianza (predicha vs empírica) | descarriada (saturación) | ±5% |
| Divergencia arena/exchange detectada y cerrada | no medida | 100% en ≤1 ciclo de reconciliación |
| Órdenes de cierre sin reduceOnly | todas | 0 |
| Backtests certificables (sin lookahead) | 0 | 4/4 generadores |
| Literales sin derivación en rutas de dinero | 24 censados | 0 |

---

*El plan se ejecuta ítem por ítem con commit y test por ítem. Ningún ítem se declara done sin su criterio de aceptación. La auditoría se repite al cierre de cada fase (mismo método multi-agente de raíz a cima) — auditar y auditar hasta que funcione.*

---

## ⚠️ REGISTRO DE EJECUCIÓN R3 (2026-09-07)

- **R3.1**: `reduceOnly=true` en one-way de `execute_reduce_only_market` + **piernas OCO corregidas** (positionSide condicional por modo: hedge lo omitiría-rechaza reduceOnly, one-way rechaza positionSide → -4061; antes se enviaba incondicionalmente). Motor de cancelación de pierna hermana `{base}_TP`↔`{base}_SL` sobre ORDER_TRADE_UPDATE: implementado.
- **R3.2 y R3.3**: implementados (hot-swap con `new_with_shared` preservando registry+arena; flatten con failover dinámico -4061).
- **R3.4 — BLOQUEADO POR EDICIÓN CONCURRENTE**: `order_registry.rs` está siendo reescrito en bucle por la sesión paralela (el contenido cambia cada minuto; 4 intentos de parche abortaron por colisión). **Parche listo para aplicar cuando el archivo se estabilice:**
  1. Campos `ack_commission`/`ws_commission` en `TrackedOrder` (+defaults en los 3 constructores); `total_commission = max(ack, ws)` en ambas rutas (elimina doble contabilización REST-max + WS-suma).
  2. `OrderStatus::lifecycle_rank()` (Unknown=0, New=1, PartiallyFilled=2, terminales=3) y guard monótono en `apply_ack`/`apply_trade_update`: jamás regresar de estado terminal.
- **R3.7**: pendiente (contabilidad local en vuelo del rate-limit).
- **Acción requerida del operador**: coordinar las sesiones — dos editores sobre `execution-engine` sin exclusión mutua es hoy el mayor riesgo operativo del repo (ya produjo flickering de archivos a medio escribir).
