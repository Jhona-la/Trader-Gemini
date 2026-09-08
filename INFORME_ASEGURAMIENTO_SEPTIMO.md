# 🔬 SÉPTIMO INFORME DE ASEGURAMIENTO Y CERTIFICACIÓN PROFUNDA
### Tres auditores senior en paralelo: Quant Researcher, Trading Infrastructure Engineer, ML/Infra Architect

**Fecha:** 2026-09-08 · **Método:** 3 agentes especializados en roles senior cubriendo los 8 módulos del Grafo Vivo. Solo documentación, sin cambios de código.
**Se agrega a:** la serie forense completa (maestro §13-18, aseguramientos 1º-6º, certificaciones).

---

## 🗺️ 0. Paradigma de Grafo Vivo — diagnóstico central

**EL HALLAZGO MÁS CRÍTICO DE TODA LA SERIE**: el certificador definitivo R2.2 (aggTrades reales) tiene el **timestamp parseado de la columna EQUIVOCADA** — `cols[4]` es `last_trade_id`, no `transact_time` (que es `cols[5]`). El archivo de 33.6M "ticks reales" que descargamos tiene **trade-IDs en el campo de tiempo**: duraciones, interpolación temporal y horizonte del motor continuo quedan corruptos. Toda certificación futura sobre ese archivo es inválida hasta que se regenere.

El grafo además revela: la interpolación por `temporal_scale` fue **revertida de logarítmica a lineal** sin documentación por la sesión concurrente (y además es prácticamente código muerto para trades reales); el sizing kelly está **neutralizado por el min-notional del exchange** en cuentas micro; y el ensamble de 14 estrategias se **evalúa 3 veces por tick** con resultados idénticos (las tres funciones son aliases).

---

## 🚦 1. Resumen de resolución

| Frente | Estado |
|---|---|
| ✅ Certificado | U-C zombie-fix (leverage lee metrics), kelly_bootstrap_cold, fee contable honesto (D-179/180), freeze() DarkAlpha completo, N-11 ReLU/predict vigente, R2.2 prioridad REAL, R-04 trades mínimos, ws_executor stub honesto, WS client sólido (pinning IP, gate bayesiano), NO_MUTEX eliminado, SQLite WAL |
| ⚠️ Reparado con regresión | Interpolación U-A/U-B revertida a lerp (C-1 quant); D-144 doble fuente temporal_scale |
| 🔴 Críticos | 8 (ver matriz §2) |
| 🔴 Altos | 12 |
| ⬜ Medios/Bajos | ~20 |

---

## 📊 2. Matriz de CRÍTICOS

| ID | Área | Hallazgo | Ubicación |
|---|---|---|---|
| **S-01** | R2.2/Datos | **aggTrades: timestamp = cols[4] (last_trade_id) en vez de cols[5] (transact_time)** — el certificador definitivo escribe trade-IDs como tiempo. Sort disfraza el bug (monótono). Regenerar BTCUSDT_ticks_REAL.bin obligatorio | `binance_vision_sync.rs:214-216` |
| **S-02** | M3/M5 | **temporal_scale interpolación REVERTIDA sin documentación**: la sesión concurrente cambió `span = 10^(2s)` → lerp lineal `scalp*(1-s) + swing*s` en AMBOS lados (evaluador y gestión) — consistente como lerp pero destruye la no-linealidad del diseño original | `risk-engine/lib.rs:671-698`; `god-engine-core/lib.rs:588-615` |
| **S-03** | M3/M5 | **La interpolación es CÓDIGO MUERTO para trades reales**: toda posición abierta recibe TP/SL del risk-engine (no-zero garantizado) → el bloque interpolado solo vive en posiciones restauradas/corruptas. El gen temporal_scale NO tiene fenotipo pleno | `god-engine-core/lib.rs:584-635` vs `1499-1511` |
| **S-04** | M5/Riesgo | **Kelly neutralizado por min_notional en micro-cuenta**: con capital $13 y kelly $0.13-0.65, el margen SIEMPRE queda por debajo del mínimo de Binance ($5.1) → el sizing kelly jamás gobierna; gobierna el piso del exchange. El PnL +0.14% está 100% determinado por exposición al piso notional, no por edge estadístico | `risk-engine/lib.rs:614-622` |
| **S-05** | M4/Ejecución | **OCO retry NO re-firma**: replay del buffer firmado exacto (mismo timestamp, firma, clientOrderId) → -4116 duplicate o -1021 recvWindow. El fix se aplicó a flatten pero NO a las piernas OCO | `executor.rs:1879-1890` |
| **S-06** | M4/Ejecución | **Margen hardcoded 10x en adopción de reconciliación**: `notional/10.0` ignora el leverage real de la cuenta — infla `used_margin` 2-5x en cuentas 20x/50x → falsa escasez → vetos de apertura | `reconciliation.rs:237, 263` |
| **S-07** | M8/Backtest | **Doble increment_tick persistente**: CEB incrementa + process_tick_dual incrementa = 2× por tick (principal y shadows). Gates por tick corren a 2× frecuencia relativa al tiempo simulado | `continuous_evolution_backtest.rs:385, 432` |
| **S-08** | M7/Ensamble | **14 estrategias evaluadas 3× por tick con resultados IDÉNTICOS**: `tensor_cont`, `tensor_scalp`, `tensor_swing` son aliases de la misma función — 42 evaluaciones para 1 señal; los tres gates (0.65/0.60/0.65) comparan la MISMA señal; solo el umbral 0.60 discrimina | `god-engine-core/lib.rs:1108-1110` |

---

## 📈 3. Módulo 3+5 — Núcleo de Decisión (Quant Researcher Senior)

### Altos
- **A-1**: Discontinuidad `is_scalp < 0.5` en Continuous → saltos discretos en leverage y stops en la frontera del gen — gradiente evolutivo en escalón.
- **A-2**: Doble fuente de verdad `temporal_scale` (D-144: constructor lo deriva de `capital_split_scalp`; el gen lo sobrescribe en `apply_to_arena`) — valor efectivo depende del orden de llamadas.
- **A-3**: Síntesis de intenciones = selector argmax, NO fusión bayesiana: el desacuerdo no atenúa tamaño, la convergencia no amplifica, y la convergencia de dos visiones produce el horizonte MÁS CORTO (Scalp forzado).
- **A-4**: Stops vs lifetime incoherentes: timeouts lineales en s (30m→4h) independientes de sl/ATR. Un trade de 16bps con 30m de vida es una eternidad de ruido.
- **A-5**: `evaluate_order` calcula edges sobre `coin.scalp/swing.kelly_fraction` = 0 (nunca escritos) → split R1.3 funcionalmente muerto.
- **ALTO adicional**: `global_max_drawdown` NO se aplica al motor unificado — el cortafuegos de drawdown solo existe en `evaluate_order` (sin callers de producción) y en el kill-switch `cap <= 0` (post-extinción).

### Arbitrariedades por impacto en edge (ranking del auditor)
1. Kelly bypass por min_notional (S-04)
2. Timeouts lifetime literales (A-4)
3. Selector argmax (A-3) — señal descartada por peso de uso
4. Pesos 0.40/0.35/0.25 de la fusión bayesiana del composite
5. Umbral is_scalp 0.5 (A-1) + doble fuente temporal_scale (A-2)
6. Literales micro-cuenta ≤$15/$30 (bimodal, incluye 3.5% fee-impact tolerado)
7. Constantes de leverage_matrix congeladas

### Top-3 mejoras de edge medible (propuestas del auditor)
1. **Kelly contractualmente vinculante**: derivar leverage del min-notional en vez de sobrescribir el margen.
2. **Lifetime derivado del stop**: `timeout = k_time · (sl/ATR)` con `k_time` gen — acopla automáticamente stops cortos→vidas cortas.
3. **Fusión bayesiana real de intenciones**: magnitud atenuada por desacuerdo, horizonte continuo `s_eff = mezcla(s_gen, acuerdo)` — edge estadístico gratis ya calculado y descartado.

---

## ⚡ 4. Módulos 4+8 — Ejecución y Backtesting (Trading Infrastructure Engineer)

### Altos
- **H1**: Rate-limit header-only reactivo — sin contabilidad en vuelo; ráfaga concurrente cruza el límite con checks en verde.
- **H2**: WS replay sin dedup por `trade_id` — `trade_id` viene en el evento crudo (`"t":1`) pero se descarta en el parse; reconexión re-emite fills → infla `ws_commission`.
- **H3**: Activity bonus (+0.015/trade, cap 6%) sigue premiando churn; la rama else-if adopta mutantes con PnL NEGATIVO si solo superan al control ("el menos malo").
- **H4**: REJECT_COUNTERS global acumulativa multi-motor — 250+ shadows escriben a los mismos contadores; el reporte diario es inservible cuantitativamente.
- **H5**: Archivo REAL truncado aceptado silenciosamente en el límite de registro; pérdida de los primeros 8 bytes = interpreta como legacy con datos desplazados.

### Ranking de gaps de realismo del fill-model (¿qué exagera el backtest vs live?)
1. **Fills al precio del trade ± 0.5 bps con liquidez implícita infinita** — sin Kyle impact en el walk-forward (solo en vectorized). El gap dominante.
2. **Latencia uniforme 25ms y cero rejects parciales** — sin jitter de cola, sin -2010, sin carrera de latencia real en TP/SL.
3. **Doble tick + tensor semi-sintético** — funding/VIX/DXY hardcodeados en el walk-forward.
4. **OBI como dirección del último agresor** — la señal más ponderada consume microestructura más "limpia" que un book real.
5. Fees planos sin BNB-discount (único gap conservador, ~10%).

---

## 🔬 5. Módulos 1+2+6+7 — IA/Infra (ML/Infra Architect)

### Altos
- **A-2 (infra)**: forensic_auditor muere permanente ante `Lagged` — fix de V7 NO llegó a main. Broadcast de capacidad 100 + OmniUpdate por tick = burst mata al forense.
- **A-3 (infra)**: `MmapTelemetryBus` tiene CERO productores — 64MB ring con NT-store SIMD, sfence, sin un solo writer. Sus 2 lectores (online_daemon, online_learning) leen vacío → **feedback loop de aprendizaje roto silenciosamente**.
- **A-4 (infra)**: = S-08 (ensamble 3×).

### Medios clave
- **M-1**: SegQueue ilimitada — el fix fue renombrar la variable a `_safe_capacity` (silenciar el warning, no arreglar).
- **M-2**: Lakehouse 1GB mmap sin flush, sin header, wrap sin generación; write-only sin lector.
- **M-3**: PPO consume hawkes VIEJO (`1+|obi|*2`) mientras el registry registra el nuevo (`|a_t|/ATR`); cointegration = momentum del líder renombrado.
- **M-4**: Fan-out de aliases del registry: 6-8 duplicados semánticos por tick (vpin×4, hurst×2, atr×2, obi×2, ml_prob×2).
- **M-5**: 30+ rutas relativas `"data/..."` — CWD-dependency.

### Inventario de cómputo descartado
| Cómputo | Costo |
|---|---|
| Ensamble 14×2 redundantes | 28 de 42 evaluaciones/tick idénticas |
| MmapTelemetryBus completo | 100% descartado (0 productores) |
| SegQueue binary/stats | Sin productores ni consumidores |
| Lakehouse 1GB | Write-only, sin lector |
| feature-engine muerto (~960 líneas) | quantum_tensor_store, tensor_ring, simd_nn, hawkes, kalman, spectral |
| profile_node! | Solo 5 sitios en todo el repo — latencia sin medir |

---

## 🎯 6. Hoja de ruta priorizada

1. **S-01 (URGENTE — minutos)**: `cols[4]` → `cols[5]` en binance_vision_sync.rs + regenerar BTCUSDT_ticks_REAL.bin. Sin esto, toda certificación R2.2 es inválida.
2. **S-08 (minutos)**: des-duplicar las 3 llamadas de consenso — una sola evaluación, tres referencias Copy.
3. **S-04 (horas)**: kelly contractualmente vinculante (leverage del min-notional, no margen sobrescrito).
4. **S-05/S-06 (horas)**: OCO retry re-firma + margen desde leverage real.
5. **S-07 (minutos)**: eliminar el increment_tick manual del CEB.
6. **S-02/S-03 (días)**: restaurar interpolación log + hacerla viva (o documentar el lerp como decisión y eliminar el código muerto).
7. **A-2 infra / A-3 infra**: Lagged fix + MmapTelemetryBus productor (o eliminar todo el pipeline muerto).
8. Los 3 del Quant: lifetime derivado del stop, fusión bayesiana real, drawdown gate en el path cuántico.

*Se agrega sin sustraer contenido. Referencias a main @ cb6dd182 + working tree 36 archivos sin commit.*
