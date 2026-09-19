# 🏛️ INFORME DE ASEGURAMIENTO DECIMOCUARTO — CERTIFICACIÓN FORENSE POST-B3 TOTAL

## Auditoría de Verificación de Cambios: 22 Commits en 2 Días, Enjambre de 5 Agentes de Solo-Lectura, Evidencia Viva en Producción

**Fecha:** 2026-09-17 | **Motor vivo:** v32 (demo testnet) | **Commits auditados:** `68f89157` → `7d7e060c` (22 en 48 horas) | **Método:** 5 agentes forenses paralelos de solo-lectura sobre 307 archivos / 23 crates + forense de logs vivos (demo_v26–v32) + contabilidad de exchange (income API)

---

# 🗺️ PARADIGMA DE GRAFO VIVO Y TOPOLOGÍA DEL SISTEMA

El sistema es un grafo dirigido donde la información fluye: **RAÍZ** (ingestión WS/REST) → **NODOS DE DECISIÓN** (espectro, ensamble ML, señales, consejo, gates de riesgo) → **NODO TERMINAL** (orden al exchange) → **RETORNO** (fills → contabilidad → Kelly → evolución). Esta auditoría certifica el estado de cada arista tras la densidad de cambios B3.1–B3.26. El hallazgo estructural: **el grafo tiene aristas cortadas (inteligencia que no fluye), nodos duplicados (residuos swing/scalp), bucles que se pierden (estado no persistido) y un subgrafo paralelo (backtest) cuya topología difiere del vivo en 15 puntos medidos**.

---

# 🚦 RESUMEN DE ESTADO DE RESOLUCIÓN

| Estado | Puntos | Nota |
|---|---|---|
| ✅ Resueltos (B3.1–B3.26, verificados) | 14 | τ core, friction floors, macro 4D, contabilidad disparos, escalado naked, breaker, maker, divergencia local↔exchange, oráculo resucitado, DXY, cap RR≥2, disciplina roster, SOLUCIONA |
| 🔴 CRÍTICOS nuevos (acción inmediata) | **11** | Ver matriz: congelamiento de precio en crash, cancel del maker ignorado, 44/54 features muertas, τ degenerada → brackets al 65%, drift leverage 10×, NN cross-símbolo en el gate, PnL triple + consejo duplicado, sizing bt≠vivo, genoma clampado, daemon que muta campos autodestruidos, kill-switch por ganancias |
| 🟠 ALTOS | **38** | Ver matriz por módulo |
| 🟡 MEDIOS | **52** | Ver matriz |
| ⚪ BAJOS | **~35** | Ver matriz |
| **TOTAL nuevos** | **~136** | Suma a la matriz maestra histórica de 305+ |

---

# 📊 MATRIZ MAESTRA CONSOLIDADA (extensión de los 305+ puntos históricos)

## CRÍTICOS (11) — defectología degenerativa activa o invariante roto

| ID | Módulo | Archivo:línea | Defecto (resumen ejecutivo; detalle completo en sección del módulo) |
|---|---|---|---|
| C-01 | MOD1/4 | data-pipeline/ws_client.rs:331-350 | **Filtro anti-glitch congela el mercado en crash >20%**: `is_extreme_glitch` eternamente true → todos los ticks rechazados para siempre → motor opera contra precio pre-crash congelado. El filtro diseñado contra corrupción se vuelve congelador en el peor régimen. |
| C-02 | MOD1/4 | data-pipeline/omni_multiplexer.rs:135-209 | **44/54 features del omni sin productor en vivo**: `binance_spot` jamás escrito, `start_feeds` nunca invocado — cross-exchange, liquidaciones, OI, funding spot, CVD, máximos de dolor… constantes. El vector entrena con historia real y sirve con constantes: paridad train/serve rota de raíz, silenciosa. |
| C-03 | MOD1/4 | execution-engine/executor.rs:2104-2147 | **Maker-chase descarta el error del cancel**: DELETE falla → GTX sigue viva en el libro → remnant se mercadoa → la límite llena después → **sobre-exposición hasta 2×qty**. Ruta del 100% de las entradas (force_maker=true). |
| C-04 | MOD1/4 | execution-engine/reconciliation.rs:472-476 | **Drift con leverage 10× hardcodeado**: el fix S-06 corrigió la adopción 30 líneas arriba pero no el drift — `new_margin = notional/10.0` infla `used_margin` en cuentas 20×/50× → falsa escasez → vetos de margen contra capital que existe. |
| C-05 | HOST | god_engine.rs:31-99 + temporal_spectrum.rs:237-248 | **τ degenerada al extremo del espectro (146 años)**: fusión 1/vol pondera escalas lentas → `dominant_tau_ms` = escala 31 → `HorizonCurve.eval` extrapola exponencial sin clamp → **brackets a +65%/−32%**. Verificado en journal vivo: 2/3 entradas con `tau_ms: 4611686018427`. El invariante de protección produce desnudez. |
| C-06 | MOD2/7 | god-engine-core/lib.rs:1700-1721 + calibration.rs:183-187 | **El gate B3.18 lee ml contaminado**: (a) la DarkAlpha NN es UN modelo de BTC votando cross-símbolo en TODOS los ensambles — B3.18b quitó el fallback del forest pero el NN sigue; (b) `spot_bias` ±0.15 absolutos por 1.5bps de spread puede cruzar el umbral sin opinión de modelo; (c) neutralización X-037 (≥0.9999→0.5) tira el ensamble a neutral. La entrada en SOL la puede decidir el modelo de BTC + el spread spot. |
| C-07 | MOD2/7 | god-engine-core/lib.rs:1246-1257, 3282-3287 | **PnL en TRES contadores + consejo duplicado**: cierre escribe metrics+scalp+swing (comentario F-014 lo niega); consejo registra cada cierre dos veces (señales scalp≡swing) → pesos adaptativos aprenden de dataset duplicado. La erradicación swing/scalp es cosmética en contabilidad. |
| C-08 | MOD3/5 | backtest-engine/booktick_replay.rs:588-660 vs risk-engine/lib.rs:417-515 | **La envolvente bayesiana (LCB+shrinkage+bootstrap+veto) existe SOLO en backtest**: el vivo dimensiona con kelly clásico + leverage_matrix. Bifurcador estructural #1 del "funciona en bt, no en prod": los tamaños que la evolución midió no son los que producción ejecuta. |
| C-09 | MOD3/5 | config_dir/genomes/prod/active.json + genome.rs:1195-1198 | **Producción opera un genoma que no es el evaluado**: campeón trae ≥5 genes fuera de banda que el vivo reescribe silenciosamente (tech_threshold 0.05→0.24: opera 4.8× el gen; OBI 0.797→0.60: 25% de la banda evolutiva invisible). Linaje: desciende de `backtest_heritage` pre-D-651 sin gate de desempeño; el único pipeline con puerta OOS no produjo al campeón vigente. |
| C-10 | MOD3/5 | evolution-engine/online_daemon.rs:520-577, 679-691 | **El daemon vivo evoluciona nada contra un fitness prohibido**: muta `scalp_tp_base` que se autodestruye en el roundtrip vector (las curvas mandan, las anclas son vistas) — la geometría TP/SL JAMÁS evoluciona en vivo; usa `wf_pnl×sqrt(trades)×wf_wr` (familia que D-652 erradicó); fricción maker+taker contra el 2×taker del gate; entrena contra macro congelada en literales 2024. |
| C-11 | MOD6/8 | god_engine.rs:2662-2690 + audit-engine/drift_auditor.rs:42-75 | **Kill-switch por drift semánticamente roto**: shadow=0 fijo → drift=−pnl → cualquier trade con |movimiento|>5% ARMA el kill-switch — **incluidas las ganancias grandes**; y pnls de adoptadas con entry roto (≈100%) lo arman por contabilidad podrida; pero el marcado unrealized envenenado (B3.13) NO lo dispara. Sensible a basura, ciego al veneno real. |

## EVIDENCIA VIVA (forense de producción, demo_v26–v32) — 5 hallazgos con runtime proof

| ID | Evidencia | Detalle |
|---|---|---|
| LIVE-01 | **Rama 7 esquivó la disciplina de roster** | ADA (retirado, sin modelo en disco) ejecutó 14 entradas bajo v32 (13 rechazadas por exchange −2019, 1 aterrizó con OCO). Mecanismo: coin_id↔símbolo desincronizado tras rotación del universo + .bin huérfanos resucitados (C-05 relacionado vía loader). |
| LIVE-02 | **Resurrección masiva de modelos retirados** | El loader B3.19e (".bin sin .json hermano se carga") cargó 20 modelos retirados desde binaries huérfanos (AVAX, DOGE, DOT, LINK, LTC…) — verificado en log de carga v32. La disciplina B3.25 socavada por la capa de carga. |
| LIVE-03 | **Anomalía de balance** | $2,341 → $320 entre v29 y v30 sin explicación en income (hipótesis: reset de testnet). El sistema opera un día en régimen micro sin registro del evento. |
| LIVE-04 | **Drag de fees repta con churn no-roster** | 0.8% → −20.9% con SOL+ADA+RENDER (símbolos no-roster) churneando — los resucitados de LIVE-02 operando. |
| LIVE-05 | **Primeros disparos de bracket contabilizados** | DOGE TP +$80.77, AVAX SL −$36.75, XLM SL, SOL SL — diario fluye, clasificador funciona, Kelly alimentado. La cadena B3.7 funciona; pnl_gross=0 honesto en adoptadas (B3.21 mitiga vía diario B3.1). |

## ALTOS (38) — por módulo (detalle completo en las secciones)

**MOD1/4 (9):** FRED pisa a Yahoo en el poller (B3.23 hueco); cancel de pierna hermana ALGO por endpoint legacy muerto; maker leg sin registro (early-exit inefectivo sin UDS); step_size 0.001 literal en emergencia; UDS sin catch-up REST tras reconexión (fills perdidos para siempre); WsExecutor stub con trampa de activación; NTP dual oscilando offset; I/O de archivo síncrono en hilo de WS (journal re-parseado por fill); comisión dedup max() sub-cuenta 43%.

**MOD2/7 (11):** scalp_forest UNIVERSAL fantasma; ml_prob_ewma/var declarados jamás usados; online-learner entrenando con target roto; funding/dex features a cero constante; aislamiento hebbiano roto por nomenclatura (racha de DOGE contamina BTC); consejo delibera sobre 7 números colineales de OBI; umbral causal del consejo neutralizado (0.88 efectivo); cascada de 17 compuertas con tautologías; consenso tensorial cutoff casi-bloqueante; consejo exige OBI no nulo; Hurst gates contradictorios (6 bandas solapadas).

**MOD3/5 (8):** doble fuente de verdad anclas/curvas frío vs hot-swap; curvas derivadas con literales serde; validación con fee de referencia 0.0010 vs vivo 0.00114; daemon fricción maker+taker; promoción viva con 3 trades de evidencia; min_trades_per_day=50 en ventana de minutos (estancamiento → recocido sin evaluación); sizing replay≠vivo (C-08); champion-producer fitness IS diario sin OOS.

**MOD6/8 (5):** maker_flag del replay condición muerta (100% taker); replay ensancha cotizaciones (OFI adulterado); calibración del ensamble 1440×/día vivo vs 1×/día replay; espectro FFT/multifractal CONGELADO en vivo post-arranque (6/10 features espectrales muertas); Brownian bridge del oráculo visita ambos extremos correlacionado con vela previa.

**HOST (5):** dedup 120s traga trades legítimos scalp; PAPER CLOSE puede cerrar posición nueva legítima; envolvente Kelly no persistida (32 reinicios = bootstrap perpetuo); launcher mainnet roto con etiqueta invertida; claves shadow con prioridad sobre reales.

## MEDIOS (52) y BAJOS (~35): inventariados por agente en secciones siguientes.

---

# 🔬 MÓDULO 1: INGESTIÓN, PARSERS, LIBROS L2 Y NORMALIZACIÓN

> 33 hallazgos del agente MOD1/4 (data-ingest, data-pipeline, execution-engine). Los críticos C-01 a C-04 arriba. Destacados adicionales:

**MOD1/4-005 — FRED sobrescribe a Yahoo (B3.23 hueco).** `omni_multiplexer.rs:504-542`: el ciclo escribe los 4 índices Yahoo y DESPUÉS el loop FRED sobrescribe los mismos slots (`sp500`, `nasdaq`, `vix`, `dxy`) con DTWEXBGS (¿otra serie!) y cierre de HOY (sin corte t-1). El comentario "FRED queda fuera del ciclo" miente sobre el código: en una red donde FRED responda, la paridad B3.4b se rompe silenciosamente y la feature dxy cambia de semántica según qué fetch ganó ese minuto.

**MOD1/4-009 — UDS sin catch-up tras reconexión.** `user_data_stream.rs:204-245`: cada reconexión re-crea listenKey SIN re-consultar openOrders/income/replay. Todo ORDER_TRADE_UPDATE del gap se pierde para siempre: los fills que caigan en el gap no se journalean (B3.7/B3.10 agujereados sistemáticamente en tormenta de red — justo cuando más se opera); `cached_positions` no se limpia (equidad falsa hasta el próximo ACCOUNT_UPDATE).

**MOD1/4-012 — I/O síncrono en el hilo del WS privado.** `trade_accounting.rs:132-139` + `user_data_stream.rs:525`: `record_bracket_close`/`record_entry_fill` hacen `std::fs` write DENTRO del handler del WebSocket; `last_journal_entry_px` (B3.21) re-lee y re-parsea el journal COMPLETO por cada disparo de TP/SL — a 10k entradas son 10k parses serde bloqueando el runtime que debe procesar fills en microsegundos.

**MOD1/4-013 — Cadena serial de entrada: 1.4–2.1s con 750ms de sleeps.** `set_leverage` → POST GTX → 400ms poll → DELETE → GET query → POST remnant → **350ms settle** → GET position_risk → 2×POST algoOrder. El `executed_qty` del query se descarta y se re-deriva por position_risk: triple serialización de la misma duda. Ventana naked compuesta ~750ms–2s cubierta solo por el watchdog de 5s.

**MOD1/4-018 — `CONDITIONAL_ORDER_TRIGGER_REJECT` ignorado.** El evento de Binance para "una orden ALGO falló al disparar" (posición desprotegida) no se enruta: `_ => {}`. Tampoco MARGIN_CALL ni ACCOUNT_CONFIG_UPDATE.

**MOD1/4-023 — Cuatro selectores de universo incompatibles.** dynamic_symbols (score pct·log10(vol), umbral 10M), dynamic_selector (1M, fallback hardcode), asset_selector (ln(1+vol)·pct·e^(−pct/15), F-003 "mismos umbrales en todos los entornos"), dynamic_ranker (misma fórmula pero min_vol=0 en testnet — contradiciendo F-003 71 líneas arriba). El universo activo depende de cuál ganó el wiring.

*(+ inventario completo de filtros del executor con evaluación teoría/vibración: 20 ops/s local mezcla órdenes y fetches; 3×429 consecutivos → kill-switch permanente sin mecanismo de reset; read-timeout = umbral de pánico — la calma del mercado castigada como fallo; piso anti-glitch 0.8% descarta los primeros ticks de todo movimiento >0.8% — entrada sistemáticamente tarde a los impulsos que la selección de símbolos elige.)*

---

# 🧠 MÓDULO 2: INFERENCIA IA, MODELOS PREDICTIVOS Y SEÑALES

> 40 hallazgos del agente MOD2/7. Críticos C-06, C-07 arriba. Destacados:

**Bloqueos de inteligencia (la información existe y no fluye):**
- **MOD2/7-001**: `scalp_forest` (modelo "UNIVERSAL") cargado de disco, mantenido en Arc, refrescado cada 1000 ticks — **cero lectores en el workspace**. Si el operador entrena un UNIVERSAL esperando que opere, el motor lo ignora silenciosamente.
- **MOD2/7-003**: el online-learner se sigue entrenando en cada cierre con `realized_ret − ml_at_entry` (fracción−probabilidad, error negativo en 100% de casos — el propio calibration.rs D-693 documenta el defecto) y se evalúa por tick — puro costo, cero voto.
- **MOD2/7-004**: funding_rate y dex_severity alimentados con 0.0 constante → 2 slots del vector 34D/48D son columnas muertas en vivo.
- **MOD2/7-005**: aislamiento hebbiano inoperante por key-mismatch — se escribe `{sym}_hebbian_weight` y se lee `{sym}_perceptron_hebbian_weight` (que nadie escribió) → cae al global: la racha perdedora de DOGE reduce la convicción de BTC.
- **MOD2/7-006**: el consejo de seniors delibera sobre 7 números donde 5 de 6 seniors direccionales son transformadas colineales de {OBI, f(OBI), Hurst} — "deliberación adversarial de 9 seniors" = re-empaquetado de 3 señales que el motor ya gateó ≥6 veces aguas arriba.

**Rigidez (cascada de 17 compuertas):** MOD2/7-011 enumera el embudo completo LONG: ATR mínimo → spread → cooldown ×6 → rama régimen (composite+OBI+z95) → veto Bayesiano → ponderación ML → veto CVD → **re-veto Bayesiano (repetición literal)** → whiplash → firewall racha → escudo macro ×3 → espectral → veto OBI-ruido → escudo neuronal → risk-engine (confianza+EV+notional) → consejo (consenso ≤1 veto supermayoría) → B3.18 (roster+ml). Con contradicciones concretas: capitulación extrema implementada DOS veces con condiciones distintas; Hurst clasificado con 6 bandas solapadas según la rama; el consenso tensorial corta a supermayoría (|net|>0.40 con 15 estrategias heterogéneas ≈ siempre Flat); cooldowns por CUENTAS de tick (no tiempo) con duración wall-clock distinta entre entornos.

**Residuos swing/scalp (erradicación cosmética):** C-07 (PnL triple + comentario que lo niega), MOD2/7-018 (consejo registra cada cierre DOS veces), MOD2/7-019 (contadores `scalp_loss_streak` gobiernan el riesgo del motor unificado), MOD2/7-020 (enum discreto vivo en lifetimes/sensibilidades del consejo), MOD2/7-021 (timestamps y márgenes gemelos), MOD2/7-022 (aliases legacy como puertas de re-bifurcación), MOD2/7-023 (genes binarios aún evolucionando).

**Arbitrariedades:** ml_threshold 0.5698=0.55×1.036 sin derivación (y GA puede subirlo a 0.95 cegando el motor sin feedback); confianzas fabricadas por 3 fórmulas distintas (0.50+0.40·|score|, |macd|·H·50 tanh, 0.5+|ml−0.5|); escalera de trailing ×fee con 8 multiplicadores y dos be_trigger con clamps distintos (12% de diferencia); be_activation=tp·0.55 CONTRADICE el cap B3.24 (SL≤TP/2 → margen útil antes del primer lock de solo 45-50% del TP); spot_bias ±0.15/1.5bps; >60 grados de libertad no evolucionables en señales.

**Lógica:** `a_t` con doble escritor de unidades incompatibles (adimensional vs $ según fase del contador %64) alimentando 3 features; EMA fast/slow con períodos distintos según el feed (20/200 en tick vs 12/26 en kline — micro_trend tiene significado según la mezcla del feed); X-037 castiga convicción legítima y protege al degenerado; normalizadores per-coin mutan durante inferencia "congelada" (tras 500 ticks, el NN de altcoins opera con calibración que nunca vio el entrenamiento); ml_prob_adaptive corregido solo lado bajista y no leído por el gate (dos verdades del mismo modelo en el mismo tick).

---

# 📈 MÓDULO 3: ESTRATEGIA MULTIACTIVO, RÉGIMEN Y HORIZONTES

*(cubierto por MOD2/7 y MOD3/5 conjuntamente)*

- Espectro temporal: HOST-001/C-05 (τ degenerada), MOD6/8-004 (FFT/multifractal congelados en vivo — 3 de 10 features espectrales muertas post-arranque; el forest swing recibe 6 features nulas en bt y congeladas-hace-días en vivo: **ninguno de los dos ve el espectro vivo que la directriz F8 proclama**).
- Hurst: gates contradictorios (MOD2/7-014), hurst_duration factor 2^((H−0.5)/0.02) clamp [0.5,2] arbitrario, hurst_swing_threshold muerto.
- Régimen: RegimeDetector con literales fallback (0.6/0.02/correlación<0.2→Chaotic) decidiendo el kill-switch de largos en Crash.
- Universo: 4 selectores incompatibles (MOD1/4-023); `max_active_coins` = 2 a $13 pero el oráculo evalúa 1 símbolo y el loop vivo 30; specs del símbolo TRIO inconsistente (oráculo tick 0.1/lev 50 vs replay 0.01/20 vs vivo real — MOD6/8-031).

---

# ⚡ MÓDULO 4: EJECUCIÓN HFT, PROTOCOLO DE RED Y CONECTIVIDAD

> Críticos C-03, C-04 arriba. Adicionales:

**MOD1/4-006 — Cancel de pierna hermana por endpoint muerto.** El "motor OCO" del streamer cancela el SL hermano vía `/fapi/v1/order` legacy — pero las piernas son ALGO que viven en `/fapi/v1/algoOrder`: el DELETE devuelve "order does not exist" sin tocar el trigger → **el SL queda armado tras un TP lleno** hasta 60s (watchdog). El endpoint correcto (`cancel_algo_order`) está a 30 líneas.

**MOD1/4-016 — Router de pánico → ruta más lenta.** Latencia >500ms o spread >0.5% rutea a maker-chase (400ms de ventana + cancel + query + market con la red ya degradada) — la decisión premia la máxima latencia cuando la latencia es el problema.

**MOD1/4-017 — 429/418 sin clasificar en 3 de 5 primitivas HTTP.** Solo el POST de órdenes alimenta la máquina de estados F1.8; un 418 (IP baneada) en cancel/leverage/GET jamás dispara cooldown — y `get_payload_account` reintenta CUALQUIER error tras 200ms incluyendo 429 con Retry-After: 60 (cada reintento fallido suma al contador 3×429 → kill-switch).

**MOD1/4-015 — Comisión dedup max() sub-cuenta 43%.** REST y WS con cobertura disjunta de fills: max(0.04, 0.03)=0.04 cuando la real es 0.07. La base con la que D-632 alimenta métricas de fricción — el mismo dominio que B3.11/B3.6 miden con exactitud.

**MOD1/4-020 — Adopción sintética re-creada por ciclo.** `apply_to_registry` sintetiza ack FILLED con comisión imputada por cada ciclo de 60s con update_time distinto: ~10 entradas adopted coexistiendo por posición — multi-contabilización de comisiones de entrada por factor N.

*(+ firma correcta en ruta principal verificada; reduceOnly/positionSide consistentes en todas las rutas de orden auditadas; hosts literales en endpoints algo divergiendo de get_base_url; overflow de buffer sin chequeo en 3 rutas.)*

---

# 🛡️ MÓDULO 5: GESTIÓN DE RIESGO, KELLY Y GENOMAS EVOLUTIVOS

> 41 hallazgos del agente MOD3/5. Críticos C-08, C-09, C-10 arriba. **La respuesta completa a la pregunta del operador** (genoma bt≠prod) son las CUATRO causas raíz verificadas:

1. **Sizing estructuralmente distinto** (C-08): envolvente bayesiana solo en backtest; el vivo fuerza notional mínimo subiendo leverage que el replay veta.
2. **Genes clampados/muertos a la entrada**: 5+ genes del campeón reescritos silenciosamente (tech_threshold ×4.8, OBI 0.797→0.60 con 25% de banda invisible, margin_cushion, zombie_timeout, dynamic_ema_trend 5.5× bajo el piso); ~32 genes sin consumidor; 12+ conectados a `evaluate_order`/`calculate_dynamic_allocation`/`check_drawdown_limit` — **circuitos apagados** (la resurrección D-650 cableó genes a funciones muertas); `maker_only_capital_threshold` congelado a literal 50.0 por la mutación misma.
3. **Linaje contaminado**: prod/demo descienden de herencia automática pre-D-651; el walkforward (único con puerta OOS) no produjo al campeón vigente.
4. **Promotores divergentes**: daemon vivo destruye sus mutaciones TP/SL en el roundtrip, usa fitness prohibido, fricción pre-D-645, macro 2024 congelada; champion-producer con fitness IS diario sin OOS y "blindaje cuántico" que clampea anclas ya re-derivadas (código muerto disfrazado de seguridad); latency_penalty piseado a 25ms anula el gen.

**Gates del risk-engine — inventario y veredicto:** los NÚCLEOS derivan correctamente (min_viable_sl=f/0.5, min_rr_for=(1−w)/w+f/(w·sl), fee medido de genes, colchón del gen). Pero: ev_fee_multiplier del campeón (1.0) clamped a 1.25 en micro — el gene inerte, el sistema exige 25% de holgura que el genoma no autorizó; literales residuales (0.66/0.62, 0.20, 0.035, 5.0+1.5, $1.20/$2.60); correlación cluster=2 fijo para CUALQUIER capital sin mirar correlación real (2 anti-correlacionadas se vetan igual que 2 altcoins); B3.24 NO se aplica en el camino de la orden (compute_tp_sl garantiza rr≥rr_required≈1.5-1.8, el gestor re-geometriza a RR≥2 al tick siguiente — dos geometrías coherentes por separado, incoherentes entre sí).

**Kelly:** piso 0.50 en expansión inflando sizing de edges marginales; rampa de exploración sin salida estadística (se auto-refuerza con clamp_min del gen); envolvente con trampa f=0 solo salvable por diseño; clamps axiomáticos en restricción de ruina (nunca prescribe <0.001 aunque el posterior grite ruina); capital_regime C¹ sólido pero micro_weight calculado con min_notional GLOBAL no del símbolo evaluado (BTC a $13 en micro pleno, otro símbolo con min 5 en estándar — régimen incoherente dentro del MISMO trade).

---

# 🔒 MÓDULO 6: ESTADO ATÓMICO, MEMORIA MMAP, TELEMETRÍA Y SO

> 40 hallazgos del agente MOD6/8. Crítico C-11 arriba. Destacados:

- **MOD6/8-010 — `used_margin`: el átomo enfermo.** RMW load→compute→store NO atómico desde ≥3 hilos (cierre core, rollback async, reconciliación, replay) — lost-update estructural. Y el else del cierre hace `store(0.0)` cuando current<margin: **ante drift contable de UNA moneda, borra el margen de TODAS** → free_margin inflado → sobre-exposición autorizada.
- **MOD6/8-012 — mmap_bus sin época**: tras restart, el Shadow Forest re-lee hasta 1M frames de sesiones anteriores como nuevos — el lazo predicción→realidad de la autoevolución contaminado con historia de otro genoma.
- **MOD6/8-014 — 64MB de telemetría fantasma locked en RAM**: zero_copy_bus sin lectores, flusher que solo avanza un tail, SystemTime::now() por emit bajo lema "3-5ns", carrera de datos documentada.
- **MOD6/8-019 — El sanitizador B3.13 tiene hueco exacto**: el writer clampa a ±2×capital y ALMACENA; el reader excluye solo si >bound (estricto) → el valor clamped (==bound) PASA y se agrega: `marking_anomalies: 0` con $4.4K fantasma en el agregado. Y el bound 2× es ilegítimamente bajo para leverage>2 (las pérdidas grandes LEGÍTIMAS también se maquillan); el sanitizer es ciego a pnl_realized/gross envenenados.
- **MOD6/8-023 — profile_node en el hot path**: 2×rdtsc + CAS a ArrayQueue en CADA evento del motor; conversión asume 3.0GHz exactos (25% de error típico); el aggregator solo lo consume OTRO binario.
- **MOD6/8-025 — Prioridades invertidas**: THREAD_PRIORITY_TIME_CRITICAL en el hilo de decisión puede STARVAR los hilos tokio que ejecutan el maker-chase — el hilo que DECIDE posterga al hilo que EJECUTA; HIGH_PRIORITY_CLASS para TODO el proceso (dashboard+SQLite+Telegram incluidos); "pinning al core 1" es un hint blando con comentario falso de garantía L3.
- **MOD6/8-027 — JobObject memory limit**: symbols×128MB+2048MB — superarlo = FALLO DE ALOCACIÓN = abort del motor vivo (no kill-switch elegante); si AssignProcessToJobObject falla, no hay límite y solo un eprintln lo dice.
- **MOD6/8-026 — ObservabilityPlane**: polling 1kHz leyendo PMU/eBPF que en Windows son MOCKS; el pinning al último core es cfg(linux) — en Windows el hilo flota libre despertando 1000×/s para alimentar un detector EWMA con constantes: telemetría fantasma Y fuente de jitter.

---

# ⚛️ MÓDULO 7: SEÑALES CUÁNTICAS, ORQUESTACIÓN Y CONFLUENCIA

*(integrado en Módulo 2; síntesis)*

La "confluencia cuántica" es en la práctica: 15 estrategias donde 2 votan con datos de OTRA moneda (MOD2/7-009: turbo_scalper y renyi_tsallis no sobreescriben evaluate_for_coin → usan el registry global — el voto de la moneda N se computa con datos de la última que escribió); el conformal calcula ACI con rigor O(200) por tick para pesar 1/15 del consenso (garantía diluida a irrelevancia); el consenso corta a supermayoría (≈siempre Flat); el consejo exige OBI no nulo (7ª vez que OBI gatea); y el umbral causal 0.75 hardcodeado es decorativo (el efectivo es 0.88 por neutralización de horizonte). **El resultado neto: la "orquestación de confluencia" es un embudo adicional de OBI sobre señales ya-gateadas, con costo de 10 evaluaciones + 2 RwLock por candidato.**

---

# 🧪 MÓDULO 8: BACKTESTING, AUDITORÍA INTERNA Y GOBERNANZA

> Divergencias bt↔vivo residuales post-réplica D-442 — **tabla D1-D15 completa** (ver informe del agente MOD6/8; resumen de sesgos):

| Divergencia | Sesgo del bt |
|---|---|
| D1/D2 maker (chase real vs fill inmediato; flag maker condición muerta → 100% taker) | **Pesimista en fees y features de flujo** |
| D3 funding ausente | **Optimista** (carry gratis para zombies/swing) |
| D4 SL en gaps (no cortable sin tick) | Optimista-pesimista según el gap |
| D5 sin OCO de exchange | Sobreestima duración de zombies |
| D7 espectro: FFT nulo en bt vs congelado en vivo | 6/44 inputs del forest OOD en AMBOS, distintos entre sí — misranking de genomas |
| D8 calibración ensamble 1440×/día vs 1×/día | Infraestima adaptación del gate B3.18 |
| D9 envolvente vacía por corrida vs memoria de sesión | Pesimista estructural de arranque (30 trades lev 1) |
| D11 macro del oráculo congelada 2024 | **Invalida la cobertura del oráculo como certificación** |
| D14 sin rechazos/AMBIGUOUS/adopciones | Optimista (nunca sufre la cola de fallos del vivo) |

**Además:** MOD6/8-029 (contabilidad interna del replay: stats pre/post-warmup inconsistentes, `initial+net_pnl ≠ final_capital` sin flag); MOD6/8-030 (value_at Err(0) → lookahead suave al inicio); HOST-008 (dedup por timestamp exacto en --daily destruye aggTrades mismo-ms — **las tablas OOS de septiembre corrieron sobre cintas adelgazadas en los bursts**); MOD6/8-032 (doble noción de warmup, Hurst inválido al inicio); MOD6/8-036 (Sharpe por trade no anualizado, media/varianza sobre muestras distintas); MOD6/8-005/006/007 (Brownian bridge correlacionado con vela previa, OFI de 3 estados, RNG con variables acopladas y retícula de 1000 valores).

**Forense:** MOD6/8-016 (forensic_auditor ACTIVO pero: panic-letal si data/ no escribible, SQLite bloqueante en async, Lagged→continue pierde eventos para siempre, DB sin retención creciendo sin cota); MOD6/8-017 (scalp_pnl persiste unrealized — columna mal etiquetada para siempre); MOD6/8-018 (media API de audit-telemetry muerta, canal bounded(1M) pre-reservando ~32-80MB, proyección institucional con 50 trades/día hardcodeado inalcanzable); MOD6/8-038 (Telegram reporta capital del arena sin saneo — el veneno por Telegram como verdad; sin alertas de kill-switch aunque send_alert existe).

---

# 🎯 HOJA DE RUTA SISTÉMICA DE REHABILITACIÓN 1-A-1 (priorizada por peligro real)

## FASE 0 — Detención de daño activo (hacer ANTES de operar más)
1. **C-05/τ degenerada**: clamp de τ a las anclas [30s, 12h] en genome_protection_prices + fix de la fusión espectral (paridad 1/vol degenera) — los brackets al 65% están vivos AHORA.
2. **C-01/anti-glitch**: el `is_extreme_glitch` debe tener vía de escape (aceptar tras N ticks anómalos consecutivos O ref-rescalado) — un crash real congela el motor hoy.
3. **C-03/cancel del maker**: inspeccionar el error del DELETE (distinguir "ya llenó" de "fallo de red") + cancel_algo_order para piernas ALGO (MOD1/4-006).
4. **LIVE-02/.bin huérfanos**: borrar los 20 binaries retirados (una línea de shell) — los tóxicos resucitados están churneando fees AHORA (LIVE-04).
5. **C-11/kill-switch por drift**: desconectar el armado por |pnl|>5% hasta rewiring contra divergencia real (o el próximo win grande mata el motor).

## FASE 1 — Paridad del grafo (cerrar el bt/prod)
6. C-08: llevar la envolvente bayesiana al vivo (o el kelly clásico al replay — decidir cuál es la verdad).
7. C-09: re-clampear o re-entrenar el campeón dentro de banda; regenerar linaje por walkforward OOS.
8. C-10: alinear el daemon (fitness D-652, fricción D-645, mutar curvas no anclas, macro real).
9. D7/D8: alimentar FFT/multifractal desde process_tick en vivo + klines sintéticos en replay; unificar el reloj de calibración.
10. MOD6/8-031: UNA fuente de specs (la del exchange) para oráculo+replay+vivo.

## FASE 2 — Erradicación real del binario
11. C-07: escribir PnL SOLO en metrics; consejo registra UNA vez; eliminar aliases y contadores gemelos; borrar el enum PositionHorizon o reducirlo a Continuous.
12. Los 12+ genes de circuitos apagados: conectar o eliminar (con test de cobertura del oráculo como trinquete al alza).

## FASE 3 — Desbloqueo de inteligencia
13. Los 5 bloqueos de MOD2/7 (UNIVERSAL, ewma/var, online-learner, features a cero, hebbian key).
14. C-02: productores para las 44 features muertas o contraer el vector a lo vivo (un contrato honesto).
15. El embudo de 17 compuertas: deduplicar tautologías, unificar bandas Hurst, dar poder de decisión real al conformal o retirarlo del hot path.

## FASE 4 — Salud estructural
16. used_margin a ownership único o CAS; NTP dual; UDS catch-up; persistencia de envolvente/espectro entre reinicios; saneador B3.13 (pase-exacto + realized); telemetría fantasma (64MB locked, buses muertos, profiler con GHz falso); prioridades de SO; launcher mainnet y claves shadow.

---

**Certificación:** este informe documenta ~136 hallazgos nuevos sobre la matriz histórica de 305+, de los cuales 11 son críticos con defectología degenerativa activa o invariantes rotos, verificados por 5 agentes independientes de solo-lectura y corroborados con evidencia de producción (logs v26–v32, journal de posiciones, contabilidad de exchange). Ningún código fue modificado durante esta certificación. Las 14 reparaciones B3.1–B3.26 se verifican como implementadas y parcialmente efectivas (arrastre de fees −108.7%→0.8-20.9%, contabilidad de disparos viva, oráculo resucitado); los defectos aquí documentados son los que emergieron de, con o pese a esos cambios.
