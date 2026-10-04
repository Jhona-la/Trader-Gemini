# Mapa de cimientos y hoja de ruta: Trader Gemini

> **Nota al versionar (2026-10-03, ciclo 8).** Foto de `40bd881b`: `main`
> avanzó después (ver `.agents/MEMORIA.md`). Los números ADR-0008…0020 de
> este documento eran PROVISIONALES y no son los del índice `docs/adr/`:
> GLM tomó 0008–0010 para otras decisiones. Correspondencia: la «identidad
> de símbolo» provisional 0008 es el **ADR-0011** (aceptado, CL-37/38); la
> «autoridad del risk-engine» provisional 0014 es el **ADR-0013** en su
> parte mínima (el host ya no envía más apalancamiento que el validado; aún
> puede bajarlo, y el rechazo en vez de ajustar sigue abierto). El
> **ADR-0012** (sólo el núcleo de ejecución sigue al almacén de genomas,
> CL-40) no estaba en esta hoja. El resto sigue sin número y tomará el
> siguiente libre del índice cuando se escriba.

> Consejo de seniors de la sesión Claude (cloud), 2026-10-01. Seis áreas, una por prioridad del operador. Cada área la mapeó un senior y la verificó un escéptico que intentó refutar las afirmaciones de ausencia y de defecto; lo refutado no se usa aquí. Los mapas se hicieron sobre `ebbd6f6b` y `568375fb`; la síntesis se contrastó con `2ea9d06e` (main `40bd881b`). Las líneas de `god-engine-core/src/lib.rs` y `src/bin/god_engine.rs` pueden estar desplazadas hasta unas 300 líneas: antes de editar, cada ancla se vuelve a buscar con grep, como exige AGENTS.md. Las áreas 4, 5 y 6 llegaron recortadas a la síntesis y se completaron con lecturas propias, que se citan.
>
> Nada de este documento es un arreglo. Cada entregable entra como un commit atómico con un test que falla antes del cambio, y es ese test el que confirma el hallazgo.

## 1. Diagnóstico

**Lo sólido.** El núcleo de decisión es uno solo. `GodEngineCore::process_event` lo usan el vivo, los replays, el entrenador y el forense. El risk-engine tiene contratos de falsos positivos y falsos negativos para entradas, geometría, EV, apalancamiento y nocional mínimo. Hay un panel de métricas ex-post bien especificado (`crates/backtest-engine/src/metrics.rs:1-40`). Hay DSR con multiplicidad (`crates/evolution-engine/src/selection_stats.rs:141,229,239`), un oráculo T-1 que hace de trinquete y un golden bit a bit que se reproduce bajo MSVC nativo. Ya existen siete ADRs en `docs/adr/` y un `TABLERO.md` compartido. Desde esta semana hay CI en `main` (`.github/workflows/replay-contracts.yml`).

**Lo frágil.**
- La identidad slot↔símbolo del vivo está rota. El host congela `symbol_to_id`. El universo dinámico arranca en minúsculas y rota, y el núcleo, el riesgo y la reserva resuelven el símbolo a través de ese universo.
- Los arneses sin `GOD_NO_HOT_RELOAD` pueden aplicar el genoma activo encima del candidato.
- Los replays de altcoins calculan el régimen global con la propia altcoin (slot 0 = "BTC").
- Varias compuertas son inertes, absorbentes o decorativas: gate ATR, REJ_DRAWDOWN en backtest, `darwin_approved`.
- El host sustituye el apalancamiento que calcula el risk-engine.
- La evolución tiene vías que se saltan el armado y GENOME-GATE.

**Lo que falta.**
- No hay un log durable del flujo vivo.
- No hay un registro de decisión que una intención, orden y fill con la versión de genoma, modelo y features.
- No hay un model registry que se consuma al cargar.
- La CI no cubre el workspace ni T-1.
- `main` no tiene protección.
- Las métricas no se miden fuera de muestra de forma sistemática.

Con esta base, ninguna cifra de rendimiento es atribuible ni reproducible. Por eso no hay nada que permita operar con capital ni prometer crecimiento.

---

## 2. Prioridades en el orden del operador

### P1. Pipeline de datos y linaje

**Estado real**
- **Ingesta viva sin grabación.** Una cola `bounded(5_000)` con política drop-oldest (`src/bin/god_engine.rs:982`). Los descartes del camino de datos siguen sin contarse en HEAD: en la rama `Full` sólo hay `try_recv()`, en `god_engine.rs:~4747`. El contador sólo sube al encolar el centinela de reconexión (`god_engine.rs:4725`). Ese mismo drop-oldest puede tirar el centinela `[SYSTEM:RECONNECT]`. `RawTickArchive`, `FlightRecorder` y `Lakehouse` existen pero no tienen llamadores en producción (`god-engine-core/src/lib.rs:677,923,1044`).
- **Identidad slot↔símbolo rota** (omisión del escéptico):
  - el universo se carga en minúsculas: `god_engine.rs:785`;
  - el mapa del host queda congelado: `god_engine.rs:1330-1336`;
  - el demonio de rotación reordena el universo en su primer tick: `src/symbol_manager.rs:143-163,280-327`;
  - la clave del modelo sale del universo: `coin_model_key` desde `try_symbol`, `lib.rs:3279-3281`.

  Consecuencia: el modelo, la spec y las claves de registro pueden ser de otro símbolo.
- **Tape distinto del vivo.**
  - El tape son aggTrades con un libro de ±0,5 pb inventado (`src/bin/binance_vision_sync.rs:203-229`).
  - El vivo es `@trade` con reloj E, depth5 real y klines (`god_engine.rs:3013-3028`; `src/parsers.rs:128-141` descarta T y el id de trade).
  - La paridad D-753 sólo compara el entrenador contra el núcleo dentro del mismo tape. En las dims 0..34 compara el getter consigo mismo (`lib.rs:1087`; `train_forest.rs:388-446`).
  - La dim 2 (OFI) se calcula en el tape con cantidades sintéticas que codifican el lado (`stateful_engine.rs:941-950,1076`). Esto pide confirmación en un run.
- **Linaje de artefactos.**
  - Genomas: el sobre lleva generación pero no hash. La historia se puede sobrescribir (`quantum-arena/src/genome_store.rs:310-332,398-409`) y hay más de 11 promotores.
  - Modelos: el artefacto sólo tiene árboles (`ml_inference.rs:12-21`). Se escriben sin atomicidad (`train_forest.rs:2194`). Existe un manifiesto con SHA-256 (`ml_registry.rs:32-50`, `src/bin/model_manifest.rs`), pero nadie lo verifica al cargar.
  - Un `.bin` huérfano resucita un modelo retirado (`god_engine.rs:1178-1183`; no existe `remove_global`).
- **Claves de registro desacopladas.** `whale_burst_z` y `spoof_score` se escriben con `set_scoped` (`god_engine.rs:3112-3114,3217-3219`) y se leen con `get_for_coin_or` (`lib.rs:6455-6463`). El Consejo lee siempre 0.
- **Diarios sin vínculo.**
  - `position_journal` guarda el `ml` del instante del OCO y se compacta (`god_engine.rs:4338-4352,601-650`).
  - `trade_fills.jsonl` no lleva orderId (`execution-engine/src/trade_accounting.rs:138-148`).
  - Journals y estado comparten `data/` entre demo y prod (`trade_accounting.rs:28`; `god_engine.rs:371-393`).

**Brechas.** Falta lo siguiente: log de sesión viva; DecisionRecord; consumo del manifiesto de modelos; hash y cerrojo de genomas; feature schema versionado (el vector se ensambla en tres sitios de `lib.rs`, más una copia en `train_forest.rs:372-386`); SHA de código en los artefactos (no hay `build.rs`); y una prueba replay-vs-vivo.

**Primeros entregables**
1. **Identidad de símbolo única** (va antes que cualquier registro).
   - Qué hace: núcleo, riesgo, reserva y host resuelven el símbolo con el mismo mapa inmutable de sesión, normalizado a mayúsculas. La rotación sólo añade al final y nunca reordena. El despacho compara `parsed_sym == reservation.symbol`.
   - Aceptación: un test de integración en `god-engine-core/tests` arranca con el universo `["btcusdt","atomusdt"]` y simula una rotación que pone ATOM primero. Hoy falla. Después, `coin_model_key(0) == "BTCUSDT_MOTOR"` y un tick de BTC nunca se puntúa con el modelo de ATOM.
   - Esfuerzo M. Depende de coordinarse con quien toca `symbol_manager.rs`.
2. **Arneses con genoma fijado.**
   - Qué hace: `GodEngineCore` recibe la política de hot-reload por constructor; los arneses de evaluación la desactivan por defecto, no por variable de entorno.
   - Aceptación: un test de `booktick_replay` con `TG_GENOME_ENV=demo` y un candidato distinto de `demo/active.json` conserva el candidato tras 1 000 ticks. Hoy fallaría si se confirma el mecanismo (`lib.rs:1186-1216`; `booktick_replay.rs:270-272`).
   - Esfuerzo S. No depende de nada.
3. **DecisionRecord append-only.**
   - Qué guarda: `decision_id`, símbolo, slot, generación y sha256 del genoma, clave y sha256 del modelo, `ml_prob`, `ml_model_base`, rama, vector 48D, `ValidatedOrder`, `client_order_id`.
   - Cómo escribe: hilo de fondo en `data/{env}/decisions/{session}.jsonl`. `trade_fills` añade `decision_id`.
   - Aceptación: un replay de 20 000 ticks con sumidero en memoria produce exactamente una apertura y un cierre por posición. Reevaluar el modelo por hash sobre el vector guardado reproduce `ml_prob` bit a bit, y todo fill se une a su decisión.
   - Esfuerzo M. Depende de los entregables 1 y 2.
4. **Correcciones puntuales de linaje.**
   - Qué cubre: claves `whale_burst_z`/`spoof_score`; diario con `pos.ml_prediction` y precio con precisión completa; contador de descartes del camino de datos; régimen global a partir de BTCUSDT por símbolo y neutral si no está.
   - Aceptación: lo que publica el host lo lee el Consejo con valor distinto de 0; una cola saturada da `dropped > 0`; un replay de altcoin sin BTC deja `market_regime` neutral.
   - Esfuerzo S. Depende de re-certificar T-1 y el golden con aprobación del dueño, porque cambia decisiones.

### P2. Vetos, bloqueos y límites

**Estado real**
- **Gate ATR inerte.** `atr_ok = atr_pct > dynamic_atr_min` (`lib.rs:~4082`). Mientras tanto, `v_t` tiene suelo de 10 pb (`stateful_engine.rs:846,872,876`) y el genoma versionado lleva el gen 27 en 1,5e-4. El "colapso evolutivo a 0,0012" es código muerto (`evolution-engine/src/lib.rs:640` dentro de `start_evolution_loop`, sin llamadores).
- **REJ_DRAWDOWN es un gatillo de pelo y está mal especificado.**
  - Con `global_max_drawdown = 0.5` (`config_dir/genomes/demo/active.json:9`) da α = 0,5 y DD ≈ 2 % (≈0,26 USD sobre 13).
  - Modela una racha consecutiva, pero se compara con la caída máxima desde el pico (`risk-engine/src/drawdown.rs:14-121`).
  - Es absorbente en backtest (`backtest-engine/src/lib.rs:431-445`) y se reinicia en cada reconexión WS (`lib.rs:1023-1030`; `god_engine.rs:2968`).
- **El host sustituye el apalancamiento del risk-engine.** Con n<30 usa L=1 sin mirar si es "operable" (`god_engine.rs:3823,3864-3876`). Luego `LEVERAGE-ADAPT` lo sube hasta 20 (`god_engine.rs:~4128-4139`). `rej(5)` y la matriz de apalancamiento deciden sobre un valor que nunca se ejecuta.
- **GENOME-GATE no se aplica al cargar.** `load_active` sólo deserializa (`genome_store.rs:140-176`). Los `active.json` versionados tienen `margin_cushion_pct = 1.0922`, por encima de la cota de 0,98.
- **Kill-switch incoherente.** La bandera de la arena detiene también la gestión de salidas del núcleo (`lib.rs:1881`; test OPEN `god-engine-core/tests/close_outcome_contract.rs:350-367`). X-009 arma esa bandera sobre una posición declarada desnuda y además apaga el watchdog de protección (`god_engine.rs:4428-4437,2500-2503`).
- **La evolución se salta el embudo.**
  - `.forensic_violation` escribe umbrales ML vivos sin comprobar el armado (`online_daemon.rs:1000-1016`).
  - La evolución viva está armada por defecto en demo (`online_daemon.rs:742-752`).
  - Darwin aplica el genoma antes de `promote` (`darwin.rs:596-640`).
- **Estado que se pierde al reiniciar.** Calibradores en identidad: el gate de EV usa la puntuación cruda como p (`calibration.rs:86-133`; `lib.rs:6059-6067`). También se pierden `cumulative_trials`, el vigilante de rollback (`online_daemon.rs:834-835`) y la racha y el enfriamiento, que además se reinician en cada reconexión (`stateful_engine.rs:208-260`).

**Brechas.** Faltan tests de FP y FN para escudo macro, condicionamiento espectral, OBI, CVD z, spread, margen libre, TOXIC_FLOW, asiento Auditor, sistema inmune y freno de comisiones. Falta paridad backtest↔vivo de B3.6, sistema inmune, deriva y envolvente. Y el registro de vetos (`risk-engine/src/veto_registry.rs`) describe mal varias compuertas.

**Primeros entregables**
1. **El host respeta al risk-engine.** Se elimina `LEVERAGE-ADAPT` o se reevalúan `rej(5)` y el techo genómico antes de enviar, y lo mismo en `booktick_replay.rs:623-747`.
   - Aceptación: un test de host y replay donde ninguna orden enviada tiene L por encima del techo genómico ni `roundtrip·L` por encima del presupuesto. Hoy falla con margen libre escaso.
   - Esfuerzo S.
2. **GENOME-GATE al cargar.** `load_active` y `load_or_baseline` llaman a `validate`. En demo y prod se rechaza o se registra "unversioned" según el ADR-0009.
   - Aceptación: cargar `demo/active.json` falla hoy por `margin_cushion_pct`. Pasa tras recortar el genoma con un commit atómico aprobado.
   - Esfuerzo S. Depende del ADR-0009.
3. **Kill-switch que no abandona posiciones.** Con la bandera de la arena activa, el núcleo sigue gestionando salidas. X-009 y `latch_degradation` aplanan o re-protegen.
   - Aceptación: el test `open_kill_switch_blocks_even_a_local_stop_close_proposal` se invierte y pasa (un tick que cruza el SL produce `closed_order`). Un test nuevo comprueba que, tras X-009, la posición queda cerrada o con bracket.
   - Esfuerzo M.
4. **Cerrar los bypasses de evolución.** `.forensic_violation` y los umbrales del shadow forest exigen armado y pasan por `promote`. Darwin aplica sólo si `promote` devuelve Ok.
   - Aceptación: con la evolución desarmada, `.forensic_violation` no cambia `ml_threshold_long`. Un genoma rechazado por `validate` no llega a la arena.
   - Esfuerzo M. Hay que coordinarlo con GLM.

### P3. CI/CD, ramas y merge limpio

**Estado real** (HEAD `2ea9d06e`; esto corrige al mapa, que se hizo antes de que la CI entrara en `main`)
- **CI en `main`.** `.github/workflows/replay-contracts.yml` corre en windows-2022 con `nightly-2026-06-30` y techo de 90 min. Hace `check --workspace --all-targets --locked`, tests de `ml_registry` y de backtest-engine, y `t1_measurement_contract`. `git diff --check HEAD^ HEAD` sólo mira el último commit. No cubre feature-engine, data-pipeline, storage-engine ni audit-engine (unos 560 tests), y no corre T-1.
- **Toolchain flotante.** `rust-toolchain.toml` dice `channel = "nightly"` sin fecha, mientras la CI fija la de 06-30 y la nube usa la de 09-27. `.cargo/config.toml:2` fija `target-cpu=native`, peligroso si se añade caché.
- **T-1 en verde de nuevo, según su autor.** HEAD tiene `COBERTURA_MINIMA = 0.110` (`t1_cobertura_genetica.rs:273`), y TABLERO.md registra 17/144 tras CL-35c. Sigue sin `#[ignore]`, así que `cargo test --workspace` lo ejecuta en perfil dev.
- **Marcadores de conflicto.** `git grep` de marcadores en HEAD no devuelve nada: los de `2080e423` ya no están.
- **`main` sin protección.** El mapa registró `protected:false` y 71 de 79 commits de primer padre hechos por push directo con una cuenta administradora. No lo volví a verificar en esta pasada.
- **Lo demás.** `scripts/quality_gate.ps1` está roto (fmt con 737 hunks; argumento de clippy mal formado). `os-guardian` impide compilar en Linux nativo e intoxica a 13 paquetes (`os-guardian/src/lib.rs:15-23`), y su rama Linux tiene un E0425 latente (`observability_plane.rs:27-32`).

**Brechas.** Falta: ruleset sin bypass, toolchain fijado con fecha, CI de workspace completa, T-1 automático (nocturno y en PRs que tocan crates), compilación nativa en Linux y canal de coordinación fuera de la rama de código.

**Primeros entregables**
1. **Toolchain con fecha.** `rust-toolchain.toml` pasa a `channel = "nightly-2026-06-30"`, igual que la CI y el operador.
   - Aceptación: `rustc --version` coincide en la CI y en la sesión cloud, y el golden pasa en ambas.
   - Esfuerzo S. Depende de re-certificar en la nube con ese compilador.
2. **T-1 explícito.** Se marca con `#[ignore = "oráculo largo: --ignored en release"]` y se crea `.github/workflows/t1-nightly.yml` (cron y `workflow_dispatch`, release, `--test-threads=1`, `--ignored`).
   - Aceptación: `cargo test -p backtest-engine` ya no lo ejecuta, y el job nocturno publica N/144.
   - Esfuerzo S.
3. **CI de workspace en matriz por crate**, con caché keyed por toolchain y `RUSTFLAGS="-C target-cpu=x86-64-v3 -C opt-level=3"`.
   - Aceptación: todos los crates del workspace corren `--all-targets` en verde y el tiempo total queda por debajo de 90 min.
   - Esfuerzo M.
4. **Ruleset en `main`.** PR obligatorio con checks requeridos y sin bypass, y la coordinación (buzón, tablero) movida a rutas exentas o a issues.
   - Aceptación: un push directo a `main` es rechazado por GitHub.
   - Esfuerzo S. Depende de la decisión del dueño (ADR-0012).

### P4. Métricas realistas y riesgo

**Estado real**
- **Panel ex-post correcto pero sin consumidores de producción.** `ex_post_metrics` (`backtest-engine/src/metrics.rs:126`) calcula CAGR, Sharpe, Sortino, Calmar, MaxDD, CVaR, turnover y PF. Busqué `rg 'ex_post_metrics\('` fuera de `metrics.rs`: sólo aparece en `tests/ex_post_metrics_contract.rs`.
- **Aptitud ≠ crecimiento log medido.** `fitness_compute` (`god-engine-core/src/lib.rs:20-25`) devuelve `-inf` con menos de 30 cierres y no tiene factor OOS (comentario en `lib.rs:15-18`).
- **DSR implementado** (`selection_stats.rs:141,239`), pero `cumulative_trials` no se persiste, así que tras cada reinicio subestima la multiplicidad.
- **Tamaño fuera del espacio de riesgo.** Kelly entra dos veces, como margen y bajo raíz en el apalancamiento. Con 13 USD el riesgo al stop lo fija el nocional mínimo (≈5·SL; `risk-engine/src/lib.rs:1086-1150`). `REJ_VIABILIDAD` casi nunca muerde.
- **Tres definiciones de q en una misma evaluación**: drawdown (`lib.rs:272-277`), correlación (`560-565`) y viabilidad (`1041-1051`). Además la reconciliación calcula win_rate como media simple sin prior (`execution-engine/src/reconciliation.rs:507-518`).
- **Fricción.** Hay una fricción única para gate, brackets y daemon (`tp_sl::roundtrip_friction`). Pero el libro es sintético en el tape, y la banda del genoma se normaliza con 10 pb de referencia.

**Brechas.** Falta: panel ex-post en el forense y en el examen del daemon; crecimiento log por operación con intervalo de confianza; split temporal obligatorio con OOS; capacidad (impacto frente a nocional); y dimensionado en espacio de riesgo.

**Primeros entregables**
1. **Panel en el forense.** `audit_forensic_backtest` y `run_backtest_native` emiten `ex_post_metrics` y el crecimiento log medio por operación con IC bootstrap por bloques, por cada mitad temporal.
   - Aceptación: un test del fixture determinista donde la línea de panel aparece y es idéntica en dos corridas. El forense falla si falta una mitad.
   - Esfuerzo S.
2. **Aptitud con componente OOS.** Para promover se exige crecimiento log positivo en el tramo de prueba posterior, además del DSR con `cumulative_trials` persistido.
   - Aceptación: un candidato que gana en selección y pierde en prueba no se promueve (hoy sí). `cumulative_trials` sobrevive a un reinicio simulado.
   - Esfuerzo M.
3. **Dimensionado en espacio de riesgo**, detrás de una bandera. Riesgo al stop = fracción de Kelly acotada por la ruina; nocional = riesgo/SL; margen = nocional/L.
   - Aceptación: un test donde riesgo al stop ≤ tope de ruina en toda orden validada, y donde una orden que exigiría arriesgar más que el tope por el nocional mínimo se rechaza en lugar de recortarse.
   - Esfuerzo L. Depende de P1-3 (medir antes de cambiar) y del ADR-0013.
4. **Una sola q.** El risk-engine recibe `q` de un único proveedor (posterior con prior del genoma) y la reconciliación deja de escribir win_rate.
   - Aceptación: un test con una primera pérdida contada por la reconciliación donde las tres compuertas ven la misma q.
   - Esfuerzo S. No toca `correlation_guard.rs`: sólo cambia el argumento que se le pasa desde `lib.rs`.

### P5. Adaptación online y drift

**Estado real**
- **Detector de cambio sin consumidor.** BOCPD publica `spectral_p_transition` (`god-engine-core/src/lib.rs:1602,2080`). Busqué `rg '"spectral_p_transition"'`: nadie lo lee, y en `lib.rs:5914-5919` el valor se descarta con `let _ = p_transition`.
- **Auditor de deriva casi inerte.** Compara contra un PnL sombra de 0,95·real con umbral 0,05 (`audit-engine/src/drift_auditor.rs:64-70`), así que sólo dispara con un PnL superior al 100 % del nocional. Si dispara, es absorbente sin posiciones (`drift_auditor.rs:46-57`).
- **Aprendizaje online que sí actúa.**
  - Platt con semivida de 12 h (`calibration.rs:78,130-145`).
  - Hedge del ensamble con `eta = 0.05` (`ensemble.rs:145,253`).
  - Umbrales del shadow forest en vivo, que aprenden de features del cierre: fuga temporal (`online_daemon.rs:885-957`).

  Nada de ese estado se persiste ni se versiona.
- **Evolución viva armada por defecto en demo** (`online_daemon.rs:751`), con rollback t ≤ −2 que no sobrevive a un reinicio.

**Brechas.** Falta: distinguir drift de datos (features vivas frente a entrenamiento), drift de concepto (habilidad del modelo) y drift de ejecución (slippage real frente a modelado). Falta un snapshot versionado del estado aprendido y un consumidor de `p_transition`.

**Primeros entregables**
1. **Monitor de drift de features.** Por cada feature viva del bosque, PSI o KS frente a la distribución de entrenamiento guardada en el manifiesto del modelo.
   - Aceptación: un test donde una distribución desplazada 2σ dispara y otra igual no. La métrica sale en telemetría con el hash del modelo.
   - Esfuerzo M. Depende de P1 (manifiesto consumido).
2. **Bus del daemon sin fuga temporal.** El frame lleva las features de la decisión (las del DecisionRecord), no las del cierre.
   - Aceptación: un test donde las features del frame son idénticas a las registradas en la apertura. Hoy difieren (`lib.rs:2750-2758`).
   - Esfuerzo S. Depende de P1-3.
3. **Snapshot del estado aprendido** (Platt, Hedge, TasaAcierto, `cumulative_trials`, vigilante de rollback) en `data/{env}/learned_state/`, con versión de esquema.
   - Aceptación: un test de serializar y cargar con decisiones idénticas antes y después de un reinicio simulado.
   - Esfuerzo M.
4. **Drift de ejecución.** El slippage real de cada fill frente al modelado por `roundtrip_friction` sale en telemetría, con un veto sólo si la evidencia es significativa.
   - Aceptación: test de FP y FN con fills sintéticos.
   - Esfuerzo M. Depende de que los fills lleven `decision_id`.

### P6. Teorías avanzadas y deep learning (sólo con validación)

**Estado real**
- **Piezas neuronales sin evidencia de habilidad.**
  - DarkAlpha se lee de `models/DarkAlpha_BTCUSDT.json` en cada `GodEngineCore::new` (`lib.rs:915-940`).
  - El auto-trainer lo sobrescribe a partir de un CSV sin esquema que rellena con ceros (`src/bin/auto_trainer_daemon.rs:108-122,394-401`).
  - `OnlinePpoPolicyEngine` se instancia en el núcleo (`lib.rs:745,945`).
  - El conformal es sólo telemetría (`lib.rs:3806-3807`).
- **Espectro con habilidad medida parcial.** `SpectralForecastBank` (`quantum-arena/src/spectral_tape.rs:601-628`) muestra R² OOS positivo en volatilidad, volumen e intensidad, no en dirección (MEMORIA 2026-09-25, pendiente de re-medir).
- **Triage como portón.** ADR-0005 ya fija el triage de cuatro estados para teorías nuevas. El commit `c71be62b` deja propuestas (Cramér-Lundberg, f(α), Ricci) pendientes del operador.

**Brecha.** Ninguna pieza avanzada tiene una prueba OOS que la compare contra un baseline simple antes de votar en vivo.

**Primeros entregables**
1. **Inventario de votantes con habilidad.** Para cada votante (bosque, DarkAlpha, PPO, ramas, espectro), log-loss o Brier OOS frente a la persistencia y frente a la base del modelo, sobre el mismo tape con paridad.
   - Aceptación: una tabla reproducible por script. Un votante sin habilidad positiva con IC que excluya 0 pasa a peso 0 en el ensamble mediante una bandera.
   - Esfuerzo M.
2. **DarkAlpha bajo contrato.** Dataset con timestamp, símbolo, versión de features y generación; el auto-trainer rechaza filas de otro esquema; artefacto con manifiesto.
   - Aceptación: una fila de 26 columnas se rechaza (hoy se rellena con ceros).
   - Esfuerzo S.

---

## 3. Catálogo de vetos

Leyenda. ⚠ = sospechoso. ✎ = corregido por el escéptico. Las líneas son las de `crates/risk-engine/src/lib.rs` (RE), `crates/god-engine-core/src/lib.rs` (CO) o `src/bin/god_engine.rs` (HO), salvo que se indique otro archivo. Abreviaturas de la columna «Test»: FP = falso positivo, FN = falso negativo, FP+FN = ambos.

| Id | Capa | Clase | Umbral y origen | Test | Sospecha |
|---|---|---|---|---|---|
| rej(0) flat/coin | riesgo | integridad | literal; RE:213 | ninguno | ⚠ mezcla «sin señal» con «moneda inválida» |
| REJ_INVALID_INPUT 15 | riesgo | integridad | dominio; RE:225-235… | FP+FN | — |
| REJ_TARGET_GEOMETRY 16 | riesgo | integridad | lado y finitud; RE:238,791 | FP+FN | ⚠ el registro de vetos no la lista bien |
| REJ_DRAWDOWN 10 | riesgo | duro | `drawdown_maximo`, H=200, α del gen; RE:254-286 | FN | ⚠ ≈2 % con gen 0,5; racha ≠ caída desde pico; absorbente en backtest; se reinicia al reconectar; lerp D-641 muerto |
| V-001 clamp_ruin | riesgo | duro | literales 0,25/0,05/0,35; `ruin.rs` | unit | ⚠ doc invertida; literales sin origen |
| rej(3) sin spec | riesgo | integridad | presencia; RE:429 | ninguno | — |
| rej(1) exposure0 | riesgo | duro | `raw_exposure==0`; RE:523 | ninguno | ⚠ el registro de vetos describe otra causa |
| rej(2) correlación | riesgo | duro | `veto_por_riesgo_real_medido`; RE:535-577 | FP+FN (Codex) | ⚠ `tope/8` en frío (zona Codex, sólo descrito) |
| REJ_TP_SL_FLOOR 11 | riesgo | estrategia | `fee/0,65`; `tp_sl.rs:275-381` | FN | ⚠ banda del genoma a 10 pb frente a fricción real; domina el embudo. ✎ el FP tras recortes es marginal (sólo con fricción > 35,8 pb) |
| V-002 tope micro 55 pb | riesgo | duro | literal 0,0055/2,25; RE:768 | ninguno | ⚠ recorta en vez de rechazar (política del dueño) |
| REJ_CONFIDENCE 12 | riesgo | estrategia | gen `min_confidence_btc` · ratio micro; RE:875-928 | sólo dominio | ⚠ sin test del umbral. ✎ el régimen micro SUBE la exigencia (hasta +6,45 %) |
| REJ_SIN_EVIDENCIA 14 | riesgo | estrategia | p y LCB ausentes; RE:953-982 | diagnóstico | ⚠ ✎ inalcanzable desde el núcleo (p siempre `Some`) |
| rej(4) EV | riesgo | estrategia | `fee·mult(gen)`; RE:992-1005 | FP+FN | ⚠ ✎ en frío y tras reinicio usa la puntuación cruda como p |
| REJ_VIABILIDAD 13 | riesgo | duro | `orden_viable`; RE:1043 | unit | ⚠ casi nunca muerde con 13 USD |
| V-003 rescate de nocional | riesgo | duro | literales 5/1,5/50; RE:1086-1115 | FP+FN | ⚠ infla el margen (contradice D-750) |
| rej(6) min_notional | riesgo | operativo | spec del símbolo; RE:1116,1164 | FP+FN | — |
| V-004 techo de margen | riesgo | duro | literales 0,90/0,25/1,20/2,60; RE:1127 | FN | ⚠ recorte silencioso |
| rej(5) fee_impact | riesgo | estrategia | lerp(gen, 0,035); RE:1160 | FN | ⚠ inefectivo: el host sustituye L |
| rej(8) allow_trade | riesgo | duro | colchón(gen) − crash; `orchestrator.rs` | FP+FN | ⚠ techo direccional redundante; literal 0,25 |
| V-005 rej 7 y 9 | riesgo | operativo | sin emisor | ninguno | ⚠ siempre 0 |
| V-006 `guard::check_drawdown_limit` | riesgo | duro | muerto | diagnóstico | ⚠ falla abierto |
| V-007 compounder/epigenético | riesgo | duro | muerto | contratos | ⚠ protección aparente |
| V-008 kill-switch arena | núcleo | duro | bandera; CO:1490,1881 | ✎ test OPEN FMT-232 | ⚠ detiene la gestión de salidas |
| V-009 interbloqueo de entradas | núcleo | operativo | latencia > max(gen, 50 ms); CO:1885-1912 | ninguno | ⚠ gen versionado ≈6,9 s |
| V-010 gate ATR | núcleo | estrategia | gen 27; CO:~4082 | ninguno | ⚠ inerte; ✎ el colapso a 0,0012 es código muerto |
| V-011 spread | núcleo | estrategia | `tp_at_tau` − (maker+taker); CO:~4040 | ninguno | ⚠ incoherente con `tp_sl` |
| V-012 enfriamiento | núcleo | estrategia | τ·2^racha ≤ 12 h; `stateful_engine.rs:517` | FP+FN | ⚠ se borra en cada reconexión |
| V-013 D-472 | núcleo | estrategia | signo del score; CO:1246-1331 | FP+FN | — |
| V-014 F-009 ML | núcleo | estrategia | −0,80/2,5/0,20; CO:1246 | FP+FN | ⚠ literales sin origen |
| V-015 CVD de puertas | núcleo | estrategia | gen; CO:1246 | FN | ⚠ duplica V-025 |
| V-016 muro L2 | núcleo | estrategia | gen; CO:1246 | ninguno | ⚠ libro sintético |
| V-017 anti-persecución | núcleo | estrategia | ±0,05 + Z95; CO:4115 | ninguno | ⚠ literal sin origen |
| V-018 freno del bosque | núcleo | estrategia | Wilson > 0,5; CO:5362 | ninguno | ⚠ R8-A abierto |
| V-019 escudo macro | núcleo | estrategia | Z95 + capitulación; CO:5386 | ninguno | ⚠ sin tests |
| V-020 espectral | núcleo | estrategia | 5 literales de flujo; CO:5480-5600 | ninguno | ⚠ sin re-derivar tras CL-32 |
| V-021 equidad cero | núcleo | duro | cap ≤ 0; CO:5680 | ninguno | ✎ el sistema inmune sí aplana |
| V-022 anti-whiplash | núcleo | estrategia | ventanas τ + signo del stretch; CO:5689-5790 | ninguno | ✎ no es código muerto: sobrevive a la reconexión |
| V-023 tras racha | núcleo | estrategia | p85/p80; CO:5815 | FN | ⚠ la racha se borra al reconectar |
| V-024 OBI | núcleo | estrategia | Z95·sd; CO:5848 | ninguno | ⚠ falla abierto tras cada reconexión |
| V-025 CVD z | núcleo | estrategia | Z95 / ±0,12; CO:5892 | ninguno | ⚠ duplicado; literal de calentamiento |
| V-026 puerta neural | núcleo | operativo | diagnóstico | ninguno | ⚠ parece veto y no lo es |
| V-027 conformal | núcleo | operativo | telemetría; CO:3806 | Codex | ⚠ parece veto y no lo es |
| V-028 slots | núcleo | estrategia | 3 slots, Δlnτ 0,80; `position.rs:639` | FN | ⚠ `same_dir_unsecured` muerto |
| V-029 Ente / cascada | consejo | duro | 0,85; `consejo_seniors.rs:441` | FP+FN | ⚠ ballena y spoof siempre 0 (claves) |
| V-030 Causal VPIN | consejo | estrategia | 0,75; `consejo_seniors.rs:522` | FN | ⚠ literal del llamador |
| V-031 Riesgo dd | consejo | duro | 0,95−0,10·s | ✎ `quantum_organism_test.rs:42-52` | ⚠ inalcanzable detrás de REJ_DRAWDOWN |
| V-032 Ejecución | consejo | estrategia | 35+65·s pb | ninguno | ⚠ no está ligado a la fricción |
| V-033 Teleonomía / Auditor | consejo | estrategia | 0,02/0,35; 0,92 | FP | ⚠ el Auditor no tiene test |
| V-034 aprobación D-738 | consejo | estrategia | 0,35/0,80/0,28 | FP+FN | ⚠ literales sin origen |
| V-035 roster ML / sonda | núcleo | estrategia | base + lift(gen); CO:3278,6041 | golden | ⚠ ✎ `trade_count` tiene un 2.º escritor (`reconciliation.rs:507`); ✎ BTCUSDT_MOTOR ya está en el manifiesto |
| V-036 margen libre | núcleo | duro | 0,95·crash; CO:6041+ | ninguno | ⚠ duplica rej(8) con otra fórmula |
| V-037 salidas de riesgo | núcleo | duro | TOXIC 0,75/3/0,85; CO:2285-2537 | ninguno | ⚠ no corre bajo kill-switch |
| V-038 salidas de estrategia | núcleo | estrategia | más de 8 literales; CO:2285 | ninguno | ⚠ tiempos absolutos que no escalan con τ |
| V-039 mainnet | host | duro | `--force-live` + `MAINNET_ARMED`; HO:711 | ninguno | — |
| V-040 `darwin_approved` | host | operativo | `true` incondicional; HO:1167 | interno | ⚠ decorativo |
| V-041 sistema inmune | host | duro | `drawdown_maximo` + 3 strikes; HO:1594-1752 | ninguno | ⚠ capital y pico distintos de REJ_DRAWDOWN; gatillo de pelo con latch |
| V-042 freno B3.6 | host | estrategia | trades ≥ 3, fees > gross; HO:2164-2340 | ninguno | ⚠ cuenta filas de income; ✎ bloqueo real ≈8–20 h; falla abierto con JSON corrupto |
| V-043 pánico de latencia | host | operativo | gen; HO:3119 | ninguno | ⚠ un solo evento; ✎ sólo deriva del reloj, no el desfase bruto |
| V-044 deriva | host | integridad | 0,05; `drift_auditor.rs:64-70` | FN | ⚠ casi inerte y absorbente |
| V-045 pánico de memoria | host | operativo | RAM > 85 %; HO:3804 | ninguno | ⚠ no existe en backtest |
| V-046 envolvente Kelly | host | duro | n<30 ⇒ L=1; z 1,64→0,85; HO:3830-3960 | FP+FN | ⚠ «no operable» no rige con n<30; sustituye la L del risk-engine |
| V-047 LEVERAGE-ADAPT | host | duro | ceil(…), [1,20]; HO:~4128 | réplica | ⚠ se salta rej(5) |
| V-048 MARGIN-GUARD | host | duro | 0,95; HO:4139 | réplica | — |
| V-049 sin filtro de símbolo | host | integridad | presencia; HO:4150 | ninguno | — |
| V-050 NAKED-ESCALATION | host | duro | HO:2644 | ninguno | ⚠ se omite bajo kill-switch |
| V-051 X-009 | host | duro | HO:4428 | ninguno | ⚠ apaga la protección de la posición desnuda |
| V-052 kill-switch del ejecutor | ejecución | duro | bandera; `executor.rs:322` | FP+FN (CL-3) | — |
| V-053 rate limit | ejecución | operativo | 3×429, 80 %, 20 ops/s | FP | — |
| V-054 flatten_all | ejecución | duro | `executor.rs:678-830` | ninguno | ⚠ sólo lo llaman el sistema inmune y el apagado |
| V-055 GENOME-GATE | evolución | duro | cotas + banda a 10 pb; `genome_store.rs:228-310` | FP+FN | ⚠ ✎ no se aplica al cargar; los genomas versionados no lo pasan |
| V-056 armado evolutivo | evolución | operativo | demo armado; `online_daemon.rs:742` | ninguno | ⚠ sin opt-in |
| V-057 DSR | evolución | estrategia | 0,95; `selection_stats.rs:229` | unit | ⚠ `cumulative_trials` no persiste |
| V-058 WF / INVIABLE | evolución | estrategia | 30 cierres, 13 USD | unit | — |
| V-059 rollback | evolución | duro | t ≤ −2, n ≥ 20 | CL-25 | ⚠ ✎ no sobrevive a un reinicio |
| V-060 `latch_degradation` | evolución | duro | n ≥ 25, t < −1,5 | ninguno | ⚠ no aplana |
| V-061 umbrales del shadow forest | evolución | estrategia | armado; `online_daemon.rs:938` | ninguno | ⚠ fuera de GENOME-GATE; fuga temporal |
| V-062 `.forensic_violation` | evolución | estrategia | 0,60/0,40; `online_daemon.rs:1000` | ninguno | ⚠ ✎ también se salta el armado |
| V-063 Darwin +5 % | evolución | estrategia | `darwin.rs:312` | unit | ⚠ aplica antes de `promote` |
| V-064 envolvente del replay | backtest | duro | `sl_at_tau`; `booktick_replay.rs:623-747` | FP+FN | ⚠ paridad rota con el stop real |
| V-065 drawdown en backtest | backtest | duro | sin reset; `backtest-engine/src/lib.rs:431` | ninguno | ⚠ absorbente |

### 3.1 Reconciliación con el registro de vetos de `main`

`crates/risk-engine/src/veto_registry.rs` (ADR-0004, «el censo es código») tiene 17 entradas en `40bd881b`. Los ids V-001…V-065 de la tabla son provisionales del censo. Una compuerta que entre en el registro toma un id con su estilo (V-RISK-nnn o V-LOGIC-nnn), y la entrada se escribe en el mismo commit que su test.

| Registro | Censo | Nota |
|---|---|---|
| V-RISK-001 exposure0 | rej(1) | El registro y el censo describen causas distintas. |
| V-RISK-002 correlación | rej(2) | Zona Codex. |
| V-RISK-003 margen_insuf | V-036 / V-048 | Su test (`replay_con_envolvente_sigue_determinista`) prueba determinismo, no FP/FN. |
| V-RISK-004 min_notional | rej(6) | — |
| V-RISK-005 drawdown | REJ_DRAWDOWN 10 | El censo lo marca absorbente en backtest y reiniciado al reconectar; el test del registro no cubre ninguna de las dos cosas. |
| V-LOGIC-001 viabilidad | REJ_VIABILIDAD 13 | — |
| V-LOGIC-002 roster ML | V-035 | — |
| V-LOGIC-003 Fisher WF | — | Retirado (CL-28). |
| V-LOGIC-004 FMT-055 | — | Retirado (CL-29). |
| V-LOGIC-005 kill-switch sobre entradas | V-008 | El registro lo da por bueno; el censo encuentra que también detiene la gestión de salidas (test OPEN FMT-232). |
| V-LOGIC-006 suelo TP/SL | REJ_TP_SL_FLOOR 11 | El registro lo clasifica como riesgo duro; el censo, como lógica de estrategia. Hay que decidirlo. |
| V-LOGIC-007 confianza | REJ_CONFIDENCE 12 | Sin test en ambos. |
| V-LOGIC-008 geometría | REJ_TARGET_GEOMETRY 16 | — |
| V-LOGIC-009 sin_evidencia | REJ_SIN_EVIDENCIA 14 | El censo lo encuentra inalcanzable desde el núcleo. |
| V-LOGIC-010 rama 15 | — | No estaba en el censo. |
| V-LOGIC-011 warmup | — | No estaba en el censo. |
| V-LOGIC-012 banda operable (#586) | — | Entró en `main` después del mapa. |

Hay tres cosas que el registro aún no hace:
- Cubre 17 de unas 65 compuertas; quedan fuera, entre otras, REJ_INVALID_INPUT (15), rej(4) EV, rej(5) fee_impact, rej(8) allow_trade, el sistema inmune del host y LEVERAGE-ADAPT.
- Su comentario dice que «el test de cobertura obliga a que cada rej nombrado tenga entrada», pero ese test no existe: los cuatro tests del módulo comprueban campos, clases, retiros y fuentes de umbral, no la cobertura de los códigos `REJ_*`.
- Dos entradas citan como test un contrato que no prueba FP ni FN (V-RISK-003 y, en parte, V-RISK-005).

Primer paso: un test que enumere los `REJ_*` y los `rej(n)` emitidos por el risk-engine y falle si alguno no tiene entrada, con cada entrada nueva acompañada de su par de tests FP/FN.

---

## 4. Arquitectura neuronal propuesta

**Principio.** La red propone y las compuertas duras disponen. Ninguna capa entra en la parte diferenciable si antes no gana a un baseline simple fuera de muestra con IC que excluya 0.

**Orden de diferenciabilidad.** De menor a mayor riesgo:
1. **Calibración**: Platt o isotónica por activo. Ya es un modelo paramétrico. Su salida es una probabilidad, y se valida con log-loss y fiabilidad OOS.
2. **Pesos del ensamble.** Hoy son Hedge. Pasarían a una combinación aprendida (stacking logístico) sobre votantes con habilidad medida, entrenada con la etiqueta de barrera.
3. **Horizonte**: τ predicha y σ(τ) a partir de `SpectralForecastBank`, que ya tiene R² OOS positivo en volatilidad. Sustituye heurísticas de la geometría, no compuertas.
4. **Dirección** (bosque o red sobre el vector 48D versionado). Sólo después de cerrar la paridad tape↔vivo de P1. Sin ella se entrena sobre otra distribución.
5. **Política de tamaño** (fracción de Kelly en espacio de riesgo). La última, y siempre acotada por fuera.

**Fuera de la red, siempre.** Quedan como código determinista con tests de FP y FN:
- tope de ruina y viabilidad;
- drawdown y sistema inmune;
- kill-switch y aplanado;
- techo de apalancamiento y presupuesto de comisiones;
- nocional mínimo y filtros del exchange;
- puerta de mainnet;
- límites de rate;
- GENOME-GATE;
- validación de dominio (REJ_INVALID_INPUT).

**Validación obligatoria antes de producción.**
- Paridad bit a bit de features entre entrenamiento y servicio sobre una sesión viva grabada (P1).
- Split temporal con purga y embargo.
- Prueba posterior que no se use para seleccionar.
- DSR con multiplicidad persistida.
- Crecimiento log por operación con IC por bloques positivo en las dos mitades.
- Shadow en demo con DecisionRecord durante un periodo fijado por ADR, sin enviar órdenes.
- Rollback automático por hash.

**Primer paso pequeño.** Un experimento offline que compare el calibrador actual (identidad en frío) contra Platt ajustado en ventana deslizante y contra la frecuencia base, en log-loss OOS por activo, sobre tapes con paridad.
- Aceptación: un script reproducible que da la tabla con IC. Si Platt no gana, no se toca nada.

---

## 5. ADRs a escribir (después de los 0001–0007 existentes; números provisionales hasta que cada uno entre en `docs/adr/README.md`)

- **ADR-0008. Identidad de símbolo:** un mapa slot↔símbolo inmutable por sesión, en mayúsculas, que comparten host, núcleo, riesgo y reserva.
- **ADR-0009. Genoma no versionado:** demo y prod no operan con un genoma sin sobre válido ni que no pase GENOME-GATE al cargar.
- **ADR-0010. DecisionRecord:** toda orden enviada lleva un `decision_id` que la une con genoma, modelo, vector y fills en un log append-only por entorno.
- **ADR-0011. Model registry:** el cargador vivo sólo acepta artefactos cuyo sha256 coincide con el manifiesto, y retirar un modelo es una operación explícita.
- **ADR-0012. Protección de `main`:** sólo se entra por PR, con checks requeridos y sin bypass. La coordinación vive fuera de las rutas de código.
- **ADR-0013. Dimensionado en espacio de riesgo:** el tamaño se deriva del riesgo al stop acotado por la ruina, y el apalancamiento es una consecuencia del margen.
- **ADR-0014. Autoridad del risk-engine:** el host no modifica el apalancamiento ni el tamaño validados; si no caben, rechaza.
- **ADR-0015. Semántica del kill-switch:** bloquea entradas, nunca la gestión de salidas, y toda vía que lo arma también aplana o re-protege.
- **ADR-0016. Drawdown:** se define la prueba estadística (caída desde el pico frente a racha), el pico persistente y la regla de recuperación.
- **ADR-0017. Evolución viva:** desarmada por defecto en todos los entornos; toda escritura en la arena pasa por `promote` con linaje.
- **ADR-0018. Estado aprendido:** calibradores, Hedge, contadores y vigilantes se persisten versionados por entorno y se restauran al arrancar.
- **ADR-0019. Toolchain fijado:** una sola fecha de nightly para operador, nube y CI; los goldens se certifican sólo con ella.
- **ADR-0020. Portón para modelos neuronales:** los criterios de la sección 4 son obligatorios antes de que una capa aprendida vote en vivo.

---

## 6. Siguientes 3 ciclos

**Ciclo 8: identidad y autoridad.** Cada entregable lleva su test y va en un commit atómico.
1. Toolchain fijado y T-1 con `#[ignore]` más un job nocturno. El test es la CI verde con `rustc` igual al del operador.
2. Arneses con genoma fijado por constructor. El test es el de `booktick_replay` con `TG_GENOME_ENV=demo`.
3. Identidad de símbolo única (ADR-0008). El test es la rotación simulada en `god-engine-core/tests`.
4. GENOME-GATE al cargar (ADR-0009), más el commit que recorta `margin_cushion_pct` en los genomas versionados. El test: cargar falla antes y pasa después.
5. El host respeta al risk-engine (ADR-0014), en host y en replay. El test: ninguna L enviada supera el techo.

**Ciclo 9: linaje y kill-switch.**
1. Correcciones puntuales: claves del Consejo, diario, contador de descartes y régimen por símbolo. Tests de P1-4 y re-certificación de T-1 y del golden aprobada por el dueño.
2. Manifiesto de modelos consumido por el cargador, con escritura atómica en `train_forest` y retiro explícito (ADR-0011). Tests: manifiesto no coincidente rechazado; `.bin` huérfano no cargado.
3. DecisionRecord (ADR-0010). Test de replay de 20 000 ticks con reproducción de `ml_prob` por hash.
4. Kill-switch coherente (ADR-0015). Se invierte el test FMT-232 y se añade el test de X-009.
5. Bypasses de evolución cerrados (ADR-0017). Tests de `.forensic_violation` desarmado y de Darwin sin `promote`.

**Ciclo 10: métricas y estado.**
1. Panel ex-post y crecimiento log con IC en el forense. Test con el fixture determinista.
2. Snapshot del estado aprendido (ADR-0018). Test de serializar, cargar y obtener decisiones idénticas.
3. Bus del daemon con features de la decisión. Test de igualdad con el DecisionRecord.
4. Drawdown redefinido (ADR-0016): pico persistente y backtest no absorbente. Tests de reconexión, de FP en micro y de recuperación tras una caída del 3 %.
5. Batería de tests de FP y FN para los vetos del núcleo, el consejo y el host que no la tienen. Cada test de FN debe fallar si se comenta la condición del veto.