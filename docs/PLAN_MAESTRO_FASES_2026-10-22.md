# Plan maestro por fases V33 — revisión desde la base

**Revisión vigente:** 2026-10-24, Kimi Quant Sr V33, en coordinación con los demás agentes.
**Base inspeccionada:** `main = 4f9ada0980fdcd155bdce522993d7ec0ac0f30c9 = origin/main`, con PR #140 integrado por fast-forward preservando los siete WIP y el stash histórico.
**Alcance:** workspace Rust de 21 crates + paquete raíz, 2.203 rutas versionadas y 847 archivos Rust. Estas cifras describen el censo mecánico, **no** archivos ya auditados.
**Objetivo de investigación:** crecimiento neto compuesto de +100 %/72 h. Es una hipótesis que debe demostrarse fuera de muestra y después de fees, funding y slippage; no una garantía ni una razón para aumentar riesgo. Producción permanece prohibida en esta revisión.

Este es el plan canónico de coordinación. Los planes de líneas aportan detalle, no fuentes paralelas de verdad. `AGENT_SYNC.md` contiene reservas y heartbeats; `.agents/MEMORIA.md` conserva los recibos de cierre.

**Resultado F1 baseline y recomposición (2026-11-04T17:14:27Z):** CAS local previo `main=1fd5adf0`; candidato integrado `79e6fbc0455a9a327158a5bc8ad4bbaac5bae2b7`, `origin/main=91bd085f`, sin push. Catorce archivos de `feature-engine` corregidos sin supresiones; se añadieron átomos en `omniscient-registry`, `flight-recorder`, SBE y raw zone. Evidencia acumulada: feature 403/403 (20 ignored, 0 filtered), nativas 914/914, DAZ 17/17, átomos 19/19, SBE 6/6, storage 50/50, LPPLS 4/4, workspace `check --all-targets` exit0 y Zero-Wait 22/22 previo. Clippy workspace deja sólo `navier_stokes::update` (QRA). Revisión agent-25 `With fixes`; Jacobi y cardinalidad multifractal con RED→GREEN. Formatter 632 targets/5 lotes con deuda baseline; `test_pre_push_gate.py` ausente. Attestation global y publicación siguen bloqueados. Producción NO-GO; ROI/PnL/WR/Sharpe/EV=`N/D`. Estado: `.agent_tx/staging/KIMI-QUANT-SR-V35/F1_FEATURE_BASELINE_STATUS.json`.

## Corte 2026-10-29 — F0: verdad instalada, candidato y purga congelada

**Corte de refs:** HEAD local `4d8db9a345bd1a78f3e479b5beba01bd0124fa3e`; `origin/main=9b2cde7c15eba058c67046317c22c1cfbc3f36c2`; remoto sólo `main`. `claude/linea-i-i6=46285e72feec1dc7b9a09dbf24b9f8f3f14bf121` se preserva con su worktree. El manifiesto por ruta, propietario, lease y SHA-256 vive en `AGENT_SYNC.md` §0A.

**Comportamiento instalado:** el A4 original ocupa diez rutas del workspace. Tres no tienen delta en la revisión posterior: `crates/execution-engine/src/execution_evidence.rs`, `crates/execution-engine/src/executor.rs` y `crates/quantum-arena/src/symbols.rs`. Esta verdad instalada no se confunde con el candidato ni recibe sus 498 pruebas históricas como evidencia transferible.

**Candidato revisado no instalado:** exactamente siete rutas bajo `.agent_tx/staging/AGENT-01-live-red/`: `crates/execution-engine/src/entry_dispatch.rs`, `crates/execution-engine/tests/entry_route_contract.rs`, `crates/execution-engine/tests/f325_min_notional_ioc_open_diagnostics.rs`, `crates/quantum-arena/src/symbol_registry.rs`, `src/bin/god_engine.rs`, `src/host_oms.rs` y `tests/host_oms_contract.rs`. T2 debe reconciliarlas; F0 no las instala ni las publica.

**Purga y estados históricos:** no se ejecutó `PurgeLegacyBranches`. La guarda exige `origin/main` fresco verificado por `ls-remote`, PR `MERGED` o merge SHA comprobado, ancestria y cero commits exclusivos, y ausencia de worktree/lease/reserva. HEAD local desactualizado no habilita borrado remoto. Los estados históricos `FUSIONADO`/`CERRADO` no son operativos ni prueban ausencia de trabajo; no se borraron ramas, locks, WIP, procesos o worktrees.

## Corte 2026-10-28 — Revisión V34 desde la base y ejecución atómica

**Base comprobada:** `origin/main = d2759073937585474a14d2f688687c9ba5746e7e`; `HEAD = 4d8db9a345bd1a78f3e479b5beba01bd0124fa3e`. `claude/linea-g-ola10` (11 commits) y `claude/linea-i-i6` (3) se preservan hasta revisión; remoto sólo `main`. QRA mantiene baseline/T1/Risk sin Native PASS; no se duplica su trabajo ni se bajan umbrales. Recursos medidos: `target/` 65 GiB, `data/` 24 GiB, `logs/` 1.5 GiB, `graphify-out/` 817 MiB, `.agent_tx/` 2.9 GiB y `.git/` 3.5 GiB; no se borra por categoría.

**Secuencia aprobada y revisada por builder:** F0 propiedad/ramas; F1 baseline; F2 fees/capital; F3 ignición/OMS/Top 10; F4 aprendizaje online y código fantasma; F5 Binance/red; F6 recursos; F7 causalidad científica; F8 Demo/OOS; F9 publicación y retirada de ramas fusionadas. Los tres primeros átomos son E1 (fees/capital), B1+B3 (venue, entrenamiento y universo) y O1 (lifecycle online), preparados en staging privado y sin promoción hasta que las puertas previas pasen.

**Aceptación prerregistrada:** mínimo 72 h y 20 trades, EV neto positivo, score secuencial ≥100 y crecimiento logarítmico ≥ ln(2)/72 por hora, con OOS y multiplicidad canónica. Se conserva el envelope de riesgo; ningún incumplimiento se compensa con apalancamiento. Disponibilidad de credenciales/entorno continuo: `unverified`; sin ella el trabajo termina en implementación/replay/shadow, no en certificación completa. Producción sigue bloqueada.

**E1 cerrado sólo localmente (2026-10-28 03:50 UTC):** parser de comisiones completo/finito/símbolo exacto compartido; veto por símbolo, precedencia offline parcial coherente, rebates y overflow sin scores inválidos; ruta duplicada de wallet retirada y patrimonio canónico conservado en `OrderExecutor::fetch_account_balance`. Evidencia: quantum 159/159, fees 10/10, ejecución 10/10 + 13/13, datos 12/12, check de tres crates con all-targets y Clippy acotado E1. Selección real HTTP con RED→GREEN y fixture antes bloqueado (timeout 124) ahora acotado (2.03 s). Revisión final GO local sin Critical/Important. Clippy global quantum permanece rojo en baseline ajena. No hay commit, push, CI, Demo ni certificación económica de E1.

**B1+B3 cerrado sólo localmente (2026-10-28 05:27 UTC):** el venue persistido se valida contra el proceso; warmup exige 1000 velas OHLCV válidas por símbolo y falla sin mapa parcial; la fase de entrenamiento ficticia fue retirada; `darwin_approved` depende de warmup y recarga reales; el gestor canónico reconcilia BIN/JSON y reintenta escrituras sin re-suscribir ni inyectar símbolos; fallback y cargador publican exactamente `ACTIVE_UNIVERSE_LIMIT = 10`; el ranker duplicado fue retirado. Evidencia final: warmup 8/8, identidad de universo 2/2, symbol_manager 8/8, cf5 2/2, universo 4/4, ingesta 10/10, recarga 16/16, autorización 5/5, `check` de `god_engine` y Clippy de la librería raíz con `-D warnings` exitosos; formato con ediciones correctas y revisión final GO local. No hay commit, push, CI, Demo ni certificación económica.

**O1 cerrado sólo localmente (2026-10-28 06:16 UTC):** el único worker comparte `shutdown_requested`; el reentrenamiento tiene cancelación cooperativa por etapas y no publica modelos parciales; la evaluación shadow recibe el mismo flag, se espera al salir y su resultado se descarta; el host espera `online_worker` antes de flatten. La revisión agent-28 detectó 2 Important que quedaron cubiertos por nuevos contratos RED→GREEN. Evidencia final: lifecycle 5/5 —reposo, retrain real con 5.000 observaciones, no duplicación, cancelación sin publicación parcial y drenaje de evaluación real—, suite evolution-engine completa verde, Clippy acotado, `check` de `god_engine` y formato con ediciones correctas. Las primitivas externas de entrenamiento no son preemptibles internamente: no se afirma una cota universal de 2 s, sólo la evidencia de estos contratos. `EvolutionLedger` conserva pesos de eventos promote/rollback; no persiste ni restaura el forest completo. Sin commit, push, CI, Demo ni certificación económica.

**F5 cerrado como auditoría acotada (2026-10-28 06:26 UTC):** data-ingest 58/58 y execution-engine lib 194/194 verdes, incluidos transporte HTTP local, cuotas, reintentos y firma; censo de motores 4/4 y XCV 2/2. Se conservan la descomposición red+reloj/cola y las protecciones del exchange. El reporte antiguo de 142 eventos no es comparable con la sonda RTT posterior (p50 ≈231 ms; suelo ≈220 ms), por lo que no se promete mejora DNS. Las nueve dimensiones cross-exchange permanecen muertas por contrato: reactivarlas exige reentrenar y validar OOS, no conectar pollers por catálogo. `live_certifier` no tiene llamador en `god_engine` y no existe `CERTIFICATION_REPORT.json`; ROI/PnL/WR/Sharpe/EV = `N/D`. Disco medido: target 65 GiB, data 24 GiB, logs 1.5 GiB, graphify 817 MiB, agent_tx 2.9 GiB y git 3.5 GiB; no se borra por categoría. Línea I conserva I-3c/I-4/I-5/I-6/I-7 y QRA conserva baseline/T1/Risk/P1. Remoto sólo `main=d275`; ramas G/I preservadas.

**Cierre local y publicación bloqueada (2026-10-28 07:23 UTC):** la revisión integral agent-29 detectó 3 Important ya corregidos con RED→GREEN: (1) `refresh_ml_models` devuelve el éxito real del escaneo/cargas y `darwin_approved` exige `warmup_ready && models_reloaded`; (2) el GBDT se entrena como candidato y sólo publica su estado completo tras el último chequeo de shutdown —el RED real `mixed model after 50ms` ahora pasa en una barrida de cancelaciones—; (3) los outcomes completados se descartan si shutdown está activo. Evidencia nueva: lifecycle 7/7, ingesta 10/10, model_reload 16/16, fases 5/5, suite evolution completa, `check --workspace --all-targets` exit 0 y formato/staging verdes. E1 no recibió hallazgos. El control QRA conserva `native_full_pass=false`, T1/Risk/P1 abiertos: **no hay commit, push, CI, merge ni daemon**. Producción y evidencia económica continúan sin certificar; ROI/PnL/WR/Sharpe/EV=`N/D`.

**Nueva revisión config/lib A1+A2 (2026-10-28 17:28 UTC):** A1 retira `QuantumConfig.min_notional`, campo inerte cuyos consumidores ya usan especificaciones del exchange. A2 hace que `dynamic_ranker` exija filtros completos, finitos y positivos; `tickSize`, `stepSize` y `minQty` son opcionales en deserialización para que un filtro incompleto no aborte todo `exchangeInfo`, y el constructor omite sólo el símbolo afectado en vez de inyectar defaults. Revisión agent-30: 1 Important corregido con RED→GREEN. Evidencia: A1 3/3 y check all-targets; data-pipeline 40 lib +45 integración; symbol_manager 8/8; `god_engine` check; workspace all-targets exit 0; Clippy, formato y staging verdes. **Deuda explícita:** maker/taker VIP0 y `max_leverage` por símbolo no tienen evidencia nueva en este átomo; `god-engine-core/src/lib.rs` permanece bajo lease QRA. Sin commit, push, CI, merge ni daemon; ROI/PnL/WR/Sharpe/EV=`N/D`.

**A3: evidencia de apalancamiento y conservación de filtros (2026-10-28 19:44 UTC):** `dynamic_ranker` ya no fabrica VIP0 ni el mapa 125/75/50: publica sólo `SymbolMarketFilters`. `SymbolLeverage` proviene de `/fapi/v1/leverageBracket` con parser estricto de símbolo, lista y enteros finitos; OMS exige evidencia completa por generación antes de `Ready` y cada entrada revalida universo, registry y caps. `REGISTRY_WRITE_LOCK` serializa escritores; `publish_leverage` valida la sesión bajo su bloqueo y actualiza sólo caps sobre el registry más reciente. La revisión agent-31 cerró stale refresh, carrera registry, bypass de `apply_snapshot` y censos de lecturas; su segunda revisión no encontró Critical/Important. Evidencia RED→GREEN: data-pipeline 41 lib +45 integración; execution-engine 196/196; quantum-arena 154/154; host lib 51/51; host_oms_contract 16/16; registry 2/2 en el árbol real; Clippy de los cuatro alcances con `-D warnings`; `cargo check -p quantum-arena -p data-pipeline -p execution-engine -p trader-gemini-v5 --lib --locked` exitoso en el entregable. `check --workspace --all-targets` quedó **bloqueado por memoria de Windows** (os error 1455/STATUS_STACK_BUFFER_OVERRUN), no certificado. No se ejecutó llamada autenticada real ni una intercalación concurrente específica; compatibilidad externa del cambio de retorno del ranker queda `unverified`. Otros defaults fuera de A3 no se declaran erradicados. Sin commit, push, CI, merge, daemon, Demo/OOS ni certificación económica; ROI/PnL/WR/Sharpe/EV=`N/D`.

**A4 instalado localmente: brackets completos, nocional vivo y admisión atómica (2026-10-29 00:58 UTC):** `SymbolLeverage` conserva `bracket`, floor/cap, MMR y `notionalCoef`; OMS usa `fetch_leverage_evidence`, mientras `fetch_max_leverage` conserva el parser A3. La proyección no usa margen ni `volume_usd * leverage`: calcula `quantity × max(route_price, taker_price)` más exposición same-side a `coin.current_price`, sin netting y sin excluir la pierna opuesta. La reserva propia sólo se excluye con slot+generación+lado y estado no confirmado. OMS normaliza ruta/taker en la costura y reevalúa antes de `configure_leverage` y `submit_entry`; el guard de registry permanece activo hasta después de `in_flight += 1`, con universo y specs bajo el mismo mutex. Se incorporaron los deltas remotos LG-32 (intenciones ausentes) y LG-35 (reloj monótono) sin perder `OrderIdentity`. Revisión final agent-32: GO local, sin Critical/Important.

RED→GREEN real: tipos/parser ausentes; exceso de nocional/leverage y ABA; reserva intercalada; fórmula Maker `900+850>1500`; cambio de precio Market; generación de reserva incorrecta; normalización OMS; writer de universo antes del commit. Evidencia staging: host_oms_contract 29/29; host lib 54/54; quantum 157/157; execution lib 199/199; entry_route 17/17; f325 2/2; data-pipeline lib+integración; `god_engine`, formato y Clippy de cuatro bibliotecas 0. En raíz, los diez archivos son byte-idénticos a staging y repiten quantum 157, ejecución 199, rutas 17, IOC 2, data-pipeline, host lib 54, Clippy, `god_engine` y formato. **Bloqueos:** host_oms_contract no compiló en raíz por memoria/metadatos; `check` de cuatro bibliotecas y `check --workspace --all-targets` tampoco se certifican por memoria de Windows (`os error 1455`, `STATUS_STACK_BUFFER_OVERRUN` y metadata incompleta). La llamada autenticada a `leverageBracket`, permisos/firma/JSON reales y compatibilidad externa de los nuevos campos `post_order_notional`, `taker_price`, `reserved_slot` y `reserved_generation` quedan `unverified`. Deuda residual: fills Market/remanente sin cota absoluta y publicaciones posteriores a admisión comprometida. Sin commit, push, merge, CI, daemon, Demo/OOS ni certificación económica; ROI/PnL/WR/Sharpe/EV=`N/D`.

**A4 re-revisado en staging; instalación pendiente (2026-10-29 01:56 UTC):** se corrigieron tres defectos: el request ya no puede reducir la cota taker viva; toda posición abierta cuenta para el tier y se retiran `reserved_slot`/`reserved_generation` por no acreditar titularidad; los batches parciales válidos fusionan evidencia por símbolo sin borrar la previa. La doble contabilización conservadora de una reserva propia es aceptada. Revisión agent-32: GO local tras añadir la aserción que discrimina la pérdida en un batch parcial válido. Verificación fresca 2026-10-29 02:21 UTC: 498 pruebas sin fallos (host_oms_contract 30/30; quantum 157/157; execution 199/199; entry_route 17/17; IOC 2/2; data-pipeline 41 lib +52 integración por targets), rustfmt y Clippy `-D warnings` de las cuatro bibliotecas, incluido el host. **Bloqueo restante:** `god_engine` y `check --lib` reprodujeron OOM/pagefile y `mmap` (`STATUS_STACK_BUFFER_OVERRUN`). El bloqueo pertenece al perfil `check` bajo presión de commit, no es específico del binario ni evidencia de un defecto A4; no se presenta como PASS ni se repiten intentos ciegos. Diagnóstico posterior: commit de 31,23 GB sobre un límite de 33,70 GB, unos 2 GB virtuales libres y pagefile fijo de 4–8 GB sin administración automática; los fallos de `mmap`/metadatos son colaterales de esa presión, no evidencia de un defecto A4. No había Cargo/Rustc activo propio al medir; no se cerraron procesos ajenos ni se modificó el SO. Por ello el nuevo candidato NO está instalado en raíz ni publicado. La llamada autenticada a `leverageBracket` sigue `unverified`. Sin commit, push, merge, CI, daemon, Demo/OOS ni certificación económica; ROI/PnL/WR/Sharpe/EV=`N/D`.

## Corte 2026-10-27 — Kimi Quant Sr V34, nueva revisión desde la base

**Base comprobada:** `main = origin/main = d2759073937585474a14d2f688687c9ba5746e7e`; el remoto solo contiene `main`. Las ramas locales `claude/linea-g-ola10` y `claude/linea-i-i6` conservan trabajo no fusionado y no se borran. El checkout mezcla WIP de varias sesiones: se integran átomos selectivos, nunca el lote completo.

**Bloqueos heredados que no se relajan:** T1 continúa bajo `0.110/144`; el canario Risk incumple `10000 ns`; P1 snapshot/época/capital/readiness y CL88 permanecen abiertos. El último CI de código observado es `0fc932957dd40766416e93f6e83fe4c74e5750c2` con Unit `40265314105` y Replay `40265314104` en SUCCESS; ese verde no se transfiere a otro sujeto.

**Hallazgos estáticos nuevos, todavía no certificados por runtime:** `IncomeEntry` fabrica fee cero si falta `income`; la cuarentena de income puede perderse al consumir una ventana agotada; UDS mezcla comisión ausente/moneda distinta; la EWMA de riesgo excluye fricción; el MMR usa respaldo ficticio sin productor; el pre-screen evolutivo normaliza con volatilidad OOS futura; un `Option` de cierre por moneda pierde ráfagas; gross/MTM y funding requieren reconciliación contable. Cada hallazgo se atacará con un contrato RED→GREEN y un propietario separado.

**Primer átomo local verificado; publicación bloqueada (Kimi V34):** commit local `13fca0978fa1521f918b2013d486b15b3ff84b58`, limitado a `crates/execution-engine/src/order_types.rs` + nuevo `crates/execution-engine/tests/income_required_field_contract.rs`. Ausencia de `income` se rechaza; cero explícito y rebates se conservan. Evidencia: RED inicial esperado, GREEN 1/1, toda execution-engine 441 PASS/0 FAIL (5 Testnet ignorados por credenciales), Clippy del átomo PASS, formato protegido 589 targets/4 lotes PASS y revisor independiente sin Critical/Important. `origin/main` permanece en `d27590739`; no hubo push ni CI nuevo. La aduana exige attestation de seis fases y el HEAD base no supera Clippy/formato obligatorios; el WIP correctivo pertenece a QRA y no se incluirá artificialmente en el átomo. Esperar baseline QRA, no `--no-verify`. Después siguen, en orden: cobertura/cuarentena de income, moneda/ausencia en UDS, fricción en EWMA, MMR real, recibos brutos/funding, ráfagas de cierres y anti-leakage OOS. No se añaden teorías sin insumo, consumidor y ablación OOS.

## Corte 2026-10-27 — AGENT-01: validación local Zero-Wait; publicación pendiente

`scripts/zero_wait_bus.ps1` vincula el recibo a base, árbol, `CratePath`, conjunto de archivos, SHA256 y blobs. El commit usa sólo el candidato validado, nunca staging ni el índice canónico; si main avanza, exige merge con dos padres, revisión de ambos diffs y revalidación workspace. Rechaza runners de candidato o del `CARGO_HOME`, fija el toolchain del candidato y exige conteo terminal de tests real. El daemon conserva base remota/head por SHA, calcula paths del push completo y entiende workflows YAML/JSON; `CI_NOT_REQUIRED` no equivale a verde.

Evidencia local: PowerShell AST PASS; fixtures de merge con base avanzada, índice compartido byte-idéntico, subconjunto/CratePath/runner externos vetados y test real `tests_passed=1`. `src/live_certifier.rs`, explícitamente Windows-only para no ofrecer un fallback Unix con locks irrecuperables, tiene 11/11 tests con `sccache`, incluidos lock local+Windows, recuperación huérfana, tolerancia dimensional y reporte sin autorización mainnet. Gates Python: 17/17 y 30/30. El alcance CI del átomo income es Replay+Unit; el workflow JSON queda cubierto y un push multi-commit usa base→head, no sólo el último commit. `.cargo/config.toml` mantiene únicamente features medidas AVX2/FMA/BMI/AES/PCLMUL.

El workspace terminó `check --all-targets` y Clippy propio con código 0; agent-22/agent-23 cerraron sin Critical/Important. El candidato selectivo de seis archivos fue detenido por `cargo fmt --check` en fuentes baseline fuera del lote (`continuous_evolution_backtest.rs`, `god_engine.rs`, entre otras). No se ejecutaron Clippy/tests del candidato ni CommitAndMerge: las correcciones de esa baseline pertenecen a QRA y no se apropiarán ni omitirán. El Clippy estricto con dependencias conserva además cuatro `expect()` ajenos en `feature-engine`. Quedan attestation y CI del SHA publicado; no hay commit de este lote, daemon, push, certificación económica real ni autorización `ready_for_mainnet`.

## 0. Contrato de admisión y de no degradación

Todo átomo debe responder antes de editar:

1. **Hipótesis causal:** qué fallo o variable mejora y qué información existe estrictamente en `t`.
2. **Ruta viva:** productor, consumidores y decisión real afectada; un helper aislado no cuenta.
3. **Costo:** RAM, SSD, CPU, latencia y fricción financiera medidos cuando aplique.
4. **Prueba:** un RED conductual previo, el camino real después y la ejecución explícita en CI.
5. **Resultado:** superioridad neta OOS frente al incumbente o deuda declarada con el experimento pendiente.

No se simplifica matemática existente para salir de un error. No se presentan `P(ruina)=0`, picosegundos, nanosegundos ni rentabilidad como hechos sin medición reproducible. Ausencia de dato significa desconocido/`N/D`, nunca cero. No se mezclan balance de wallet, escenario sintético y numerario adimensional.

## 1. Estado medido y sincronización inmediata

| Área | Estado verificado | Consecuencia |
|---|---|---|
| `main` | `4f9ada098`, integrado por fast-forward; los siete archivos locales y el stash histórico fueron preservados | Kimi no toca Rust ni arrastra cambios ajenos al staging |
| CI del último código integrado | `4f9ada098`: **10/10 checks SUCCESS**; Unit `40159701782` y Replay `40159701719`, ambos con salida 0 | Verde acotado al código integrado; el siguiente push se valida por su propio SHA |
| Ramas remotas | `main=4f9ada098` y `claude/auditoria-deslizamiento-apalancamiento-sqtc08=2beda16c3`; PR #140 fusionado y sin PR abierto al corte | La rama activa de Línea C se preserva; no se borra por antigüedad |
| Ramas y worktrees locales | `main`, `glm/162`, `claude/linea-g-ola10`, `claude/linea-i-i6` y `codex/verify-pr65-20261022` bloqueado | Todos se preservan; el stash histórico ajeno también permanece intacto |
| RAM | 23,38 GiB visibles; 0,81 GiB disponibles en la última medición; Cargo y Rustc ajenos activos | No compilar ni editar Rust hasta admisión real del semáforo; no cerrar procesos ajenos para forzar admisión |
| Disco | Tamaño total del proyecto pendiente; no se ejecutan `du` ni barridos durante el build ajeno | No afirmar consumo actual ni limpiar por intuición |
| Hooks | `core.hooksPath=.githooks`; solo existe `pre-commit` de whitespace | `pre-push` sigue pendiente como átomo separado |
| Arquitectura documental | `docs/ARCHITECTURE.md` describe el sistema Python histórico; `ARQUITECTURA_VIVA.md` está sincronizado al 2026-10-06 | El código y el linaje real mandan sobre diagramas históricos |

### Propietarios que no se pisan

- **AGY:** titular propuesto de la implementación N8 al precio límite antes de reservar; Kimi V33 revisa RED/GREEN y Línea C consulta. Fases 10/11 ya fusionadas y sus ramas remotas retiradas.
- **Codex QRA:** PR #141 incorporado; preparación de formato y gates. Sus cambios locales de `user_data_stream.rs`, `os-guardian`, `.agents/MEMORIA.md` y su fila de `AGENT_SYNC.md` no se incluyen en commits de Kimi.
- **Claude Línea I:** worktree `claude/linea-i-i6`; Línea C conserva su rama activa; Línea G conserva su worktree de revisión.
- **GLM:** worktree `tg-mps` en `glm/162`; `verify-pr65` permanece bloqueado.
- **Kimi V33:** revisión N8, este plan, inventario y su propia fila de `AGENT_SYNC.md`; el átomo de capital queda pendiente hasta que la RAM y el semáforo admitan la puerta local.

## 2. FASE 0 — Recuperación, recursos y continuidad multiagente

1. Antes de editar: leer `AGENT_SYNC.md`, registrar heartbeat, archivos (máximo 1–3 por microtarea), rama y huella. Respaldar en `.agent_tx/<agente>/`; nunca copiar el proyecto completo.
2. Un solo `CARGO_TARGET_DIR=<raíz>/target`; builds únicamente mediante `bash scripts/cargo_semaforo.sh ...`, con `CARGO_BUILD_JOBS=2` y al menos 4 GiB disponibles. Salida 75 significa sin turno, no permiso para bypass.
3. No lanzar `du`, higiene ni barridos mientras otro agente compila. Medir primero y aplicar limpieza solo con inventario, cerrojo propio y costo de recompilación aceptado. `data/` y `models/` son intocables.
4. No matar procesos por nombre. Verificar PID + `StartTime` + titularidad. Los procesos `du`/`sort` iniciados por Kimi ya terminaron; no se detuvo ningún proceso ajeno.
5. Cuando no haya ventana local, no dejar ediciones Rust sin verificar en el checkout compartido. La falta de RAM bloquea el push de código; no convierte CI en excusa para saltarse la puerta local exigida por esta revisión.
6. Failover: conservar primero el trabajo; completar o restaurar según claridad, certificar y actualizar bitácora. Ningún worktree activo o bloqueado se elimina por antigüedad sin aplicar el procedimiento completo.

**Salida:** máquina estable, bitácora actualizada, cero procesos propios huérfanos, ningún artefacto borrado por intuición.

## 3. FASE 1 — Ramas, CI y puertas de publicación

1. Revalidar refs antes de cualquier higiene: `git fetch --prune`, `git status`, ancestria y `git rev-list --left-right --count origin/main...origin/<rama>`. El lado derecho debe ser cero y la rama no debe tener reserva.
2. V33 retiró las ramas remotas `agy/fase10-sbe-helmholtz-hodge` y `agy/fase11-feynman-instantones` tras confirmar PR #135/#137 fusionados, cero commits exclusivos, ancestria completa y ausencia de worktree. El remoto quedó con `main` y el PR #138 activo. Los cinco worktrees locales y el stash ajeno se preservan; no hay borrado local automático pendiente.
3. Diagnosticar siempre el primer fallo dominante con `gh run view --log-failed`; no cambiar de tarea hasta repararlo o dejar bloqueo explícito.
4. Crear `.githooks/pre-push` como átomo independiente: procesar stdin y validar el SHA exacto recibido, distinguir borrado/docs/código, rechazar pesos/prohibidos y ejecutar la validación Rust por semáforo. Añadir pruebas herméticas y conectarlas al job `resource-safety`.
5. Antes de commit: `git status`, diff staged y `git add` solo de rutas propias. Separar docs de código. Después de push: `gh run watch <id> --exit-status` y confirmación de `headSha`; todos los workflows disparados deben quedar verdes.

**Salida:** ningún merge por ancestria parcial, ningún check heredado y ningún hook que certifique un árbol distinto del SHA publicado.

## 4. FASE 2 — Primer átomo de código: procedencia de capital e informe

**Archivos previstos:** `crates/backtest-engine/src/bin/continuous_evolution_backtest.rs`, `tests/continuous_capital_contract.rs` y `.github/workflows/replay-contracts.yml`. Aunque la fuente vive bajo `crates/backtest-engine`, el binario está declarado en el paquete raíz `trader-gemini-v5`; no se usará `-p backtest-engine`.

**Fallos actuales:** capital fijo 13 USD; `equity_curve.csv` se crea/trunca antes de validar configuración; el crecimiento vuelve a usar 13 USD; el binario promueve al final sin que este átomo certifique una puerta de rendimiento.

**Alcance estricto del primer cambio:**

1. Validar `INITIAL_CAPITAL` como primera sentencia del cuerpo de `main` (actualmente en `continuous_evolution_backtest.rs:78`), antes del primer `set_var` (`:82`), crear archivos, leer ticks o genomas. Ausencia, vacío, texto inválido, NaN, ±Inf, cero y negativos retornan error no cero sin efectos laterales.
2. Extraer funciones puras para parseo y crecimiento; el informe real debe llamarlas. Eliminar la redeclaración `initial_capital = 13.0` posterior para que no sombree la variable validada. Usar esa única variable en simulación, “Capital Inicial” y crecimiento, etiquetado como escenario backtest, no wallet.
3. RED en dos capas con certificados explícitos:
   - **Aritmética pura dentro del binario:** `Some(" 20.0 ")` es válido; ausente/vacío/malformado/NaN/±Inf/cero (incluido `-0.0`)/negativos se rechazan; `growth_percent(20.0, 30.0) == 50.0`. Un test de integración no puede importar funciones de un binario. Sin cinta no se puede producir el informe completo: este caso demuestra la fórmula, **no** una corrida E2E ni ROI financiero.
   - **Proceso real:** en `tests/continuous_capital_contract.rs`, ejecutar `env!("CARGO_BIN_EXE_continuous_evolution_backtest")` con `std::process::Command` y cwd temporal distinto por hijo. Para el caso ausente usar `.env_remove("INITIAL_CAPITAL")`; fijar `SIM_DAYS=1`, `MULTI_COIN=0`, `TG_GENOME_ENV=backtest` y tanto `TGM_CONFIG_DIR` como `TG_CONFIG_ROOT` dentro del fixture, sin `set_current_dir` ni `set_var` globales. Exigir salida no exitosa y diagnóstico de `INITIAL_CAPITAL`, no aceptar un pánico cualquiera. Comparar bytes de un CSV centinela, cubrir también la variante sin CSV y exigir que `config_dir` siga ausente o que todo su árbol quede idéntico: la configuración inválida no debe crear, truncar ni promover nada. Los tests puros no sustituyen esta prueba de arranque.
4. Conectar explícitamente en Replay el binario y su contrato:
   `cargo +nightly-2026-06-30 test -p trader-gemini-v5 --bin continuous_evolution_backtest --test continuous_capital_contract --locked -- --test-threads=1`.
   Verificar nombres y número de tests en el log; `check --all-targets` solo compila.
5. Puerta local cuando el semáforo admita: fmt, clippy del binario con `-D warnings`, el comando anterior y `check --workspace --all-targets --locked`.

**Fuera de este átomo:** promoción por rendimiento, wallet API y los fallbacks de `evolver`, `evolution`, `multi_coin_simulator`, `benchmark` y `ReplayConfig`. Quedan registrados, no ocultos.

## 5. FASE 3 — Correctness financiera y evolutiva en la ruta viva

Orden de ataques posteriores, cada uno con su propio RED y reserva. Los seis primeros proceden de la revisión estática cruzada; aún no tienen pruebas ejecutadas por V33. Antes de tocar `reconciliation`, host o ejecutor se coordinará con Línea C/AGY/Sol para no invadir sus reservas:

**N8 / IOC bajo `minNotional` ya asignado:** la revisión `main@5e3557c17` confirma que el mínimo se garantiza al precio medio/entrada modelada, pero el IOC final queda en `4.1 × 1.219 = 4.9979 USDT < 5`. **AGY implementa, Kimi V33 revisa y Línea C consulta**: dimensionar la mínima al precio límite antes de publicar la reserva, sin aumentar cantidad después; RED debe exigir `qty=4.2`, `notional=5.1198` y veto previo a reserva/configuración/envío cuando la mínima no cabe en el riesgo aprobado. Alcance máximo: `god-engine-core/src/lib.rs`, la costura IOC de `src/bin/god_engine.rs` y `execution-engine/tests/f325_min_notional_ioc_open_diagnostics.rs`. No mezclar N12/fricción ni salidas reduce-only.

1. **OMS ante exposición desconocida (crítico):** `positionRisk` puede deserializar campos ausentes como cantidad `0.0` y llegar a `Ready` con `openOrders=[]`. Exigir campos de evidencia y `InvalidRemoteEvidence`/`Blocked`; probar que una fila sin `positionAmt` no abre admisión ni cierra una posición local como fantasma.
2. **Cierre sin fill (crítico):** el host puede aprender/contabilizar antes de confirmar la salida y reduce-only acepta ACK `FILLED`/`PARTIALLY_FILLED` con `executedQty=0`. Exigir ejecución positiva y reconciliación; no purgar OCO ni llamar `record_net_pnl` con una salida desconocida.
3. **Equity y funding:** `ACCOUNT_UPDATE` alimenta otro plano atómico que el usado por sizing; el funding realizado no entra en el cierre que aprende. Una equity real inferior debe limitar margen antes de la siguiente entrada y el funding debe figurar en el neto con procedencia explícita.
4. **TP maker ficticio frente a taker real:** el núcleo clasifica la salida TP como maker mientras host/brackets usan MARKET/TAKE_PROFIT_MARKET. Alinear precio y fee con el transporte real y contrastar el contrato existente de fees.
5. **PnL desconocido:** `user_data_stream.rs` puede transformar `entry_price=0` en `0 - fees` y activar evidencia independiente. Rechazar ese registro económico, conservar diagnóstico y admitir breakeven genuino con entrada/salida positiva. La cuantización tampoco puede aumentar la cantidad aprobada ni dejar el nocional por debajo del mínimo.
6. **Fee histórico desconocido:** una posición adoptada nace con `entry_fee=0` y después se presenta como confirmada. Sin evidencia nativa no puede alimentar Kelly como si el fee fuese cero.
7. **Backpressure:** canales no acotados en `god_engine.rs` y `trade_accounting.rs`; descarte invisible del canal independiente. Acotar, contar saturación, impedir que la muestra se considere completa y probar recuperación sin duplicados.
8. **ADN coherente:** publicación perdedora que devuelve éxito, aplicación de un candidato no activo y preferencia de `active.bin` viejo sobre JSON nuevo. Exigir resultado observable y versión consistente o fallo cerrado.
9. **Evolución con fills reales:** `last_closed_trade` sobrescribe cierres, faltan snapshots de apertura y el canal independiente no alimenta RF/GBDT. Dos cierres antes del muestreo deben producir dos observaciones causales; sin snapshot no se ingiere.
10. **Paridad de evaluación:** diezmado `msg_count % 10`, envolvente/riesgo/exchange ausentes en ShadowForest, macro ficticia en walk-forward y fitness rotulado como dólares. La cosecha no puede promover trayectorias que la operativa habría rechazado.
11. **Anti-leakage:** volatilidad calculada con toda la serie, validación solapada con selección y micro-ticks que conocen el cierre. Dos tapes idénticos hasta `k` y con futuro distinto deben generar las mismas decisiones hasta `k`.
12. **Procedencia del ADN:** `backtest_heritage` cargado en Prod, SA que promueve directo y papel/Testnet compartiendo silo. La validación numérica del sobre no prueba promoción humana/OOS; separar entornos y exigir linaje autorizado.

**Regla:** no resolver estas familias con una reescritura grande. Una ruta, un fallo, un contrato y un recibo por ciclo.

## 6. FASE 4 — Fees, OMS, red, datos y recursos adaptativos

1. Medir esperanza neta después de comisión real por símbolo/sesión, funding, spread, slippage y selección adversa; Maker no equivale a fill gratis ni garantizado.
2. Exigir reconciliación OMS completa antes de admisión: balance real Demo/Prod, posiciones, órdenes, mismo ejecutor/entorno y cero incidencias no resueltas.
3. Línea I lidera la descomposición `recv−E` frente a espera local; otros agentes no duplican su bloque. Después medir p50/p95/p99, freshness, cola, pérdidas y reconexiones por segmento.
4. Auditar `reqwest`/WS y duplicados con metadata (`Cargo.lock`/`cargo tree`). El inventario V33 encontró 40 paquetes duplicados (`docs/audit/V33_DEPENDENCIAS_2026-10-24.md`): `telemetry-server` usa Axum/Hyper 1.10/tungstenite 0.24, mientras warp/reqwest usan Hyper 0.14/tungstenite 0.21. Unificar solo si la convergencia de frameworks demuestra beneficio y conserva comportamiento; no hacer bumps ciegos ni añadir dependencias por costumbre.
5. Acotar buffers y persistencia; restaurar el último estado al recuperar. Ningún drop, canal desconectado o dato obsoleto puede quedar invisible.
6. Validar continuidad temporal, NaN/Inf, partial fills, reduceOnly y Mark/Last Price antes de optimizar instrucciones o silicio.

## 7. FASE 5 — Demo, métricas y evidencia económica

No lanzar Producción. Demo solo con RAM suficiente, credenciales Testnet disponibles y las puertas anteriores:

1. Confirmar endpoint Testnet, origen del balance y reconciliación OMS antes de admitir.
2. Registrar inicio/fin, dataset/ticks, fills, endpoint, modo, commit y configuración.
3. Reportar obligatoriamente **ROI bruto/neto, PnL bruto/neto, WR antes/después de costos, Sharpe y EV/trade**, con fees, funding y slippage.
4. Separar latencia p50/p95/p99 por segmento, freshness, profundidad de cola y rechazos.
5. Si no hay fills o muestra suficiente, escribir `N/D` y “no certificado”. Arrancar no es certificar. Un backtest no prueba Demo y Demo no prueba Producción.

## 8. FASE 6 — Matemática de frontera con admisión científica

Las teorías no se integran por catálogo o prestigio. Se prioriza por utilidad neta medible:

1. Corrección causal y estabilidad numérica de lo ya conectado.
2. Persistencia y aprendizaje online real, con anti-leakage y estado recuperable.
3. Microestructura, fricción, liquidez y riesgo de cola pesada.
4. Koopman/DMD, rough paths, Tsallis, Wasserstein/TDA y MPS/DMRG solo con insumo y consumidor verificados.
5. MFG/HJB/Navier-Stokes y demás teorías avanzadas solo si superan al incumbente en A/B o ablación OOS dentro del presupuesto de recursos.

Cada nueva teoría exige: definición operativa, estimadores, guardas NaN/Inf, causalidad, complejidad de tiempo/memoria, consumidor en la decisión, prueba de solución conocida, datos reales OOS, ablación y recibo de impacto. Si no hay evidencia, permanece como investigación, no como verdad operativa.

## 8B. Backlog científico V33.1 — mandato de fases 28/29

Estas familias se incorporan a la cola, **no** a la lista de capacidades certificadas. Fase 9 (Marchenko-Pastur/Wilson/Lyapunov) fue integrada posteriormente en `main` como `8c96ff25a` y su rama remota fue retirada tras verificar ancestria y ausencia de worktree. La revisión previa había confirmado su ausencia, por lo que no se le atribuía integración ni OOS. Su CI candidata terminó 10/10; la CI del push de `main` se evalúa por separado. La existencia de módulos no demuestra consumidor ni superioridad: esos puntos siguen en el backlog de linaje y ablación.

1. **Motor de capital:** descomposición logarítmica de la meta, Grinold-Kahn con apuestas efectivas y Grossman-Zhou. Aceptación: crecimiento neto OOS con intervalos de confianza, límites de liquidez y drawdown; nunca invertir la fórmula para forzar trades o apalancamiento. Las cifras de bps por ciclo son hipótesis, no expectativa.
2. **Ejecución multi-riel:** libro implícito triangular, idempotencia `newClientOrderId`, predictor de Smith, socket sharding y numerario anti-depeg. Aceptación: inventario real de mercados y fees, paridad de filtros/cantidades, cola FIFO y selección adversa; prueba de que reintentos no duplican ni provocan rechazos inesperados; medición por segmento de latencia, sin llamar “exacto” a RTT/2. Un libro sintético no crea liquidez gratuita.
3. **Aprendizaje online:** Cover-Vovk y potenciales de Orabona. Aceptación: derivación correcta del potencial y del arrepentimiento, implementación con guardas numéricas, comparación contra el incumbente y ablación. No se afirma best-in-hindsight ni eliminación de todos los hiperparámetros sin demostrar el alcance del teorema.
4. **Modelos de localización y campo:** Filoche-Mayboroda, nemáticos activos y Wetterich FRG. Aceptación: operadores discretos bien definidos, solución conocida, interpretación física de parámetros, datos suficientes y superioridad neta OOS; no se adopta por analogía estética.
5. **Optimización y muestras finitas:** Hadamard/geodésica y Malliavin-Stein. Aceptación: demostrar geodésica-convexidad real del objetivo (no suponerla por el espacio), cotas aplicables al estimador y robustez ante muestras cortas, con control de colinealidad y multiplicidad.
6. **Silicio y recursos:** IMC throttling, SIMD-DFA, RNS, Elias-Fano y compresión. Aceptación: benchmark antes/después con latencia p50/p95/p99, RAM/SSD, exactitud bit a bit y fallback cuando la CPU no soporte la instrucción. La velocidad nunca sustituye validación ni reduce el presupuesto de 16 GB.

Ningún elemento de este backlog justifica prometer +100 %/72 h, ruina cero, 0 ns o fees insignificantes. Cada candidato entra primero como medición desacoplada y solo avanza con consumidor, causalidad y mejora neta OOS.

## 8C. Gobernanza anti-cosmética, tres relojes y ejecución de próxima generación (fase 30)

Estas reglas elevan el contrato de admisión, **no** certifican que las capacidades ya existan ni autorizan a activarlas. Toda implementación debe demostrar la ley que dice implementar, su procedencia de datos y su consumidor en la decisión real.

1. **Puerta de invariantes analíticos exactos:** cada motor nuevo o modificado debe probar la invariante correspondiente a su teoría, no una aproximación con el mismo nombre. Para Lindblad, traza uno y positividad del estado; para haces/Hodge, `δ²=0` y ortogonalidad; para integradores simplécticos, `JᵀΩJ=Ω` (la conservación `det(J)=1` es consecuencia, no sustituto de la estructura); para PDEs, convergencia contra solución analítica y orden esperado. Añadir dimensiones, estabilidad NaN/Inf, fronteras y casos degenerados. El test debe cruzar la ruta productor→decisión→efecto; un test aislado con el nombre de la teoría no basta.
2. **Tres relojes espectrales:** Tier-0 queda limitado a operaciones acotadas demostradas en el hot path; Tier-1 a filtros de streaming; Tier-2 a física profunda y evolución. Publicar solo instantáneas coherentes de un escritor mediante SeqLock, sin bloqueos en el lector y sin valores parciales. Aceptación: p50/p95/p99 por anillo, frescura, cola, RAM, canary de linaje extremo a extremo y fallback de CPU. No afirmar `<40 ns`, picosegundos ni swap JIT seguro antes de medirlos; el intercambio de ecuaciones exige versión validada y rollback.
3. **Inferencia temprana de llenado propio:** cola FIFO y flujo público de trades/depth se evalúan primero como **sombra**. No confirmar fill, aprenderlo dos veces ni disparar TP/cobertura con una estimación no certificada. Aceptación: replay con posición en cola conocida, selección adversa, fills parciales, paquetes ausentes y reconciliación idempotente con `ORDER_TRADE_UPDATE`. Solo tras paridad Demo se evalúa actuación temprana, nunca como sustituto del user data stream.
4. **Margen multi-activo y comisiones:** `multiAssetsMargin`, `commissionRate`/BNB, colateral multi-moneda y tensor de liquidaciones requieren descubrimiento de capacidades y compatibilidad de cuenta, balance real, numerario anti-depeg, escalones MMR y prueba de rollback. No activar modos por literal ni afirmar cero comisiones. Certificar en Demo reconciliación de wallet/margen/liquidación antes de cualquier Producción.
5. **Backlog matemático nuevo:** Wasserstein-Fisher-Rao, Eyring-Kramers, Ramanujan/expander, conexiones duales de Amari y Sonine/Krasovskii. Cada candidato exige mapeo operacional, estimador estable, unidades, modelo nulo, causalidad, costo de tiempo/memoria, comparación con incumbente y ablación OOS. El nombre de la geometría no demuestra ventaja predictiva ni económica.

## 9. Censo archivo por archivo y recepción entre agentes

El inventario mecánico `docs/audit/V33_INVENTARIO_2026-10-24.tsv` conserva las 2.186 rutas de la base de auditoría `2fb4633d3` y los deltas posteriores hasta `main=4f9ada098`: **2.203 rutas**, 26 `REVISADO_ESTATICO`, 1 `RED` (capital inválido con evidencia ejecutada) y 2.176 `INVENTORIADO`. La revalidación actual añadió `crates/data-ingest/tests/inmune_latencia_contract.rs` y `docs/audit/linea_i/rtt_probe.ps1`, sin bajas; el conjunto completo fue comparado en orden y sin duplicados contra `git ls-files`. Las filas históricas conservan el blob de su base y las altas llevan el blob del índice vigente, como aclara la cabecera. No es una certificación de revisión completa. El ledger incremental `docs/audit/V33_REVISIONES_2026-10-24.tsv` conserva los 34 hallazgos concretos, con N8 revalidado y asignado, sin reescribir censos históricos. Campos mínimos:

`ruta | blob | dueño | ruta_viva | hallazgo | severidad | prueba | CI_SHA | estado | siguiente`

Estados permitidos: `INVENTORIADO`, `REVISADO_ESTATICO`, `RED`, `VERDE_LOCAL`, `INTEGRADO_CI`, `BLOQUEADO`, `SIN_CAMBIO_JUSTIFICADO`. Ningún archivo pasa a `INTEGRADO_CI` por revisión de otro agente o por pertenecer a un crate verde.

**Protocolo de cada microciclo:** sincronizar → reservar → respaldar → RED → cambio mínimo → pruebas → integración → CI del SHA → recibo → liberar reserva. Si la cuota se agota, el estado queda escrito y otro agente continúa sin adivinar.

## 10. Puertas locales y remotas de referencia

Cuando la RAM lo permita:

```bash
cargo fmt --all -- --check
bash scripts/cargo_semaforo.sh clippy -p trader-gemini-v5 --bin continuous_evolution_backtest --locked -- -D warnings
bash scripts/cargo_semaforo.sh test -p trader-gemini-v5 --bin continuous_evolution_backtest --test continuous_capital_contract --locked -- --test-threads=1
bash scripts/cargo_semaforo.sh check --workspace --all-targets --locked
cargo tree --workspace --duplicates --locked
```

Después de cada push de código: todos los checks disparados (base actual: 8 de Unit + 2 de Replay, más workflows de dominio), verificados contra el SHA publicado. Documentación sola sigue la exención de CI ya existente, pero no se utiliza para ocultar código en un commit documental.


## Corte V35-F1R3 — formato, Fisher-Rao y attestation pendiente (2026-11-04T19:20:56Z)

**Estado:** CAS local completado en `b8529f40ec7ee7681f68b495003c3b10eaabc301`; publicación no autorizada por attestation incompleta. `origin/main=f1ca374aa00d1c84124ae803799db5c0b4fa4b95` permanece sin cambios.

1. Se integraron cuatro átomos exclusivamente de formato (`fac7fbe0`, `d6e1896b`, `41d7c615`, `c362239b`) en siete archivos de audit-engine y dos de backtest-engine. No se alteraron reglas, umbrales, contabilidad, OOS ni APIs; QSR/QRA permanecen intactos.
2. El candidato tiene tree `c00748bdfaab9e640e9fd327dfd474ffa6cccfdb`, diff-check contra ambos padres PASS, Fisher-Rao 4/4, workspace all-targets exit 0, Zero-Wait 22/22 y backtest 43/43 +2 ignored. Revisión agent-25: **Yes**, sin Critical/Important.
3. No se hereda ese verde parcial como attestation. Permanecen bloqueantes: `navier_stokes::update` y `adversarial_gate` con llamadores QRA, familia legacy `test_pre_push_gate.py`, formatter baseline y las seis fases + contratos Python/Golden9 sobre un único SHA final. Los warnings MXCSR y errores no fatales de checksum dep-info permanecen registrados.
4. Siguiente acción: QRA aplica o cede explícitamente sus alcances; continuar formato sólo con propiedad verificada; repetir el sobre completo antes de cualquier push. Producción NO-GO; ROI/PnL/WR/Sharpe/EV=N/D. +100 %/72 h es hipótesis a falsar con evidencia OOS neta.

## Corte V35-R4/R5 — audit-engine formateado e integrado localmente (2026-11-04T19:42:25Z)

- `main` local=`dbdd79f7bb255347fc5a45a3fbbd1a08e8c45729`, tras CAS desde `b8529f40`; `origin/main=f1ca374a` sin cambios, sin push.
- R4 `950882897` y R5 cierran `rustfmt --check` de todo `audit-engine` exclusivamente con cambios de formato. Cada suite: 90/90 en 14 targets, sin ignorados ni filtrados. Workspace exit 0, Zero-Wait 22/22 y revisión agent-25 **Yes** sin hallazgos.
- Estos recibos no son attestation global. Permanecen QRA Navier/adversarial/legacy, formatter de otros crates, seis fases, Python legacy y Golden9 del SHA final. Producción NO-GO; métricas económicas N/D.

## Corte V35-R6 — revisión causal config/lib y evidencia de fricción (2026-11-04T19:58:07Z)

1. **Revisión desde base:** `dbdd79f7b` y archivo inmutable R5; ramas/agentes/WIP preservados. La constante `CANONICAL_COUNCIL_MAX_DRAWDOWN` está derivada de Cramér-Lundberg y no se considera hardcode arbitrario. El balance 13 USD no aparece como literal en las rutas config/lib inspeccionadas.
2. **Defecto concreto seleccionado:** `route_order` fabrica maker 2 bps/funding 1 bp cuando faltan arena, símbolo o libro; `expected_funding_cost` devuelve 0.0 sin funding. Esto puede cambiar admisión y PnL.
3. **Átomo autorizado:** sólo `execution-engine/src/router.rs` y `tests/adverse_selection_viability_contract.rs`. Ausencia ⇒ error tipado; evidencia estricta de funding ausente ⇒ error. Escenarios explícitos permanecen compatibles.
4. **Prueba y entrega:** RED antes del cambio, GREEN, toda execution-engine, workspace, revisión independiente y CAS local sólo con recibos. Attestation/push siguen bloqueados por QRA, formatter restante y seis fases/Golden9. Producción NO-GO; ROI/PnL/WR/Sharpe/EV=N/D.

## Corte V35-R6A — evidencia de fricción fail-closed integrada localmente (2026-11-04T20:42:02Z)

- `main` local=`5e021ebb9d17cf5db586b023f3d414cba12be306`, tras CAS desde `dbdd79f7b`; `origin/main=cfe08162` sin cambios, sin push.
- `route_order` deja de fabricar maker/funding por defecto: ausencia de arena/símbolo/tarifa ⇒ `REJ_FEE_EVIDENCE_MISSING`; funding requerido ausente/no finito ⇒ `REJ_FUNDING_EVIDENCE_MISSING`, antes de cualquier envío. Escenarios explícitos permanecen compatibles.
- Evidencia: contratos 11/11, ejecución 514/514 en 49 suites, workspace exit 0, Zero-Wait 22/22, revisión agent-25 **Yes** sin defectos. La cadena truncada anterior fue reconstruida y no integrada; blob final `37f7b3a7`.
- Attestation/publicación siguen bloqueados por QRA, formatter restante y seis fases/Golden9. Producción NO-GO; métricas económicas N/D.

## Corte V35-R7 — evidencia de señal finita antes de ejecución (2026-11-04T21:09:55Z)

- Base `5e021ebb`; R6A queda integrado localmente y no se reabre. QRA sigue sin cesión y sus archivos no se tocan.
- `route_order` admite actualmente NaN/Inf en confianza, volatilidad y spread porque `evaluate_viability` los reemplaza por heurísticas. Se exige evidencia finita y producto de edge finito antes de cualquier envío, con `REJ_INVALID_SIGNAL_EVIDENCE`.
- Átomo sólo en `router.rs` y `adverse_selection_viability_contract.rs`. RED de tres entradas corruptas, GREEN, toda execution-engine, workspace, Zero-Wait, revisión y CAS local con recibos. Sin push/attestation. Producción NO-GO; ROI/PnL/WR/Sharpe/EV=N/D.

## Corte V35-R7B — evidencia finita de señal integrada localmente (2026-11-04T22:02:00Z)

- `main` local=`6f95120f0cafefc0c857e062f6d1141411f6a836`, tree `f7966b523fa2d09cb96d77e1b840c7a5fb78988d`, tras CAS desde `5e021ebb`; padres `ead6d683` y `9bbb0b99`. `origin/main=ead6d683` revalidado, sin push.
- La guarda `REJ_INVALID_SIGNAL_EVIDENCE` exige confianza finita en [0,1], volatilidad/spread finitos no negativos y producto edge finito antes de fee/IOC/Market. Flat retorna antes; R6A y el reúso del edge validado permanecen intactos.
- R7A no fue promovido porque upstream avanzó a GLM-192. Se adopta `is_nan()` de `ead6d683`; el `to_bits()` duplicado se descarta. Frente a R7 sólo se añaden los dos archivos Fisher upstream.
- Evidencia terminal: ejecución 517/517 en 49 suites/5 ignored/0 filtered, Fisher 4/4, workspace exit 0 (4m07s), Zero-Wait 22/22 (189.610s), revisión agent-25 **Yes** sin Critical/Important/Minor.
- Handoff documental separado pendiente de Zero-Wait y CAS. Attestation sigue bloqueada por QRA Navier/adversarial/legacy, formatter restante, productor ROOT registrado, seis fases + Python legacy/Zero-Wait/bus y Golden9 del SHA final. Producción NO-GO; métricas económicas N/D; +100 %/72 h sigue siendo hipótesis OOS neta.


## Corte V35-R8 — dark-alpha formateado e integrado localmente (2026-11-04T22:34:38Z)

- `main` local=`92950e1b832ea562675620151fcd4cff1714c099`, tree `44a2810d`, tras CAS desde `13b3cc0d`; `origin/main=ead6d683` sin cambios, sin push.
- Nueve rutas `dark-alpha-engine` quedan en rustfmt con equivalencia byte a byte contra los nueve blobs WIP censados. No se adoptó ningún otro WIP ni se alteró comportamiento, APIs, umbrales o contabilidad.
- Evidencia: 66/66 en 6 targets, workspace 4m12s, Zero-Wait 22/22 (267.253s), revisión Yes sin defectos. `core.fsmonitor` se desactiva sólo en índices privados antes de staging; la configuración global no cambia.
- Handoff separado pendiente. Attestation sigue bloqueada por QRA Navier/adversarial/legacy, formatter restante, productor ROOT, seis fases + Python y Golden9. Producción NO-GO; métricas económicas N/D.
