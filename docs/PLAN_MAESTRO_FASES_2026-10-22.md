# Plan maestro por fases V33 — revisión desde la base

**Revisión vigente:** 2026-10-24, Kimi Quant Sr V33, en coordinación con los demás agentes.
**Base inspeccionada:** `main = f6436716bee5c953d497d70d2104baf6c4a859cb`.
**Alcance:** workspace Rust de 21 crates + paquete raíz, 2.175 rutas versionadas y 829 archivos Rust. Estas cifras describen el censo mecánico, **no** archivos ya auditados.
**Objetivo de investigación:** crecimiento neto compuesto de +100 %/72 h. Es una hipótesis que debe demostrarse fuera de muestra y después de fees, funding y slippage; no una garantía ni una razón para aumentar riesgo. Producción permanece prohibida en esta revisión.

Este es el plan canónico de coordinación. Los planes de líneas aportan detalle, no fuentes paralelas de verdad. `AGENT_SYNC.md` contiene reservas y heartbeats; `.agents/MEMORIA.md` conserva los recibos de cierre.

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
| `main` | `f6436716b`, actualizado por fast-forward; solo `.workbuddy-ai/` untracked y ajeno | No hay cambios propios de código en el checkout |
| CI del último código integrado | Fase 6 `e83be23c5`: Unit `40118206857` y Replay `40118206773`, ambos SUCCESS; el push posterior `f643` es solo docs | No se transfiere ese verde a commits futuros |
| Ramas remotas | Solo `main`, `agy/fase7-ergocicidad-rough-paths` y `sol/f1101-snapshot-coherente` | Fase 5/Fase 6/Línea C ya fueron retiradas por terceros; no repetir borrados |
| PR #128 / Fase 7 | Nuevo SHA `a0e15be457` tras el E0689 inicial de `rough_ergodic.rs`; CI del nuevo candidato en curso al corte | AGY conserva titularidad; no editar |
| PR #129 / Sol F1-101 | Falló `test_profiler_financials_atomics_end_to_end` por mezcla de ticks; Sol ya publicó `45b319abb` para corregir el contrato | No tocar profiler ni la instantánea financiera |
| RAM | 23,38 GiB visibles; lecturas de 1,03 GiB libre, luego 368 MiB y 699 MiB disponibles; cargo/rustc ajenos vivos | No compilar ni editar Rust en el checkout compartido hasta admisión del semáforo |
| Disco | Volumen C: 305 GiB libres a las ~13:10; tamaño total del proyecto pendiente porque la medición se canceló para no competir con el build | No afirmar consumo actual ni limpiar por intuición |
| Hooks | `core.hooksPath=.githooks`; solo existe `pre-commit` de whitespace | `pre-push` sigue pendiente |
| Arquitectura documental | `docs/ARCHITECTURE.md` describe el sistema Python histórico; `ARQUITECTURA_VIVA.md` está sincronizado al 2026-10-06 | El código y el linaje real mandan sobre diagramas históricos |

### Propietarios que no se pisan

- **AGY:** Fase 7/rough paths y su worktree.
- **Sol:** F1-101, instantánea financiera coherente y su worktree.
- **Claude Línea I:** descomposición de latencia del feed y reloj/cola.
- **Claude Línea G:** revisión estática de commits; Línea C: censo/retiros en su carril de nube.
- **GLM:** trabajo local `glm/160`; `verify-pr65` permanece bloqueado.
- **Kimi V33:** este plan, `AGENT_SYNC.md`, `docs/ARCHITECTURE.md`; después, un átomo de capital en backtest continuo cuando la RAM lo permita.

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
2. Las tres ramas inicialmente candidatas ya desaparecieron del remoto por acción de terceros. No hay borrado V33 pendiente. Las dos ramas activas y todos los worktrees se preservan.
3. Diagnosticar siempre el primer fallo dominante con `gh run view --log-failed`; no cambiar de tarea hasta repararlo o dejar bloqueo explícito.
4. Crear `.githooks/pre-push` como átomo independiente: procesar stdin y validar el SHA exacto recibido, distinguir borrado/docs/código, rechazar pesos/prohibidos y ejecutar la validación Rust por semáforo. Añadir pruebas herméticas y conectarlas al job `resource-safety`.
5. Antes de commit: `git status`, diff staged y `git add` solo de rutas propias. Separar docs de código. Después de push: `gh run watch <id> --exit-status` y confirmación de `headSha`; todos los workflows disparados deben quedar verdes.

**Salida:** ningún merge por ancestria parcial, ningún check heredado y ningún hook que certifique un árbol distinto del SHA publicado.

## 4. FASE 2 — Primer átomo de código: procedencia de capital e informe

**Archivos previstos:** `crates/backtest-engine/src/bin/continuous_evolution_backtest.rs`, `tests/continuous_capital_contract.rs` y `.github/workflows/replay-contracts.yml`. Aunque la fuente vive bajo `crates/backtest-engine`, el binario está declarado en el paquete raíz `trader-gemini-v5`; no se usará `-p backtest-engine`.

**Fallos actuales:** capital fijo 13 USD; `equity_curve.csv` se crea/trunca antes de validar configuración; el crecimiento vuelve a usar 13 USD; el binario promueve al final sin que este átomo certifique una puerta de rendimiento.

**Alcance estricto del primer cambio:**

1. Validar `INITIAL_CAPITAL` como primera operación de `main`, antes de `set_var`, crear archivos, leer ticks o genomas. Ausencia, vacío, texto inválido, NaN, ±Inf, cero y negativos retornan error no cero sin efectos laterales.
2. Extraer funciones puras para parseo y crecimiento; el informe real debe llamarlas. Usar la misma variable validada en simulación, “Capital Inicial” y crecimiento, etiquetado como escenario backtest, no wallet.
3. RED conductual: inicial 20/final 30 ⇒ 50 %. Casos ausente/vacío/malformado/NaN/±Inf/cero/negativo/válido distinto de 13. En `tests/continuous_capital_contract.rs`, ejecutar el binario real con `CARGO_BIN_EXE_continuous_evolution_backtest` y un CSV/config_dir centinela: la configuración inválida no debe crear, truncar ni promover nada. Los tests puros no sustituyen esta prueba de arranque.
4. Conectar explícitamente en Replay el binario y su contrato:
   `cargo +nightly-2026-06-30 test -p trader-gemini-v5 --bin continuous_evolution_backtest --test continuous_capital_contract --locked -- --test-threads=1`.
   Verificar nombres y número de tests en el log; `check --all-targets` solo compila.
5. Puerta local cuando el semáforo admita: fmt, clippy del binario con `-D warnings`, el comando anterior y `check --workspace --all-targets --locked`.

**Fuera de este átomo:** promoción por rendimiento, wallet API y los fallbacks de `evolver`, `evolution`, `multi_coin_simulator`, `benchmark` y `ReplayConfig`. Quedan registrados, no ocultos.

## 5. FASE 3 — Correctness financiera y evolutiva en la ruta viva

Orden de ataques posteriores, cada uno con su propio RED y reserva. Los seis primeros proceden de la revisión estática cruzada; aún no tienen pruebas ejecutadas por V33. Antes de tocar `reconciliation`, host o ejecutor se coordinará con Línea C/AGY/Sol para no invadir sus reservas:

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

## 9. Censo archivo por archivo y recepción entre agentes

El inventario mecánico `docs/audit/V33_INVENTARIO_2026-10-24.tsv` contiene las 2.183 rutas del índice actual: 26 `REVISADO_ESTATICO`, 1 `RED` (capital inválido con evidencia ejecutada) y 2.156 `INVENTORIADO`. No es una certificación de revisión completa. El ledger incremental `docs/audit/V33_REVISIONES_2026-10-24.tsv` conserva los 34 hallazgos concretos sin reescribir censos históricos. Campos mínimos:

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
