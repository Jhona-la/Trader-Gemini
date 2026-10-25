# Plan maestro por fases V33 — revisión desde la base

**Revisión vigente:** 2026-10-24, Kimi Quant Sr V33, en coordinación con los demás agentes.
**Base inspeccionada:** `main = 4f9ada0980fdcd155bdce522993d7ec0ac0f30c9 = origin/main`, con PR #140 integrado por fast-forward preservando los siete WIP y el stash histórico.
**Alcance:** workspace Rust de 21 crates + paquete raíz, 2.203 rutas versionadas y 847 archivos Rust. Estas cifras describen el censo mecánico, **no** archivos ya auditados.
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
