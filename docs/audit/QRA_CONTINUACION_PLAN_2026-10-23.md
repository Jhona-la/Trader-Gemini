# Continuación QRA por fases — 2026-10-23

Estado actual 2026-10-23T09:56:51-05:00: átomo F0/F1 integrado en main688, candidato10/10 y push real10/10 propios. [Recibo y límites](QRA_F0_F1_RESULTADOS_2026-10-23.md), [expediente](QRA_F0_F1_CI_2026-10-23.json) y [ledger5](QRA_F0_F1_RUTAS_2026-10-23.tsv). F2 no está cerrado. El resto de cortes fechados conserva la cronología; censo2193 histórico, sin auditoría integral.

Corte 2026-10-23T01:35:52-05:00. Base 75be5d6751b0075bbf061ce9f5b2ee09a36b427f; rama codex/qra-capital-20261023 y worktree propio locked ACTIVE. El cierre Omega220 PR102/104 permanece válido para su código e286 y sus 45 objetivos; no certifica esta nueva ola. El [inventario](QRA_CONTINUACION_INVENTARIO_2026-10-23.tsv) enumera 2193 rutas del árbol base por blob. Es una cola completa de archivos, con revisión pendiente; las 23 coincidencias de Omega220 sólo heredan su alcance concreto. La fase de la tabla es clasificación mecánica, no auditoría ni prioridad científica demostrada.

## Orden y aceptación

| Fase | Trabajo | Estado y condición de cierre |
|---|---|---|
| 0 | Recursos, semáforo, perfiles, dependencias, higiene | Átomo de tres launchers cerrado:84 recursos en candidato y main, sin Cargo real local;32 métodos F0. Target canónico/jobs2/guardas/cleanup propio. Censo completo76.31GB lógicos; cargo tree68 entradas/65 pares metadata, sin poda semver ni ahorro certificado. Presupuesto16GB global y300GB futuro no se infieren de estos cortes. |
| 1 | Ramas locales/remotas, rescate, integración y CI | PR105 en main688 con10/10 propios; Sol1095/GLM154 y rescate selectivo AGY103 preservados. Nueva AGY PR106, Sol1097, GLM155 y Verify ACTIVE siguen independientes. Retirar sólo rama/WT QRA tras recibo integrado; no declarar abandonada actividad ajena ni transferir verdes. |
| 2 | Capital/config/API, fees, OMS, red y precisión | Pendiente: [capital cero observado](QRA_F2_CAPITAL_DISENO_2026-10-23.md), presencia/sesión/cache completo y wallet/equity; [ruta UDS](QRA_F2_UDS_ENDPOINT_2026-10-23.md) por entorno. PR106 AGY en revisión separada. FFI13/null/overflow mantienen prioridad independiente; M aún no los corrige. No mezclar estos pendientes con el verde F0/F1. |
| 3 | Evolución online, Top10, persistencia y costo total | Pendiente: grafo productor-consumidor por motor, causalidad temporal, estado duradero y ahorro de recursos medido. Ingreso de wallet real ya existe en el host; escenario FFI debe declarar su capital. No introducir red dentro de un replay reproducible sin snapshot de cuenta fechado. |
| 4 | Modelos, canarios, concurrencia y Demo | Pendiente: insumo causal, dominio numérico, unidades/formas, productor y consumidor reales, baseline, costo y experimento OOS por teoría. Hipótesis MPS/Koopman/ACI y restantes del adjunto no quedan certificadas por sus nombres. No declarar ruina cero, nanosegundos o crecimiento100%/72h sin medición. |
| 5 | Visor y meta-analítica | Pendiente: linaje hasta ejecución, pérdidas/backpressure visibles y reporte con dataset/periodo, PnL/ROI/WR bruto/neto, Sharpe y EV. Telemetría fuera de hot path y almacenamiento acotado. No inventar métricas financieras a partir de tests sintéticos. |

La implementación de cada átomo sigue hipótesis → ruta de datos → contrato real → CI candidato → CI push de main → recibo → retirada propia. Se mantienen pequeños commits y se reevalúa main cuando avance. El inventario se consume archivo por archivo en esa secuencia, sin marcar miles de archivos como revisados por un grep.

## Reservas y coordinación

Root: plan/ledger y análisis RO de FFI en crates/backtest-engine/src/lib.rs. resource_safety implementa sólo los tres lanzadores F0 y amplía test_qra_resource_safety.py (ya en CI); sin modificar el semáforo compartido. omega220_review: revisión adversaria RO de contrato/ABI. git_audit: ramas/CI RO. resource_safety: mediciones RO. AGY mantiene parsers/MXCSR/shard; Sol su cobertura CI; GLM DMRG. El primary compartido no recibe checkout/pull, fuentes ni cambios de índice. No se eliminan archivos o procesos ajenos.

[Coordinación AGY](https://github.com/Jhona-la/Trader-Gemini/pull/103#issuecomment-6359000192): su shard ejecutó112 casos (50 comunes con Sol, uno nuevo MXCSR y61 de telemetry-server). Resolver el solapamiento conservando cobertura; una suma mecánica duplicaría jobs. Este aviso no toma posesión de sus ramas.

## Hipótesis FFI y prueba propuesta

El consumidor C recibe métricas de una cuenta ficticia cuando falta capital; también puede alcanzar creación de slices con volumes nulo y aritmética de extensión desbordada. No hay llamadores internos de esos exports; se conserva ABI para consumidores externos y no se afirma impacto live medido. La API Rust canónica ya recibe capital explícito, y el host usa fetch_account_balance con reintento.

Probar ausencia, vacío, texto, cero, negativos, NaN/Inf; positivos de diferente escala exactos; rechazo antes de mutar outputs/registries, puntero volumes nulo y extensión desbordada sin dereferenciar; control positivo del export con datos válidos hasta el motor canónico. No inferir validez de una dirección arbitraria, longitud real, terminación CStr, aliasing o inmutabilidad externa del mmap: permanecen precondiciones unsafe. El cero legado no distingue error de cero trades; debe acompañarse de diagnóstico y el límite de ABI queda explícito.

## Deuda científica y operativa

Verificar en documentación primaria los contratos reales de Binance antes de STP/GTX/reduceOnly, batchOrders, modify, cuotas, SBE y firma; ninguna instrucción genérica se aplica indiscriminadamente a Hedge/One-Way. Investigar FTZ/FMA frente a la reproducción IEEE, y postselección de estados futuros frente a leakage. Criterios para promover modelos y pasar a Producción requieren evidencia OOS, costos observados y control de posiciones vivas; este átomo no ejecuta fondos ni promueve entorno.

## Límites del átomo F0

No se atribuye el Cargo ajeno a estos lanzadores sin prueba. No se fuerza la terminación de procesos activos ni se barre target durante su uso. Corregir admisión no mide ahorro de RAM ni certifica ejecución nanosegundo. Oracle conserva por ahora una observación de hashes: no se acreditará identidad entre un glob y el ejecutable de una invocación Cargo posterior. La certificación de ese linaje requerirá otro contrato; no inventar un PASS a partir de esa observación.

## 2026-10-23T08:35:09-05:00 — F0 preparado y base renovada

Los tres lanzadores usan el semáforo real. quality_gate drena sin buffer ilimitado, no duplica Clippy y propaga fallos75/101 incluso en el paso informativo; fmt con warning+exit0 no falla. Ambos Bash fuerzan target canónico/jobs2 y rechazan overrides CLI; el medidor retira sólo altas propias no modificadas al fallar, sin --force. Doce contratos nuevos,28métodos totales PASS en272.35s con Cargo sentinel y repos externos, PowerShell7.6.7 y WindowsPowerShell5.1.26100.9549 reales, cero omisiones y cero Cargo real local. CI propio candidato y push de main todavía PENDIENTES.

Main75e17d4635c3c12eaf377119fb2518df18b10ba1 integra Sol1095 y GLM154 y tiene9/9 propios (Unit39918514517/Replay39918514488); DMRG6/MPS7 medidos. La base original75be del inventario se conserva como snapshot, no se reetiquetan blobs como actuales. Reanclar el átomo sobre75e y rescatar PR103 si sigue sin respuesta/avances; sólo en rama propia, jamás editando el primary AGY.

Censo de almacenamiento20s:110071 archivos y44,359,003,062bytes lógicos contados,8099directorios pendientes. Target contado44,345,299,482bytes, NO su total. Volumen libre362,375,606,272bytes. Método metadata streaming sin seguir reparse; no ahorro físico ni presupuesto300GB certificado. RAM13:34Z0.75GiB<4; no Cargo local. cargo tree --duplicates permanece sin ejecutar por presión de recursos; no podar semver a ciegas.

F2 adicional prioritario demostrado RO: ACCOUNT_UPDATE con wb0 pierde publicación en stream/puente y capital_del_freno usa el arena positivo cuando exchange<=0. Distinguir cero observado de ausencia ligada a sesión, sin aceptar wb ausente que serde(default) convierte a0. Requiere su propio átomo y contratos reales; no está corregido por F0. Fuentes primarias: https://developers.binance.com/en/docs/catalog/core-trading-derivatives-trading-usd-s-m-futures/api/rest-api/account y https://developers.binance.com/en/docs/products/derivatives-trading-usds-futures/user-data-streams .

## Candidato F0/F1 congelado para CI

Source ancla ea05e99f742b9b6740161ba3dd7e4bd7654de118. Cinco rutas de comportamiento: tres lanzadores, contrato Python y shard CI. Treinta y dos métodos en recursos (16existentes+16nuevos):28pase completo previo PASS272.35s; luego6focusedPASS50.553s(3previos+3nuevoscleanup) y1método parametrizado de12negativos de flags PASS12.416s. Ambos PowerShell7.6.7/5.1.26100.9549 realmente ejecutados, sin skips ni Cargo real. La suite completa del candidato32 y el total esperado84recursos siguen pendientes de CI; no se presenta esa expectativa como resultado.

Cleanup sólo de altas de esta invocación: valida ruta física/common-dir/HEAD, hash raw de la copia propia y ausencia de stage antes de restaurarla o retirar el archivo originalmente ausente. Modificación ajena distinta, cambio de HEAD o contenido staged conserva el árbol. Propaga75/101 y retira con Git sinforce. Los tres launchers rechazan RUSTFLAGS/CARGO_ENCODED_RUSTFLAGS/CARGO_BUILD_RUSTFLAGS; Bash también bloquea target/config/jobs alternativos. La salida de quality se drena sin acumularla toda; Clippy workspace una sola vez, errores nunca informativos.

Rescate AGY103: f174 es ancestro por merge49f9, con exclusiones deliberadas registradas en buzón/PR103 (comentario6366222051). Shard nuevo sólo telemetry-server completo; preserva seis paquetes Sol y sus doctests, Host OMS/golden/storage sin recorte. Todos los854archivos Rust, Cargo.toml/lock y .cargo/config son idénticos a75e. No se incorpora DAZ global (UB según contrato oficial Rust) ni parsers sin consumidores vivos. La historia f174 conserva su desarrollo y los motivos no se ocultan tras un merge limpio. No se heredan sus10verdes antiguos. Composición esperada8Unit+2Replay=10jobs; cobertura objetivo50Sol+61telemetry-server=111 sinduplicados, pendiente de logs propios.

Cierre exigido: candidato10/10 y push real main10/10 por separado, fuente anclada, recibo con límites, rama/WT propios retirados. Primary AGY con cambios ajenos y Verify locked ACTIVE se preservan. No activar motor, fondos ni entornos. El átomo F2 de capital observado0 y los fallbacks FFI siguen pendientes.

El candidato añade inventario cargo tree --workspace --duplicates --locked una sola vez en el shard telemetry-server de CI, sin Cargo local ni unificación semver a ciegas. Esa evidencia de dependencias sigue pendiente hasta completar el paso; sustituye al candidato9a9 de arriba antes de su cierre, sin heredar parciales.
