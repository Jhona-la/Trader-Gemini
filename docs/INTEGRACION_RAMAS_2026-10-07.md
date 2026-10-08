# Integración de trabajo pendiente — 7–8 de octubre de 2026

Responsable: Codex, rama `codex/integration-recovery-2026-10-07`.
Base observada: `adeb8d1b1f8171b14fffa8abc9f54c396e4794dd`.
Estado: integración local en curso; sin publicación ni certificación económica.

## Censo inicial

Refs actualizadas por el coordinador con `git fetch origin --prune`.
`main` y `origin/main` coincidían en la base indicada. Los únicos heads
remotos eran `main` y `codex/root-audit-2026-10-04`; `origin/HEAD` es un alias.
Las cantidades son commits por ascendencia, no conteos de cambios funcionales.

| Rama | HEAD | Detrás / exclusivos frente a main | Árbol de trabajo |
|---|---|---:|---|
| codex/root-audit-2026-10-04 | d0e02da1 | 137 / 28 | limpio; publicado |
| codex/oos-partition-2026-10-04 | db3171c1 | 139 / 32 | limpio |
| codex/sa-score-monotonicity-2026-10-04 | 94790c5c | 139 / 32 | limpio |
| codex/forest-evaluation-2026-10-04 | cace007d | 154 / 1 | limpio |
| codex/feature-clock-2026-10-04 | 36b0063c | 154 / 2 | limpio |
| codex/evidence-expiry-2026-10-04 | 7aadf509 | 154 / 2 | limpio |
| codex/model-reload-contract | 3139f444 | 206 / 4 | sin worktree |
| codex/ruin-input-contract-2026-10-04 | f6087378 | 139 / 27 | 2 modificados y 4 nuevos sin commit |
| codex/review-plan-2026-10-07 | a054495c | 5 / 0 | 3 archivos nuevos sin commit |
| glm/h2-7-paridad-flow | adeb8d1b | 0 / 0 | activo, cambios GLM y Graphify |
| qoder/ola67-lows-limpieza | adeb8d1b | 0 / 0 | activo, 10 fuentes/tests modificados |

`f6087378` está contenido en root-audit. Sus cambios sin commit no lo están:
`crates/risk-engine/src/{lib,ruin}.rs`, dos contratos y auditorías MD/JSON.
Review-plan conserva `c07_ruin_composition_contract.rs`, el inventario y el
plan del 7 de octubre sin commit. No sobrescribir ni eliminar esos worktrees.
Los directorios `target/` ignorados tampoco son preservados por Git.

## Secuencia y condiciones

Orden acordado: raíz → OOS → selección SA → bosque → reloj → caducidad de
evidencias → recarga de modelos. Después se integra el trabajo independiente
del coordinador y la versión publicada de las sesiones externas, si cambió.

Cada merge conserva ambos historiales, recibe comparación contra sus dos
padres y `cargo check --workspace --all-targets --locked` antes del commit.
Las suites afectadas y T-1 se ejecutan sobre el candidato final. No se
transfiere un resultado de pruebas a un árbol distinto sin calificarlo.
Los recibos locales detallados se guardan en `target/integration-receipts/`.

### 1. Auditoría raíz

Tres conflictos documentales (`MEMORIA`, coordinación, forense) se resuelven
por unión de entradas completas. El conflicto de `orchestrator.rs` requiere
composición semántica: se conservan el rechazo de evidencia no finita de RA
y la contracción continua de Ω12. El veto discreto de Crash permanece sólo
como fallback cuando `p_crash <= 0`; el veto sistémico conserva `p_crash >= .9`.
El cálculo reutiliza la presión validada, sin otra lectura atómica intermedia.

El parser conserva `AggTradeEvent::timestamp_ms` de Qoder y añade las
validaciones de escalares y desbordamiento de RA. La corrección del contexto
OOS calcula el warm-up según la cola IS disponible. El workflow conserva los
pasos existentes y añade contratos de parser, riesgo no finito y frontera OOS.

Merge: `56eefd3145a39f856e769df43e21dc7fe0ee45c6`.
Check completo: exit 0, 11 min 59 s; diff del candidato idéntico antes/después.
Se interrumpieron dos intentos de compilación propios para reutilizar una
caché compatible y reducir contienda; no son fallos de código ni éxitos.
No se detuvieron procesos externos. Las pruebas de comportamiento quedan
pendientes del candidato combinado. Ninguna afirmación económica resulta del check.

### 2. Particiones OOS

Merge: `2fdb55ca1226b4909d58fe2057b60790c6e1f9c1`.
Se valida `0 < train_len < total_len` antes de activar NanoForest, consultar
FRED o iniciar ensayos; el warm-up conserva la cola IS realmente disponible.
Se conservaron por unión dos conflictos de bitácoras. Comparación de código
contra cada padre: la política OOS y los cambios recientes de main coexisten.
Check completo exit 0, 303.432 s; árbol `58120d9fae00a9cf301e298ff00cd5519949dd8e`
idéntico antes y después. Validez estructural no acredita suficiencia estadística.

### 3. Selección SA

Merge: `9869f8c` (abreviado).
Conserva penalización monótona de scores negativos, campeón finito explícito
y enfriamiento/recalentamiento tras rechazar utilidad inválida. La interfaz
canónica sigue siendo `pub use risk_engine::selection_stats`; no se restaura
la copia eliminada en Ω11. Sólo se añaden los dos módulos SA independientes.
El workflow conserva las suites de ejecución/datos y añade los contratos SA;
las bitácoras se unen. La partición OOS previa permanece antes de los efectos.
Check completo exit 0, 129.841 s; árbol `0d09c8d13cd7f52badaf2e85d3d12ca3b454bf10`
idéntico antes y después. Las anclas legacy del CLI siguen siendo una deuda
separada; esta integración no inventa sincronización de curvas.

### 4. Bosque de evaluación

Merge: `3e82e64d6f62d9d3d93c1cd0a4d3e988208faf41`.
El resultado conserva `control_realized_pnl_pct` de XCIV y añade
la historia de drawdown realizado observado del satélite. Se elimina el
override de latencia que diferenciaba el genoma evaluado del almacenado.
No se cambia a MTM, ni se incorpora DSR o una política nueva de promoción.
Los únicos conflictos son bitácora y pasos CI; ambos se conservan por unión.

Check completo exit 0, 155.829 s; árbol `256cceb63aab0348657257d97b327ccce6eeb7ad`
idéntico antes y después.

### 5. Reloj de features aceptadas

Merge: `aa74ddeabef30bf103ac22abd9d320689310946a`.
La transición de `last_event_ms` ocurre después de las guardias de admisión;
un tick futuro rechazado no envejece cooldowns ni rachas. Fuente y contratos
coinciden con el satélite; coordinación/CI se compusieron por unión.
Check completo exit 0, 105.984 s; árbol `1e83e7809815b82322948bd156913b8c297fd1b7`.

### 6. Caducidad de evidencia

Merge: `b327fde2efeb6eb60d999984f8c535e529167bea`.
La publicación de coherencia ocurre después del consumo de maduración H1-1
y antes de los retornos TTL/kill-switch; se conserva la prueba familiar de
Ville vigente. También publica el camino dual y las actualizaciones outcome.
Ausencia de estimador retira el veto/cota caducada según el contrato existente.
El fixture antiguo de 40 pares ya no superaba la política familiar actual:
se ajustó a 120, con afirmación intermedia explícita de ausencia a 40 y
presencia a 120. No se rebajó el umbral del producto e ni la corrección familiar.
Check completo exit 0, 304.924 s; árbol `44f8f7278b01e0266340eb38a80b00f2a76e03de`.

### 7. Recarga de modelos

Merge: `964d94a` (abreviado).
El tracker marca sólo cargas exitosas, reintenta errores y prioriza JSON sobre
BIN hermano. Startup/polling usan el mismo helper. Se preservan Ω14, sus curvas
y los precios protectores del host; CI incorpora la suite del watcher.
Persisten MW04/05/07: revocación, identidad efectiva del BIN servido y contenido
que cambia conservando tamaño/mtime. No se declaran cerradas por estos tests.
Check completo exit 0, 169.768 s; árbol `7d0979dd4c444cf3b56f69640da60be1abba8104`.

### 8. Fundamentos R4 y recuperación C07

Merge `2a6ff1b` (abreviado), segundo padre `639c3e0d`: mantiene el daemon
Darwin entre rondas del proceso,
conservando `cumulative_trials`; persistencia entre reinicios queda abierta.
Recupera sólo la guardia de producto no finito de RUIN y el contrato consumidor
C07 con procedencia verificable. No sustituye `ruin.rs` por la copia antigua.
Los worktrees originales sucios permanecen intactos; este port acotado no
significa que todo su contenido sin commit haya sido integrado.
El guard CI inspecciona el candidato original sin un commit normalizador que
ocultaba su diff. C07 queda incluido en la ejecución CI del paso de riesgo.

Check completo exit 0, 353.412 s; árbol `dd507d117cb04708f3e43e7a8dd3ff13847c1527`.

### 9. Main publicado por el consejo

Candidato sobre `d4e4583897105cff58a1c335ffc63398d5f36573` (GLM103),
confirmado con `git ls-remote origin refs/heads/main` antes de integrar.
Conflicto de coordinación resuelto por unión. Los tres archivos Rust conservan
exactamente el delta publicado: constantes de igual valor, contrato y comentarios.
El nuevo test H2-7 ejerce `vote`/`voto_espectral`; su comentario sobre el camino
vivo no constituye una prueba directa de `evaluate_flow_impulse`.
Merge: `f3f86960190d99aeacdb75b14cdfab0adceaa186`.
Check completo exit 0, 517.848 s; árbol `462e1c69c4d08c7e95f22a81893ad23855bb5495`.
`git -c core.whitespace=-blank-at-eof,-blank-at-eol diff --check d4e45838 HEAD`
pasó sobre el rango completo. Las suites y T-1 se registran por separado.

## Entorno local de verificación

`nightly` y `nightly-2026-06-30` resuelven aquí al mismo compilador:
`rustc 1.98.0-nightly (096694416a41840709140eb0fd0ca193d1a3e6ba 2026-06-29)`,
host `x86_64-pc-windows-msvc`, LLVM 22.1.8. Los checks usan `--locked`,
`--workspace --all-targets`, un job y la caché de `model-lineage-audit`.
Las advertencias heredadas se conservan en logs. Un check no sustituye a tests.


## Regresión causal C07: RED

Candidato histórico de la regresión causal: `f3f86960190d99aeacdb75b14cdfab0adceaa186`.
No se atribuye este RED a la revisión posterior de main del 8 de octubre.
Se retiró sólo el hunk `405d9a15` de `risk-engine/src/lib.rs`, sin cambiar el
índice ni crear commit. Árbol RED temporal `6d6eb1daf1e27056664f01aa347da208c355de09`;
blob de fuente `35022083e831b8813d56d82b34041c232ef4abf2`.
El árbol se obtuvo con índice alternativo descartado después del registro.

Comando: `cargo +nightly test --release -p risk-engine --locked --test
c07_ruin_composition_contract c07_t02_nonfinite_micro_product_cannot_reopen_through_lerp
-- --exact --nocapture --test-threads=1`. Entorno `TG_GENOME_ENV=backtest`, jobs=1.
Resultado esperado: exit 101, **un test ejecutado y fallido**, 3 filtrados;
129.430 s incluidos compilación/enlace, test 0.19 s. No se confunde con los
intentos de compilación cancelados detallados abajo.

La causa reproducida fue `nonfinite product admitted`, no sólo un diagnóstico:
- C=25: Long y Short admitidos con volumen `1.2007780104736843`.
- C=100: Long y Short admitidos con volumen `12.401746228688301`.

La restauración automática dejó el SHA-256 de bytes idéntico antes/después:
`888b974ecfb3cbf866c084cbdedcf1351049add0ce8ecb3cb1a1707610311e60`.
El diff volvió a vacío y HEAD/índice permanecieron iguales. Log RED SHA-256:
`76384385b8bb66449bd3663b55c356e3427788f34c96b9f0a16a9f0563fa38d9`.
GREEN: C07 **4/4**, entradas no finitas **7/7** y admisión de cartera **7/7**,
18/18 en total, exit 0 y 16.581 s incluidos compilación/enlace. Mismo perfil
release, entorno y candidato; los controles saludables permanecen abiertos.
Log GREEN SHA-256 `0096f4c77940c976140c482398820c6dac49d44bcd0aa9116de0363bc683e401`.
Las otras suites históricas terminaron después: el total sobre f3 fue
**209 aprobadas, cero fallidas y una ignorada**. T-1 no se inició en f3.
Estos resultados no se trasladan al candidato actualizado del 8 de octubre.

Recibos versionables: `docs/audit/RECIBOS_INTEGRACION_R4_2026-10-07.json`.
Sus listas pendientes y su estado deben consultarse junto con los conteos;
cada árbol compilado coincide con su commit de integración respectivo.
El esquema 2 distingue los resultados históricos de los del candidato actual.

Límite: se inyecta `min_confidence_btc=-Inf` en seis escenarios con controles
saludables. No cubre todas las configuraciones, ni prueba overflow del producto
con operandos exclusivamente finitos. La configuración NaN puede ocultarse
antes de esta guardia por `.max(0)`; la validación integral permanece abierta.

### Intentos cancelados y elección de caché

Antes de ejecutar RED hubo dos intentos cancelados durante compilación: debug
(756.291 s) y release en model-lineage (196.529 s), ambos con **cero tests**.
Se detuvieron sólo procesos propios con PID/StartTime/parent verificados;
`finally` restauró la fuente exactamente en ambos. Los recibos completos se
conservan en `target/integration-receipts/cancelled-debug/` y
`cancelled-model-release/`. No son resultados RED ni fallos de código.

El grafo de features del paquete risk aislado usa Chrono sin serde; el grafo
T-1 disponible inicialmente en model-lineage no garantizaba esa variante.
Se verificaron en la caché raíz el fingerprint y rlib exactos de Chrono-f29,
Polars-utils-f3 y Polars-arrow-d05. Pruebas finales usan
`C:/Users/jhona/Documents/Proyectos/Trader Gemini/target`, perfil release,
con los mismos toolchain/flags. Cargo recompila las fuentes del worktree del
candidato. Ningún binario previo se aceptó como evidencia de ejecución.


## 10. Reconciliación con main del 8 de octubre

El fetch confirmó `origin/main=18bbd1d90a073ae49ca8083c5898537013e4bfeb`,
98 commits remotos adicionales respecto del corte anterior. El merge
`7acf36aadf33a323b368f8e59fb77e2aafd40720` tiene padres f3 y 18bb y árbol
`f26f3b83e784dac821b53a5363f19bb29baf3c4d`.

Se revisó el resultado contra ambos padres. Memoria, coordinación y auditoría
conservan las historias; el plan de revisión mantiene completas las secciones
1–10 comunes y la nueva sección 11 de Sol. El ledger conserva el snapshot
histórico consistente de `8938cf41`; el coordinador regenerará el censo tras
el merge. Los borradores anteriores no reemplazaron las adendas de Sol.

En runtime se conserva el reloj de eventos aceptados y se adopta el nuevo
nombre `last_exit_fastband_ms`. El contrato de transición adapta ocho usos
de campos retirados (`last_scalp_exit_ms` y las tres rachas `scalp_*`) a los
nombres fastband. Mantiene las mismas entradas, snapshots y aserciones.
El host conserva el tracker MW y el Arc persistente de Darwin, junto con el
nuevo tercer fallback SL por curva de main. Se preservan los hooks de
caducidad/coherencia, Ville, C07, reporting diario de Sol y el guard CI sin
commit normalizador. OU, registry, Darwin y selection_stats mantienen los
deltas remotos; esta composición no certifica sus propiedades más generales.

El intento 20 de check se canceló durante compilación tras detectar seis
referencias de rachas antiguas todavía presentes en el fixture. No completó
la validación ni ejecutó tests. Sólo se detuvo el Cargo propio identificado.
Tras corregirlas, check 21 terminó **exit 0 en 110.744 s**, con árbol estable
antes/después. El log tiene SHA-256
`f3f626a34d7c169928448c70f645064249d1dc315559bb10aea29e2cef261aa2`.
Las revisiones independientes de runtime y del delta final de seis renombres
no encontraron pérdida de contratos recuperados. El guard de espacios del
rango completo `18bb..7acf36a` pasó con la política CI vigente.

Las suites renovadas usan release, jobs=1, `TG_GENOME_ENV=backtest`, el mismo
compilador y caché raíz. Sus recibos se guardan por separado bajo
`target/integration-receipts/2026-10-08-refresh`. Consulte los campos
`completed_behavioral_suites`, `current_counts` y `pending_suites` del JSON
versionable para distinguir resultados completados de ejecución pendiente.
T-1 se ejecutará explícitamente sobre el candidato actual, con un único test
pesado y ratchet 0.110 sin modificar; no equivale a una prueba de rentabilidad.

La revisión aislada de OU descubrió un defecto del padre remoto: su API opt-in
puede caer al evaluador legacy con timestamp repetido o regresivo y producir
Short con duración cero. No se encontraron callers productivos fuera de tests.
Queda como I5 y lote posterior coordinado; no se modificó runtime para corregirlo
como parte de este merge. Tampoco se cierran las reservas estadísticas, MW,
MTM, dominio de Ville o persistencia entre reinicios descritas por el consejo.


## 11. Candidato final del 8 de octubre

Merge `e9c6a435eb1780da7f2c6e29103f3bd7d227d94c`, árbol
`10b8a427dfbdd5b203595b039b2dad50c69d6f68`, incorpora `origin/main=5842c8e3`
y sus commits Ω26/Ω27. La memoria se compuso por unión íntegra; las únicas
diferencias de código frente a 7acf son retirar dos overrides de latencia
25 ms del backtest continuo y renombrar una prueba de Hawkes ya reparada.
Las fuentes recibidas coinciden con el segundo padre en esos dos archivos.

Check 37 completo antes del commit: exit 0 en 15.803 s, árbol estable;
log SHA-256 `f9b0ab1184ad537ef57e978441505d7b0e5af7bc84a9dbe9d2f36d0f8fb7162d`.
Se revisaron ambos padres y pasó el guard de espacios sobre `5842..e9`.
No se tocó el umbral T1 0.110 ni se sobrescribieron los worktrees activos.

La matriz de aplicabilidad compara **todos los objetos versionados** por
modo, tipo y OID, y falla ante cualquier cambio fuera de la lista explícita:
los dos archivos indicados y MEMORIA, coordinación y TRIAJE. Registra 530
entradas idénticas en fuentes, tests, manifiestos, lockfile, configuración,
fixtures y assets. También exige recibos exitosos con HEAD/árbol estables y
verifica el compilador y los flags seleccionados. Git no demuestra constancia
del estado externo o de cualquier entorno heredado no registrado.

Las 731 aprobadas de 7acf se conservan con su SHA original. Sus dos pruebas
de reporting se sustituyen por la ejecución sobre e9 y no se suman dos veces.
La tabla siguiente contiene las ejecuciones aplicables, con su HEAD real:

| Suite | HEAD ejecutado | Aprobadas | Fallidas | Ignoradas |
|---|---|---:|---:|---:|
| 22-risk-contracts | 7acf36aa | 54 | 0 | 0 |
| 23-parser | 7acf36aa | 19 | 0 | 0 |
| 24-oos | 7acf36aa | 23 | 0 | 0 |
| 25-sa | 7acf36aa | 22 | 0 | 0 |
| 26-forest | 7acf36aa | 16 | 0 | 0 |
| 26b-ast | 7acf36aa | 3 | 0 | 0 |
| 27-core-contracts | 7acf36aa | 115 | 0 | 1 |
| 28-core-lib | 7acf36aa | 170 | 0 | 0 |
| 29-host | 7acf36aa | 2 | 0 | 0 |
| 30-risk-lib | 7acf36aa | 142 | 0 | 0 |
| 31-signal-lib | 7acf36aa | 119 | 0 | 0 |
| 32-strategy-lib | 7acf36aa | 27 | 0 | 0 |
| 33-strategy-contracts | 7acf36aa | 11 | 0 | 0 |
| 34-registry-lib | 7acf36aa | 6 | 0 | 0 |
| 35b-reporting-final | e9c6a435 | 2 | 0 | 0 |
| 35c-stateful-open-final | e9c6a435 | 5 | 0 | 0 |
| 36-t1 | e9c6a435 | 1 | 0 | 0 |

Resultado aplicable en este corte: **737 aprobadas, 0 fallidas y 1 ignorada**;
729 se conservan por identidad desde 7acf y 8 se ejecutaron en e9.
T1 completó exactamente un test pesado, exit 0, 1611.203 s incluidos compilación/enlace.
Cobertura observada: **16/144 = 11,1%**, superior al ratchet **0,110**, sin cambiarlo.
Se registraron las 144 perturbaciones; en cinco la proyección no cambió ninguna
coordenada efectiva. Esto no prueba inercia global ni rentabilidad.
Main remoto seguía en `5842c8e3` antes y después de T1.

El inventario local de modelos permanece ignorado por su anotación original;
no es un fixture portátil. Las pruebas `open_debt` y ciertos diagnósticos
verifican limitaciones conocidas: aprobarlas no significa haberlas reparado.
El cambio Ω27 trata honestidad del triaje; la heurística NaN sigue abierta.

Entorno de cada ejecución: release, `+nightly` (rustc 096694416), `--locked`,
`CARGO_BUILD_JOBS=1`, `TG_GENOME_ENV=backtest`, caché raíz y un hilo de tests.
Los dos tests Ω14 del host se ejecutaron como harness en `release/deps`;
no se reconstruyó ni reinició el ejecutable operativo. T1 usa exclusivamente
la serie sintética y el predictor del contrato, con filtro exacto del único
test pesado. Mide sensibilidad condicional; no acredita crecimiento del
100% en 72 horas, validez OOS suficiente ni ausencia de los defectos abiertos.

No se ha efectuado push desde este worktree. La publicación, sincronización
local de main y limpieza segura de referencias quedan coordinadas por el
responsable raíz después de los recibos y el cierre documental.

Respaldo externo verificado de 175 archivos (27.860.258 bytes):
`C:/Users/jhona/.codex/audit-evidence/trader-gemini/r4-e9c6a435-2026-10-08`.
El manifiesto conserva SHA-256 y tamaño de cada log, recibo, diff y script local;
se excluyen cachés compiladas, bytecode Python y el hash circular del manifiesto.
