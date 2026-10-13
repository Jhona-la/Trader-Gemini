# R4: cierre de T1 y contratos de ceros y replay del veto espectral

Fecha: 2026-10-11, America/Bogota. Responsable: Codex; revisión independiente
por subagentes de Git, contratos espectrales y planificación. Base congelada:
`1d1f8bdd35640e5ab12b6e55edb9bf582c41c33a` (merge del PR #38, Línea C).
Estado del corte intermedio: RED unitario recibido; RED del consumidor en
compilación fría, sin conclusión. La corrección permanece congelada hasta
recibir ese RED. GREEN, regresión final y publicación siguen pendientes.

Esta adenda continúa el [plan compartido](../PLAN_MAESTRO_SINCRONIZACION.md),
el [recorrido R0–R9](../PLAN_REVISION_ARCHIVO_POR_ARCHIVO_2026-10-07.md) y las
[fases C-F0…C-F11](../PLAN_LINEA_C_DESDE_LA_BASE_2026-10-11.md). Reutiliza sus
recibos y preserva los cortes históricos; no crea un plan rival ni declara
que un inventario equivalga a una auditoría.

## Corte provisional de ejecución y sincronización

- Base y último remoto informados por el integrador: `1d1f8bdd`; esta línea
  no ha publicado una revisión nueva. Los workflows de esa base constan
  verdes según la comprobación del integrador: `38133310816` y `38133310892`.
  Esos resultados no certifican la reparación todavía sin integrar.
- RED unitario: `01-red-unit.json` en `target/r4-zero-replay/` registra
  2 aprobados, 5 fallidos, 0 ignorados, 126 filtrados y exit 101 sobre
  `1d1f8bdd`, con árbol de índice antes/después
  `59ca3abfa606a2191028fa5e74162e4e04287a30`. El log tiene SHA256
  `c56abc4fab2cb363d23f685d98eccf49792efdfabe2c1669d49d8cc0c60ab6e6`;
  el ejecutable retenido tiene SHA256
  `809796f1913e0d44b307be22c79dc1302778f9d032fe90a60106fa5e33992112`.
  Los fallos documentan los contratos adversos sobre la base, no un GREEN.
- RED del consumidor: los dos contratos de publicación/consumo siguen en
  build frío con el target privado `target/r4-build`. No se infiere un
  resultado del tiempo transcurrido, la ausencia de errores ni el progreso
  de compilación. El cierre incorporará resultado y ejecutable exactos.
  La fixture v1 contempla publicación/apriete de `rho_final`, sin un
  cambio de admisión reproducido todavía. La revisión corrigió la firma
  de `Position::open`: cantidad 8, margen 0,1; la cantidad no era 0,1.
  Sus riesgos son 0,08 para la posición previa y 0,1875 para la candidata;
  con ρ=1, el grupo alcanza 0,2675 > 0,25. Se retira la afirmación
  anterior de 0,1885 < 0,25: la fixture puede cruzar el límite, pero
  contiene una inconsistencia entre margen y nocional y su assert de
  base ρ<0,8 no garantiza admisión (corte aproximado ρ=0,6978).
  Después del primer RED, el integrador reforzará las precondiciones con
  riesgo aproximado de 0,13 por pierna para exigir aceptación en frío y
  `REJ_CORRELATION` en caliente, y repetirá el RED de esa fixture antes
  de corregir producción. No hay aún resultado confirmado de ese
  contrato reforzado.
- Línea C llegó a `1f752037bafca6817e2551eb3d21fd87882fb27b` y trasladó
  C-F2 al PC del operador, en `.claude/worktrees/determined-tu-0d7126`.
  Sus 27 archivos nuevos de tests y documentos estudian soluciones
  conocidas, sin cambio de producción. Sus archivos `cf2_*` y OPEN se
  conservan separados de los contratos mecánicos de este lote.
- Si informa un primer all-targets aprobado en su línea y conserva los
  commits `8404bbdf` (pronósticos emitidos/transiciones), `c9d877da`
  (decaimiento epigenético) y `06ce88ef` (ingesta del benchmark). La
  verificación de consumidores y replay sigue pendiente; estos commits no
  se atribuyen como integrados ni certificados por esta línea.
- El integrador retiró `qoder/r7r4-nucleo-vivo` y `qoder/ronda7-plan`
  después de comprobar limpieza y ancestría; preservó `.r7r4` y `.ola73`
  detached en los mismos SHAs. Claude E continúa activo y se preserva;
  también se preservan C-F2, Si, Qoder R7-R6 y el replay reservado de Sol.
- En el censo comunicado por el integrador a las 14:12, C y Si tienen
  cambios locales pendientes. Las descripciones de commits anteriores no
  certifican esos árboles nuevos; se preservan y requieren recibos propios.
  El checkout compartido pasó al trabajo activo de AGY Ω83: no efectuar
  checkout ni fast-forward de ese directorio desde esta línea. La base
  remota informada continúa en `1d1f8bdd`.

Este corte es provisional. Los resultados finales y la publicación se
añaden como una adenda posterior con recibos; ninguna fila pendiente se
convierte en cerrada por compilar o por escribir este documento.

## 1. Base, metas y coordinación

El pedido vigente es verificar que T1 terminó, conciliar lo pendiente con
`main`, retirar ramas cerradas y corregir ceros y replay en los vetos
espectrales. El crecimiento de la cuenta se mide con patrimonio neto,
costes, funding, periodos sin operar, capacidad e incertidumbre. El objetivo
de ×2 cada 72 h equivale a `ln(2)/3 = 0,231049` de crecimiento log diario y
`2^(1/3)-1 = 25,992105 %` diario compuesto. Es una referencia financiera
del operador; estos contratos de software no miden su viabilidad.

Las fuentes doctrinales divergen. La RULE `growth_over_wr` y el Contexto de
[ADR-0016](../adr/ADR-0016-riesgo-geometria-y-falsacion-de-la-meta.md) piden
duplicación en 72 h. [ADR-0003](../adr/ADR-0003-doctrina-meta-geometrica.md),
aceptado como directiva del operador, pide maximizar crecimiento geométrico
sujeto a riesgo, drawdown, costes y capacidad. Línea C §9 propone como
puerta `g_LCB,95 > 0` en cada mitad OOS y reportar la duplicación como
brecha; atribuye el reencuadre al dueño. Esta adenda registra ambas fuentes
sin inventar una ratificación humana nueva. `LC-F0-01` sigue abierto:
ninguna frase de este plan sustituye un ADR aceptado. Los estimadores y
resultados económicos siguen sin medir en este lote.

| Línea | Reserva y estado observado | Coordinación de este lote |
|---|---|---|
| Codex | Rama `codex/root-spectral-zero-replay-2026-10-11`, worktree aislado; integración y contratos transversales | `quantum-arena/src/espectral_multiactivo.rs`, sus tests y recibos; un escritor por archivo |
| Subagentes de Codex | Lectura de Git, contratos/QA y planificación; no se les atribuye una rama separada | Contratos adversos, revisión causal y recibos bajo la coordinación del integrador |
| Si, chat «Audita y evoluciona Trader Gemini» | Rama `codex/spectral-evidence-1011`; reserva `quantum-arena/src/spectral_tape.rs` y `state.rs`; commits `8404bbdf`, `c9d877da`, `06ce88ef` | Target aislado; primer all-targets informado, consumidores/replay pendientes; este lote no edita sus rutas |
| Línea C, Claude, CL-n | PR #38 integrado; C-F2 activo en `1f752037`, worktree `.claude/worktrees/determined-tu-0d7126`, rama `claude/auditoria-deslizamiento-apalancamiento-sqtc08`; 27 archivos nuevos de tests y documentos, sin producción | Mantener separados `cf2_evalues_cuantiles_solucion_conocida.rs` y `cf2_multiactivo_solucion_conocida.rs`; sus OPEN de potencia/solape no son ceros/replay; no editar sus archivos ni activar sus OPEN |
| Línea E, Claude «elegant», QS-n | Ramas 1–10 del núcleo, admisión D2/D3, T2 y `unit-suites.yml`; T2 v2 con cambio de régimen anunciada | Preservar `REJ_EV`; T1 no reemplaza T2 ni contratos de puertas |
| Qoder | R7-R6 «Aprender y medir», worktree `.r7r6`, rama `qoder/r7r6-aprender-medir`; dominios de espectro/consenso | Informar alcance multiactivo y resultado; no atribuirle acuse nuevo a partir de una nota histórica |
| Sol | `.sol-replay-2026-10-09`; `booktick_replay.rs` y `metrics.rs` | No confundir replay de maduraciones con su reserva de contabilidad de replay; preservar sus archivos |
| Antigravity, AGY | Registro, señales físicas, consejo y continuo; Ω83 activo en el checkout compartido según el integrador | «Desconocido ≠ 0» del registro no cierra el bug de `signum(0)` del multiactivo; preservar sus cambios posteriores de OU y no cambiar su checkout |

Los avisos se publican en el buzón
[`COORDINACION_CODEX_2026-09-28.md`](../../COORDINACION_CODEX_2026-09-28.md).
Una reserva observada no prueba un acuse nuevo. El plan rector de AGY
[del 11 de octubre](../PLAN_MAESTRO_REVISION_BASE_UNIVERSO_ESPECTRAL_2026-10-11.md)
declara Rust para motores, análisis y pruebas; es una declaración de AGY,
sin atribuirla aquí a una instrucción humana nueva. Las correcciones y
contratos de este lote se escriben en Rust, con PowerShell para Git y recibos. Se
preservan las herramientas y evidencias históricas de otras líneas.

## 2. T1 histórico y reconciliación de OU

T1 de la composición Ω28 **terminó y pasó** sobre
`eb32ac9613c847eec75bd8e2591add8ae88234cf`, árbol
`10ecee6c773a37d6ac6a0ff7908f3c2c800d8ea0`: 16/144 genes sensibles,
11,1 %, mínimo 11,0 % intacto; 1 test aprobado, 0 fallidos y 1 filtrado.
Duración del test: 1 481,02 s; duración total registrada del comando:
1 619,667 s. El recibo `48-omega28-t1.json` identifica HEAD y árbol antes
y después, comando, entorno y exit 0. Su log tiene SHA256
`864c7daf715b4536288d06287e07b31e25933ac59843a26d8ca44a9fd41cef68`.
Ambos archivos se conservan en `target/integration-receipts/2026-10-08-refresh/`
del worktree histórico `integration-recovery`; el integrador debe respaldar
y publicar el recibo verificable que cierre su alcance; la conciliación
actual se documenta en [cierre de T1 e integración](R4_CIERRE_T1_INTEGRACION_2026-10-11.md).

`eb32ac96` no es ancestro de la base `1d1f8bd`; eso no significa que su
arreglo se perdiera. En esta base, `multivariate_coint.rs:125` ya rechaza
`sde.count > 0 && timestamp_ms <= sde.last_ts_ms` antes de consumir el
z-score. El test `omega28_consumer_clock_contract.rs` conserva el blob
`f8812c55c5a5c2fbfb1152576c7e5886b5eea320`. El consumidor evolucionó después:
vida media hasta 12 h, curva de confianza y media previa del regresor.
La integración debe verificar convergencia y recuperar sólo los aportes
faltantes, preservando esas evoluciones. Una fusión amplia que reintroduzca
el código anterior no constituye una reparación.

El T1 histórico acredita aquella perturbación/fixture y revisión. No se
hereda como prueba de la base actual. El oráculo usa 1 000 USD y no ejecuta
el host; sus 16 genes sensibles tampoco certifican cobertura de todas las
puertas, el régimen de 13 USD o rentabilidad OOS. Una ejecución nueva exige
fuentes, compilador, ejecutable y resultados identificados.

## 3. Conceptos, unidades y ruta causal

La ruta observada en la base es:

`TemporalSpectrum → último bloque maduro → EspectralMultiactivo →
coherencia_media_con_todas → publicar_coherencia → qo_613_rho_tau →
rho_final → correlation_guard`.

El núcleo alimenta maduraciones por escala (`god-engine-core/src/lib.rs`,
`observar_maduracion`) y publica coherencia en sus rutas de evento, tick
dual y feedback. `evidence_publication::publicar_coherencia` elige el nodo
de menor distancia absoluta a la τ dominante. El riesgo usa ese IC sólo
para apretar una base escalar existente cuando el IC la supera; no basta
con que un test abra el gate del estimador para demostrar que una orden
real fue vetada.

La malla implementada en `temporal_spectrum.rs` es
`τ_k = 10^-6 ms · 4^k = 1 ns · 4^k`, `k=0…31`, hasta aproximadamente
146,15 años. La cifra 208 días corresponde al nodo 27, no al extremo del
array. El feed tiene resolución mínima declarada de 1 ms: una malla
numérica fina no produce observaciones independientes de nanosegundos.
Las salidas operativas heredadas siguen acotadas a 30 s…12 h. Las frases
`4^k µs` y «1 µs…208 días» de planes recientes no describen este array.

Separar tres estados: retorno cero observado, ausencia de dato y dato no
finito. Los ceros válidos pueden actualizar momentos/conteo; su factor de
apuesta debe ser 1. Los no finitos y timestamps inadmisibles se rechazan
antes de mutar el reloj, el último retorno o los pares acumulados.

## 4. Matemática: reparación mecánica y nulo estadístico

El defecto de ceros nace en `IcEscala::acumular`: `(ra * rb).signum()`
convierte `+0.0` en `+1.0` y `-0.0` en `-1.0` antes de que `EProceso`
pueda aplicar su abstención. Con una pareja informativa positiva seguida
por ceros, los momentos pueden conservar IC=1 mientras el capital de
apuesta sube artificialmente. El contrato debe preservar ceros antes de
la transformación de signo y comprobar invariancia al bit de signo.

El replay nace en el guard `previo.0 == ts_ms`: un timestamp menor puede
reemplazar el bloque más reciente y volver a sumar observaciones antiguas.
La identidad temporal debe crecer estrictamente por moneda y escala;
todo replay o regresión se descarta antes de cualquier mutación. El
consumo de bloques por par debe mantener la misma disciplina causal.

Para el producto de apuestas `M_t = M_(t-1) (1 + λ_t S_t)`, una condición
suficiente es que `M_0=1`, cada factor sea no negativo, `λ_t ≥ 0` sea
previsible y `E[S_t | F_(t-1)] ≤ 0` bajo cada distribución del nulo.
Entonces el capital es supermartingala; Ville controla el cruce secuencial
del umbral. Hay que definir la filtración y el momento de la apuesta en
la llegada asíncrona de ambas monedas.
[Definiciones primarias de Ramdas, ICML 2025, pp. 1–2](https://stat.cmu.edu/~aramdas/icml25/evalue-checklist.pdf).

Ni independencia de bloques ni una etiqueta «Ville» prueban por sí solas
ese contrato condicional. La publicación original diferencia procesos
adaptados y apuestas previsibles y permite datos dependientes cuando se
satisfacen sus condiciones.
[Ramdas, Grünwald, Vovk y Shafer, 2023, §§1.1–1.2](https://safestatistics.com/wp-content/uploads/2023/10/RamdasSAVIStatScience23.pdf).

Dos contraejemplos algebraicos propios mantienen abierta la revisión del
nulo declarado en el código:

- `Y=0` con probabilidad 0,51 y `Y=1` con 0,49 tiene mediana 0, pero
  `E[sign(Y)]=0,49`: con λ=0,1, el factor medio es 1,049.
- `Y=1` con probabilidad 0,9 y `Y=-9` con 0,1 tiene media 0, pero
  `E[sign(Y)]=0,8`: el factor medio es 1,08. Un producto cruzado medio
  cero no implica equilibrio de sus signos.

Corregir ceros y replay no demuestra validez anytime de las afirmaciones
«mediana ≤ 0» de `evalues.rs` o «IC=0» de `espectral_multiactivo.rs`.
Esa calibración y la política de muestreo permanecen en C-F2/R2. Las
analogías físicas y las teorías nuevas siguen el contrato de transferencia
y medición de [ADR-0005](../adr/ADR-0005-triage-teorico-y-medicion.md).

La cobertura de familia es otro expediente. El código conserva
`M=C(30,2)·5=2175`, `α=0,05`, umbral 43 500 y λ=0,1. La selección por
vecino absoluto con τ entre 30 s y 12 h puede consultar los seis nodos
17…22; por ejemplo, 30 s elige 17. Una capacidad de 30 monedas y seis
nodos produciría 2 610 celdas; un roster fijo de 26 produciría 1 950.
Esto exige demostrar el universo efectivamente consultado por época y
congelar una familia que lo cubra. Este lote no reduce el umbral ni
confunde la reparación mecánica con el cierre de ese expediente.

**SI-O02, abierto: escala e inversión del pronóstico.** En la línea Si,
madurar con un objetivo `log(Y + suelo)` y publicar `sqrt(exp(z))` para
σ o `exp(z)` para volumen requiere aclarar el desplazamiento: la
exponencial representa la variable transformada con su suelo, no
automáticamente `Y`. Deben identificarse el suelo y la normalización
causales al emitir y al madurar, y una transformación inversa consistente
con el estimando y sus unidades. Éste es un contrato de escala distinto
del sesgo de Jensen: incluso una inversión coherente de una predicción
de `E[log(Y + suelo)]` no la convierte en `E[Y]`. El primer all-targets
de Si no prueba ese contrato; se necesitan sus casos de escala conocida,
publicadores y consumidores. No se afirma que SI-O02 esté corregido ni
se modifica su código desde este lote.

## 5. Fases y pruebas de aceptación

| Fase | Relación con los planes | Entregable y criterio |
|---|---|---|
| A. Base y reservas | R0/C-F0 | SHA de fuentes, censo actual, ramas por OID, acuses y evidencia T1 histórica sin transferirla |
| B. Conceptos | R1/C-F1 | Identidad moneda/escala/bloque, cero/ausencia/no finito, tiempo de evento y ruta productor→consumidor |
| C. Matemática y estadística | R2–R3/C-F2 | Factor neutral de ceros; estado previsible; nulo y familia declarados abiertos con contraejemplos |
| D. Contratos de código | R4–R5/C-F2/C-F10 | Test que falla en la base por cada bug, sin alterar λ, α, M ni los controles positivos |
| E. Replay y continuidad | R4/C-F10 | Duplicados, retrocesos, replay completo, orden asíncrono, escalas/monedas independientes y continuación fresca |
| F. Regresión y resultados | R7/R9/C-F10 | Suites afectadas, all-targets/locked, contrato OU, T1 aplicable y pruebas de publicación/consumo; recibos por SHA |
| G. Publicación y limpieza | R9/C-F11 | Comparación contra cada padre, main local/remota verificadas y retirada sólo de refs cerradas; conservar worktrees/evidencia |
| H. Recibo por archivo | R0/R8/C-F10–11 | Blobs e intervalos revisados, límites del veredicto y delta del inventario; ningún archivo certificado por listado automático |

Contratos causales mínimos, con controles que impiden «arreglar» la prueba
suprimiendo toda evidencia:

- Una pareja positiva y 112 parejas con retornos cero no abren el veto;
  variantes `±0.0` y una sola pierna cero preservan el mismo capital neutral.
- 112 parejas positivas únicas no bastan: `1,1^112≈43249,46<43500`.
  La pareja 113 sí basta: `1,1^113≈47574,41`. Se comprueban conteo y gate.
- Alimentar 40 bloques y luego repetir prefijo hasta 110 produce 110
  parejas únicas, no 150; el gate sigue cerrado. Tres parejas frescas
  adicionales lo abren con el umbral original.
- Un timestamp igual o menor no cambia último bloque, momentos, capital
  ni consumo de pares. Cambiar el retorno del replay tampoco lo hace.
  Después, un timestamp fresco sí avanza; invertir el orden de llegada
  de las monedas no altera la identidad del par.
- Cuando corresponda, comprobar la cadena de publicación `qo_613_rho_tau`
  y el consumidor del veto, conservando base escalar y precondiciones
  suficientes para probar el efecto. Distinguir este contrato de un
  backtest multiactivo real o de una medición de coste de oportunidad.

El test histórico `qo_665_ic_veto_exige_capital_de_familia` repite 40
observaciones al llamar otra vez a su feed desde cero. Debe convertirse
en control de conteo y replay: no bajar el umbral para conservar una
afirmación anterior de «110/120 perfectas». El cierre precisa RED y GREEN
de la misma fixture, exit codes, fuente exacta y logs con hash.

El lote contiene siete contratos privados en
`crates/quantum-arena/src/espectral_zero_replay_contract.rs` y dos contratos
de publicación/consumo reales en
`crates/god-engine-core/tests/spectral_zero_replay_contract.rs`. El RED
unitario ya registró 2 aprobados y 5 fallidos; el RED del consumidor sigue
sin conclusión en este corte. La fixture v1 contempla publicación/apriete;
su cantidad y riesgo sí permiten cruzar el límite, pero sus precondiciones
no garantizan la admisión de base. No se declara reproducido un cambio de
aceptación/rechazo de la orden: requiere el contrato reforzado y su propio
RED previo a la reparación. Esa cuenta no constituye una ejecución GREEN.
Los recibos finales deben separar reproducción, reparación y regresión,
con su alcance exacto.

## 6. Verificación, publicación y continuidad archivo por archivo

Usar `nightly-2026-06-30`, `--locked` y un solo dueño por target de build.
El integrador usa `target/r4-build` de su worktree para el RED del
consumidor y la validación posterior, separado del target de Si y de
otras líneas. Con targets
compartidos, identificar y conservar cada ejecutable; cargo puede reutilizar
un binario de otro worktree. La lección CL-60b/`scripts/t1_oraculo.sh` del
PR #38 invalida comparaciones de base/rama que ejecuten el mismo binario
con fuentes distintas. Registrar versión del compilador, revisión, árbol,
fuentes, comando, entorno, resultados y hash del ejecutable/log.

El paso de compilación es `cargo +nightly-2026-06-30 check --workspace
--all-targets --locked`. Las suites se eligen por el cambio real y sus
consumidores. El medidor de `process_event` de Línea C es evidencia distinta
del contrato funcional. Para cambios de su camino caliente, adjuntar la
comparación frente a la base con binarios distintos por revisión y máquina
quieta, y declarar entorno e incertidumbre; una comparación no comparable
no equivale a una puerta aprobada. No
heredar las afirmaciones de «<25 ns, cero heap» del plan como mediciones.
Conservar la política CI que no cancela corridas de `main` por pushes nuevos.

El [borrador de sincronización](BORRADOR_SINCRONIZACION_R4_2026-10-11.md)
propone textos para memoria, buzón y plan compartido; sigue fuera del
índice y no acredita su incorporación a esos documentos ni a `main`.

Antes de cada commit, `git status` y stage explícito de archivos propios.
Antes del merge, actualizar `origin/main`, reconciliar cambios posteriores,
revisar el resultado frente a cada padre y compilar all-targets. Después,
verificar que el SHA publicado es el anunciado y que todas las piezas
propias están en su árbol. La limpieza requiere ancestría/OID, ausencia
de cambios exclusivos y cierre de reservas; no borrar una rama reutilizada
por otra línea sólo porque su HEAD sea ancestro. Conservar directorios de
worktrees, archivos desconocidos, receipts y cachés de evidencias.

El corte `1d1f8bd` contiene 1 583 archivos versionados y 533 Rust (censo
`git ls-tree -r --name-only`, no lectura completa). El
[TSV de Línea C](PLAN_LINEA_C_RECIBOS_2026-10-11.tsv) contiene 1 561 filas
de archivos de `1025f750`; sigue siendo un snapshot histórico. La siguiente
revisión debe calcular su delta y conservar una fase dueña por archivo,
OID del blob, lector, revisor, intervalos cubiertos, contrato, productores
y consumidores, hipótesis, reproducción, tests y veredicto. Un cambio en
productor, datos, compilador o política puede invalidar evidencia del
consumidor aunque su propio blob no cambie. El recorrido completo se
mantiene en los planes existentes; este sublote sólo certificará sus
contratos concretos cuando existan sus recibos.

Al cerrar, enlazar desde plan compartido, buzón y memoria: SHAs de reparación
y publicación, resultados RED/GREEN y regresión, T1 histórico/fresco con
alcances separados, ramas retiradas/preservadas y expedientes abiertos de
nulo, familia, malla, latencia y validación económica. Los estados pendientes
se actualizan con evidencia, sin reescribir aportaciones de otras líneas.

## 7. Adenda conceptual: estimando del IC y correspondencia con la cartera

Hallazgos abiertos de lectura estructural, referidos exclusivamente a la
base `1d1f8bdd35640e5ab12b6e55edb9bf582c41c33a`. No sustituyen el corte
histórico de pruebas ni amplían la reparación mecánica de ceros/replay.
Los enlaces abren las rutas del worktree; los blobs e intervalos siguientes
identifican la versión leída mediante `git show <base>:<ruta>`:

| Fuente de la base | Intervalos usados | Blob de `1d1f8bdd` |
|---|---|---|
| [espectral_multiactivo.rs](../../crates/quantum-arena/src/espectral_multiactivo.rs) | 93–118: apuesta y momentos; 315–339: media con todas las monedas | `a480b0d2cc3449fc07ca569dbd0606af84fcbf8c` |
| [correlation_guard.rs](../../crates/risk-engine/src/correlation_guard.rs) | 126–160 y 184–220: fallback Pearson; 253–326: HY; 612–720: slots, signo, selección y ajustes | `e51b53de148d83d69be3065d42b7d80a738b6880` |
| [risk-engine/src/lib.rs](../../crates/risk-engine/src/lib.rs) | 594–653: dependencia y riesgos; 675–705: apriete espectral | `337b5c822732ace3bed1faa18fe846bf0725148a` |
| [evidence_publication.rs](../../crates/god-engine-core/src/evidence_publication.rs) | 52–76: publicación escalar por moneda a la escala más próxima | `b0d34fe01fe4e08476ed05e8def2996fc4b3d30b` |

**R4-O01, abierto: momento normalizado y nulo de la apuesta.** El IC
implementado es `eab / sqrt(ea2·eb2)`, con momentos exponencialmente
ponderados y sin restar las medias. Su nombre no lo convierte en la
correlación de Pearson centrada. Como contraste algebraico poblacional,
`X` e `Y` independientes, cada uno equiprobable en `{1,3}`, tienen
`E[X]=E[Y]=2`, `E[X²]=E[Y²]=5` y `E[XY]=4`. Por tanto, Pearson es 0 y el
momento cruzado normalizado es `4/5=0,8`; una EWMA finita no tiene por qué
coincidir exactamente con ese valor poblacional. Todos los productos son
positivos, la señal previa del código es positiva y el score de la
apuesta es +1. Este ejemplo no es un falso positivo bajo el nulo
condicional del score: `E[S_t|F_(t-1)]≤0` no se cumple aquí. Sí impide
interpretar independencia, Pearson cero y el nulo de esa apuesta como
afirmaciones intercambiables. El expediente debe identificar el estimando
y el nulo pretendidos antes de elegir centrado, transformación o
calibración; no se rebautiza el IC como Pearson ni se cambia en este lote.

La base escalar de riesgo tampoco es siempre Pearson: primero intenta
`hayashi_yoshida_correlation`, que suma productos de incrementos
solapados y normaliza por sus variaciones cuadráticas sin centrarlos. Si
esa ruta devuelve `None`, recurre a `correlacion_de_retornos`, cuyo
fallback en rejilla sí resta medias mediante `pearson`. El contrato
futuro debe distinguir ambas ramas, sus soportes temporales y el efecto
del drift; no deducir equivalencia por el nombre de un test o comentario.

**R4-O02, abierto: IC de precios frente a dependencia de PnL.**
`coherencia_media_con_todas` promedia por igual los pares que superan el
gate, recorriendo todas las monedas. No recibe posiciones abiertas,
lado, cantidad ni riesgo al stop. `publicar_coherencia` escribe un único
`qo_613_rho_tau` por moneda y escala seleccionada; la consulta de ese
escalar no distingue candidata Long de Short.

En cambio, `dependency_exposure` recorre slots abiertos, transforma la
dependencia de precios con el producto de signos de candidata y posición
(`candidate_long` frente a `slot.is_long`), selecciona la misma apuesta
con `es_la_misma_apuesta` y conserva riesgos individuales al stop. Su
base efectiva incorpora además amplificación de contagio Hawkes, la
señal registrada como Hodge/curl y dependencia de colas. Leer esas claves
no certifica sus teorías ni calibración. Comparar aquel promedio de IC
de precios con esta ρ efectiva de PnL no demuestra que compartan universo,
signo, τ, pesos ni soporte de observación.

Consecuencias estructurales que requieren contratos futuros:

- Una moneda sin posición puede elevar o diluir la media espectral sin
  pertenecer al grupo de riesgo seleccionado.
- Una posición de cobertura excluida por la selección de misma apuesta
  puede contribuir a un IC de precios positivo que apriete ese grupo.
- Anticorrelación de precios con lados opuestos se transforma en
  dependencia positiva de PnL. Su IC de precios firmado negativo no
  añade apriete frente a una base positiva; eso no borra la protección
  que ya aporta la base de dependencia de riesgo.

`only-tighten` conserva la base cuando el IC no la supera y nunca la
afloja; esa propiedad no prueba correspondencia semántica entre ambos
estimadores. Esta adenda no declara una orden concreta mal vetada, una
frecuencia de ocurrencia ni una mejora de retorno medida.

| Expediente y fase | Dueño futuro propuesto, pendiente de acuse | Contrato para cerrar |
|---|---|---|
| R4-O01 — estimando/nulo; C-F4/R3 con revisión estadística C-F2/R2 | Línea C para la fase C-F4, con revisión cuantitativa independiente y coordinación Codex; sin nueva reserva de producción | Fixture de soluciones conocidas con drift y sin drift; separar Pearson centrado, HY y momento espectral sin centrar; declarar score, filtración y nulo condicional; medir calibración sólo bajo ese nulo |
| R4-O02 — correspondencia de cartera; C-F4/R3 | Línea C para la fase C-F4, coordinación Codex para publicador/consumidor y revisión de riesgo; confirmar reservas antes de editar | Mapa por candidata, slot, lado, cantidad/riesgo, par, escala, tiempo y selección de misma apuesta; pruebas Long/Short, hedge excluido, monedas sin posición, anticorrelación con lados opuestos y cambios de tamaño; separar ajustes Hawkes/curl/colas y preservar `only-tighten` como control |

Los dueños son una propuesta de enrutamiento, no un acuse ni autorización
para ocupar sus archivos actuales. C-F4/R3 permanece abierto; el mapa de
cartera y los contratos descritos se desarrollarán en un lote propio con
RED/GREEN y mediciones aplicables. El arreglo presente conserva estimador,
publicador, política de riesgo y parámetros de apuesta.

**R4-EPOCHS, abierto: historia, épocas y frescura.** Los momentos olvidan
con paso 1/64, mientras el capital E acumula los factores de toda la
historia. Tras corregir el factor de cero, 120 pares idénticos positivos
seguidos de ceros conservan `1,1^120≈92 709,07>43 500` y razón IC=1 en
aritmética exacta: no prueban dependencia vigente. El tramo alternativo
pasado hace falso el nulo histórico; esto no se describe como falso
positivo bajo un nulo válido. Deben distinguirse el objeto histórico del
actual y su frescura: retirar una decisión por política de frescura es
distinto de modificar el e-proceso. Un descuento determinista en `[0,1]`
puede preservar una supermartingala no negativa bajo un nulo válido, pero
no convierte un rechazo histórico en certificación actual. En la base
`1d1f8bdd`, el constructor del core crea un espectral nuevo en 1129–1131;
`reset_engines` de reconexión, 1294–1299, sólo reemplaza feature engines y
limpia el último libro, conservando el espectral. Esta lectura no acredita
persistencia completa. El expediente sigue abierto: no se implementan
descuento ni reset ni se afirman frecuencia o PnL medidos.

| Deuda futura y dueño propuesto, pendiente de acuse | Contrato de cierre | Fuente de la base `1d1f8bdd` |
|---|---|---|
| Objeto actual/histórico y frescura — Línea C C-F4/R3, revisión cuantitativa C-F2/R2 y coordinación Codex | Alternativa→ceros, alternativa→ruido y alternativa→signo opuesto; identificar qué afirma el gate y cuándo retirar su uso operativo sin confundirlo con una modificación del capital | [espectral_multiactivo.rs](../../crates/quantum-arena/src/espectral_multiactivo.rs), 75–77 y 93–124; [evalues.rs](../../crates/quantum-arena/src/evalues.rs), 87–124, blob `758ff0af9725b73ffdbb054e84e3e025c0435a94` |
| Reconexión, restart y replay — coordinación Codex/core con el dueño de la futura fase de persistencia | Contrastar identidad de época, relojes, pares consumidos, momentos y capital tras reconexión frente a reconstrucción/reinicio; probar replay y continuación fresca; delimitar el estado que realmente se conserva o restaura | [god-engine-core/src/lib.rs](../../crates/god-engine-core/src/lib.rs), 1129–1131 y 1294–1299, blob `5fb67b3b572a9ea670242163366a21f94c8619de` |
| Presupuesto de error por épocas — revisión estadística C-F2/R2 con Línea C C-F4/R3 | Si se abren procesos por época, definir causalmente su inicio y nulo, exigir `Σα_epoch≤α_global` y corregir cada familia con su `M_epoch` realmente consultado; no reutilizar un presupuesto completo en cada reset ni tomar evidencia alternativa antigua como certificación vigente | [evalues.rs](../../crates/quantum-arena/src/evalues.rs), 111–130; contrato condicional y familia abiertos en §4 de esta adenda |

## 8. Corte de seguimiento 2026-10-11, 15:35 UTC: ejecución todavía abierta

El último main remoto observado en este corte es `8dd51dfb32481ba900038ee279bfd61f45e1b9b4`. El WT R4 sigue en `1d1f8bdd35640e5ab12b6e55edb9bf582c41c33a`; no se ha aplicado ni publicado el parche. El RED unitario terminó con 2 aprobados y 5 fallos esperados. El RED consumidor `02-red-consumer` sigue compilando en su target privado: todavía no hay resultado de tests. Los diez hashes de fuentes del recibo inicial se compararon sin cambios. Esta adenda no modifica el índice ni las fuentes congeladas.

T1 histórico sigue cerrado en eb32: 16/144 y ratchet 0,110 intacto; el agregado histórico 42–48 es 334/0/1. No prueba la composición actual. La sensibilidad genética usa fixture sintética y ocho estadísticas; el veto multiactivo tendrá controles propios de publicación, admisión y rechazo.

SI publicó `fcda9d262a7ec529cfe2c2d16e38d994e2ea3258` en [PR #39](https://github.com/Jhona-la/Trader-Gemini/pull/39), componiendo 6b1d con Ω83/8dd. Consulta propia de GitHub: riesgo, núcleo, espectro/consejo y entrenador aprobados; replay en curso; PR abierta. SI reporta 252 tests locales de arena/registro y T1 todavía compilando. Se conserva main hasta su aviso de integración. El reinicio global y su inventario pendiente se unirán por bloques añadidos, preservando las tablas de los otros agentes.

Borradores verificados con `git apply --check`, aún sin aplicar: fixture v2 SHA256 `b9346bee3b4d7eb34884ba887d9e66803a78118090c632c8328b50e2fef8d97a`; parche productivo SHA256 `a1966e7939deff2bf99e375598b48ee2b8006dbf2888ed23320841b34a9261c5`. La [documentación primaria de Rust](https://doc.rust-lang.org/std/primitive.f64.html#method.signum) confirma que signum transforma los ceros con signo en direcciones +/-1. Pasar el producto sin esa transformación permite conservar la abstención ya implementada en EProceso.

Se mantienen abiertos el nulo condicional, la familia efectiva, R4-O01, R4-O02 y R4-EPOCHS. Los contrastes R0/R1 de SI sobre presupuesto neto, barrera de drawdown y timestamps del registro siguen separados como hallazgos delimitados, sin impacto financiero medido. RAM limitada: una sola cadena pesada propia, jobs=2, pruebas posteriores secuenciales; no se detienen procesos ajenos ni el motor.

## Corte posterior de R4 — 13 de octubre de 2026

La validación vigente se ejecutó en `f62590e33a76a61a67320fed0e1ecc9ae0514ebd`: 713 pruebas aprobadas, cero fallos y siete ignoradas. T1 exacto pasó con 18/144 genes sensibles (12,5 %), cinta sintética, mínimo 0,110 intacto y un diagnóstico filtrado; no mide rentabilidad. Fuentes, genomas y candidatos de modelos permanecieron estables.

La composición con LH12 `92792116` quedó en `4191c65f`; se resolvieron dos conflictos documentales conservando historia y se aprobó all-targets en release antes del merge. La equivalencia posterior preservó los 664 inputs compilables y 637 Rust de T1. La publicación en main sigue pendiente en este corte local; su estado final se consulta en la [adenda del 13 de octubre](ADENDA_CIERRE_R4_2026-10-13.md).

Los recibos del 8 y 11 de octubre conservan sus fechas y alcances originales. Toda auditoría de archivo completo continúa pendiente; sólo los hunks de ceros/replay y sus contratos tienen evidencia mecánica acotada. Nulo condicional, familia efectiva, frescura/épocas y representación de cartera permanecen abiertos.