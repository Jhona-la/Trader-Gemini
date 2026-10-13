# Adenda R4: validación terminal y continuidad del recorrido — 13 de octubre de 2026

R4 13Oct: composición documental b661dcfaca00a1d46a26ebd7e44b2dc252e6348c; fuente ejecutada f62590e33a76a61a67320fed0e1ecc9ae0514ebd. Agregado terminal: 713 aprobadas / 0 fallidas / 7 ignoradas / 1 filtrada; son ejecuciones registradas, con repeticiones, no pruebas únicas. T1: 18/144 genes sensibles (12.5000%). Ledger: 1816 rutas pendientes, cero archivos completos auditados y sólo tres rutas con hunks ceros/replay verificados. evalues.rs sigue documental pendiente. SI final declarado: 927921160e87a425a793d6a23460ab7f2fbff033; no se transfiere CI de padres ni se certifica main remoto.

[Agregado terminal](RECIBO_VALIDACION_R4_2026-10-13.json), SHA256 **5d5188a7bf6834b7bdbaa62a49f8f2bd1a56522c9545bf633c3c99b03f80848e**; [ledger terminal](INVENTARIO_FINAL_R4_2026-10-13.json), SHA256 **4d00bedb9b3ce5154ad7ed056d4b5ce921a35cb13193ef67dbf3d6edb0981ce1**. Ambas copias conservan bytes exactos. Los logs y binarios crudos referenciados se conservan localmente; estos dos JSON no publican por sí mismos un bundle de ejecutables.

## Evidencia ejecutada

| Recibo | Aprobadas / fallidas / ignoradas / filtradas | Completado UTC |
|---|---|---|
| 12-final-four-libs | 539 / 0 / 2 / 0 | 2026-10-13 06:39:20 |
| 13-final-core-contracts | 59 / 0 / 0 / 0 | 2026-10-13 06:46:47 |
| 14-final-risk-contracts | 69 / 0 / 0 / 0 | 2026-10-13 07:16:27 |
| 15-final-core-persistence | 10 / 0 / 3 / 0 | 2026-10-13 07:17:00 |
| 16-final-leverage-admission | 7 / 0 / 0 / 0 | 2026-10-13 07:17:10 |
| 17-final-ou-consumer-clock | 1 / 0 / 0 / 0 | 2026-10-13 07:17:18 |
| 18-final-multiactivo-known | 1 / 0 / 2 / 0 | 2026-10-13 07:17:48 |
| 19-final-backtest-contracts | 26 / 0 / 0 / 0 | 2026-10-13 07:56:43 |
| 20-final-t1 | 1 / 0 / 0 / 1 | 2026-10-13 08:10:47 |

Se derivan 9 recibos del agregado; 664 inputs compilables coinciden por SHA de bytes, blob y modo con el ledger de la composición. La suma conserva repeticiones de scopes y los tests filtrados/ignorados; no certifica cobertura global, causalidad económica ni todos los archivos. La sensibilidad T1 es una realización del fixture sintético y del oráculo identificado, no rentabilidad, suficiencia estadística ni prueba global de genes inertes.

## Alcance mecánico y deudas

Los hunks reproducidos pertenecen a [multiactivo](../../crates/quantum-arena/src/espectral_multiactivo.rs), [unitario privado](../../crates/quantum-arena/src/espectral_zero_replay_contract.rs) y [consumidor real](../../crates/god-engine-core/tests/spectral_zero_replay_contract.rs). Los contratos ejercitan cero físico ±0, replay y timestamps no crecientes; el consumidor mide admisión fría/rechazo caliente bajo su fixture. La prueba no certifica la nulidad o multiplicidad del estadístico. [evalues.rs](../../crates/quantum-arena/src/evalues.rs) sólo recibió alcance documental: su revisión está pendiente en el ledger.

SI-O02 (escala/estimando/Jensen), R4-O01 (IC sin centrar, nulo condicional y familia real), R4-O02 (relación entre precio, cartera y PnL) y R4-EPOCHS (historia, frescura, reinicio/replay y gasto de error) siguen abiertos. Riesgo parte de HY sin centrar y usa Pearson como fallback; la regla de endurecimiento no acredita igual universo, signo, peso u horizonte. Las referencias antiguas a 29 rutas ausentes describen el texto anterior; los 48 ejemplos corregidos existen. Nueva deuda semántica: independencia o centrado de retornos no implican E[χ(s_t A_tB_t)|Fprev]<=0; el comentario multiactivo43 sigue pendiente de corrección documental y evidencia de hunk, sin cambio runtime.

LG22: el producto de factores exige filtración, integrabilidad, nulo condicional E[R_t|F_(t-1)]>=mu, cotas y apuestas predecibles; Ville y ARL del e-detector son garantías distintas bajo sus respectivos supuestos. La formulación marginal E[R]>=mu de un resumen histórico no debe transferirse como garantía general.

Deuda semántica nueva — 2026-10-13: el [comentario multiactivo](../../crates/quantum-arena/src/espectral_multiactivo.rs), línea 43, que atribuye |IC| pequeño a maduraciones independientes es falso para IC=eab/sqrt(ea2*eb2), momento cruzado normalizado sin centrar. Si A,B son independientes y cada una toma epsilon o 3epsilon equiprobablemente, E[AB]/sqrt(E[A²]E[B²])=4/5; Pearson 0, IC 0.8 y score +1 no verifican el nulo del score. Además, para A,B independientes centradas, cada una +epsilon con probabilidad 0.9 y -9epsilon con probabilidad 0.1, E[AB]=0 pero E[sign(AB)]=0.64; con s_t=+1 y lambda=0.1, el factor 1+lambda*χ(s_t A_tB_t) tiene esperanza 1.064. El nulo correcto del proceso espectral exige E[χ(s_t A_tB_t)|Fprev]<=0, con orientación y apuesta predecibles, acotación e integrabilidad según contrato; independencia y centrado de retornos no bastan. Los dos contraejemplos incumplen ese nulo, por lo que no son falsos positivos bajo un nulo válido. El comentario y su evidencia de hunk siguen pendientes; esta adenda no cambia código, límites, política ni runtime.

El promedio de coherencia entre todas las monedas, sin posición/lado/tamaño, no acredita el mismo universo ni signo que riesgo de cartera; una exposición ausente o un hedge excluido puede afectar el apriete. Esta deuda estructural no mide frecuencia, DD o PnL.

R4-EPOCHS: momentos con olvido 1/64 y capital E histórico sin el mismo olvido estiman objetos distintos. Retirar política por frescura y modificar el e-process requieren contratos separados, alternativas que cambian a ceros/ruido/opuesto y gasto de error entre épocas con suma(alpha_epoch)<=alpha_global sobre familia real. Un rechazo histórico no certifica dependencia vigente; un pasado alternativo tampoco es un falso positivo bajo el nulo histórico válido. No se añade aquí decay/reset ni se afirma persistencia completa auditada.

## Historial de ejecución conservado

Se conservan 05 rechazado por drift del índice y su repetición 05b; 06 RED; 07, que reutilizó caché RED tras Copy-Item con mtime antiguo, y 07b, que recompiló tras renovar mtime antes del freeze. Fuente congelada con hashes correctos no prueba por sí sola correspondencia de ejecutable en una caché. El agregado mantiene las referencias RED/excluidas sin sumarlas al conjunto final. Los documentos y recibos Oct8/Oct11 permanecen como snapshots históricos; su T1 no se atribuye a esta composición.

## Recorrido y coordinación

| Fase R | Nombre exacto del ledger | Rutas propuestas pendientes |
|---|---|---:|
| R0 | objetivos y doctrina | 41 |
| R1 | conceptos y estado | 281 |
| R2 | fundamentos y representación | 41 |
| R3 | fundamentos y representación | 96 |
| R4 | datos y tiempo | 55 |
| R5 | capital, vetos y ejecución | 118 |
| R6 | genoma y aprendizaje | 59 |
| R7 | plataforma y rendimiento | 30 |
| R8 | UI, scripts, generados y documentación | 1077 |
| R9 | integración y evidencia | 18 |

R0–R9 y C-F0–C-F11 mantienen nombres y referencias independientes; no son equivalentes por número. Los 1816 estados de archivo completo siguen pendientes: evidenciar tres hunks no cierra tres archivos. Cambios de dependencias pueden reabrir contratos aunque el blob del consumidor no cambie. El [master Oct13](../PLAN_MAESTRO_REVISION_BASE_UNIVERSO_ESPECTRAL_2026-10-13.md) contiene 48 ejemplos existentes y no reemplaza el ledger exhaustivo.

Se preservan reservas actuales de SI/calibration/LH12, AGY, C-F5, G8/H3, Sol y los commits/documentos exclusivos de Qoder. Un apéndice de fase cerrada no equivale a integración ni a liberación de rama. El SHA SI final proviene del argumento explícito de ROOT y no de SI04e histórico; este helper no ejecuta Git ni acredita por sí mismo su ascendencia. La publicación y limpieza remotas requieren recibos separados y verificación fresca.

## Meta y límites científicos

La meta sigue G72>=ln(2), sobre patrimonio neto multivariante MTM y unitizado. g=ln(2)/3=0.2310490602 por día y exp(g*1día)-1=0.2599210499 son identidades distintas. Contrastar probabilidad de cumplimiento, déficit, DD y ruina bajo restricciones/modelos declarados, sin inventar un umbral universal. La malla nominal es tau_k=(1ns)*4^k, k=0..31; datos ms y anclajes de 30s/12h no se vuelven continuos por nombrarlos espectro. El getter de posición conserva tres categorías. Hodge necesita topología/máscara; ecuaciones inspiradas en física requieren unidades/operadores/identificación. No se demuestra hardware cuántico, solución de problemas del milenio, ventaja económica o latencia a partir de estas pruebas.

## Composición, modelos y controles posteriores

T1 se ejecutó en f625 con mínimo 0,110 intacto; detectó 18 respuestas distintas a la huella base `6f346613f56f43ce` entre 144 perturbaciones. Los índices sensibles observados fueron 1, 10, 11, 17, 18, 20, 24, 27, 33, 39, 68, 69, 129, 130, 131, 141, 142 y 143. Los restantes 126 no cambiaron bajo esa perturbación y fixture; esto no demuestra inercia global ni ausencia de interacciones.

El [check de todos los targets en release](CHECK_ALL_TARGETS_R4_2026-10-13.json) pasó antes del merge 4191c65f, con Cargo exit 0 y 25,124 s incluyendo el proceso; quedaron los doce invariantes verdaderos. El intento 21 en debug se interrumpió deliberadamente por omisión de `--release` y no se contabiliza como validación aprobada. Se preservó su recibo y se repitió con etiqueta 21b. Los dos conflictos resueltos afectaron sólo coordinación y plan; ambos bloques se conservaron.

La [equivalencia posterior al merge](EQUIVALENCIA_COMPILABLES_R4_POST_SI_2026-10-13.json) compara 664 rutas, modos y blobs canónicos, además de SHA de bytes, con los inputs de f625. Los 24 candidatos de genoma del check 21b mantienen la identidad de T1. [Modelos antes](MODELOS_T1_R4_2026-10-13.before.json) y [después](MODELOS_T1_R4_2026-10-13.after.json) pasaron con identidad `5364d6d17afb0e80f08d647c76401f21f20cadb41dd2479539c931a1ab94bac4`; el snapshot posterior al merge mantiene esa identidad. Estos inventarios verifican los candidatos declarados, no la activación efectiva en memoria, syscalls, entradas de build de otros tipos ni cambios transitorios restaurados. La [revisión independiente de LH12](REVISION_INDEPENDIENTE_R4_LH12_2026-10-13.json) verificó los nueve recibos, logs y 25 binarios retenidos sin repetir las pruebas.

Los siete ignorados permanecen abiertos: F3.8-02/04 (espejo de primer toque y base de puertas frente al gate), F2.2-05 (selección, cobertura y pérdida entre aceptadas), F2.2-06 (extrapolación Platt fuera de soporte) y F2.3-07 (solape/fase temporal del IC). Sus testigos se conservan; no se reactivaron ni relajaron para presentar el resultado como cobertura completa.

El ledger de 1.816 rutas identifica el snapshot b661 posterior a todos los merges y anterior a esta adenda y sus nuevos recibos. Los documentos añadidos después del snapshot entran en la cola pendiente de R8/R9; no se incluyen a sí mismos ni se declaran auditados. El censo de publicación y el recibo remoto deben conservar esta separación entre código ejecutado, equivalencia de composición y metadatos posteriores.

Los ocho JSON de evidencia de este cierre tienen reglas explícitas -text en .gitattributes para conservar sus bytes CRLF y SHA originales al publicar. Las reglas de Rust, Cargo y los demás archivos permanecen iguales. .gitattributes es un cambio de metadatos posterior al ledger y conserva su revisión pendiente.
