# Auditoría desde la base y coordinación del consejo — 8 de octubre de 2026

La meta del operador es duplicar patrimonio neto cada 72 horas desde 13 USD.
El objetivo de esta ronda es construir y comprobar la cadena que permite
medir esa hipótesis, corregir contratos rotos y recorrer cada archivo con
evidencia. El objetivo financiero sigue sin validación económica; las
pruebas de software y T1 tienen estimandos distintos.

Este es el punto de entrada del plan. El [recorrido por archivo](PLAN_REVISION_ARCHIVO_POR_ARCHIVO_2026-10-07.md)
define los recibos y las fases; el [plan compartido](PLAN_MAESTRO_SINCRONIZACION.md)
registra responsabilidades y reservas. Los expedientes anteriores se
conservan con sus fechas y bases. Prevalecen las adendas fechadas para el
estado operativo, sin transferir evidencias entre versiones.

## Base, medición y estado

Base de código integrada: `e9c6a435eb1780da7f2c6e29103f3bd7d227d94c`; árbol `10b8a427dfbdd5b203595b039b2dad50c69d6f68`.
La publicación, pruebas y limpieza tienen recibos separados en
[integración](INTEGRACION_RAMAS_2026-10-07.md). El
[ledger de código](audit/REVISION_CODIGO_2026-10-08.json) fija todos los
archivos de su snapshot: censo no significa auditoría completa. Cada blob
modificado invalida su revisión; el revisor debe invalidar también las
dependencias semánticas afectadas aunque sus blobs no hayan cambiado.

La identidad aspiracional es `C(t)=13*2^(t/72h)`: `ln(2)/3` de crecimiento
log diario y `25.992105%` de rendimiento diario compuesto equivalente.
Medir capital realizado + todas las posiciones abiertas a precios y tiempos
identificados − costes pendientes; separar depósitos y retiros. Reportar
distribución de retornos de 72 h, periodos sin operar, incertidumbre,
drawdown, expected shortfall, ruina bajo un modelo declarado y capacidad.
«Consistente» necesita un criterio preregistrado de frecuencia y severidad
de incumplimiento. Ventanas disjuntas tampoco garantizan independencia.

## Concepto rector

El sistema estima un campo causal `X(activo, tiempo de evento, escala,
procedencia)` con incertidumbre. Horizonte, volatilidad, liquidez y
dependencia se tratan como coordenadas continuas; una malla finita debe
declarar banda, resolución, interpolación y error. Los nombres legacy no
demuestran particiones económicas; renombrarlos tampoco prueba continuidad.

Seguir cada contrato desde evento y reloj hasta estado multiactivo,
predicción, decisión, veto, orden, fill, patrimonio, evaluación y aprendizaje.
Una política autoevolutiva debe demostrar etiquetas maduras, información
disponible en cada decisión, identidad de modelos/genoma, selección válida,
publicación atómica, consumo efectivo y rollback. Las reglas de adaptación
se preregistran; sus parámetros pueden actualizarse causalmente durante el
examen según esas reglas. Toda consulta y actualización queda registrada.

## Fases y criterio de salida

| Fase | Prioridad y trabajo | Evidencia requerida |
|---|---|---|
| R0 — metas | Magnitudes, criterios económicos, SHA, roster, modelos/datos y reservas | Especificación y censo completo; owners confirmados o marcados como propuestos. |
| R1 — conceptos | Topología real, consumidores, estado/tiempo y rutas vivo/sombra/replay | Trazas productor→consumidor→efecto; contratos de identidad y unidades. |
| R2 — matemática/estadística | Derivaciones, dominios, dependencia, selección, filtración y familias | Nulos/contraejemplos reproducibles; incertidumbre y supuestos verificables. |
| R3 — física/cuántica/algoritmos | Teorías transferibles, discretización, estabilidad, complejidad | Observable y predicción falsable, baseline clásico, ablación y coste medido. |
| R4 — datos/reloj | Duplicados, retrocesos, huecos, sincronización, warm-up y OOS | Replays adversos; rechazo sin mutación; disponibilidad causal por etiqueta. |
| R5 — capital/vetos/ejecución | Contabilidad MTM, costes, sizing, reservas, margen y fills | Conciliación por evento y terminal; composición de fallbacks y contrafactual en sombra. |
| R6 — aprendizaje | Sensibilidad genética, linaje, presupuesto de ensayos, promoción/reinicio | Cambio efectivo de parámetros, evaluación prequential, promoción y rollback trazables. |
| R7 — plataforma | Compilación, concurrencia, presión, recuperación y rendimiento | All-targets; comportamiento bajo fallos; p50/p95/p99 y asignaciones medidas. |
| R8 — cobertura restante | UI, scripts, manifests, derivados y documentación | Revisión individual por blob/esquema/procedencia/consumidor, más su generador. |
| R9 — integración/economía | Composición, regresiones, T1 aplicable y examen separado de selección | Diff contra ambos padres, recibos exactos; publicación verificada; informe económico OOS. |

R0→R1→R2/R3 preceden interpretar el código como teoría válida. Después se
priorizan rutas capaces de perder capital o contaminar evidencia. Cada lote
abarca aproximadamente 10–20 archivos, dividido por contrato si el módulo
es grande. Su dueño reserva rutas/base antes de editar; otro senior revisa.
No hay cierre por conteo de matches ni por un único test verde.

El ledger conserva etiquetas F0–F8 para agrupar superficies y proponer
responsables. R0–R9 define el orden de evidencia de esta ronda; ambos ejes
se cruzan por contrato, sin equiparar automáticamente sus números.

Por archivo: blob/commit, rangos leídos, contrato, ecuación y unidades,
productores/consumidores, invariantes, hallazgo/severidad/certidumbre,
reproducción, fix, RED/GREEN, entorno/datos/semilla, límites, revisor y SHA
integrado. Revisar una función no equivale a cerrar el archivo completo.
Conservar hallazgos abiertos y errores económicos, incluida insolvencia.

## Primeros lotes concretos

| Lote | Rutas principales | Contrato de salida |
|---|---|---|
| Medición económica | `booktick_replay.rs`, `metrics.rs`, `darwin.rs`, reporting continuo y contabilidad del core | Equity común multiactivo, DD intraventana y terminal, fronteras de warm-up/72 h, costes reconciliados. |
| Inferencia y selección | `selection_stats.rs`, Darwin y online daemon | Dependencia temporal y entre ensayos, observaciones inválidas, identidad de muestras y genealogía; calibración bajo nulos antes de interpretar DSR. |
| Tiempo y OU | Parser, `ws_client.rs`, arena state, `multivariate_coint.rs`, `vecm_arbitrage.rs` | Timestamp ausente/inválido y no creciente; rechazo sin mutación; modo OU exclusivo o híbrido explícito y duración de toda señal. |
| Espectro y vetos | `espectral_multiactivo.rs`, `evalues.rs`, publicación y callers | Dominio realmente alcanzable, capacidad/configuración, familia completa y evidencia única madura; contrafactual por veto sin rebajar límites para cumplir la meta. |
| Genoma y modelos | `evolution.rs`, curvas, `model_reload.rs`, loaders/manifests | Mutaciones cambian parámetros efectivos; identidad contenido/cache/fuente, revocación y reintento; presupuesto acumulado entre rondas/reinicios. |

Los ocho archivos con hallazgos parciales abiertos y el cierre completo del contrato C07
quedan en el ledger; no representan cierre completo de fuentes runtime.
Los [fundamentos revalidados](audit/REVALIDACION_FUNDAMENTOS_R4_2026-10-08.md),
[protocolo OOS](audit/PROTOCOLO_OOS_R4_2026-10-07.md) y
[diagnósticos OU](audit/REVISION_MERGE_CORE_OU_REGISTRY_2026-10-08.md)
fijan reproducción, alcance y límites. Los testigos Rust son repetibles con
`python scripts/audit_r4_rust_witnesses.py --repo .`. El script fija fuentes
de `18bbd1d9`; sus aserciones verdes caracterizan defectos de esos blobs,
sin repararlos. Reejecutarlo desde otro HEAD no verifica ese HEAD: trasladar
el diagnóstico exige comprobar igualdad de blobs y dependencias del nuevo
candidato, como hace la revalidación enlazada.

## Consejo y sincronización

Codex coordina la integración y documentación; los subagentes separan
reconciliación Git/pruebas, fundamentos científicos e inventario/revisión.
Qoder y Antigravity conservan sus reservas activas observadas. GLM112 y
el reporting de Sol están incorporados; se preservan sus contribuciones.
Las asignaciones futuras por dominio son propuestas hasta acuse, nunca
actividad asumida por el mero nombre de una rama.

Publicar ruta, dueño, base, contrato, prueba prevista y estado antes de
comenzar. No editar simultáneamente el mismo consumidor. Resolver el choque
de knobs cuánticos Qoder/GLM mediante decisión de contrato y prueba, antes
de importar trabajo en vuelo. El buzón local avisa de inmediato y
`COORDINACION_CODEX_2026-09-28.md` guarda la bitácora versionada; un aviso no
demuestra recepción.

Trabajar en ramas propias, stage explícito y commits atómicos. Comparar
cada merge contra ambos padres y compilar all-targets antes del commit.
Antes de publicar, conservar las regresiones afectadas y T1 con trinquete
intacto. Verificar el SHA remoto; borrar sólo refs cerradas por ancestría
y OID, excluyendo actividad y preservando bytes de checkouts sucios.

## Transferencia de teorías

Investigar control estocástico/crecimiento log, procesos OU/saltos/primer
paso, wavelets y multirresolución, Hawkes, matrices aleatorias/cópulas/grafos,
inferencia secuencial y aprendizaje online. Cada propuesta requiere decisión
útil, observable, unidades, dominio, estimador, supuestos, baseline, falsación,
complejidad y efecto incremental OOS neto de costes.

Los problemas del milenio son fuentes de ideas que exigen una correspondencia
demostrable. Hodge sobre grafos y la conjetura algebraico-geométrica son
objetos diferentes; un algoritmo clásico con nombre cuántico no acredita
ventaja de hardware cuántico. Una teoría entra al sistema cuando su contrato
es implementable y su evidencia mejora una comparación justa.


Estado de validación: 736 regresiones aplicables + T1 (1 test exacto, 16/144 sensibles, mínimo 0,110 intacto); total 737 aprobadas, 0 fallidas y 1 ignorada. 729 aprobadas se retienen por identidad desde 7acf y 8 se ejecutaron en e9, incluido T1. Check all-targets aprobado antes del merge; estos resultados no validan crecimiento económico. Publicación pendiente del recibo remoto.

Continuación durante T1: lectura completa de selection_stats (427 líneas/11 tests)
y parser (462 líneas/19 pruebas existentes), con hallazgos aún abiertos.
El [lote R2](audit/R2_SELECTION_STATS_AUDIT_2026-10-08.md) separa matemática
muestral, fallos numéricos y calibración. El [lote parser/Depth](audit/REVISION_PARSER_DEPTH_R4_2026-10-08.md)
añade R4-I6: mezcla de lados con JSON válido, reproducida sin modificar el
pipeline congelado. La lectura íntegra no convierte esos módulos en verificados.

Revisiones independientes: [ciencias](audit/REVISION_PAQUETE_CIENTIFICO_2026-10-08.md), [Depth y alcanzabilidad](audit/DEPTH_INDEPENDENT_REVIEW_2026-10-08.md) y [ensamblado documental](audit/COPY_PACKET_PREFLIGHT_REVIEW_2026-10-08.md). Cada dictamen conserva el corte leído; el recibo de integración posterior acredita el resultado T1 y separa sus límites.
