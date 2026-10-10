# Censo de cobertura RA — 2026-10-04

## Objeto y frontera del recibo

[RA_COBERTURA_2026-10-04.tsv](RA_COBERTURA_2026-10-04.tsv) contiene una fila
por entrada del árbol Git `12456048d7c5cbafb23a75a530b3e030d53be913`:
**1435 rutas, 433 bajo crates/**. Fuente: `git -c core.quotePath=false
ls-tree -r 12456048d7c5cbafb23a75a530b3e030d53be913`. No se enumeran archivos
ignorados, no versionados, datos descargados, procesos o modelos externos.
El propio censo y esta explicación son posteriores al snapshot; excluirlos
de él evita una referencia circular. Los conteos anteriores de 1427/1434
pertenecen a otros árboles y se conservan como evidencia histórica.

Campos: snapshot, modo Git, tipo de objeto, OID, ruta y estado. El OID permite
detectar cambios de contenido, no demuestra lectura, exactitud o seguridad.
Todas las filas empiezan como **inventariado_no_certificado**. Esta etiqueta
no niega las revisiones acotadas descritas en el informe; evita convertir la
lectura de un tramo, una búsqueda o una compilación en una certificación de
todo el archivo. No hay un porcentaje de corrección derivado de este censo.

## Cómo convertir inventario en cobertura verificable

Mantener recibos separados y aditivos por `(ruta, OID, contrato)` con lector,
fecha, alcance exacto, hallazgos, consumidores y comandos/salidas. Estados:
inventariado → leído → flujo/teoría contrastados → oráculo ejecutado →
integrado/revalidado. Una transición exige su evidencia; no se avanza en
lote por ejecutar `rg` o `cargo check`. Una ruta puede tener contratos
revisados y otros pendientes. Un cambio de OID exige revalidar el contrato
afectado y sus consumidores, no desechar silenciosamente el recibo anterior.

Método por clase: Rust/Python/CI requiere flujo y semántica; configuración,
unidades/ámbito y consumidores; documentación, reconciliación con fuente;
datos, esquema/causalidad/grano/procedencia; binarios/modelos, hash, formato,
metadatos de entrenamiento/carga y equivalencia del artefacto desplegado.
Inventariar un `.pt` no audita el modelo ni autoriza cargarlo o publicarlo.

## Prioridad desde la raíz

Aplicar [G0–G8](../PLAN_MAESTRO_SINCRONIZACION.md#8-grafo-de-dependencias-y-criterios-de-aceptación)
y [RA §12](../AUDITORIA_RAIZ_REVALIDACION_2026-10-04.md#12-cobertura-y-siguiente-orden-de-trabajo):
identidad y cronología, admisión/reservas, estado causal del genoma y skill,
trayectoria de capital, frontera OOS, equivalencia cargado/evaluado y ejecución.
Estos son contratos, no motores separados por etiquetas scalping/swing.

El censo facilita distribuir revisión sin solapamientos. No atribuye a
Claude/GLM/Qoder lecturas no recibidas ni convierte reservas propuestas en
asignaciones aceptadas. Los hallazgos y sus límites permanecen en el
[informe RA](../AUDITORIA_RAIZ_REVALIDACION_2026-10-04.md).

## Topología declarada: recibo estructural, no grafo vivo certificado

[RA_TOPOLOGIA_DECLARADA_2026-10-04.json](RA_TOPOLOGIA_DECLARADA_2026-10-04.json)
describe los **24 paquetes del workspace (23 crates más el paquete raíz),
202 targets y 88 aristas internas declaradas** del mismo snapshot12456048.
Fuente: `cargo +nightly-2026-06-30 metadata --no-deps --locked --offline
--format-version 1`. Cada manifiesto conserva su OID; cada arista distingue
dependencia normal/dev/build, alias, opcionalidad y condición de plataforma.
No incluye bibliotecas externas ni resuelve activación efectiva de features.

La orientación A→B significa **A requiere B**, no A envía un evento a B.
En el subgrafo de dependencias normales no hay ciclo; esto es propiedad de
la declaración de compilación, no prueba de ausencia de ciclos de feedback,
deadlocks, carreras, llamadas recursivas o retrasos entre nodos operativos.
Un target es un elemento declarado de Cargo (lib/bin/test/etc.), no una
estrategia, un proceso en marcha o un contrato ya probado.

| Nivel de dependencia normal | Paquetes |
|---:|---|
| 0 | feature-engine, flight-recorder, graph-4d, graph-architecture, omniscient-registry, telemetry-engine |
| 1 | quantum-arena, storage-engine |
| 2 | data-pipeline, os-guardian |
| 3 | data-ingest, metacortex-engine, phase-runner, telemetry-server |
| 4 | audit-engine, dark-alpha-engine, strategy-core |
| 5 | signal-engine |
| 6 | risk-engine |
| 7 | execution-engine, god-engine-core |
| 8 | backtest-engine |
| 9 | evolution-engine |
| 10 | trader-gemini-v5 |

El nivel es `1 + máximo nivel de los proveedores normales` (0 si no hay
proveedores internos normales). Es una guía de lectura de abajo hacia arriba,
no profundidad neuronal ni latencia medida. La raíz alcanza23/24 paquetes
por esas aristas. `flight-recorder` no aparece en su cierre transitivo de
dependencias normales: **no se demuestra que esté muerto o desconectado**;
un ejecutable independiente puede comunicar por IPC/archivos/u otra ruta.
Su uso exige localizar productor/consumidor y observar evidencia real.

Validación del artefacto: JSON con claves únicas; conjuntos exactos de
paquetes/aristas/atributos y conteos cotejados con metadata; blobs de los
24 manifiestos comparados con Git12456048; aciclicidad y alcance recalculados.
Para el grafo vivo faltan contratos por arista: identidad de activo, reloj,
unidad, generación, vigencia/invalidación, entrega/ack, reintento, backlog,
latencia y consumidor. No se infiere sincronía perfecta de la topología.

## Nuevo corte main237 y plan archivo-a-archivo (adenda, no sustitución)

[COBERTURA_MAIN237_2026-10-04.tsv](COBERTURA_MAIN237_2026-10-04.tsv) y
[recibo JSON](COBERTURA_MAIN237_2026-10-04.json): árbol inmutable23701ecb,
1432rutas,461Rust,23directorioscrates y24manifiestosCargo.136Rust bajo
tests/ incluye1soporte. Clases/fases/owners sólo heurísticas por ruta;
todos inventariado_no_certificado. QA propia y del subagente verifica
conjuntos/OIDs/modos/orden, hashes, conteos y estados, no contenidos.

[Plan exhaustivo](../PLAN_REVISION_EXHAUSTIVA_2026-10-04.md) conserva
F0–F8 con subfases de fundamentos, métodos por clase, recibos por contrato,
rollback y cobertura externa separada. C-versionado no es C-sistema.
644JSON de grafo y11artefactos de grafo también siguen pendientes; no
certificar sincronía operativa mediante topologías exportadas.

Las cifras1435/433 de este README arriba corresponden a12456048, no
a main237.461Rustmain y465RustRA se diferencian además por4fuentes nuevas
RA (helper OOS/test, testparser ytestestado no finito). No equiparar conteo
de todas las rutas bajo crates con conteo de .rs bajo crates. El universo
debe acompañar cada cifra; un cambio de árbol pide reconciliar por ruta/OID.
