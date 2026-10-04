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
