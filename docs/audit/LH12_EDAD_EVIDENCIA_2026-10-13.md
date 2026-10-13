# LH-12: envejecimiento de cada cierre y restauración del reloj

Consejo Codex, 13 de octubre de 2026. Hallazgo reproducido en `6bb08fdf`;
composición con `3dc88468` de Ω93. Alcance: calibración de Platt, su estado
guardado y dos contratos nuevos. No cierra la revisión de todo el núcleo.

## Problema y comportamiento resultante

La ventana retenía 512 pares sin fecha. La consulta mezclaba el modelo con
la identidad según la edad del último cierre. Incorporar otro cierre volvía
a ajustar todos los resultados con peso uno. El historial antiguo recuperaba
peso completo aunque hubieran transcurrido varias semividas operando.

Ahora cada resultado conserva su timestamp de evento. En cada cierre se
reajusta con la edad de todas las filas, en el máximo reloj conocido.
Un resultado tardío conserva su edad y no hace retroceder ese reloj.
`calibrate_at_tau`, añadido por Ω93, se conserva: gobierna la mezcla entre
consultas, no la edad individual del ajuste.

El modelo es explícito. Con `n` filas retenidas, score `s_i`, resultado
`y_i`, logit `x_i`, semivida `H`, masa del prior `κ` y ridge `λ`:

\[
p_i=\sigma(a x_i+b),\quad a\ge0,\qquad
w_i(t)=2^{-(t-t_i)/H},
\]

\[
L_t(a,b)=\sum_i w_i(t)\ell(y_i,p_i)
+\frac{\kappa}{n}\sum_i\ell(s_i,p_i)
+\frac{\lambda}{2}[(a-1)^2+b^2].
\]

Aquí `ℓ` es la entropía cruzada binaria. Una fila se computa con peso
`w_i+κ/n` y etiqueta `(w_i*y_i+(κ/n)*s_i)/(w_i+κ/n)` en objetivo,
gradiente y Hessiana. No normalizar los pesos reales: esa normalización
eliminaría la pérdida de masa frente al prior. La semivida heredada de 12 h
es una política contrastable, no una duración de régimen demostrada.
`WINDOW=512` limita almacenamiento y trabajo y también recorta memoria a
alta cadencia. La restricción de pendiente no negativa sigue vigente.

La consulta conserva la mezcla aproximada con la identidad, sin volver a
resolver Newton. Por tanto no representa exactamente `L_t` a cada instante.
No se midió una latencia de nanosegundos; el ajuste de cierre conserva trabajo
acotado por filas e iteraciones y una asignación de buffer ya existente.

## Reloj, compatibilidad y restauración

La API antigua `update()` sigue sin envejecimiento hasta que llega un reloj
explícito. Entonces sus filas se anclan una sola vez a ese instante: sus
edades anteriores son desconocidas. Un `update()` posterior usa el último
reloj conocido y también envejece. No se convierte una fecha desconocida en
evidencia de antigüedad reconstruida.

El reloj máximo sobrevive independientemente del anillo. Ejemplo: un cierre
en `10*H`, seguido por 512 cierres recibidos tarde y fechados en cero,
expulsa la fila de `10*H`. Guardar sólo las filas restantes rejuvenecería
los resultados al restaurar. Se guardan relojes opcionales por moneda y
global, y se restauran tras reproducir los registros.

`validar()`/`cargar()` rechazan relojes anteriores a una fila retenida. La API
directa `restaurar_estado_aprendido()` continúa confiando en su llamador.
Los JSON antiguos sin esos campos siguen siendo legibles: permiten inferir
sólo el máximo de las filas retenidas, no un máximo que nunca guardaron.
Volver a un binario antiguo puede perder los campos adicionales; no se
certifica esa operación como reproducción del nuevo algoritmo.

## Evidencia ejecutada y validación de composición

El testigo Rust aislado extrae el prefijo de Platt del blob base
`826e4615947dd6368c0b2e2556142200b2de2054`, hasta el comentario D-693.
La única dependencia sustituida es la constante temporal `43_200_000 ms`,
contrastada con `temporal_spectrum.rs`. No ejecuta el núcleo completo.

* RED: siete contratos, tres aprobados y cuatro fallidos, exit 101.
* GREEN del prefijo modificado: ocho aprobados, cero fallidos, exit 0.
* Ejemplo de cierres continuos: `p=0.86677677` con el código anterior,
  frente al oráculo ponderado `0.52840499`.
* Oráculos independientes: solución cerrada de la entropía ponderada para un score y
  dos bins que identifican pendiente e intercepto, con tolerancia explícita
  a la perturbación del ridge. Incluyen igualdad de timestamps, cambio de
  origen del reloj, actualizaciones mixtas y scores inválidos.
* Tres contratos nuevos ejercitan el snapshot JSON del núcleo: reloj máximo
  expulsado, lectura de formato anterior y rechazo de reloj inconsistente.
  Los ocho contratos de edad y los tres de restauración pasaron en la
  composición R4 descrita abajo. El check privado con 3dc terminó según el
  recibo propio de esta sección; la composición con 98aadd2 aún es posterior.

El otro Codex ejecutó seis contratos core en release sobre
`f62590e33a76a61a67320fed0e1ecc9ae0514ebd`: 59 aprobados, cero fallidos,
cero ignorados, exit 0, sin filtros y con `--test-threads=1`. Incluye
`platt_evidence_age_contract` (8), `platt_evidence_restore_contract` (3)
y `platt_optimizer_contract` (8). El recibo terminó a
`2026-10-13T06:46:47.0834988Z`, con las doce invariantes de su runner
satisfechas, compiler nightly fijado y target privado. ROOT contrastó
los cuatro blobs LH12 con su propio índice: son idénticos, incluido
`calibrate_at_tau` de main. Es evidencia de esa composición, que también
contiene R4/98aadd2, no una ejecución terminal de la cadena fría ROOT
ni CI del candidato por publicar. Recibo local del otro WT:
`target/r4-zero-replay/13-final-core-contracts.json`, SHA-256
`af447c05a0ae38c78dc7cc95fa88504de0663b9da79955e8c8c853934270514b`.

Otros recibos de esa misma composición, comprobados por ROOT:

| Recibo R4 | Resultado | SHA-256 del JSON local |
|---|---|---|
| `12-final-four-libs` | 539 aprobados, 0 fallidos, 2 ignorados; incluye core | `9bed756336b65038ffbb5646dfa6e8442a521ff522257a147042aed785d6aec2` |
| `15-final-core-persistence` | 10 aprobados, 0 fallidos, 3 ignorados; cf2/cf4 existentes | `45cb86bcfb5163705e313c492892c4f6fd4389ce2ed0d815a61ae0c88e861938` |

Los dos contratos cf2/cf4 también conservan sus blobs en el índice ROOT.
Los tres ignores de cf2 permanecen: estas pruebas no cierran cobertura,
selección de resultados ni extrapolación. Los dos recibos tienen exit 0
y doce invariantes satisfechas. Los recuentos describen ejecuciones
distintas; no son un porcentaje de auditoría completa del repositorio.

La compilación fría de tests ROOT se interrumpió voluntariamente al contar
con esos tests integrados y fuentes equivalentes. No ejecutó tests ni se
declara verde: su Cargo terminó con exit 101 por terminación del proceso
de la dependencia `polars-ops`. Registro propio
`logs/lh12-cold-test-stop.json`; log SHA-256
`4721b82bcd61e6865e63d04d0c1817c978dea0f95537ab62aefeb7088ad043c7`.
La cadena propia se concentró en `check --workspace --all-targets`,
con snapshot antes/después y target aislado, antes de cerrar el merge.

ROOT ejecutó `cargo +nightly-2026-06-30 check --workspace --all-targets
--locked` sobre padres `04eae49b` y `3dc88468`: exit 0, 24 min 52 s,
terminado a `2026-10-13T07:25:03.8251259Z`. Los 662 inputs registrados
(635 Rust), sus bytes/blobs, HEAD y MERGE_HEAD permanecieron idénticos.
Target propio, un job y debuginfo/incremental desactivados; rustflags del
proyecto conservados. Hubo un warning heredado por `serie` no usada en
soporte del test `t2_cobertura_puertas`, sin modificarlo para ocultarlo.
El check compila targets y no ejecuta los tests. Recibo
`logs/lh12-all-targets-3dc.json`, SHA-256
`bd38076914a4d4a9bd0e766636fa44a3468ac77a7e677f44df3adf18ee185085`;
log SHA-256 `13deab5c192d0ddb6eb42c522fd2f92598b2b0ff23058545d28dcfe6ff074423`.

Las lecturas matemáticas y la revisión adversarial de persistencia se hicieron
por revisores separados. El segundo detectó el caso de reloj expulsado y
la necesidad del oráculo de dos bins antes del cierre de esta ola.

El revisor matemático resolvió además el objetivo de dos bins con ridge:
`p≈(0.5715656564, 0.7514567046)`, pendiente `a≈0.2950869248>0`.
La perturbación máxima frente al posterior cerrado sin ridge es `1.02e-5`,
dentro de la tolerancia `1e-3` del contrato. Es un contraste independiente
del modelo, no una ejecución adicional del núcleo.

Comentarios heredados sobre volver a sondear/Kelly o sobre decaimiento por
τ no garantizan decisiones posteriores: esta API sólo cambia calibración
y mezcla idle; el ajuste individual continúa usando H=12 h. LG-5 y la
intervención de gates siguen pendientes de sus contratos económicos.

## Grafo de consulta y alcance de la extracción

Se ejecutaron `graphify query` y `graphify update .` en el WT aislado,
según la regla del proyecto. Update terminó con exit 0: 838 archivos de
código extraídos, 16.716 nodos y 26.559 aristas. El grafo curado anterior
tenía 83.025 nodos; la herramienta respaldó cinco artefactos y sustituyó
la vista por AST. Su salida exige otro flujo para documentos/teorías.

Se conservó el resultado AST en `logs/lh12-graphify-ast/` y el respaldo
curado en `logs/lh12-graphify-curated-backup-2026-10-13/`, ambos ignorados.
Se restauraron únicamente los ocho derivados que esta ejecución cambió,
preservando el grafo curado versionado. SHA-256 del AST `graph.json`:
`b2bbc7a35e8fbd39ad008925bc145696b4921ad52b102a489ad54af43791bc64`.
Estos resultados no certifican lectura, semántica ni actualidad de todo
el conocimiento. Re-grep y blobs de fuente gobiernan los contratos.

## Límites y continuación

Este cambio corrige la memoria temporal del ajuste. No demuestra que la
política seleccione mejor operaciones ni que alcance +100 % cada 72 h.
`P(PnL neto>0)` sigue sin ser `P(TP antes de SL)` (LG-5); el resultado
sigue condicionado a órdenes ejecutadas (C-F2.2). No ofrece una LCB, no
cablea LG-22 y no permite consultas históricas causales anteriores al
último dato. El reloj de evento no sustituye un tiempo separado de
disponibilidad/conciliación de cada etiqueta.

Siguiente contraste económico: algoritmo de actualización completo congelado,
control anterior y nuevo sobre las mismas trayectorias multivariantes, costes
y flujos; medir log-loss/Brier prequential, soporte, exposición, `G72`, DD
y ruina conjuntamente. Usar datos posteriores no empleados en seleccionar
semivida/modelo. El T-3 v2 continúa reservado a Línea H; sus cifras no se
atribuyen a este patch.
