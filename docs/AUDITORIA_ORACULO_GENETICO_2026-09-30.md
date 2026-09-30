# GO — Auditoría del oráculo genético y validez de sus mediciones

Fecha: 2026-09-30. Autor: Codex. Rama: `codex/genome-oracle-evidence`.
Base de código: `76de9935cb21075414f6d50eda64089393e0a173`.
Estado: reparación diagnóstica local; no publicada, no fusionada, sin certificación
de cobertura genética total ni de rentabilidad. Esta adenda conserva los
informes previos; sus afirmaciones históricas no se convierten en evidencia vigente.

## 1. Dictamen ejecutivo y límites de la revisión

El problema inmediato no es escoger otra teoría sofisticada, sino conocer qué
variable está observando el instrumento usado para juzgar la evolución.
T-1 compara ocho estadísticas de un replay monoactivo, tras perturbar una
coordenada hacia un único extremo y reconstruir el genoma. No mide directamente
la aptitud canónica del promotor ni prueba que una coordenada sea globalmente inerte.

Se reparó una correspondencia errónea de campos en el diagnóstico. Se añadieron
trazas del genoma efectivo y contraejemplos ejecutables para evitar conclusiones
inválidas. No se cambian producción, genoma, fixture, predictor, comparador,
denominador, criterio de sensibilidad ni trinquete de cobertura de 0,110.
Las nueve pruebas rápidas NO sustituyen la corrida extensa T-1, que no se
ejecutó en esta pasada. Un test que caracteriza un fallo abierto no es su reparación.

Cobertura real de lectura: archivo T-1 completo y nuevos helpers/contratos;
secciones de construcción/serialización/restricciones del genoma, escritura de
estadísticas y sanitización del replay; workflow, notas GLM de PR20 y revisión
MR. No es una lectura íntegra de todos los crates ni de todos los archivos.

## 2. Grafo vivo de la evidencia: raíz, decisión y terminal

```text
RAÍZ: fixture OHLCV + baseline + predictor sintético + símbolo/capital
  └─ vector serializado, 144 coordenadas
      └─ petición: cambiar g hacia su extremo más lejano
          └─ from_vector: clamp + enforce_curve_rr + derive_anchors + sync
              ├─ cambio borrado: no llegó ninguna intervención al motor
              ├─ cambio acoplado: intervienen varias coordenadas efectivas
              └─ cambio realizado
                  └─ replay / señales / vetos / riesgo / fills simulados
                      └─ ocho estadísticas agregadas
                          └─ DECISIÓN: alguna diferencia supera tolerancia
                              └─ TERMINAL: proporción / 144 >= 0,110
```

La ausencia de cambio terminal tiene múltiples causas posibles. Este grafo no
permite saltar directamente a «gen muerto» o «veto arbitrario». El veto debe
auditarse en su consumidor, con entradas, unidades, disponibilidad de información,
margen numérico y estado de activación. No se retira un límite para aumentar el
número de coordenadas sensibles.

## 3. Matriz de expedientes

| ID | Prioridad | Hallazgo | Estado en este corte |
| --- | --- | --- | --- |
| GO-01 | P1 | Diagnóstico intercambia trades, PnL y WR | Reparación candidata local con RED/GREEN |
| GO-02 | P1 | Ruido sintético estrictamente negativo, no centrado | Abierto; fixture histórico preservado |
| GO-03 | P1 | Un extremo no prueba inercia global ni detecta interacciones | Lenguaje corregido; método limitado, abierto |
| GO-04 | P1 | Petición vectorial distinta de intervención efectiva | Trazas añadidas; problema de coordenadas abierto |
| GO-05 | P1 | Cambio en estadísticas no equivale a cambio de aptitud | Documentado; integración con fitness abierta |
| GO-06 | P1 | Comparador de no finitos no valida resultados | Abierto; semántica histórica preservada |
| GO-07 | P1 | Fixture único no acredita multi activo ni continuo temporal | Abierto; alcance delimitado |
| GO-08 | P2 | Diagnóstico depende de estado global y variable de entorno | Abierto; aislamiento de ejecución necesario |
| GO-09 | P1 | Fallo combinado no identifica causa ni legitima re-baseline | Abierto; diseño de comparación especificado |

GO-04 incluye antecedentes D-656: no se presentan las cuatro anclas derivadas
como cuatro descubrimientos nuevos. GO-06 conecta con los contratos numéricos
del replay. Los expedientes no se suman mecánicamente a una antigua matriz
de «305 fallos» sin conciliación de identidad, estado y duplicados.

## 4. GO-01 — Etiquetas diagnósticas equivocadas

**Evidencia.** El replay escribe, en orden:
`[net_wr, trades, final_capital, max_dd, sharpe, gross_pnl, net_pnl, gross_wr]`.
Fuente: `crates/backtest-engine/src/lib.rs`, escritura de `out_stats[0..7]`.
El test imprimía stats[0] como trades, stats[1] como PnL y stats[2] como WR.

**Reproducción.** Para [0,25; 8; 975; 0,05; −0,4; −10; −25; 0,50],
el mensaje histórico era `trades=0.25 pnl=8 wr=Some(975.0)`.
La asignación correcta es `trades=8 pnl=-25 wr=0.25 capital=975`.
El contrato falló antes de corregir los índices, con seis pruebas aprobadas
y una fallida. Después se amplió y pasó la batería de nueve.

**Impacto.** Contar fracciones como operaciones y dinero como probabilidad
puede dirigir una investigación hacia el subsistema equivocado. La reparación
no cambia capital, órdenes, rechazos ni selección genética; corrige la lectura.

**Reparación/aceptación.** Helper compartido por el test real y la regresión,
con acceso a índices 1/6/0/2. Pendientes CI y revisión del candidato.
No se infiere que toda la telemetría del sistema tenga ya un contrato tipado.

## 5. GO-02 — Sesgo determinista del supuesto ruido

**Fórmula existente.** Sea S un entero de 64 bits del LCG. El término
u = (S >> 33)/(2^32−1) − 1/2 usa únicamente 31 bits en el numerador.
Por tanto, −1/2 <= u <= −1/[2(2^32−1)] < 0.
No hace falta asumir aleatoriedad uniforme para demostrar el signo.
Bajo distribución uniforme de esos 31 bits, su esperanza sería cercana
a −1/4; el valor medido del fixture no se sustituye por esa aproximación.

El precio sigue p_i = p_(i−1) [1 + 0,003 sin(i/180) + 0,004u_i + 0,00004].
El ruido aporta entre −20 y casi 0 puntos básicos por evento, frente al
drift nominal de +0,4 pb. Su escala nominal «40 pb» no significa ruido
simétrico de esa amplitud. El ciclo, la multiplicación y el drift también
afectan el precio final: no se atribuye toda la trayectoria al ruido.

**Medición reproducida, 3.000 observaciones.** Ningún residual positivo;
media u = −0,2525882664848767; último precio = 7592,089743463237, desde 60000.
La media se recupera de la recurrencia de precios; incluye redondeo de f64.

**Impacto.** La prueba puede favorecer rutas direccionales y no activar
otras; su resultado no representa una distribución neutral de escenarios.
No se demuestra con esto un sesgo equivalente en datos de mercado reales.

**Alcance.** La expresión también aparece en fixtures de
bt_vivo_parity_audit, booktick_replay, booktick_causality_contract y lib.rs.
Se observó por búsqueda; no se repararon esas fixtures ni se alteró el golden.

**Cierre exigido.** Nueva familia versionada de escenarios con contrato
distribucional y controles de signo, escala y persistencia, comparada en
paralelo con la histórica. No reemplazar silenciosamente el fixture ni
rebajar el trinquete para obtener verde. Estado: abierto.

## 6. GO-03 — Extremos, interiores e interacciones

**Error lógico.** «Si no cambia con el extremo más lejano, no cambia con
nada» sólo sería defendible bajo condiciones adicionales que aquí no se
han demostrado. Contraejemplo: f(x)=x(1−x), x en [0,1], base 0.
f(0)=f(1)=0, pero f(0,5)=0,25. La prueba rápida usa el comparador real.

**Interacción.** f(a,b)=ab, base (0,0): mover una sola coordenada a 1
no cambia nada; mover ambas produce 1. Son contraejemplos matemáticos
del procedimiento, no una afirmación de que el motor implemente esas funciones.

**Impacto.** Falsos negativos al atribuir inercia, retirar genes o cambiar
vetos; una banda saturada en un solo contexto no prueba ausencia global de
efecto. También puede haber rutas discontinuas por cuantización o admisión.

**Acción.** Comentario y salida ahora dicen «sin cambio observado bajo esta
perturbación/fixture». El protocolo de un extremo se preserva para continuidad
histórica. Pendiente: diseños de interiores, pares y contextos, con presupuesto
computacional, semillas, trazabilidad y contraste fuera de muestra. Una malla
finita, por densa que sea, tampoco certifica exhaustividad del continuo.

## 7. GO-04 — La geometría del vector no es la del genoma efectivo

**Fuente.** `SuperGenotype::from_vector` aplica restricciones y deriva
anclas; D-656 ya documenta que slots 13–16 son vistas de curvas.
La transformación debe entenderse como P(v), no como identidad.

**Medición actual.** Roundtrip del baseline: ninguna coordenada cambió bajo
la tolerancia usada. Con las 144 peticiones históricas, se borraron por completo
13,14,15,16,140. Acoplamientos observados:
141 → [13,14,15,16,140,141,142,143];
142 → [13,14,15,16,140,142];
143 → [13,14,15,16,140,142].
En la última petición ni siquiera cambia necesariamente la coordenada 143
reconstruida. Esto se refiere a este baseline y a este extremo, no a toda su banda.

**Significado.** La intervención es P(v+delta e_g)−P(v). Puede ser cero
o tener varias componentes. Asignar su efecto a un gen independiente sin medir
esa transformación confunde representación, restricción y causalidad.

**Acción.** Trazas `T1-PROBE` con solicitado, realizado y slots cambiados;
contrato rápido que hace la reconstrucción real sin abrir modelos.
No se cambia el genoma ni se redefine el denominador de 144. Excluir cuatro
slots tampoco demostraría que las otras 140 dimensiones sean independientes.

**Cierre.** Especificar grados de libertad admisibles, reglas de proyección,
direcciones tangentes o perturbaciones factibles y su relación con el optimizador.
Tratar aliases como aliases y restricciones financieras como restricciones,
sin borrarlas para producir sensibilidad. Estado: abierto con observabilidad mejorada.

## 8. GO-05 — Estadísticas, fitness y selección no son equivalentes

T-1 marca sensibilidad cuando cualquiera de ocho estadísticas difiere.
No llama a la agregación final de aptitud del promotor para hacer esa decisión.
Un cambio de conteo puede no alterar fitness; un agregado puede ocultar cambios
en distribución de retornos, exposición, cola o atribución por activo.

Su comparación finita usa |x−y| > 10^-9 max(|x|,1), con x como referencia.
Es tolerancia mixta absoluta/relativa por componente, no significancia estadística,
intervalo de confianza ni materialidad económica. Mezcla dimensiones: dinero,
conteos, proporciones y Sharpe. El umbral 11% es histórico de regresión,
no una ley teórica que pruebe autoevolución o ventaja financiera.

**Cierre.** Contrato explícito desde genoma solicitado hasta fitness efectivo,
con versión del promotor, métricas tipadas/unidades, trazas de selección,
control de incertidumbre y comparaciones de resultados netos. No sustituir la
aptitud por WR ni confundir 100% de crecimiento con 100% de acierto.
Estado: abierto; la aclaración documental no conecta todavía esas capas.

## 9. GO-06 — No finitos y sanitización

El comparador retorna distinto estado de finitud si alguno de los dos valores
no es finito. NaN frente a +Inf resulta «sin diferencia»; 0 frente a NaN resulta
«sensible». La regresión reproduce ambas situaciones y permanece verde
precisamente porque caracteriza el defecto abierto.

Además, `run_backtest_native` reemplaza estadísticas no finitas por cero
o capital inicial. Exigir finitud sólo al final puede perder la procedencia
de un fallo anterior. Esta observación no demuestra que la corrida actual
haya producido NaN; demuestra una limitación del contrato.

**Cierre.** Canal separado de validez/error, fallar explícitamente la medición
ante valores inválidos y conservar causa/etapa antes del saneamiento. Cambiar
el comparador requiere una nueva medición reconciliada, no borrar la historia.
El helper changed_slots sólo se usa aquí con genomas finitos, no como validador
numérico general. Estado: abierto.

## 10. GO-07 — Alcance monoactivo y horizonte observado

La raíz fija BTCUSDT, capital 1000, una serie de 3000 observaciones y un
predictor direccional sintético. El test condiciona la medición a cooperación
predictiva; no evalúa la exactitud ni la calibración de modelos operativos.
No hay diversidad simultánea de activos, shocks cruzados, competencia por
capital, precisión propia de cada instrumento ni activación de todo el host.

Un esquema continuo de parámetros no crea datos a todas las escalas. Resolución
del reloj, observabilidad, soporte de datos, coste y horizonte de decisión deben
separarse. No se acredita cobertura entre un nanosegundo y cien años con este
fixture. Tampoco sería correcto añadir iteraciones de nanosegundo sin datos
nuevos y llamar a ello mayor información.

**Cierre.** Matriz de contextos por activo, escala observada y condiciones de
mercado continuas, kernels definidos sobre tiempo físico, admisibilidad de datos,
validación causal y evaluación de interacciones. Es una agenda pendiente, no
una capacidad implementada por GO. No se añaden ecuaciones por prestigio;
cada modelo necesita hipótesis, unidades, identificabilidad y criterio refutable.

## 11. GO-08 — Aislamiento y trazabilidad del diagnóstico de vetos

El test de una evaluación instala un forest global y hace
`unsafe std::env::set_var("TG_TRACE_NATIVO","1")`. Su comentario presupone
ejecución single-threaded, pero el atributo test no impone ese modo a Cargo.
El reporte de rechazos usa contadores globales; debe distinguir delta de esta
evaluación de acumulado del proceso. Dos tests del target pueden compartir estado.

GO no modifica este comportamiento. Las regresiones ejecutadas usan
`--test-threads=1`; los nuevos nueve contratos no ejecutan el core ni escriben
esa variable. Esto acota nuestra ejecución, no repara la seguridad de cualquier
invocación del target T-1.

**Cierre.** Configuración explícita por contexto, snapshots/deltas de contadores,
restauración de recursos e aislamiento en proceso para mediciones históricas.
No interpretar cero rechazos como prueba automática de inexistencia de señales
sin conocer qué rutas y contadores participaron. Estado: abierto.

## 12. GO-09 — Atribución de la regresión y gobierno del trinquete

GLM reportó en main76de9935: merge local PR20+CX limpio, train_forest39/39,
y T-1 por debajo de11% tras49min. Son resultados reportados por GLM,
no corridas ejecutadas por Codex. La nota no adjunta cifra ni lista por gen.

El fallo combinado no determina qué cambio lo causa ni si reducir el umbral es
correcto. Se necesita el detalle por coordenada y etapas; comparar sólo porcentajes
puede ocultar que unas ganan sensibilidad y otras la pierden.

Diseño mínimo para investigar interacción:
- Y00: base sin CX ni cambios CL-30..32.
- Y10: CX sin CL-30..32.
- Y01: CL-30..32 sin CX.
- Y11: ambos.

Mantener fixture, predictor, compilador y configuración comparables, registrar
genomas efectivos y el vector de diferencias, no sólo Y agregado.
El contraste Y11−Y10−Y01+Y00 describe interacción en una métrica escalar
comparable; no identifica por sí mismo causas dentro del código.
Esos cuatro cortes NO se ejecutaron aquí. Deben definirse con commits exactos,
incluyendo dependencias para que ninguna combinación sea un artefacto inválido.

**Decisión.** Mantener 0,110 y PR20 sin integrar mientras el fallo no se explique.
Un arreglo causal correcto puede cambiar una referencia, pero cualquier
re-baseline exige justificación por gen, revisión independiente y evidencia
de no ocultar otra regresión. Estado: abierto.

## 13. Verificación local y preservación del instrumento

RED controlado: siete contratos iniciales, seis aprobados y uno fallido por
etiquetas. GREEN: nueve aprobados, cero fallidos/ignorados, 0,01s de ejecución
de tests (compilación16,26s). Las nuevas pruebas verifican igualdad bit a bit
del fixture extraído frente a las ecuaciones originales y de los144 destinos
de perturbación. El comparador y el trinquete conservan su código lógico.

Comando rápido:
`cargo +nightly-2026-06-30 test -p backtest-engine --locked --test t1_measurement_contract -- --test-threads=1 --nocapture`.

La regresión ampliada y check all-targets se registran en una adenda tras
terminar; no se presentan como aprobados por estar lanzados. No se ejecutó
el oráculo T-1 completo, entrenamiento, promoción, operación ni cambio
de artefactos/modelos. El workflow añade solamente el target rápido.

## 14. Integración, colaboración y condiciones de cierre

El checkout compartido se deja a GLM/Claude. Aviso pasivo en el buzón local;
su existencia no acredita lectura. GO se conserva en rama propia y no está
incluido en MR. MP permanece local con autorización pública pendiente.

MR/PR23 recibió revisión GLM COMMENTED favorable condicionada a documentar
`legible`; no es un estado formal APPROVED. MR no valida GO ni MP.
Antes de publicar GO: autorización específica por tratarse de repositorio
público; antes de merge: CI del candidato, revisión cruzada y comparación
contra ambos padres si hay integración concurrente. No eliminar ramas activas.

La meta financiera del operador no se convierte en evidencia por reparar
tests. No se promete duplicar capital cada tres días ni erradicar todos los
bugs. Esta pasada mejora la fiabilidad del diagnóstico y deja límites
explícitos para la siguiente investigación.

## 15. Adenda de regresión ampliada terminada

117 aprobadas / 0 fallidas / 2 ignoradas: biblioteca51, paridad8/0/2,
métricas21, labels25, riesgo espectral3 y contratos GO9. Las ignoradas
son mediciones manuales en tapes reales, no fallos omitidos del oráculo.
Compilación40,85s; biblioteca83,78s, paridad37,71s, resto0,10s.
No sumar de nuevo las nueve rápidas: ya están dentro de117.
El check all-targets y el commit se registran después de su confirmación.

Check confirmado: workspace/all-targets/locked aprobado en38,18s, sin
errores de compilación. No equivale a ejecutar todos los tests del workspace.
Hashes SHA256 de los tres archivos de tests constan en el JSON; corresponden
a los bytes locales validados, no a un modelo o a evidencia de producción.

## 16. Recepción de MR en main y reconciliación local

Durante este trabajo, GLM integró PR23 en589a591db26fc4e30af931d71b0fb1f9cc3103f4
(2026-09-30T14:11:15Z) y añadió la condición documental del campo legible
enfbf8e9ea12554c8a7a4494859b631a3280ab06f9. Remoto main verificado en ese SHA.
Codex no ejecutó ese merge ni duplicó la aclaración. La rama remota MR
ya estaba eliminada al consultar. Una review COMMENTED favorable no se
presenta como estado formal APPROVED.

En la consulta posterior, CI36725338162 del candidato f3145e27 y
CI36727198985 de mainfbf8e9ea seguían IN_PROGRESS. Por tanto, el merge
es un hecho observado, pero no se acredita que se cumpliera el gate de CI
vigente antes de hacerlo. El verde histórico36718230501 corresponde a
otro SHA. La protección de rama no se inspeccionó ni modificó; no se
deduce su configuración de este evento.

La recepción y las precisiones MR-02/MR-03 se comunicaron en
[PR23](https://github.com/Jhona-la/Trader-Gemini/pull/23#issuecomment-5913065634).
Aceptación estructural JSON no prueba identidad del predictor cacheado;
main de train_forest sigue sin parsear --test-in en la fuente observada.
PR10 y PR20 continúan abiertas; PR20 en borrador.

GO774cbac5 incorpora localmente mainfbf8e9ea. Tres conflictos de apéndices
resueltos conservando ambos lados. Verificación de subsecuencia ordenada,
líneas HEAD/MERGE_HEAD: atlas1933/1934, coordinación1760/1795,
maestro8742/8750, memoria954/1010, workflow49/50. Comparación contra ambos
padres realizada; ningún delta de producción GO respecto al nuevo main.
Check workspace/all-targets/locked aprobado en27,29s. Nueve contratos de
registro aprobados; la repetición ampliada se registra al finalizar.

Backups conservados: backup-before-cleanup tiene3 commits exclusivos,
v7-unificacion-wip1 y TH4 frente a este main. No son ramas demostradas como
integradas. MP40f8685b sigue local; GO no se publica mezclándola con MR.

### Cierre de la validación integrada

Repetición terminada sobre GO + mainfbf8e9ea: registry9/0/0 y
replay117/0/2, total disjunto126 aprobadas / 0 fallidas / 2 ignoradas.
Biblioteca73,35s; paridad42,55s; resto de targets de replay0,08s.
El contrato bit a bit se reforzó con igualdad de longitudes: zip solo
no detectaría truncamiento. Repetición final de las nueve rápidas9/0
y check workspace/all-targets/locked5,64s aprobados tras esa única aserción.
No contar las repeticiones como nuevos tests ni confundirlas con T-1 completo.

JSON conserva hashes del primer candidato y añade final_source_sha256
del resultado integrado. Código de producción idéntico a mainfbf8e9ea;
las diferencias GO siguen siendo tests, CI y documentación. Las referencias
remotas MP/GO no existen al verificar; ambas ramas permanecen sólo locales.
Las CI MR/main seguían en ejecución en la última comprobación de este corte.

## 17. Autorización de publicación recibida

El operador respondió «Hazlo» a la pregunta explícita sobre publicar MP y GO
en el repositorio PÚBLICO Jhona-la/Trader-Gemini mediante dos PR separadas,
con CI y revisión cruzada antes del merge. Queda levantado el bloqueo de
publicación de ambas ramas; las notas anteriores se conservan como historia.
No autoriza trading, entrenamiento, promoción de modelos ni retirar gates.
GO se publica separada de MP. Fuente9cda4677,126/0/2 y check5,64s; hashes
de código verificados sin delta. Push/PR/CI se acreditarán por su resultado.
