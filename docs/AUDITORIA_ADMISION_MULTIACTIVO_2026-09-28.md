# Auditoría de admisión multiactivo y evidencia de integración — 2026-09-28

Estado: correcciones verificadas en el checkout compartido; auditoría acotada,
NO certificación integral, validación de rentabilidad ni autorización operativa.

[Artefacto estructurado](artifacts/auditoria_admision_multiactivo_2026-09-28.json) ·
[Informe espectral anterior](AUDITORIA_CONTRATOS_ESPECTRALES_2026-09-28.md) ·
[Contratos numéricos](AUDITORIA_INTEGRACION_Y_CONTRATOS_NUMERICOS_2026-09-28.md) ·
[Buzón entre editores](../COORDINACION_CODEX_2026-09-28.md).

## 1. Dictamen ejecutivo, alcance y límites de aseguramiento

Se confirmó y corrigió un bypass lógico de riesgo: después de contar una
dependencia no observada como incierta, el consumidor podía fabricar ceros en
una matriz parcial, interpretarla como independencia y retirar ese conteo.
También ignoraba slots y exposiciones de signo opuesto que podían concentrar
el mismo PnL. Estos defectos no se resolvían añadiendo más teoría al solver:
estaban en el transporte de evidencia y en el significado de sus entradas.

Esta ola corrige la admisión numérica de la media de correlación, el dominio
de la agregación escalar y el consumidor D-748. Conserva las APIs matemáticas
existentes, pero no permite que una EWMA de pérdida al stop se disfrace de
desviación típica para justificar descuentos de cartera. La política lineal
heredada continúa siendo una aproximación, no una cota demostrada de pérdida.

Resultado reproducido: risk-engine, 195 pruebas pasan, 0 fallan, 0 ignoradas;
quantum-arena, 165 pasan, 0 fallan, 0 ignoradas. Workspace all-targets compila
con advertencias. Esto verifica los contratos ejercitados, no todos los
caminos, la suficiencia estadística, las colas extremas o la rentabilidad.

Durante la revisión otra sesión publicó 988f0478 en main, incorporando cambios
de varios autores, incluidos arreglos inspeccionados aquí. Quedaron fuera las
cuatro suites nuevas de riesgo y los informes nuevos. El PR #7 pasó de CLEAN a
CONFLICTING. Véase §10: estar 0/0 frente a origin/main no significa que todo
el trabajo del directorio esté versionado o integrado.

Cobertura real: lectura/revisión de correlation_guard, random_matrix,
consumidor D-748 y escritura de EWMA, slots/snapshot de posiciones, cuatro
archivos de regresión y los diffs del genoma/PR. No se afirma una relectura de
los 1.300 archivos del censo histórico, de los 24 manifiestos ni de todos los
crates. Las métricas del censo pertenecen al corte anterior, no son cobertura
de auditoría. No se ejecutaron órdenes, demo, entrenamiento, promoción o
despliegue; tampoco se inspeccionaron claves.

## 2. Grafo diagnóstico: de observación a decisión

```text
RAÍZ: ticks del activo candidato + ticks de cada activo con exposición
  │ validez del reloj/precio, soporte común, variación observada
  ▼
ESTIMADOR: HY; rejilla/Pearson como respaldo cuando corresponda
  │ Some(r) finito y en dominio, o None (causa todavía no tipificada)
  ▼
EXPOSICIÓN: todos los activos × todos los slots
  │ snapshot de lado; rho_PnL = signo_candidata × signo_posición × rho_precio
  │ mismo activo: identidad de precio; no fabricar muestras
  ▼
EVIDENCIA: {abiertas, misma_apuesta, desconocidas}
  │ desconocida cuenta como caso adverso; no se convierte en cero
  ▼
DECISIÓN D-748: presupuesto lineal heredado sobre el conteo
  │ no descuento MP, no matriz estrella, no rho “Hodge”
  ▼
TERMINAL DE ESTA REVISIÓN: rechazo 2 o continuación de validación
  └ NO equivale a orden enviada, fill, PnL conciliado ni eficacia del genoma
```

Este grafo es el camino inspeccionado, no una descripción completa del sistema.
La propiedad importante es conservar la incertidumbre entre nodos. Una
correlación ausente, una correlación estimada cero y una matriz inválida no
deben colapsar en el mismo estado de “diversificación acreditada”.

## 3. Matriz de actualización de hallazgos

Los IDs SPECTRAL mantienen las fichas anteriores; no se renumeran ni se añaden
automáticamente a la matriz histórica de 305. “Corregido” se refiere al defecto
concreto, no al cierre del área completa. P1/P2 expresan prioridad técnica,
no pérdidas financieras medidas.

| Hallazgo | Estado de esta ola | Alcance exacto / deuda restante |
|---|---|---|
| SPECTRAL-003, P1 | Bypass retirado del consumidor | No hay Unknown→0 ni exención AllNoise; no se implementó matriz conjunta identificable |
| SPECTRAL-004, P1 | Signos y slots corregidos; parcial | Todos los slots y ambos signos; faltan pesos monetarios, stops, tau y reserva atómica global |
| SPECTRAL-008, P1 | Contrato numérico corregido | Media exige matriz completa válida; no demuestra procedencia estadística |
| SPECTRAL-009, P1 | Dominio escalar corregido | rho imposible no obtiene cobertura; política del piso de una apuesta sigue heredada |
| SPECTRAL-010, P1 | Uso dimensional indebido retirado; OPEN | No se aplica descuento sigma sobre EWMA; falta riesgo real de cada posición |
| SPECTRAL-012, P2 | Afirmación corregida en código; parcial | Frustración de signos no es Hodge; algoritmo legacy queda sin actuador vivo |
| SPECTRAL-013, P2 | OPEN | Muestra efectiva, calidad, frontera temporal y error del estimador no viajan con Some |
| SPECTRAL-014, P2 | Evidencia ampliada; OPEN | Dos expectativas inválidas rectificadas sin borrar testigos; no cobertura integral |
| SPECTRAL-015, P1 | OPEN, sin validación | +100%/72 h no se acredita con tests ni con reparar bugs |
| ADM-GIT-01, P1 | Confirmado, OPEN | Publicación concurrente incorporó código pero omitió 44 regresiones locales y documentos vinculados |
| ADM-GIT-02, P2 | Confirmado, OPEN | Mensaje XLIV describe actuador varianza/Hodge que su propio árbol ya no ejecuta |
| ADM-MERGE-01, P1 | Confirmado, pendiente | PR #7 abierto, conflictos de tests y composición semántica de reparadores por revisar |

SPECTRAL-001/002 y las correcciones previas de 005/006/011 se conservaron. Las
limitaciones de soporte HY parcial y de inferencia siguen en los informes
previos. Una tabla actualizada no elimina los testigos históricos.

## 4. SPECTRAL-003: evidencia ausente convertida en independencia

**Causa raíz anterior.** El consumidor construía un grupo a partir de aristas
candidata→posición. Las relaciones posición→posición no se habían estimado,
pero la tabla se inicializaba con ceros. El valor None tampoco se propagaba
como ausencia: quedaba un cero. Un solver numéricamente correcto acepta una
identidad como PSD; no puede saber que sus ceros son imputaciones sin evidencia.
El resultado AllNoise se usaba para dispensar el guard.

**Cadena de impacto.** Falta de ticks → helper prudente → matriz imputada →
comparación MP con T=512 → exención → admisión. Se había validado el objeto
algebraico equivocado. 512 era capacidad máxima del snapshot, no número de
observaciones conjuntas efectivas. No detectar un factor con esa comparación
no constituye una demostración de independencia.

**Reproducción.** El contrato
missing_evidence_is_not_imputed_to_independence_in_live_admission abre una
posición sin historial de correlación, en cada uno de los tres slots y ambos
lados. Fija riesgo por operación de modo que dos apuestas excedan el tope y
llama a RiskEngine::evaluate_quantum_order. Antes se admitía; ahora devuelve
Flat y aumenta el contador de rechazo 2. El caso sin posiciones sigue
admitiendo la señal: no se añadió un veto indiscriminado.

**Corrección.** dependency_exposure devuelve un conteo estructurado; None
persiste como evidencia desconocida y cuenta en el caso adverso. El consumidor
no construye la matriz estrella ni invoca systematic_mode/rho efectivo. No se
eliminan esas bibliotecas: siguen disponibles y probadas como herramientas,
no como certificados de admisión.

**Límite.** Unknown se trata adversamente respecto del conteo heredado. Eso no
garantiza que el importe de riesgo asignado a cada posición sea suficiente.
No es una prueba de solvencia ni una política óptima de recuperación de datos.

## 5. SPECTRAL-004: signos, slots y unidad económica

Para incrementos de precio comparables y exposiciones lineales de signo fijo:

```text
X_i = s_i a_i R_i, a_i > 0, s_i ∈ {-1,+1}
Corr(X_i, X_j) = s_i s_j Corr(R_i, R_j)
```

Las magnitudes positivas se cancelan al calcular correlación, pero NO al
calcular varianza o pérdidas monetarias. Long A + short B con rho_precio=-0,9
produce rho_PnL=+0,9; filtrar primero por “misma dirección” eliminaba una
concentración. Long A + long B con la misma anticorrelación no debe contarse
como la misma apuesta bajo este criterio.

El consumidor anterior leía únicamente una parte del almacenamiento. La
implementación nueva recorre PositionManager::slots() sin convertir sus
nombres históricos en motores temporales. Incluye posiciones del propio
activo; usa identidad de precio 1 para éste y transforma por el signo. No
inventa 512 muestras para justificar esa identidad estructural.

Se estima una vez por activo con posiciones y se reutiliza el resultado en
sus slots. Cada posición se lee mediante snapshot. Si se observó abierta
pero no hay snapshot utilizable, cuenta abierta y desconocida; no desaparece.
El snapshot por posición no convierte la secuencia completa en transacción:
otra apertura entre lectura y envío requiere control/reserva de cartera.

**Evidencia.** Pruebas end-to-end cubren cada slot del propio activo y de otro,
anticorrelación con dirección opuesta y cobertura medida. Un barrido adicional
de 24 combinaciones cubre dos signos de correlación × dos lados candidatos ×
dos lados de posición × tres slots. Otra prueba exige cuatro posiciones
abiertas, cuatro contadas adversamente y tres desconocidas, preservando la
distinción en el resultado público. ID candidato inválido devuelve None.

**No cerrado.** El guard cuenta, no pondera tamaños/stops. Dos posiciones con
distintas duraciones y reglas de salida no son dos PnL a horizonte común.
No se afirma que correlación instantánea garantice cobertura hasta sus stops,
liquidación, gaps o después de comisiones. La arquitectura temporal continua
requiere transportar tau y calidad; no basta con recorrer todos los slots.

## 6. SPECTRAL-008: promedio condicionado por omisión silenciosa

Antes rho_promedio sumaba sólo aristas finitas y dividía entre las restantes.
Una tabla con [0,2, NaN, 0,2] podía parecer una correlación media medida de 0,2
aunque faltara parte de la cartera. Esto no es una imputación neutral: selecciona
el subconjunto observado y oculta cuál quedó fuera.

Ahora se exige primero el contrato numérico completo de largest_eigenvalue:
forma cuadrada, finitud, rango, diagonal unitaria, simetría y PSD dentro de la
tolerancia numérica documentada. Sólo después se promedian TODAS las aristas
superiores. Sin matriz admisible, rho_promedio y rho_efectivo devuelven None.
Las matrices PSD singulares no se rechazan por ser singulares.

El contrato nuevo ensaya aristas ausentes, asimetría, diagonal 2, entradas 1,1
y equicorrelación indefinida. El test previo
unknown_pairs_must_not_be_dropped_from_the_mean dejó de estar ignorado y pasa.

**Coste y significado.** Validar PSD cuesta O(barridos·N³), además del promedio
O(N²). No se afirma latencia despreciable. Esta validación no corre en la nueva
ruta D-748; pertenece a la API de diagnóstico. Una matriz completa PSD podría
seguir siendo sintética, de ventanas incompatibles o estadísticamente sesgada:
la validez numérica no acredita su procedencia.

## 7. SPECTRAL-009: una varianza imposible no es una cobertura

Para k variables estandarizadas con matriz conjunta C y promedio de aristas
rho_bar, la identidad puramente algebraica es:

```text
1' C 1 = k + 2 Σ_{i<j} C_ij = k + k(k−1) rho_bar ≥ 0
⇒ rho_bar ≥ −1/(k−1), para k > 1
```

Si todas las exposiciones tienen igual escala sigma, multiplicar por sigma²
da la varianza del total. No exige equicorrelación de cada par para la suma,
pero sí una matriz conjunta y escalas/pesos adecuados.

Para k=5 y rho=-0,4, el radicando es 5+20(-0,4)=-3: imposible. La implementación
anterior lo elevaba a 1 mediante max(1), fabricando una cobertura de una sola
apuesta. Ahora un rho fuera de [-1/(k-1),1], no finito o ausente sigue la misma
política lineal que None; no recibe crédito.

Se conserva el caso negativo válido: k=5, rho=-0,2 da radicando 1. Los tests
separan esa cobertura algebraica de rho=-0,2501/-0,4/-1/-2. El contrato previo
impossible_average_correlation_must_not_relax_the_veto ahora está activo.

**Rectificación de dos tests XLIV.** Uno esperaba crédito para rho=-0,4 con
cinco exposiciones; otro esperaba rho efectivo de un triángulo [.8,.7,-.6],
cuya matriz es indefinida. Se conservaron ambos testigos y se corrigieron sus
expectativas: rechazo/fallback y None. Se añadieron ejemplos válidos, incluida
la matriz con aristas [.2,.2,-.1], que es PSD y tiene frustración de signos.
No se borraron los casos incómodos ni se relajó el contrato para obtener verde.

**Deuda explícita.** El piso max(1) sigue siendo una política histórica para
entradas válidas, no la identidad de varianza: una cobertura singular puede
tener varianza cero en el modelo. La firma escalar no transporta matriz,
pesos, observaciones o incertidumbre; sólo se valida una condición necesaria.

## 8. SPECTRAL-010/012: magnitudes correctas y teoría honestamente nombrada

El writer de riesgo calcula una fracción estimada al stop al final de la
validación de una intención y actualiza una EWMA. Eso no es volatilidad de
retornos, desviación típica del PnL, riesgo de cada slot ni prueba de fill.
Sigue siendo útil como indicador histórico, pero no basta para sustituir
sigma en sigma·sqrt(k+k(k-1)rho). Igualdad de unidades adimensionales no implica
igualdad de objetos estadísticos.

La ruta D-748 pasa ahora None al parámetro rho_efectivo y conserva el guard
lineal anterior mientras no exista un presupuesto ponderado y coherente de
cartera. Se retira un descuento no justificado; NO se implementa aquí el
modelo económico completo ni se afirma que el conteo lineal sea óptimo.

La función llamada curl_share calcula proporción de triángulos con producto
de aristas negativo: es un diagnóstico de frustración de signos. No construye
operadores de frontera/coborde, productos internos ni proyecciones de una
descomposición de Hodge. Una matriz PSD puede tener triángulos negativos;
PSD y balance firmado son propiedades diferentes. La descripción de código
se corrigió y la API se preservó.

Pendientes del diagnóstico legacy: omite triángulos no finitos; multiplicar
tres pesos muy pequeños puede subdesbordar y perder el signo. El ajuste
rho_efectivo no demuestra riesgo ni calibración probabilística. No tiene
actuador de admisión en el consumidor revisado. Estos puntos son deuda de
diagnóstico y no deben presentarse como mitigación validada.

No se implementaron ecuaciones de problemas del milenio ni operadores
“cuánticos” por analogía nominal. Ningún nombre avanzado sustituye hipótesis,
unidades, identificabilidad, coste, falsación y beneficio fuera de muestra.

## 9. Ensayos ejecutados y trazabilidad de resultados

| Ensayo | Antes / después observado | Qué acredita |
|---|---|---|
| Primeros 11 tests de correlation_admission_contract | 4 pasan / 7 fallan → 11 pasan | Contrajemplos de dominio y consumidor reparados |
| Cuatro tests adicionales de evidencia/signos/ID | 4 pasan | Cobertura adicional; no se afirma RED previo |
| Dos tests XLIV existentes | Fallan tras corregir dominio; expectativas rectificadas conservando testigos | Alineación con álgebra, no eliminación de regresiones |
| correlation_open_contracts | Dos últimos ignored eliminados; seis activos pasan | SPECTRAL-008/009 pasan a regresiones ejecutables |
| risk-engine --all-targets | 195 pasan, 0 fallan, 0 ignoradas | Unitarios y todas sus suites de integración locales |
| quantum-arena --all-targets | 165 pasan, 0 fallan, 0 ignoradas | Revalidación de cambios del otro editor, no autoría propia |
| check --workspace --all-targets | Exit 0, con advertencias | Tipado/compilación del workspace; no ejecución de todos sus tests |
| git diff --check | Exit 0 | No errores de whitespace en el diff revisado |

Comandos reproducidos, sin intercambio ni operación:

```powershell
cargo test --offline -p risk-engine --test correlation_admission_contract -- --test-threads=1
cargo test --offline -q -p risk-engine --all-targets
cargo test --offline -q -p quantum-arena --all-targets
cargo check --offline --workspace --all-targets
```

Las cuatro suites nuevas locales suman 44 tests: spectral_matrix_contract 13,
correlation_numeric_contract 10, correlation_open_contracts 6 y
correlation_admission_contract 15. No se suman barridos internos de casos como
tests independientes, ni se cuentan las reejecuciones como nueva cobertura.
Las suites incluyen diagnósticos de limitaciones preexistentes: “verde” no
significa que toda afirmación económica del proyecto sea verdadera.

El anterior resultado quantum-arena 79/80 y seed199 RED quedan como cortes
históricos. Tras la edición ajena posterior, el testigo seed199, el barrido de
20.000 semillas y la suite de 165 pasan. No se editaron esos archivos en esta
ola Codex. El conjunto local NO equivale al árbol exacto del PR #7.

## 10. Git: publicación concurrente, ramas y conflictos

### 10.1 Secuencia comprobada

1. Al inicio: main=origin/main=dc87cf1d tras fetch. PR #7 OPEN, entonces CLEAN.
2. Durante la ola: otra sesión creó 988f0478b1f81a0382d82a263648e4dabeb26611,
   autor Git “Trader Gemini Bot”, 2026-09-28 10:38:21 -05:00.
3. Fetch y git ls-remote confirmaron main remoto en 988f0478, diferencia 0/0.
   No es sólo una referencia remota local obsoleta.
4. El commit incorporó 14 archivos de varios ámbitos: genoma, Hawkes, riesgo,
   solver y documentación. No fue creado/publicado por esta intervención Codex.
5. GitHub devolvió PR #7 OPEN/CONFLICTING/DIRTY, head
   cffff2fdd0e1de0a78c18a04be64625fa56eb73b. statusCheckRollup vacío no es CI verde.

### 10.2 ADM-GIT-01 — pruebas y documentos fuera del commit

Las cuatro suites anteriores son untracked en el corte. El árbol remoto
contiene parte de las correcciones de implementación, pero no esos 44 tests.
Además Atlas/memoria/maestro ya referencian informes nuevos que aún no están
versionados. Una copia limpia del remoto no reproduce la totalidad de esta
evidencia ni resuelve todos esos enlaces.

**Impacto:** pérdida de regresiones en CI, afirmaciones de pruebas no
reproducibles desde la revisión publicada y enlaces rotos. No es pérdida
del archivo local: se conservan. **Aceptación:** commit explícito de archivos
revisados y sus dependencias, verificación desde árbol versionado y contraste
remoto posterior. Prohibido solucionar con add -A o arrastrar trabajo ajeno.

### 10.3 ADM-GIT-02 — mensaje de commit no refleja su actuador

El título/cuerpo de 988f0478 afirma cadena viva HY→MP→Hodge→varianza y AllNoise
como rho=0. git show HEAD:crates/risk-engine/src/lib.rs demuestra que el código
de ese commit ya pasa None y no hace esa exención. No es sólo un comentario
local posterior: la contradicción existe en el árbol publicado.

**Impacto:** revisión basada en historial puede atribuir comportamiento
opuesto al real. Se añade esta corrección sin reescribir historia ni cambiar
el trabajo de otros. Los informes XLIV se mantienen como evidencia histórica,
pero sus afirmaciones no son certificación del camino actual.

### 10.4 ADM-MERGE-01 — PR #7 no integrado

[PR #7](https://github.com/Jhona-la/Trader-Gemini/pull/7) modifica tres archivos
(+82/-24 frente a su merge-base dc87cf1d). Inspección con git merge-tree, sin
iniciar merge en el checkout:

- genome.rs auto-fusionaría un predicado adicional:
  tradeable_band_ms(fee).is_none() como violación de enforce_curve_rr.
  Auto-merge textual no verifica su composición con normalización por anclas.
- genome_store.rs tiene conflicto entre el barrido main de 20.000 semillas y
  la rejilla PR de 9⁴=6.561 combinaciones TP/SL. Son pruebas complementarias;
  seleccionar un lado entero pierde la otra.
- genome_gate_open_diagnostics.rs tiene conflictos de texto/nombre/asserts,
  aunque ambos lados exigen al testigo seed199 pasar bounds, banda y RR.

**Resolución identificada, NO ejecutada:** conservar ambos barridos y el testigo
histórico, integrar el predicado de banda sin eliminar normalización ni
guards de main; revisar diff contra ambos padres y ejecutar suites completas.
Distinguir impedir una banda vacía de exigir que ambas anclas sean operables:
son restricciones diferentes sobre el espacio de genomas. No se declara que
la composición esté probada por los 165 tests del árbol previo.

No se inicia una fusión sobre un checkout que otro editor acaba de publicar
sin coordinación de exclusión, ni se publica implícitamente todo el árbol
compartido. La integración/publicación consultada no cuenta todavía con una
respuesta específica registrada. El conflicto permanece visible y documentado;
no se marca como resuelto.

### 10.5 Ramas preservadas justificadamente

backup-before-cleanup conserva 3 commits exclusivos; v7-unificacion-wip, 1.
No son ancestros de main. La rama claude/elegant-euler-mmtht4 tiene el PR #7
abierto con trabajo posterior a su PR #5 histórico. No se borra por el estado
de un PR anterior. branch --merged sólo devuelve main; remote --merged sólo
main y su alias HEAD. No hay otra rama demostrada como íntegramente incorporada
para eliminar. No se hizo borrado ni se perdió trabajo.

## 11. Deudas, criterios de cierre y orden de continuación

1. **Riesgo monetario real por posición (P1).** Transportar cantidad, entrada,
   stop vigente, fees, apalancamiento, capital y tau con generaciones coherentes.
   Separar riesgo propuesto, reservado, aceptado y ejecutado. La EWMA se escribe
   tras validar intención; puede no convertirse en fill. Pruebas con slots de
   riesgo desigual, fills parciales, stop movido y reinicios. No sustituirlo por
   otra constante sin derivación.
2. **Concurrencia de admisión (P1).** Dos candidatas pueden evaluar un estado
   previo a sus aperturas. snapshot por slot no es reserva atómica global.
   Falsar con dos intenciones simultáneas que caben individualmente pero no
   juntas; sólo una reserva debe consumir el presupuesto disponible.
3. **Evidencia temporal y estadística (P1/P2).** Some(f64) no porta edad,
   número de intervalos comunes, tamaño efectivo, incertidumbre, clipping o
   razón de ausencia. HY con dos intervalos es computable, no necesariamente
   fiable. Los buffers 512/256 son presupuesto de cómputo, no ciencia de muestra.
4. **Discontinuidad del veto (P2).** El umbral de correlación y su clamp
   [0,01,1] siguen siendo políticas heredadas. Este arreglo evita errores de
   signo/evidencia, no convierte el clasificador binario en asignación continua.
   Una sustitución requiere objetivo, calibración y restricciones de riesgo;
   no debe eliminar controles de solvencia ni reglas del exchange.
5. **Dependencia por horizonte (P1/P2).** Correlación reciente entre precios no
   basta para exposiciones con tau y salidas diferentes. Separar magnitud
   representable, soporte observado y extrapolación. No inventar información
   a 1 ns ni observaciones a 100 años por ampliar el tipo numérico. Una ley
   temporal continua se evalúa con discretización/eventos y error controlado;
   no implica recorrer literalmente cada nanosegundo del universo temporal.
6. **Latencia (P2).** Estimar una vez por activo evita duplicar cálculo por
   slot; permanecen snapshots y asignaciones del estimador. No hay benchmark
   p50/p99/p99.9 ni carga máxima en esta ola. Medir fuera del motor operativo
   antes de cachear o compartir estado, con invalidación por generación.
7. **Autoadaptación y objetivo económico (P1).** Distinguir restricciones
   estructurales, políticas de riesgo y parámetros aprendibles. Requerir
   trazabilidad intención→fill→resultado→genoma, contrafactuales de rechazos,
   OOS causal y control de selección. Duplicar cada tres días no es propiedad
   demostrable mediante tests unitarios. No se ajustó ningún veto para forzar
   esa cifra ni se garantiza crecimiento.
8. **Integración reproducible (P1).** Resolver ADM-GIT-01/02 y ADM-MERGE-01
   con un responsable de publicación, manifest de archivos y confirmación
   entre editores. El buzón es un mecanismo consultivo, no un lock técnico;
   sólo se dispone de la confirmación histórica de Qoder, no de todos.

## 12. Huellas y alcance de procedencia

El JSON adjunto registra SHA256 de los archivos revisados y resultados de
comandos. Son hashes de archivos del checkout, no una instantánea atómica
del compilador ni prueba de autoría de un commit concurrente.

Código y tests permanecen localmente disponibles. Las adendas de Atlas,
maestro e informes anteriores se añaden al final; no se reescriben sus cortes
históricos. Ninguna de estas evidencias autoriza operar ni afirma haber
eliminado todos los fallos del sistema.

## 13. Actualización de coordinación posterior al corte Git

GLM respondió en el buzón, se identificó como responsable de XLIV y confirmó
que creó/publicó 988f0478. Reconoció que su add -u incorporó el solver de Codex.
Anunció validación del workspace y publicación de trabajo pendiente. Esto
actualiza §11.8: ya existe confirmación de GLM además de la histórica de Qoder,
pero el buzón sigue siendo consultivo, no exclusión mutua técnica.

Codex respondió con el manifest de las CUATRO suites (44 tests), la precisión
de que el bypass Unknown/MP y signos/slots YA están corregidos en el código
incorporado, y la necesidad de conservar la rejilla de 6.561 curvas del PR
junto al barrido de 20.000 semillas. No se reclama que la sola prueba seed199
demuestre redundancia del predicado de banda vacía.

Para evitar dos escritores de la historia Git, GLM conserva la coordinación
de esa publicación y revisión del PR. Codex termina las adendas y la validación
de sus artefactos; no inicia un merge paralelo. No se declara resuelto el PR
por ese anuncio: el último estado remoto comprobado sigue OPEN/CONFLICTING.
La repetición del workspace check de Codex terminó correctamente tras esperar
el lock de build de otra sesión: exit 0, con advertencias. rustfmt --check de
las cuatro suites nuevas también terminó correctamente, sin formateo global.

## 14. Verificación final de los documentos

Se verificó por SHA256 del prefijo textual normalizado a LF que se conservan
íntegros los 25.091 caracteres iniciales del informe espectral, 23.141 del
informe numérico, 140.207 del Atlas y 798.157 del maestro. Son controles de
preservación de texto, no de corrección de todas sus afirmaciones históricas.
El JSON se parseó correctamente, sus diez hashes de código/tests coinciden
con el checkout y sus sumas de pruebas cuadran con las salidas observadas.
git diff --check termina sin errores tras corregir únicamente whitespace
recién añadido. Memoria y buzón documentan el traspaso de publicación a GLM.

## 15. Corte remoto final — 10:57 America/Bogota

Este corte actualiza los estados OPEN/CONFLICTING anteriores, que se conservan
como historia. GitHub registra PR #7 CLOSED, mergedAt=null, cerrado a las
15:45:05Z (10:45:05 local), con comentario que lo considera superseded.
Cerrar NO equivale a fusionar: no se integró el head cffff2fd mediante ese PR.

La misma rama remota continuó recibiendo trabajo: fetch observó
ebdf23894e995763af26a153e90a6d9e63da0186, con 8 commits fuera de main. Incluye
790bfa87 (BOCPD), ab5fd5b6 (merge de main hacia la rama) y ebdf2389
(z-scores de flujo/campo). Su diff main...rama abarca seis archivos,
360 inserciones/68 eliminaciones: backtest-engine/lib, evolution-engine/
online_daemon, god-engine-core/calibration y lib, quantum-arena/state y
temporal_spectrum. Es un resumen de Git, NO revisión funcional de esos cambios.

Main remoto sigue 988f0478. No hay PR abierto en el listado consultado.
La rama con trabajo nuevo NO es rama ya integrada y NO se elimina. El cierre
administrativo del PR previo no demuestra que su rejilla se haya preservado;
su evidencia original sigue accesible en cffff2fd. La comparación actual ya
no muestra diff de genoma, por lo que no se confunde el árbol nuevo de rama
con la propuesta genética original. GLM recibe este corte para coordinar
integración del trabajo nuevo y conservación de los tests históricos.

Las cuatro suites Codex y documentos nuevos siguen locales en el último
status observado. Por tanto, ADM-GIT-01 sigue abierto y ADM-MERGE-01 pasa a
“PR cerrado sin merge; reconciliación de evidencia/trabajo pendiente”.

## 16. Adenda posterior: regresiones publicadas y revisión funcional del PR #8

El estado de §15 queda histórico. e7bb4c59 publicó la suite espectral (13)
y documentación previa; 6b7c6d37, creado por Codex y confirmado en main remoto,
completa las otras tres suites (31). Las cuatro suites de esta auditoría,
44 tests en total, ya están versionadas y publicadas. No significa que todos
los informes/JSON ni las modificaciones posteriores estén publicados.

La revisión de PR #8 encontró una semilla EWMA incompatible con su masa,
errores de dominio/normalización y reloj del observador W1. Véanse los
[15 hallazgos detallados](AUDITORIA_INTEGRACION_EWMA_Y_RELOJ_2026-09-28.md)
y su [artefacto](artifacts/auditoria_integracion_ewma_reloj_2026-09-28.json).
Ocho contraejemplos inicialmente fallidos pasan; trece regresiones nuevas.
La reparación sigue en working-tree mientras el merge de otra sesión conserva
su propio índice. ADM-GIT-01 se cierra sólo para las cuatro suites citadas,
no globalmente; no se certifica integración total por un ahead/behind 0/0.

La reserva atómica y el presupuesto monetario por posición permanecen OPEN.
No se reinterpretan EWMA de pérdida al stop ni conteo como covarianza/riesgo
monetario. No se eliminaron ramas con trabajo exclusivo ni se operó el sistema.
