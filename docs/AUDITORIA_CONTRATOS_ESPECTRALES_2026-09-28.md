# Auditoría de contratos espectrales y riesgo multiactivo — 2026-09-28

Estado: reparación numérica acotada y auditoría de integración; NO certificación
integral ni autorización operativa. Rama: main. Base observada:
dc87cf1d0e65231aee54596c7b92fc949e77b041. Hay cambios concurrentes de otros agentes.

[Artefacto estructurado](artifacts/auditoria_contratos_espectrales_2026-09-28.json)
· [Coordinación](../COORDINACION_CODEX_2026-09-28.md)
· [Atlas](../ATLAS_ANALITICO.md)
· [Informe maestro](../INFORME_FORENSE_MAESTRO.md).

## 1. Dictamen y alcance verificable

El defecto prioritario confirmado no es la falta de nombres de teorías: es que
la cadena transforma evidencia insuficiente o algebraicamente inválida en
conclusiones de diversificación. Una fórmula correcta fuera de sus supuestos no
es una protección de riesgo. Añadir más operadores encima amplifica ese error.

Se reparó exclusivamente el solver y la admisión numérica de matrices en
random_matrix.rs. Se conservaron firmas públicas y las pruebas históricas; se
añadieron contratos adversarios. El consumidor y la nueva agregación XLIV están
siendo editados por otra sesión y NO fueron sobrescritos. Sus fallos quedan
abiertos, con cinco contratos ejecutables que siguen fallando explícitamente.

Inventario versionado observado: 1.300 archivos, 396 Rust, 24 Cargo.toml.
Ese censo NO implica lectura integral. La cifra histórica 169/289 del informe
XXXIX corresponde a otro universo y no puede dividirse por 396 sin reconciliar
identidades, altas y modificaciones. No se afirma haber revisado cada archivo.

Lectura completa de módulos en este tramo: random_matrix.rs,
correlation_guard.rs, guard.rs y ruin.rs; lectura dirigida del bloque D-748 y
actualización de riesgo de risk-engine/lib.rs, PositionManager::slots y el
contrato de admisión de cartera. También se inspeccionaron reglas, memoria,
manifiestos y los informes recientes pertinentes. Los dos archivos de pruebas
nuevos se revisaron íntegros. No se reauditaron íntegramente los demás crates.

## 2. Topología diagnóstica: de la raíz a la decisión

```text
ticks por activo, con event_time y calidad
  → incrementos / ventana común / resolución observada
  → estimador de covariación (HY o rejilla)
  → matriz conjunta + identidad de activos + soporte de muestra
  → solver espectral / hipótesis estadística
  → exposiciones por TODOS los slots, signo, tamaño, stop y tau
  → presupuesto de pérdida / decisión de admisión
  → intención → orden → fills → PnL neto conciliado
  → atribución al genoma y actualización causal de evidencia
```

No son intercambiables nodo raíz (datos), nodo de inferencia (estimador), nodo
de decisión (política) y nodo terminal (fill/resultado). Validar una intención
no prueba que se ejecutó; una matriz PSD no prueba que sus aristas se observaron;
un test sintético verde no prueba capacidad predictiva ni rentabilidad.

## 3. Matriz de hallazgos de este corte

Los IDs SPECTRAL son locales a este informe; no renumeran los 305 puntos
históricos ni cuentan como nuevos los defectos heredados que se reexaminan.

| ID | Prioridad | Estado | Contrato afectado |
|---|---|---|---|
| 001 | P1 | Reparado localmente | Autovalor máximo oculto por vector uniforme |
| 002 | P1 | Reparado numéricamente | Matriz finita, simétrica, unitaria y PSD |
| 003 | P1 | OPEN | Falta convertida en cero; MP usado como exención de riesgo |
| 004 | P1 | OPEN | Signos de exposición y slots incompletos |
| 005 | P1 | OPEN, RED | HY normalizado fuera de la ventana común |
| 006 | P1 | OPEN, RED | HY acepta intervalos de duración cero |
| 007 | P2 | OPEN | HY confundido con Pearson y certeza asintótica |
| 008 | P1 | OPEN, RED | Media de correlación sobre pares parcialmente conocidos |
| 009 | P1 | OPEN, RED | Varianza imposible convertida en cobertura admisible |
| 010 | P1 | OPEN | Riesgo al stop / EWMA usado como sigma de cartera |
| 011 | P2 | OPEN, RED | Overflow de Pearson con entradas finitas |
| 012 | P2 | OPEN | Balance firmado presentado como curl de Hodge |
| 013 | P2 | OPEN | Soporte temporal y tamaño efectivo no propagados |
| 014 | P2 | OPEN | Pruebas y documentación sobreafirman garantías |
| 015 | P1 | OPEN | Meta de crecimiento sin validación causal/OOS |

P1 expresa urgencia de revisión del contrato, no una medición de pérdidas
materializadas. Esta ronda no consultó la cuenta ni midió impacto económico real.

## 4. SPECTRAL-001 — un modo dominante puede desaparecer del cálculo

**Ubicación:** random_matrix::largest_eigenvalue, antes de esta reparación.
El vector inicial uniforme era un autovector del modo equivocado. La igualdad
entre dos cocientes de Rayleigh sucesivos solo probaba estacionariedad de esa
iteración, no que se hubiese encontrado el mayor autovalor.

Para C=[[1,-0.9],[-0.9,1]], los autovalores son 0.1 y 1.9. El vector (1,1)
pertenece al primero y es ortogonal al modo dominante (1,-1). La implementación
devolvía 0.09999999999999987. Con N=2 y T=512, el borde usado es 1.12890625:
el error cambiaba el diagnóstico a AllNoise. Con anticorrelación perfecta la
primera multiplicación daba cero y se perdía una matriz PSD perfectamente válida.

**Reparación:** rotaciones cíclicas de Jacobi sobre toda la matriz; convergencia
por norma de Frobenius fuera de diagonal, no por inmovilidad de un único vector.
Se prueban anticorrelación, singularidad, signos de exposición, autovalores
repetidos y un factor equilibrado long/short de 30 activos, cuyo máximo es 12.6.
Las rotaciones son una técnica numérica estándar; su estabilidad requiere
atender al escalado y a la generación de rotaciones. [Anderson, LAPACK Working Note 150](https://www.netlib.org/lapack/lawnspdf/lawn150.pdf).

**Límite del cierre:** se corrigió el resultado numérico y su dominio, no la
identificabilidad estadística de la matriz ni la legitimidad del veto posterior.

## 5. SPECTRAL-002 — no toda tabla cuadrada es una correlación

Antes solo se comprobaba la forma cuadrada. Matrices asimétricas, diagonales
distintas de uno, entradas fuera de [-1,1] y matrices indefinidas podían recibir
un veredicto de mercado. Ejemplo:
C=[[1,.9,.9],[.9,1,0],[.9,0,1]] tiene autovalor mínimo
1-.9*sqrt(2)=-0.2727922061357857. No es una correlación realizable.

Ahora se comprueban finitud, simetría, diagonal y rango; después se examina
todo el espectro diagonalizado para rechazar negatividad material. Se admite
singularidad PSD. No se proyecta silenciosamente a PSD, no se imputan aristas
y no se carga la diagonal para fabricar validez.

La tolerancia 64*EPSILON*N es un presupuesto numérico adimensional para entradas
unitarias; no es un umbral de mercado. Se permiten hasta 64 barridos; si no se
alcanza el residual, el resultado es None, no una aceptación por agotamiento.
Complejidad O(barridos*N^3), memoria O(N^2); falta benchmark de p99 en el host.
Las validaciones numéricas NO pueden detectar que un cero se inventó antes
de llamar a la función.

## 6. SPECTRAL-003 — inferencia de independencia sin evidencia

**Ubicación:** risk-engine/lib.rs, construcción grupo_ids/pares_r/corr y rama
MppVerdict::AllNoise. Solo se miden aristas candidata→posición; las aristas
entre posiciones existentes permanecen en cero. También una correlación None
se convierte en NaN y después se deja en cero al copiar únicamente valores finitos.

**Mecanismo:** el helper pairwise cuenta prudentemente lo desconocido como misma
apuesta; después una identidad fabricada puede producir AllNoise y borrar ese
conteo. Que el solver acepte la identidad es matemáticamente correcto: el fallo
es afirmar que esa identidad fue medida. La reparación 002 no puede descubrirlo.
Una matriz estrella inválida ahora se rechaza en el solver, pero el consumidor
puede continuar entregándola al cálculo de rho efectivo: tampoco queda cerrado.

Además T se fija en MAX_TICKS_MUESTRA=512, capacidad de almacenamiento, no número
de retornos conjuntos válidos. Los estimadores HY por pares no generan por ello
una matriz muestral iid con 512 filas compartidas.

No superar un borde asintótico no prueba independencia; superar el borde tampoco
es por sí solo una prueba calibrada de factor real. Los resultados clásicos del
máximo de Wishart especifican distribución, independencia y límites dimensionales.
[El Karoui, 2003](https://arxiv.org/abs/math/0309355).

**Aceptación exigida:** matriz y máscara de observación inseparables; ventanas
y tamaños efectivos por estimación; validación conjunta/PSD; no aplicar descuento
de riesgo por falta de datos. El consumidor debe manejar explícitamente
Unknown/Invalid/Estimated, y calibrar cualquier prueba con un modelo nulo adecuado.
Test de integración: sin ticks en uno de los activos no se debe convertir el
conteo conservador en independencia.

## 7. SPECTRAL-004 — riesgo incompleto por dirección y por slot

El llamador descarta toda posición cuya dirección difiera de la candidata antes
de estimar r. Para exposiciones lineales firmadas, la correlación de sus PnL es
s_i*s_j*r_ij: long A y short B con r=-0.9 tienen correlación de PnL +0.9.
Descartar el short omite justamente una concentración que interesa al guard.

También se lee únicamente positions.position. PositionManager conserva tres
slots y ofrece slots(); otros controles de cartera sí recorren todos. Los nombres
históricos scalp/swing siguen en el layout, aunque su semántica declarada es
Continuous. La corrección no consiste solo en renombrarlos: el riesgo debe incluir
todas las exposiciones, su tau y su tamaño, conservando compatibilidad de mmap.

**Aceptación:** casos long/long, short/short y signos opuestos; posiciones abiertas
en cada slot; exposición de la propia moneda; contribuciones ponderadas con signo;
snapshots coherentes y pruebas de concurrencia. Una cobertura aparente de precio
no elimina riesgo de basis, ejecución o liquidez.

## 8. SPECTRAL-005 y 006 — contrato temporal de HY roto

**005, ventana:** el numerador usa productos de intervalos que se solapan, pero
las varianzas del denominador incluyen todos los retornos de ambas series,
incluidos los que quedan completamente fuera del intervalo común. En el test
nuevo, dos series coincidentes dan 1. Añadir historia solo a A antes del inicio
de B cambia la medida a 0.00028143784570703424, sin cambiar ningún movimiento
común observado. Esa caída puede crear diversificación artificial.

**006, tiempo:** se construyen intervalos con windows(2) sin exigir timestamps
estrictamente crecientes. La serie [1,2,2,3] produce Some en vez de rechazar el
intervalo nulo. Los punteros presuponen orden; duplicados, desorden e invalidaciones
de precio necesitan un contrato de reparación/rechazo explícito.

**Cierre requerido:** determinar una ventana común y política de bordes, aplicar
el mismo soporte a covariación y variaciones, validar tiempos y precios finitos,
retener diagnósticos de datos descartados y no convertir rechazo del estimador
en observación cero. Pruebas: invariancia a historia disjunta, relojes desordenados,
duplicados, entradas no finitas y tamaños de muestra insuficientes.
Ambos contratos se ejecutaron y siguen RED; no se modificó la implementación ajena.

## 9. SPECTRAL-007 — HY no es Pearson centrado ni una garantía finita

La documentación y el nombre de una prueba afirman equivalencia con Pearson
bajo sincronía. En general es falso: HY suma productos de incrementos sin
restar sus medias; Pearson sí las resta. Con incrementos [0.1,0.2] y [0.2,0.1],
la razón HY sincronizada es 0.8 y Pearson es -1. El nuevo test pasa demostrando
precisamente esa diferencia. Que A se correlacione consigo misma a 1 no prueba
la equivalencia general entre estimadores.

El resumen primario de Hayashi-Yoshida establece consistencia al reducirse los
intervalos de observación; no afirma exactitud para tres ticks ni inmunidad
universal al ruido de microestructura.
[Hayashi y Yoshida, resumen de 2008](https://www.ism.ac.jp/editsec/aism/60/367.pdf).

También se trunca la razón a [-1,1] sin exponer cuánto excedió el intervalo:
una salida acotada no demuestra PSD conjunta ni precisión. La prioridad HY con
tres ticks evita el control de muestra mínima del fallback Pearson.
**Cierre:** nombrar la variable estimada, documentar supuestos, propagar soporte
y calidad, simular procesos comunes sobre un reloj subyacente y medir error,
no solo verificar autocorrelación o una semilla favorable.

## 10. SPECTRAL-008 y 009 — datos incompletos y varianza imposible

**008:** rho_promedio omite aristas no finitas y divide entre las restantes.
Con un par desconocido, aún devuelve una media. Esa media es del subconjunto
observado, no necesariamente de la cartera. El contrato que exige None para una
matriz parcialmente desconocida sigue RED. No basta con comprobar dimensiones.

**009:** para k exposiciones de igual desviación, la varianza es
sigma²*[k+k(k-1)*rho_medio]. Como 1'C1>=0 en una matriz PSD,
rho_medio>=-1/(k-1). Para k=5, rho=-0.4 viola la cota -0.25 y produce -3*sigma².
La implementación aplica max(1) y autoriza una cobertura que no existe.
El test histórico XLIV la llama consistente; el nuevo contrato normativo falla
mostrando la aceptación. Un piso de seguridad sobre la pérdida no es una
validación algebraica de los datos.

**Cierre:** rechazar evidencia inconsistente antes de usarla, propagar la misma
identidad/dimensión de cartera a media y conteo, y probar el borde PSD, su vecindad
y matrices incompletas. Proyectar a PSD, si se adopta, debe informar distorsión y
ser una decisión metodológica validada, nunca una reparación invisible.

## 11. SPECTRAL-010 — pérdida al stop no equivale a volatilidad de cartera

La fuente arena.riesgo_por_operacion se actualiza con
safe_vol*safe_lev*sl_pct/current_cap, suavizada por EWMA. Es una fracción
estimada de pérdida al stop para una intención validada, NO una desviación
estándar de retornos a un horizonte común ni un vector de riesgos por posición.
Se actualiza antes de devolver ValidatedOrder, no tras confirmar el fill.

Usarla como sigma compartida por todas las posiciones introduce supuestos de
igual magnitud, misma variable aleatoria y misma tau que no se han demostrado.
El candidato final aún puede dimensionarse distinto de esa media histórica.
El fallback tope/8 para riesgo desconocido fija capacidad de arranque, pero su
etiqueta de conservador no está respaldada por riesgo medido de posiciones adoptadas.

**Cierre:** separar presupuesto de pérdidas (stops, gaps, fees, slippage,
liquidación) de covarianza del PnL a horizonte dado; usar exposición realmente
ejecutada y reservas de órdenes pendientes; admitir incertidumbre sobre adopción.
Ningún cambio a ese modelo ni a su límite se implantó en esta ronda.

## 12. SPECTRAL-011 y 012 — error numérico y sustitución de teoría

**011:** pearson calcula sqrt(saa*sbb). Con a=b=[-1e150,1e150], cada suma de
cuadrados y el numerador son finitos, pero su producto desborda: el coeficiente
devuelto es 0 en vez de 1. Es un fallo de invariancia de escala, reproducido RED.
No se afirma que precios de esa magnitud hayan llegado al motor. Reparación
pendiente: normalización escalada/estable, tests de homotecia y extremos finitos.

**012:** curl_share_desbalanceado cuenta triángulos con producto negativo.
Eso es un indicador de balance de signos, no una descomposición de Hodge:
no hay cochain orientada, operadores de incidencia ni proyecciones ortogonales.
Una correlación 3x3 equicorrelacionada a -0.2 es PSD (autovalores .6,1.2,1.2)
y tiene triángulo negativo. La ausencia de un solo factor de signos no demuestra
inconsistencia de una distribución multivariante. Empujar rho hacia 1 o 0 con
esa fracción es una política de estrés heurística, no la identidad de Markowitz.

**Cierre:** conservar, si interesa, el diagnóstico con nombre correcto; separar
penalización de estrés de correlación estimada; validar utilidad incremental OOS.
Para Hodge genuino hay que definir el flujo de aristas, las incidencias y el
producto interno antes de discutir qué decisión justifica su energía residual.

## 13. SPECTRAL-013 y 014 — espectro observado y evidencia de pruebas

**013:** el camino inspeccionado toma 512 ticks y una rejilla de hasta 256 puntos.
Cuando esta satura, construye desde el inicio del solape, no desde su final.
La medida no depende de la tau candidata. Un límite de memoria puede ser legítimo,
pero no representa por sí mismo todo el espectro ni demuestra estacionariedad.
La fórmula 1/sqrt(K-3) es una aproximación asociada a Fisher-z bajo supuestos,
no una garantía de resolver cualquier correlación en datos dependientes.

Un motor continuo puede representar funciones de tau mediante bases/adaptación
event-driven sin enumerar cada nanosegundo. Necesita distinguir dominio matemático,
resolución del feed, edad observada, incertidumbre y horizonte operativo. Declarar
soporte 1 ns–100 años no crea datos ni resolución predictiva a esas escalas.

**014:** los informes XLI–XLIV dicen que la cadena separa ruido/estructura y queda
completa; los contraejemplos anteriores limitan expresamente esas conclusiones.
El fixture HY asíncrono usa incrementos por índice con relojes distintos: eso
no establece por sí solo muestreo de una misma trayectoria latente. Su generador
usa seed>>33 dividido por u32::MAX y resta .5: el supuesto ruido no queda centrado
como afirma el comentario. Los verdes existentes no equivalen a validación OOS.

No se borran esos informes. Esta adenda corrige su interpretación. La prueba
histórica de varianza imposible sigue verde: demuestra por qué hay que auditar
también los asserts, no solo contabilizar tests.

## 14. SPECTRAL-015 — objetivo económico y autoevolución verificable

Duplicar cada 72 horas corresponde a log-crecimiento ln(2)=0.693147 por periodo
y, si se supone tasa diaria constante, 2^(1/3)-1≈25.9921% neto diario. Es una
conversión algebraica del objetivo, no una estimación de factibilidad. Ninguna
prueba de este tramo demuestra que el sistema lo alcance o lo sostenga.

La evaluación debe distinguir retorno esperado, mediana del capital, probabilidad
de alcanzar la meta, drawdown, cola de pérdidas, capacidad y costes. El crecimiento
logarítmico esperado requiere considerar insolvencia: no puede ocultarse una
trayectoria quebrada eliminándola del promedio o aplicando un epsilon al capital.

Para hablar de autoevolución se necesita: identidad/versionado de genoma, features
as-of, intención y causas de veto, evaluación contrafactual sin look-ahead,
atribución de fills/costes, actualización con evidencia nueva, holdout temporal,
criterio de promoción y rollback. Los pendientes históricos de paridad, feedback
selectivo y calibración no se declaran resueltos aquí.

Las teorías se admiten por variable, supuestos y falsación, no por prestigio.
El cálculo espectral de correlaciones no implementa la brecha de masa de
Yang-Mills; Clay la presenta como problema sin resolver. Esa analogía no otorga
una garantía financiera. [Clay Mathematics Institute](https://www.claymath.org/millennium-problems/).

## 15. Resultados reproducibles y límites

| Ejecución | Resultado observado | Significado |
|---|---|---|
| spectral_matrix_contract, antes | 4 pasan, 9 fallan | RED funcional, no fallo de compilación |
| spectral_matrix_contract, después | 13 pasan | Reparación numérica acotada |
| correlation_open_contracts --ignored | 5 fallan | Cinco deudas reproducidas, NO reparadas |
| risk-engine --all-targets | 165 pasan, 5 ignoradas | Incluye diagnósticos históricos; no certificación total |
| workspace --all-targets, cargo check | Éxito con warnings | Tipado/compilación; no ejecución integral |

Nuevas pruebas únicas: 19 (13 contratos del solver, 1 contraejemplo semántico,
5 contratos OPEN). Reejecutarlas no aumenta la cuenta. Las cinco ignoradas están
marcadas por ID y motivo; se ejecutaron aparte en rojo para no ocultarlas bajo
el resultado verde de la suite normal. No se ignoró ninguna prueba previa.

```text
cargo test -p risk-engine --test spectral_matrix_contract -- --nocapture
cargo test -p risk-engine --test correlation_open_contracts -- --ignored --nocapture
cargo test --offline -p risk-engine --all-targets
cargo check --offline --workspace --all-targets
```

Persisten advertencias en quantum-arena, risk-engine, god-engine-core,
evolution-engine y bins/tests. No se ejecutó cargo fix sobre archivos ajenos.
No hubo órdenes, consultas privadas, entrenamiento, promoción, reinicio del motor,
deploy, commit, push, merge o fetch. El test de riesgo no invoca el motor operativo.

Fuentes: se leyó la habilidad Firecrawl y sus reglas. Su índice de papers falló;
la revisión recurrió a la herramienta web y a fuentes primarias enlazadas arriba.
No se enviaron código privado, credenciales ni datos de cuenta. La investigación
guio la separación entre algoritmo numérico, hipótesis estadística y analogía física.

## 16. Hoja de ruta ordenada por dependencia

1. Acordar con Claude/GLM los cambios del consumidor: no otorgar independencia
   a datos desconocidos y cubrir todos los slots/signos. Confirmación aún pendiente.
2. Sustituir matriz suelta por evidencia con IDs, ventana, máscara, estimador,
   versión, N válido y soporte efectivo; hacer imposible confundir capacidad con T.
3. Corregir los cinco contratos OPEN y añadir integración de decisiones/vetos,
   no solo tests de helpers; revisar los asserts históricos que aceptan entradas imposibles.
4. Definir pérdida al stop frente a volatilidad, con exposición realizada y pendiente.
   La covarianza no reemplaza límites operativos, de liquidez o de insolvencia.
5. Validar inferencia multiescala causal con baseline y datos retenidos; decidir
   ruido microestructural/prepromediado, regularización y pruebas finitas.
6. Auditar ledger/genoma/feedback y paridad de rutas. No promover un genoma por
   el mismo tape con el que se eligieron teorías, thresholds y calibraciones.
7. Medir latencia/asignaciones/p99 y mantener presupuesto de cómputo explícito.
8. Reconciliar inventario histórico con los 396 Rust actuales y continuar las
   lecturas por archivo. No reemplazar esta deuda con una afirmación de omnisciencia.

## 17. Coordinación y conservación documental

Canal compartido: COORDINACION_CODEX_2026-09-28.md, referenciado desde la memoria
que cargan Codex y Claude. GLM usa otro editor; no se verificó que lea esa memoria.
No hay confirmación de recepción de ninguno de los dos. El acceso local falló
temporalmente y se recuperó; se usó el ejecutable apply_patch para las ediciones.

Los hashes del artefacto son del corte inspeccionado, no de una versión remota.
Los cambios ajenos quedan preservados y pueden evolucionar después del chequeo.
Este informe es aditivo y no sustituye la matriz histórica ni los informes previos.


## 18. Actualización de coordinación y referencias al cierre

Después del estado inicial de §17, el agente que firma como Qoder confirmó
lectura del buzón. Sus archivos son god-engine-core/lib.rs y
signal-engine/flow_excitation_confluence.rs; NO pertenecen al trabajo XLIV de
riesgo. Se leyó su diff: el nombre de publicación/consumo del gen coincide y
no se solapa con este solver. Sus conteos 59/59 y 114/114 son resultados
reportados por él, no reejecutados aquí; el chequeo global sí se ejecutó aquí.

Feedback remitido por buzón: el cambio suma hasta 0.9 a un ratio base 1.6;
eso equivale a +56.25% respecto de 1.6, no +90%. El test mostrado usa el registro
global; falta un contrato que distinga dos símbolos con valores simultáneos.
No se reescribió su fix ni se certificó el estimador subyacente de Hawkes.
La segunda sesión no ha confirmado recepción al cierre de este apartado.

Se reparó adicionalmente el enlace al artefacto XXXIX mediante un JSON histórico
reconstruido y una adenda explícita; no se recrearon hashes ni logs ausentes.
Las fichas SPECTRAL corresponden a este corte y no inflan la matriz histórica.


## 19. Adenda de reparación numérica e integración Git (segundo corte del día)

[Informe nuevo](AUDITORIA_INTEGRACION_Y_CONTRATOS_NUMERICOS_2026-09-28.md) ·
[Artefacto](artifacts/auditoria_integracion_numerica_2026-09-28.json).
Se conservan los estados históricos de §§1–18; esta adenda los actualiza:
SPECTRAL-005 tiene reparación parcial (historia totalmente disjunta); 006
valida reloj/quotes en HY y fallback; 011 corrige escala/constancia/pares.
Los tres contratos antes ignorados pasan activamente. Diez nuevas pruebas
reproducen y cubren los casos numéricos. Risk-engine: 178 pasan, 2 ignoradas
OPEN; las dos OPEN de 008/009 ejecutadas aparte siguen fallando. Los bloques
XLIV y el consumidor no se modificaron. No es un cierre de riesgo integral.

Fetch y GitHub verificados: main=origin/main=dc87cf1d, PR #7 OPEN con trabajo
no integrado. Backups locales contienen commits exclusivos: no se borraron.
Durante la revisión el genoma cambió en otra sesión; seed199 primero falló,
luego su barrido pasó. Última suite quantum-arena observada: 79 pasan/1 falla
(D-658: igualdad exacta tras reconstrucción). Véase la secuencia y sus límites
en el nuevo informe. Sin commit/push/merge; publicación consultada pendiente.
El JSON anterior permanece como corte histórico, no se reescriben sus hashes.

## Adenda 2026-09-28 — admisión multiactivo y tercer corte de evidencia

[Informe detallado](AUDITORIA_ADMISION_MULTIACTIVO_2026-09-28.md) · [Artefacto](artifacts/auditoria_admision_multiactivo_2026-09-28.json).
Se actualizan los estados anteriores sin borrar testigos: SPECTRAL-003 retira
Unknown→cero y exención AllNoise; 004 incorpora TODOS los slots y ambos signos;
008 exige matriz completa numéricamente válida; 009 no acredita rho imposible.
010 retira el descuento sigma sobre EWMA de pérdida al stop, pero el presupuesto
monetario por posición sigue OPEN. Frustración de signos no es Hodge ni PSD.

Riesgo: 195 pruebas pasan, 0 ignoradas. Genoma del otro editor: 165 pasan.
Los antiguos 79/80 son historia, no fallo actual. Se añaden 15 tests de admisión;
7 testigos RED→GREEN entre los 11 iniciales, más 4 de cobertura. Dos expectativas
XLIV inválidas se rectifican CONSERVANDO sus testigos y añadiendo casos válidos.
No hay auditoría integral archivo-por-archivo ni validación de +100%/72 horas.

Publicación concurrente: main remoto cambió a 988f0478, confirmado con fetch y
ls-remote. Ese commit incorpora reparaciones pero en el corte deja fuera cuatro
suites nuevas (44 tests) e informes enlazados desde documentos publicados.
Su mensaje describe actuador varianza/Hodge, mientras su código ya pasa None.
PR #7 sigue abierto, ahora CONFLICTING; main y PR aportan barridos complementarios
que no deben perderse. No hay otra rama íntegramente integrada para borrar.

GLM confirmó en el buzón que creó/publicó 988f0478 y que prepara integración de
lo pendiente; reconoció arrastre de trabajo de Codex. Se le notificó preservar
las cuatro suites y la rejilla de 6.561 curvas del PR, además de los 20.000 casos
de main. Codex no abre un merge/commit en paralelo a ese responsable; añade
documentación y evidencia. Riesgo ponderado, reserva atómica, muestra efectiva y
tau siguen abiertos. No hubo operación, promoción, entrenamiento ni despliegue.

Actualización de Git a las 10:57: PR #7 CLOSED sin merge, no integrado. La rama
remota reutilizada avanzó a ebdf2389, con 8 commits fuera de main y cambios
nuevos BOCPD/calibración (6 archivos +360/-68), aún no auditados funcionalmente
en esta ola. Main remoto sigue 988f0478. No borrar rama por PR cerrado. Véase
§15 del informe de admisión; GLM recibió el aviso y mantiene la coordinación.

