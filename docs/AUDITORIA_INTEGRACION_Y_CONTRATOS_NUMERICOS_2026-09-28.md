# Auditoría de integración y contratos numéricos — 2026-09-28

Estado: reparaciones locales verificadas de estimadores; integración Git y
consumidor de riesgo pendientes. NO es certificación integral ni permiso de
operación. Esta ronda continúa el informe espectral, no reemplaza sus evidencias.

[Artefacto de este corte](artifacts/auditoria_integracion_numerica_2026-09-28.json) ·
[Informe espectral previo](AUDITORIA_CONTRATOS_ESPECTRALES_2026-09-28.md) ·
[Buzón entre editores](../COORDINACION_CODEX_2026-09-28.md).

## 1. Dictamen ejecutivo y alcance

Se reprodujeron nueve contratos numéricos fallidos y, durante la revisión del
parche, un décimo: una serie constante de longitud 49 recibía varianza ficticia
por redondeo. Los diez pasan tras la corrección. Tres contratos antes ignorados
de HY/Pearson vuelven a la suite activa. El total observado de risk-engine es
178 aprobados y dos ignorados OPEN; esos dos se ejecutaron aparte y FALLARON.

La revisión Git demuestra que main local y origin/main coinciden en el commit
observado, pero NO que todo el trabajo haya llegado a main remoto: existen
cambios sin commit, un PR abierto y ramas históricas con commits exclusivos.
La ausencia de marcas de conflicto tampoco demuestra integración semántica.

En paralelo se ejecutó el nuevo diagnóstico local de otra sesión sobre genomas.
Falló con semilla 199, tasa 0.5: una reparación de SL anterior al reparador
general no evita que este vuelva a eliminar la banda operable. Se remitió el
contraejemplo al editor responsable. No se modificaron sus archivos.

Inventario versionado: 1.300 archivos, 396 Rust, 24 manifiestos Cargo. No es una
métrica de lectura. En este tramo se releen las funciones de correlación, su
consumidor D-748, los tests propios completos y fragmentos de reparación,
serialización y validación del genoma; se revisa íntegro el diff de tres
archivos del PR #7. NO se han auditado todos los archivos ni toda la teoría.

## 2. Estado Git realmente comprobado

Corte local registrado alrededor de 10:02 America/Bogota. El fetch remoto se
hizo en esta ronda; la observación no garantiza que un tercero no publique después.

| Objeto | Resultado | Decisión |
|---|---|---|
| HEAD/main | dc87cf1d0e65231aee54596c7b92fc949e77b041 | Mantener main |
| origin/main tras fetch | mismo SHA; divergencia 0/0 | Commit sincronizado, árbol de trabajo NO publicado |
| PR #7 | OPEN; base main; head cffff2fdd0e1de0a78c18a04be64625fa56eb73b | Revisado; no integrado por esta sesión |
| Rama remota claude/elegant-euler-mmtht4 | cuatro commits alcanzables adicionales; diff efectivo de tres archivos | No borrar |
| backup-before-cleanup | tres commits exclusivos; no ancestro ni equivalencia de parches | Conservar |
| v7-unificacion-wip | un commit exclusivo; no ancestro ni equivalencia de parche | Conservar |
| Índice | sin rutas no fusionadas; sin staging al inspeccionarlo | No hay conflicto textual que resolver en el índice |
| Worktrees | solo checkout actual de main observado | No crear checkout adicional |
| Ramas ya integradas disponibles para limpieza | ninguna aparte de main y su referencia simbólica | Cero eliminaciones |

Comandos de evidencia: git fetch --prune origin; rev-parse; rev-list --left-right
--count; branch --merged; branch -r --merged origin/main; git cherry; status;
diff --diff-filter=U; gh pr list/view. No se dedujo el estado remoto de una
referencia local obsoleta. La ejecución de fetch actualizó referencias; no
publicó código ni modificó archivos de trabajo.

### GIT-001 — confundir sincronía de commits con publicación completa (P1, abierto)

Un indicador 0 ahead / 0 behind solo compara dos commits. No incluye archivos
modificados ni nuevos sin seguimiento, ni el contenido de PRs abiertos. Dar por
terminada la integración a partir de ese indicador perdería las mejoras locales
en una clonación nueva, además de ocultar los contratos que siguen en rojo.

Criterio de cierre: atribuir y revisar cada bloque, probar el árbol exacto que
se va a publicar, crear commits acotados autorizados, comprobar recepción en
origin/main y volver a inspeccionar lo que permanezca sin commit. No usar add -A,
stash global, reset, descarte de cambios ni una limpieza basada en nombres.

### GIT-002 — ancestro en el grafo no implica contenido aplicado (P2, advertencia)

La rama del PR #7 contiene el merge 864177f9, realizado con estrategia ours:
su árbol es idéntico al primer padre 0bb9885f. El segundo padre ae470b95
(memoria del PR #6 cerrado) queda en la historia sin aplicar su contenido.
Se comparó con AMBOS padres. No debe describirse como aplicación de todos los
cambios del segundo padre; tampoco corresponde resucitar automáticamente una
memoria obsoleta para hacer coincidir la historia con esa descripción.

El PR #5 se fusionó antes, pero la misma rama recibió trabajo posterior. Por
eso su etiqueta histórica MERGED no permite borrar la punta actual del PR #7.
GitHub registra el PR #4 como MERGED, aunque un bloque antiguo de la memoria
lo describía como cerrado/sustituido. Se conserva ese bloque y se añade esta
rectificación basada en el estado remoto observado.

### Situación del PR #7

[PR #7: reparación de banda operable](https://github.com/Jhona-la/Trader-Gemini/pull/7).
Estado observado: MERGEABLE/CLEAN, pero statusCheckRollup vacío. No equivale a CI
aprobada. El diff tiene 82 inserciones y 24 eliminaciones en genome.rs,
genome_store.rs y genome_gate_open_diagnostics.rs.

La corrección incorpora banda inexistente al predicado violated de
enforce_curve_rr. Es relevante: reducir SL en el paso 2 puede eliminar la
banda después del chequeo inicial. El PR añade un barrido de 9^4 = 6.561
combinaciones de coeficientes y convierte la semilla 199 de diagnóstico de
defecto a regresión normativa. Se conserva la exigencia del gate.

Esos tests del PR NO fueron ejecutados aquí sobre su árbol: se inspeccionaron.
No atribuirles los resultados del árbol local ni los publicados por su autor.
Durante el trabajo aparecieron cambios locales concurrentes en los mismos
archivos. La integración requiere comparar ambos enfoques, preservar autoría
y comprobar la composición; ya no basta la condición MERGEABLE de GitHub.

## 3. Nodo raíz: precio, reloj y dominio numérico

### SPECTRAL-006 — duración no positiva y datos inválidos (P1, reparación local)

Antes, HY construía intervalos con windows(2) sin exigir orden estricto.
Duplicados [1,2,2,3] y retrocesos [1,3,2,4] infringían las hipótesis de los
punteros de solape. Con cotizaciones inválidas, algunos intervalos se omitían
o se calculaba un mid desde una sola cotización válida: el número devuelto
dejaba de describir la muestra entregada.

Ahora ambos estimadores, HY y la rejilla de respaldo, validan todas las
cotizaciones bid/ask como positivas y finitas y todos los tiempos como
estrictamente crecientes. Invalididad devuelve None, no cero. Validar solo HY
no bastaría: or_else podía volver a producir una estimación con la misma
entrada temporalmente inválida en la rejilla.

No se inventa un epsilon temporal. Dos eventos legítimos pueden compartir un
timestamp de milisegundos: requieren una política upstream de agregación o
secuenciación con reloj adecuado, no la fabricación de un nanosegundo ficticio.
Este parche no implementa esa política; conserva la incertidumbre. En el
consumidor actual None aún puede terminar convertido en cero (SPECTRAL-003),
por lo que NO se certifica seguridad extremo a extremo.

Pruebas: duplicados, desorden, cero, negativo, NaN e infinito en cotización;
duplicado dentro de una serie larga que antes pasaba la rejilla. El rechazo
de datos fuera del dominio matemático es un contrato de integridad, no una
clasificación rígida del mercado.

### NUMERIC-001 — mid y log-retorno fuera del rango intermedio (P2, reparado)

Dos precios positivos y finitos próximos a f64::MAX no garantizan que su suma
sea finita. (bid+ask)/2 podía desbordarse. También ln(p1/p0) fallaba para
p0=1e-300 y p1=1e300 aunque ambos logaritmos fueran perfectamente representables.

El mid se calcula como bid+(ask-bid)/2 después de validar el dominio positivo.
El log-retorno utiliza ln_1p((p1-p0)/p0) para preservar variaciones pequeñas;
si esa razón no es finita o redondea a -1 se usa ln(p1)-ln(p0). No se inventan
precios sustitutivos ni se descartan silenciosamente intervalos inválidos.

La prueba compara una serie consigo misma con precios extremos y exige 1,
no un valor arbitrario finito. Esto valida aritmética de entrada, no que esos
precios tengan sentido de mercado. Tick size, spread cruzado, edad de quotes,
secuencia de exchange y procedencia son controles separados no cerrados aquí.

## 4. Nodo de estimación: ventana, escala y degeneración

### SPECTRAL-005 — contaminación de HY por historia disjunta (P1, cierre parcial)

Definamos W=(max(t_A0,t_B0), min(t_Afinal,t_Bfinal)]. Cada incremento r_i conserva
su intervalo de observación I_i. Solo se retienen intervalos con intersección
de duración positiva con W. Sobre ese soporte se calculan:

    Q_A = suma r_Ai²; Q_B = suma r_Bj²
    H_AB = suma r_Ai * r_Bj para pares con solape positivo
    h = H_AB / (sqrt(Q_A) * sqrt(Q_B))

La implementación anterior incluía historia enteramente ajena a W en Q_A/Q_B,
aunque no pudiera contribuir a H_AB. Añadir un gran movimiento de A antes de
que B comenzara reducía h sin cambiar ningún movimiento común. El contraejemplo
histórico bajaba de 1 a aproximadamente 0.0002814. No era evidencia de menor
dependencia sino una inconsistencia de soporte.

La corrección excluye intervalos completamente disjuntos de numerador y
denominador; se prueba historia tanto anterior como posterior y simetría al
intercambiar A/B. El contrato histórico antes ignorado pasa ahora activamente.

Límite importante: un incremento que CRUZA el borde de W se conserva entero.
No existe un precio observado justo en ese borde para dividirlo sin un modelo.
El parche documenta esta convención, pero no demuestra estimación exacta sobre
W ni elimina incertidumbre finita de frontera. Faltan diagnósticos del tamaño
de esos intervalos, edad, número de cruces y ruido microestructural. Por eso la
ficha completa queda PARCIAL, no cerrada por completo.

### SPECTRAL-011 — Pearson no invariante a la escala (P2, reparación local)

Pearson es el producto interior de vectores centrados dividido por el producto
de sus normas. La implementación formaba cuadrados y luego saa*sbb en las
unidades originales. Para [-1e150,1e150], el producto de varianzas desbordaba
aunque el coeficiente exacto de la serie consigo misma fuera 1. En escalas
pequeñas, los cuadrados podían desaparecer por underflow.

Cada serie se divide ahora por su máximo absoluto antes del centrado, llevando
los datos a [-1,1]; una escala positiva se cancela en el cociente. Las raíces
se multiplican después de calcularse. Se verifican escalas 1e-300, 1, 1e150,
1e300 y f64::MAX, signos opuestos y traslaciones con reescalado.

Además se exige igual longitud y finitud de TODOS los pares. Usar min(lenA,lenB)
descartaba silenciosamente observaciones finales, incluso NaN. No corresponde
al estimador decidir una alineación por truncado; los llamadores deben construir
los pares que representan el mismo contrato de observación.

### NUMERIC-002 — una constante puede obtener varianza de redondeo (P2, reparado)

El primer parche de escala superó nueve pruebas, pero una prueba adversaria
adicional encontró n=49 como testigo. Al calcular suma*(1/n), el redondeo de la
media puede dejar un residuo idéntico y no nulo en todas las observaciones.
Sus productos forman una falsa correlación unitaria aunque la desviación real
sea cero y la correlación esté indefinida.

Se comprueba constancia exacta de cada serie finita antes de centrar. La prueba
recorre longitudes 2..512 y ambas orientaciones contra una serie variable.
El fallo se reprodujo antes de añadir la comprobación; después pasa. Es un
ejemplo de por qué revisar la reparación es distinto de reejecutar solo sus
casos favorables. No se afirma exactitud ilimitada de coma flotante.

### NUMERIC-003 — piso absoluto de variación disfrazado de validez (P2, reparado)

HY exigía denom > 1e-18. Una serie [100,100.00000001,100] era no constante,
pero resultaba None por ese literal. El cociente normalizado no requiere un
piso absoluto de volatilidad: requiere variaciones positivas representables,
finitud y evidencia estadística aparte.

Se exige denominador positivo y finito, calculado como sqrt(Q_A)*sqrt(Q_B).
Dos incrementos siguen siendo el mínimo computable preexistente; NO se elevan
a prueba de precisión, capacidad predictiva o suficiencia muestral. Un filtro
de incertidumbre futuro debe derivarse del error y datos observados, no de la
unidad numérica de esta multiplicación.

## 5. Nodo de decisión: deudas que esta reparación NO oculta

| Contrato | Evidencia vigente | Criterio mínimo de cierre |
|---|---|---|
| SPECTRAL-003/013 | None/aristas no medidas dejan ceros; T se fija en 512 | Evidencia con máscara, tiempos, tamaño efectivo y resultado Unknown/Invalid/Estimated |
| SPECTRAL-004 | consumidor lee solo slot principal y descarta signo opuesto | Exposición firmada de todos los slots y tamaños, con snapshot coherente |
| SPECTRAL-008 | rho_promedio omite NaN; test normativo sigue RED | No certificar media completa de matriz parcial |
| SPECTRAL-009 | k=5, rho=-0.4 implica 5+20*(-0.4)=-3; se clampa a 1 | No convertir varianza imposible en cobertura admisible |
| SPECTRAL-010 | EWMA de riesgo al stop se usa como sigma | Separar pérdida al stop, volatilidad y peso real por exposición |
| SPECTRAL-007/012/014 | HY/Pearson, Hodge/balance firmado y garantías finitas mezcladas | Operador, hipótesis, prueba nula y documentación correspondientes |

Los bloques XLIV y risk-engine/lib.rs pertenecen a trabajo concurrente. Se
preservaron y se avisó por el buzón. Los dos contratos ignorados no son éxitos:
al ejecutarlos con --ignored ambos FALLAN. Una prueba histórica que acepta
rho=-0.4 para cinco posiciones sigue verde porque su expectativa es incorrecta.

El clipping HY a [-1,1] sigue siendo compatibilidad heredada, no reparación
estadística. También queda por resolver el transporte de diagnósticos: Option
no distingue reloj inválido de ventana insuficiente. Ni el solver PSD corregido
ni las nuevas validaciones descubren un cero inventado previamente.

### SPECTRAL-014, ampliación — el fixture asíncrono no prueba lo que afirma

El test xliii_b_hy_asincrona_mantiene_correlacion construye el factor mediante
((seed >> 33)/u32::MAX - 0.5)*0.02. El desplazamiento entrega 31 bits:
la fracción está entre 0 y aproximadamente 0.5; el factor queda no positivo,
no centrado alrededor de cero. La misma secuencia por índice se asigna a
relojes de pasos 10 y 13: el índice común no es el mismo tiempo físico.

Un h alto entre incrementos con deriva común no prueba recuperación de la
correlación de innovaciones ni consistencia asintótica con observación asíncrona.
Tampoco se evalúa convergencia de malla, sesgo o cobertura sobre réplicas. Esto
se deduce de la construcción exacta, no de su nombre ni de que esté verde.
No se alteró ese test concurrentemente: queda documentado para reemplazar su
afirmación por un proceso latente común muestreado en tiempos distintos y
un contraste reproducible con verdad conocida. El test de identidad sigue
siendo útil como identidad numérica, no como equivalencia general con Pearson.

## 6. Genoma: dos reparadores pueden contradecirse

### GEN-001 / FMT-216 — banda reparada y después destruida (P1, reproducido)

El árbol local concurrente añade normalize_sl_curve_friction_floor al mutar SL
y una búsqueda determinista de 20.000 semillas. La normalización se aplica antes
de enforce_curve_rr; el reparador posterior puede reducir SL otra vez. Se ejecutó
el diagnóstico en el estado entonces presente, sin modificar esos archivos:

    seed = 199; mutation_rate = 0.5
    SL mutante = 0.0012410326295169774, pendiente b = 0
    SL tras roundtrip = 0.0013026105844166744, pendiente b = 0
    piso de fricción = 0.0015384615384615385
    resultado: GenomeEnvelope::validate rechaza por banda inexistente

Una pendiente cero hace constante el SL: no hay un horizonte alternativo que
salve ese ejemplar. El fallo no demuestra que todo genoma sea inviable ni que
relajar el gate sea correcto. Demuestra que los invariantes del generador y
del verificador no están cerrados bajo composición.

El PR #7 actúa en el predicado del reparador general; el cambio local actúa
antes y usa dos anclas. Proyectar ambas anclas sobre un piso puede modificar
la pendiente aunque el comentario inicial diga que permanece libre. Esa
composición necesita comparación semántica, no aceptar ambas por ausencia de
conflicto textual. No se afirma haber probado aquí el árbol del PR.

Cierre exigido: semilla 199; barrido de coeficientes; mutación y from_vector
por separado; roundtrip repetido; límites, finitud, banda y RR conjuntamente;
paridad entre carga, hot-swap y rutas reales. Evitar que el mismo defecto se
repare de formas incompatibles en dos sitios. La identidad/generación activa
debe acompañar a cada decisión y resultado si se pretende atribuir aprendizaje.

Los archivos del genoma continuaron cambiando durante esta ronda. El resultado
rojo anterior es evidencia del corte ejecutado, no una garantía sobre una
edición posterior no probada. El buzón conserva los valores y el comando.

## 7. Continuidad temporal y propósito de los cálculos

Un motor continuo no necesita etiquetas scalping/swing para decidir. Sí necesita
un dominio temporal representable, una base numérica finita, evidencia por escala
y errores explícitos. Un timestamp más fino no crea observaciones; evaluar una
ley a 100 años no crea historial a ese horizonte. Hay que distinguir resolución
de reloj, tiempo de evento, horizonte de decisión y soporte estadístico.

Este parche opera sobre la validez de observaciones y cocientes, no sobre un
clasificador de regímenes. No introduce umbrales de volatilidad ni dos motores.
La rejilla de respaldo, límites de buffer y anclas existentes NO se declaran
universales ni óptimos: su justificación y propagación quedan en las deudas.

La propuesta de duplicar capital cada 72 horas sigue siendo una meta no
validada, no una restricción que permita forzar resultados o retirar protecciones.
No se ha demostrado edge neto fuera de muestra, capacidad de ejecución ni
estabilidad de promoción. Las fórmulas adicionales requieren variables,
unidades, hipótesis y pruebas falsables; su complejidad o procedencia física
no sustituye estos contratos.

## 8. Pruebas ejecutadas y estado de aceptación

| Ejecución | Resultado observado |
|---|---|
| Nuevos nueve contratos antes del parche | 0 pasan / 9 fallan |
| Primer lote tras el parche | 9 pasan |
| Caso adicional de constancia antes del segundo parche | FALLA en n=49 |
| correlation_numeric_contract final | 10 pasan |
| correlation_open_contracts normal | 4 pasan / 2 ignorados OPEN |
| correlation_open_contracts --ignored | 0 pasan / 2 fallan |
| risk-engine --all-targets final | 178 pasan / 2 ignorados |
| Diagnóstico local genoma seed199 | 0 pasan / 1 falla en el corte ejecutado |
| workspace --all-targets, cargo check | Éxito con advertencias; no ejecuta los tests |

Los 178 incluyen diagnósticos históricos que documentan conductas pendientes.
No equivalen a 178 garantías económicas. Se añaden diez pruebas únicas; activar
tres pruebas existentes no crea tres pruebas nuevas. Los hashes del artefacto
identifican las fuentes de riesgo inspeccionadas, no un commit publicado.

Comandos reproducibles:

    cargo test --offline -p risk-engine --all-targets
    cargo test --offline -p risk-engine --test correlation_open_contracts -- --ignored --nocapture
    cargo test --offline -p quantum-arena --lib xliv_diag_r11_encuentra_el_rechazo_determinista -- --nocapture
    cargo check --offline --workspace --all-targets

No se hizo build de producción, operación, promoción, entrenamiento, reinicio,
merge, commit, push, borrado de rama ni staging por esta sesión. Se usó fetch.
Se consultó al usuario la integración/publicación de trabajo revisado; sigue
pendiente al redactar este corte. No hay justificación para borrar ramas con
trabajo exclusivo ni para publicar juntos bloques ajenos todavía en curso.

## 9. Coordinación y siguiente orden de integración

1. Resolver con el editor del genoma el solape PR #7 / normalizador local;
   verificar el árbol combinado antes de cualquier publicación.
2. Corregir en una ola coordinada los dos contratos rojos y la procedencia
   completa de la matriz, incluyendo el camino None→cero→AllNoise.
3. Separar máscara de datos, inferencia y decisión; probar decisiones finales,
   todos los slots y signos, no solo helpers aislados.
4. Medir latencia y asignaciones con cardinalidad real. La validación añade
   pasadas O(n+m); HY conserva sus vectores temporales, sin benchmark p99 aquí.
5. Publicar únicamente bloques atribuibles, revisados y autorizados. Revalidar
   remoto y ramas al terminar; eliminar solo puntas integradas y no activas.
6. Continuar el inventario de lecturas y contratos de los demás módulos.
   Esta ronda no liquida el trabajo de auditoría integral solicitado.

## 10. Revalidación del genoma tras edición concurrente (10:06 America/Bogota)

Después del fallo descrito en §6, otra sesión añadió normalización de SL también
DESPUÉS de enforce_curve_rr y una restricción por anclas en su paso 2. Cambió
además el fixture de D-658. Codex no aplicó esos cambios; sí volvió a ejecutar
cargo test --offline -p quantum-arena --lib sobre la revisión posterior.

Resultado: **79 pasan / 1 falla**. El diagnóstico de 20.000 semillas ahora pasa,
igual que el test aleatorio test_r11_evolution_pipeline_never_blocked_by_gate.
El fallo vigente observado es d658_mutacion_de_curvas_respeta_las_cotas_declaradas:
assert_eq compara -6.400000000000001 con -6.4 tras reconstruir la curva.

Esto sustituye el estado actual de seed199, no borra su reproducción anterior.
El nuevo fallo exige aclarar idempotencia exacta frente a tolerancia numérica;
no demuestra por sí mismo un impacto económico material. Un no-op explícito
para curvas ya válidas y un contrato de tolerancia razonado son alternativas a
revisar por el editor, no parches aplicados aquí. También hay que conservar un
testigo del fixture antiguo, pues cambiar el dato para que pase no demuestra
ausencia de regresiones en todo el dominio previamente admitido.

SHA256 leído inmediatamente antes de esa revalidación:
- genome.rs: 76633AA574108429BEB8005176246A16C378144E8D40E198193F19BA671DF67F.
- genome_store.rs: 432A2102406EA7DF5906AB37C509F8A21183CC7CCB57E4A3BD40680C0B4DD92C.

Son observaciones de un checkout compartido, no una instantánea atómica del
compilador. No se asigna ese hash al fallo anterior ni se certifica el PR #7.
La corrección debe integrarse con la semántica de banda continua y RR: preservar
el piso en DOS anclas es una condición más fuerte que tener alguna banda
operable y puede restringir curvas que sí eran viables en otras escalas.
El aviso actualizado y el resultado exacto quedan en el buzón entre editores.

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

