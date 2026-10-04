# Revisión exhaustiva desde los fundamentos — plan por archivo y contrato

## 1. Objetivo, frontera y evidencia

Mandato del operador: recorrer todos los archivos, uno por uno, y mejorar
el sistema multiactivo sobre un espectro temporal continuo; preservar lo
existente, explicar los cálculos y verificar comportamiento e integración.
Este plan complementa, NO reemplaza, PLAN_MAESTRO_SINCRONIZACION,
BARRIDO_EXHAUSTIVO_FASES, PLAN_MAESTRO_2026-10-04 y los informes forenses.

Universo reproducible inicial de esta ronda: árbol Git inmutable
23701ecb05486eb13b661d3d50afaadbc9a0015a. Una fila por ruta en
[COBERTURA_MAIN237](audit/COBERTURA_MAIN237_2026-10-04.tsv), con modo,
tipo/OID, clase, fase y owner PROPUESTOS, estado y razón. El
[recibo JSON](audit/COBERTURA_MAIN237_2026-10-04.json) permite cotejar
conjuntos y conteos. Clasificación automática no es lectura ni certificación.
El censo RA12456048 anterior se conserva; son cortes distintos.

Fuera de ese árbol están datos privados/descargados, modelos externos,
archivos ignorados/no versionados, servicios y procesos en ejecución.
Exigen un anexo de inventario por formato/hash/procedencia sin exponer
secretos. No eliminar ni cargar esos recursos para simular cobertura.
Los propios artefactos de cobertura son posteriores al snapshot: evitar
referencia circular; incluirlos como nuevas rutas en el siguiente corte.

La auditoría no puede garantizar encontrar todos los bugs. Su criterio
operacional es cobertura trazable y revalidable de contratos, no omnisciencia.
No declarar éxito financiero a partir de una compilación o un inventario.

## 2. Primero definir lo que se quiere medir

La meta es duplicar equity neta en72h. Para equity positiva, sin aportes ni
retiros en la ventana: G72=E(t+72h)/E(t), g72=ln(G72), objetivo G72>=2,
g72>=ln2. Tasa diaria equivalente2^(1/3)-1≈25,9921%; no es una tasa
demostrada. Con flujos externos, definir retorno ajustado y distinguirlo
de cambio de saldo. Equity, capital disponible y margen no son sinónimos.
Incluir PnL marcado, costes, funding y obligaciones; especificar base y reloj.

Si equity<=0, el logretorno no es finito: registrar insolvencia/indefinición,
no inventar crecimiento al recortar el log. Ventanas superpuestas son
dependientes; no contarlas como ensayos independientes. Registrar el total
de hipótesis y descartes, drawdown, probabilidad de pérdida, capacidad,
exposición, incertidumbre y cobertura. La meta no autoriza aumentar riesgo
sin evidencia, desactivar frenos o seleccionar retrospectivamente ventanas.

La cifra Sharpe≈0,68 citada en un plan anterior depende de aproximaciones
particulares; no usarla como teorema universal de viabilidad. El criterio
principal es crecimiento geométrico neto y supervivencia bajo restricciones,
no WR perfecto, retorno esperado aislado ni score normalizado por10000.

Decisiones a explicitar por el dueño: pérdida/drawdown tolerables, capital
operable, venues/activos, uso de margen y límites de despliegue. Si faltan,
la auditoría sigue; la promoción/operación no queda autorizada por defecto.

## 3. Orden desde los fundamentos, sin alterar la numeración compartida

Una fase tiene dos estados: inventario/revisión y reparación/validación.
F1 «cerrada» en la bitácora de Qoder significa inventario publicado; sus
23 hallazgos pendientes no pasan a resueltos ni validan toda ruta de F1.
La precedencia siguiente se aplica a cada contrato, aunque los equipos
preparen lecturas en paralelo. Fases operativas F3–F7 dependen de la raíz
causal de datos F6: revisar sus contratos de entrada antes de los consumidores.

| Fase/subfase | Pregunta y alcance por archivo | Evidencia mínima antes del siguiente nivel |
| --- | --- | --- |
| F0a: metas | objetivos, balances, unidades de retorno, restricciones y métricas | definición falsable, dato necesario, reloj y criterio de fracaso |
| F0b: conceptos | diccionario de señal, riesgo, evidencia, genoma, escala, nodo y estado | contratos entrada/salida; ausencia diferente de cero; mapeo doc↔consumidor |
| F1a: matemática | curvas/interpolación, álgebra, probabilidades, optimización, límites | dominio/unidades, signo, invariantes, derivación y contraejemplos |
| F1b: estadística | soporte efectivo, nulos, dependencia, selección y calibración | orden prequential, conteo de ensayos, controles nulos y separación causal |
| F2a: física/cuántica | modelos físicos, espectro, supuestos y metáforas | correspondencia de magnitudes; falsación; baseline clásico; no ventaja por nombre |
| F2b: algoritmos | exactitud, convergencia, complejidad y representación | condiciones del algoritmo, fallos extremos, costo y error de aproximación |
| F6 raíz anticipada | reloj/datos/libro/normalización/memoria antes de consumir | esquema, orden, identidad, frescura, pérdida/duplicados y replay determinista |
| F3: núcleo/grafo vivo | root→decisión→terminal, snapshots y sincronización | aristas realmente consumidas, causalidad, generación/as-of/ack y paridad |
| F4: dinero/riesgo/ejecución | todos los vetos, reservas, sizing, fills y conectividad | unidades monetarias, rechazo explicable, límites válidos, evidencia terminal |
| F5: medir/aprender | genomas, modelos, evolución, backtest y promoción | expresión genética, selección finita, IS/validación/test, costes y equivalencia live |
| F6 completo | persistencia, parsers, metacortex e ingestión por ruta | recuperación/crash, atomicidad, configuración, incompatibilidad y linaje |
| F7: plataforma/observabilidad | servicios/UI/scripts/config/docs/logs/binarios y resto | escritor+lector, verdad de paneles, seguridad, recursos, latencia y recuperación |
| F8: integración | cada test/toolchain/CI y composición de todas las capas | candidato exacto, diff por padre, regresión y gates económico-operacionales |

Todos los .rs, scripts, manifiestos, CI, documentación, configuraciones,
frontend, ejemplos y artefactos versionados quedan en alguna fase primaria.
Una ruta puede necesitar contratos secundarios de otras fases. Archivo no
clasificado se revisa manualmente en F7: no se excluye silenciosamente.
No llamar «menos esencial» a autenticación, configuración o una ruta rara:
prioridad final depende de impacto y dependencias, no del tamaño del archivo.

## 4. Concepto continuo sin fingir resolución o información inexistentes

Tiempo de evento, recepción y ejecución son distintos. Guardar nanosegundos
no demuestra eventos observables cada nanosegundo. Repetir datos cacheados
en cada tick no crea muestras nuevas: F1-C2 es ejemplo del riesgo de contarlas.
La evolución de parámetros necesita evidencia causal, no duplicar el reloj.

τ es escala/horizonte con unidad explícita. Frecuencia, periodo y constante
de decaimiento no son intercambiables sin definir el operador. Un banco
finito y curvas sobre lnτ son una representación numérica: explicar rango,
interpolación, soporte, error y adaptación, sin imponer scalp/swing ni
regímenes rígidos. La continuidad conceptual no implica C∞ de toda función:
límites del exchange, estados terminales y decisiones de seguridad pueden
ser discretos. Justificar discontinuidades relevantes; no suavizar un rechazo
de entrada corrupta para satisfacer una estética de continuidad.

1ns–100años puede ser dominio declarado o ensayo de escala; no inventa
granularidad de mercado ni cien años futuros de observación. Extrapolación
debe marcar ausencia de soporte e incertidumbre. Multiasset exige identidad
por activo/par/venue/τ, unidades comparables y reloj admisible de ambas ramas.
IC histórico con mismaτ no equivale a evidencia vigente: MG05 permanece abierto.

## 5. Recibo obligatorio de CADA archivo y contrato

No editar el inventario histórico para borrar estado anterior. Añadir recibos
por (snapshot,ruta,OID,contrato); múltiples contratos pueden compartir archivo.
Plantilla mínima de cada recibo:

1. Ruta/OID, corte, lector/revisor y fecha; lectura completa o intervalo exacto.
2. Finalidad, inputs/outputs, unidades/dominio, invariantes y dependencias.
3. Teoría usada, supuestos, fórmula, por qué se calcula y qué NO significa.
4. Productor→transformaciones→consumidor terminal; clocks/as-of/generaciones.
5. Estado inicial, gatillos, datos inválidos/ausentes y transición de salida.
6. Hallazgo ID/severidad/confianza, evidencia, contraejemplo y radio de impacto.
7. Diferencia entre error demostrado, hipótesis y deuda de diseño/calibración.
8. Prueba RED, extracción o fixture y sus límites; test real vs estructural.
9. Reparación aislada, preservación de invariantes y GREEN del mismo contrato.
10. Comandos/códigos/logs/hashes; warnings/ignoradas y resultados no ocultados.
11. Compilación del candidato; integración por padre; CI del SHA realmente probado.
12. Prueba OOS/económica si procede, límites, review cruzada y estado de despliegue.
13. Rollback: destino compatible de código/esquema/config/estado/modelo,
    disparador, responsable, respaldo verificable y prueba de reversión.
    Revertir código no restaura por sí mismo mmap, reservas o formatos.

Estados separados: inventariado → leído → teoría/flujo contrastado → defecto
reproducido → reparado local → probado → integrado → revalidado/desplegado.
Puede haber «leído sin defecto encontrado», que NO es «libre de fallos».
Un OID cambiado abre revalidación de contratos afectados y consumidores;
no invalida ni reemplaza silenciosamente los recibos históricos. Compilar
el workspace no avanza todas las rutas a leído/probado.

## 6. Pruebas de comportamiento antes de certificar cambios

| Tipo de archivo/contrato | Control mínimo | No confundir con |
| --- | --- | --- |
| fórmula/helper | dominio, signos, extremos, finitos, monotonía/simetría y tests metamórficos | rentabilidad o validez de todos los inputs upstream |
| filtro/veto | límites, motivos, ausencia/invalidez, precedencia y casos de aceptación/rechazo | aumentar número de trades o eliminar controles |
| tiempo/estado | prefijos causales, igualdad de timestamp, reorder, duplicados, resets y as-of | precisión del tipo timestamp o sleep de pared |
| genoma/modelo | mismo hash/config/consumidor, expresión de genes, replay/live, selección y carga | un backtest favorable o archivo serializado existente |
| parser/IO | fuzz/fixtures, truncado, overflow, escapes, formato, errores sin estado parcial | tokenizar ejemplos felices |
| concurrencia | interleavings, snapshots, generaciones, reservas/reintentos y recuperación | DAG de Cargo o absence de merge conflict |
| rendimiento | workload fijo, warmup, p50/p95/p99, backlog, allocations y límites de capacidad | throughput sintético o nanosegundos como unidad |
| docs/config/UI | cotejo con fuente/datos reales, unidades, defaults, contratos y render si aplica | fórmulas correctas sólo en prose |
| datos/binarios | esquema/hash/linaje/corte/formato/compatibilidad y carga autorizada | lectura de bytes, prueba de entrenamiento o permiso de promoción |

RED debe fallar por el comportamiento alegado, no por no compilar o por una
aserción mal planteada. Conservar ese error del test si se descubre, como
en SA. GREEN no se obtiene eliminando casos, relajando el criterio o saltando
la prueba. Si una prueba global consume modelos/datos reales o abre procesos,
se requiere control explícito de efectos; fixtures aisladas primero.

Por reparación: helper aislado + consumidor relevante + checkalltargets
locked del candidato; por fase: regresiones de área, paridad integrada y
tests adversos; por integración económica: T1/oráculo pertinente del tip,
walk-forward y costes/capacidad. T1 de expresividad genética no demuestra
rendimiento. Ignoradas no cuentan como aprobadas. Registrar qué no se ejecutó.

Caché compartida entre worktrees puede dar evidencia de fuente equivocada:
RA§43–44 conserva fallo y reconstrucción. Preferir target aislado cuando
viable; si se comparte, verificar identidad/hash y reconstrucción acotada.
No matar procesos ajenos, borrar caches globales o cambiar perfiles sin recibo.

## 7. Vetos, límites y teoría: clasificación antes de evolución

Inventariar por consumidor TODOS los return/rej/clamp/min/max/timeouts y
umbrales de configuración, no sólo símbolos llamados filtro. Una coincidencia
de texto es candidato, no prueba de arbitrariedad. Para cada uno registrar:
unidad, dominio, finalidad, autoridad, dato, fórmula, parámetros, owner,
frescura, motivo, precedencia, calibración, bypass y efecto terminal.

Separar invariantes de mercado/red/solvencia, protección numérica,
restricciones del dueño, evidencia estadística y heurísticas no calibradas.
Las primeras no se eliminan por una aspiración de aprendizaje continuo.
Las últimas requieren hipótesis, contrafactual shadow y calibración OOS.
Unknown/invalid/stale no son independencia, riesgo0 o confianza0 medida.
Evitar aprender límites con los mismos resultados usados para certificarlos.

Teoría nueva debe aportar magnitud observada, supuestos, implementación,
baseline, coste de cómputo y falsación marginal. Como referencias de raíz:
[Kelly1956](https://www.kiv.zcu.cz/~vavra/zti/KELLY.PDF) para crecimiento
logarítmico y [DSR](https://www.davidhbailey.com/dhbpapers/deflated-sharpe.pdf)
para sesgo de selección/no normalidad. Ninguna repara por sí misma un score
invertido, datos contaminados, ruina o mala ejecución. Los
[problemas del milenio](https://www.claymath.org/millennium-problems/)
no proporcionan un requisito de ingeniería ni un edge por asociación.
Algoritmos clásicos inspirados en física/cuántica se identifican como tales;
hardware/ventaja cuántica requiere evidencia específica, no nomenclatura.

## 8. Coordinación, ramas y composición sin colisiones

| Línea propuesta | Alcance reservado / coordinación | Evidencia/pendiente relevante |
| --- | --- | --- |
| Qoder | F1–F3 productores espectrales, próximo F2 | inventario F1 publicado23; cambios de C2/C4/A1 requieren su ola |
| Claude | F4 ejecución/riesgo, GENOME-GATE | A3/A4/B1/B2; RA-RUIN-F01 reproducido, sin parche paralelo de Codex |
| GLM | F5 medición/modelos y F6 de su línea | L2v1 bloqueado por sus medidas; decisión FDUSD del dueño |
| Codex | contratos raíz, F6/F7 y SA/OOS de F5 aislados | SA22/0 revisado; OOS14/0; RA reconciliada con docs; aún locales/pendingCI |
| Revisor independiente | oráculos/consumidores distintos al autor | no confundir revisión subagente con aprobación humana/CI |

Son propuestas alineadas a los buzones observados, NO nuevas asignaciones
aceptadas. Avisar por ID/ruta/ancla/SHA, pedir acuse y documentar entrega.
Documentos no pueden demostrar que otro editor recibió el aviso. Cada agente
usa rama/worktree propio, jamás cambiar la rama/índice compartidos para
resolver un choque. Reservar anclas y re-grep antes de editar.

Commit atómico por contrato; integrar main vigente en worktree propio;
conservar unión de informes/memoria; diff contra CADA padre; compilar y
probar el candidato; publicar sólo alcance autorizado; CI y review del
SHA exacto; confirmar ancestro/contenido en main remoto. Sólo entonces
borrar referencias inactivas integradas, sin exclusivos y sin checkout ocupado.
Una rama con cero commits exclusivos pero trabajo activo no es basura.

## 9. Ruta crítica inmediata y cierre de una fase

1. Terminar inventario exacto main237 y QA del plan; preservar checklistF1.
2. Cerrar merge documental RA con compilación comprobada; esperar CI74be
   sin extrapolarla ni cancelarla por documentación; nueva CI para nuevo SHA.
3. Mantener SA yOOS separados; composición con base vigente por padre,
   contratos relevantes y review antes de su publicación/integración.
4. Coordinar F1-C2/C4/A1/C1 por owner; ICas-of/MG05 yRA-RUIN-F01 quedan
   abiertos, sin reemplazar ausencia por cero o inventar n efectivo.
5. Avanzar filas por dependencias y riesgo; no por tamaño ni por número de
   hallazgos esperado. Cada fase entrega leídos/pendientes, contratos,
   falsaciones/reparaciones, gates y alcance explícito no examinado.

Cierre de fase exige exactitud de conjunto/rutas, recibos por contrato,
compilación/pruebas pertinentes y revisión de consumidores; no exige afirmar
imposibilidad de nuevos fallos. Cambios futuros disparan revalidación por
OID/arista. Crecimiento72h sólo se declara medido con evaluación neta causal
representativa; nunca se promete consistencia por este plan o por ciencia
avanzada. Sin autorización de trading/promoción/entrenamiento en esta ronda.

## 10. Adenda de revisión: nomenclatura canónica, reversión y anexo externo

Revisor independiente detectó una colisión de definición: maestro§6.1 usa
G72 para incremento LOGARÍTMICO, mientras este plan§2 usa G72 como FACTOR.
Se conservan ambos textos como historia, pero el contrato canónico nuevo es:

- growth_factor_72h (G72_factor) = E(t+72h)/E(t), adimensional, objetivo>=2.
- log_growth_72h (g72_log) = ln(growth_factor_72h), objetivo>=ln2.
- G72 de maestro§6.1 y su distribución§6.2 se interpretan como g72_log,
  NUNCA como G72_factor. G72 de§2 de este plan corresponde a G72_factor.

Duplicación:factor2/log0,693147; capital igual:factor1/log0. No comparar
una métrica con el umbral de la otra. Todo recibo nuevo usa nombres completos,
unidad y transformación; un campo G72 sin linaje/semántica es ambiguo y
BLOQUEA G8 hasta reconciliarlo. Esta corrección documental no demuestra
que código/paneles/datasets ya hayan migrado: se deben rastrear consumidores.

Rollback se exige en el recibo13 de toda reparación, también esquemas y
configuración. El ensayo es aislado, con respaldo/hash y compatibilidad;
despliegue/carga/activación real requieren su autoridad separada.

El anexo de cobertura externa tiene dos gates: C-versionado (todas rutas del
árbol y sus recibos) y C-sistema (también recursos no versionados/runtime).
Owners PROPUESTOS: GLM manifiestos de datos/modelos; Claude órdenes/specs y
estado de ejecución; Qoder productores/features y bancos; Codex concilia
el índice de recursos y evidencias de linaje. Sin acuse no son asignaciones.
Cada recurso requiere ubicación lógica sin secreto, hash/formato/versión,
identidad temporal, productor/cargador, contrato y prueba de compatibilidad.
Recurso ausente, no accesible o sin prueba queda pendiente con impacto/gate,
no aprobado ni omitido. C-sistema no cierra mientras haya recursos críticos
sin inventario/contrato; acceso/publicación/ejecución no autorizados se piden
separadamente. El censo Git por sí solo sólo sirve a C-versionado.

## 11. Recibo del inventario y del método, no de auditoría completa

Inventario main237 creado por subagente Galileo y revalidado por Codex:
1432 rutas únicas,461Rust (405 dentro de crates/56 fuera),23 directorios
de crates y24manifiestosCargo.136Rust bajo tests/ incluye1soporte; no es
conteo de funciones #[test].21rutas requieren triage manual. Todas1432
filas son inventariado_no_certificado;0certificadas por este artefacto.

| Fase primaria sugerida | Rutas |
| --- | ---: |
| F0 |46|
| F1 |29|
| F2 |70|
| F3 |32|
| F4 |39|
| F5 |58|
| F6 |78|
| F7 |930|
| F8 |150|

F7 contiene644JSON de grafo y11otros artefactos de grafo, además de docs/
plataforma. No son930módulos operativos. Cada artefacto también requiere
contrato de esquema/procedencia y cotejo de referencias con su fuente;
parsear un grafo o verificar un OID no certifica conexiones vivas.
Duplicados por OID pueden compartir oráculo de contenido, pero cada ruta
conserva recibo de contexto/consumidor y no se da por revisada en bloque.

TSV SHA25634f5139dfaaad6317924af71584c77a2724123a38918f30c42e2c63247017aae;
JSON SHA256fdd2b412698d9c4d7d8f0ff044fd6d47c047f999aeb5129bea0a7f22dcf8e215.
Metadatos/modos/OIDs/snapshot/orden/conjunto exactos yconteos porfase/clase/
paquete/owner/estado cotejados con ls-tree. Owner sigue propuesto_sin_acuse.
Para detalles de clasificación y denominadores usar el JSON, no sumas de
categorías que se solapan (crate, tipoRust, directoriotests).

Revisión documental independiente Sartre: colisiónG72 cerrada por§10 y
maestro§17; rollback/anexoexterno añadidos;0bloqueadores documentales nuevos.
Consumidores ambiguos y gates económicos siguen pendientes. No ejecutó
fuentes/CLI/Cargo ni revisó el TSV. Esa independencia no es aprobación
humana de PR, ejecución de fase ni acreditación de rentabilidad.
