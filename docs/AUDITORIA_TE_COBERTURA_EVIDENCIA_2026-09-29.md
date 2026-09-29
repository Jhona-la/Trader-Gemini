# Auditoría TE — cobertura, información condicional y evidencia

Fecha: 2026-09-29. Responsable: Codex. Rama: `feat/quant-sr-codex-te`.
Estado: correcciones y documentación LOCALES; sin push, PR nuevo ni merge remoto.

Este informe es aditivo al informe forense, al atlas y a la auditoría ST.
No reemplaza los hallazgos previos, no renumera los 305 puntos históricos
y no certifica que se hayan revisado semánticamente todos los archivos.
No convierte una aspiración de rentabilidad en una garantía financiera.

## 1. Dictamen y alcance verificable

La implementación incorporada en main como `cb1a0604` no calculaba una CMI
de una distribución coherente. Además aceptaba dominios inválidos, confundía
ausencia de cobertura con silencio y reservaba memoria por cada bin vacío.
Sus cuatro pruebas cualitativas originales no detectaban esos problemas.

Corte de esta auditoría: **19 hallazgos; 11 corregidos en código local,
2 contratos documentados y 6 cuestiones abiertas**. No son 19 nuevos fallos
operativos de producción: se incluyen errores, limitaciones del modelo y
carencias de evidencia, diferenciados por estado. Severidad = impacto
potencial si se consume la herramienta, no pérdida financiera observada.

Se preservan las cuatro pruebas originales, la API histórica y la decisión
de GLM de no cablear la feature. Se añade una API con cobertura explícita,
conteo disperso sin coste proporcional al tiempo vacío y 27 contratos nuevos.
La incorporación al genoma, consejo, registry operativo o ejecución NO se realiza.

### 1.1 Cobertura real de revisión

| Elemento | Profundidad en este corte |
|---|---|
| `feature-engine/src/transfer_entropy.rs` | Lectura completa, derivación matemática, reparación, pruebas |
| `feature-engine/tests/transfer_entropy_contract.rs` | Nuevo; revisión completa y oráculo independiente |
| `feature-engine/tests/transfer_entropy_real.rs` | Lectura completa del aporte de GLM; lector y evidencia corregidos |
| `backtest-engine/src/tick_replayer.rs` | Solo contrato de cabecera/layout/alineación, primeras 110 líneas |
| `feature-engine/src/lib.rs`, manifiesto del crate | Exportación/dependencias, sin conexión operativa |
| Triage, coordinación, memoria, atlas, informe forense | Adendas pertinentes; preservación del histórico |
| Símbolos TE en `crates/` y `src/` | Búsqueda transversal de consumidores, no lectura íntegra de todos esos archivos |
| Seis crates + replay unitario | Regresión ejecutada; probar no equivale a auditar cada línea |
| Resto del repositorio | No se certifica en este corte; permanecen auditorías/deudas anteriores |

Inventario durante la integración: 1.358 rutas versionadas y 23 manifiestos
`crates/*/Cargo.toml`. Es un inventario Git, NO un contador de archivos
auditados. La lectura profunda de todo el proyecto sigue siendo trabajo
pendiente y debe tener un registro de cobertura propio.

### 1.2 Cortes de código y conservación

- Main inicial TE: `ca3ea5d4fbfe85228cb8a0faa60cfc85177c2248`.
- Integración local conservando ST: `3ef64475ebd06bf5a75c95bc81e27c785fe656b7`.
- Arreglo del estimador y 20 contratos: `00a118de7cd72fc295dcb0d112ba41316b5b49e9`.
- Aporte posterior de GLM: `40c263c2`, coordinado en main `c0f6510f`.
- Integración de ese aporte: `d98db3e7aa865f282745f8464192f25a3d51e469`.
- Lector y evidencia manual: `d08a840b5e4a1ff667d733c4b357b9c88c37d98e`.
- Último main documental: `817d5882a1870ebe80bfb20dad5b0bf89ef121ad`.
- Integración local de ADRs: `057cb4910b8cea2fb6871d6c272618ba59cf07a5`;
  diff de código/manifiestos vacío frente al corte validado.

Los conflictos fueron adendas al buzón y al triage; se retuvieron ambos lados.
Se revisaron diferencias frente a ambos padres y se compiló con
`--workspace --all-targets` ANTES de cada commit de merge: 42,69 s y 59,91 s.
Estos son merges hacia la rama aislada, NO integración de las correcciones
en el main remoto. ST y TH mantienen sus ramas y commits de conservación.
La tercera integración incorpora exclusivamente los seis ADRs, su índice y
el aviso de GLM; se leyeron completos y se preservaron. El check previo a su
commit pasó en 3,46 s. ADR-0004/0005 repiten afirmaciones limitadas por ST-19
y TE-17: incorporarlos no cierra esas deudas ni certifica sus conclusiones.

## 2. Paradigma de grafo vivo: desde la raíz hasta el terminal

| Nodo / arista | Contrato actual | Estado diagnóstico |
|---|---|---|
| Raíz: eventos y captura | Instrumento, venue, reloj, origen, continuidad | El módulo recibe timestamps; cobertura autenticada pendiente |
| Normalización temporal | Cobertura común [inicio,fin), bins completos | API explícita; wrapper conserva aproximación por extremos |
| Nodo de conteo | N[destino futuro][destino pasado][fuente pasada] | u64, disperso, direcciones separadas |
| Nodo probabilístico | Una sola conjunta suavizada y sus márgenes | Corregido; no mezcla priors de tablas distintas |
| Nodo estadístico | CMI en bits/transición + soporte | Escalar descriptivo, sin significancia acreditada |
| Nodo de decisión | Condicionamiento, selección, coste, OOS | No implementado para esta feature; no inventar veto |
| Nodo genoma/terminal | Feature versionada → gen → orden → fill → PnL | Arista no conectada; no atribuir impacto a TE |
| Bucle evolutivo | Resultado fuera de muestra → actualización/rollback | Debe validarse por separado; fuera del arreglo TE |

Los ceros observados y los datos ausentes son estados diferentes de la raíz.
Si se confunden, ningún cálculo posterior rescata la interpretación.
La incertidumbre debe viajar por las aristas; un escalar finito no la sustituye.

## 3. Qué calcula cada expresión y qué NO significa

Definimos A=Y[t+1], B=Y[t], C=X[t]. Cada símbolo solo indica si hubo
al menos un evento en un bin de Δ milisegundos. Nabc cuenta transiciones,
M=Nventanas−1. Con α=1/2:

```text
qabc = (Nabc + α) / (M + 8α)
qab  = suma_c qabc
qbc  = suma_a qabc
qb   = suma_a,c qabc

T(X→Y) = suma_a,b,c qabc · log2(qabc · qb / (qab · qbc))
       = H(A,B) + H(B,C) − H(B) − H(A,B,C)
```

La primera fórmula pondera cuánto cambia la distribución de Y futuro al
conocer X pasado, además de Y pasado. La segunda es el oráculo independiente
usado en tests. Ambas deben coincidir. α evita logaritmos de cero, pero añade
información previa: no crea evidencia empírica. Esta CMI de la ley suavizada
no es la esperanza posterior de la CMI.

En aritmética real 0≤T≤H(A|B)≤1 bit, porque A es binaria. La implementación
solo admite una tolerancia de redondeo de 64·EPSILON fuera de [0,1]; un
desvío mayor no se oculta con un clamp. Ese margen es numérico, no un filtro
de mercado ni una cota de error estadístico.

T mide bits por transición de la rejilla elegida. Dividir por Δ para presentar
una tasa descriptiva no convierte el cálculo en el límite de un proceso
continuo ni permite comparar evidencia sin controlar discretización y soporte.
La memoria retenida es de un paso. No se estima «todo el espectro» a la vez.

### 3.1 Prueba de exactitud del conteo disperso

Toda transición no adyacente a un bin activo tiene tupla 000. Se inicializan
M transiciones de ese tipo. Un bin activo b solo puede afectar t=b y t=b−1.
Cuatro iteradores ordenados (dos por stream) enumeran esos índices, deduplican
las coincidencias y reemplazan una transición silenciosa por la observada en
ambas direcciones. Búsquedas binarias determinan la actividad de t y t+1.

No se materializa la rejilla ni se recorre el vacío temporal. Complejidad:
O(E log(E+1)) en tiempo y O(1) en memoria auxiliar, sin contar las entradas.
Las tablas se conservan en u64; f64 se usa únicamente al estimar probabilidades.
Más allá de 2^53, la conversión a f64 no distingue todos los enteros.

La comprobación incluye 3.840 rejillas contra tablas densas, anchuras 1/7/200,
diversas ocupaciones y longitudes, duplicados, jitter y epochs >2^53.
El caso centenario prueba conteos/recursos; NO observa un siglo de mercado.

## 4. Matriz maestra de este corte

| ID | Severidad | Estado | Hallazgo |
|---|---|---|---|
| TE-01 | alta | corregido_local | La supuesta conjunta no conservaba masa probabilística |
| TE-02 | alta | corregido_local | Los condicionales usaban priors incompatibles con la conjunta |
| TE-03 | alta | corregido_local | Símbolos fuera del alfabeto causaban panic |
| TE-04 | alta | corregido_local | Truncamiento silencioso simulaba alineación de dos series |
| TE-05 | alta | corregido_local | Retrocesos del reloj se plegaban a cero o se descartaban |
| TE-06 | alta | corregido_local | La unión de extremos fabricaba observaciones fuera del solape |
| TE-07 | media | corregido_local | Se contaba como completa una ventana terminal parcialmente observada |
| TE-08 | alta | corregido_local | Overflow y conversiones del número de ventanas corrompían el soporte |
| TE-09 | alta | corregido_local | El coste dependía de todos los bins vacíos del intervalo |
| TE-10 | media | contrato_documentado | La descripción «sin lag» contradice el operador implementado |
| TE-11 | media | contrato_documentado | 64 ventanas se presentaban como frontera matemática de estimabilidad |
| TE-12 | alta | abierto | Regularización positiva no constituye evidencia de transferencia |
| TE-13 | alta | abierto | La cobertura declarada aún no dispone de certificación de feed |
| TE-14 | alta | abierto | TE bivariada no identifica causalidad económica multiactivo |
| TE-15 | media | abierto | No existe transferencia demostrada de esta feature al genoma o al motor |
| TE-16 | media | abierto | La representación actual no cubre un continuo temporal universal |
| TE-17 | alta | abierto | El «resultado negativo» histórico no tiene un contraste calibrado |
| TE-18 | media | corregido_local | El lector diagnóstico aceptaba tapes truncados, vacíos o desordenados |
| TE-19 | alta | corregido_local | La prueba manual podía pasar sin medir ningún par |

Los IDs son específicos de TE y no sustituyen ST/TH/CF ni el censo global.
`corregido_local` no significa publicado, desplegado ni económicamente validado.
`contrato_documentado` corrige la descripción; no acredita una capacidad nueva.

## 5. Fichas profesionales de hallazgo

### TE-01 — La supuesta conjunta no conservaba masa probabilística

- Severidad: alta. Estado: `corregido_local`.
- Ancla reproducible: `transfer_entropy.rs:te_binaria_kt (cb1a0604)`.

**Causa y mecanismo.** Cada una de las ocho celdas recibía 2·KT=1, pero el denominador solo añadía 4. Para M=n−1 transiciones, la suma era (M+8)/(M+4), no 1. Con n=64 resulta 71/67≈1,05970. No es un detalle de tolerancia: se ponderaba el log-cociente con una medida no normalizada.

**Consecuencia y alcance.** Los valores perdían la interpretación de información mutua condicional y sus cotas. Una prueba cualitativa de dirección podía seguir pasando, ocultando el defecto. No se atribuye pérdida operativa: no se encontró consumidor de ejecución.

**Evidencia/reproducción.** RED: n=64 produjo 0,38625361467532393 bits frente al oráculo por cuatro entropías, 0,4091039241579857. Test cmi_matches_one_normalized_distribution.

**Corrección o trabajo requerido.** q=(N+1/2)/(M+4), con ocho pseudoconteos de 1/2; conteos enteros y cálculo final f64. Oráculo independiente H(A,B)+H(B,C)−H(B)−H(A,B,C).

**Criterio de cierre y límite residual.** Oráculo y equivalencia disperso/denso en 3.840 rejillas; pendiente publicación, no pendiente arreglo local.

### TE-02 — Los condicionales usaban priors incompatibles con la conjunta

- Severidad: alta. Estado: `corregido_local`.
- Ancla reproducible: `transfer_entropy.rs:cmi`.

**Causa y mecanismo.** La versión original suavizaba por separado p(a|b,c) y p(a|b) con add-1/2. Marginalizar la nueva conjunta de ocho celdas exige sumar dos pseudoconteos por resultado en p(a|b), no reiniciar un Bernoulli add-1/2. Arreglar únicamente el denominador de TE-01 no cerraría este defecto.

**Consecuencia y alcance.** El cálculo mezclaba leyes distintas. Recortar resultados negativos a cero no demuestra no negatividad del estimador ni restaura una identidad de CMI.

**Evidencia/reproducción.** Derivación algebraica: q(a|b,c)=(Nabc+1/2)/(Nbc+1); q(a|b)=(Nab+1)/(Nb+2). El oráculo por entropías exige consistencia de todos los márgenes.

**Corrección o trabajo requerido.** Derivar márgenes de una sola tabla suavizada. El nombre te_binaria_kt se conserva, pero se explica que es CMI plug-in de la media conjunta Dirichlet(1/2), no esperanza posterior de CMI ni un contraste KT secuencial.

**Criterio de cierre y límite residual.** Tests de referencia y complementos binarios pasan. Solo se recorta un residuo numérico de 64·EPSILON; violaciones sustanciales devuelven fallo.

### TE-03 — Símbolos fuera del alfabeto causaban panic

- Severidad: alta. Estado: `corregido_local`.
- Ancla reproducible: `transfer_entropy.rs:te_binaria_kt`.

**Causa y mecanismo.** Los u8 se convertían directamente en índices de longitud dos. El tipo aceptaba 2…255 aunque la fórmula solo admite {0,1}. El último símbolo de la fuente no se usaba y podía quedar sin validar.

**Consecuencia y alcance.** Datos mal codificados podían abortar la evaluación; diferentes posiciones del mismo dato inválido recibían trato distinto. Es un contrato de dominio, no un veto económico.

**Evidencia/reproducción.** RED: x[0]=2 causa index out of bounds. La regresión recorre valores 2 y 255 y posiciones 0,31,63, en ambas orientaciones.

**Corrección o trabajo requerido.** Validar todo el alfabeto de ambas series antes de indexar, devolviendo None sin truncar ni reinterpretar categorías.

**Criterio de cierre y límite residual.** malformed_alphabet_is_rejected_without_panicking pasa; no se ampliaron categorías del modelo.

### TE-04 — Truncamiento silencioso simulaba alineación de dos series

- Severidad: alta. Estado: `corregido_local`.
- Ancla reproducible: `transfer_entropy.rs:te_binaria_kt`.

**Causa y mecanismo.** min(x.len(),y.len()) ocultaba diferencias de longitud. Coincidir en un prefijo no prueba que ambos vectores compartan origen, cadencia ni máscara de observación.

**Consecuencia y alcance.** Una desalineación aguas arriba podía producir un escalar aparentemente válido. El algoritmo no dispone de timestamps en esta API para reconstruir la alineación.

**Evidencia/reproducción.** RED: longitudes 64/65 devuelven Some(0,3862536…), aunque el contrato de series alineadas está incumplido.

**Corrección o trabajo requerido.** Exigir igual longitud, además de declarar que la alineación semántica sigue siendo responsabilidad del llamador.

**Criterio de cierre y límite residual.** mismatched_series_are_not_silently_truncated pasa en ambas orientaciones; no se inventa un emparejamiento temporal.

### TE-05 — Retrocesos del reloj se plegaban a cero o se descartaban

- Severidad: alta. Estado: `corregido_local`.
- Ancla reproducible: `transfer_entropy.rs:sorted y transfer_entropy_observada`.

**Causa y mecanismo.** El intervalo se calculaba con primer/último elemento sin validar orden. saturating_sub plegaba timestamps anteriores al origen; eventos posteriores al último aparente podían quedar fuera del vector.

**Consecuencia y alcance.** Se alteraban actividad y dirección temporal sin diagnóstico. Ordenar silenciosamente sería también incorrecto si ocultara corrupción del feed.

**Evidencia/reproducción.** RED: [0,40,20,100] frente a [0,50,100] se acepta y produce valores finitos.

**Corrección o trabajo requerido.** Validación no decreciente de ambos streams antes del conteo. Duplicados permitidos porque actividad binaria colapsa eventos simultáneos. Error tipado UnsortedTimestamps.

**Criterio de cierre y límite residual.** Regresiones del wrapper, errores tipados, duplicados y epochs grandes verificadas.

### TE-06 — La unión de extremos fabricaba observaciones fuera del solape

- Severidad: alta. Estado: `corregido_local`.
- Ancla reproducible: `transfer_entropy.rs:transfer_entropy_eventos y ObservationWindow`.

**Causa y mecanismo.** Se usaban min(primeros) y max(últimos), aunque la documentación prometía None sin solape. Los espacios externos se rellenaban con cero como si el feed se hubiera observado.

**Consecuencia y alcance.** Ausencia de cobertura se confundía con silencio de mercado. El sesgo puede modificar ambas direcciones de manera distinta.

**Evidencia/reproducción.** RED: [0,100] y [200,300] devuelve aproximadamente (0,01312;0,01866) bits en vez de ausencia de estimación.

**Corrección o trabajo requerido.** Wrapper histórico con intersección conservadora de extremos. Nueva API con intervalo común declarado [inicio,fin), capaz de representar streams vacíos observados y eventos de extensiones disjuntas dentro de cobertura real.

**Criterio de cierre y límite residual.** Tests distinguen explícitamente silencio observado, falta de cobertura inferible y ausencia de solape. La certificación externa de cobertura permanece abierta en TE-13.

### TE-07 — Se contaba como completa una ventana terminal parcialmente observada

- Severidad: media. Estado: `corregido_local`.
- Ancla reproducible: `transfer_entropy.rs:transfer_entropy_observada`.

**Causa y mecanismo.** span/anchura+1 incluía el bin del último evento aunque solo estuviera observado su comienzo. Esto podía satisfacer artificialmente MIN_VENTANAS.

**Consecuencia y alcance.** Se incrementaba el soporte nominal y se introducían ceros no observados al final. El efecto es mayor en registros cortos.

**Evidencia/reproducción.** RED: extremos [0,639] con anchura 10 daban Some; solo hay 63 bins completos, no 64.

**Corrección o trabajo requerido.** N=floor((fin−inicio)/Δ); fin exclusivo. Solo N bins y N−1 transiciones, con discarded_tail_ms explícito.

**Criterio de cierre y límite residual.** Pruebas de frontera, registros externos al intervalo y cola parcial contrastadas contra un oráculo denso.

### TE-08 — Overflow y conversiones del número de ventanas corrompían el soporte

- Severidad: alta. Estado: `corregido_local`.
- Ancla reproducible: `transfer_entropy.rs:transfer_entropy_observada`.

**Causa y mecanismo.** span/anchura+1 puede exceder u64, y el cast a usize añade otra dependencia de plataforma. Con [0,u64::MAX], anchura 1, el ejecutable usado devolvió None tras el cálculo envuelto.

**Consecuencia y alcance.** Un intervalo enorme podía aparecer vacío o fallar antes de estimar. No se afirma que se observara un panic de overflow en esta configuración: el RED observado fue None y el unwrap del test falló.

**Evidencia/reproducción.** maximum_timestamp_does_not_overflow_bin_count falló antes del arreglo. Casos adicionales prueban origen cercano a u64::MAX y anchura próxima a u64::MAX/2.

**Corrección o trabajo requerido.** Restar el intervalo con checked_sub, usar extremo exclusivo y conteos u64; eliminar +1 y la asignación indexada por usize para bins temporales.

**Criterio de cierre y límite residual.** Casos máximos y épocas >2^53 pasan. La etapa f64 no promete precisión unitaria por encima de 2^53.

### TE-09 — El coste dependía de todos los bins vacíos del intervalo

- Severidad: alta. Estado: `corregido_local`.
- Ancla reproducible: `transfer_entropy.rs:affected_transitions y active`.

**Causa y mecanismo.** Dos Vec<u8> de N bins exigían memoria y recorrido O(N), aunque solo hubiera dos eventos. Para 100 años a 1 ms, N≈3,15576·10^12: unos 6,31 TB para ambos indicadores, sin contar overhead.

**Consecuencia y alcance.** Riesgo de agotamiento de memoria/latencia por reloj corrupto, separación larga o exploración de escalas. No se provocó deliberadamente una asignación masiva para demostrarlo.

**Evidencia/reproducción.** Análisis de las dos reservas originales; century_silence_uses_exact_counts_without_dense_allocation y equivalencia exacta de tablas sobre 3.840 rejillas.

**Corrección o trabajo requerido.** Inicializar M transiciones 000; visitar solo t=b y t=b−1 alrededor de bins activos. Cuatro iteradores ordenados, deduplicación y búsquedas binarias: O(E log(E+1)) tiempo, O(1) memoria auxiliar.

**Criterio de cierre y límite residual.** Conteos u64 exactos del intervalo centenario y del extremo u64; falta benchmark productivo de latencia, no se declara aptitud HFT.

### TE-10 — La descripción «sin lag» contradice el operador implementado

- Severidad: media. Estado: `contrato_documentado`.
- Ancla reproducible: `transfer_entropy.rs:documentación; docs/TRIAGE_TEORICO_2026-09-29.md`.

**Causa y mecanismo.** El estadístico usa X[t], Y[t] y Y[t+1]. Con bins de Δ ms hay historia de un paso y horizonte de predicción Δ; el default de 200 ms no elimina esa elección.

**Consecuencia y alcance.** Podía interpretarse erróneamente como estimador universal de todos los horizontes y memorias, o como superioridad automática sobre Hawkes.

**Evidencia/reproducción.** Definición del código y Schreiber (2000), ecuación 4 con historias de longitudes k,l. El formalismo continuo consultado también condiciona en historias.

**Corrección o trabajo requerido.** Corregir el módulo y añadir corrigendo sin borrar los textos históricos de GLM. Separar estimador discretizado de futuro candidato de procesos puntuales en tiempo continuo.

**Criterio de cierre y límite residual.** Contrato documental corregido localmente; una familia adaptativa de escalas sigue pendiente, TE-16.

### TE-11 — 64 ventanas se presentaban como frontera matemática de estimabilidad

- Severidad: media. Estado: `contrato_documentado`.
- Ancla reproducible: `transfer_entropy.rs:MIN_VENTANAS y ObservationWindow.min_windows`.

**Causa y mecanismo.** Un conteo mínimo fijo no mide n efectivo ni ocupación de las ocho celdas. El suavizado permite computar una cifra con menos datos; eso tampoco la vuelve significativa.

**Consecuencia y alcance.** Rigidez opaca y confusión entre validez algebraica, política de soporte y evidencia estadística. Reducir el mínimo no resuelve la inferencia.

**Evidencia/reproducción.** La API explícita estima una transición y coincide con el oráculo; el caso de 100 bins silenciosos sigue dando un valor positivo debido al prior.

**Corrección o trabajo requerido.** Conservar 64 por compatibilidad, etiquetarlo como política y permitir min_windows explícito >=2. Exponer conteos/transiciones para diagnósticos sin fabricar p-values.

**Criterio de cierre y límite residual.** Configuración inválida produce error diferenciado; significancia y calibración permanecen abiertas en TE-12.

### TE-12 — Regularización positiva no constituye evidencia de transferencia

- Severidad: alta. Estado: `abierto`.
- Ancla reproducible: `transfer_entropy_contract.rs:explicit_observation_distinguishes_silence_from_missing_coverage`.

**Causa y mecanismo.** La masa del prior en celdas no observadas puede crear dependencia condicional en la ley suavizada. Esta es una limitación del estimador, no necesariamente un bug de normalización restante.

**Consecuencia y alcance.** Usar TE>0, una razón entre direcciones o un umbral universal como veto puede producir falsos liderazgos. Los bins vacíos repetidos no son eventos independientes de información financiera.

**Evidencia/reproducción.** Tras corregir la fórmula, dos streams vacíos con cobertura declarada de 100 bins producen 0,02477933540853777 bits. La prueba lo registra explícitamente.

**Corrección o trabajo requerido.** Diseñar nulos y surrogates que preserven dependencia/marginales pertinentes, calibrar tamaño y potencia, reportar sensibilidad al prior, ocupación y tamaño efectivo. Corregir multiplicidad antes de explorar pares, escalas y símbolos.

**Criterio de cierre y límite residual.** Exigir falsos positivos controlados bajo modelos nulos relevantes y potencia bajo alternativas predefinidas en datos separados. No se implementa un umbral arbitrario de sustitución.

### TE-13 — La cobertura declarada aún no dispone de certificación de feed

- Severidad: alta. Estado: `abierto`.
- Ancla reproducible: `transfer_entropy.rs:ObservationWindow`.

**Causa y mecanismo.** Un intervalo start/end no contiene huecos de captura, sincronización entre exchanges, origen del reloj, correcciones o incertidumbre de timestamps. El wrapper solo infiere una intersección de extremos.

**Consecuencia y alcance.** Un outage puede continuar interpretándose como silencio si el caller declara incorrectamente observación continua. La nueva API hace el supuesto visible, no lo verifica por magia.

**Evidencia/reproducción.** Los tipos de entrada solo incluyen timestamps y una ventana. No hay máscara de disponibilidad ni registro de gaps en el contrato local.

**Corrección o trabajo requerido.** Aportar lineage de captura por activo/venue/reloj, segmentos contiguos válidos y versión de datos. No contar transiciones atravesando gaps. Distinguir event-time, receive-time y tiempo de decisión.

**Criterio de cierre y límite residual.** Pruebas con outage conocido, relojes desfasados y segmentos separados; salida invalidada o segmentada con razón trazable, y paridad replay/serving.

### TE-14 — TE bivariada no identifica causalidad económica multiactivo

- Severidad: alta. Estado: `abierto`.
- Ancla reproducible: `transfer_entropy.rs:te_binaria_kt`.

**Causa y mecanismo.** La conjunta solo tiene tres variables binarias. No condiciona por mercado común, otros activos, calendario, liquidez ni historial más largo.

**Consecuencia y alcance.** Un driver común, latencias distintas o dinámica omitida pueden explicar la asimetría. Actividad predictiva no implica retorno neto predecible ni intervención causal.

**Evidencia/reproducción.** Ocho celdas y ausencia de variables Z; no hay control multivariante en el estimador. Schreiber contempla condicionar por un driver común conocido.

**Corrección o trabajo requerido.** Definir grafo y conjunto de condicionamiento identificable, baselines y controles de latencia. Evaluar capacidad, costes, fills y ablación fuera de muestra antes de inferir utilidad de trading.

**Criterio de cierre y límite residual.** Evidencia multivariante preespecificada y estabilidad fuera de muestra; no declarar causalidad solo por orientación de un arco.

### TE-15 — No existe transferencia demostrada de esta feature al genoma o al motor

- Severidad: media. Estado: `abierto`.
- Ancla reproducible: `crates/src: búsqueda de transfer_entropy y te_binaria_kt`.

**Causa y mecanismo.** Los usos encontrados son la exportación del módulo, funciones internas y tests. La nueva prueba real es consumidor diagnóstico, no un consumidor operativo.

**Consecuencia y alcance.** El módulo no puede explicar por sí mismo mejoras o discrepancias backtest/demo/producción. Conectarlo prematuramente introduciría una nueva política sin validación.

**Evidencia/reproducción.** Búsqueda transversal de símbolos tras integrar main c0f6510f; decisión de GLM de no cablear conservada. No se modifica ningún gen, veto de riesgo ni capa de ejecución.

**Corrección o trabajo requerido.** Si supera validación: definir payload versionado activo/par/τ/as-of, lineage hasta features/genes/decisión, comparación shadow y paridad de transformaciones en replay y serving.

**Criterio de cierre y límite residual.** Trazas de extremo a extremo y ablación con costes reales. Hasta entonces, estado experimental aislado.

### TE-16 — La representación actual no cubre un continuo temporal universal

- Severidad: media. Estado: `abierto`.
- Ancla reproducible: `transfer_entropy.rs:ObservationWindow.bin_width_ms`.

**Causa y mecanismo.** El contrato usa u64 en milisegundos, bins homogéneos, actividad binaria y orden uno. No representa una observación de nanosegundos ni memoria de cien años. Un conteo de bins enormes prueba escalabilidad, no existencia de datos.

**Consecuencia y alcance.** Etiquetarlo como espectro completo ocultaría límites de resolución, pérdida de marcas/conteos y aliasing. En activos muy activos la simbolización puede saturarse.

**Evidencia/reproducción.** Unidades y variables del API; timestamps iguales conservan actividad, no reconstruyen tiempos sub-ms. No hay selección adaptativa de escalas ni modelo de memoria.

**Corrección o trabajo requerido.** Tratar escala, memoria, resolución, incertidumbre y coste como parámetros verificables. Candidato: TE de procesos puntuales en tiempo continuo con intensidades dependientes de historia; compararlo con el baseline discreto y alternativas de símbolos marcados.

**Criterio de cierre y límite residual.** Datos con resolución acreditada, estudio de convergencia/sensibilidad, control de selección y coste. No implementar teorías del milenio o cuánticas sin variable observable, operador y falsación.

### TE-17 — El «resultado negativo» histórico no tiene un contraste calibrado

- Severidad: alta. Estado: `abierto`.
- Ancla reproducible: `docs/TRIAGE_TEORICO_2026-09-29.md:ADENDA XLVIII·E`.

**Causa y mecanismo.** Se compara ~0,001 bits en tapes con <0,01 de una simulación y >0,15 de un acople fuerte. Son referencias descriptivas, no distribución nula, intervalo de confianza ni potencia de la prueba. Además se usó la fórmula anterior.

**Consecuencia y alcance.** «No detectamos evidencia validada» no equivale a «el fenómeno no existe». Tampoco puede asumirse un piso universal del sesgo KT. La decisión prudente de no cablear es compatible con reconocer esta incertidumbre.

**Evidencia/reproducción.** Texto y tabla históricos del commit 40c263c2; no hay surrogates, significancia o potencia calculados en la prueba real. No se han recalculado esos tapes en este corte.

**Corrección o trabajo requerido.** Conservar tabla con versión del estimador; añadir esta salvedad. Repetir medición corregida con hashes/intervalos/cobertura y protocolo nulo/OOS, sin reutilizar la misma muestra para elegir símbolo y certificar rendimiento.

**Criterio de cierre y límite residual.** Reporte reproducible de efecto, incertidumbre, potencia y pruebas múltiples. No se sustituye la tabla histórica por números inventados.

### TE-18 — El lector diagnóstico aceptaba tapes truncados, vacíos o desordenados

- Severidad: media. Estado: `corregido_local`.
- Ancla reproducible: `transfer_entropy_real.rs:timestamps_de_bytes`.

**Causa y mecanismo.** La división entera de bytes entre 40 descartaba el resto. Una cabecera sola producía Some([]), y no se comprobaba orden. El comentario declaraba cinco f64 aunque el timestamp se decodificaba correctamente como u64.

**Consecuencia y alcance.** La medición podía usar evidencia incompleta o no observable sin reportar corrupción. La cabecera no acredita autenticidad ni calidad completa del contenido.

**Evidencia/reproducción.** RED del lector: 2 pasan/3 fallan. Un byte extra conservaba Some([10,20]); cabecera sola Some([]); [20,10,30] se aceptaba.

**Corrección o trabajo requerido.** Validar payload no vacío y múltiplo exacto de 40, y orden no decreciente. Extraer parser puro para fixtures; corregir descripción del layout sin alterar los cuatro campos ignorados.

**Criterio de cierre y límite residual.** Siete contratos finales del lector/medición pasan; no valida precios, continuidad ni procedencia autenticada, que quedan explícitamente fuera del lector.

### TE-19 — La prueba manual podía pasar sin medir ningún par

- Severidad: alta. Estado: `corregido_local`.
- Ancla reproducible: `transfer_entropy_real.rs:medir_par_obligatorio`.

**Causa y mecanismo.** Faltaban tapes → continue; estimador None → println. Si todas las iteraciones quedaban en esos casos, el harness terminaba verde sin ninguna estimación válida.

**Consecuencia y alcance.** Un resultado de infraestructura podía confundirse con evidencia cuantitativa. Que un test sea manual/ignored no justifica éxito cuando se solicita pero no tiene datos.

**Evidencia/reproducción.** Control de flujo del commit 40c263c2; contratos nuevos ejercitan ausencia de cada lado, ausencia de ambos y soporte insuficiente mediante catch_unwind.

**Corrección o trabajo requerido.** Exigir ambos tapes y ambas estimaciones cuando se ejecuta manualmente; fallo explícito ante ausencia/corrupción/soporte insuficiente. Se conserva ignored por requerir archivos externos. La razón entre direcciones sin denominador positivo se muestra como no definida.

**Criterio de cierre y límite residual.** Tests de invalidez y par válido pasan. La prueba con tapes reales permanece ignorada y no cuenta como validación económica.

## 6. Validación ejecutada y resultados no ejecutados

El corte ejecutable final es `d08a840b`; después solo se añaden documentos.

| Validación | Aprobadas | Fallidas | Ignoradas | Alcance |
|---|---:|---:|---:|---|
| RED del estimador original | 1 | 7 | 0 | Ocho regresiones, antes del arreglo |
| RED del lector original | 2 | 3 | 0 | Cinco contratos; extracción pura sin cambio semántico |
| Contratos nuevos del estimador | 20 | 0 | 0 | Incluye 3.840 rejillas dentro de un test |
| Contratos nuevos del lector/medición | 7 | 0 | 1 | La ignorada es la medición real preexistente |
| Cuatro simulaciones originales de TE | 4 | 0 | 0 | Conservadas, sin bajar tolerancias |
| Seis crates, todos los targets, corte final | 1.133 | 0 | 7 | 89 bloques; incluye las pruebas TE anteriores |
| Backtest-engine, lib | 37 | 0 | 0 | Separado de los seis crates |
| Total disjunto final | **1.170** | **0** | **7** | No sumar otra vez los subconjuntos TE |

Las siete ignoradas son cinco pruebas de testnet, una de inventario de modelos
y una medición TE con tapes externos. No son siete aprobaciones implícitas.
`cargo check --workspace --all-targets`: exit 0, 4,03 s en el corte de código;
exit 0, 3,46 s después del último merge exclusivamente documental.
Hay warnings preexistentes fuera del alcance; no se eliminan mediante cambios
de flags, supresiones globales o debilitamiento de tests.

Comandos reproducibles:

```powershell
cargo test -p feature-engine --test transfer_entropy_contract
cargo test -p feature-engine --test transfer_entropy_real
cargo test -p feature-engine --lib transfer_entropy
cargo test -p feature-engine -p signal-engine -p risk-engine -p god-engine-core -p execution-engine -p evolution-engine --all-targets
cargo test -p backtest-engine --lib
cargo check --workspace --all-targets
git diff --check
```

No se ejecutan el test manual `--ignored`, T-1, demo, producción, entrenamiento
o promoción de modelos. No se cambia golden, kill-switch, política de capital,
límites de exchange ni criterios de aceptación de una orden.
El total de tests no certifica cobertura matemática exhaustiva ni beneficios.

### 6.1 Huellas de archivos del corte

SHA-256 de bytes locales (pueden diferir de un blob Git por CRLF):

| Archivo | SHA-256 |
|---|---|
| Módulo TE original | `AEB1BC0345646E7CEAFECAAE1C19D9AFCD650F2BEB0315A2033D80EFDBCF8DF6` |
| Módulo TE corregido | `C420D26F0A507AECA099B092B4C02637C3761816268BCC579B5C37384212AC06` |
| Contratos TE nuevos | `50BA9122E884B83F029574323689039722BAA59AC3C16D7CE49FD2D141C08482` |
| Lector/medición corregidos | `1B1031CE3391DACC8021F163863A7053F8E983C43527A20586F9707219F8FB89` |

El [artefacto JSON](artifacts/auditoria_te_cobertura_evidencia_2026-09-29.json)
contiene estados, fichas, commits, fuentes, pruebas y estas huellas.
No se incluyen tapes ni documentos científicos completos en los commits.

## 7. Registro de vetos, filtros y rechazos de este alcance

| Control | Justificación | ¿Es una política económica? | Comportamiento |
|---|---|---|---|
| Símbolos {0,1} | Dominio de la tabla binaria | No | Rechazar dato inválido, no indexarlo |
| Longitud idéntica | Contrato mínimo de alineación | No | None; no truncamiento silencioso |
| Timestamps no decrecientes | Causalidad/orden de índices | No | Error tipado; duplicados admitidos |
| Δ>0, fin>inicio | Definición de división/intervalo | No | Error específico |
| N≥min_windows≥2 | Al menos una transición; soporte declarado | No; 64 sigue siendo política heredada | Configurable en API explícita, no significancia |
| Bins completos | No fabricar cola observada | No | Devolver resto temporal descartado |
| T∈[0,1] con tolerancia numérica | Cota algebraica de CMI binaria | No | Fallo si viola sustancialmente la cota |
| Tape íntegro/ordenado | Integridad del experimento | No | No aceptar restos/cabecera vacía |
| Evidencia obligatoria en test manual | Evitar aprobación vacía | No | Fallar si no puede medir ambos lados |
| No cablear TE al motor | Falta de evidencia OOS y operativa | Gobernanza experimental | Se conserva; no se agrega un veto de trading |

El riesgo-duro no se elimina por ser un límite. La deuda ST-19 del registro
de vetos permanece: un nombre de test como string no demuestra que exista
ni que pruebe comportamiento FP/FN. El registro operativo no se modifica
en esta ola, porque no se añade/retira/cambia ningún veto de trading.

## 8. Hoja de ruta de rehabilitación y transferencia científica

Orden de dependencia, no catálogo de teorías por prestigio:

1. **Raíz/lineage:** certificar calendario de observación, unidades, calidad,
   causalidad temporal y gaps por activo. Sin ello no comparar tasas.
2. **Estimación:** fijar alfabeto, memoria, escala y prior con trazabilidad;
   medir ocupación, sensibilidad, resolución efectiva y coste.
3. **Inferencia estadística:** construir nulos relevantes y medir potencia.
   Preespecificar cómo se exploran pares/escalas para evitar selección optimista.
4. **Multiactivo:** estudiar condicionamiento por factores/activos observables,
   evitando confundir difusión común con un vínculo causal directo.
5. **Replay/serving:** idéntica feature versionada y as-of; validación shadow,
   retrasos, colas, costes, fills, capacidad y rollbacks medidos.
6. **Adopción evolutiva:** solo después, demostrar mediante ablación que
   una feature modifica útilmente el genoma y sobrevive fuera de muestra.

### 8.1 Fuentes primarias verificadas y familia candidata

- [Schreiber, Measuring Information Transfer](https://arxiv.org/abs/nlin/0001042):
  se verificó la definición con historias de longitudes k,l y el uso de
  condicionales. También permite considerar un driver común conocido.
  La fuente fundamenta el operador, no valida este código ni aporta edge.
- [Spinney, Prokopenko y Lizier, Transfer entropy in continuous time](https://arxiv.org/abs/1610.08192):
  pasajes del cuerpo verificados. Su formulación usa medidas de trayectorias
  e historias de duración s,r; para procesos de saltos/puntuales distingue
  contribuciones de eventos y espera. Ofrece una vía diferente al binning,
  pero no elimina hipótesis de memoria ni problemas de estimación.
- [Estimating Transfer Entropy in Continuous Time Between Neural Spike Trains or Other Event-Based Data](https://doi.org/10.1371/journal.pcbi.1008054):
  candidato vecino localizado por índice. No se recuperaron pasajes del
  cuerpo mediante el conector; no se usa como evidencia de rendimiento.
- [An information-theoretic framework to measure the dynamic interaction between neural spike trains](https://arxiv.org/abs/2012.08667):
  vecino por metadatos/abstract, para comparación posterior de información
  dinámica y TE en procesos de eventos; no implementado ni validado aquí.

La búsqueda y lectura se hicieron con la skill Firecrawl; las extracciones
quedan en caché ignorada `.firecrawl/te-audit/`. El desarrollo algebraico de
la normalización y la prueba del algoritmo disperso son de esta auditoría.
No se atribuyen a los papers cifras calculadas por nuestros tests.

Para física/cuántica/problemas del milenio, el criterio de admisión sigue
siendo observable → operador → supuestos → identificabilidad → falsación →
coste → evidencia OOS. No se ha encontrado en este módulo un experimento
que justifique introducir esos formalismos. Matemática más avanzada no
repara por sí sola un denominador o un contrato de captura incorrectos.

## 9. Git, coordinación y bloqueo de publicación

Primera lectura remota con `git ls-remote` durante el cierre:

- main: `c0f6510f5ea8eb2e03a832ea6103aff2615133fc`.
- PR #10, abierto y no draft: head `01d1985aa10d2ee2d88a6e081deb97330602107f`.
- PR #20, abierto y draft: head `43ed1411afd27b8db40c0580bed48eda69b623e6`.

Main avanzó después a `817d5882` con ADRs; se incorporó en `057cb491`
sin tocar código. La rama local incluye ese corte y conserva ST. Las correcciones TE
NO están confirmadas en el remoto. No hay conflicto Git pendiente en el corte
ejecutable. Una referencia local equivalente a origin/main no significa que
los commits exclusivos de Codex se hayan publicado.

Se observó a GLM pasando de main a `glm/xlviii-f-adrs` en su propio checkout.
No se cambió su rama, índice, proceso ni ficheros versionados. Se dejó un
aviso ignorado en el checkout compartido:
`.firecrawl/coordination-codex-te-2026-09-29.md`.
No hay acuse: no se afirma comunicación efectiva con Claude/GLM.

No se borraron ramas: main se conserva; la rama activa de GLM no se elimina
aunque momentáneamente apunte a main; ST/TH/backups contienen trabajo no
integrado en main. Borrar solo por antigüedad o por un HEAD momentáneamente
alcanzable interferiría con las otras sesiones.

Publicación: un intento previo fue bloqueado por divulgación de detalles
internos hacia un repositorio público. No se evade la revisión usando push,
otro API o un comentario alternativo. El adjunto actual no responde
explícitamente a esa autorización informada. Los commits quedan locales.
Se requiere confirmación específica antes de publicar en
[Jhona-la/Trader-Gemini](https://github.com/Jhona-la/Trader-Gemini).

## 10. Conclusión de aseguramiento

Se cerraron defectos reproducibles del estimador y del experimento, se
preservó el trabajo concurrente y se delimitaron seis deudas científicas.
No se certifica un sistema omnisciente, sin bugs o rentable; tampoco se
presentan 1 ns–100 años como datos observables por este contrato de ms.
La siguiente decisión depende de evidencia de captura e inferencia
estadística, y la integración remota depende además de autorización.
