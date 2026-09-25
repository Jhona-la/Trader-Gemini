# Auditoría científica XIII — persistencia fiel, continuidad causal y arranque verificable

Fecha: 2026-09-24. Corte local observado: main, HEAD 59a76de4. Continuación aditiva de XII. Este documento no sustituye la matriz histórica de 305 puntos, no modifica sus identificadores ni declara cerrado el proyecto. Añade FMT-145–154 y el diseño experimental T36. Las reparaciones son locales y están diferenciadas de los defectos abiertos.

## 1. Resultado y perímetro de evidencia

Se corrigieron contratos de datos en tres archivos de producción: HistoryStore, StateDb y el parser compartido de warmup. No se cambió el genoma activo, la selección de estrategias, las órdenes, los límites de riesgo ni el estado de una cuenta. También se agregó un archivo de pruebas diagnósticas del checkpoint, deliberadamente sin cambiar el formato persistido.

Se ejecutaron **28 tests distintos**, incluidos **22 nuevos**. Doce pruebas reprodujeron fallos antes de las reparaciones: seis del histórico, cuatro de intenciones y dos del parser. Tres tests nuevos son caracterizaciones de defectos que siguen abiertos: su resultado verde confirma la debilidad actual, no su eliminación. Los cuatro tests adicionales de apertura/rangos/poda/rollback del histórico se añadieron después de la reparación y no se atribuyen retrospectivamente a una ejecución roja.

**Cobertura conservadora acumulada: 107/289 archivos Rust preexistentes leídos completos; 182 pendientes.** Las seis nuevas lecturas completas son history_store.rs, state_db.rs, bootloader.rs, state_continuity.rs, persistence.rs y storage-engine/src/lib.rs. La lectura dirigida del consumidor god_engine.rs y la relectura de temporal_store.rs no incrementan el acumulado. El inventario base sigue siendo 1.119 archivos versionados y 24 manifiestos Cargo. Buscar todos los símbolos no equivale a haber auditado todos los archivos.

La búsqueda en crates y src no localizó consumidores operativos de HistoryStore, StateDb, WalStorage ni save/load_checkpoint fuera de sus módulos y tests. Sí localizó la llamada de arranque vivo a SystemDiagnostics::execute_phase_3_warmup y al placeholder execute_phase_4_training. Por ello, una corrección de biblioteca no se presenta como mejora económica observada en producción.

| ID | Prioridad y alcance | Estado XIII | Resumen |
| --- | --- | --- | --- |
| FMT-145 | P2, contrato de biblioteca | Corregido y probado | Escrituras OHLCV inválidas, parciales o reconstruidas como éxito |
| FMT-146 | P2, contrato de biblioteca | Corregido en APIs fallibles; wrappers legacy limitados | Errores de lectura/apertura confundidos con ausencia y coordenadas truncadas |
| FMT-147 | P1 de diseño si se usa multifuente | Abierto | Clave histórica sin mercado, intervalo ni generación de universo |
| FMT-148 | P2, contrato de biblioteca | Corregido y probado | Etiquetas desconocidas convertidas a swing y estado inválido admitido |
| FMT-149 | P1 de recuperación | Abierto | Esquema nominalmente continuo incapaz de identificar todas las posiciones/estados |
| FMT-150 | P1 de recuperación | Abierto, caracterizado | Checksum parcial, colisiones y metadatos globales sin verificar |
| FMT-151 | P2 de persistencia/concurrencia | Abierto | Publicación temporal compartida, coste y garantía de recuperación no justificados |
| FMT-152 | P1, ruta de warmup llamada por el binario | Dominio numérico corregido; tiempo/completitud pendientes | Velas inválidas contaminan el estado inicial |
| FMT-153 | P1, gobernanza del arranque | Abierto | Entrenamiento vacío y verificaciones que sólo enumeran el estado existente |
| FMT-154 | P2, API WAL y documentación | Abierto en WalStorage | Ticks inválidos convertidos a cero y durabilidad confundida con atomicidad |

La prioridad expresa impacto potencial y contrato, no prueba de un incidente de mercado. No se midieron latencias productivas, rentabilidad, drawdown ni tasa de pérdida de eventos.

## 2. Paradigma de grafo vivo: del origen de la evidencia al efecto terminal

El problema no es solamente sustituir dos etiquetas por la palabra “continuo”. Un nodo puede llevar ese nombre y seguir almacenando una sola intención por activo; otro puede aceptar una vela falsa y producir miles de cálculos perfectamente finitos sobre evidencia inexistente.

| Nivel del grafo | Responsabilidad verificable | Hallazgo de esta ronda | Lo que todavía no demuestra |
| --- | --- | --- | --- |
| Nodo raíz: observaciones | Identidad, unidades, finitud, causalidad, cierre y procedencia | FMT-145/146/147/152/154 | Una validación OHLCV no acredita cobertura temporal |
| Nodos de estado | Conservar filtros, normalizadores, escalas y versiones | FMT-149/150/151 | Una lista de posiciones no reproduce el estado predictivo |
| Nodo de decisión | Vincular candidato, estado y evidencia disponible en ese instante | FMT-153 | Invocar una función llamada training no acredita aprendizaje |
| Nodo terminal | Identificar órdenes, fills, costes y efectos ya realizados | Requisito de T36; no reparado aquí | Repetir un evento no autoriza repetir una orden |
| Camino de recuperación | Restituir un corte consistente y reanudar sin omisión/duplicación observable | T36 y FMT-149–151 | Un checksum local no prueba consistencia del grafo |

Esta topología separa dependencias reales de nombres aspiracionales. No se identificó en los módulos revisados una demostración de ventaja cuántica ni de observación omnisciente. La continuidad temporal del modelo es una propiedad distinta de la resolución del feed, del scheduler y del almacenamiento.

## 3. FMT-145 — el histórico modificaba o descartaba evidencia sin expresar el fallo

**Evidencia:** [validación y escritura histórica](<C:/Users/jhona/Documents/Proyectos/Trader Gemini/crates/storage-engine/src/history_store.rs:203>).

Antes del cambio, una escritura individual podía devolver éxito sin almacenar una observación inválida. La ruta por lotes aplicaba otro criterio: omitía ciertas filas, reconstruía extremos inválidos a partir de apertura/cierre y sustituía volumen inválido por cero. Un lote podía producir un subconjunto o valores distintos de los recibidos y aun así devolver Ok.

Ese comportamiento mezclaba tres objetos estadísticos diferentes: dato observado, dato ausente y dato imputado. Por ejemplo, reemplazar un máximo desconocido por max(apertura,cierre) reduce artificialmente el rango intrabar. Un estimador de volatilidad que después use ese rango no puede recuperar la información perdida; el error se transmite a normalización, stops, sizing y fitness si se conecta esta biblioteca a esos consumidores. No se afirma que esa cadena se haya observado funcionando en producción.

El contrato ahora exige valores finitos, precios positivos, volumen no negativo y:

`low ≤ open ≤ high; low ≤ close ≤ high`.

Son invariantes del formato OHLCV usado aquí, no umbrales de alpha ni filtros de volatilidad. Se mantienen extremos válidos, saltos grandes y volumen cero. No se introdujo winsorización, clipping estadístico, bandas arbitrarias ni un mínimo de volumen operativo.

La escritura individual devuelve Err ante cualquier violación. El lote valida todas las filas antes de abrir la transacción y almacena exactamente los valores aceptados; un error SQL posterior revierte también las actualizaciones anteriores de ese lote. Un trigger de prueba aborta la segunda inserción y demuestra que una modificación de la primera vela no queda comprometida.

**Compatibilidad y coste:** los datos válidos mantienen el contrato, mientras que lotes parcialmente inválidos dejan de tener éxito parcial. La validación es O(n) sobre el lote y no se ha microbenchmarkeado. Si se desea ingestión parcial, debe diseñarse explícitamente un resultado con filas aceptadas/rechazadas, motivos y cuarentena; no restablecer silenciosamente la imputación. La reparación no limpia registros ya existentes.

**Cierre local:** pruebas rojas→verdes para dominio OHLCV y lote inválido; prueba posterior de rollback SQL; comparación de inserción individual y por lote sobre datos válidos. Sigue pendiente la integración operativa y la trazabilidad de revisiones de velas.

## 4. FMT-146 — ausencia, corrupción, truncamiento y destino alternativo eran indistinguibles

**Evidencia:** [apertura](<C:/Users/jhona/Documents/Proyectos/Trader Gemini/crates/storage-engine/src/history_store.rs:263>), [lecturas fallibles](<C:/Users/jhona/Documents/Proyectos/Trader Gemini/crates/storage-engine/src/history_store.rs:369>).

El constructor anterior redirigía fallos de apertura a un nombre fijo de SQLite en el directorio temporal. Ese fallback podía hacer que dos destinos lógicos distintos compartieran una base inesperada. También ocultaba fallos de creación de directorios y configuración. El éxito ya no certificaba estar leyendo el archivo solicitado.

Ahora try_new propaga errores de apertura, directorio, pragmas y esquema. El constructor de compatibilidad new conserva su firma, pero falla explícitamente mediante panic si no puede inicializar el destino; se recomienda try_new en integración operativa. La prueba usa una ruta que es un directorio, no una base productiva, y comprueba Err. No se ejecutó el antiguo fallback durante esa prueba ni se abrió su archivo compartido.

Las consultas usaban rows.flatten(): una fila cuyo tipo no se podía decodificar desaparecía de la respuesta. Además, casts de u64/usize a i64 podían envolver coordenadas; un timestamp persistido negativo podía reaparecer como una fecha enorme. El limit se recortaba al intervalo 1–100.000: pedir cero devolvía una fila y pedir más aparentaba completar una solicitud truncada.

Se agregaron conversiones comprobadas, decodificación estricta del dominio y recolección de Result por fila. Un error deja de generar una muestra parcial aparentemente sana. El límite cero retorna cero filas; superar el presupuesto existente de 100.000 retorna Err. No se presenta ese presupuesto de memoria como horizonte natural de mercado: una futura API paginada puede sustituirlo con semántica explícita.

try_get_range valida extremos y orden; try_get_latest_timestamp diferencia archivo sin datos de error y no toma como cursor la última fila corrupta; try_prune_older_than devuelve cantidad borrada y rechaza un cutoff no representable. Antes, saturar un cutoff excesivo podía ampliar una poda hasta casi todo el archivo. La prueba de poda usa SQLite en memoria y conserva la vela ante u64::MAX.

**Límite importante:** los wrappers legacy aún retornan Vec vacío/None o sólo registran errores. Sus firmas no pueden comunicar cobertura. Se documentaron como APIs con pérdida de información; no deben fundamentar entrenamiento ni certificación de integridad. Tampoco se implementó detección de huecos entre filas numéricamente válidas.

## 5. FMT-147 — la identidad histórica no identifica un universo multiespectral

**Evidencia:** [esquema de klines](<C:/Users/jhona/Documents/Proyectos/Trader Gemini/crates/storage-engine/src/history_store.rs:283>).

La clave primaria es (coin_id,timestamp). No contiene venue, mercado/instrumento estable, intervalo de agregación, unidad temporal, fuente, versión del registro de símbolos ni estado de cierre/revisión. El timestamp está documentado en milisegundos, pero no hay un contrato serializado que evite interpretar otra unidad.

Dos barras del mismo activo y apertura temporal, una de un minuto y otra de cinco, colisionan aunque ambas sean correctas. El ON CONFLICT actualiza la primera con la segunda. Si coin_id cambia de significado después de modificar el universo, el histórico no conserva por sí solo el símbolo que tenía ese identificador. El fallo surge bajo esas precondiciones; no se encontró una mezcla real en una base activa.

**Impacto:** una investigación multiescala podría entrenar sobre un mosaico no identificable, y un replay aparentemente determinista utilizar observaciones de otro soporte. Cambiar el nombre del horizonte no repara esa pérdida de identidad.

**Cierre exigible:** migración versionada que preserve el archivo original; clave de instrumento/fuente/intervalo o eventos primarios identificados; metadatos de tiempo de evento, recepción, cierre y revisión; pruebas de colisión deliberada y round-trip. No se asigna retroactivamente una escala a datos que no la registraron. Estado: abierto, sin cambios de esquema ni migraciones destructivas.

## 6. FMT-148 — recuperar una etiqueta desconocida como swing inventaba una decisión

**Evidencia:** [decoder de intenciones](<C:/Users/jhona/Documents/Proyectos/Trader Gemini/crates/data-pipeline/src/state_db.rs:146>), [escritura](<C:/Users/jhona/Documents/Proyectos/Trader Gemini/crates/data-pipeline/src/state_db.rs:237>).

Ambos getters seleccionaban Continuous para CONTINUOUS, Scalp para SCALP y Swing para cualquier otra cadena. Una etiqueta corrupta o de una versión futura adquiría una semántica operativa válida. No era una migración ni una inferencia estadística: era el brazo por defecto de un match.

Además, save_position_intent devolvía Ok al recibir precio/cantidad inválidos y no escribía nada. El consumidor podía creer que el último estado estaba persistido mientras permanecía el anterior. Las lecturas no comprobaban dominio de precio/cantidad, booleano, símbolo o timestamp. Las conversiones de identificadores y tiempo tampoco verificaban representabilidad.

Se incorporó un decoder común: sólo reconoce las tres etiquetas declaradas, exige is_long igual a 0 o 1, precio/cantidad finitos positivos, símbolo no vacío y timestamp no negativo. Escrituras inválidas retornan Err; IDs y tiempo se convierten mediante comprobaciones antes del SQL. Las operaciones de borrado y lectura también validan IDs. Se comprueba que las tres etiquetas históricas válidas continúan siendo legibles.

**Por qué no se borró Scalp/Swing:** aquí son valores de un formato persistido existente. Eliminar sus variantes sin una migración puede volver irrecuperables registros antiguos. Mantener lectores de compatibilidad no autoriza nuevos motores separados. El paso correcto es un esquema continuo nuevo y una migración trazable, no reinterpretar toda cadena como Continuous.

**Cierre local:** cuatro tests rojos→verdes, uno de compatibilidad y dos tests preexistentes. No se consultó una base real ni se demostró que este módulo sea el restaurador del motor vivo. La semántica de un estado válido sigue limitada por FMT-149.

## 7. FMT-149 — una etiqueta continua y una intención no bastan para recuperar el genoma fenotípico

**Evidencia:** [clave de StateDb](<C:/Users/jhona/Documents/Proyectos/Trader Gemini/crates/data-pipeline/src/state_db.rs:221>) y [get_position_intent](<C:/Users/jhona/Documents/Proyectos/Trader Gemini/crates/data-pipeline/src/state_db.rs:312>).

La tabla permite una intención por (coin_id,horizon). No hay coordenada tau, identidad de slot/posición, generación de candidato ni versión de política. Dos posiciones legítimas del mismo activo etiquetadas CONTINUOUS se reemplazan entre sí. El getter general recupera sólo la más reciente, no todas; si hay empate en updated_at no define un desempate adicional.

El upsert tampoco impide que un evento atrasado reemplace un updated_at más reciente. No se corrigió con un simple máximo de timestamp: primero debe definirse si el tiempo representa evento, recepción, revisión o secuencia de autoridad. Una corrección legítima tardía no necesariamente es un estado obsoleto.

El checkpoint paralelo mantiene is_scalp, pero no coordenadas espectrales ni estado de filtros, normalización, modelos, optimizador, RNG, orden de eventos o deduplicación. Guardar sólo genes y posiciones no preserva el fenotipo online: el mismo vector de genes puede actuar de otra manera si cambia el estado acumulado que transforma observaciones en features.

**Impacto condicionado:** explica un mecanismo posible de divergencia después de warmup/reinicio, pero esta ronda no establece que sea la causa medida de la diferencia backtest–demo. Faltan comparación de trazas y evidencia de cableado.

**Cierre:** esquema versionado con identidades estables, estado suficiente por nodo y contrato de reanudación; lectura de todas las posiciones; política de revisiones; prueba de replay con reinicio y los mismos futuros eventos. Debe incluir balances/órdenes reconciliados, no derivarlos de una memoria local potencialmente vieja.

## 8. FMT-150 — el checksum no cubre el estado que dice validar

**Evidencia:** [compute_state_checksum y validate_snapshot](<C:/Users/jhona/Documents/Proyectos/Trader Gemini/crates/quantum-arena/src/state_continuity.rs:44>), [pruebas diagnósticas](<C:/Users/jhona/Documents/Proyectos/Trader Gemini/crates/quantum-arena/tests/checkpoint_characterization.rs:1>).

La expresión es una rotación de bits:

`checksum = rotate_left(bits(size) XOR bits(entry_price), coin_id mod 64)`.

Su función real es mezclar dos patrones binarios y seis bits del identificador. No incluye símbolo, dirección, stops, extremos, etiqueta temporal ni timestamp. validate_snapshot sólo recomputa esa expresión.

Hay colisiones estructurales, no una discusión abstracta de probabilidad: intercambiar size y entry_price conserva XOR; aumentar coin_id en 64 conserva la rotación. Un test hace ambas modificaciones y el snapshot sigue validándose. Otro altera dirección, stops, símbolo y tiempo, incluso dejando campos no finitos, y validate_snapshot devuelve true.

load_checkpoint valida únicamente esos checksums por posición. No verifica version, global_checksum, capital, timestamp global ni consistencia entre posiciones. La prueba guarda/carga un archivo temporal propio con version=u32::MAX, capital=−100 y un global_checksum arbitrario; la carga resulta aceptada. No se alteró ningún checkpoint del usuario.

**Consecuencia:** “checksum válido” no acredita identidad ni seguridad de la exposición restaurada. Un hash más fuerte aplicado a los mismos tres campos seguiría omitiendo el resto del contrato. Si se requiere protección contra manipulación adversaria, un digest sin autenticación tampoco basta.

**Reparación pendiente:** formato nuevo con serialización canónica, integridad sobre todos los campos relevantes, dominio y versiones permitidas, validación de invariantes de cartera y compatibilidad explícita del lector legacy. No se cambió el checksum en el mismo formato porque invalidaría archivos anteriores sin mecanismo de migración. Los tres tests son caracterizaciones abiertas y deben sustituirse/reclasificarse cuando se implemente esa migración.

## 9. FMT-151 — atomicidad del archivo no equivale a snapshot consistente ni a recuperación O(1)

**Evidencia:** [save_checkpoint](<C:/Users/jhona/Documents/Proyectos/Trader Gemini/crates/quantum-arena/src/state_continuity.rs:57>).

Se serializa todo el Vec de posiciones a JSON, se escribe un temporal obtenido por with_extension("tmp"), se sincroniza el archivo y se renombra. Dos destinos como estado.json y estado.db comparten estado.tmp. Dos escritores del mismo destino comparten también ese temporal; File::create puede truncarlo. El módulo no implementa coordinación de escritores.

La sincronización del temporal es una medida real, pero no prueba por sí sola durabilidad del renombrado en todos los sistemas ni consistencia de una captura concurrente de varios nodos. No hay sincronización explícita del directorio, protocolo de generación/commit ni una prueba de recuperación tras interrupciones entre pasos. No se simularon cortes de energía.

Las afirmaciones de serialización cero copia, O(1) y recuperación garantizada en menos de un milisegundo no se sostienen en este algoritmo: serializar/deserializar P posiciones exige trabajo y memoria proporcionales al contenido, además de IO. Un alineamiento de 64 bytes de la estructura sin estado no transforma JSON en un formato cero copia.

**Cierre:** publicación con temporal exclusivo y protocolo de escritor; generaciones verificables; política de backup y lectura de último commit íntegro; pruebas con fallos inyectados antes/después de cada paso y benchmark por tamaño/plataforma. Antes debe definirse qué corte del grafo representa el archivo. Estado abierto: ni migración ni almacenamiento operativo fueron modificados.

## 10. FMT-152 — warmup numéricamente inválido y cobertura temporal sin contrato

**Evidencia:** [parser compartido](<C:/Users/jhona/Documents/Proyectos/Trader Gemini/crates/god-engine-core/src/bootloader.rs:10>), [ruta utilizada](<C:/Users/jhona/Documents/Proyectos/Trader Gemini/src/bin/god_engine.rs:1172>).

La ruta SystemDiagnostics validaba el cierre finito y algunas desigualdades, pero permitía apertura infinita, volumen NaN/negativo, low negativo y apertura fuera de [low,high]. La otra ruta aceptaba conversiones fallidas como cero y enviaba los valores al feature engine. La primera prueba falló con open="inf"; otra con open=13, high=12, low=9, close=11.

Se extrajo primero la lógica original a una función pura para reproducir los errores, después se reparó y se conectó a ambas rutas. Ahora los cinco campos deben parsearse y ser finitos, con la geometría y dominio descritos en FMT-145. No se fabrican valores. El log de la ruta moderna cuenta realmente las velas aceptadas y rechazadas; la ruta de compatibilidad avisa cuando rechaza filas por esquema/dominio.

**Pendiente temporal:** ambas peticiones siguen usando interval=1m y limit=1000. El retorno del diagnóstico es Vec<[f64;5]>, sin timestamps, cierre, edad, fuente ni motivos de huecos. No se prueba orden, unicidad, continuidad o cierre de la última barra. El parser valida el cuerpo OHLCV, no los tiempos que descarta la interfaz. La respuesta HTTP tampoco se convierte en un contrato tipado de estado/cobertura, y una carga fallida puede dejar una lista vacía sin bloquear por sí misma el arranque.

**Rigidez y latencia:** hay recorridos seriales y esperas fijas de 50 ms por símbolo en SystemDiagnostics y 200 ms en la otra ruta. Son elecciones de planificación/rate-limit, no constantes deducidas del espectro. No se eliminaron sin medir límites del proveedor, paralelismo admisible y p99 de arranque. Mil barras de un minuto son soporte de warmup concreto, no evidencia sobre todo el universo temporal.

El problema de rangos/relojes de FMT-083 y la ausencia de tiempo físico en otros estimadores ya estaban registrados: esta ronda aporta el defecto de parseo y la interfaz de origen, no los cuenta otra vez como nuevos hallazgos. Cierre restante: eventos timestamped, watermarks/cobertura, tratamiento explícito de revisiones y comparación causal con replay.

## 11. FMT-153 — las fases de arranque no acreditan entrenamiento ni continuidad

**Evidencia:** [placeholder training](<C:/Users/jhona/Documents/Proyectos/Trader Gemini/crates/god-engine-core/src/bootloader.rs:382>), [llamada y parámetros derivados del genoma](<C:/Users/jhona/Documents/Proyectos/Trader Gemini/src/bin/god_engine.rs:1178>).

execute_phase_4_training recibe símbolos, histórico, learning rate y epochs, pero sólo retorna Some(()). No usa esas entradas ni produce un modelo. El host deriva learning rate y epochs de quantum_mutation_rate y asigna el retorno a _swing_nn. En esta ruta concreta, modificar ese componente del genoma no puede cambiar un entrenamiento que no existe. No se deduce que todo el entrenamiento del proyecto esté vacío: hay otros mecanismos que deben evaluarse por separado.

SystemBootloader::phase_4_state_recovery recorre las posiciones ya presentes en memoria, calcula un checksum y lo imprime; no carga un checkpoint ni compara contra una referencia. Después afirma continuidad verificada. La fase ML sólo cuenta GLOBAL_FORESTS; no vincula versiones, universo o estado de entrenamiento. La integridad inicial compara capital con cinco, pero NaN no activa esa comparación, y fija treinta slots como prueba de topología. Son verificaciones incompletas, no teoremas de validez.

La búsqueda no encontró una llamada externa a execute_boot_sequence, por lo que no se confunde esa ruta alternativa con el warmup real del binario. El defecto de entrenamiento vacío sí tiene un caller identificado.

**Cierre:** estados tipados de fase (no ejecutada, fallida, parcial, verificada), evidencias y modelo resultante identificados, control de dependencias y un test de integración que modifique un input y compruebe el efecto declarado. Si una fase es intencionalmente un no-op, debe decirlo. Retirar etiquetas no puede sustituir esa prueba. No se implementó un entrenamiento arbitrario para llenar el nombre de la función ni se cambió el consumidor concurrente.

## 12. FMT-154 — almacenar cero no conserva un tick inválido; WAL no es una política completa de durabilidad

**Evidencia:** [WalStorage::insert_tick](<C:/Users/jhona/Documents/Proyectos/Trader Gemini/crates/data-pipeline/src/persistence.rs:48>).

WalStorage convierte precio no finito/no positivo a cero y volumen inválido a cero, inserta la fila y devuelve Ok. Su test legacy exige precisamente aceptar NaN con volumen negativo. No registra la condición original ni diferencia observación y corrección. También convierte timestamp u64 a i64 sin comprobación, no valida símbolo y no aporta identidad de evento para deduplicación. No se modificó este módulo en XIII; queda documentado como la siguiente ruta de datos a endurecer, sin fingir que arreglar HistoryStore lo arregló también.

Los comentarios prometen que synchronous=NORMAL relaja fsync sin riesgo crítico. SQLite distingue consistencia de durabilidad: con WAL y NORMAL una transacción comprometida puede perderse ante caída del sistema o energía, aunque la base mantenga consistencia. FULL agrega sincronización por commit en WAL. La elección debe responder a un presupuesto explícito de pérdida y latencia, no a una etiqueta HFT. [Documentación oficial de PRAGMA synchronous](https://www.sqlite.org/pragma.html#pragma_synchronous).

Se conservaron los pragmas existentes; no se impuso FULL universalmente sin medir coste y necesidades por tipo de dato. Los comentarios de HistoryStore/StateDb ahora reconocen el límite. Faltan RPO/RTO por evento, recuperación reproducible y mediciones en almacenamiento real. No hay benchmark que acredite “nanosegundos” de persistencia SQLite.

## 13. T36 — estado suficiente, cortes consistentes y equivalencia observable tras reinicio

Esta es una propuesta de diseño y validación, no una teoría implementada ni una garantía productiva. Se utilizaron Firecrawl Research Index y Scrape para contrastar fuentes primarias; no se envió código ni información de cuentas. La consulta de artículos aportó fragmentos y abstracts, no una lectura completa de toda la bibliografía.

### 13.1 Qué deben significar los cálculos

Sea e_k un evento con identidad, tiempo, procedencia y unidades; S_k el estado suficiente de los nodos antes de procesarlo; theta_v una versión de política; y a_k los efectos observables. El contrato idealizado es:

`(S_(k+1), a_k) = F(S_k, e_k; theta_v)`.

No se afirma que mercado y estrategia sean Markovianos por naturaleza. S debe incluir la memoria necesaria para que esa representación sea válida, o declarar su aproximación. Guardar un subconjunto S* sin filtros/modelos/orden no permite inferir que F(S*,e) producirá la misma acción que F(S,e).

La prueba operacional deseada compara dos trazas con iguales inputs futuros: ejecución ininterrumpida y ejecución con checkpoint, fallo y replay. Sus efectos terminales deben coincidir bajo la tolerancia declarada, sin duplicar órdenes ni omitir fills. Igualdad de PnL final es demasiado débil: puede ocultar decisiones diferentes compensadas por azar.

En un grafo asíncrono, un corte es consistente si no incluye la recepción de un mensaje sin su envío causal. La literatura de snapshots de dataflows distingue estado de operadores y mensajes en tránsito; los ciclos requieren tratamiento adicional. Esto justifica registrar fronteras de eventos y recuperación coordinada, no copiar unas estructuras por separado. [Lightweight Asynchronous Snapshots for Distributed Dataflows](https://arxiv.org/abs/1506.08603).

### 13.2 Tiempo continuo no significa fabricar observaciones cada nanosegundo

Una coordenada tau positiva puede modelarse continuamente, por ejemplo mediante funciones sobre log(tau), con evaluación numérica adaptativa y error declarado. El contrato debe separar unidad del reloj, resolución observada, soporte estimado y horizonte de decisión. Un reloj de nanosegundos no suministra evidencia nueva cada nanosegundo.

Para un operador de propagación P(delta_t), cuando las hipótesis lo permitan, una prueba útil es `P(a+b)≈P(a)P(b)`: dos particiones del mismo intervalo sin eventos nuevos no deben crear información extra. La primitiva temporal de T35 es un baseline relacionado; no fue integrada ni alterada por XIII. Si cambian controles, parámetros o régimen dentro del intervalo, esa igualdad necesita una formulación condicionada y no debe imponerse ciegamente.

Consultar un horizonte de cien años con datos de minutos es extrapolación/modelo de escenarios, no calibración identificada. La representación continua es compatible con recursos finitos; evaluar todas las escalas en todo instante no es un contrato computacional realizable literalmente. Deben publicarse soporte y error, no vender discretización oculta como omnisciencia.

### 13.3 Protocolo experimental propuesto

1. Congelar identidad de dataset, universo, modelo, política y estado inicial. Separar genotipo, estado aprendido y contexto operativo.
2. Registrar eventos primarios con secuencia e identidad suficientes para detectar duplicados, revisiones y saltos.
3. Definir por nodo qué estado y fronteras necesita reanudar. El nodo terminal requiere reconciliación e idempotencia de efectos, no sólo replay interno.
4. Inyectar fallos antes/después de persistir estado, publicar checkpoint y confirmar efectos; introducir duplicados, eventos tardíos y pausas.
5. Comparar features, señales, asignaciones y efectos por evento contra la traza sin fallo. Explicar tolerancias de punto flotante y fuentes no deterministas.
6. Sólo después comparar candidatos fuera de muestra con costes, latencia y disponibilidad iguales. Consistencia no implica rentabilidad, y aprendizaje no implica mejora.

No se integraron ecuaciones de problemas del milenio ni componentes “cuánticos” por su prestigio. Una incorporación futura necesita variable observable, correspondencia dimensional, hipótesis contrastables, baseline, presupuesto de cálculo y mejora reproducida. El formalismo no sustituye identidad de datos, causalidad o un entrenamiento ausente.

## 14. Matriz de filtros, estados y hoja de rehabilitación

| Capa | Regla o restricción | Evaluación | Siguiente cierre verificable |
| --- | --- | --- | --- |
| OHLCV de histórico/warmup | Finitud y geometría de precios; volumen≥0 | Invariante de dominio, corregida; no filtro económico | Compartir especificación sin duplicar contratos divergentes |
| Histórico reciente | 100.000 filas máximo | Presupuesto heredado ahora explícito; no escala física | Paginación con cursor y cobertura verificables |
| Warmup | 1 minuto, 1.000 barras | Soporte fijo, no espectro universal | Eventos con tiempos, resolución y criterios de suficiencia |
| Arranque | 30 slots, capital≥5 | Restricciones heredadas sin prueba universal | Topología desde registro versionado y validación finita de recursos |
| Persistencia | WAL/NORMAL, cachés y mmap fijos | Configuración técnica sin benchmark de esta ronda | RPO/RTO y percentiles bajo carga, por clase de estado |
| Recuperación | Checksums parciales y última intención | Insuficientes para inferir continuidad | Snapshot versionado y equivalencia observable T36 |

La rehabilitación 1-a-1 debe empezar por entradas y recuperación, seguir por integración de las APIs fallibles y terminar con evaluación económica. En concreto: FMT-154; migraciones FMT-147/149/150; publicación FMT-151; contrato temporal FMT-152; fases de arranque FMT-153. El orden expresa dependencias, no autorización para migrar bases activas o alterar trading automáticamente.

En los ocho módulos del maestro, XIII aporta evidencia nueva principalmente a Módulo 1 (ingestión/normalización), Módulo 3 (soporte temporal), Módulo 6 (persistencia/estado) y Módulo 8 (pruebas/gobernanza). Los Módulos 2 y 5 reciben una conexión concreta con entrenamiento y estado del genoma, no una nueva auditoría completa; Módulos 4 y 7 no se consideran nuevamente cubiertos por esta ronda.

## 15. Verificación, limitaciones y conservación

| Comando | Resultado |
| --- | --- |
| cargo test -p storage-engine --lib history_store:: --offline | 15 tests pasan |
| cargo test -p data-pipeline --lib state_db:: --offline | 7 tests pasan |
| cargo test -p god-engine-core --lib bootloader::audit_xiii_tests --offline | 3 tests pasan |
| cargo test -p quantum-arena --test checkpoint_characterization --offline | 3 caracterizaciones pasan; defecto continúa |
| cargo check --bin god_engine --offline | Pasa; no ejecuta el motor |
| rustfmt --check, ediciones 2021/2024 según archivo | Pasa en los cuatro archivos intervenidos |
| git diff --check en los tres fuentes modificados | Pasa |

La compilación conserva tres warnings preexistentes: latest_ts y mode sin uso, y RealWfOutcome.trades sin lectura. No se ejecutó la suite completa del workspace ni una sesión de trading. Una compilación correcta no prueba paridad dinámica ni performance.

El formateador encontró dos veces un bloqueo de escritura por sección mapeada de Windows en history_store.rs. Se obtuvo la salida formateada sin modificar el archivo y se aplicó mediante parche; la comprobación final pasó. No se detuvo ningún proceso del usuario.

El [artefacto JSON XIII](<C:/Users/jhona/Documents/Proyectos/Trader Gemini/docs/artifacts/auditoria_fundamentos_XIII_2026-09-24.json>) registra cobertura, estados, pruebas, fuentes, hashes y límites. Los informes anteriores se conservan; sólo se agregan adendas al maestro, atlas y XII. Los archivos concurrentes god_engine.rs, core/lib.rs, risk/lib.rs y booktick_replay.rs mantienen los hashes observados en XII.

No se hizo commit, push, merge, fetch, despliegue, reinicio, consulta autenticada de exchange, migración de base activa ni promoción de genomas. Main local no demuestra que las demás ramas estén integradas o que el remoto incluya estos cambios. La auditoría integral y las reparaciones arquitectónicas siguen abiertas.

### Integridad final de esta entrega

El JSON se parseó correctamente. Coinciden sus once hashes de fuentes, prueba nueva y archivos protegidos. Los diecisiete enlaces locales del informe resuelven a archivos existentes y sus números de línea están dentro del archivo. Los prefijos completos previos del maestro, atlas y XII conservan su SHA-256 después de normalizar CRLF a LF: las adendas no eliminaron ni reescribieron contenido histórico. La advertencia de Git sobre acceso a un archivo global de ignore no impidió las comprobaciones locales; no se modificó esa configuración.

## Continuidad — ronda XIV (2026-09-24)

Se añade [XIV](<C:/Users/jhona/Documents/Proyectos/Trader Gemini/docs/AUDITORIA_FUNDAMENTOS_CIENTIFICOS_XIV_2026-09-24.md>) y su [artefacto JSON](<C:/Users/jhona/Documents/Proyectos/Trader Gemini/docs/artifacts/auditoria_fundamentos_XIV_2026-09-24.json>) sin cambiar conclusiones ni evidencia histórica de XIII. La nueva ronda incorpora FMT-155–164, reparación numérica parcial de FMT-093 y caracterizaciones de FMT-023.

Se corrigen bypass temporal/revocación de autorización, evaluación repetida y preferencia nominal swing en adaptadores, admisión tras init fallido, estados financieros no finitos y dos funciones numéricas. Permanecen documentadas las desconexiones Hawkes/CVPIN, base del modelo no publicada, genes de helpers que no gobiernan el voto, estimadores temporales sintéticos, targets y telemetría. T37 es propuesta de expertos disponibles con feedback causal retardado, no despliegue.

XIV valida 84 tests distintos, 21 nuevos; once fueron rojo→verde y cinco caracterizan fallos todavía abiertos. La cobertura acumulada pasa a 114/289 Rust preexistentes completos, 175 pendientes. Compilación comprobada sin ejecutar el motor; no certificación integral ni publicación Git.
