# Auditoría de fundamentos científicos XXX — evidencia, vetos y persistencia

Fecha: 2026-09-25. Estado: reparación local verificada y deuda abierta; no certificación integral. Rama local main, HEAD observado 59a76de4. Continuación acumulativa de [XXIX](<C:/Users/jhona/Documents/Proyectos/Trader Gemini/docs/AUDITORIA_FUNDAMENTOS_CIENTIFICOS_XXIX_2026-09-25.md>), sin sustituir los informes anteriores ni la matriz histórica de 305 puntos.

[Artefacto estructurado XXX](<C:/Users/jhona/Documents/Proyectos/Trader Gemini/docs/artifacts/auditoria_fundamentos_XXX_2026-09-25.json>).

## 1. Dictamen ejecutivo y límites de la certificación

Se contienen dos defectos de admisión de telemetría y se corrige el contrato de lectura de una biblioteca de posiciones. Se hace visible el fallo de lectura en el consumidor de evolución. Se reproducen, sin repararlos todavía, la pérdida de frames pendientes, la supresión de propuestas defensivas por el kill-switch, el descarte de un snapshot de exposición cero, el descarte de exposición pequeña y la pérdida de eventos con la cola llena.

El resultado no equivale a un sistema autoevolutivo validado. Una optimización matemática no puede restaurar una observación que se perdió, distinguir un cero imputado de un estado plano observado ni transformar una intención en un fill. Estas son dependencias causales del aprendizaje, no tareas periféricas de almacenamiento.

Se añade una lectura integral al inventario preexistente: ledger.rs, 294 líneas en HEAD antes de esta intervención. La cobertura acumulada declarada pasa de 144 a **145 de 289 archivos Rust**, con **144 pendientes**. El inventario base conserva 1.119 archivos versionados y 24 manifiestos Cargo. Los nuevos tests y las relecturas no incrementan ese numerador. No se afirma haber leído cada archivo no Rust ni certificado todos los archivos ya leídos.

Resultado verificado de esta ronda: **35 pases distintos = 30 funcionales/compatibilidad + 5 diagnósticos que confirman deuda abierta**, sin ignorados en las selecciones ejecutadas. Hay 22 pruebas nuevas: 18 funcionales y cuatro diagnósticos. Las repeticiones no se suman. Se observaron cinco fallos de aserción antes de las correcciones y un proceso de test terminado por violación de acceso; después pasan los contratos correspondientes. No se ejecutó toda la suite del workspace.

Cargo check offline de god_engine, feature_exporter, train_forest y train_dark_alpha pasa. Permanecen tres warnings previos de evolution-engine. Los 41 modelos del snapshot XXIX conservan sus hashes. No se consultaron cuentas ni enviaron órdenes; tampoco se entrenaron/promovieron modelos operativos, construyó/reinició el ejecutable operativo ni publicó Git.

## 2. Paradigma de grafo vivo: de la raíz al resultado económico

La raíz del grafo debe ser evidencia identificada, con procedencia y tiempo, no una etiqueta de estrategia. El siguiente diagrama distingue el circuito deseado de los atajos que siguen sin justificar:

```text
RAÍZ: evento de mercado / evento de cuenta / reloj / generación de configuración
  │ identificador, activo, fuente, tiempo del evento y de recepción, validez
  ▼
Estado observado ──► representación temporal-espectral multiactivo
  │                         │ soporte, incertidumbre, unidades
  │                         ▼
  │                  NODO DE DECISIÓN
  │                  candidato + razón + contexto congelado
  │                         │
  │                  autorización por acción y evidencia
  │                         ▼
  └──────────────────► intención / reserva
                            │ ACK ≠ fill
                            ▼
                  fills / comisiones / conciliación
                            │ identidad y deduplicación
                            ▼
                  NODO TERMINAL ECONÓMICO
                  exposición, caja, resultado realizado
                            │ procedencia conservada
                            ▼
                  muestra causal → aprendizaje → candidato
                            │ validación fuera de muestra y gobernanza
                            └────────► nueva generación

Atajos todavía abiertos:
estimación local de cierre ──► aprendizaje legacy             [FMT-225]
head de reserva mmap ──► cursor de lectura ya consumido       [FMT-230]
kill-switch global ──┤ propuesta defensiva local              [FMT-232]
enqueue sin respuesta ──?──► commit durable                   [FMT-236]
```

El dibujo es diagnóstico, no una afirmación de que todos esos nodos estén implementados. Un nodo terminal de procesamiento no acredita liquidación económica. Una transacción de SQLite tampoco acredita haber recibido todos los eventos del exchange.

La biblioteca PositionLedger aquí intervenida no se encontró consumida por el runtime en la búsqueda de referencias en crates y src: aparecen su implementación, reexportación y tests. Por tanto, su mejora es una corrección de contrato auxiliar; **no repara por sí misma el circuito económico usado en demo/producción**. No se sustituyó la ruta operativa por esta biblioteca.

## 3. Resumen de resolución y matriz incremental

Los identificadores FMT-233…236 se agregan a la serie de fundamentos; no renumeran la matriz histórica de 305 puntos. No debe sumarse el último identificador FMT al número histórico para anunciar un total consolidado de bugs: hay dependencias y posibles solapamientos de raíz.

| ID | Prioridad y exposición | Evidencia | Estado al cerrar XXX |
|---|---|---|---|
| FMT-231 | P1, lector de telemetría | Crash aislado y contratos de tamaño | Contención de longitud; formato/concurrencia abiertos |
| FMT-233 | P1, escritor de telemetría | Archivo parcial aceptado antes; rechazo preservador después | Reparado para archivo existente no vacío e incompleto |
| FMT-234 | P2, biblioteca auxiliar | Tres aserciones fallan antes; once contratos de lectura pasan | Lectura unificada corregida; no es ledger de fills |
| FMT-235 | P2, biblioteca auxiliar | Dos diagnósticos de admisión | Abierto: cero absoluto y exposición pequeña descartados |
| FMT-236 | P2, biblioteca auxiliar | Cola llena reproducida; revisión de writer | Abierto: aceptación, persistencia y pérdida no distinguibles |
| FMT-230 | P1, canal hacia aprendizaje | Diagnóstico de reserva/commit | Abierto y reproducido otra vez |
| FMT-232 | P1, decisión defensiva | Diagnóstico sobre core real con fixture | Abierto; pasa de evidencia estática a reproducción |
| FMT-225 | P1, liquidación/aprendizaje | Deuda heredada de XXVIII–XXIX | Contención previa conservada; sin ledger económico completo |

P1 significa prioridad alta por alcance potencial, no incidente observado en una cuenta real. P2 en PositionLedger refleja que no se encontró consumidor operativo. La severidad deberá reevaluarse si se conecta la biblioteca a reconciliación o autorización de riesgo.

## 4. FMT-231 — longitud no validada antes de interpretar memoria

**Localización.** [open_mmap y admisión](<C:/Users/jhona/Documents/Proyectos/Trader Gemini/crates/storage-engine/src/mmap_bus.rs:272>), [guarda previa al acceso de cabecera](<C:/Users/jhona/Documents/Proyectos/Trader Gemini/crates/storage-engine/src/mmap_bus.rs:316>) y [contratos de admisión](<C:/Users/jhona/Documents/Proyectos/Trader Gemini/crates/storage-engine/tests/mmap_admission_contract.rs:66>).

**Contrato violado.** La validez del descriptor o del objeto mmap no implica que exista memoria respaldada suficiente para crear una referencia a AtomicUsize ni para recorrer el ring. Antes se interpretaba la cabecera antes de comprobar el tamaño. Un archivo vacío y una cabecera sin ring no son lotes vacíos válidos.

**Reproducción observada.** La primera corrida de contratos alcanzó un fallo de aserción para un archivo de cabecera y terminó después con 0xc0000005, STATUS_ACCESS_VIOLATION, al probar un archivo vacío. Fue un proceso de test aislado sobre su propio temporal, no god_engine ni un archivo operativo. Ese aborto impidió ejecutar los casos posteriores de esa corrida; no se registran como reproducciones pre-fix los casos que no llegaron a ejecutarse. Hubo también un error inicial de compilación del test por usar unwrap_err con un tipo sin Debug; se corrigió el test y no se cuenta como evidencia del defecto.

**Cambio.** Se exige longitud mínima antes de crear el mmap y de nuevo antes de interpretar su cabecera. Para el layout actual, tamaño requerido = 64 + 1.000.000 × 64 = 64.000.064 bytes. La comprobación precede al acceso unsafe. La apertura fallida del lector ya no se convierte en Ok(vec![]); read_latest_frames devuelve el error y conserva la posibilidad de reintento.

**Semántica.** Ok(lote vacío) representa una lectura exitosa sin frames nuevos conforme al protocolo legacy. Err representa evidencia no disponible o formato incompleto. Ni siquiera un Ok vacío acredita continuidad: FMT-230 y los saltos del cursor siguen existiendo.

**Propagación operativa acotada.** [El daemon](<C:/Users/jhona/Documents/Proyectos/Trader Gemini/crates/evolution-engine/src/online_daemon.rs:344>) registra el primer fallo de una racha y su recuperación. No satura el log con el mismo fallo en cada poll. El mensaje de recuperación aclara que no se ha acreditado continuidad histórica. Esto mejora observabilidad; no implementa backfill, ni contador durable de gaps, ni pausa general de entrenamiento/promoción. Si cambia la causa dentro de una racha de errores, el log no distingue cada causa nueva: queda pendiente un estado de salud estructurado.

**Verificación.** Siete tests cubren vacío, inexistente, cabecera aislada, longitudes cortas, reintento tras recuperación explícita y compatibilidad de archivos completos. El barrido de tamaños cortos incluye 0, 1, 7, 63, 64, 65 y requerido−1. Ese barrido se ejecutó después de añadir la guarda.

**Residual.** No hay magic/version/frame-size/generación en la admisión; un archivo arbitrario suficientemente grande no queda semánticamente validado. No se acredita seguridad ante truncamiento o sustitución posterior al mapeo. La inicialización simultánea de archivos vacíos, la vida del mapa, los accesos concurrentes y el protocolo de publicación necesitan diseño propio. No se realizó fault injection concurrente ni una prueba de comportamiento indefinido.

**Criterio de cierre sistémico.** Especificar un formato versionado, propiedad del archivo y límites de concurrencia; separar generación del escritor y secuencia absoluta; imponer una recuperación explícita; probar interrupción del escritor, reinicio, wraparound y pérdida observable. La guarda de tamaño no autoriza marcar todo FMT-231 como resuelto.

## 5. FMT-233 — inicialización que sobrescribía evidencia parcial

**Localización.** [Constructor del writer](<C:/Users/jhona/Documents/Proyectos/Trader Gemini/crates/storage-engine/src/mmap_bus.rs:105>) y [regresión de preservación](<C:/Users/jhona/Documents/Proyectos/Trader Gemini/crates/storage-engine/tests/mmap_admission_contract.rs:50>).

**Mecanismo anterior.** Un archivo más pequeño que el ring esperado se ampliaba y rellenaba con ceros. Se trataba igual un archivo nuevo que uno no vacío, incompleto por corte de proceso, formato incompatible o error externo. La supuesta reparación podía borrar evidencia que habría permitido diagnosticar su origen.

**Impacto.** El sistema podía presentar una nueva cabecera como si fuese una continuidad válida; además, el diagnóstico perdía el contenido previo. Es un defecto de recuperación y de preservación forense, no solamente una pérdida de logs. No se ha probado que haya ocurrido sobre archivos operativos.

**Reproducción y cambio.** Un test con contenido propio incompleto mostró que el constructor anterior aceptaba el archivo. El nuevo constructor devuelve InvalidData para 0 < tamaño < requerido antes de modificarlo. La prueba posterior verifica que longitud y bytes permanecen iguales. El caso de tamaño cero sigue permitido para inicialización; no se convierte una condición insegura en una prohibición global del arranque.

**Residual y tradeoff.** La recuperación deja de ser automática para esos archivos: un operador o proceso de recuperación deberá distinguir evidencia inválida, archivo en construcción y formato antiguo. Esa menor disponibilidad es deliberada hasta contar con protocolo de recuperación. No se ha añadido cuarentena automática, borrado, migración ni sincronización entre inicializadores. La prueba no acredita durabilidad después de una caída eléctrica.

**Criterio de cierre ampliado.** Archivo nuevo creado por un único propietario, validación de versión, política recuperable de preservación y publicación de un archivo completamente inicializado. El error debe llegar a salud operativa; init_global_telemetry aún puede fijar None y no se ha rediseñado su política de reintento en esta ronda.

## 6. FMT-234 — contrato dual que ocultaba posiciones y convertía errores en ceros

**Localización.** [Nuevo registro y lector](<C:/Users/jhona/Documents/Proyectos/Trader Gemini/crates/storage-engine/src/ledger.rs:20>), [read_ownership](<C:/Users/jhona/Documents/Proyectos/Trader Gemini/crates/storage-engine/src/ledger.rs:171>) y [tests](<C:/Users/jhona/Documents/Proyectos/Trader Gemini/crates/storage-engine/tests/ownership_read_contract.rs:49>).

**Mecanismo anterior.** get_ownership sólo podía devolver cuatro números correspondientes a dos etiquetas. Las filas con otra etiqueta se ignoraban y podían producir Some((0,0,0,0)) aun existiendo exposición. Los fallos de conversión de columnas se sustituían por cero. La iteración sobre filas podía terminar ante error sin propagarlo. La apertura habitual de SQLite podía crear un archivo inexistente durante una consulta supuestamente de lectura.

**Por qué es un fallo lógico, no cosmético.** Cambiar el nombre scalp por continuous no recupera las filas omitidas ni hace distinguibles ausencia de evidencia y ausencia de exposición. El contrato debe conservar todas las filas relevantes o rechazar una representación que no pueda expresarlas. La procedencia histórica puede conservarse para trazabilidad sin tener autoridad para particionar la política de trading.

**Cambio.** read_ownership devuelve un Result de registros con provenance_label, quantity y entry_price. Abre con SQLITE_OPEN_READ_ONLY; filtra por activo y lado de posición; conserva todas las etiquetas; propaga errores de apertura, esquema, ejecución y conversión. Rechaza etiqueta vacía, cantidades/precios no finitos o negativos y precio cero con exposición positiva. Un error en cualquier fila invalida la consulta completa, no devuelve un subconjunto que parezca una cartera completa. No aplica un epsilon de exposición mínima.

**Compatibilidad.** El adaptador de cuatro números permanece por compatibilidad, pero devuelve None ante etiquetas desconocidas, repetidas o error. None no debe interpretarse como autorización de nueva exposición. Sólo etiquetas históricas conocidas y sin ambigüedad pueden representarse mediante ese adaptador. La nueva API no crea motores de scalping y swing. Se mantiene la información antigua; no se migró ni eliminó una base real.

**Reproducción.** Tres tests fallaban por los motivos esperados: exposición continuous reportada como cero por el adaptador; cantidad mal tipada aceptada como cero; consulta de ruta inexistente que creaba archivo. El test de compatibilidad legacy ya pasaba y no se cuenta como rojo→verde. La ampliación posterior cubre mezcla de etiquetas, scoping por activo/lado, corrupción de una fila, dominios inválidos, exposición no nula muy pequeña, esquema ausente frente a tabla observada vacía y duplicado legacy.

**Límites.** Es una lectura de estado local, no prueba de fill ni reserva. No añade cuenta/proveedor, secuencia de evento, identidad de orden, generación de genoma, freshness, moneda de comisión o procedencia temporal. Se mantiene el timeout de SQLite de cinco segundos: no se ha certificado aptitud para hot path. Tampoco se ha implementado migración de todos los consumidores a la nueva API, porque no se encontró consumidor operativo de esta biblioteca.

**Criterio de cierre.** El contrato de lectura local queda reparado en los casos cubiertos. Antes de usarlo para autorizar riesgo deben diseñarse identidad de cuenta, snapshots coherentes y freshness, y conectarse a un ledger económico autoritativo. No se hereda certificación de producción de once pruebas aisladas.

## 7. FMT-235 — un filtro de cantidad impide representar el estado plano

**Localización.** [push_event](<C:/Users/jhona/Documents/Proyectos/Trader Gemini/crates/storage-engine/src/ledger.rs:156>), [borrado por cantidad](<C:/Users/jhona/Documents/Proyectos/Trader Gemini/crates/storage-engine/src/ledger.rs:91>) y [diagnósticos de admisión](<C:/Users/jhona/Documents/Proyectos/Trader Gemini/crates/storage-engine/src/ledger.rs:361>).

**Mecanismo confirmado.** El campo qty_delta se usa tanto para incrementos como para snapshots absolutos. La guarda abs(qty_delta) < 1e−12 descarta el snapshot absoluto cero antes de que llegue al writer. Sin embargo, el writer contiene una rama destinada a eliminar la posición cuando el override absoluto indica cantidad no positiva/pequeña. Para el cero exacto recibido por push_event, esa rama resulta inalcanzable.

**Consecuencia lógica.** Una conciliación que intente comunicar «la exposición ya es exactamente cero» puede no actualizar el estado persistido. Una posición fantasma podría sobrevivir en esta biblioteca. No se ha observado esa condición en una cuenta operativa, y no se ha reparado el writer.

**Segundo defecto.** El mismo filtro descarta 1e−13 aunque sea finito y no nulo. El writer además elimina cantidades <=1e−8. Son dos umbrales distintos en unidades de activo, no en riesgo monetario ni en precisión admitida por venue. Para cantidad q y precio P, exposición monetaria aproximada N=qP; el mismo epsilon q impone un corte monetario dependiente de P. La posibilidad de ejecutar una orden pequeña no decide si una exposición existente debe representarse.

**Pruebas.** Dos tests construyen sólo un canal acotado y el sender privado de la biblioteca. Comprueban que el cero absoluto y 1e−13 nunca llegan al receptor. No lanzan hilos escritores ni crean bases, por lo que demuestran rechazo de admisión, no un incidente de persistencia real. Sus pases significan que el defecto sigue presente.

**Reparación pendiente.** Separar SnapshotAbsolute de DeltaFill; permitir q=0 como estado absoluto conocido; validar signo según semántica del evento; conservar cualquier exposición representable; separar restricciones de orden de representación de estado; retornar motivo estructurado. Una sustitución ciega de todos los epsilons por cero no resolvería identidad, duplicados ni diferencias entre activo/venue.

**Aceptación.** Pruebas snapshot no cero→cero, delta de cierre, sobrecierre, subnormal, distinta escala de precio, hedge por lado y replay idempotente. No basta con que un test de apertura incremental pase.

## 8. FMT-236 — enqueue no constituye aceptación durable

**Localización.** [try_send ignorado](<C:/Users/jhona/Documents/Proyectos/Trader Gemini/crates/storage-engine/src/ledger.rs:165>) y el worker transaccional del mismo archivo.

**Mecanismo demostrado.** push_event devuelve unidad y descarta el Result de try_send. Una cola llena o desconectada es indistinguible de una admisión exitosa para el llamador. El diagnóstico usa capacidad uno: el primer evento queda en el receptor y el segundo evento válido desaparece, sin respuesta de rechazo.

**Mecanismo adicional por inspección.** El worker agrupa eventos, registra fallos de sentencias y continúa. Puede intentar commit después de errores individuales. No hay respuesta por evento ni identidad para que el productor determine qué se persistió, qué debe reconciliarse o qué reintento duplicaría una cantidad. El inicio de transacción y commit pueden fallar sin notificación al productor. No se inyectaron fallos de disco o SQL en esta ronda: esos modos se clasifican como evidencia estática, no como reproducción.

**Impacto potencial.** Si se conectase a autoridad económica, «no bloquear el hot path» podría significar perder estado sin marcar incertidumbre. La asimetría puede afectar inventario, margen, resultado atribuido y aprendizaje. Un canal acotado sí protege memoria, pero necesita una política de saturación acorde al tipo de dato. Telemetría best-effort y fills no tienen la misma tolerancia de pérdida.

**Especificación pendiente.** Diferenciar validación rechazada, backpressure, encolado, persistido y conciliado. Un ACK de enqueue no debe llamarse ACK durable. Necesita event_id, ámbito de cuenta/venue, generación y deduplicación; definir atomicidad de batch y rollback ante fallo; exponer salud/lag/offset; recuperar después de reinicio. Ninguna de estas garantías aparece por activar WAL o por denominar lock-free al canal.

**Aceptación.** Saturación, receptor caído, SQL rechazado, fallo de commit, reinicio tras commit antes del ACK y replay duplicado. Antes de reparar, decidir qué autoridad posee la exposición y qué eventos pueden descartarse. Esta deuda se relaciona con otros journals asíncronos ya auditados, pero se documenta en este componente sin adjudicarle causalidad de incidentes no observados.

## 9. FMT-230 y FMT-232 — deudas activas reproducidas

### 9.1. Cursor de reserva no equivale a cursor de publicación

[El diagnóstico de mmap](<C:/Users/jhona/Documents/Proyectos/Trader Gemini/crates/storage-engine/tests/mmap_open_diagnostics.rs:6>) conserva fases seriales sobre un archivo propio. Se presenta head=1 y seq impar; el lector devuelve vacío y avanza. Después se publica seq par con payload válido; el mismo lector no lo recupera, pero un lector nuevo sí lo ve.

La guarda de tamaño no cambia ese comportamiento. Avanzar por todo lo reservado y descartar los slots pendientes produce pérdida dependiente del interleaving. No se ha medido la tasa en producción. Si la pérdida depende de congestión y ésta depende del estado de mercado, no hay base para asumir missing completely at random; se trata de un riesgo de sesgo, no de un sesgo empírico ya cuantificado.

El límite de 10.000 frames por batch puede saltar observaciones anteriores sin gap tipado. El ring de un millón de slots es capacidad física, no un horizonte estadístico. Debe especificarse si el objetivo es últimas métricas aproximadas o evidencia completa para aprendizaje. Los nuevos tests de compatibilidad no convierten al bus en un journal lossless.

### 9.2. Kill-switch bloquea la propuesta de cierre local

[Retorno temprano del core](<C:/Users/jhona/Documents/Proyectos/Trader Gemini/crates/god-engine-core/src/lib.rs:922>) y [nuevo diagnóstico](<C:/Users/jhona/Documents/Proyectos/Trader Gemini/crates/god-engine-core/tests/close_outcome_contract.rs:345>).

Sobre arena y posición propias de test, con entrada 100 y stop 99, se procesa un bid 97. Con kill_switch_active=true, la ruta no emite entrada ni cierre y la posición local sigue abierta. Desactivar sólo ese flag de fixture y repetir permite el cierre con pérdida. Esto reproduce el veto de esa vía defensiva; no demuestra que desaparezcan brackets del exchange ni todas las otras rutas de salida.

No se retira el veto operativo. Un fallo de integridad de precios, una prohibición del operador y una alarma de degradación predictiva pueden requerir políticas distintas. Debe distinguirse aumentar exposición, reducir, cancelar y reconciliar, con autoridad y evidencia disponibles. «Cerrar siempre pese a cualquier alarma» tampoco es una política segura si se desconoce la posición o la conectividad.

Criterio de cierre: matriz causa × acción × evidencia; pruebas de posición confirmada, entrada pendiente, cierre parcial, bracket externo, fallo del canal de órdenes y reinicio. Debe observarse rechazo estructurado y conservarse la intención defensiva cuando su ejecución no esté autorizada.

## 10. Auditoría de vetos: qué es rígido y qué protege un contrato

| Condición | Naturaleza | Decisión de XXX | Justificación y deuda |
|---|---|---|---|
| Archivo incompleto | Integridad de memoria/evidencia | Rechazar con error | No es un régimen de mercado; evita interpretar memoria no respaldada |
| Archivo existente parcial | Recuperación/preservación | No rellenar con ceros | Preservar bytes permite diagnóstico; falta recuperación versionada |
| Error de lectura mmap | Disponibilidad de evidencia | Propagar y registrar | Ausencia de evidencia no equivale a ausencia de observaciones |
| Etiqueta distinta de scalp/swing | Representación arbitrariamente dual | Admitir en nueva lectura | Etiqueta como procedencia, no límite del universo temporal |
| Fila numérica inválida | Dominio de estado | Rechazar consulta completa | No imputar una cartera segura con ceros o subconjuntos |
| Cantidad absoluta cero | Estado válido de reconciliación | Defecto abierto de writer | Debe poder representar ausencia observada de exposición |
| Exposición < epsilon fijo | Rigidez dimensional | Defecto abierto de writer | No confundir minimum order size con inventario existente |
| Cola llena | Capacidad/latencia | Defecto abierto de señalización | Necesita backpressure y salud, no silencio |
| Kill-switch global | Seguridad con ámbito excesivo | Conservar y rediseñar | No liberar entradas; separar autoridad defensiva |
| Slot pendiente | Protocolo de publicación | Pérdida abierta | Una espera acotada requiere estado/gap, no salto invisible |

La evolución adaptativa no debe aprender a desactivar invariantes de memoria, dominio numérico, autenticación, identidad económica o límites mandatados por el operador. Sí puede optimizar políticas económicas bajo esos contratos y expresar incertidumbre en vez de sustituirla por categorías arbitrarias.

## 11. Teoría y significado de los cálculos

### 11.1. Continuidad temporal: representación, observación y presupuesto no son lo mismo

El objetivo de un motor continuo es que la política y su estado no dependan de elegir entre dos motores discretos. No exige un bucle exhaustivo que evalúe todos los instantes representables. Un intervalo de 100 años de 365,25 días contiene aproximadamente 3,15576×10^18 nanosegundos; almacenar un solo escalar de ocho bytes por instante requeriría aproximadamente 25,24608×10^18 bytes, antes de activos y dimensiones. Es un cálculo de escala, no una estimación del coste actual del sistema.

Una representación funcional puede definirse sobre τ>0 sin afirmar que todas las escalas sean identificables con los datos disponibles. Resolución del timestamp, frecuencia de llegada, horizonte de memoria y horizonte de predicción son cantidades diferentes. Un timestamp en nanosegundos no acredita una observación independiente ni una decisión completada cada nanosegundo; un parámetro de cien años no acredita datos predictivos para cien años.

Como especificación candidata, u=log(τ/τ_ref) permite parametrizar escalas positivas en una coordenada adimensional. τ_ref sólo fija unidad/coordenada. Una memoria causal de relajación puede usar:

```text
m(t,τ) = exp(−Δt/τ)·m(t−Δt,τ) + [1−exp(−Δt/τ)]·x(t)
```

Sirve para hacer explícita la dependencia entre tiempo transcurrido y memoria, no para probar rentabilidad. Requiere Δt≥0, unidades compatibles y una política ante desorden temporal. Para integración espectral, los nodos de cuadratura son una aproximación numérica con error y presupuesto: no deben convertirse silenciosamente en dos estrategias operativas ni en una afirmación de cobertura infinita. Esta ronda no implementa una nueva integración espectral.

### 11.2. Volatilidad y multi-asset: estados continuos con incertidumbre

Tratar volatilidad como estado continuo no elimina discontinuidades reales del mercado, límites de ejecución ni cambios de calidad de datos. Un detector de cambios puede producir evidencia probabilística sin imponer que todas las decisiones se reduzcan a un enum de «regímenes».

Para un vector de exposiciones w y matriz de covarianza Σ, la forma wᵀΣw tiene sentido de varianza sólo si unidades, sincronización y ventana/ponderación son compatibles y Σ es semidefinida positiva. Sin sincronización y política causal de precios atrasados, una fórmula correcta puede calcular una cantidad distinta de la pretendida. No se ha revalidado toda la estimación de Σ ni su proyección PSD en esta ronda; son criterios de verificación, no capacidades añadidas.

La API de ownership conserva activo y lado; eso evita mezclar filas de dos activos en la consulta, pero no demuestra control conjunto de cartera. Leer labels universales tampoco implementa una estructura tensorial, una cópula o una matriz espectral cruzada.

### 11.3. Genoma: sensibilidad económica y observabilidad

Para un parámetro g_j y una salida económica y, una aproximación diagnóstica es D_j(h)=[y(g+h e_j)−y(g−h e_j)]/(2h). Su propósito es medir si el gen alcanza la ruta de decisión y con qué sensibilidad local; no optimizar por sí mismo. Debe probarse convergencia respecto a h, normalizar unidades y mantener idénticos datos/semillas/contexto. Un cero puede provenir de saturación, una rama no activada, discretización del venue o desconexión, no necesariamente de irrelevancia teórica.

La comprobación completa debe recorrer genoma→features→score→riesgo→payload→fill→costes→aprendizaje. Las rondas anteriores repararon una desconexión concreta de horizonte después del cierre; XXX muestra que todavía pueden perderse muestras en el bus. Añadir más genes sin identidad de evento y procedencia estable haría menos interpretable, no más verificable, el proceso evolutivo.

No se atribuye la brecha backtest/demo/prod a una causa única. Esta ronda aporta mecanismos comprobados que deben contrastarse con replay causal y evidencia operativa autorizada. No hay experimento de PnL real ni estimación del tamaño de efecto económico.

### 11.4. Error, cero y dato ausente son tipos distintos

Un resultado numérico cero puede significar retorno exactamente nulo, exposición observada plana, imputación, fallo de parser o dato no observado. Mezclarlos altera recuentos, medias y variancias. La corrección de lectura usa Result para separar error de observación válida; sigue pendiente transportar tipos de evidencia por todas las rutas.

Que la pérdida de frames no sea uniforme puede invalidar el supuesto de muestra representativa. Una corrección estadística de selección exigiría un modelo identificable de inclusión, no un peso inventado a posteriori. La prioridad es medir gaps y conservar procedencia.

### 11.5. Física, cuántica y problemas del milenio: criterio de admisión científica

La complejidad de una ecuación no es evidencia de utilidad económica. En esta ronda no se añadió una ecuación de los problemas del milenio ni se afirmó resolver alguno. Tampoco se implementó computación cuántica ni se acreditó ventaja cuántica.

Para incorporar una teoría, debe documentarse: variable observable y unidad; hipótesis física/estadística; correspondencia entre objeto matemático y mercado; identificabilidad con datos; aproximación numérica; coste de cálculo; baseline clásico; criterio falsable fuera de muestra; ablación; estabilidad ante costes y latencia. Un formalismo cuántico simulado en hardware clásico debe nombrarse como tal; no hereda una ventaja de hardware ni probabilidades calibradas por su nomenclatura.

La deuda observada aquí corresponde a semántica de evidencia, concurrencia y autorización. Un formalismo más sofisticado no sustituye esos contratos. Las analogías de red viva pueden orientar el diseño del grafo, pero sus aristas deben representar dependencias y mensajes verificables.

## 12. Memoria y latencia: contraste primario y límites

Se usaron las habilidades Firecrawl de búsqueda de desarrollo y extracción para consultar contratos primarios. Al no disponer de CLI se recurrió al conector. No se enviaron fuentes privadas del proyecto. La síntesis local se conserva en [nota de evidencia](<C:/Users/jhona/Documents/Proyectos/Trader Gemini/.firecrawl/XXX-memory-contract-evidence.md>).

La documentación de Rust distingue un acceso volatile de una operación atómica y de sincronización entre hilos. No basta una segunda lectura de secuencia para dar por demostrado todo el contrato de memoria. [Rust: read_volatile](https://doc.rust-lang.org/std/ptr/fn.read_volatile.html).

memmap2 identifica como unsafe los mapas respaldados por archivo por riesgos asociados a cambios del archivo, incluso externos al proceso. De ahí la limitación explícita: la guarda inicial no acredita seguridad durante toda la vida del mapa. [memmap2: MmapOptions, safety](https://docs.rs/memmap2/latest/memmap2/struct.MmapOptions.html#safety).

Se corrigieron comentarios que presentaban repr(C) como garantía de persistencia, atomicidad de traces o latencia picosegundo/cero. También se aclara que cachear el mapa evita reapertura frecuente, pero recorrer un batch tiene trabajo dependiente de su tamaño. No se midieron percentiles de latencia ni throughput; el término «hot path» no reemplaza un benchmark. Las fuentes públicas consultadas no constituyen una certificación del toolchain completo instalado.

## 13. Estado de los ocho módulos

| Módulo | Trabajo pertinente de XXX | Lo que no queda certificado |
|---|---|---|
| 1. Ingestión/parsers/L2 | Distinción error versus lote vacío en telemetría; preservación de archivo parcial | Libros L2, continuidad del feed y normalización completa |
| 2. Inferencia/señales | Visibilidad de interrupción del canal hacia aprendizaje | Calibración, features causales, PPO real y ausencia de fuga |
| 3. Multiactivo/horizontes | Lectura sin partición obligatoria por dos etiquetas; especificación de escala/unidades | Universo temporal íntegro, dependencias cruzadas y sincronización global |
| 4. Ejecución/conectividad | Diagnóstico de supresión de defensa local | Fill de salida, reconciliación y paridad transporte/demo/prod |
| 5. Riesgo/genomas | Distinción inventario versus minimum order; preservación de regresión Kelly | Kelly conjunto óptimo, sensibilidad de todos los genes y ledger económico |
| 6. Estado/mmap/SO | Guardas de tamaño, no sobrescritura de archivo parcial, lectura SQLite explícita | Protocolo concurrente, durabilidad, backpressure y gaps |
| 7. Cuántica/orquestación | Criterio científico y grafo causal explícitos | Ventaja cuántica o conexión íntegra de todos los nodos |
| 8. Backtest/gobernanza | Fixtures propios; regresiones anteriores conservadas; auditoría acumulativa | Certificación del workspace, promoción estadística y validación operativa |

## 14. Evidencia de verificación y alcance de las pruebas

Comandos ejecutados:

```text
cargo test --offline -p storage-engine --test mmap_admission_contract --test mmap_open_diagnostics --test ownership_read_contract -- --test-threads=1
cargo test --offline -p storage-engine --lib mmap_bus::tests -- --test-threads=1
cargo test --offline -p storage-engine --lib ledger::tests::xxx_open -- --test-threads=1
cargo test --offline -p god-engine-core --test close_outcome_contract -- --test-threads=1
cargo check --offline --bin god_engine --bin feature_exporter --bin train_forest --bin train_dark_alpha
```

| Selección | Funcionales/compatibilidad | Diagnósticos abiertos | Interpretación |
|---|---:|---:|---|
| mmap_admission_contract | 7 | 0 | Admisión, preservación y recuperación explícita |
| ownership_read_contract | 11 | 0 | Semántica de lectura y compatibilidad |
| mmap_bus::tests | 3 | 0 | Compatibilidad serial existente; no prueba de concurrencia |
| close_outcome_contract | 9 | 1 | Conserva XXIX; añade deuda de kill-switch |
| mmap_open_diagnostics | 0 | 1 | Pérdida de frame sigue presente |
| ledger::tests::xxx_open | 0 | 3 | Rechazo de cero, epsilon y cola llena siguen presentes |
| Total distinto | 30 | 5 | 35 pases, no 35 fallos reparados |

El test legacy de sanitización NaN en mmap se cuenta como compatibilidad: preservar su comportamiento no acredita que convertir valores inválidos en cero sea estadísticamente correcto. Los diagnósticos abiertos están nombrados como tales y no deben presentarse como una certificación verde del componente.

Rustfmt --check pasa para los dos nuevos archivos de integración y el test de cierres intervenido. git diff --check pasa en las tres fuentes versionadas modificadas. No se aplicó formato global ni cargo fix sobre el árbol compartido.

No se ejecutaron los dos tests legacy de writer SQLite asíncrono en esta selección: no se acredita por ellos escritura durable. No se ejecutaron benchmarks, tests de carrera real, recuperación de cuenta ni todo el workspace. El cambio de log del daemon tiene check de compilación, no una prueba end-to-end de su servicio en ejecución.

## 15. Preservación, artefactos y concurrencia de sesiones

Los informes anteriores se mantienen; el atlas, el maestro y XXIX reciben únicamente adendas. El artefacto XXX registra hashes SHA-256 de las fuentes y tests de esta ronda, 41 modelos y los prefijos documentales normalizados CRLF→LF. Los hashes anclan un snapshot, no una garantía contra cambios posteriores de otras sesiones.

Se respetaron los cambios ajenos y no se ejecutó git add, commit, push, merge, fetch, reset ni checkout. Las referencias remotas no se actualizaron ni verificaron. No se atribuye el contenido completo del diff compartido a esta intervención.

Los tests crean temporales de propiedad exclusiva. El aborto del test vacío dejó una carpeta propia con un archivo vacío; se verificó su ruta exacta bajo Temp y se eliminó únicamente ese archivo y esa carpeta vacía, sin borrado recursivo. No se borraron datasets, traumas, modelos ni archivos operativos del usuario. El temporal era un fixture generado para esta prueba, no evidencia de mercado.

La comprobación de hashes de modelos no incluye una certificación semántica del modelo ni una afirmación de que los genomas compartidos no hayan sido editados por otras sesiones. Esta ronda no modificó deliberadamente genomas activos.

## 16. Hoja de ruta por dependencia y criterios de aceptación

1. **Definir autoridad económica.** Completar FMT-225 con identidad de intención, fill, cuenta y generación, reservas y conciliación. El resultado de cierre debe distinguirse de la estimación local. Probar partial fill, cierre pendiente, replay y reinicio.
2. **Separar seguridad por acción.** Resolver FMT-232 mediante matriz de causas y capacidades; preservar veto de nueva exposición y probar rutas defensivas con datos válidos e inválidos. No desactivar globalmente protecciones.
3. **Elegir el contrato del bus.** Best-effort de métricas versus evidencia para aprendizaje. Para el segundo, resolver FMT-230/231 con publicación, secuencia, generación, gaps, recuperación y memoria formalmente revisada.
4. **Arreglar escritura de estado antes de promover la biblioteca a autoridad.** Resolver FMT-235/236 con eventos tipados, ACK por etapa, idempotencia y política de saturación. El nuevo lector no compensa pérdida previa.
5. **Cerrar trazabilidad del aprendizaje.** Features congeladas en la decisión, costes/fills observados y procedencia de cada muestra. El log de error no garantiza dataset causal ni elimina FMT-006/052.
6. **Validar sensibilidad temporal y multiasset.** Barridos continuos de genes, unidades, soportes de datos, error numérico y presupuesto; comparar replay idéntico entre backtest/demo/prod con una tolerancia económica definida.
7. **Completar cobertura pendiente.** Leer los 144 Rust preexistentes restantes y el inventario no Rust, sin convertir una búsqueda de texto en lectura integral. Mantener deuda, reproducciones y límites de evidencia por archivo.

No se acredita omnisciencia, ausencia de bugs, rentabilidad, duplicación periódica de capital ni ventaja cuántica. El avance concreto es que ciertos estados inválidos ya no se convierten silenciosamente en evidencia aparentemente válida y que varios bloqueos antes sólo sospechados ahora tienen reproducciones acotadas.

## 17. Adenda de continuidad XXXI — consejo y aprendizaje

[Informe XXXI](<C:/Users/jhona/Documents/Proyectos/Trader Gemini/docs/AUDITORIA_FUNDAMENTOS_CIENTIFICOS_XXXI_2026-09-25.md>) y [artefacto XXXI](<C:/Users/jhona/Documents/Proyectos/Trader Gemini/docs/artifacts/auditoria_fundamentos_XXXI_2026-09-25.json>). Esta ampliación no modifica los resultados históricos de XXX ni cierra sus deudas de mmap, writer de ledger o kill-switch.

Se añaden FMT-237…245. Se corrigen validación del consejo, alineación de una excepción causal, conexión del parámetro de cascada, referencia ML de Teleonomia, contabilización de abstenciones y orientación del feedback de cortos. Siguen abiertos el shrinkage con población incompatible, el consenso sin independencia acreditada y la compresión de horizontes al intervalo30s–12h.

XXXI registra23 tests nuevos y13 contratos rojo→verde. Selección distinta:37 funcionales +4 diagnósticos abiertos =41 pases. Check de cuatro binarios pasa. Nueva lectura integral consejo_seniors.rs: cobertura146/289 Rust preexistentes,143 pendientes. Los41 modelos se preservan. No hubo operación real, limpieza de cachés compartidas ni publicación Git. Las cifras anteriores de XXX permanecen como evidencia de aquella ronda.
