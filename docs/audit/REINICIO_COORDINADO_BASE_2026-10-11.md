# Reinicio coordinado desde la base — 11 de octubre de 2026

Estado: **plan operativo publicado; revisión semántica completa y certificación económica abiertas**. Coordinación Codex SI; consejo de revisión independiente: teoría/estadística, riesgo/recursos e integración. Mandato ampliado del operador: volver a metas, conceptos, teoría, algoritmos y recorrer cada archivo del sistema modificado.

## 1. Precedencia, versiones y alcance

Esta adenda operacionaliza el [plan rector del universo espectral](../PLAN_MAESTRO_REVISION_BASE_UNIVERSO_ESPECTRAL_2026-10-11.md), el [recorrido R0–R9](../PLAN_REVISION_ARCHIVO_POR_ARCHIVO_2026-10-07.md) y la [asignación C-F0–C-F11](../PLAN_LINEA_C_DESDE_LA_BASE_2026-10-11.md). El [plan compartido](../PLAN_MAESTRO_SINCRONIZACION.md) mantiene los contratos entre agentes. No reinicia sus números ni elimina bitácoras, reservas o resultados negativos. Para este reinicio, las cifras y estados vigentes son los ligados al snapshot de esta sección; los cierres históricos conservan su SHA y alcance.

Base remota observada: `1d1f8bdd35640e5ab12b6e55edb9bf582c41c33a`. Candidato SI previo a esta adenda: `a44c22e6e55f051ccdace7e95f06a12651f4d665`, [PR #39](https://github.com/Jhona-la/Trader-Gemini/pull/39). Un nuevo commit documental no se incluye retrospectivamente en ese censo.

| Artefacto | Qué acredita |
|---|---|
| [Inventario por blob](REINICIO_BASE_INVENTARIO_2026-10-11.json) | Las **1.583 rutas versionadas** de la base, todas inicialmente `inventariado`; cero archivos declarados revisados por generar el censo. |
| [Asignación completa de Línea C](REINICIO_BASE_FASES_C_2026-10-11.tsv) | Una fase dueña por ruta, 1.583 asignaciones explícitas, ninguna por fallback; recibos pendientes. La fase no asigna por sí sola un escritor humano. |
| [Delta base → candidato SI](REINICIO_BASE_DELTA_SI_2026-10-11.json) | Cuatro adiciones, siete cambios y cero eliminaciones en ese corte; no incluye esta adenda posterior. |
| [Informe SI](../AUDITORIA_EVIDENCIA_ESPECTRAL_SI_2026-10-11.md) | Contratos específicos corregidos y límites de esa evidencia. El recibo terminal de validación e integración está en el PR. |

Categorías del censo: 357 fuentes, 196 tests, 57 configuraciones, 34 operaciones, 197 documentos, nueve datos/modelos, 656 generados y 77 otros. El manifest declara **23 miembros y el paquete raíz**. Cantidades como «39 crates» o «21 crates» en tablas del plan rector son cantidades de **archivos asignados a fases**, no de crates.

Reutilizar `scripts/audit_inventory.py` y `scripts/plan_linea_c.py`. El primer clasificador usa F0–F8 heurísticos: son rutas propuestas, **no equivalencias numéricas** con R0–R9 o C-F0–C-F11. El TSV C gobierna la asignación de este recorrido. El checker estructural no prueba verdad científica ni cobertura conductual. Recibo observado: checker `ok=true`, 1.583 filas; siete tests del inventariador aprobados. Herramientas auxiliares de auditoría no pertenecen al runtime Rust.

Antes de cada lote resolver `origin/main` a SHA. Si cambia, producir un delta y regenerar el censo con `--previous` sólo cuando se desea conservar recibos de blobs idénticos. Un cambio de blob/modo invalida el recibo del archivo. Un cambio semántico en productor, consumidor, esquema, datos o modelo obliga además a reabrir los contratos dependientes aunque su blob no cambie. Archivos eliminados conservan evidencia histórica; archivos nuevos empiezan pendientes. No editar un ledger fechado para fingir que describía otra base.

## 2. Meta y criterios que pueden refutarla

Sobre patrimonio multiactivo neto MTM, unitizado para separar aportes/retiros, definir:

\[
G_{72}(t)=\log\frac{V(t+72h)}{V(t)},\qquad
\Delta_{72}(t)=G_{72}(t)-\log 2.
\]

La referencia es `ln(2)/3 = 0,23104906` por día, equivalente a `25,992105 %` diario compuesto. Un promedio equivalente no demuestra duplicación en cada ventana. Publicar distribución, frecuencia de cumplimiento, déficit, intervalos de incertidumbre bajo supuestos declarados, drawdown, insolvencia, costes, frecuencia de oportunidades y capacidad. Ventanas disjuntas sirven de resumen y móviles de diagnóstico; ninguna división acredita independencia automáticamente. Capital no positivo implica fallo/ruina, no un logaritmo válido que se pueda omitir.

El significado operativo de «consistente» aún requiere frecuencia y severidad admisibles de incumplimiento; no se inventa un umbral. Separar tres estados: corrección de software, evidencia predictiva y resultado económico. Cada requisito lleva estimando, unidad, fuente, falsador y estado medido/no medido.

Se conserva la meta del operador como hipótesis exigente. ADR-0003 y ADR-0016 contienen encuadres históricos distintos; esta adenda no sustituye la meta por otra ni convierte su posibilidad algebraica en un resultado. La «propiedad ergódica», `p>0,43` y las sustituciones de modelos del plan rector quedan como **hipótesis pendientes**: requieren pagos, costes, dependencia, frecuencia, supervivencia y capacidad definidos. Ninguna medición de esta ola acredita +100 %/72 h.

## 3. Recorrido global, dependencias y cola de cada archivo

| Recorrido existente | Pregunta y salida exigida | Asignación C y tamaño de la base |
|---|---|---|
| R0: objetivos y doctrina | Requisitos económicos e ingeniería; resolver afirmaciones contradictorias con fuentes y medición. | C-F0: 30 archivos. |
| R1: conceptos y estado | Identidad de instrumento, unidades, reloj, autoridad, procedencia, desconocido/cero/inválido/obsoleto. | C-F1: 49. |
| R2–R3: fundamentos y representación | Teoría → observable → operador → estimación causal → baseline → falsación → consumidor. | C-F2: 42. |
| R4: datos y tiempo | Disponibilidad causal, reloj, gaps, parsers y replay. | C-F6: 47. |
| R5: capital, vetos y ejecución | Contratos del campo multiescala consumido por decisiones, dinero y órdenes. | C-F3: 39; C-F4: 21; C-F5: 27. |
| R6: genoma y aprendizaje | Etiquetas, particiones, selección, promoción, persistencia y rollback reproducibles. | C-F7: 73. |
| R7: plataforma y rendimiento | Telemetría fiel, recursos, concurrencia e incidentes recuperables. | C-F8: 71; revisión de rendimiento también transversal. |
| R8: UI, scripts, generados y documentación | Herramientas sin efectos imprevistos, afirmaciones y procedencia exactas. | C-F9: 86; C-F11: 900. |
| R9: integración y evidencia | Pruebas aplicables, árbol integrado y examen económico separado. | C-F10: 198; contratos entre todas las fases. |

Las fases C son la clasificación de rutas; R expresa dependencias de razonamiento. La revisión de datos/relojes C-F6 se adelanta cuando los fundamentos dependen de ellos. El código se inspecciona durante todas las fases para contrastar la teoría con el comportamiento real. No se difiere la revisión de causalidad hasta terminar todo el catálogo científico.

Para cada fila del TSV, sin omitir documentación, generados, scripts, manifests ni modelos:

1. Reservar ruta, SHA/blob, fase, escritor, revisor, contrato y consumidores; un escritor por ruta. Acusar la reserva antes de editar zonas activas ajenas.
2. Leer el archivo completo en rangos registrados. Para binarios/datos registrar formato, procedencia, hash, esquema y validación adecuada; no tratar bytes como código ni exponer secretos.
3. Trazar productores y consumidores productivos. Distinguir definición, código muerto, alias, configuración histórica y vía efectivamente alcanzada.
4. Revisar dominio, unidades, reloj, estado inicial, errores, concurrencia, persistencia, complejidad, coste y supuestos. Anotar contrato por contrato, incluso si no se encuentra defecto.
5. Para fallos reproducibles: test de comportamiento que falle antes de corregir; corrección mínima; GREEN y regresiones de consumidores. Una búsqueda de texto no sustituye la ejecución del caso.
6. Registrar recibo y pendientes. Revisar una función o pasar sus tests no cierra el archivo completo. Documentos y generados requieren comprobar afirmaciones/procedencia, no tests artificiales.
7. Revisar composición en el SHA integrado y actualizar el delta. «Leído sin defecto encontrado» no significa ausencia de bugs.

Recibo mínimo por contrato: `ruta`, `commit`, `blob`, `modo`, `rangos_leidos`, `whole_file_closed`, `owner`, `revisor`, `contrato`, `productores`, `consumidores`, `hallazgos`, `supuestos`, `pruebas/fixtures`, `comandos`, `exit`, `SHA_ejecutable`, `limites`, `reversion`, `evidencia_main`. Los estados existentes son `inventariado`, `en_revision`, `hallazgo_abierto`, `verificado`; todo `verificado` necesita evidencia referida al blob y un alcance explícito. Sólo completar **todos** los rangos/contratos permite cerrar el archivo. La cobertura total de rutas es una obligación del plan; su revisión semántica total sigue pendiente.

## 4. Concepto continuo, multiactivo y observabilidad

Estado causal indexado por `(instrumento, tiempo_evento, horizonte τ, procedencia)`, con incertidumbre y reloj de disponibilidad. Volatilidad, liquidez, dependencia y horizonte son coordenadas; anclas finitas aproximan un campo continuo y necesitan declarar resolución, interpolación, error y banda observable. Etiquetas scalp/swing se clasifican primero como alias o particiones reales; renombrarlas no cambia la arquitectura.

Pruebas transversales: permutar activos junto con sus specs/modelos/estado; aislar A de B salvo dependencias comunes declaradas; impedir doble gasto de margen/reservas concurrentes; eventos tardíos/duplicados/gaps/timestamps iguales; invariancia frente a cadencia de consulta; simetría largo/corto bajo costes simétricos; barrer τ y pesos cerca de anclas y antiguas fronteras. Correlación desconocida no equivale a cero. Continuidad de señales no elimina lotes, tipos de orden o controles de solvencia discretos.

Primeras observaciones para reproducir, todavía sin cierre de archivo:

| Observación | Contrato a comprobar y reserva |
|---|---|
| SI-O02: suelo móvil de objetivos, climatología en coordenadas distintas e inversión que omite restar el desplazamiento. | Transformación versionada y congelada al emitir; separar expectativa en log de expectativa física. Coordinar con C-F2; SI no declara corregido este hallazgo. |
| Cooldown común de moneda usa τ del último cierre para todos los candidatos. Las antiguas funciones con fronteras scalp/swing consultadas no tienen caller productivo. | Medir si un cierre de 4 h bloquea indebidamente un candidato de 30 s; comprobar interacción de dependencia y riesgo antes de cambiar política. Núcleo/estrategia: entregar reproducción al propietario. |
| `SPECTRUM_SCALES_MS = 1e-6 ms × 4^i`, i=0…31, representa aproximadamente 1 ns…146 años. | Resolver diferencias con «1 µs…208,5 días» del [plan cuántico del 10 de octubre](../PLAN_MAESTRO_CUANTICO_UNIVERSO_ESPECTRAL_CONTINUO_2026-10-10.md) y «4⁰ µs…4³¹ µs» del rector del 11 (aproximadamente 146.000 años). Timestamps enteros en ms no observan escalas sub-ms; duración representada no es historia disponible. |
| README conserva «dos motores scalping+swing», diagrama y cifra «17/17 tests». | Contrastar cada afirmación con callers, CI y SHA actual; no usar README como prueba de arquitectura vigente. |
| ADR-0014 declara F0 doctrinal cerrada; ADR-0015 conserva rearme del latch pendiente; ADR-0016 deja D1/D4 pendientes. | Un cierre doctrinal no cierra implementación ni evidencia económica. Reevaluar los abiertos en consumidores actuales. |

## 5. Teoría avanzada y siguiente evolución de cada modelo

Aplicar [ADR-0005](../adr/ADR-0005-triage-teorico-y-medicion.md): implementada, parcial, candidata o especulativa. Cada teoría necesita magnitud observable, unidades, dominio, condiciones de contorno, identificabilidad, estimador causal, incertidumbre, referencia sencilla, falsación y consumidor concreto. Una analogía física debe justificar su correspondencia con variables de mercado. Los problemas del milenio no conceden garantía financiera ni justifican sustituir modelos por su nombre.

| Familia a revisar | Experimento antes de conexión |
|---|---|
| Pronóstico: RF, boosting, modelos lineales/online, redes y operadores aprendidos | Mismos datos disponibles, costes y presupuesto; calibración, score emitido, ablation y examen temporal. XGBoost es un candidato comparativo, no una mejora universal de RF. |
| Dependencia: cópulas, grafos/factores, random matrices, cointegración | Nulo y señal implantada, asincronía, estabilidad entre escalas, multiplicidad y coste por activo/par. |
| Dinámica: Hawkes, SDE, filtrado, Koopman/DMD, firmas de camino, transporte óptimo | Identificabilidad, unidades, cambios de muestreo, referencia simple y error del consumidor. Una PIN-SDE no sustituye RF sin evidencia comparativa. |
| Control/evolución: optimización robusta, aprendizaje online, control estocástico, genética/PPO | Conservar genealogía y reglas de examen; medir adaptación frente a incumbente fijo, adaptación desactivada y baseline simple. |
| Conducta humana: flujo, respuestas a noticias, reflexividad, sentimiento | Fuente lícita y disponible históricamente, sesgo de selección, manipulación, privacidad, estabilidad y contribución marginal OOS. Son hipótesis sobre observables, no lectura de pensamientos. |
| Cuántica | Declarar cálculo clásico, simulación clásica, inspiración cuántica o hardware cuántico real. Comparar calidad y coste total incluyendo preparación, medición y ruido; una amplitud compleja no acredita ventaja cuántica. |

Las limitaciones del hardware ruidoso y la profundidad ejecutable importan al evaluar una propuesta cuántica ([Preskill, 2018](https://arxiv.org/abs/1801.00862)). El catálogo se explora fuera del camino crítico hasta demostrar beneficio. Resultados negativos y candidatos rechazados son entregables; no se cablean teorías pendientes para satisfacer una etiqueta de sofisticación.

## 6. Vetos, límites, literales y recuperación

Conservar `RiesgoDuro/Logica` como eje de finalidad del registro. Añadir al **recibo de auditoría**, sin cambiar todavía el enum productivo, dos dimensiones: naturaleza y procedencia.

| Naturaleza | Fundamento y actualización legítima |
|---|---|
| Invariante | Identidad, causalidad, conservación contable y restricciones oficiales del instrumento. Contrato y prueba; refrescar specs desde fuente autorizada. |
| Política | Riesgo tolerado, presupuesto de exploración, α estadístico y alcance. Decisión explícita con responsable/unidades; no llamarla ley física. |
| Parámetro estimado/aprendido | Probabilidad, costes, dependencia, memoria, pesos. Datos, estimador, incertidumbre, vigencia, adaptación causal acotada y reversión. |
| Cota numérica | Redondeo, dominio representable, denominadores. Derivar de precisión/escala/error tolerable; no ajustar para elevar PnL. |

Procedencia separada: exchange, medición, genoma, fórmula, configuración o literal histórico. Un gate mezcla componentes. `z=1,645` depende del nivel elegido y del modelo; `d*=1−α^(c/(2−c))` es una cota condicional, no universal. Con c=0,5 y α=0,05 vale aproximadamente 0,631597; no transferir una caída desde W0 fijo a drawdown desde pico móvil sin justificarlo.

Buscar `return`, `None`, `clamp`, `min/max`, defaults, saturaciones, timeouts y vías de bypass además del registro nominal. Cada entrada registra productor, consumidor terminal, unidad, dominio, activo/estrategia/τ/cartera, reloj, estado inicial, autoridad, fórmula/estimador, ausencia/caducidad, rechazo exacto, recuperación y pruebas.

Contrato exigido: admisión y rechazo alcanzando el gate; igualdad y ambos lados de frontera; NaN/Inf/ausencia/cero válido; ausencia de mutación parcial de dinero/reloj/aprendizaje al rechazar; precedencia entre gates; caminos alternativos; persistencia/reinicio/expiración. Un bloqueo de entradas conserva las salidas previstas por ADR-0015. Si promete recuperación, probar desde cero posiciones con el veto activo: no puede requerir exclusivamente cierres que él mismo impide.

Reducir rechazos no mide calidad. Falsos positivos/negativos exigen resultado esperado independiente. El libro sombra conserva procedencia contrafactual y modelo de fills; no se confunde con ejecución real. Con probabilidad de selección cero no existe soporte para estimar por ponderación inversa el resultado real de candidatos nunca ejecutados. Ninguna adaptación autoriza retirar una restricción porque impide alcanzar la meta.

## 7. Autoevolución falsable y examen independiente

Secuencia auditable: `observación → pronóstico emitido → etiqueta madura → evaluación → actualización → candidato → selección → promoción → consumo → rollback`. Persistir genealogía, versiones, estado inicial, datos/modelos disponibles, candidatos rechazados y resultado tras reinicio.

Reutilizar el [protocolo OOS R4](PROTOCOLO_OOS_R4_2026-10-07.md). Separar desarrollo, selección y examen; purgar según maduración real y solapamiento; registrar consultas y multiplicidad entre sesiones/reinicios. Prerregistrar reglas de actualización: una política adaptativa puede aprender causalmente durante el examen, pero no se retocan sus reglas con el resultado del propio examen. Si se hace, ese conjunto pasa a selección.

La reutilización adaptativa de un holdout puede sobreajustarlo ([Dwork et al., 2015](https://arxiv.org/abs/1506.02629)). La inspección secuencial necesita un procedimiento válido bajo supuestos comprobados; las secuencias de confianza no eliminan esos supuestos ([Howard et al.](https://arxiv.org/abs/1810.08240)). Comparaciones con incumbente congelado, adaptación desactivada y referencia simple usan mismos costes, recursos y disponibilidad temporal. Primero etiquetas/unidades correctas; luego sombra causal, evaluación independiente y actualización acotada.

## 8. Rendimiento, escalabilidad y recursos limitados

Cada presupuesto pertenece a un componente y workload. Distinguir acceso a registros, núcleo completo y recepción→decisión→envío. Las aspiraciones `<25 ns` y cero asignaciones requieren una medición correspondiente: no se extrapolan del acceso atómico al motor completo ni de Wine a Windows. `O(1)` declara respecto de qué variable; no fija una duración. Esta ola SI no incluye certificación nueva del camino crítico.

Reutilizar `medidor_camino_caliente.rs` y el comparador de Línea C: SHA, hash del ejecutable, compilador/flags, plataforma, fixture, modelos, semilla, activos/escalas y carga externa controlada; compilar antes de medir. El comparador informa p99 pero no lo usa como puerta: registrar esa limitación. La aceptación nueva deberá definir p99/p99,9 desde frescura y latencia admisibles, backlog máximo, recuperación de ráfagas, memoria pico/crecimiento residual, asignaciones por evento y pérdida/duplicación. Un activo, varios y máximo soportado son workloads diferentes. Sin presupuesto o identidad comparable, el estado es **no medido/no comparable**.

Para validar en este equipo: una cadena Cargo pesada propia a la vez, targets privados por worktree, `CARGO_BUILD_JOBS=2` en la ola actual y T1 con cuatro workers internos. Es una política de recursos, no un parámetro del modelo. Dos cadenas previas se interrumpieron deliberadamente ante presión de memoria y se registran como interrumpidas, no fallos de tests. No compartir binaries de targets entre worktrees ni alterar fixtures, ignores o umbrales para ahorrar tiempo. Observar memoria/commit y coordinar compilaciones externas; no detener procesos ajenos ni el motor vivo.

## 9. Reservas, primeros lotes e integración

| Frente | Estado y siguiente recibo |
|---|---|
| SI / Codex chat01a0f2fc | Escritor de `spectral_tape.rs`, feedback epigenético acotado en `state.rs`, consumidor forecast de `spectral_bench`, contratos y esta adenda. Funciones publicadas en PR #39; validación terminal e integración pendientes al redactar. |
| R4 / Codex chat01a116f0 | ACK de estrategia multiactivo, ceros/admisión/replay y e-values; conserva recuperación histórica separada. Aviso cruzado antes de avanzar main. No editar sus rutas desde SI. |
| Línea C / Claude | Plan C y C-F2/host/ejecución reservados según coordinación existente. Hallazgos SI-O02 y fundamentos se entregan con reproducción; no asumir que una reserva de fase permite editar todos sus archivos. |
| AGY / checkout primario Ω83 | Activo según ACK del otro chat. Mantener HEAD, archivos y ramas del checkout compartido. |
| Línea E, Sol y otras refs | Preservar reservas observadas y commits exclusivos. No inventar ACK o inactividad por el nombre de una rama. Renovar propietario/rutas antes del siguiente lote. |

Orden concreto: R0 metas/ADR/afirmaciones → R1 identidad/unidades/desconocido/relojes y C-F6 datos necesarios → C-F2 representaciones/targets → C-F3/C-F4 decisiones/dinero → C-F5 ejecución → C-F7 adaptación/evolución → C-F8/C-F9 recursos/operación → C-F10/C-F11 cierre de pruebas/documentos/generados. Los hallazgos críticos de contabilidad, causalidad o salidas se priorizan al detectarlos, conservando el dueño de ruta. Cada lote publica reserva, hallazgo reproducible, pruebas, límites y recibo remoto; los demás agentes revisan sin escritura concurrente.

Ramas con prefijo `codex/` y nombre propio. Antes de cada commit revisar status; stage sólo archivos propios explícitos. Antes de integrar: fetch, SHA de base/head, diff contra **cada padre**, pruebas aplicables y `cargo check --workspace --all-targets --locked`; si main cambió, reevaluar composición, no heredar un resultado por nombre. CI y merge ligados al head exacto. Después comprobar padres/árbol, ancestry y SHA remoto de main. El recibo terminal del PR evita autorreferencias documentales infinitas.

Limpiar únicamente refs cuyo OID está incorporado a main remoto, sin commits exclusivos ni reserva/actividad; conservar main y worktrees activos. Una ref desaparecida no prueba merge. Retirar la rama propia sólo después de integrar y terminar sus procesos. No hacer checkout/fast-forward del primario ocupado. El cierre de esta entrega será plan/censo, correcciones SI, validación e integración verificadas; no se declararán 1.583 archivos auditados, sistema libre de fallos ni rentabilidad demostrada.
