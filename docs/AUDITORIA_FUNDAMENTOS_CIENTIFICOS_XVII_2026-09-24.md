# Auditoría científica XVII — universo multiactivo, identidad estable y selección adaptativa

Fecha: 2026-09-24. Corte local: main, HEAD 59a76de4. Continuación aditiva de XVI; se preservan el maestro, el atlas y la matriz histórica de 305 puntos. Los IDs FMT-171 a FMT-177 son registros complementarios, no una suma automática a esa matriz.

## 1. Resultado ejecutivo y frontera de certificación

Se revisaron completos ocho archivos Rust preexistentes que no figuraban como lecturas completas en las rondas anteriores: active_universe, symbol_ranker_engine, symbol_registry, symbols, dynamic_ranker, asset_selector, dynamic_selector y symbol_manager. La cobertura conservadora avanza de 116 a **124/289 Rust; quedan 165 pendientes**. El inventario base sigue siendo 1.119 archivos versionados y 24 manifiestos Cargo. Buscar un símbolo o leer un fragmento del host no equivale a auditar ese archivo completo.

Se modifican dos fuentes y se agregan tres archivos de pruebas. **35 tests distintos pasan: 23 nuevos y 12 existentes. Doce contraejemplos fallaron antes de las correcciones y pasan después; seis pruebas nuevas refuerzan contratos; cinco caracterizan deuda abierta.** No se cuentan las reejecuciones. Cargo check del binario pasa sin arrancarlo. No hubo backtest económico, validación HTTP contra un exchange, benchmark p99 ni suite completa del workspace.

La revisión confirma el problema conceptual del usuario en un lugar concreto: el selector auxiliar conserva dos scores nominales scalp/swing y el radar operativo usa retorno neto de 24 horas como si fuera volatilidad. Además, el universo puede cambiar sin que todos los estados por índice cambien de significado conjuntamente. Una teoría de dependencia multiactivo no corrige una clave de instrumento mal atribuida.

| ID | Prioridad y alcance | Estado XVII |
| --- | --- | --- |
| FMT-171 | P2, API auxiliar de admisión | Corregidos capital/dimensiones/finitud, prioridad forzada, duplicados y factibilidad mínima; optimización y lifecycle pendientes |
| FMT-172 | P1 de diseño en ruta de ranking; componentes auxiliares diferenciados | Retorno diario confundido con volatilidad, dos scores discretos y penalidad no monótona; abierto |
| FMT-173 | P2, selector data-ingest sin caller operativo localizado | No fabrica universo con feed inválido; duplicados y empates tratados; publicación global sigue limitada |
| FMT-174 | P1, host/estado multiactivo | IDs congelados y universo reordenable pueden atribuir estados a otro símbolo; abierto, reproducción aislada |
| FMT-175 | P2, validador auxiliar sin caller operativo localizado | NaN admitido y redondeo que aumenta cantidad/sale de la malla; abierto y reproducido |
| FMT-176 | P1, políticas de universo utilizadas por symbol_manager | Filtros divergentes, blacklist sin condición de entorno y defaults de metadatos; abierto |
| FMT-177 | P1, publicación del universo y metadatos | Dos snapshots sin epoch común, specs sin refrescar cuando no cambia la lista y errores ignorados; abierto |

No se afirma que todos los mecanismos observados alcancen actualmente una orden. La búsqueda estática localizó fetch_dynamic_universe en symbol_manager y el daemon con resuscripción en el host. No localizó llamadas operativas a calculate_active_universe/calculate_dynamic_universe, DynamicSelector::select_top_10, SymbolRankerEngine::run_daemon ni SymbolSpec::validate_order fuera de sus definiciones/pruebas en crates/src. Esa búsqueda no descarta consumidores externos, dinámicos o futuras conexiones.

## 2. Grafo vivo: la identidad es una invariante de raíz a terminal

```text
Ticker24h + ExchangeInfo
   └─ dynamic_ranker: políticas, ranking, specs
       └─ symbol_manager: roster + histéresis + límite
           ├─ archivos de configuración (dos representaciones)
           ├─ universo ordenado mutable
           ├─ registro de specs por símbolo
           └─ URL WS y solicitud de reconexión
                    ↓
Host: mapa símbolo→ID del arranque, con fallback dinámico
                    ↓
arena.coins[ID], features[ID], inventario, modelos y órdenes
                    ↑
Otros consumidores: try_index / try_symbol sobre universo actual

Rutas auxiliares:
capital/precios/specs → active_universe → lista + bitmap64
ticker JSON → DynamicSelector → top10 → publicación directa del universo
```

La revisión sigue el grafo de datos y sus consumidores, no solo los nombres de funciones. Una lista ordenada de oportunidades es una vista; no debe actuar a la vez como dirección estable de posiciones y como índice de un estado que sobrevive a la rotación. La sincronía requerida es semántica: instrumento, venue, observación, versión y posición deben conservar su correspondencia. Un ArcSwap protege una publicación individual, no demuestra coherencia entre todas estas aristas.

El eje temporal continuo tampoco elimina la necesidad de identidad discreta de instrumentos. La disponibilidad contractual de una acción, un veto de solvencia y una coordenada espectral son objetos distintos. Se debe eliminar arbitrariedad donde no hay justificación, sin confundir una condición de seguridad verificable con un régimen económico aprendido.

## 3. FMT-171 — Admisión de universo con capital ficticio, prioridad débil y datos inválidos

**Evidencia:** [active_universe](<C:/Users/jhona/Documents/Proyectos/Trader Gemini/crates/quantum-arena/src/active_universe.rs:8>), [APIs checked](<C:/Users/jhona/Documents/Proyectos/Trader Gemini/crates/quantum-arena/src/active_universe.rs:52>), [contratos reproducibles](<C:/Users/jhona/Documents/Proyectos/Trader Gemini/crates/quantum-arena/tests/universe_selection_contract.rs>).

### Mecanismos anteriores

El límite de activos se calculaba como ceil(sqrt(max(capital,1))/2), acotado por el tamaño del universo. Capital no finito se reemplazaba por 13; capital cero o negativo aún admitía al menos una moneda. Es una imputación de capacidad financiera desconocida, no una estimación de capital.

Los campos dinámicos se indexaban por prices.len() sin comprobar los tamaños de volumes_usd y price_changes_pct. Una única moneda con vectores dinámicos vacíos provocaba un panic. El filtro price≤0 no rechazaba NaN; posteriores max sobre flotantes podían convertir propiedades desconocidas en scores finitos artificialmente atractivos.

Las monedas forzadas recibían un bono de un millón. Sin embargo, la granularidad puede aportar cinco millones por el suelo 10⁻⁶ en la fórmula estática, y el momentum dinámico no tiene esa cota. El bono no era una garantía de permanencia. Con tres activos y cupo dos, el forzado podía ser expulsado. Repetir el mismo ID cuatro veces ampliaba take_count hasta cuatro aunque solo existiera una obligación distinta.

El control de accesibilidad comparaba min_qty·price contra capital_por_moneda·20. Ignoraba min_notional y max_leverage del instrumento. Un lote de nocional pequeño puede estar sujeto a un mínimo nocional mayor; un producto con apalancamiento máximo uno no adquiere capacidad veinte por una constante del selector.

### Correcciones

Se introduce UniverseSelectionError: InvalidCapital, LengthMismatch, InvalidForcedCoin, InvalidForcedEvidence y BitmapCapacityExceeded. Las APIs try_calculate_* permiten conservar el motivo de rechazo. Los wrappers previos retornan selección vacía ante error, sin fabricar capacidad; **esa salida no autoriza a abandonar la gestión de posiciones abiertas**. Se documenta que admisión y lifecycle deben vivir separados.

Las obligaciones forzadas se deduplican mediante conjunto y se ordenan lexicográficamente antes del score, sin bonos finitos. Primero se distingue si la permanencia es obligatoria; después se compara la heurística entre elementos de la misma categoría; el ID desempata. Se valida que los forzados tengan ID, spec y evidencia utilizables. Si faltan, se devuelve error explícito en lugar de anunciar una cobertura inexistente.

Para admisión no forzada se comprueba:

- N_lote = min_qty·precio.
- N_requerido = max(N_lote, min_notional).
- L_admisión = min(max_leverage_del_spec, 20).
- N_requerido/L_admisión ≤ capital/max(K, número_de_forzados_únicos).

Se valida finitud/positividad de precio, tick, step, min_qty y resultados intermedios; min_notional debe ser finito no negativo y max_leverage positivo. Las fees deben ser finitas. Volumen negativo/no finito y cambio no finito no llegan al ranking. Una moneda no forzada inválida se excluye; una forzada inválida produce error. El filtro evita multiplicar capital por leverage para comparar nocionales y usa una división por el límite validado.

El resultado seleccionado no puede anunciar IDs que su bitmap de 64 bits no representa: si un seleccionado excede 63 se devuelve error. El universo global puede tener más elementos; la API concreta no finge representar todos ellos. La prueba de 65 candidatos elegidos verifica ese contrato. No se cambia la estructura global ni se declara que 64 sea un límite científicamente óptimo.

### Qué no se resolvió

La curva raíz de capital, el límite de política 20, los pisos de score y los dos scores históricos se conservan como compatibilidad, expresamente sin alegación de optimalidad. La fórmula mezcla magnitudes de distinta naturaleza con coeficientes fijos; no se convierte en Kelly, asignación de riesgo ni adaptación espectral por retirar el bono.

El filtro de nocional es necesario, no suficiente: no calcula redondeo al lote, margen ya ocupado, fees, reserva, límites por orden, liquidación ni cantidades de una cartera conjunta. Los forzados pueden saltarse la admisión presupuestaria porque representan permanencia, no permiso de ampliar exposición. Un consumidor necesita separar esa obligación de su gate de nuevas órdenes.

Las specs se consultan mediante APIs globales sin snapshot compuesto de versión. Validar sus números no prueba frescura ni una observación común del universo. La clonación de specs incluye String y búsqueda por símbolo; no se declara coste cero ni latencia nanosegundo. Ningún consumidor operativo de esta API se conectó en XVII.

Cierre de FMT-171: contrato por instrumento y epoch, separación vigilancia/gestión/admisión, política de capital con función objetivo validada y cantidades factibles verificadas antes del terminal. No sustituir sqrt(capital) por otra curva lisa elegida sin modelo.

## 4. FMT-172 — Retorno neto no es volatilidad; una banda suave sigue siendo una preferencia arbitraria

**Evidencia:** [scores heredados](<C:/Users/jhona/Documents/Proyectos/Trader Gemini/crates/quantum-arena/src/active_universe.rs:161>), [ranker de metadatos](<C:/Users/jhona/Documents/Proyectos/Trader Gemini/crates/quantum-arena/src/symbol_ranker_engine.rs:124>), [ranking de la ruta usada por el manager](<C:/Users/jhona/Documents/Proyectos/Trader Gemini/crates/data-pipeline/src/dynamic_ranker.rs:147>).

Varios selectores llaman volatility al valor absoluto de priceChangePercent de 24 horas. El recorrido 100→100→100 y el recorrido 100→150→100 tienen retorno neto cero, pero solo el segundo tiene variación realizada positiva. Un campo con esos endpoints no identifica volatilidad intraperiodo, dependencia espectral, liquidez ni riesgo de salto. El test del selector confirma que retorno cero excluye al activo de esa API, incluso con volumen elevado; no incorpora el recorrido, porque su interfaz ni siquiera lo recibe.

La fórmula s=ln(1+V)·x·exp(−x/15), x=abs(cambio_porcentual), tiene derivada de banda exp(−x/15)·(1−x/15). Su máximo está exactamente en x=15, por construcción. El 15 no se aprende en estos módulos. Que la curva sea suave no demuestra que ese retorno diario maximice utilidad, oportunidad ejecutable o relación riesgo/coste. El logaritmo además presupone una escala de referencia de volumen; cambiar unidades de V no conserva necesariamente las relaciones del ranking.

active_universe conserva dos recetas con coeficientes distintos para scalp_score/swing_score. La ruta estática usa su máximo y la dinámica solo scalp_score. No existe una integración explícita de utilidad sobre tau, pesos de soporte por escala ni incertidumbre asociada. No se renombraron como “universal” para ocultar que el cálculo sigue siendo el anterior.

### Penalidad de apalancamiento no monótona

SymbolRankerEngine usa b(x)=x·exp(−x/15) tanto para oportunidad como para penalización: p(x)=clip(b(x)/5,1,5), y L≈L_base/p(x). Como max b=15/e≈5,518, la penalidad máxima es aproximadamente 1,104; jamás alcanza cinco para x≥0 finito en aritmética exacta. En x=100 vuelve a uno. Así, una variación absoluta diaria mucho mayor puede recibir menos penalidad que x=15. El diagnóstico reproduce la expresión y sus extremos; no invoca la función de red privada ni acredita su despliegue.

Hay además imprecisión semántica: volume/trades es tamaño medio observado de operación, no profundidad de libro; trades/1440 es frecuencia media por minuto de esa ventana, no intensidad instantánea identificada. order_flow_imbalance se fija a cero en el ranker. Los nombres tensorial/omnisciente no añaden observables.

Cierre: distinguir retorno acumulado, variación realizada por escala, spread, profundidad, frecuencia y coste; declarar cada estimando y su disponibilidad. Separar función de oportunidad y restricción de riesgo. Ajustar o seleccionar sus parámetros sobre evidencia independiente y comparable, con benchmark y regularización. No extrapolar todos los horizontes a partir de un agregado de 24 horas.

## 5. FMT-173 — Un feed vacío no prueba que existan diez oportunidades

**Evidencia:** [DynamicSelector checked y publicación](<C:/Users/jhona/Documents/Proyectos/Trader Gemini/crates/data-ingest/src/dynamic_selector.rs:52>), [parser puro](<C:/Users/jhona/Documents/Proyectos/Trader Gemini/crates/data-ingest/src/dynamic_selector.rs:97>), [tests](<C:/Users/jhona/Documents/Proyectos/Trader Gemini/crates/data-ingest/tests/dynamic_selector_contract.rs>).

Antes, un array vacío, un objeto de error o una respuesta compuesta solo por valores inválidos acababan en una lista fija de diez símbolos. select_top_10 publicaba esa lista en el universo global. La ausencia de evidencia se transformaba en un aparente descubrimiento de mercado, mezclando mecanismo de arranque con resultado científico.

Los duplicados consumían varios slots y podían reducir la cardinalidad real del universo. Empates de score mantenían el orden del feed, por lo que invertir dos registros idénticos en métricas invertía sus posiciones. Dado que otros componentes usan posiciones como identidad, ese detalle no es solamente cosmético.

Se agrega SelectorError para volumen mínimo inválido, payload no array y duplicados contradictorios. Un array válido sin elegibles retorna Ok(vacío); el wrapper conserva Vec como compatibilidad pero devuelve vacío ante error. Duplicados numéricamente idénticos se deduplican; si un símbolo presenta dos observaciones válidas diferentes sin un orden temporal declarado, se rechaza el lote. No se escoge el máximo score como si significara dato más reciente.

El ranking conserva su fórmula heredada. Se usa ln_1p y se evalúa primero la banda acotada antes de multiplicar por log-volumen, evitando una forma intermedia desbordada. Los empates usan símbolo como desempate determinista; esto evita dependencia del orden de entrada, pero no elimina el sesgo que cualquier desempate determinista pueda tener al cortar top10.

La ruta HTTP exige estado de respuesta exitoso. Ante error de parsing o selección vacía retorna error y no publica un universo inventado ni vacía el anterior. Conservar la última publicación no demuestra que siga siendo fresca: faltan edad, estado degradado y criterios de expiración. La función pública sync_with_quantum_arena conserva su capacidad de reemplazar directamente el universo, por lo que el riesgo de identidad de FMT-174 no se cierra.

Cuatro tests iniciales fallaron y ahora pasan. Dos nuevos tests comprueban errores y duplicados conflictivos; otro caracteriza la pérdida del recorrido cuando el retorno diario es cero. No hubo llamadas HTTP en estas pruebas. Los filtros de formato, stablecoins, volumen por entorno y límite diez no se recalibraron ni se validaron como política universal.

## 6. FMT-174 — Rotar un ranking puede cambiar el dueño semántico de una posición

**Evidencia:** [universo mutable](<C:/Users/jhona/Documents/Proyectos/Trader Gemini/crates/quantum-arena/src/symbols.rs:8>), [lookup sobre universo actual](<C:/Users/jhona/Documents/Proyectos/Trader Gemini/crates/quantum-arena/src/symbol_registry.rs:99>), [preferencia del host por mapa inicial](<C:/Users/jhona/Documents/Proyectos/Trader Gemini/src/bin/god_engine.rs:3108>), [diagnóstico aislado](<C:/Users/jhona/Documents/Proyectos/Trader Gemini/crates/quantum-arena/tests/universe_identity_diagnostics.rs>).

El host crea symbol_to_id a partir de symbols_clone al arrancar y lo prefiere cuando procesa un símbolo conocido. Solo usa try_index dinámico si el símbolo no está en ese mapa. Por otra parte, try_spec/try_symbol/try_index interpretan el ID como posición del universo actualmente publicado. El manager puede cambiar tanto contenido como orden.

Reproducción mínima: arranque [A,B] asigna A→0/B→1. Publicar [C,A] cambia el lookup dinámico a C→0/A→1, pero el mapa inicial conserva A→0. Consultar try_symbol con el ID que el host utilizaría para A devuelve C. La prueba usa las funciones reales del registro/universo y un mapa equivalente al del host; no abre posiciones reales ni simula todo el event loop.

Las estructuras arena.coins[ID], feature_engines[ID], indicadores y otros vectores no se migran en la publicación observada. Cambiar la identidad de un slot puede mezclar historial, inferencia, margen y atribución; procesar un símbolo antiguo o un fill retrasado exige conservar su binding original. Un fallback dinámico para nuevos símbolos no repara esa propiedad. El comentario histórico D-725 solucionó una divergencia entre dos espacios de índice, pero el carácter mutable de la posición deja esta condición aún sin resolver.

No se cambió el host ni se desactivó un motor. Sustituir su lookup por uno siempre dinámico movería el problema: podría dirigir A al slot uno sin migrar allí su posición/features. El arreglo requiere un espacio estable de IDs y una vista de ranking independiente; o un protocolo de migración/versionado que cubra todos los consumidores y eventos en vuelo.

Cierre verificable: un instrumento conserva su ID durante toda la vida de posiciones, órdenes y eventos; el epoch acompaña la configuración; los nuevos instrumentos no heredan estado ajeno; los retirados siguen gestionables hasta quedar reconciliados. Probar permutación, entrada/salida de símbolos, fills retrasados, reinicio y rotación concurrente con inventario abierto. La prueba XVII confirma el mecanismo de lookup, no pérdidas financieras efectivamente ocurridas.

## 7. FMT-175 — Validación de órdenes que admite NaN y redondea hacia arriba

**Evidencia:** [SymbolSpec::validate_order](<C:/Users/jhona/Documents/Proyectos/Trader Gemini/crates/quantum-arena/src/symbol_registry.rs:20>) y los dos diagnósticos numéricos del archivo anterior.

Las comparaciones price≤0 y raw_qty≤0 no rechazan NaN. Luego, comparaciones con mínimos también resultan falsas y la API puede devolver Ok(NaN). Una cantidad normal con precio NaN también se admite como Ok(cantidad). No hay validación de finitud de los metadatos ni del nocional final. El test reproduce ambos casos.

El código calcula floor(raw_qty/step)·step y después infiere “decimales exactos” mediante round(−log10(step)). Esa inferencia no es válida para pasos decimales generales. Con step=0,05 y cantidad solicitada 0,06, el lote inferior es 0,05, pero redondear a un decimal lo convierte en 0,1: aumenta la cantidad por encima de lo autorizado. Con step=0,025 y raw=0,09, el segundo redondeo puede producir 0,08, que no pertenece a la malla de múltiplos de 0,025.

No se ha localizado consumidor operativo de esta función en la búsqueda actual. El hallazgo se clasifica como fallo de contrato de API, no como demostración de órdenes reales sobredimensionadas. Otras rutas de ejecución pueden aplicar sus propios redondeos; no se certifican por ausencia de caller de este helper.

Cierre: contrato decimal/lattice desde metadatos originales, representación de unidades enteras adecuada, validación de rango/finitud y demostración de q_ajustada≤q_solicitada para modo round-down. Separar precisión de visualización y paso negociable; cubrir pasos no potencia de diez, mínimos, overflow, subticks y serialización final. No se sustituye por otro epsilon ni se conecta este validador a la ruta operativa antes de resolver el contrato.

## 8. FMT-176 — Políticas de universo distintas y pruebas que no ejecutan el filtro real

**Evidencia:** [dynamic_ranker](<C:/Users/jhona/Documents/Proyectos/Trader Gemini/crates/data-pipeline/src/dynamic_ranker.rs:94>), [asset_selector](<C:/Users/jhona/Documents/Proyectos/Trader Gemini/crates/data-pipeline/src/asset_selector.rs:59>), [symbol_manager](<C:/Users/jhona/Documents/Proyectos/Trader Gemini/src/symbol_manager.rs:29>).

Los selectores no tienen el mismo contrato: asset_selector usa mínimo de volumen de un millón en ambos entornos y mínimo de cambio 0,1; dynamic_ranker usa volumen cero en testnet y un millón en producción; DynamicSelector usa 10.000 y un millón. Comparten una fórmula parecida, pero distinto universo admisible. Eso no acredita paridad de pruebas, demo y producción.

En dynamic_ranker, una lista denominada testnet_blacklist se aplica también cuando is_testnet=false. El filtro de base de tres caracteres y ASCII precede a consultar ExchangeInfo; no demuestra estado negociable ni calidad económica. Se documenta el alcance de la condición, sin afirmar aquí qué símbolos están actualmente listados por el exchange.

El test qo_u1a_sanitizer_rechaza_basura_testnet replica una versión del closure sin la blacklist. Por construcción no detectaría un error en esa lista ni prueba la función realmente usada por fetch_dynamic_universe. Algo similar ocurre con tests que recopian la fórmula de score en vez de llamar al helper operativo. Un test verde de una copia no protege una segunda implementación divergente.

Además, se trunca el ranking antes de validar que existan metadatos de instrumentos TRADING. Puede agotarse el límite con candidatos no utilizables sin rellenar huecos con los siguientes válidos. Parámetros de lote/tick/notional malformados o ausentes se sustituyen por constantes; max_leverage se asigna por grupos de símbolos y las fees por defaults. No se ha consultado una cuenta para validar esos valores; llamarlos oficiales no los convierte en observados.

Cierre: separar parsing, elegibilidad de instrumento y ranking; extraer funciones puras usadas tanto por HTTP como por tests; aplicar políticas de entorno explícitas/versionadas; validar metadatos antes de truncar; etiquetar fees/leverage como observados, configurados o desconocidos. No eliminar restricciones que tengan una función de seguridad sin sustituir su evidencia.

## 9. FMT-177 — Publicación incoherente y specs obsoletas aunque el ranking no cambie

**Evidencia:** [daemon y condición changed](<C:/Users/jhona/Documents/Proyectos/Trader Gemini/src/symbol_manager.rs:182>), [publicaciones](<C:/Users/jhona/Documents/Proyectos/Trader Gemini/src/symbol_manager.rs:211>), [registry merge](<C:/Users/jhona/Documents/Proyectos/Trader Gemini/crates/quantum-arena/src/symbol_registry.rs:62>).

El manager publica universo y specs en dos operaciones. Un lector puede observar la lista nueva con metadatos antiguos o ausentes. La ruta SymbolRankerEngine publica en el orden inverso; tampoco existe un epoch común. ArcSwap hace segura cada referencia, pero no transforma dos referencias y numerosos vectores consumidores en una transacción.

La actualización de specs está dentro de if changed, calculado solo con la lista de símbolos de configuración. Si un instrumento conserva asiento y cambia su step/min_notional, la nueva spec puede no publicarse en esta ruta. Comparar con configuración persistida, en lugar de verificar todo el estado publicado, tampoco acredita coherencia con el proceso. La información adquirida y descartada no debe contarse como refresco efectivo.

Los writes/renames de configuración y try_send de reconexión ignoran errores. Dos archivos pueden pertenecer a versiones distintas; la notificación puede no entregarse; aun así se imprime éxito. Un rename aislado reduce riesgo de archivo parcial, no acredita confirmación compuesta de configuración, registro y suscripción.

update_registry hace load→clone→store sin compare-and-swap del conjunto: escritores concurrentes pueden perder incorporaciones ajenas. Preservar el índice dentro de su vector no vuelve estable el índice del universo, que tiene otro contrato. El merge tampoco retira automáticamente specs obsoletas, por lo que su existencia no identifica vigencia.

Cierre: snapshot versionado de identidad y metadatos, publicación consistente, política explícita para eventos en vuelo y reintentos/ack de suscripción; refrescar specs independientemente de cambios de pertenencia. Persistencia debe registrar resultado real, recuperación y versión, no solo intención. No se ejecutó fault injection contra procesos operativos en XVII.

## 10. T40 — Selección online con disponibilidad, feedback y coste de transición

Se utilizó Firecrawl Research Index porque la mejora exige contrastar hipótesis, no buscar una fórmula más compleja. El CLI no estaba disponible y se usó el conector de investigación; no se enviaron fuentes privadas ni datos de cuenta. Se leyeron pasajes del cuerpo de dos trabajos y se amplió la familia mediante referencias relacionadas. No se implementó un nuevo algoritmo de inversión.

[Adversarial Online Learning with Changing Action Sets](https://arxiv.org/abs/2003.03490) distingue expertos con información de pérdidas de todas las acciones disponibles y bandits que solo observan la elegida. El conjunto disponible se revela antes de elegir; el estudio usa pérdidas binarias y un benchmark basado en rankings. Sus garantías aproximadas no equivalen a superar al mejor activo imaginable en cada instante ni a obtener beneficios. La disponibilidad de instrumentos debe registrarse como dato, no reconstruirse después de ver sus rendimientos.

[Optimal Comparator Adaptive Online Learning with Switching Cost](https://arxiv.org/abs/2205.06846) formaliza el coste del movimiento de decisiones. En su ejemplo de inversión, las decisiones son cantidades de acciones, las variaciones por acción están acotadas y el coste es por acción transada; la comparación es aditiva frente a buy-and-hold. El dominio permite cortos/margen y el ejemplo idealiza interés cero. No es directamente el contrato de comisiones proporcionales al nocional, funding y restricciones de futuros de este sistema.

### Traducción propuesta al grafo; requiere validación

Un diseño candidato separaría estado continuo estimado x_a(t,τ), disponibilidad observable A_t, restricciones de riesgo y decisión u_t. El universo para observar mercado, el conjunto que debe seguirse por posiciones abiertas y el conjunto admitido para nueva exposición no son la misma lista. FMT-171 muestra por qué un único vector puede mezclar obligaciones incompatibles.

La evaluación podría registrar pérdida predictiva o utilidad neta definida antes de seleccionar, más un coste de transición de cartera y de infraestructura. Ese coste debe tener unidades verificables: comisión, slippage, funding, pérdida de prioridad de cola y coste de suscripción no son intercambiables por una constante sin calibrar. La histéresis “top N+3” es una política, no una cota de ese coste.

Regret compara pérdidas de una política con un benchmark bajo un protocolo de información. Un benchmark que también pierde dinero permite regret pequeño y PnL negativo. Un óptimo retrospectivo que dispone de todas las oportunidades futuras no es la misma clase de comparadores. Para feedback parcial deben conservarse selección, observaciones y soporte; los resultados de operaciones elegidas no describen sin sesgo las acciones omitidas.

La propuesta exige replay causal con conjunto disponible as-of, identidad estable, costes homogéneos entre candidatos, manejo de feedback retrasado y evaluación independiente. T39 aporta el contrato de dependencia asíncrona; T40 añade el de selección/adaptación. No se obtiene una función objetivo de cartera por sumar indiscriminadamente ecuaciones de ambas familias.

### Familia relacionada, todavía solo abstracts en esta ronda

- [Online learning with feedback graphs and switching costs](https://arxiv.org/abs/1810.09666): información parcial y costes de cambiar acciones.
- [Non-stationary Online Learning with Memory and Non-stochastic Control](https://arxiv.org/abs/2102.03758): pérdidas dependientes de decisiones anteriores.
- [Adaptive Online Learning in Dynamic Environments](https://arxiv.org/abs/1810.10815): comparación con secuencias variables.
- [Parameter-free Dynamic Regret: Time-varying Movement Costs, Delayed Feedback, and Memory](https://arxiv.org/abs/2602.06902): candidato para feedback retrasado.
- [Unconstrained Dynamic Regret via Sparse Coding](https://arxiv.org/abs/2301.13349): complejidad del comparador y diccionarios.
- [Revisiting Smoothed Online Learning](https://arxiv.org/abs/2102.06933): distingue información disponible y criterios de evaluación.
- [Smoothed Online Convex Optimization Based on Discounted-Normal-Predictor](https://arxiv.org/abs/2205.00741): adaptación por intervalos y coste de cambio.

Se conserva esta familia como cola de lectura, sin copiar cotas de abstracts cuya notación puede estar degradada por extracción. Ningún resultado garantiza duplicar capital ni elimina restricciones de solvencia.

## 11. Teoría avanzada, espectros y límites de las analogías

El trabajo científico necesario no consiste en reemplazar todos los if por funciones suaves. Un precio inválido necesita un contrato de rechazo; una posición abierta necesita identidad estable; una incertidumbre estadística debe permanecer visible. Esos límites no son regímenes arbitrarios. Por otra parte, 15% diario, sqrt(capital)/2 o dos scores separados sí requieren una justificación que el código inspeccionado no proporciona.

Una representación multivariante continua puede indexarse por activo, tiempo, escala y observable, con dependencias entre activos. Debe declarar medida de integración sobre escalas, resolución efectiva, errores y condiciones de extrapolación. Un dato agregado de 24 horas no identifica todas sus componentes espectrales; consultar 1 ns o 100 años no crea evidencia a esas escalas.

No se introduce una teoría física o cuántica únicamente por analogía lingüística. Transferir una ecuación de otro ámbito exige estado y observable correspondientes, unidades, hipótesis, condiciones iniciales/de contorno, identificabilidad y ganancia frente a un baseline. Tampoco se deduce un modelo de mercado de las ecuaciones de un problema del milenio sin ese enlace. Esta ronda no demuestra ventaja cuántica, omnisciencia ni autoevolución productiva.

La autorregulación requiere distinguir errores de datos, cambios de soporte, cambios del proceso generador, restricciones de ejecución y feedback de la política. Adaptar un gen a una lista que silenciosamente cambia identidades optimiza un experimento incoherente. La reparación de esas raíces es parte del objetivo avanzado, no una tarea separada de la teoría.

## 12. Matriz de los ocho módulos y hoja de ruta

| Módulo histórico | Aporte XVII | Pendiente |
| --- | --- | --- |
| 1. Ingestión, parsers, L2 | Payload inválido, duplicados y contratos del universo | Frescura, metadatos completos y causalidad del feed |
| 2. IA y señales | Disponibilidad/feedback; roster se detecta por archivos en manager | Validez del modelo no probada por nombre de archivo; calibración conjunta |
| 3. Multiactivo y horizontes | Ocho archivos completos, retorno versus variación y scores duales | Campo espectral y ranking con objetivos identificados |
| 4. Ejecución | Diagnósticos de cantidad y atribución por símbolo | Validador decimal, lifecycle de órdenes e identidad terminal |
| 5. Riesgo/genoma | Admisión con límites de spec y errores explícitos | Asignación conjunta, costes reales y evaluación de genes sobre mismo tape |
| 6. Estado/telemetría/SO | Trazado de snapshots y errores de publicación | Epoch compuesto, pérdida de actualización y fallos de persistencia |
| 7. Confluencia/grafo | Identidad de raíz a terminal, vistas separadas | Migración de todos los consumidores sin estado heredado ajeno |
| 8. Backtesting/gobernanza | 35 tests y teoría T40 con hipótesis | Pruebas económicas independientes, p99 y resto del inventario |

Prioridad de rehabilitación:

1. Diseñar y probar IDs estables más epoch de configuración; no conectar otro algoritmo antes de saber a qué instrumento pertenece cada estado.
2. Separar observación, gestión de inventario existente y admisión; registrar motivo de no disponibilidad y su caducidad.
3. Sustituir los validadores de lote por un contrato numérico/decimal verificable con límites de cantidad.
4. Consolidar parsing/elegibilidad de rankings y probar la función real, no copias de closures; publicar specs aunque no rote la lista.
5. Identificar retorno, volatilidad, coste y dependencia por escala antes de aprender utilidad/riesgo; preservar incertidumbre.
6. Evaluar adaptación con feedback y comparadores explícitos sobre tapes exógenos comunes, con costes de transición.
7. Continuar los 165 Rust preexistentes pendientes y los demás archivos del proyecto sin inflar cobertura con búsquedas.

## 13. Pruebas, integridad y condiciones de reproducción

| Suite ejecutada offline | Cantidad | Interpretación |
| --- | ---: | --- |
| quantum-arena, universe_selection_contract | 12 | Ocho rojo→verde y cuatro refuerzos |
| quantum-arena, universe_identity_diagnostics | 4 | Fallos abiertos: binding, finitud, lattice y penalidad de leverage |
| data-ingest, dynamic_selector_contract | 7 | Cuatro rojo→verde, dos refuerzos y una limitación abierta |
| quantum-arena --lib active_universe | 6 | Tests existentes; capital cero ahora no admite |
| data-ingest --lib dynamic_selector | 6 | Tests existentes; vacío ahora abstiene |
| Total | 35 | 23 nuevos, doce rojo→verde, seis refuerzos y cinco diagnósticos abiertos |

```powershell
cargo test -p quantum-arena --test universe_selection_contract --test universe_identity_diagnostics --offline -- --test-threads=1
cargo test -p data-ingest --test dynamic_selector_contract --offline -- --test-threads=1
cargo test -p quantum-arena --lib active_universe --offline -- --test-threads=1
cargo test -p data-ingest --lib dynamic_selector --offline -- --test-threads=1
cargo check --bin god_engine --offline
```

Los tests nuevos que manipulan el registro en universe_selection_contract usan mutex propio para aislar casos. Las suites se ejecutaron en modo serial para evitar interacción entre los tests preexistentes de estado global. Los diagnósticos de FMT-175 usan el helper real; el de la penalidad de leverage reproduce la expresión y no finge llamar al ranker de red.

El primer intento de rustfmt encontró temporalmente archivos con una sección mapeada abierta en Windows (error 1224). Se verificó el contenido y se reintentó sin detener procesos ni restaurar archivos. Cargo tuvo espera transitoria de lock. El formato final y git diff --check sobre fuentes propias pasan. Check mantiene tres warnings existentes de evolution-engine: latest_ts, mode y RealWfOutcome.trades.

El [artefacto XVII](<C:/Users/jhona/Documents/Proyectos/Trader Gemini/docs/artifacts/auditoria_fundamentos_XVII_2026-09-24.json>) registra mecanismos, estados, tests, fuentes primarias y hashes. Se capturaron doce hashes iniciales de fuentes: cambian solo los dos archivos intervenidos; los otros diez permanecen protegidos. La cobertura nueva se acredita por lectura completa de los ocho archivos indicados, no por los fragmentos del host ni por interfaces adicionales leídas parcialmente.

Se añaden adendas al maestro, atlas y XVI, conservando el contenido histórico. No hubo commit, push, fetch, merge, despliegue, reinicio, operación de cuenta ni escritura intencional de genomas activos. Main/59a76de4 describe el checkout local; no se verificó el remoto. La auditoría global y la migración a un motor verdaderamente continuo/autoadaptativo siguen abiertas.

### Control final de integridad

JSON parseado con siete hallazgos y siete IDs nuevos. Coinciden los quince hashes del inventario y las diez fuentes protegidas mantienen sus hashes iniciales. Los veintiún enlaces locales del informe resuelven; sus referencias de línea están dentro del archivo correspondiente. Los prefijos completos del atlas, maestro y XVI conservan su SHA-256 normalizado CRLF→LF; se agregaron respectivamente 2.098, 5.355 y 1.141 caracteres. El NUL histórico del maestro permanece sin limpieza. El primer intento de adenda al atlas informó fallo de escritura: se inspeccionó el final real del archivo antes de reintentar, sin duplicar secciones ni restaurar el árbol. La apertura del informe puede quedar en cola en la aplicación; no equivale a validación científica.

## Continuación XVIII — estado posterior, sin reescribir XVII

El [informe XVIII](<C:/Users/jhona/Documents/Proyectos/Trader Gemini/docs/AUDITORIA_FUNDAMENTOS_CIENTIFICOS_XVIII_2026-09-24.md>) amplía FMT-175: dominio finito y proyección conservadora minQty+n·step reparados, pero aritmética decimal/serialización integral pendientes. Dos diagnósticos de cantidades de XVII pasan a ser regresiones; los diagnósticos de identidad y penalidad del ranker permanecen abiertos.

Se añaden FMT-178–184 sobre números no finitos, ACK vacío, ejecución incierta, moneda/signo de comisiones, otro selector no operativo localizado, códigos de error y reconciliación por pierna. Esta última reproduce neteo de hedge a plano y dirección local obsoleta tras inversión; no se ha reparado. Cincuenta pruebas distintas pasan, con veintitrés nuevas; check sin ejecutar motor pasa. Cuatro lecturas nuevas elevan cobertura a128/289 Rust;161 pendientes. Véase el [artefacto XVIII](<C:/Users/jhona/Documents/Proyectos/Trader Gemini/docs/artifacts/auditoria_fundamentos_XVIII_2026-09-24.json>) para mecanismos, estados y evidencia. No hubo publicación Git, operación de cuenta ni promoción de genoma.
