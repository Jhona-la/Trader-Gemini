# Auditoría de fundamentos científicos XXXI — consejo, vetos y atribución del aprendizaje

Fecha: 2026-09-25. Estado: reparaciones locales verificadas, no certificación sistémica. Continuación de [XXX](<C:/Users/jhona/Documents/Proyectos/Trader Gemini/docs/AUDITORIA_FUNDAMENTOS_CIENTIFICOS_XXX_2026-09-25.md>).

[Artefacto estructurado XXXI](<C:/Users/jhona/Documents/Proyectos/Trader Gemini/docs/artifacts/auditoria_fundamentos_XXXI_2026-09-25.json>).

## 1. Dictamen y alcance real

El consejo de decisión tenía fallos comprobables que afectaban admisión, semántica de veto, dirección de señal y actualización de pesos. Se corrigen seis familias de contratos: validación de evidencia/configuración; alineación direccional de una excepción; aplicación del umbral configurable de cascada; referencia ML consistente; abstenciones sin votos ficticios; atribución del resultado según el lado ejecutado.

También se demuestran tres deudas de diseño que siguen abiertas: consenso aparentemente total a partir de una sola fuente direccional; mezcla de win-rate de un activo con recuento agregado, además de extracción y deliberación con contextos distintos; y compresión de horizontes extremos a los mismos extremos de la banda 30s–12h.

No se eliminan los cortacircuitos de seguridad para aumentar aceptación. No se añade una teoría por prestigio ni se presenta como solución un cambio cosmético de nombres. La matemática debe describir el estimando que realmente recibe el código; la física o una analogía cuántica no proporcionan datos ni identidad de ejecución ausentes.

Cobertura acumulada: **146/289 archivos Rust preexistentes leídos íntegramente; 143 pendientes**. Se añade consejo_seniors.rs, 1.521 líneas antes de esta intervención, 1.614 en el snapshot final. Las lecturas dirigidas del core y de posición no se cuentan como nuevas lecturas completas. Los tests nuevos no inflan el inventario histórico de 1.119 archivos versionados y 24 manifiestos Cargo.

Verificación seleccionada: **41 pases distintos = 37 funcionales/compatibilidad + 4 diagnósticos de deuda abierta**, sin ignorados. Son 23 tests nuevos: 20 funcionales y tres diagnósticos. Trece contratos pasaron de una aserción fallida en el código anterior a pasar después de la corrección: doce del consejo y uno sobre el cierre real del core con estado sintético.

No se certifica rentabilidad, causalidad completa del aprendizaje, independencia del ensamble, ausencia de bugs, operación a cada nanosegundo ni una capacidad predictiva de cien años. La auditoría de todos los archivos sigue incompleta.

## 2. Grafo vivo y topología de esta revisión

```text
RAÍZ: activo + libro + modelo + espectro + métricas + evidencia de cuenta
 │       [procedencia/freshness no representadas íntegramente en el payload]
 ▼
MarketSnapshotPayload ──► admisión de dominio y política         FMT-237
 │
 ├─ OBI ──────────────────► Microestructura
 ├─ score + persistencia ─► SeriesTemporales
 ├─ probabilidad/base ────► ML
 ├─ OBI + espectro + ML ──► Metacognitivo   [reutiliza raíces]
 ├─ espectro + ML + H ────► Teleonomia     [reutiliza raíces]     FMT-240/244
 └─ riesgo/impacto/cascada ► permisos y moduladores               FMT-238/239
                            │
                            ▼
                  NODO DE DECISIÓN
           capacidad ponderada + señal + política de veto
                            │
               comprobación del lado por el core
                            ▼
              intención / posición / ejecución
                            │
           estimación de cierre o liquidación confirmada       FMT-225 abierto
                            ▼
                atribución por lado ejecutado                  FMT-242
                            │
           exclusión de abstenciones y moduladores              FMT-241
                            ▼
                 tracker agregado → pesos
                            │
                n global + wr del activo                       FMT-243 abierto
                            └──────────► próxima decisión

Eje temporal del consejo:
τ positivo → log(τ/30s) / log(12h/30s) → clamp[0,1]               FMT-245 abierto
```

El consejo está conectado al núcleo: se construye en GodEngineCore, recibe el snapshot de entrada y el núcleo consulta su aprobación/dirección. A diferencia del PositionLedger auxiliar de XXX, aquí sí se identifica una ruta operativa de consumo. Esto no prueba que cada condición defectuosa se haya producido en una cuenta real.

La salida approved del consejo no basta por sí sola para describir qué orden ejecutar. El core ya comprueba que final_signal tenga el lado de la orden: esa protección previa se conserva. No se documenta de nuevo como bug actual una discrepancia que ese consumidor ya contiene.

## 3. Matriz incremental de resolución

| ID | Prioridad | Contrato | Estado |
|---|---|---|---|
| FMT-237 | P1 | Evidencia y parámetros inválidos podían convertirse en permiso | Reparado en admisión de deliberación; alcance auxiliar limitado |
| FMT-238 | P1 | Excepción de breakout aceptaba libro contrario | Reparación local de alineación |
| FMT-239 | P2 | Parámetro de cascada no activaba el veto bajo el literal antiguo | Reparado en el consejo configurado; política por defecto conservada |
| FMT-240 | P1 | Teleonomia centraba ML en 0,5 aunque el modelo tuviese otra base | Referencia corregida; utilidad/calibración abiertas |
| FMT-241 | P1 | Cero/abstención aprendían como voto de permiso acertado | Contado y expulsión de ventana corregidos |
| FMT-242 | P1 | PnL de corto se comparaba con dirección de mercado | Consumidor de cierre pasa lado explícito; atribución local corregida |
| FMT-243 | P1 | Recuento, win-rate y señales guardadas no comparten procedencia | Abierto y reproducido; explicación bayesiana corregida |
| FMT-244 | P1 | Cociente de consenso confundido con número de fuentes independientes | Abierto y reproducido; documentación acotada |
| FMT-245 | P2 | Una coordenada saturada y una escala dominante sustituyen al espectro | Abierto y reproducido |

Los nuevos IDs se añaden a la serie FMT. No sustituyen la matriz histórica de 305 puntos, ni su numeración representa un conteo deduplicado de todos los bugs del proyecto. P1/P2 son prioridades de ingeniería, no incidentes financieros observados. FMT-239 requiere variar la configuración respecto al valor por defecto: no se atribuye ese caso al despliegue actual.

## 4. FMT-237 — validación incompleta y configuración inválida interpretada como política

**Evidencia:** [validación del payload](<C:/Users/jhona/Documents/Proyectos/Trader Gemini/crates/metacortex-engine/src/consejo_seniors.rs:116>), [CouncilParams](<C:/Users/jhona/Documents/Proyectos/Trader Gemini/crates/metacortex-engine/src/consejo_seniors.rs:794>), [contratos de datos](<C:/Users/jhona/Documents/Proyectos/Trader Gemini/crates/metacortex-engine/tests/council_evidence_contract.rs:31>) y [multiplicadores](<C:/Users/jhona/Documents/Proyectos/Trader Gemini/crates/metacortex-engine/tests/council_evidence_contract.rs:102>).

**Mecanismo.** validate examinaba sólo parte de los campos. No validaba finitud de umbral causal, τ dominante, burst, cascada, OI, spoofing, ratios de crowd o base ML. Otros campos tenían finitud pero no dominio. Los scorers aplicaban clamps, defaults o comparaciones que podían transformar lo desconocido en neutralidad o permiso. Una base NaN y un umbral NaN no son decisiones económicas calibradas.

CouncilParams podía contener umbral negativo o concentración k inválida; los multiplicadores externos NaN, infinitos, negativos o cero se ignoraban, recuperando implícitamente el peso por defecto. El usuario no podía distinguir una configuración no aplicada de una política realmente ejecutada.

**Reproducción.** Antes del cambio, una prueba mostró que causal_veto_threshold=NaN pasaba validate; otra que book_imbalance=1,1 se aceptaba. Un approval_threshold negativo permitía una decisión y un multiplicador NaN no la rechazaba. Los bucles de prueba contienen más casos, pero antes del fix se detenían en su primera aserción fallida: no se afirma que todos los elementos del barrido fueran observados individualmente en rojo.

**Cambio.** La deliberación valida primero payload y parámetros. Se comprueba finitud de todos sus escalares flotantes y los dominios declarados: probabilidades/índices normalizados, scores firmados, τ>0, costes/ratios no negativos. Se rechaza win-rate fuera de [0,1] y multiplicador externo no finito o no positivo. Los parámetros de umbral/penalización están en [0,1]; k es finito y positivo para el modo simétrico existente. No se interpreta k=0 como un modo sin prior que no está especificado.

La respuesta conserva approved=false y una justificación de fallo de integridad. Se reutiliza el formato de resultado existente; no se introduce todavía un enum de causas separado de SeniorRole. En particular, un rechazo por integridad atribuido a Riesgo no equivale a una opinión económica de ese asiento.

**Lo que no cambia.** Un τ positivo de 1ns o cien años no se rechaza por estar fuera de la banda legacy; su dominio es válido, aunque la representación siga saturando. Las condiciones de mercado válidas no se etiquetan como corrupción. Se mantiene el cortacircuitos de cascada para severidad alta válida.

**Residual.** No hay mask de observabilidad, antigüedad, cuenta/venue, generación del modelo o identidad causal dentro del payload. El productor todavía usa valores por defecto para algunas fuentes ausentes. Los scorers individuales y extract_senior_signals son APIs auxiliares que no heredan automáticamente todo el gate de deliberación; no se certifica cada entrada pública ni cada vector de agentes personalizado. El tracker rechaza una muestra numéricamente inválida, pero todavía no devuelve un motivo durable al productor.

**Aceptación ampliada.** Datos tipados con procedencia y freshness; estados Missing/Invalid/Observed distintos; razones estructuradas; contrato explícito de extensibilidad del conjunto de agentes; pruebas de rutas de construcción del payload. Las guardas actuales contienen dominios inválidos, no detectan un número finito que ya fue imputado incorrectamente.

## 5. FMT-238 — magnitud del libro confundida con alineación

**Evidencia:** [SeniorCausal](<C:/Users/jhona/Documents/Proyectos/Trader Gemini/crates/metacortex-engine/src/consejo_seniors.rs:505>) y [regresión simétrica](<C:/Users/jhona/Documents/Proyectos/Trader Gemini/crates/metacortex-engine/tests/council_evidence_contract.rs:113>).

**Mecanismo.** La excepción declarada como breakout alineado requería |OBI|>0,25, coordenada temporal rápida y riesgo <0,88. La magnitud no distingue presión compradora de vendedora. Un libro fuerte contrario al lado propuesto podía desactivar un veto causal que de otro modo correspondía aplicar.

**Caso reproducido.** Con riesgo 0,8, umbral 0,75 y τ=30.000ms, un OBI de signo contrario a intended_direction obtenía is_veto=false. El test recorre ambos lados. La dirección neutra también debe impedir afirmar alineación.

**Corrección.** Se añade OBI·intended_direction>0. Los demás criterios permanecen para no presentar como calibrada una política nueva. Si libro e intención están alineados, la excepción legacy sigue disponible; si son opuestos o la intención es cero, ya no se concede esa excepción.

**Importancia y límite.** Se repara un error de signo que aumentaba permisos, no se garantiza la bondad del breakout. No se ha demostrado que 0,25, 0,88, el corte spectral_s<0,5 o el clamp causal [0,60,0,90] sean óptimos. La posibilidad de override posterior del consejo permanece; esta prueba acredita el scorer y su excepción, no una prohibición económica global de toda operación con esos valores.

**Ciencia pendiente.** El nombre do_calculus_risk recibe VPIN en el core. En esta ruta no se identifica un grafo causal, una intervención do(X=x) ni una identificación de efectos causales. El resultado debe tratarse como score de toxicidad usado heurísticamente. Cambiar la etiqueta tampoco calibraría el veto: hacen falta target, costes, falsos positivos, sensibilidad y validación fuera de muestra.

## 6. FMT-239 — parámetro de cascada desconectado de la activación

**Evidencia:** [aplicación efectiva del parámetro](<C:/Users/jhona/Documents/Proyectos/Trader Gemini/crates/metacortex-engine/src/consejo_seniors.rs:952>) y [reproducción](<C:/Users/jhona/Documents/Proyectos/Trader Gemini/crates/metacortex-engine/tests/council_evidence_contract.rs:128>).

**Mecanismo.** El asiento EnteMercado activaba is_veto con el literal 0,85. Más adelante el consejo consultaba cascade_severity_breaker, pero sólo dentro de una rama a la que se llegaba si existía un veto. Bajar el parámetro a 0,5 no provocaba veto para severidad 0,6: el productor seguía usando 0,85.

**Consecuencia.** Un parámetro expuesto como control de seguridad no gobernaba de forma consistente el inicio de esa protección. Es una desconexión config→comportamiento, análoga al problema gen→consumidor. No se midió un impacto real ni se afirma que el despliegue usase 0,5.

**Cambio.** En deliberación, el umbral configurado determina tanto activación de ese veto como la condición que impide sobreescribirlo por supermayoría. La justificación contiene severidad, límite aplicado y estado efectivo, sin mantener una etiqueta de veto que correspondiera sólo al valor por defecto. El scorer standalone conserva su default legacy: el contrato configurable es el del consejo.

**Pruebas.** Con límite 0,5 y severidad 0,6 se rechaza con EnteMercado como causa. El caso de frontera conserva la semántica estricta “>”: 0,5 no activa el breaker por sí solo, 0,500001 sí. Un mercado válido con severidad 0,99 sigue siendo rechazado por política, no por integridad.

**Residual.** 0,85 no queda demostrado como P99 por un comentario. No hay en este cambio calibración empírica, distribución condicionada por activo o horizonte, ni política de incertidumbre de la severidad. La configuración sigue siendo mutable en la API pública; no se añade versionado transaccional ni actualización en caliente segura.

## 7. FMT-240 — contradicción entre referencias ML del mismo consejo

**Evidencia:** [componente ML de Teleonomia](<C:/Users/jhona/Documents/Proyectos/Trader Gemini/crates/metacortex-engine/src/consejo_seniors.rs:707>), [neutralidad en base](<C:/Users/jhona/Documents/Proyectos/Trader Gemini/crates/metacortex-engine/tests/council_evidence_contract.rs:139>) e [invariancia de traslación](<C:/Users/jhona/Documents/Proyectos/Trader Gemini/crates/metacortex-engine/tests/council_evidence_contract.rs:148>).

**Mecanismo.** ML y Metacognitivo comparaban la probabilidad con ml_model_base. Teleonomia seguía usando 2(p−0,5). Para un modelo cuya base es 0,3, observar p=0,3 no contiene lift respecto a su referencia, pero Teleonomia generaba contribución bajista.

**Reproducción.** Con contribución espectral cero y p=base=0,3, el scorer anterior devolvía −0,15444800000000003. No es una pérdida económica medida: es la salida del score del fixture. Tras corregir la referencia devuelve cero.

**Corrección.** Se usa 2(p−base) y se conserva la escala/peso posterior de la fórmula para aislar el defecto de referencia. Una traslación común de p y base, dentro de sus dominios, deja igual el componente ML. La regresión comprueba esa invariancia. Se mantiene la compatibilidad en base 0,5.

**Interpretación.** Lift respecto a la base no es por sí mismo expectativa monetaria, probabilidad de ganar una orden concreta ni prueba de dirección óptima. La definición del label del modelo y sus costes debe acompañar al modelo. Si p predice un evento condicionado que no es la dirección del mercado, centrarlo correctamente no resuelve esa discrepancia de target.

**Residual.** La fórmula suma score espectral y lift con 0,6/0,4, usa |H−0,5| como predictibilidad, multiplica penalidades y lo denomina expected_utility. No integra distribución de payoff, costes/fills realizables ni preferencias de utilidad. Tampoco los distintos dead zones/normalizaciones de los asientos quedan armonizados. Es un score heurístico referenciado consistentemente, no una utilidad económica certificada.

## 8. FMT-241 — abstenciones y moduladores convertidos en evidencia de éxito

**Evidencia:** [tracker](<C:/Users/jhona/Documents/Proyectos/Trader Gemini/crates/metacortex-engine/src/consejo_seniors.rs:1212>), [prueba de abstención](<C:/Users/jhona/Documents/Proyectos/Trader Gemini/crates/metacortex-engine/tests/council_evidence_contract.rs:160>), [máscara](<C:/Users/jhona/Documents/Proyectos/Trader Gemini/crates/metacortex-engine/tests/council_evidence_contract.rs:168>).

**Mecanismo.** record_outcome del consejo sustituía por cero las señales de Volatilidad, Riesgo y EnteMercado, explicando que así no entrenaban. El tracker no implementaba esa semántica: incrementaba total_counts para los once asientos y consideraba un cero correcto si el retorno de la operación era positivo. El comentario y el algoritmo se contradecían.

**Reproducción.** Una muestra con once ceros elevaba total_counts a once unos. Diez operaciones ganadoras dejaban diez ensayos en los asientos enmascarados. Tras expulsar de la ventana un voto real y conservar sólo abstenciones, los contadores seguían contabilizando esas abstenciones. Son contadores de fixtures, no métricas de ejecución real.

**Cambio.** Sólo una señal no nula ingresa como ensayo direccional. Al retirar una muestra vieja se aplica la misma condición antes de decrementar. Los moduladores enmascarados permanecen sin ensayos direccionales y sus pesos adaptativos quedan neutros en el contrato probado. Un vector de señales no finito o fuera de [-1,1] no contamina el tracker.

**Por qué la simetría de ventana importa.** Un filtro de inserción sin el mismo criterio de expulsión puede restar ensayos que nunca se sumaron, deformando tasas. La prueba llena la ventana y fuerza expulsión, no verifica únicamente el primer incremento.

**Residual.** El histórico de cierres sigue contando muestras aunque todos se abstengan, porque su longitud se usa como cantidad de outcomes; eso no es lo mismo que cantidad de ensayos de un asiento. La máscara está indexada por posición de rol en el vector, y la API pública permite personalizar agentes: el orden/identidad/extensión de una topología dinámica requiere un contrato adicional. No se ha migrado estado persistido ni introducido aprendizaje del valor de un veto: evaluar rechazos necesitaría outcomes observables o contrafactuales identificables, no atribuir automáticamente el beneficio de una entrada a un permiso.

## 9. FMT-242 — rentabilidad de la operación y signo de mercado en espacios distintos

**Evidencia:** [nuevo contrato con lado](<C:/Users/jhona/Documents/Proyectos/Trader Gemini/crates/metacortex-engine/src/consejo_seniors.rs:1157>), [consumidor del cierre](<C:/Users/jhona/Documents/Proyectos/Trader Gemini/crates/god-engine-core/src/lib.rs:1880>) y [regresión sobre core](<C:/Users/jhona/Documents/Proyectos/Trader Gemini/crates/god-engine-core/tests/close_outcome_contract.rs:365>).

**Mecanismo.** El caller pasaba retorno neto de la operación. El tracker comparaba signo(signal) con signo(return). Un corto ganador tiene retorno neto positivo, aunque la predicción de dirección acertada fuese negativa. La regla premiaba la señal alcista por ese resultado y penalizaba la bajista.

**Caso reproducido.** Un fixture de core abre estado propio, configura posición corta y señales guardadas de signo contrario en dos asientos, y produce un cierre ganador. Antes, correct_counts del asiento bajista quedaba cero en vez de uno. Tras el cambio, cuenta uno para acuerdo bajista y cero para el voto opuesto.

La primera construcción del fixture asignó señales al slot 0, mientras positions.position corresponde al slot 2. Se corrigió esa preparación y se repitió el caso antes de cambiar el consumidor. La reproducción válida es la segunda corrida sobre el slot correcto; la primera no se usa para demostrar el bug de orientación.

**Corrección y significado.** Se añade record_trade_outcome(signals, net_return, is_long). Si d=+1 para largo y d=−1 para corto, se transforma a_i=d·s_i y se evalúa el acuerdo con la acción ejecutada frente al retorno de esa acción. Esto conserva la simetría largo/corto. No se sustituye r_net por una supuesta rentabilidad de mercado: comisiones, slippage y financiación hacen que esa conversión sea conceptualmente incorrecta.

El adaptador legacy permanece para datos que ya compartan orientación; el caller operativo identificado usa el contrato nuevo. Se prueban largos/cortos y ganancias/pérdidas de forma simétrica.

**Residual crítico.** La regla es atribución de acuerdo sobre operaciones seleccionadas, no exactitud predictiva contrafactual. Si un corto pierde por costes aunque baje el precio, su señal direccional de precio puede haber sido correcta y su acción económica no. Un desacuerdo premiado ante pérdida no demuestra que la operación opuesta hubiese ganado. Los IDs FMT-225/052 siguen abiertos: estimaciones de cierre, features y fill real no son intercambiables.

También sigue pendiente guardar un trace único de la deliberación efectivamente autorizada; recuperar un array por slot sin una identidad integral de decisión/fill/generación no garantiza atribución entre concurrencia, rechazos y reinicios.

## 10. FMT-243 — shrinkage con población incompatible y doble evaluación

**Evidencia:** [n agregado y mezcla](<C:/Users/jhona/Documents/Proyectos/Trader Gemini/crates/metacortex-engine/src/consejo_seniors.rs:925>), [extracción de señales](<C:/Users/jhona/Documents/Proyectos/Trader Gemini/crates/metacortex-engine/src/consejo_seniors.rs:1138>), [diagnóstico](<C:/Users/jhona/Documents/Proyectos/Trader Gemini/crates/metacortex-engine/tests/council_evidence_contract.rs:342>).

**Mecanismo.** El core pasa win-rate desde las métricas del activo. La deliberación toma n de la longitud del tracker del consejo compartido. El payload no incluye un asset_id con el que seleccionar esa evidencia. La historia agregada puede crecer por otros activos y cambiar cuánto se confía en un win-rate que no procede de esos mismos ensayos.

**Derivación correcta, uso condicionado.** Para prior Beta(α,β) y s éxitos de n ensayos Bernoulli de la misma población, el posterior es Beta(α+s,β+n−s), con media (s+α)/(n+α+β). Si α=β=k/2 y s=n·wr, se obtiene:

```text
wr_shrunk = (n·wr + k/2) / (n+k)
```

k=8 corresponde a Beta(4,4), no Beta(1,1), que tendría concentración dos. La documentación del código se corrige. La derivación es propia a partir de la densidad beta y su parametrización media/concentración de [Stan](https://mc-stan.org/docs/functions-reference/continuous_distributions_on_0_1.html#beta-distribution). No se atribuye a la fuente una validación del software ni de su muestra.

**Diagnóstico.** Un scorer sintético que expone el wr recibido obtiene 0,5 con n=0. Tras diez outcomes agregados, con el mismo wr=0,1 y sin identidad de activo, recibe (1+4)/(10+8)=5/18≈0,2777778. El resultado prueba sensibilidad al recuento agregado; su improcedencia como posterior por activo se deriva además del caller real que suministra wr desde coin.metrics.

**Segunda desconexión.** extract_senior_signals evalúa con wr crudo; deliberar evalúa con wr_shrunk. El mismo diagnóstico obtiene 0,1 en extracción y 5/18 en deliberación. Guardar el primero no es congelar el segundo. Dos evaluaciones pueden divergir por estado, política y tiempo aunque compartan el payload. La corrección de orientación del cierre no corrige esa discrepancia.

**Reparación pendiente.** Elegir el estimando: desempeño de cada asiento por activo/escala, utilidad de la acción conjunta o tasa global. Conservar suficiente estadístico y soporte del mismo conjunto de eventos; producir decisión y trace en una sola evaluación; persistir identificadores de evento, muestra y generación. Si se usa pooling multiactivo, debe ser un modelo jerárquico explícito y validado, no un n global prestado.

**Criterio de cierre.** Un evento del activo B no modifica la evidencia local de A salvo a través de un mecanismo de pooling documentado; la muestra de cierre corresponde exactamente al trace autorizado; las pruebas distinguen ventana, pesos, censura, duplicados y ausencia de datos. No se cambió la mezcla numérica ni el pooling en XXXI.

## 11. FMT-244 — capacidad ponderada no equivale a fuentes independientes

**Evidencia:** [denominador del consenso](<C:/Users/jhona/Documents/Proyectos/Trader Gemini/crates/metacortex-engine/src/consejo_seniors.rs:1005>) y [caso de una raíz direccional](<C:/Users/jhona/Documents/Proyectos/Trader Gemini/crates/metacortex-engine/tests/council_evidence_contract.rs:294>).

**Mecanismo.** El cociente suma capacidad w_i·c_i de asientos con señal del lado elegido y divide por capacidad total de asientos direccionales. Las abstenciones con c_i=0 no contribuyen al denominador. Por ello un conjunto pequeño de voces activas alineadas puede alcanzar uno. No representa cuatro de cinco opiniones ni dos o tres fuentes independientes por aplicar 0,35.

Además, Metacognitivo reutiliza OBI/ML/espectro y Teleonomia reutiliza ML/espectro/Hurst. Diferentes funciones y responsabilidades no demuestran independencia estadística. Los moduladores heredan la dirección de la entrada y, aunque no cuenten en el ratio de consenso, sí influyen en final_signal, usado para habilitar override.

**Reproducción.** Se dejan ML neutral respecto a base, score espectral cero y Hurst 0,5. Se mantiene sólo OBI como fuente direccional bruta. El consejo aprueba y total_consensus_pct=1,0; la voz metacognitiva es derivada del mismo OBI. El diagnóstico pasa porque el problema permanece.

**Alcance.** No se demuestra que cada aprobación de una sola fuente sea económicamente incorrecta: lo incorrecto es afirmar que el ratio acredita pluralidad independiente y usarlo como tal para justificar autoridad. Una estrategia puede legítimamente actuar con una fuente si incertidumbre, costes y mandato lo permiten; debe declararlo y evaluarlo con ese soporte real.

**Cambios de documentación.** Se aclara que el umbral es heurístico, que el ratio no impone un mínimo de fuentes y que multiplicar por 0,75 después de un override no es una actualización bayesiana derivada. No se cambiaron arbitrariamente los pesos o el umbral para producir más o menos operaciones.

**Aceptación pendiente.** Grafo de procedencia por raíz, evaluación de dependencia/ablación y un criterio económico explícito de override. Las restricciones de datos/seguridad no deben sobreescribirse por una suma de señales correlacionadas. La política necesita distinguir veto de integridad, límite mandatado, coste esperado y abstención por falta de habilidad.

## 12. FMT-245 — continuo acotado no es cobertura temporal universal

**Evidencia:** [spectral_s](<C:/Users/jhona/Documents/Proyectos/Trader Gemini/crates/metacortex-engine/src/consejo_seniors.rs:104>) y [diagnóstico de saturación](<C:/Users/jhona/Documents/Proyectos/Trader Gemini/crates/metacortex-engine/tests/council_evidence_contract.rs:312>).

La coordenada es s=clamp(log(τ/30s)/log(12h/30s),0,1). Dentro de la banda es continua; fuera, toda escala inferior se vuelve cero y toda superior uno. El test muestra que 1ns y 30s dan el mismo s; 12h y cien años dan el mismo s. Esto se refiere a esta coordenada del consejo, no a una afirmación sobre todos los cálculos del proyecto.

Riesgo usa un límite 0,95−0,10s y Ejecución usa 35+65s bps. Al variar τ fuera de banda, esos límites no cambian. El asiento causal conserva una excepción con s<0,5. El payload además resume con τ dominante, fused_score y persistence: no transporta la distribución completa sobre escalas, sus incertidumbres ni una matriz espectral multiactivo.

Sustituir los enums de scalping/swing por Continuous no elimina estas restricciones. Tampoco basta con cambiar 30s por 1ns y 12h por cien años: eso desplazaría todas las políticas sin datos, error numérico ni justificación económica.

**Rediseño pendiente.** Separar dominio matemático τ>0, soporte identificable de datos, discretización de cálculo, memoria efectiva, cadencia de actualización y horizonte económico. Puede representarse una función sobre log τ con nodos adaptativos y error de aproximación medido; no supone procesar cada nanosegundo ni pronosticar donde no hay soporte. La asignación de riesgo necesita unidades y costes, no una indulgencia de drawdown derivada únicamente del horizonte.

**Aceptación.** Continuidad y sensibilidad donde hay soporte; comportamiento explícito fuera de soporte; invariancia de unidades; incertidumbre creciente ante extrapolación; comparación por replay de la misma representación entre backtest/demo/prod. Esta ronda conserva la curva legacy y marca su límite, sin inventar calibración universal.

## 13. Vetos, rechazos y rigidez: interpretación por clase

| Clase | Ejemplo auditado | Tratamiento |
|---|---|---|
| Integridad | NaN, ratio negativo, τ no positivo | Rechazo explícito; no se aprende a desactivar el dominio matemático |
| Configuración inválida | Umbral fuera de rango, k negativo, multiplicador NaN | Rechazo con justificación; no fallback silencioso |
| Veto de mercado válido | Cascada supera límite | Conservar; aplicar el parámetro real de manera consistente |
| Excepción mal razonada | La magnitud de OBI implica alineación | Corregir el signo, conservar el resto pendiente de calibración |
| Proxy heurístico | “utilidad” sin distribución de payoff | No presentarlo como utilidad esperada demostrada |
| Confianza sin soporte | Consenso 100% de una raíz | Medir procedencia/dependencia; no equipararlo a evidencia independiente |
| Adaptación sesgada | Neutrales premiados, corto ganador penalizado | Corregir contabilidad y orientación de feedback |
| Rigidez temporal | Clamps de s a 30s–12h | Documentar pérdida de sensibilidad; rediseñar con soporte y error |

Los vetos no se juzgan por cuántas órdenes rechazan, sino por la proposición que afirman y la evidencia necesaria para sostenerla. Una frontera de solvencia o dominio no se vuelve incorrecta por ser discreta. Un estado de volatilidad continuo sí puede representarse sin casillas arbitrarias; eso no elimina mínimos del venue, integridad de datos o decisiones discretas de enviar/cancelar.

## 14. Teoría matemática, estadística, física y cuántica: programa verificable

### 14.1. Ensamble sobre un grafo de evidencia, no un consejo de voces supuestamente independientes

Cada arista debe conservar fuente, tiempo, activo, transformación y generación. Dos nodos derivados de una misma raíz pueden aportar funciones útiles, pero no dos observaciones independientes. El número de asientos no es tamaño muestral. El objetivo de validación es cuantificar aporte marginal mediante ablación y desempeño fuera de muestra, con costes e incertidumbre.

Antes de añadir nodos “más avanzados”, hace falta especificar el evento que cada salida predice y su pérdida de entrenamiento. Para confianza probabilística se requieren calibración y scoring apropiado al target; para utilidad, unidades monetarias o de crecimiento logarítmico y restricciones explícitas. Un score acotado no se convierte en probabilidad por aplicar clamp.

### 14.2. Adaptación online con causalidad y universo multiactivo

El estado adaptativo necesita un índice coherente: activo, escala, acción, modelo, tiempo y procedencia. El pooling puede aprovechar relaciones entre activos, pero debe reconocer heterogeneidad y no sumar n ajeno como certeza local. La covarianza y la dependencia temporal importan para el tamaño efectivo de muestra; n bruto no prueba n ensayos independientes.

La regla de acuerdo ahora simétrica corrige un bug, no resuelve selección: sólo observa resultados de operaciones ejecutadas o estimadas. Para evaluar los vetos faltan outcomes de decisiones rechazadas bajo una metodología de contrafactuales defendible. No se usa como evidencia que una orden rechazada hubiera sido ganadora mirando sólo el precio posterior.

### 14.3. Evolución y epigenética como contratos medibles

Un parámetro verdaderamente conectado debe cambiar el comportamiento donde su contrato lo prescribe y permanecer invariante donde no corresponde. FMT-239 aporta precisamente una prueba config→veto; FMT-240 comprueba una invariancia de referencia; FMT-242 comprueba simetría bajo cambio de lado.

Para el genoma completo hacen falta perturbaciones controladas, mismo replay/semilla/contexto, trazabilidad de efecto en payload, fills/costes y desempeño fuera de muestra. Los rangos y clamps requieren justificación o clasificación como restricciones, no una mutación automática de todo literal. Conservar todos los campos y hacerlos mutables no garantiza identificabilidad.

### 14.4. Integración de teoría avanzada: criterios de entrada

Una propuesta basada en geometría, control, procesos puntuales, mecánica estadística o formulaciones cuánticas debe especificar observables, unidades, hipótesis, estimando, discretización, complejidad, baseline y criterio falsable. Debe aportar una mejora medible más allá de estos contratos básicos.

No se ha implementado una nueva ecuación de los problemas del milenio ni acreditado una ventaja cuántica. No se usa esa complejidad como sustituto de validez estadística. Una analogía física de ondas o interferencia necesita demostrar qué cantidad conserva o predice; el nombre no justifica un cutoff de escala ni una autoridad de veto.

El contraste primario se limitó a beta y su parametrización, mediante Firecrawl con conector porque no estaba disponible la CLI. [Nota de evidencia](<C:/Users/jhona/Documents/Proyectos/Trader Gemini/.firecrawl/XXXI-beta-evidence.md>). Ese contraste corrigió la explicación del prior, no validó el supuesto Bernoulli ni la procedencia de los datos. No se enviaron archivos privados.

## 15. Estado de los ocho módulos

| Módulo | Avance de XXXI | Pendiente relevante |
|---|---|---|
| 1. Ingestión/L2 | Se valida dominio de OBI y campos consumidos por consejo | Freshness, faltantes imputados y continuidad del feed |
| 2. IA/señales | Referencia ML consistente; señal inválida no contamina contador | Targets, calibración y evidencia congelada |
| 3. Multiactivo/horizontes | Saturación de escala y pooling implícito reproducidos | Representación espectral conjunta y soporte por activo |
| 4. Ejecución | Orientación de retorno en caller del cierre | Fills reales, reservas y reconciliación FMT-225 |
| 5. Riesgo/genomas | Excepción alineada y parámetro de cascada efectivo | Calibración de umbrales y autoridad por causa |
| 6. Estado/telemetría | Ventana de conteos coherente con abstención | FMT-230/231/235/236 y pérdidas; no modificados aquí |
| 7. Orquestación/cuántica | Grafo real de dependencias y contrato de consejo | Independencia, causalidad y cualquier ventaja cuántica |
| 8. Backtest/gobernanza | Regresiones sintéticas y evidencia rojo→verde | Validación operativa, promoción y suite integral |

Se conservan las deudas XXX del kill-switch, mmap y escritura de ledger auxiliar. El diagnóstico de kill-switch se reejecutó; los demás no se cuentan como nuevos tests de esta ronda.

## 16. Verificación reproducible

```text
cargo test --offline -j 1 -p metacortex-engine --test council_evidence_contract -- --test-threads=1
cargo test --offline -j 1 -p metacortex-engine --lib consejo_seniors::tests -- --test-threads=1
cargo test --offline -j 1 -p god-engine-core --test close_outcome_contract -- --test-threads=1
cargo check --offline -j 1 --bin god_engine --bin feature_exporter --bin train_forest --bin train_dark_alpha
```

| Selección | Funcionales/compatibilidad | Diagnósticos abiertos |
|---|---:|---:|
| council_evidence_contract | 19 | 3 |
| consejo_seniors::tests | 8 | 0 |
| close_outcome_contract | 10 | 1 |
| Total distinto | 37 | 4 |

No se cuentan las repeticiones. Los tests legacy de “diversidad” comprueban sensibilidad funcional a inputs, no independencia estadística; su pase no cierra FMT-244. Los cuatro diagnósticos abiertos pasan precisamente porque las limitaciones siguen presentes.

La primera compilación falló con os error 112 por espacio en disco antes de ejecutar los tests. Un reintento serial funcionó sin borrar cachés o datos. No se interpreta ese fallo de infraestructura como un bug matemático ni como una prueba roja. El estado de espacio es compartido y transitorio; no se atribuye su recuperación a una limpieza realizada por esta sesión.

Las doce primeras aserciones fallidas del consejo se observaron antes del fix; el test corto se reprodujo después de corregir su slot de fixture y antes del cambio de consumidor. La primera selección final fue repetida tras afinar el texto de diagnóstico de cascada. Cargo check pasa con tres warnings previos: latest_ts, mode y trades en evolution-engine. No se ejecutó cargo fix.

No se ejecutaron toda la suite del workspace, replay económico, benchmarks, pruebas de concurrencia operativa ni backtests que pretendan medir rentabilidad. La falta de benchmark impide prometer latencia: la deliberación ya construye vectores/strings y toma RwLock; XXXI no acredita un presupuesto HFT.

## 17. Preservación y límites de operación

Se añaden adendas al atlas, informe maestro y XXX conservando sus prefijos normalizados CRLF→LF. El artefacto conserva hashes de fuentes/tests/nota y de 41 modelos; éstos permanecen sin cambios respecto a XXX. Los hashes anclan este snapshot y no protegen contra modificaciones concurrentes posteriores.

La rama local sigue siendo main y HEAD 59a76de4. No se hizo commit, push, merge, fetch, reset ni checkout, ni se atribuye todo el árbol sucio a esta ronda. No se modificaron deliberadamente genomas activos. Se respetan cambios ajenos.

No hubo peticiones de cuenta, órdenes, entrenamiento/promoción operativos ni terminación/reinicio de procesos. Cargo check no construye el ejecutable operativo. Sí se ejecutan actualizaciones de aprendizaje sintéticas en tests para validar su contrato. Los temporales del test de cierre pertenecen al fixture y se limpian dentro de su ruta validada; no se borraron cachés compartidas, datasets, modelos ni traumas operativos.

## 18. Hoja de ruta y condiciones de cierre

1. Unificar decisión y trace: una sola evaluación, features/contexto congelados, identidad por activo/slot/generación y vínculo posterior a intención/fill.
2. Resolver FMT-225 antes de tratar estimaciones como evidencia económica. Mantener separación simulación/demo/prod y la contención XXIX.
3. Sustituir n global prestado por suficiente estadístico de la población correcta o pooling explícito; medir dependencia y calibración, no sólo número de muestras.
4. Definir veto por clase y autoridad. Integridad, mandato de riesgo y condiciones de ejecución no deben confundirse con una opinión direccional sobreescribible.
5. Replantear consenso sobre procedencia y aporte marginal. Preservar nodos útiles, pero no contar transformaciones correlacionadas como observaciones independientes.
6. Rediseñar interfaz temporal del consejo con soporte, error y presupuesto. No ampliar endpoints sin datos ni prometer cálculo exhaustivo por nanosegundo.
7. Completar lectura de los 143 Rust pendientes y del inventario no Rust, con hallazgos, reproducciones y residual por archivo. Esta ronda no sustituye esa cobertura.

La mejora demostrada es coherencia local entre datos, políticas, señales y feedback. La autoevolución universal, multiactivo y temporal-espectral sigue requiriendo trazabilidad económica, representación adecuada y evidencia fuera de muestra; no se certifica por estos 41 pases.

## 19. Adenda de continuidad XXXII — 2026-09-25

El [informe XXXII](<C:/Users/jhona/Documents/Proyectos/Trader Gemini/docs/AUDITORIA_FUNDAMENTOS_CIENTIFICOS_XXXII_2026-09-25.md>) y su [artefacto](<C:/Users/jhona/Documents/Proyectos/Trader Gemini/docs/artifacts/auditoria_fundamentos_XXXII_2026-09-25.json>) amplían esta auditoría sin modificar lo anterior.

FMT-243 avanza: deliberación y votos salen de una sola evaluación, con contador/pesos de una misma lectura. Sigue ABIERTO el uso de población agregada para wr local. Se añaden FMT-246/247: identidad estable de roles, integridad de opiniones y vínculo por símbolo/slot/generación/lado, consumido una vez; el array legacy ya no es evidencia suficiente para aprender.

FMT-248 corrige ratio de flujo dependiente de unidades/overflow. FMT-249 queda ABIERTO: consumo macro vacía el dato que el consejo espera observar; sin símbolo/tiempo en el canal global. Esto limita el impacto operativo de la reparación de threshold FMT-239, sin invalidar su contrato local. Nuevas deudas250/251/252: calidad/frescura del offset, promoción ficticia del helper y dominio extremo de resta de timestamps. Se corrigen descripciones numéricas y de capacidades sin habilitar rutas latentes.

23tests nuevos;8aserciones rojo→verde;60funcionales+6OPEN=66pases únicos. Cinco lecturas nuevas:151/289 Rust,138pendientes. Modelos preservados, check offline de cuatro binarios pasa. No se certifican settlement, rentabilidad, universalidad temporal ni revisión completa; sin acciones de cuenta ni publicación Git.
