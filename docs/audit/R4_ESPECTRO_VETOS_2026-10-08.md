# R4 — Espectro multiactivo, evidencia y vetos — 2026-10-08

Corte único: `eb32ac9613c847eec75bd8e2591add8ae88234cf`. Auditoría de lectura y testigo Rust aislado autorizado; sin Cargo, sin cambios de runtime, HEAD, índice, umbrales o reservas de edición AGY f4. No se observaron operaciones ni pérdidas del sistema vivo. Ningún archivo queda cerrado por haberlo leído completo.

**Resultado:** se reprodujeron dos defectos con fuente exacta: (A) el adaptador de producto a signo transforma ceros en apuestas y puede abrir evidencia de familia después de un único bloque informativo; (B) la API multiactivo acepta replay regresivo, y el test Qo665 utiliza precisamente ese replay. A tiene una cadena activa de productor/publicador/consumidor trazada; B queda demostrado en API/test y no en el flujo normal, cuyo productor impone monotonía. I4 permanece como deuda de descripción/cobertura del dominio: seis índices seleccionables frente a cinco declarados, sin demostrar subcobertura del roster configurado de 26. La validez estadística de Ville continúa abierta y no se resuelve corrigiendo ceros o contando escalas.

## Fuentes y alcance de lectura

Todos los rangos son inclusivos y pertenecen al commit de corte; se consultaron los objetos Git fijados. “Completa” describe lectura, no certificación.

| Path relativo | Blob | Rangos leídos / alcance |
|---|---|---|
| `crates/quantum-arena/src/espectral_multiactivo.rs` | `a480b0d2cc3449fc07ca569dbd0606af84fcbf8c` | **1–627 completos**, incluidos nueve tests |
| `crates/quantum-arena/src/evalues.rs` | `758ff0af9725b73ffdbb054e84e3e025c0435a94` | **1–278 completos**, incluidos cinco tests |
| `crates/god-engine-core/src/evidence_publication.rs` | `b0d34fe01fe4e08476ed05e8def2996fc4b3d30b` | **1–77 completos** |
| `crates/quantum-arena/src/temporal_spectrum.rs` | `45c2bba7f8af60b3f160b1327de8468d8c094013` | 1–174, 340–690, 1145–1250; archivo de 2712 líneas, lectura parcial |
| `crates/god-engine-core/src/lib.rs` | `b045efbf82766090478c9da17412b05acb7d2810` | 920–952, 1033–1053, 1600–1680, 2733–2763, 3744–3778; búsqueda global de símbolos relevantes |
| `crates/risk-engine/src/lib.rs` | `554e02d060059d9d15247a2194df6d65f0f3ea36` | 600–725; búsqueda del consumidor rho_tau |
| `crates/signal-engine/src/skill_motores.rs` | `f8d1c9f81d2c145b4a20b6504658624b78fed579` | 1–270, parcial; fuente de 512 líneas |
| `crates/signal-engine/src/voto_espectral.rs` | `a170fcce199beae30e312bd6327ee266cf441788` | 1–40, constante dimensional |
| `crates/quantum-arena/src/state.rs` | `a9b0813d22d967458416c2e5b8c050cdb014a211` | constante MAX_COINS, línea 15; usos localizados por búsqueda |
| `crates/god-engine-core/src/bootloader.rs` | `6ae4163dbef146f40ecf8b1633417e1f6dfe76c1` | 311–358, roster y retorno |
| `src/bin/god_engine.rs` | `eea831560cce74f00570b230724f5eef3f106420` | 760–835, arranque y configuración; búsqueda de símbolos |

## A — Cero transformado en evidencia direccional

**Prioridad propuesta: P2, siguiente lote de corrección de evidencia. Diagnóstico Rust reproducido; integración productiva y frecuencia de ocurrencia no medidas.**

Hunk causal: `espectral_multiactivo.rs:94–102` toma el signo previo de eab como señal y calcula `(ra * rb).signum()` antes de llamar `EProceso::observar`. El guard de entrada de `182–188` sólo descarta retornos no finitos, índice/moneda inválidos, tau inválida y timestamp cero: acepta r=0. En Rust `signum(+0.0)=+1.0` y `signum(-0.0)=-1.0`, por lo que la abstención por `producto == 0.0` de `evalues.rs:90–98` ya no ve el cero original. La documentación oficial de [f64::signum](https://doc.rust-lang.org/std/primitive.f64.html#method.signum) confirma esa semántica.

La secuencia reproducida es **un par informativo positivo + 112 pares de ceros**, no 113 bloques informativos ni 113 planos. Ambos activos tienen r=0.01 en el primero y r=+0 en los siguientes; los timestamps son nuevos, crecientes y contemporáneos, y tau es el nodo real k19 de la malla. Los tres momentos ea2, eb2, eab quedan inicialmente iguales y positivos; cada plano los multiplica por gamma=63/64. Tras 112 planos, gamma^112≈0.171389595 y cada momento es aproximadamente 2.6779624e-7; el denominador sigue positivo y el IC permanece 1. El olvido es **por muestra de par**, no por reloj físico.

Para M=2175, alfa=.05 y lambda=.10, el umbral es 43500. El cálculo de factores da 1.1^112≈43249.4648 y 1.1^113≈47574.4113: hacen falta **113 aciertos**. No se extrajo el capital privado del par como telemetría; se verificó su gate público y se distingue este cálculo algebraico de las salidas observadas.

| Control ejecutado en el testigo aislado | Resultado público observado |
|---|---|
| EProceso directo: 1 producto informativo, 112 retornos cero | n=113, capital=1.1, familia=false |
| Multiactivo: 1 par informativo, 112 pares (+0,+0) | n=113, IC=Some(1), veto=Some(1), media_veto=Some(1) |
| Igual, cambiando sólo el signo de cero de un activo: (+0,-0) | n=113, IC=Some(1), veto=None |
| 113 pares cero desde estado vacío | IC=None, veto=None; el denominador cero impide emisión |

El control de ceros con signo opuesto aísla una diferencia de representación IEEE-754 en datos económicamente iguales. El caso completamente vacío no abre el gate: el defecto necesita dispersión no nula en el sentido implementado (segundos momentos positivos) procedente de historia previa. Esa comprobación de segundo momento no es una certificación de varianza centrada o de correlación estadística.

### Alcanzabilidad exacta y precondiciones del consumidor

1. `TemporalSpectrum::update`, líneas 413–457: exige precio positivo finito y timestamp creciente; T0 puede ser origen. Líneas 482–508: al cerrar un bloque calcula `r=ln_p-bloque_ln_p0` y publica ts/r **también cuando r=0**. `ultimo_bloque_maduro`, 402–410, sólo comprueba índice y existencia de timestamp.
2. `GodEngineCore`, 1633–1651: llama update y alimenta `observar_maduracion` para cada cierre de las 32 escalas, sin eliminar r=0. La fuente multiactivo puede acumular por tanto la secuencia plana. Espectral 190–230 exige recencia y consumo por par; los pares nuevos sincrónicos del testigo satisfacen ambas condiciones.
3. Core 1667–1672 publica el resultado para `spec.dominant_tau_ms`. Las otras publicaciones son 2749–2754 (dual directo) y 3764–3769 (feedback). `evidence_publication.rs:56–76` selecciona escala y guarda el IC finito o -1 de ausencia en `qo_613_rho_tau`.
4. `risk-engine/src/lib.rs:636–665`: sólo usa rho_ic para endurecer cuando rho_escalar es `Some(base)` y rho_ic>base; luego evalúa el veto de grupo con sus riesgos, tope y posible R. Si rho_escalar es None, el IC no activa esa sustitución.

Para afectar una decisión deben coexistir: par con momento previo coherente positivo, posteriores cierres planos que pasan recencia/consumo, escala elegible seleccionada con evidencia madura, publicación vigente de esa escala, base escalar presente menor que rho_ic y llegada al gate con el resto de precondiciones satisfechas. El testigo no instancia TemporalSpectrum, el core, el registry o risk-engine; **traza estática de ruta activa** y **reproducción del módulo exacto** son evidencias separadas. No prueba que hubo órdenes vetadas, trades, pérdidas ni un beneficio al corregirlo.

Propuesta falsable para el siguiente lote: conservar el significado de producto cero como abstención antes de la reducción a signo, tanto para +0 como -0; decidir explícitamente qué contadores representan observaciones recibidas o apuestas informativas. Congelar M/alfa/lambda. La regresión debe comparar contra EProceso directo y preservar un control con productos genuinamente informativos que sí acumule evidencia. No es propuesta de bajar thresholds ni de abrir admisiones; requiere RED/GREEN y evaluación del consumidor según protocolo antes de implementar durante otra reserva.

## B — Replay API y test Qo665 que no certifica unicidad

**Prioridad propuesta: P2 para contrato API/test; llegada por productor normal no demostrada.**

`observar_maduracion:190–194` descarta sólo `previo.ts == nuevo.ts`; un timestamp menor reemplaza ultimo. `223–230` impide reusar únicamente timestamps iguales a la última pareja consumida. Un bloque histórico distinto de esa pareja vuelve a contar después de otros bloques.

El test propio `qo_665_ic_veto_exige_capital_de_familia`, líneas 585–602, define feed desde i=0 hasta el límite; llama feed(40) y luego feed(110). El segundo llamado repite los primeros 40 timestamps. La reproducción exacta de esa secuencia mediante la API pública obtuvo **n=150 y gate Some(1)**. El control de **110 timestamps únicos** obtuvo **n=110 y gate None**. La aserción del test puede pasar gracias al reconsumo; no certifica unicidad ni que 110 muestras nuevas basten. Los comentarios “104 aciertos” (559/582) también contradicen el umbral real de 113, y el mensaje “120 perfectas” (601) no describe el conteo ejecutado.

El productor normal `TemporalSpectrum::update:451–452` sí rechaza ts<=last y conserva una última maduración monotónica; el core consume ese accessor. No se atribuye el replay a ese camino activo. Es un defecto de API pública y una falsa tranquilidad del test; otros callers o una restauración de estado requieren contrato aparte.

Próximo contrato propuesto: historial por moneda/escala monotónico y no mutación/reconsumo ante replay, junto a tests que midan muestras_par, no sólo gate; una secuencia incremental de 40 y luego 70 timestamps nuevos debe equivaler a 110 únicos. Probar aparte cruce a 113 y la continuación causal después del rechazo. No implementado en este corte.

## I4 revalidado: dimensiones, malla y familias

La malla real es **32 nodos, tau_k=10^-6 ms·4^k = 1 ns·4^k**, no 4^k microsegundos. k17=17179.869184 ms, k18=68719.476736 ms y k22=17592186.044416 ms. La unidad µs en comentarios de `espectral_multiactivo.rs:60–62` (y comentarios vecinos de habilidad) es incorrecta por factor 1000. La malla representable no añade observaciones submilisegundo ni evidencia de 146 años: la interfaz usa u64 en ms y la fuente temporal distingue esa limitación.

El selector de publicación (`evidence_publication:62–72`) busca distancia **absoluta** en toda la malla. Para tau_dom en [30000,43200000] puede seleccionar k17…k22 (seis nodos): la frontera 17/18 es 42949.67296 ms, y la 22/23 es 43980465.11104 ms, por encima del límite superior. `refresh_fusion:669–678` permite el fallback a 30000; k17 posee almacenamiento/proceso y no es excluido por las guardas de consulta 258–263 y 320–338. `habilidad_en:380–395`, en cambio, usa distancia en log(tau); no se confunden ambos selectores ni se propone cambiarlos sin contrato de observable.

| Familia | Fuente de conteo | M / umbral alfa=.05 |
|---|---|---|
| Selección univariante de tau* | `temporal_spectrum:646–663`, paraguas de 32 procesos | 32 / 640 |
| Motores por escala | `skill_motores:34–38,94–100`, 13×32 | 416 / 8320 |
| Veto multiactivo | `espectral_multiactivo:60–70`, C(MAX_COINS=30,2)×5 | 2175 / 43500 |

El host recibe **26 símbolos configurados** desde `bootloader:320–350`, los registra mediante `god_engine:791–836`; el core y arena tienen **capacidad 30**, no prueba de 30 feeds efectivos (`MAX_COINS:15`, core 934/1041–1043). Para 26, C(26,2)×6=1950<2175: no se demuestra insuficiencia numérica de esa reserva en el roster actual. Para capacidad 30, C(30,2)×6=2610>2175: el comentario de cinco escalas no cubre el dominio completo de capacidad. La API permite consultar hasta 32 y construir otras capacidades; su dominio genérico no queda garantizado por la familia fija. Hace falta declarar la familia de hipótesis permitida antes de ampliar el roster o la banda, no asignar automáticamente M=32 ni rebajar el gate.

Ninguna comparación de cardinalidad demuestra FWER real: requiere e-procesos válidos para el nulo y la filtración declarados. Tampoco se extrapolan 32/416 a activos, reinicios, búsquedas adicionales o selección posterior de familias.

## Contrato estadístico pendiente (no subsanado por A/B)

El observable del módulo es un producto cruzado EWMA normalizado por segundos momentos; no es una matriz completa de espectros cruzados de Fourier, un estimador de correlación centrada con error certificado ni una descripción exhaustiva del universo multivariante. La alineación de finales dentro de 0.5tau no certifica duración idéntica, cobertura íntegra ni ausencia de gaps de ambos bloques. El olvido 1/64 mide historia de pares, no un horizonte fijo de segundos.

Para un capital E_t=E_(t-1)(1+lambda_t Z_t), el argumento de supermartingala exige factor no negativo, lambda predecible y, bajo el nulo adecuado, `E[Z_t | F_(t-1)] <= 0` cuando lambda>=0. Que el signo de IC sea anterior a la muestra y lambda=.1 esté fijo no demuestra la última condición. Se necesitan además política de muestreo/emparejamiento, filtración y familia completas. Es la condición estructural de la inferencia secuencial, no una consecuencia de pasar un test de ruido; véase [Ramdas, Grünwald, Vovk y Shafer, SAVI](https://arxiv.org/abs/2210.01948).

**Contraejemplo algebraico del nulo**, no simulación ni tasa empírica: si condicionado al pasado el producto ra·rb vale 1 con probabilidad .9 y -9 con .1, entonces su media es cero, pero la media de su signo es .8. Con señal previa +1 y lambda=.1, el factor esperado es 1.08>1. Por tanto `E[ra·rb | Fprev]=0` no implica la condición de signo que necesita el capital. La afirmación “bajo H0 IC=0 es supermartingala” de `espectral:86–89` no está justificada por ese nulo. Debe especificarse un nulo condicional correcto para el score usado o diseñarse otro score admisible; corregir ceros, evitar replay y ajustar M no cierra esa validez.

**Contraejemplo adicional del nulo de mediana en EProceso**, distinto de A: `evalues.rs:14–16,52–54` afirma que mediana(señal·retorno)<=0 implica factor esperado<=1. Con Y=0 de probabilidad .51 e Y=1 de probabilidad .49, i.i.d. e independiente del pasado, la mediana es 0. El guard correcto de cero de EProceso deja factor 1 cuando Y=0, y aplica 1.1 cuando Y=1: factor esperado=1.049>1. No se simuló una tasa de rechazo; basta la esperanza de un paso para refutar esa implicación. El problema aquí es el nulo declarado, no el guard de abstención directo.

Para el score de signo con cero neutral, un nulo suficiente de ese paso es `P(Y>0 | Fprev) <= P(Y<0 | Fprev)`, equivalente a esperanza condicional del score <=0; “mediana no positiva” con átomos no basta. La independencia i.i.d. del contraejemplo tampoco lo salva. El archivo EProceso permanece **hallazgo_abierto** por su contrato estadístico, separado del adaptador multiactivo y del conteo de familia. Sus dominios aritméticos de contador/capital tampoco se declaran cerrados aquí; no se amplió este lote para reproducir sus extremos.
También hay documentación obsoleta: `espectral:48–52` afirma cero consumidores de política, mientras el core 1640–1642 y el lector de riesgo muestran una ruta activa. La descripción debe decir qué observable consume cada decisión y con qué evidencia/error aún pendientes; no “dependencia resuelta” como certificado estadístico.

## Recibos del testigo y paquete reproducible

Se compiló **fuente Rust exacta**, sin port matemático Python: EProceso y EspectralMultiactivo completos; sólo las constantes necesarias del módulo temporal (70–103) y state (15), extraídas literalmente. Los tests cfg(test) contenidos en las fuentes no se ejecutaron: el wrapper main incorpora controles diagnósticos por la API pública. El compilador fue `rustc 1.98.0-nightly (096694416 2026-06-29)`, x86_64-pc-windows-msvc, `--edition=2021 -O`. El runner Python sólo extrae, verifica hashes y orquesta rustc; no reescribe fórmulas.

- Ejecución inicial: `target/r4-review/spectral_zero_witness_2026-10-08/receipt.json`, SHA256 `e5755b35aa4484edc5df6027cc80fdb869d63370321d30dfa48db5cd7a70a748`.
- Paquete portable propuesto, **estado al preparar este informe (2026-10-08): bajo target ignorado**: `target/r4-review/spectral_witness_bundle/scripts/audit_r4_spectral_witness.py`, SHA256 `e0399e860d53c26385fa954c984da19500f4960eb4aacd8f814db84f59ef6947`; fixture `scripts/audit_fixtures/r4_spectral_zero_witness.rs`, SHA256 `f034a8e108c567563ec7e8c076cd38a10f7625131abce9ee547fa1476f688690`.
- Verificación del paquete portable: `target/r4-spectral-witnesses/a47b5539-cbcc-4ad2-8697-e22e61e3c071/receipt.json`, SHA256 `55abd6921004fbafbac3093b90917e644a432c89f624cbf6b0b5de4efcde09a3`. Tres comandos (versión, compilación, ejecución), todos exit0; salida `DIAGNOSTICS_REPRODUCED_NOT_REPAIRED`. Mismo stdout SHA256 `5ecb6dbd91b872805389c25413a7ec7597c895b10c03909f03b8ce0076ee760c`.

El comando propuesto tras copiar el paquete es `python scripts/audit_r4_spectral_witness.py --repo .`. Fija el commit eb32, verifica sus cuatro blobs y el hash del fixture, y crea una carpeta UUID bajo target sin borrar archivos ni modificar el checkout. El recibo registra fuentes/slices, hashes, comandos, compilador, salidas y exits. Su éxito seguirá caracterizando el defecto histórico aunque una fuente futura lo corrija: no prueba el estado futuro por ejecutar sobre un checkout más nuevo.

El cierre documental incluye la copia y el versionado de este paquete; para el estado de publicación posterior prevalece la adenda vigente de `docs/PLAN_AUDITORIA_BASE_2026-10-08.md`. Las rutas bajo target describen el corte de preparación y no acreditan por sí solas commit, merge o push.

## Siguiente lote propuesto, pendiente de acuse y reserva, sin ejecución ahora

1. Acusar ownership para corregir A y la caracterización B en una rama separada después de liberar el freeze; mantener invariantes de riesgo y thresholds.
2. Reproducir RED/GREEN de cada contrato con estado y control causal, y añadir un caso productor→publicación→consumidor con tau elegible y rho base explícitos. Registrar por separado toda diferencia de admisión; ninguna cifra de oportunidad perdida está medida aquí.
3. Especificar familia, nulo condicional, reloj de observación y error del observable antes de afirmar significancia o ampliar la capacidad. Vincular a R2/R3 y al diseño OU multivariante condicionado; no usar complejidad matemática como sustituto de evidencia.
4. Mantener abiertas ambas fuentes y la calibración. Este informe no añade resultados T1, evaluación OOS ni evidencia para la meta aspiracional 100%/72h.
