# LG5, ceros y replay: integración sobre main — 2026-10-15

Estado de este corte: composición local en revisión, sin publicación ni validación dinámica del nuevo árbol. Responsable ROOT, rama `codex/root-lg5-main-2026-10-15`, worktree aislado `root-lg5-main-1015/Trader Gemini`.

## Fuente y evidencia que se conserva

La composición une `b9cd773538fae9467751883eafdf3ee6ed5142f9` (main Ω128) con `971ec9ed212eabb888a3f306739d37e84be44e57` (LG5). El remoto observado después avanzó a `4096a4c60e52889850040ce7eab981fc0980c9ae`: dos documentos y 25 líneas, sin cambios en inputs compilables respecto de b9. Debe incorporarse antes de publicar, junto con cualquier avance posterior validado.

T1 histórico **sí terminó** el 13 de octubre sobre 971: dos pruebas aprobadas. El recibo 016 registra backtest/T1 **137 aprobadas, 0 fallidas, 2 ignoradas**, ocho binarios retenidos. El log no captura el número exacto de genes; **18/144 pertenece al corte R4 anterior**, no se transfiere a 971 ni a esta composición. El recibo 015 registra Core/risk **1006 aprobadas, 0 fallidas, 53 ignoradas**, 137 binarios. Son ejecuciones históricas con su alcance, no certificación de main actual ni de rentabilidad.

Ambos recibos y 145 binarios (448.736.768 bytes) están en una copia externa sellada, además del bundle de los seis commits exclusivos LG5. Directorio: `C:/Users/jhona/.codex/audit-evidence/trader-gemini/lg5-terminal-015-016-2026-10-15-8649302e-7e22-4a2d-9f52-807d0f2207f7`. SHA256 del manifest: `bc3b665e4e7b9ed4e1cacee0fbc47a6ed3616c71ee2cbebb0657e353a18449d0`. El bundle requiere la base 332; no se presenta como repositorio autónomo ni como caché reproducible.

## Decisiones de integración

- Core conserva el reloj causal del evento para estimar payoff y validar su disponibilidad; el ledger económico se captura tras apertura y se consume una sola vez al cierre elegible. No se confunde PnL por nocional con R por distancia de stop.
- La revisión estática de Core contra ambos padres conserva exactamente los cambios de cada lado: LG5 56 líneas añadidas/4 eliminadas y main 1005/70. Esto no sustituye pruebas del árbol compuesto.
- Riesgo conserva LH04, headroom de cartera, topes continuos, tres lecturas OU al último tick y sonda/SPRT de main. LG5 sustituye sólo el EV binario obsoleto: consume payoff neto explícito y prima de coste `(m−1)c`, sin fabricar retornos desde WR/PF/confianza. El reloj de evidencia económica es independiente del reloj OU.
- Los fixtures maduros nuevos de LH04 y margen aportan una cohorte sintética explícita para aislar sus respectivos contratos. Esa cohorte no se reconstruye de las métricas agregadas LH04 y no acredita una cohorte conjunta ni calibración condicional.
- Se preservan los exports de parámetros nuevos de main y los tipos de payoff de LG5. Los conflictos de documentos conservan ambos cuerpos.

## Hallazgos nuevos y oráculos

**LH04, unidades y objetivo limitado.** `ValidatedOrder.volume_usd` es margen; dividirlo por leverage subestima el margen comprobado. `sl_target` es precio absoluto; el riesgo de stop en USD es `margen * L * abs(entry−stop)/entry`. Las correcciones de unidades no modifican sizing productivo. La cota antigua `R_stop >= 0.5*f*C` puede ser inviable bajo el tope de margen: necesariamente `R_stop <= C*tope*L*stop_fraction`. Además, LH04 divide por `continuous_sl`, que no siempre coincide con la geometría final. Se conserva inicialmente la cota con unidades corregidas para obtener evidencia RED real; no se eleva leverage ni se rebaja una puerta para producir PASS. La equivalencia con Kelly óptimo y el objetivo de riesgo tras topes permanecen por demostrar.

**Ω127, parámetros numéricos y replay.** La revisión encontró un candidato concreto: las APIs directas `with_params`/`update_params` de `EspectralMultiactivo` aceptan `olvido=NaN`; una pareja fresca puede contaminar permanentemente sus momentos EWMA. Core usa esa ruta directa desde config atómica. Se prepara un contrato con control válido maduro, replay igual/anterior y pareja fresca; el estado previo debe ser observable y no se aceptan comparaciones vacías `None == None`. En este corte aún no hay reproducción dinámica ni reparación certificada.

**Ceros/replay anteriores.** Se conserva el producto de retornos crudos, con cero como abstención del e-proceso; la guardia de timestamp precede a las escrituras y ambos cierres deben avanzar para consumir una nueva muestra. Los cambios de parámetros no certifican por sí mismos filtración, familia efectiva ni vigencia estadística.

## Secuencia de validación y coordinación

1. Resolver y revisar el diff contra ambos padres, inventariar únicamente las rutas del cambio y compilar `--workspace --all-targets` antes del commit de integración, según `.agents/AGENTS.md`.
2. Congelar el commit y ejecutar contratos RED concretos, reparar su causa y obtener GREEN. La cadena debe abortar ante fallo; conservar logs, JSON y binarios antes de otro Cargo.
3. Ejecutar suites relevantes de quantum-arena, Core/risk y backtest/T1 sobre el candidato final. T1 usará `--nocapture`, snapshot de modelos/genomas antes/después incluso ante fallo y umbral existente íntegro. Un snapshot no acredita activación de modelos.
4. Incorporar avances de main, revisar cada padre y volver a validar los inputs que cambien. Publicar sólo el candidato revisado; verificar ascendencia y SHA remoto y comprobar su CI sin heredar el estado de un padre.
5. Cerrar refs realmente integradas, conservando ramas activas, commits exclusivos, worktrees y evidencia. Registrar las excepciones.

BASE15 trabaja separadamente en fundamentos/inventario y sólo dos fuentes productivas (`multifractal.rs`, `execution-engine/src/simulator.rs`). Su plan es `docs/audit/PLAN_BASE15_2026-10-15.md`. Los cambios de Core/risk/genoma/replay quedan reservados a esta integración. La compilación pesada de BASE15 tiene prioridad y su caché es privada; ROOT aún no inició Cargo en este corte. La ventana histórica SI de `r4-build` fue cerrada por ROOT después de revisar recibos y procesos; no se atribuye un acuse de liberación a SI.

Continúan abiertos R4-O01 (nulo condicional y familia), R4-O02 (mapping precio/cartera/PnL), R4-EPOCHS y SI-O02; también apertura automática→cierre→siguiente admisión LG5 y recuperación de legacy maduro sin payoff. Una media neta descriptiva no es una cota inferior ni una expectativa condicional validada. La meta de duplicación cada 72 horas sigue siendo una meta, sin demostración económica en estos recibos.

## Continuación ROOT: composición y correcciones preparadas

Este corte sustituye los pendientes de preparación anteriores; conserva su historia. La rama propia contiene `0823ca50` (b9+971), `e443c4d5` (main310), las tres correcciones siguientes y `a4573bdef526b1e75c3329e6f3816c8b0a74de36` (main940). No se ha publicado esta rama ni integrado sus cambios en main en este corte.

| Cambio | Comportamiento y prueba preparada | Estado |
| --- | --- | --- |
| `98b6d3d2` | `olvido` debe ser finito en (0,1]. El constructor sustituye sólo ese campo; una actualización inválida conserva el conjunto válido completo y la historia. Controles maduros, replay, continuación y nueve valores inválidos. | Revisión estática; comportamiento pendiente de ejecución. |
| `d14db933` | Evita el panic de `clamp` cuando `.90+.05` supera `.95` por redondeo. La API Bernoulli directa continúa muestreando ante parámetros no finitos. Registry conserva su fallback existente. | Revisión estática independiente sin bloqueantes; contratos de frontera, ULP y fallback pendientes. |
| `4a735f1a` | El positivo LH04 exige Long, margen 250 USD y riesgo de stop 2,5 USD con C=1000/L=1/stop=99. El ATR diminuto exige el rechazo exacto previo al sizing. El contrato LG5 permite el cap de main y comprueba que la salida SPRT no evita payoff ni disponibilidad. | Oráculos revisados; no se modifica sizing productivo ni se afirma GREEN. |

SPRT recibe actualmente `round(n*WR)`. El productor publica WR posterior, de modo que ese valor es un pseudoconteo y no recupera las victorias observadas. La reparación numérica no acredita los errores de Wald, IID, épocas ni esperanza neta. Una futura corrección estadística necesita conteos y prior trazables; no debe reconstruirlos inventando un prior.

Los checks ordinarios `002-precommit-b9-971`, `003-precommit-main310` y `004-precommit-main940` pasaron antes de sus respectivos commits de merge. El último ejecutó `cargo +nightly-2026-06-30 check --workspace --all-targets --release --locked`, un job, exit 0, 7 min 40 s: 708 inputs compilables y árbol preparado idénticos antes/después. JSON SHA256 `af0069e02b4e973aeac43973011a02a9b9f2bbf14934c9d2da246252735478a2`; log `d5049253947c46e06791e325ea9db1b6f735ae91a38066a9cfdcf874f8100177`. El check compila tests, pero no los ejecuta. El intento 001 conserva su fallo de integridad por metadatos físicos del índice; no se recalifica.

La ejecución protegida `1015-004-nan-red-arena-params-red`, sobre e443, falló al enlazar `futures_channel` antes de ejecutar pruebas. Cargo 101, cero tests y cero binarios retenidos; sus 12 invariantes y ocho guardas permanecieron verdaderas, igual que la identidad Before/After de modelos. No se obtuvo RED funcional. La revisión de 154 metadatos de caché no demostró rlib perdido ni mezcla: las cadenas build y check observadas son coherentes. El siguiente ensayo será `--no-run -vv --jobs 1`, sin limpiar caché, para capturar los `--extern` reales; sólo compilar no equivale a GREEN. Runner protegido y bundle de orquestación permanecen intactos.

Una copia externa preserva este punto previo a las correcciones: `C:/Users/jhona/.codex/audit-evidence/trader-gemini/lg5-integration-pre-red-2026-10-15-482f5b01-fbef-4ede-8fb0-8af47a717035`, 368 archivos/1.837.212 bytes; manifest SHA256 `e234cd46144c4a62b55d7081ea2cde8af8399e5817f1c13a219f4a724644f936`. Incluye bundle hasta e443 y recibos 001–004; no respalda aún los commits posteriores ni la caché completa.

Se prepara un merge documental de tres padres: a457, main `b46dd28624c2b0325f954066ea5c89be84495e7b` y SI `7af2bbcb11c27ea43d145366f86dca90775f664d`. No cambia inputs compilables respecto de a457. El check específico de todos sus padres sigue pendiente en este instante. El cierre terminal MTM de GLM se conserva con su alcance: no cierra causalidad tras break/tick rechazado, drawdown de equity ni invariancia de rankings históricos. Su T1 declarado de 709,76 s tampoco certifica esta composición LG5.

Orden acordado: BASE15 primero ([PR65](https://github.com/Jhona-la/Trader-Gemini/pull/65)); Hawkes después, sólo tras sus pruebas; incorporar también T3 v2 ([PR64](https://github.com/Jhona-la/Trader-Gemini/pull/64)) y SI; ROOT ejecuta las suites y T1 finales con fuente congelada. Los autores conservan sus worktrees y reservas. ROOT obtuvo sólo ventanas metadata a un job y no encadena enlace/tests mientras BASE15 mantiene su compilación. La limpieza de refs se hará por ascendencia comprobada, preservando ramas ocupadas, commits exclusivos y evidencia.
