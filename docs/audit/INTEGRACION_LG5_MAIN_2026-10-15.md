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
