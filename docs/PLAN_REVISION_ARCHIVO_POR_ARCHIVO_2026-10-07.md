# Revisión desde la base: plan y recibos por archivo

Fecha: 2026-10-07, America/Bogota. Responsable de coordinación: Codex.
Base de esta ronda R4: `adeb8d1b1f8171b14fffa8abc9f54c396e4794dd`.
Rama documental: `codex/quant-foundations-2026-10-07`.
Rama de recuperación: `codex/integration-recovery-2026-10-07`.
Estado: **auditoría iniciada; cobertura semántica y validación económica abiertas**.

Este plan recupera y actualiza el borrador no versionado del worktree
`codex-review-2026-10-07`, basado en `a054495c`. No borra ese original ni
transfiere sus conclusiones al nuevo SHA. Es el recorrido operativo de esta
ronda; [el plan compartido](PLAN_MAESTRO_SINCRONIZACION.md) conserva contratos
entre agentes, [el barrido](BARRIDO_EXHAUSTIVO_FASES.md) conserva hallazgos
F/G/H históricos y [la arquitectura](ARQUITECTURA_VIVA.md) describe el flujo.
Un cierre histórico no certifica la versión actual.

## 1. Meta, magnitudes y criterio de evidencia

El objetivo del operador es duplicar patrimonio neto cada 72 horas desde
13 USD. No exige acertar todas las operaciones. Su identidad matemática es
`C(t)=13*2^(t/72h)`: crecimiento logarítmico objetivo `ln(2)/3 =
0.23104906 por día`, distinto del rendimiento simple diario equivalente
`2^(1/3)-1 = 25.992105 %`. El logaritmo objetivo por segundo es
`ln(2)/259200 = 2.67417894e-6`.

Esta identidad especifica una aspiración; no demuestra que exista una
estrategia capaz de alcanzarla de forma consistente. Los criterios de
corrección, evidencia predictiva y resultado económico se registran por
separado. Ninguna cantidad de tests verdes sustituye evidencia OOS.

Patrimonio económico: saldo realizado + PnL abierto valorado con precio y
timestamp identificados, menos costes pendientes, separando aportes/retiros.
Registrar comisiones, spread una sola vez, slippage, funding, latencia,
órdenes rechazadas, ejecuciones parciales y liquidación. Unidades: nocional,
margen, capital y riesgo al stop nunca son intercambiables.

En el modelo idealizado GBM sin fricciones, drift/volatilidad conocidos y
constantes, tasa libre de riesgo cero y exposición sin restricciones,
`g(f)=f*mu-f²*sigma²/2` y
`g(f*)=SR²/2`; de ahí `SR_diario=sqrt(2*ln(2)/3)≈0.680`. Es una
derivación condicionada a sus supuestos, no una cota universal ni una
justificación de apalancamiento. Saltos, colas, margen, costes y capacidad
invalidan esa simplificación como receta operativa.

El cuadro económico debe publicar crecimiento log neto por unidad de tiempo,
distribución de retornos de 72 h, frecuencia y severidad de incumplimiento,
drawdown, expected shortfall, exposición correlacionada, probabilidad de
ruina bajo un modelo declarado y capacidad. Ventanas solapadas no cuentan
como ensayos independientes. Se muestran también periodos sin operar.

## 2. Concepto rector: estado causal multiactivo multiescala

La unidad conceptual es `X(instrumento, tiempo_evento, escala, procedencia)`;
las políticas operan sobre una estimación causal del campo y su incertidumbre.
Horizonte, volatilidad, liquidez y dependencia son coordenadas continuas.
Los nombres legacy se clasifican como alias, ancla, persistencia o partición
real antes de decidir migración. Renombrar no demuestra continuidad.

Un banco finito aproxima un continuo: debe declarar banda observable,
resolución, interpolación, error y límites. Representar un horizonte no
equivale a haberlo observado. El símplex de cuatro etiquetas es una
representación particular; no demuestra por sí solo un universo continuo
completo ni probabilidades calibradas. La geometría de Fisher debe tener
un consumidor verificable antes de presentarse como implementación.

Ruta que cada contrato debe cerrar:

```mermaid
flowchart LR
  A[Eventos y especificación del instrumento] --> B[Reloj, libro y estado causal]
  B --> C[Campo multiactivo por escala]
  C --> D[Predicción e incertidumbre]
  D --> E[Decisión, tamaño y vetos]
  E --> F[Orden y evidencia de ejecución]
  F --> G[Patrimonio y atribución]
  G --> H[Evaluación fuera de muestra]
  H --> I[Promoción y rollback]
  I --> D
```

También se sigue la intención rechazada hasta su resultado contrafactual en
sombra. De otro modo un veto puede impedir obtener la evidencia que necesita
para rehabilitarse. Esto no autoriza experimentar con capital real.

## 3. Paquete científico mínimo y transferencia de teorías

Por cálculo: decisión que sirve; variables observables; unidades; dominio;
supuestos; ecuación/fuente primaria; estimador e incertidumbre; dependencia;
coste; límites; consumidores; baseline simple; falsación; resultado; límite.

| Familia | Pregunta útil | Prueba antes de integrar |
|---|---|---|
| Crecimiento log y control estocástico | ¿Qué exposición maximiza crecimiento bajo riesgo y fricción? | Costes, colas, utilidad neta OOS y capacidad; comparar sizing simple. |
| OU, difusión, saltos y primer paso | ¿Es identificable reversión o distribución de tiempos de salida? | Reloj físico, estacionariedad local, residuales, saltos, calibración de barreras. |
| Wavelets, multirresolución y operadores de estado | ¿Separa escalas observables sin fuga de futuro? | Causalidad, resolución, estabilidad, reconstrucción y ablación contra filtros simples. |
| Hawkes y procesos puntuales | ¿Existe excitación adicional a intensidad base? | Residuales por time-rescaling, estabilidad, identidad por activo, baseline Poisson. |
| Matrices aleatorias, cópulas, grafos/Hodge | ¿Se estima dependencia aprovechable? | PSD, asincronía, muestras efectivas, estrés de colas y beneficio incremental. |
| Ville/e-procesos, PSR/DSR | ¿La evidencia sobrevive selección y observación repetida? | Filtración, hipótesis nula, apuestas predecibles, familia completa, dependencia temporal. |
| Control adaptativo/online learning | ¿Detecta pérdida de habilidad y se recupera sin fuga? | Drift, evaluación prequential, datos diferidos, restricciones, rollback y baseline congelado. |
| Física y métodos inspirados en cuántica | ¿La analogía produce una predicción adicional medible? | Correspondencia observable, unidades, estabilidad, comparación clásica y coste. |

Navier–Stokes, Yang–Mills, Riemann o la conjetura de Hodge no son mecanismos
de trading por el nombre. La descomposición de Hodge de un grafo y la
conjetura algebraico-geométrica son objetos distintos. La relación RMT/Riemann
no convierte autovalores financieros en señal. Un algoritmo clásico llamado
cuántico no acredita ventaja de hardware cuántico. Cualquier transferencia
queda como experimento hasta superar baseline y OOS con complejidad penalizada.

Fuentes iniciales: [problemas del milenio, Clay](https://www.claymath.org/millennium-problems/),
[reglas del exchange](https://developers.binance.com/docs/derivatives/usds-margined-futures/market-data/rest-api/Exchange-Information),
[nocional y apalancamiento](https://developers.binance.com/docs/derivatives/usds-margined-futures/account/rest-api/Notional-and-Leverage-Brackets).
Fuentes estadísticas y contraejemplos: [informe R4](audit/FUNDAMENTOS_R4_2026-10-07.md).

## 4. Fases, dependencias y definición de cierre

| Fase | Alcance y prioridad | Evidencia de salida |
|---|---|---|
| R0 | Metas, unidades, procedencia, versiones y coordinación | SHA, censo completo, erratas, mapa de ramas y criterios económicos. |
| R1 | Conceptos y topología real | Productor→consumidor→efecto→reconciliación; mapa vivo/sombra, claves y relojes. |
| R2 | Matemática y estadística | Derivaciones, dominio, dependencia, nulos, multiplicidad y falsación reproducible. |
| R3 | Física, cuántica y algoritmos | Correspondencia medible, estabilidad y ablación; coste/latencia medidos. |
| R4 | Datos y tiempo | Replay de duplicados, huecos, orden adverso, warm-up, OOS, L2 y timestamps. |
| R5 | Capital, vetos y ejecución | Contabilidad conservativa, entradas no finitas, sizing, reservas, fills y reinicios. |
| R6 | Genoma y aprendizaje | Genealogía, sensibilidad, conjuntos congelados, promoción, rollback y reanudación. |
| R7 | Plataforma y rendimiento | All-targets, concurrencia, presión, recuperación y latencias p50/p95/p99. |
| R8 | UI, scripts, datos generados y documentación | Unidades/estado exactos, procedencia del generador, validación de consumidores. |
| R9 | Integración y evidencia económica | Revisión contra padres, regresión, T-1 aplicable, OOS sellado y recibo de main. |

R0→R1→R2/R3 preceden interpretar el código como teoría válida. Después se
priorizan rutas que pueden perder capital o contaminar evidencia, y luego
el resto del censo. Ninguna fase se cierra por ausencia de matches de `rg`.

Cada lote debe ser acotado (orientativamente 10–20 archivos; módulos grandes
en sublotes por contrato), con un escritor y revisor independiente. La lectura
de un archivo completo es distinta de revisar una función. Un archivo de
10.000 líneas puede requerir varios recibos; el estado completo exige unir
los intervalos y todas las rutas dependientes.

## 5. Inventario exhaustivo y recibos

La base contiene **1.434 archivos versionados, 460 Rust y 23 crates miembros
más el paquete raíz**. De ellos, 655 pertenecen a `graphify-out`: el total
incluye derivados, no sólo código. El conteo anterior de ~377 Rust no se
traslada a esta base. [Ledger por archivo](audit/REVISION_2026-10-07.json),
generado por `scripts/audit_inventory.py`, registra blobs y clasificación.
La clasificación automática NO acredita lectura ni auditoría.

El campo `phase` conserva F0–F8 del barrido histórico como enrutamiento por
archivo: F0/documentos→R0/R8; F1/matemática→R2; F2/espectro→R3;
F3/núcleo→R1/R5; F4/riesgo-ejecución→R5; F5/evaluación→R6;
F6/datos→R4; F7/plataforma→R7/R8; F8/tests→R9 más su fase de sujeto.
Las fases R describen el recorrido de revisión; no se confunden con los
IDs de hallazgos ni con el estado automático del ledger.

El ledger fija un SHA previo para evitar autorreferencia. Los archivos
añadidos por esta ola se recogen en el siguiente delta. `check` debe probar
cobertura exacta de esa base; `delta` identifica añadidos, modificados y
eliminados. Un blob modificado invalida la evidencia dependiente, conservando
historial. No se importan secretos, datasets privados ni logs operativos.

La herramienta invalida cambios del objeto propio; no resuelve un grafo de
dependencias semánticas. Si cambia un productor, datos, parámetros o entorno,
el revisor debe invalidar manualmente los recibos de sus consumidores aunque
su blob permanezca idéntico. `verificado` exige evidencia asociada al blob,
pero validar el esquema no comprueba que esa evidencia sea verdadera.

Un recibo semántico incluye: ruta/blob/commit, responsable y revisor, alcance,
contrato, productores/consumidores, ecuaciones/unidades, invariantes, escenario
de fallo, severidad y certeza separadas, reproducción, solución, prueba
antes/después, entorno/semilla/datos, límites y SHA integrado.

Los derivados se revisan individualmente por hash, esquema, procedencia y
consumidor, más una auditoría de su generador. No se simula una lectura
semántica de binarios. Toda exclusión debe quedar motivada; el tamaño o
la extensión no justifican omitir una ruta.

## 6. Auditoría de vetos: composición y recuperación

| Tipo | Condición que debe preservar |
|---|---|
| Exchange/integridad | Identidad, precio/cantidad, reglas actuales, datos finitos, causalidad y terminales. |
| Solvencia/riesgo | Capital y margen disponibles, exposición, pérdida al stop, ruina y política del operador. |
| Evidencia | Incertidumbre, antigüedad, familia de pruebas, multiplicidad y calibración. |
| Heurística | Beneficio incremental medido; documentar si se elimina, suaviza o sustituye. |

Cada veto necesita causa tipada, denominador, origen/versionado del límite,
unidad, entradas, tasa de activación, interacciones, resultado contrafactual,
coste de oportunidad e itinerario de recuperación. Probar umbral±epsilon,
NaN/Inf/overflow, moneda barata/cara, capital micro/normal, ambos sentidos,
evidencia fría/antigua y composición con todos los fallbacks posteriores.

Un cero devuelto por un guard no puede reabrirse como apuesta mínima mediante
mezcla o clamp. La suavidad no es requisito para un fill discreto o una regla
de exchange. Una función tanh suave puede saturarse y seguir siendo inútil.
No suprimir límites para alcanzar la meta; primero medir por qué bloquean.

Prioridad recuperada: contrato `c07_ruin_composition_contract.rs` del worktree
anterior y cambios locales de `ruin-input-contract`. Su estado es **pendiente
de reejecución/revisión**, no integrado por aparecer en disco.

## 7. Autoadaptación y evolución: criterios verificables

No basta con mutar genes. El ciclo completo debe demostrar observación causal,
predicción previa, etiqueta madura, puntuación contra nulos, adaptación,
selección OOS, publicación atómica, consumo correcto y rollback trazable.

1. Dataset congelado por hash, versión de física y generación; separar
   entrenamiento, selección y examen final. Reutilizar OOS para elegir candidatos
   convierte ese conjunto en validación; hace falta un examen nuevo/sellado.
2. Retornos de portafolio MTM en reloj definido, incluyendo posiciones abiertas
   y costes; el muestreo frecuente no crea innovaciones independientes.
3. Multiplicidad acumulada entre rondas, instancias y reinicios; registrar
   familia, candidatos evaluados y presupuesto de error, no sólo población nominal.
4. Separar dispersión entre estrategias de error estándar del Sharpe de una
   estrategia; modelar dependencia temporal y no normalidad.
5. Promoción reproducible contra incumbente congelado, con contratos de costes,
   drawdown, capacidad y pérdida de habilidad. Rollback al padre real, conservando
   evidencia rechazada para evitar redescubrimientos falsos.
6. Paridad sombra/replay/vivo por identidad de parámetros, entradas y relojes;
   reentrenar/regenerar evidencia al cambiar la física que produjo el dataset.

Hallazgos de esta ronda quedan en el informe R4: capital realizado presentado
como MTM, reloj mixto de retornos, DSR sin control de dependencia y contador
de ensayos reiniciado por el caller. Se registra alcanzabilidad por flags;
no se atribuye a la ruta viva por defecto una función opt-in.

Durante la revisión de ramas aparecieron otros pendientes de datos/genoma
y recarga: [residuales de integración](audit/RESIDUALES_INTEGRACION_R4_2026-10-07.md).
Los dos [contratos de continuidad](audit/CONTINUIDAD_R4_2026-10-07.md) conservan
estado estático hasta su replay y evaluación contrafactual. Integrar ramas
no convierte estos pendientes en hallazgos cerrados.

## 8. Consejo de agentes y sincronización

Estado observado, no aceptación inventada:

- Qoder: `qoder/ola67-lows-limpieza`, worktree `.ola67`, cambios activos.
- GLM: `glm/h2-7-paridad-flow`, checkout compartido, `flow_impulse.rs` activo.
- Antigravity: Ω16 (`adeb8d1b`) integrada; no nueva reserva confirmada.
- Claude: línea de ejecución histórica; no acuse nuevo de esta ronda.
- Codex: censo y fundamentos; recuperación de ramas anteriores en worktree
  aislado. Tres subagentes: estadística, inventario y reconciliación Git.

Reparto propuesto: Qoder R2/R3 y consenso; GLM R4/R6 y datos/aprendizaje;
Claude ejecución R5; Codex censo, contratos transversales y R9. Son propuestas
hasta acuse. Antes de editar, publicar rutas/SHA y re-grep del símbolo. El
buzón local permite avisos inmediatos; la bitácora versionada es
`COORDINACION_CODEX_2026-09-28.md`. Publicar aviso no equivale a recepción.

Prohibidos `add -A`, `stash -u`, reset del árbol compartido o limpieza de
worktrees ajenos. Stage sólo rutas propias. Conflictos documentales se
resuelven por unión; conflictos de código por contrato y pruebas, revisando
ambos padres aunque Git no marque conflicto.

## 9. Recuperación Git, pruebas y publicación

La auditoría encontró `codex/root-audit-2026-10-04` publicado pero sin merge,
satélites locales y `model-reload-contract` con commits exclusivos. No todos
los cambios habían llegado a main. Orden propuesto tras inspección:
root-audit → OOS → SA → forest → feature-clock → evidence-expiry → model-reload.
Los worktrees sucios `ruin-input-contract` y `codex-review-2026-10-07` se
preservan; sus archivos no versionados requieren revisión separada.

Antes de cada commit: status, staging explícito, diff y búsqueda de conflictos.
Cada merge exige diff contra CADA padre y `cargo check --workspace --all-targets
--locked` antes del commit según las reglas del proyecto. Luego ejecutar
tests de contratos afectados y regresión de consumidores. Cambios a pipeline
vivo conservan el oráculo T-1 y su trinquete; no reducirlos ni reutilizar el
resultado de otro SHA. T-1 mide expresividad, no rentabilidad.

Por prueba: SHA/árbol, toolchain, comando, variables pertinentes, semilla,
datos por hash, salida/exit, duración y cobertura. No promover una fórmula
portada a Python como si fuera ejecución del Rust: es un experimento sobre
supuestos y se etiqueta así. Compilación, tests, CI y OOS son recibos distintos.

Push normal (sin force), fetch y verificación del SHA remoto/ancestro. El
avance de main por otro agente requiere reintegración y revalidación aplicable.
Sólo borrar ramas cuyos commits estén contenidos en main y que no estén
reservadas/activas; conservar archivos nuevos aunque su rama ya esté fusionada.
No borrar worktrees sucios ni interpretarlos como basura.

## 10. Próximos lotes y criterios para declarar avance

1. Cerrar R0 con erratas y ledger comprobado, sin afirmar revisión de 1.434 archivos.
2. Registrar/reproducir R2 estadística y R5 composición de riesgo; acordar
   ownership con los agentes que modificaron esos consumidores.
3. Recuperar ramas probadas y dejar recibo de lo integrado y lo pendiente.
4. Para cada lote del censo: lectura completa → hipótesis → reproducción →
   corrección → revisor → pruebas → merge → refrescar hashes/delta.
5. Ejecutar examen económico sellado con costes y estrés; publicar resultados
   negativos y tamaño efectivo de muestra. Sólo ese examen puede aportar
   evidencia sobre crecimiento; aún no está satisfecho.

El cierre de esta ola documentará resultados realmente ejecutados y bloqueos.
No se declara ausencia total de bugs, rentabilidad consistente ni revisión
completa por el mero hecho de tener un plan o un inventario.
