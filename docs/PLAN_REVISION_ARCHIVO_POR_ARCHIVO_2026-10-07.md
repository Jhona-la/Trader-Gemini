# Revisión desde la base: plan y recibos por archivo

## Estado vigente — 2026-10-08, candidato de integración R4

Esta adenda prevalece para estados operativos; las secciones 1–10 conservan
el recorrido y sus snapshots del 7 de octubre. La sección 11 de Sol se
preserva íntegra como bitácora con su propio cierre. No se transfieren
porcentajes históricos de revisión a esta composición.

Base de código: `e9c6a435eb1780da7f2c6e29103f3bd7d227d94c`, árbol
`10b8a427dfbdd5b203595b039b2dad50c69d6f68`; padres `7acf36aa` y
`5842c8e3`. El código recuperado y las aportaciones remotas están unidos en
ese commit local. **T1: APROBADO en e9: 16/144 sensibles (11,1%), mínimo 0,110 intacto, 1 test exacto y 1 filtrado; fixture sintético; publicación pendiente del recibo remoto.**
Un commit de integración no acredita por sí solo llegada a `main` remoto.

El censo nuevo [ledger anotado de código](audit/REVISION_CODIGO_2026-10-08.json) fija ese commit y contiene
**1.478 rutas versionadas, 474 Rust y 655 bajo `graphify-out`**. La generación
inicial dejó las 1.478 filas en `inventariado`. Una anotación posterior sobre
los mismos blobs registra **1.469 inventariado, 8 hallazgo_abierto y 1
verificado**. Los ocho hallazgos describen alcances parciales por contrato,
con blob, informe y `whole_file_closed=false`; no son ocho archivos completos
auditados. El único `verificado` es el archivo de tests
`crates/risk-engine/tests/c07_ruin_composition_contract.rs`: lectura 1–139,
revisión independiente y C07 4/4 dentro del recibo de riesgo 54/54 en
`7acf36aa`. Ese cierre no se extiende a `risk-engine/src/lib.rs` ni certifica
configuraciones completas o resultados económicos.

El check posterior de la anotación tuvo exit 0: valida estructura, cobertura
y metadatos de blobs, no la verdad de las conclusiones ni una auditoría total.
[Delta de integración](audit/DELTA_INTEGRACION_2026-10-08.json) conserva el contraste entre bases. Los
1.434/460 iniciales y 1.474/473 de `f3f8696` son snapshots distintos.
El cierre documental posterior requiere un siguiente delta por SHA; no se
incluye a sí mismo por autorreferencia ni se declara ese censo aún generado.

| Frente | Estado de esta composición y continuación |
|---|---|
| Recuperación Git | Root-audit, satélites y model reload están incorporados en la línea integrada. Faltan publicación verificada y censo final de refs antes de limpiar. |
| Riesgo C07 | Guard y contrato incorporados; el registro de 209 tests pertenece a `f3f8696`, no a este candidato. RED usó el árbol `6d6eb1daf1e27056664f01aa347da208c355de09`; no es el árbol corregido `462e1c69c4d08c7e95f22a81893ad23855bb5495` de `f3f8696`. Los recibos y su alcance quedan en [recuperación de ruina](audit/RECUPERACION_RUIN_R4_2026-10-07.md). |
| Aprendizaje | El caller mantiene un `Arc` compartido de Darwin entre workers; sigue pendiente persistencia entre reinicios. Q2 conserva cash/DD de cierres; Q3 quitó el disparador extra de cierre, pero gaps de ticks aún producen duraciones desiguales. DSR es nominal; dependencia, contaminación de entradas y dispersión entre ensayos requieren contratos nuevos. |
| Medición Sol | SOL-R5-01 de reporting diario ya fue publicado en su línea (`f8c433f8`) y está incorporado aquí. Su conciliación diaria y dos tests históricos no certifican equity común del core vivo, costes completos ni crecimiento OOS. |
| Continuidad y extensiones Ω24..Ω28 | OU-R4-01 y OU-R4-02 CERRADOS (Ola Ω28, `c0ccdf5f`): exclusividad de modo SDE continuo (abstinencia honesta sin caída legacy) y monotonicidad temporal estricta ante timestamps no crecientes (sin mutar estado acumulado ni reloj). 40/40 tests verdes en `strategy-core`. Incorporado a `origin/main` (`a5c69af3`). |

Reservas observadas, sin inventar acuses: Qoder mantiene
`qoder/ola72-lows-residuales` en `.ola72`; Antigravity mantiene
`antigravity/quant-sr-ronda6-f4-riesgo-ejecucion` en `.antigravity`. GLM112
está incorporado. Sol cerró/publicó reporting y su checkout separado se
preserva como evidencia; no se infiere actividad por conservar una ref.
Codex coordina esta recuperación, recibos y lotes R0–R9. Antes del próximo
lote se renuevan owner, rutas, base y revisor. La eliminación propuesta de
knobs por Qoder debe acordarse con el contrato GLM112 que los conserva
configurables; las modificaciones concurrentes de `quantum_oscillator.rs`
y `soliton.rs` no se importan de archivos sucios por su mera presencia.

La limpieza se limita a refs con nombre/OID verificados, ancestros de `main`
y `origin/main` publicado, sin actividad reservada ni commits exclusivos.
`main`, Qoder, Antigravity y toda rama externa nueva quedan fuera. Un checkout
histórico sucio sólo admite detach al MISMO HEAD, sin force, con status y
hashes de archivos iguales antes/después; se conserva el directorio y todo
archivo. Cualquier discrepancia aborta. Ningún borrado automático de worktrees.

Próximos recibos: validación aplicable del candidato y T1, publicación con
SHA remoto comprobado, delta documental/censo de refs y lotes semánticos
por blob y consumidor. La revisión económica exige protocolo causal,
costes/MTM, modelos y datos identificados y examen OOS separado de selección.
Ninguno de esos resultados se presume por completar estos documentos.


Validación aplicable: 736 regresiones aplicables + T1 (1 test exacto, 16/144 sensibles, mínimo 0,110 intacto); total 737 aprobadas, 0 fallidas y 1 ignorada. 729 aprobadas se retienen por identidad desde 7acf y 8 se ejecutaron en e9, incluido T1. Check all-targets aprobado antes del merge; estos resultados no validan crecimiento económico. Los recibos conservan sus SHA
de ejecución: la matriz de equivalencia compara todas las entradas
versionadas y 530 inputs compilables/de fixtures, con modo/kind/OID.
Reporta por separado los dos archivos afectados y reejecutados en
`e9c6a435`: reporting continuo 2/2 y stateful_open 5/5. Los estados
externos no registrados por Git tienen límites explícitos; no se
finge que las suites anteriores corrieron en este SHA.

## Contexto histórico de inicio R4 — 2026-10-07

Fecha: 2026-10-07, America/Bogota. Coordinación inicial R4: Codex; revisión independiente y continuación: Sol.
Base histórica de R4: `adeb8d1b1f8171b14fffa8abc9f54c396e4794dd`.
Base nueva verificada por Sol: `8938cf41d5b2d181e807de1a3fa41f36b09dfc32`.
Ramas Codex: `codex/quant-foundations-2026-10-07` y `codex/integration-recovery-2026-10-07` (trabajo activo preservado).
Rama propia Sol: `sol/plan-auditoria-2026-10-07`, worktree aislado `.sol-plan-2026-10-07`.
Estado: **plan operativo recuperado con atribución; auditoría reiniciada; cobertura semántica y validación económica abiertas**.

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

Los dueños por lote de §11.2 son propuestas hasta acuse. Antes de iniciar
un lote R0–R9 se fijan escritor y revisor, rutas/blobs, comprobaciones
apropiadas a su tipo, criterio de aceptación y recibo de salida. Para runtime
incluir comportamiento adverso y regresión; para documentos/derivados,
esquema, procedencia, enlaces y consumidores. No se inventa ejecución por
completar el censo ni se exige una prueba duplicativa a cada archivo.

R0→R1→R2/R3 preceden interpretar el código como teoría válida. Después se
priorizan rutas que pueden perder capital o contaminar evidencia, y luego
el resto del censo. Ninguna fase se cierra por ausencia de matches de `rg`.

Cada lote debe ser acotado (orientativamente 10–20 archivos; módulos grandes
en sublotes por contrato), con un escritor y revisor independiente. La lectura
de un archivo completo es distinta de revisar una función. Un archivo de
10.000 líneas puede requerir varios recibos; el estado completo exige unir
los intervalos y todas las rutas dependientes.

## 5. Inventario exhaustivo y recibos

> Snapshot de inicio del 7 de octubre. Para estados, cifras y reservas actuales
> prevalece la adenda del 8 de octubre; se conservan los contratos metodológicos.

La base contiene **1.434 archivos versionados, 460 Rust y 23 crates miembros
más el paquete raíz**. De ellos, 655 pertenecen a `graphify-out`: el total
incluye derivados, no sólo código. El conteo anterior de ~377 Rust no se
traslada a esta base. [Ledger histórico de inicio R4](audit/REVISION_2026-10-07.json),
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

> Snapshot de inicio del 7 de octubre. Para estados, cifras y reservas actuales
> prevalece la adenda del 8 de octubre; se conservan los contratos metodológicos.

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

> Snapshot de inicio del 7 de octubre. Para estados, cifras y reservas actuales
> prevalece la adenda del 8 de octubre; se conservan los contratos metodológicos.

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

> Snapshot de inicio del 7 de octubre. Para estados, cifras y reservas actuales
> prevalece la adenda del 8 de octubre; se conservan los contratos metodológicos.

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

> Snapshot de inicio del 7 de octubre. Para estados, cifras y reservas actuales
> prevalece la adenda del 8 de octubre; se conservan los contratos metodológicos.

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

> Snapshot de inicio del 7 de octubre. Para estados, cifras y reservas actuales
> prevalece la adenda del 8 de octubre; se conservan los contratos metodológicos.

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

## 11. Continuación Sol: reinicio verificable, no plan rival

Se reutilizan con atribución el plan y las herramientas de Codex del commit
`639c3e0d`. Los informes históricos siguen ligados a sus blobs originales.
No se importan sus fixes de runtime ni sus archivos no versionados, y no se
atribuye a Sol la creación del censo o del diagnóstico estadístico original.

El ledger `audit/REVISION_2026-10-07.json` se regenera para la nueva base
`8938cf41`, conservando historia cuando proceda. Incluye CADA archivo
versionado, también derivados, configuración, scripts y documentos. La
clasificación automática no acredita lectura. Los archivos no versionados de
otros worktrees quedan como alcance pendiente de su dueño, no como basura.

### 11.1 Escalera de evidencia que deben usar todos los agentes

| Estado | Recibo obligatorio | Qué NO acredita |
|---|---|---|
| Inventariado | Ruta, blob, modo y base exacta | Lectura o corrección |
| Inspección parcial/completa | Rangos leídos, contratos, dependencias, autor | Ejecución correcta |
| Hallazgo reproducido | Fixture, resultado esperado/observado, comando y exit | Impacto económico real |
| Corregido local | Regresión RED→GREEN y diff propio | Integración |
| Integrado | SHA en main, diff contra padres y regresión de consumidores | Rentabilidad |
| Validado económicamente | OOS sellado, costes, riesgo, incertidumbre y protocolo | Garantía de crecimiento futuro |

El esquema actual del ledger conserva sus cuatro estados técnicos; esta
escalera se registra en `evidence.kind/result` y en recibos de revisión, sin
convertir un campo `verificado` en certificación universal. `check` valida
metadatos, NO la verdad de una revisión. Un archivo enorme sólo completa
lectura al unir sus rangos; un consumidor requiere revalidación si cambian
productor, modelos o datos aunque su propio blob permanezca idéntico.

### 11.2 Lotes prioritarios y coordinación propuesta

| Lote | Rutas concretas | Recibo y límite | Dueño propuesto |
|---|---|---|---|
| R0-A | Plan compartido, este plan, BARRIDO, ADR-0003, Cargo manifests, ledger | Meta y unidades reconciliadas; denominador por SHA; mapa de ramas y acuses | Sol + revisión Codex |
| R1-A | `crates/god-engine-core/src/lib.rs`, `src/bin/god_engine.rs`, `crates/god-engine-core/src/reality_physics.rs`, `crates/quantum-arena/src/position.rs` | Productor→consumidor por evento, activo, slot y generación; ninguna etiqueta sustituye trazabilidad | Consejo núcleo; acuse pendiente |
| R2-A | `selection_stats.rs`, `darwin.rs`, `online_daemon.rs`, `return_evidence.rs` | Reloj, MTM, dependencia, familias y uso de holdout; no subir N fabricando observaciones | Codex/GLM; Sol revisor |
| R3-A | `temporal_spectrum.rs`, `espectral_multiactivo.rs`, `evalues.rs`, `skill_motores.rs`, Hawkes/OU/Hodge | Fuente primaria, unidades, nulo y ablación; distinguir continuo y malla numérica | Qoder/Antigravity; acuse pendiente |
| R4-A | `tick_replayer.rs`, loaders de datos, `OmniHistory`, trainers y manifests | Cobertura real 7/15/30/180 días, orden, disponibilidad causal, hashes y fuentes sintéticas | GLM/Codex; acuse pendiente |
| R5-A | `booktick_replay.rs`, `continuous_evolution_backtest.rs`, `backtest_windows.rs`, `audit_forensic_backtest.rs`, `metrics.rs` | Cash/equity diario/terminal, duración ejecutada, fees/funding, reservas e insolvencia | Sol diagnóstico; runtime por acuerdo |
| R5-B | `envio.rs`, `correlation_guard.rs`, `ruin.rs`, executor/reconciliation, `veto_registry.rs` | Veto tipado, composición, coste de oportunidad y recuperación; exits defensivos | Claude/Codex; acuse pendiente |
| R6-A | `genome.rs`, `genome_store.rs`, `ml_inference.rs`, `ml_registry.rs`, promotores | Fenotipo evaluado==servido, publicación antes de activar, concurrencia/rollback e identidad | GLM/Codex; acuse pendiente |
| R7-A | Workflows, Cargo targets, locking, storage y telemetría | All-targets, determinismo, fallos inyectados, recursos p50/p95/p99 medidos | Consejo plataforma |
| R8-A | TODOS los restantes del ledger | Cada ruta con generador/consumidor o exclusión explícita; no cierre por extensión | Dueño asignado con acuse |
| R9-A | Recibos anteriores y harness final | Regresión en tip integrado + examen OOS multiactivo; pérdidas y no-operación incluidas | Revisor independiente + operador |

Los lotes son rutas iniciales, NO exclusiones del resto de archivos. Procesar
por prioridad de daño/contaminación y dependencia, después por ruta estable.
Cada reclamo de archivo debe publicar base/blob, escritor, revisor, rangos y
entregable en el buzón. No asignar aceptación o disponibilidad a otro agente
sin respuesta. Un conflicto documental se resuelve por unión con atribución.

### 11.3 Hallazgos nuevos y convergencia

- R4-Q1..Q4 de Codex siguen abiertos o pendientes de integración en esta base.
  Sol confirmó el caller que recrea Darwin, cash presentado como MTM y reloj
  mixto. El fix del caller ya está en rama Codex: no duplicarlo.
- SOL-R5-01: el backtester continuo usa `equity_final - cash_inicial_dia` y
  suma resultados diarios. Flotante arrastrado se vuelve a contabilizar;
  probar identidad telescópica de equity neta por día antes de cambiar sizing.
- SOL-R5-02: replay compartido termina y calcula drawdown sobre cash; informar
  slots abiertos, unrealized, equity y regla de valoración terminal.
- SOL-R4-01: días solicitados se limitan a cobertura disponible en continuo;
  el replay puede terminar antes y annualizar con el fin original del tape.
  Registrar intervalo solicitado/disponible/ejecutado y causa de fin.
- H2-7 no se reabre: contrato GLM103 `h2_7_paridad_de_ganancias_pinned`
  presente en 8938cf41. La observación antigua de Ola67 quedó superada.

Se entrega evidencia conductual separada de los ports/modelos de trazas.
No se ejecutó un backtest económico 7/15/30/180 días ni un examen final en
esta ola. Sus requisitos y bloqueos quedan visibles; no se sustituye por
proyecciones de capital, inferencias de tests verdes o afirmaciones de excelencia.

### 11.4 Recibo de esta ola y bloqueo de publicación

- Ejecutado: generación y `check` del ledger sobre 8938cf41, 1.434 rutas,
  cero errores de cobertura/esquema; todos los estados siguen inventariados.
- Ejecutado: testigo compilado con `rustc 1.98.0-nightly` que importa blobs
  Rust congelados de `selection_stats.rs` en 8938cf41, no un port Python ni
  una validación de código cambiado posterior. DSR 40→400
  observaciones repetidas: 0.248357355→0.999724533; NaN/Inf añadidos no
  cambian verdict; expresión diaria compilada da 20 acumulado frente a 10
  de equity terminal en fixture controlado. Comando terminó con exit 0.
- NO certificado: suite completa de inventario ni `cargo check --offline
  --locked --workspace --all-targets` iniciados en target propio; no acabaron
  y se detuvieron únicamente los jobs de Sol. Comandos Git posteriores
  terminaron con SIGTERM sin salida útil; no se inventa causa ni resultado.
- Último remoto confirmado: main 8938cf41 y root-audit d0e02da1; CI Replay
  contracts run 37660783497 success para 8938cf41. No es CI del plan nuevo.
- NO realizado: commit, push, merge, refresh final del remoto, eliminación
  de ramas ni examen económico. Worktree Sol y fuentes de otros agentes
  preservados. Continuar publicación cuando la ejecución vuelva a responder,
  con status/diff/ancestría/pruebas nuevos; no reutilizar el verde de otro SHA.

### 11.5 Continuación tras recuperar ejecución (16:02 Bogotá en adelante)

- Main remoto refrescado a `171db3c6`; el censo/testigo 8938cf41 sigue
  histórico y congelado, no una certificación de ese main posterior.
- Suite completa `scripts/tests/test_audit_inventory.py`: 7/7 PASAN,
  198.212 s. Ancestría de cada rama local/remota y status de cada worktree
  completados: cero candidatos seguros de borrado. Recibo:
  `audit/SOL_RAMAS_2026-10-07.md`. No borrar review-plan/Sol ocupados/sucios;
  root y recovery mantienen 28/64 commits exclusivos contra main al corte.
- Reserva nueva: SOL-R5-01 en `continuous_evolution_backtest.rs`, únicamente
  reporting diario. Helper extrae valuación existente sin alterar fees/fallback;
  tests ejercitan slots reales, carry positivo/negativo, multiactivo, cierre y
  cash. LOCAL TEST PASSED: full-bin reporting_contract 2/2, direct exit 0;
  source blob 5a0af98e3fefc168feb2c1649b8d2d94d25c909e, source SHA256
  9a3ea9e2ca8ec1c0b4b0f1fd4dfc4c26e9f4b6c0d79c46116b2eaf265ed02459.
  Receipt: target/reporting-contract-execution-evidence-20261007.txt.
- Baseline offline locked workspace all-targets check passed in the own
  target, cargo/tee exits 0: target/sol-baseline-all-targets-20261007.log
  and .exit. Integrated all-targets and reporting_contract validation,
  commit and publication are
  pending; prior base success is not integrated-candidate evidence.
- T-1 was not rerun: daily reporting only, not the live strategy pipeline.
  No live orders, engine execution, strategy/sizing/shadow-selection changes,
  model/arming changes, or economic validation. Historical receipts above
  remain historical; no current CI or future publication success is claimed.

### 11.6 Cierre de integración SOL-R5-01 — publicación verificada

- Commit propio `ac6c945634d7bbb20a0ae9f19382babe660b7de5`: plan/censo
  atribuídos y corrección únicamente de reporting diario. Merge publicado
  `f8c433f8c99d230b1708cf850b44fa0204eba25b`, padres ac6c9456 y
  `e285193ef3397c65fe957e6191d5b1296806e1df`; árbol probado
  `89ef5c61b9dd1bb879ea81d32e5d2e4bd8198b9e`. SHA remoto confirmado por
  ls-remote y ambos commits propios acreditados como ancestros (exit 0).
- `cargo check --offline --locked --workspace --all-targets` en candidato
  integrado: cargo_exit=0. Test full-bin `reporting_contract`: 2/2 PASAN,
  cargo_exit=0. No transferencia del verde anterior. Review contra AMBOS
  padres preservó documentación por unión y cambios nuevos de Qoder/GLM;
  sólo las 18 rutas autorizadas difieren respecto de main integrado.
- Helper real calcula `equity_final - equity_previa` y el porcentaje usa
  equity previa. Acumulado telescópico y residual final expuesto. Tests con
  arena real prueban carry positivo/negativo, cambios de marca, cierre,
  comisiones en cash, dos activos y múltiples slots. No se altera sizing,
  selección shadow, fees/marks/fallback existentes ni promoción del motor.
- Fuente integrada: blob `4361b2b8557f34d28b6cea172a4561f3af79d404`,
  SHA256 `7f8503371f84d27e890d9e3d5575695baa2ed6b430c8a95674c3489b7dd53b6c`.
  Verified publication receipt: `target/sol-publication-evidence-20261007.txt`
  in the Sol worktree. Behavioral log: `target/sol-integrated-reporting-contract-20261007.log`;
  both-parent review: `target/sol-integrated-parent-review-20261007.json`.
  These are local, ignored receipts, not repository-relative published artifacts;
  no `outputs/` copies exist in this worktree.
- El counterejemplo histórico sigue congelado en 8938cf41; la etapa RED
  completa del harness no llegó a producir un resultado antes de la corrección.
  No reclamar RED→GREEN del bin completo; sí histórico reproducido + tests
  reales de corrección GREEN. No examen económico, T-1 nuevo ni producción.
- CI remoto del merge: run 37722336175 en progreso al último chequeo.
  No declarar resultado futuro. Publicación Git, all-targets local, tests
  conductuales y CI son recibos distintos.
- Próximo contrato R5: equity/drawdown terminal y duración efectivamente
  procesada en replay compartido, reconciliación fees/funding y posiciones
  abiertas; R2/R6 dependencia DSR/holdout/publicación sigue coordinado con
  las ramas de otros agentes. La auditoría semántica de todos los archivos
  y meta de crecimiento siguen abiertas. Los estados históricos anteriores
  se conservan como bitácora, no como estado final de esta ola.
