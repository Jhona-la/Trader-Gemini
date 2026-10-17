# B15 — dominio de parámetros y estabilidad de Soliton/Stochastic

Unidad parcial de la revisión desde fundamentos, 15 de octubre de 2026.
Base Ω131 `dfa4ae6b7d5f89757cb24a70c012cc90534e376e`; fuente reparada
`208c3c07b5fc4cd4b289cc94e51700ca1c6b5a8f`.

El recorrido por todos los archivos sigue el paquete BASE15:
`docs/PLAN_MAESTRO_REVISION_BASE_UNIVERSO_ESPECTRAL_2026-10-15.md`,
`docs/audit/PLAN_BASE15_2026-10-15.md` y su inventario/cola/validación.
Su snapshot `b9cd773538fae9467751883eafdf3ee6ed5142f9` contiene 1863 rutas;
el paquete está en la rama de BASE y su integración no se presupone aquí.
Esta adenda agrega **cero certificaciones de archivo completo**. El estado
de cada hunk y de cada teoría se conserva separado de ese censo.

BASE aclaró la numeración: constructores y fronteras de dominio pertenecen
a la familia B15-02/B15-14; B15-03 es escritor/persistencia de parámetros y
B15-04 sus lectores productivos. Repetir un patrón en dos módulos no crea
otra causa única. F3.6-05, sobre resonancia física, permanece abierto.

## Problema y cambio verificable

Ambos constructores parametrizados asignaban directamente un struct que
sus builders/updates rechazaban. Los campos públicos y los overrides
también podían eludir el dominio declarado. En resonancia, ganancias
`base=-2, peak=-1` invertían una señal positiva. Un pico `100` o un Q `0.06`
eran aceptados por el lector pese a estar fuera del validador público.

Los constructores ahora usan el builder validado: una entrada inválida
conserva los defaults. La evaluación de un objeto cuyo campo público fue
invalidado devuelve cero sin reparar ni mutar ese objeto. El helper SR
también rechaza parámetros inválidos antes de calcular. Cada override se
valida de forma independiente y conserva el campo local configurado cuando
falla; otros overrides válidos siguen aplicándose. No cambian los rangos
de `validate`, los defaults ni las interfaces públicas.

La ganancia SR implementada, fuera de sus suelos/saturaciones numéricas, es

\[
F(s,v)=s\{b+(p-b)\,\operatorname{sech}[q\log(|s|/\sqrt v)]\}.
\]

Con `s=±1e308`, `v=1e-6` y defaults, el cociente intermedio desbordaba.
La rama no finita lo convertía en log-cociente cero, seleccionaba ganancia
máxima 2 y finalmente devolvía cero por overflow del producto. El resultado
representable del perfil lejos del pico es aproximadamente `±1e308`.
Se conserva la división/logaritmo originales cuando el cociente cabe;
sólo ante ese overflow se calcula `ln(|s|)-ln(sigma)` antes de la saturación
numérica existente. El dominio validado garantiza `sigma>0` y Q finito.
El overflow de un resultado matemático no representable conserva su
política previa de devolver cero; esta reparación no redefine esa política.

El productor identificado de `weak_alpha_signal` en Core la acota a
`[-1,1]`. El extremo `1e308` demuestra una frontera de API; no es evidencia
de una operación real perdida ni de una mejora de rentabilidad.

## Teoría y adaptación que siguen pendientes

El código SR es un kernel determinista de ganancia. La resonancia estocástica
física exige relacionar señal débil, umbral/barrera y ruido con una mejora
observada de la respuesta. Un perfil de ganancia por sí solo no acredita ese
mecanismo. Véase [Gammaitoni et al., Rev. Mod. Phys. 70, 223](https://journals.aps.org/rmp/abstract/10.1103/RevModPhys.70.223).
El contrato ignorado `cf3_6_resonancia_estocastica_tiene_maximo_interior`
conserva su aserción y su estado abierto; no se ha ajustado para producir verde.

Soliton conserva su pulso `sech`, su voto espectral `tanh`, el cálculo de fase
y los gates `mid_price>1e-8`, `|velocity|>1e-12`. Su continuidad/covarianza
en esas fronteras y el significado de las alternativas de velocidad requieren
un contrato de unidades. El productor de `microstructure_noise_variance`
usa `(atr_pct*0.1).max(0.00001)`; relacionarlo con una varianza compatible con
las señales exige contraste dimensional y empírico. Core/arena no se editan.

Un campo mutable, AtomicF64 o registry no demuestra un circuito adaptativo.
Para B15-03/B15-04 hay que seguir observación disponible → estimador →
parámetro/versionado → snapshot coherente → lector decisorio → resultado
maduro → actualización causal y persistencia/restore. Deben medirse también
aislamiento por activo y diferencias entre camino escalar y voto espectral.
Los helpers de envolvente, los suelos y el resto del archivo no se declaran
íntegramente auditados por esta reparación.

## Plan de validación y decisión

1. **F0, observable.** La meta de cada ventana es
   `G72=ln(E_final/E_inicial)>=ln(2)`, con equity neta de fees, funding,
   spread, slippage, fills parciales y exposición abierta. Equity no positiva
   se registra como ruina. WR y una curva sintética no prueban esa condición.
   El crecimiento logarítmico esperado y la restricción de drawdown son
   objetivos distintos: [Busseti, Ryu y Boyd, Risk-Constrained Kelly](https://web.stanford.edu/~boyd/papers/kelly.html).
2. **Fundamentos y código.** Por archivo/hunk: fórmula efectiva, supuestos,
   unidades, soporte, causalidad/reloj, invariantes, productor y consumidor
   reales, oráculo y evidencia por blob. Reutilizar las fases F0–F8 y la cola
   BASE, sin otro maestro o recuento competidor.
3. **Literales.** Distinguir identidades matemáticas, límites de representación,
   cotas de interfaz/hardware y parámetros estimables. Una constante no se
   vuelve correcta por hacerla mutable. Calibrar las magnitudes estimables
   con información pasada y registrar procedencia, ventana, incertidumbre y
   política de fallback; conservar pendientes los rangos sin justificación.
4. **Utilidad del modelo.** Comparar candidato y baseline congelado sobre el
   mismo flujo de eventos causal, con estimación sólo en entrenamiento,
   separación temporal de etiquetas y comparación OOS predefinida. Evaluar
   pérdida/calibración, log-crecimiento neto, drawdown, rotación/capacidad y
   sensibilidad a costes. La dependencia entre activos/ventanas debe entrar
   en la incertidumbre y la multiplicidad; no contar observaciones como IID
   sin justificarlo. Esto aplica también a candidatos RF/boosting u otros.
5. **Promoción y recursos.** Fijar la regla de promoción y el presupuesto antes
   de inspeccionar resultados. Medir latencia/colas/asignaciones y RAM con
   carga representativa e inputs fijados. El testigo funcional de esta ola no
   mide latencia de trading ni certifica nanosegundos. La validación nueva
   agrega comprobaciones; su coste productivo sigue pendiente de medición.

## Evidencia de esta ola

Un único `rustc` nightly 2026-06-30 compiló los dos módulos reales y
`VotoEspectral` real con doubles explícitos de registry/trait. La única
ejecución dio **28 aprobadas, 0 fallidas, 0 ignoradas y 11 filtradas**:
12 contratos nuevos, 15 tests existentes y un test positivo que reproduce
los contrajemplos de los originales capturados. Ese último test no es un
RED 0/12. El RED previo de la revisión exploratoria conserva su alcance aparte.

Compilación: 2.849 s; RSS máximo muestreado del compilador 94,216,192 bytes.
Proceso de test: 1.208 s, runner 0.42 s; RSS muestreado 5,779,456 bytes.
Son observaciones de esos procesos, no picos absolutos del host/árbol ni un
benchmark productivo. Hashes de fuentes antes/después iguales. Cero Cargo,
cero target Cargo nuevo. El [recibo versionado](VALIDACION_SIGNAL_B15_2026-10-15.json)
delimita la evidencia; logs, binario, harness y recibo completo quedan conservados
fuera de los targets del proyecto.

La revisión independiente no encontró bloqueos en las tres rutas funcionales.
**Cargo real, all-targets y CI de la nueva PR están pendientes al escribir este
documento**; su estado posterior se consulta en la PR y sus runs por SHA.
La aprobación de PR67 sobre otro candidato no se transfiere a esta rama.

## Coordinación, integración y recuperación

Se reutilizó el worktree existente en `codex/signal-domain-20261015`, desde
Ω131; PR67 mantiene su ref `de534` inmutable y su propia cola de integración.
ROOT/BASE reservaron estos dos módulos; Hawkes, Core, riesgo, configuración,
maestro y censo quedan con sus dueños. Los cambios de dominio se revisan
contra esos consumidores sin ampliar su reserva.

La cola Codex conserva un build pesado por turno, `jobs=1` y cache con dueño.
El humano decidió mantener activos WorkBuddyAI/GLM: no hay nuevas pausas ni
interrupciones y no se afirma un límite global de una compilación. No se
limpian caches activos, modelos, datos, logs o recibos. Reutilizar checkout
evita otro árbol de archivos; la prueba breve se acordó aparte del lease Cargo.

Antes de integrar, ROOT compara contra cada padre y valida el árbol compuesto
con all-targets y los gates acordados. Una versión posterior de main o de sus
dependencias exige renovar esa evidencia. Reversión de la ola: revertir su
commit funcional en una rama propia, validar los consumidores y coordinar la
composición; no ejecutar rollback de procesos/modelos. No hay migración de
datos ni cambio de serialización en estas tres rutas.
