# ADR-0017 — Dos frenos (duro de la cuenta y de la estrategia) y estado aprendido persistente

- **Estado**: Aceptado. Implementado en CL-99, CL-106, CL-107 y CL-108; corregido tras la revisión previa de la Línea G (`docs/audit/linea_g/ola5_tanda_cf4.tsv`) en CL-106b, CL-108d, CL-108e y CL-108f, y tras la revisión posterior del PR #62 (`/mnt/project-files/auditoria/revision_pr62_2026-10-17.md`) en CL-130 a CL-134 (§2b).
- **Fecha**: 2026-10-12
- **Responsables**:
  - Decisiones del dueño en sus tarjetas del 2026-10-12: el freno de drawdown («Ambos») y la fase de sonda tras el reinicio («Guardar todo»).
  - Implementación: Claude (Línea C).

## Contexto

Había dos frenos de drawdown distintos y ninguno era una prueba con nivel:
- **Veto de entradas del riesgo.** Con 13 USD el umbral era d* = 63,2 % (ADR-0016 D3). Es un tope de ruina: con −0,10 R por operación, sólo dispara en el 21 % de los caminos de 2 000 operaciones.
- **Sistema inmune del host**, que aplana y echa el kill-switch.
  - Con riesgo medido usaba una prueba de racha: 2,2-3,8 % con 13 USD, 8-12 R.
  - Con +0,30 R de ventaja, falsa alarma del 36-49 % en 200 operaciones.
  - Sin riesgo medido, el gen: 50 %.

Además, el pico desde el que medían la caída no era compartido y se perdía al reiniciar. Y todo el estado aprendido, incluido el recuento de cierres, nacía de cero en cada arranque. Con D4 (ADR-0016), eso dejaba la cartera en una sola posición nueva a la vez durante unos 8 días con 26 monedas.

Los números salen del Monte Carlo de LC-F4-10: 13 USD, orden mínima de 5,10 USD, RR 2,25 y 14,5 pb de fricción.

## Decisión

### 1. Un solo pico (CL-99)

El pico es del arena, sólo sube, lo comparten los dos frenos y el host lo guarda por entorno (`pico_capital.txt`).

### 2. Freno duro de la cuenta (CL-106)

**Umbral** = `K(w, l)·r`, acotado por d*. Es el mismo para el veto de entradas del riesgo y para el sistema inmune del host (`drawdown::umbral_freno_duro_de`). Desde CL-130 la caída se cuenta en stops de cada posición y se compara con `K` directamente (§2b).
- `K` es el cuantil 95 % de la caída máxima en 200 operaciones bajo H0 = +0,05 R, con R el riesgo al stop más la fricción. Se mide en stops con un Monte Carlo determinista.
- `w = (TP − f)/SL` y `l = (SL + f)/SL` son la geometría medida de las órdenes (EWMA).
- `r = s̄/pico`, con `s̄` el riesgo medido al stop en USD (nocional × SL, EWMA). Así la condición es `(pico − capital)/s̄ ≥ K`: la caída en stops, que es lo que mide el cuantil (CL-106b, T1-02 de la Línea G). Con el riesgo en fracción del capital al dimensionar, el umbral se alejaba al caer con nocional fijo: con stop del 2 % y 13 USD, nunca disparaba antes de d*.
- Sin riesgo medido (al arrancar) rige d*.

**Resultado:**
- Umbral de 31-35 R: 8 % de la cuenta con stop del 0,5 % y 15,7 % con stop del 1 %.
- Falsa alarma: ≈ 5 % en 200 operaciones con +0,05 R, y 0 con +0,30 R.
- Detección: en ~520 operaciones si no hay ventaja, y en ~220 con −0,10 R.

**Límite conocido:** una caída medida desde el pico móvil sólo tiene nivel con horizonte. En 2 000 operaciones la falsa alarma sube al 52-57 %. Por eso hay un segundo freno.

### 2b. Revisión posterior del PR #62 (CL-130 a CL-134, 2026-10-17)

**La caída se cuenta en stops de cada posición (CL-130, R1).** `K` mide la caída máxima en stops de operaciones que ganan `w` o pierden `l` stops de su propia posición. La condición `(pico − capital)/s̄ ≥ K` sólo es esa caída con tamaño fijo. Desde el dimensionado en espacio de riesgo (LH-04) y el factor de racha (GLM 118), el tamaño baja en una racha perdedora: `s̄` encoge y la misma caída en USD contaba como muchos más stops. Con L = 10 disparaba tras 13 SL en vez de 31, y con +0,30 R de ventaja la falsa alarma subía al 4,5-23 %.
- Cada cierre del núcleo suma su resultado en stops de su posición: PnL neto / (cantidad × |entrada − SL|). La cuenta guarda la suma y su máximo (`resultado_en_stops`, `pico_en_stops`), que se persisten con el estado aprendido.
- El freno duro (`drawdown::freno_duro`, el mismo para el host y el riesgo) dispara cuando `pico_en_stops − resultado_en_stops ≥ K(w̄, l̄)`, o cuando la caída del capital desde el pico llega a d*.
- Una posición sin SL propio (adoptada del exchange) no suma: su caída la cubre d*. Tampoco suman los cierres por pierna del exchange que el núcleo no ve (T1-09 de la Línea G, abierto).
- La parte en stops sólo cuenta lo cerrado; las pérdidas abiertas las acotan los stops de las posiciones y d* sobre el capital.
- Sin geometría medida rige sólo d*. Al actualizar, un estado anterior sin la suma empieza con caída 0 en stops.

**Lo medido sale de las órdenes abiertas (CL-131, R2).** El riesgo al stop y la geometría se medían al final de `evaluate_single_intent`, antes de los vetos del núcleo, y la misma intención vetada se medía en cada tick. Ahora el riesgo deja pendiente la medida de la orden validada y el núcleo la confirma al abrirla (`RiskEngine::registrar_orden_abierta`).

**Un solo plano para el pico (CL-132, R3).** El host publica el saldo del exchange en el arena (`capital_exchange`). El host y el riesgo registran el pico y miden la caída con `drawdown::capital_del_freno`, el mínimo de los dos planos. Antes el riesgo subía el pico con el capital del arena, y en papel sobre mainnet una ganancia en papel se veía como una caída del saldo real.

**Un pico por venue, y un pico ilegible no es un pico ausente (CL-133, R5 y B2).** Testnet y papel de mainnet comparten `data/demo`, así que cada venue guarda su pico en su archivo (`drawdown::archivo_pico`). El archivo sin venue de antes se adopta una vez y se renombra. Si el archivo es ilegible, no entran posiciones nuevas y no se sobrescribe.

**Replantar el bosque sombra reinicia los frenos de cada universo (CL-134, R6):** el pico, lo medido, la caída en stops y el freno de la estrategia.

### 3. Freno de la estrategia (CL-107)

**Prueba:** un proceso e de Ville (`VilleEProcess`) sobre la R de cada cierre.
- H0 = +0,30 R por operación, la ventaja que pide la meta.
- Rechaza con M ≥ 20 (α = 0,05).
- La falsa alarma queda ≤ α para cualquier momento en que se mire.
- Detección: E[R] = 0 en ~117-136 operaciones; −0,10 R en ~63-79.

**Al rechazar:**
- El riesgo no admite entradas nuevas (`estrategia_falsada`).
- Las posiciones abiertas se siguen gestionando.
- No aplana ni dispara el kill-switch: eso es del freno duro.
- Queda echado. Otra estrategia es otra hipótesis, y para soltarlo se borra el estado.

### 4. Estado aprendido persistente (CL-108)

**Qué se guarda, por símbolo:**
- el recuento de cierres y la evidencia por moneda (acierto, PF, Kelly, PnL);
- las observaciones de los calibradores de Platt, por moneda y global, que se reproducen al restaurar;
- el sesgo y el umbral epigenéticos, con el ancla del umbral (el último cierre), y el anillo de siniestros de Cramér-Lundberg (CL-108e, T2-2).

El PnL de las sesiones anteriores se guarda aparte. No entra en los acumuladores de la sesión, que leen el demonio en línea, la deriva frente al control y el validador de paridad (CL-108d, T2-3).

**Qué se guarda además, para toda la cuenta:**
- el freno de la estrategia;
- el riesgo (en fracción y en USD) y la geometría medidos;
- la caída del freno duro en stops (CL-130).

El host restaura antes de operar y guarda en cada cierre (`estado_aprendido.json`, por entorno). La escritura se sincroniza con el disco antes de renombrar, y un estado que no se podría volver a leer no se escribe.

**Un archivo ilegible no es un archivo ausente (CL-108f, T2-1).** Si el archivo está corrupto, truncado, tiene un valor fuera de rango o es de otra versión:
- el freno de la estrategia queda echado (no se sabe si lo estaba);
- la sesión no guarda nada, así que el archivo queda intacto hasta que el operador lo repare o lo retire;
- las posiciones abiertas se siguen gestionando.

## Consecuencias

- **Con 13 USD el veto de entradas se aprieta:** antes d* (63,2 %), ahora ~8-16 % con stops del 0,5-1 % (unos 40 stops de la orden mínima; ~31 % con stop del 2 %).
- **El host deja de aplanar por rachas normales:** con ventaja real, antes aplanaba en el 36-49 % de los caminos de 200 operaciones.
- **Una caída bloqueante sobrevive al reinicio** (CL-99): el pico guardado no perdona.
- **Tras un reinicio las monedas conservan su historia.**
  - Las que tienen 5 cierres o más no vuelven a la fase de sonda.
  - Una moneda sin `_MOTOR` sigue vetada por el ML-GATE tras su primera sonda, y ahora también tras reiniciar. Es la parte abierta de F3.4-05: la evidencia realizada no cuenta en ese gate, y es una decisión pendiente.
- **No se persisten todavía:**
  - la evidencia por rama (`TasaAcierto`);
  - los calibradores conformales;
  - la ganancia epigenética de las 32 escalas del espectro;
  - los pesos del ensamble y del perceptrón.

## Alternativas descartadas

- **Mantener la prueba de racha del host.** Su falsa alarma con ventaja real hace inoperable la cuenta.
- **Sólo el umbral α, sin Ville.** No tiene nivel más allá de su horizonte.
- **Sólo Ville, sin freno duro.** Ville prueba la ventaja, pero no protege la cuenta de una caída rápida dentro de su tiempo de detección.
- **Persistir sólo el recuento (P1).** Las monedas con modelo saldrían de la sonda con el calibrador vacío.
