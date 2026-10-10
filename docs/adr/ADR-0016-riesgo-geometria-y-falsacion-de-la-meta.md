# ADR-0016 — Riesgo en espacio de stop, geometría ligada a τ y caída de falsación

- **Estado**: Aceptado (D2 y D3 implementados; D1 y D4 decididos, se
  implementan en la ola siguiente)
- **Fecha**: 2026-10-10
- **Responsables**: Claude (sesión «elegant», prefijo QS-n). El dueño delegó
  las cuatro decisiones de §30.5 del plan de sincronización («Tú decide»).
  Revisión cruzada: en el PR que publica QS-C22/QS-D2/QS-D3.

## Contexto

La meta es duplicar la cuenta cada 3 días. En crecimiento logarítmico:

```
g = ln 2 / 3 = 0,231049 por día
```

Eso no es una preferencia, es un requisito medible sobre la estrategia. Con
una operación binaria de pago b (TP/SL neto de fricción), probabilidad p de
tocar TP primero y fracción f del capital arriesgada al stop:

```
G(f) = p·ln(1 + f·b) + (1 − p)·ln(1 − f)        (por operación)
f*   = p − (1 − p)/b                             (Kelly)
```

Ejemplo: p = 0,40 y b = 2,25 (edge de +0,30 R por operación). Medio Kelly da
f = 6,7 % y G = 0,01451 por operación. Se necesitan 0,2310/0,01451 ≈ 16
operaciones al día para cumplir la meta.

Hoy el riesgo al stop es ~0,22 % del capital. Con eso, aun suponiendo un
edge de +0,2 R, harían falta ~540 operaciones al día. El ledger QS-R1
(`docs/audit/LEDGER_DECISION_VIVA_2026-10-10.md`) encontró además:

- R-20: el dimensionado micro es binario (orden mínima o nada).
- R-13: el stop se acortaba a 55 pb sin acortar τ.
- R-12: horizontes que no pagan la fricción se admitían en micro.
- R-03: el veto de drawdown micro era el literal 0,85.
- R-18: las primeras 5 operaciones de cada moneda pasan el gate EV con un
  prior 0,55 que nadie midió.

## Decisiones

### D1 — Dimensionar en espacio de riesgo con ½ Kelly sobre la cota inferior

```
p_LCB = cota inferior de Jeffreys de P(TP antes que SL) (mismo z del sistema)
b_net = (TP − f_rt) / (SL + f_rt)                 (f_rt: fricción ida y vuelta)
f_R   = ½ · max(0, p_LCB − (1 − p_LCB)/b_net)     (fracción del capital al stop)
f_R   ≤ clamp_ruin(...)                           (tope de ruina vigente)
nocional = f_R · C / SL ;  margen = nocional / L
```

- El apalancamiento es consecuencia del margen disponible, no una palanca
  de convicción (ADR-0013 ya impide enviar más del validado).
- Si `nocional < min_notional` y `min_notional · SL / C > f_R`: se rechaza
  (D-750). No se infla el tamaño.
- Sin edge medido, `p_LCB ≤ 1/(1 + b_net)` y f_R = 0: no hay crecimiento
  que dimensionar. **Esa es la lectura honesta de la meta**: es alcanzable en
  matemáticas sólo si existe un edge del orden de +0,3 R con ~16 operaciones
  al día; ninguna medición actual lo muestra.
- Implementación pendiente (QS-R4, zona dimensionado de la Línea C): antes
  de activarla, un libro contrafactual en sombra de las intenciones vetadas,
  para que un p_LCB bajo no se convierta en estado absorbente (hallazgo 1 de
  la auditoría del PR #5).

### D2 — La geometría de la orden es la de su τ, en todo capital

- Un τ cuyo stop difusivo no paga la fricción se rechaza
  (`REJ_TP_SL_FLOOR`) también en micro. La puerta QO-586 del núcleo ya
  filtra esos τ antes.
- Se retira el tope micro de 55 pb. Para barreras a (SL) y b (TP) de un
  browniano con deriva μ y volatilidad σ:

  ```
  E[ganancia] = μ · E[T],   E[T] ≈ a·b/σ²   (tiempo medio de salida sin deriva)
  ```

  Acortar las dos barreras un factor k reduce lo que cobra la deriva en k²,
  mientras la comisión por operación no cambia. En unidades de R, el edge
  cae en k y la fricción sube en 1/k.
- Lo que se arriesga en dólares lo acotan el dimensionado y la viabilidad
  (R-19, `orden_viable` con el stop real), no la geometría.
- Implementado en QS-D2. Contrato:
  `crates/risk-engine/tests/qs_d2_d3_geometria_y_drawdown_contract.rs`.
  RED sobre el código anterior: stop micro 55 pb frente a 85,25 pb con
  10 000 USD.

### D3 — El veto de drawdown no pasa de la caída de falsación

Para un apostador a fracción c de Kelly con la ventaja estimada
(Thorp/Breiman, aproximación de difusión):

```
P(DD ≥ x) = (1 − x)^{2/c − 1}
d*(c, α)  = 1 − α^{c/(2 − c)}
```

- Con c = ½ y α = 0,05: d* = 1 − 0,05^{1/3} ≈ 0,632. Con 13 USD el piso es
  4,79 USD.
- Una caída mayor ocurre con probabilidad < 5 % si el modelo es cierto, así
  que la caída falsifica el modelo: se deja de abrir hasta recuperar o
  re-medir.
- Umbral del veto de risk-engine:

  ```
  umbral = min(lerp(dd_max_medido, d*, micro_w), d*)
  ```

  El literal 0,85 deja de existir.
- El «suelo de 3 USD» de los documentos AGY no existe en el código. Este
  ADR lo sustituye por d*.
- Implementado en QS-D3 (`risk_engine::drawdown::drawdown_de_falsacion`).
- Pendiente: el freno del host (`src/bin/god_engine.rs`, Línea C) usa
  `drawdown_maximo` sin esta cota. Hay dos semánticas; se pide alinear.

### D4 — La sonda es exploración presupuestada, no EV inventado

- Una sola sonda abierta en toda la cartera. Las sondas son información, no
  alfa; abrirlas a la vez multiplica un riesgo sin medir.
- Tamaño: la orden mínima del símbolo (D-750).
- Máximo 5 por moneda (vigente).
- Las pérdidas de sonda cuentan en el drawdown de D3: no hay una
  contabilidad aparte que las esconda.
- El prior 0,55 de la exención (R-18) se marca como literal sin origen.
- Implementación pendiente (zona núcleo/riesgo, coordinar con la Línea C y
  el PR #29). Hay que medir antes si el dimensionado de la sonda es de
  verdad el mínimo, y sustituir el 0,55 por la tasa base medida del modelo
  para esa geometría (cambia conducta: requiere T-1).

## Consecuencias

- D2 y D3 cambian conducta en régimen micro. Se certifican con el oráculo
  T-1 en el mismo PR (trinquete 11,0 %).
- Con D2, en micro algunas órdenes salen con un stop más ancho. El riesgo en
  dólares sigue acotado por R-19 y el tope de ruina. Otras se rechazan por
  el suelo en vez de entrar con una geometría que no cobra.
- Con D3, el sistema se detiene antes (piso 4,79 USD en vez de 1,95 USD).
  Detenerse no es un fallo: es la señal de que el edge supuesto no existe.
- Nada de esto autoriza a operar ni demuestra rentabilidad. La meta queda
  expresada como requisito (edge × frecuencia) que se mide fuera de muestra.

## Alternativas descartadas

- Mantener el 55 pb como presupuesto de ruina en dólares: con la orden
  mínima de 5,10 USD, un stop de 150 pb arriesga 0,077 USD (0,59 % de 13
  USD), muy por debajo del tope de ruina. El presupuesto no exige recortar
  el stop.
- Suelo absoluto de 3 USD: no deriva de ninguna cuenta, y con 13 USD
  equivale a tolerar una caída del 77 %, más laxa que d*.
- Declarar la meta inalcanzable y fijar otra: se prefiere dejarla como
  requisito medible (D1). Si el edge medido no llega, el sistema no crece y
  no se engaña a nadie con el tamaño.
