# ADR-0013 — El host nunca envía más apalancamiento que el validado

- **Estado**: Aceptado
- **Fecha**: 2026-10-03
- **Responsables**: Claude (ciclo 8, CL-41); revisión cruzada en el PR del
  ciclo 8

## Contexto

El núcleo abre la ranura con la orden que validó el risk-engine (margen
`volume_usd`, nocional `margen · leverage`) y suma ese margen a
`arena.used_margin`. Después, el host recalcula el apalancamiento de envío
con la envolvente Kelly y lo adapta al margen libre (D-382), y el replay
copia esa cadena. Dos defectos:

1. El margen libre era `capital − used_margin`, que ya descuenta la reserva
   de esa misma orden. En capital micro se vetaban órdenes que caben y se
   subía el apalancamiento sin necesidad.
2. La adaptación subía hasta 20× sin mirar el apalancamiento validado, y la
   envolvente podía pedir más que él: el host decidía por encima del riesgo
   y la liquidación quedaba más cerca de lo que el riesgo aceptó. El host
   además leía el margen de la primera ranura abierta, no de la reservada.

## Decisión

1. Una sola función pura, `risk_engine::envio::apalancamiento_de_envio`,
   decide el envío en el host y en el replay: parte del apalancamiento de
   la envolvente recortado al validado; si el margen requerido supera el
   85 % del libre, lo sube como mucho hasta el validado; si aun así pasa
   del 95 %, veta.
2. El apalancamiento validado se reconstruye de la ranura reservada
   (nocional / margen) y el margen libre no descuenta la reserva propia
   (`margen_libre_sin_la_propia`).
3. Se conservan el arranque exploratorio (1× con menos de 30 cierres) y el
   veto de la envolvente: cambiarlos es política de riesgo del dueño.

## Consecuencias

- Las decisiones de riesgo duras (apalancamiento y liquidación) quedan del
  lado del risk-engine; el host sólo puede bajar el apalancamiento, nunca
  subirlo por encima de lo validado.
- Queda abierto, y documentado: la envolvente sólo decide el apalancamiento
  del exchange (margen bloqueado), no el nocional, que fija el núcleo; el
  dimensionamiento en espacio de riesgo (riesgo al stop ≈ Kelly fraccional)
  sigue sin existir. La envolvente también evalúa su capital descontando
  la reserva propia (`cap_now`); no se toca aquí porque cambia sus vetos.
