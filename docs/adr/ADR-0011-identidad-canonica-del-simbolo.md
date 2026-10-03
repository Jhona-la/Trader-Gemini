# ADR-0011 — Un solo nombre por símbolo y slots estables en caliente

- **Estado**: Aceptado
- **Fecha**: 2026-10-03
- **Responsables**: Claude (ciclo 8, CL-37 y CL-38); revisión cruzada en el
  PR del ciclo 8

## Contexto

El bootloader entregaba el universo en minúsculas (`atomusdt`), el registro
guarda los specs en MAYÚSCULAS (B3.38) y los bosques se cargan por el stem
del archivo (`ATOMUSDT_MOTOR`). La clave `"atomusdt_MOTOR"` nunca
coincidía: `has_roster_model` quedaba falso en todos los slots y la base del
modelo caía al 0,5 neutro. Además, el demonio de rotación reescribía el
universo vivo en su primer tick (y luego cada ciclo) mientras el host
mantiene congelado `symbol_to_id`: el slot `i` del núcleo dejaba de ser el
símbolo que el host le enruta.

## Decisión

1. La grafía canónica de un símbolo es MAYÚSCULAS ASCII
   (`symbols::simbolo_canonico`). El universo se publica canónico, el
   bootloader lo entrega canónico y la histéresis de rotación compara en
   esa grafía. Los streams WS siguen construyéndose en minúsculas aparte.
2. El mapa slot ↔ símbolo es inmutable durante la vida del proceso. El
   demonio de rotación persiste su propuesta y actualiza specs, pero no
   reescribe el universo vivo ni re-suscribe el WS; la propuesta se aplica
   al reiniciar.

## Consecuencias

- `try_symbol`, `try_spec` y `try_index` son el mismo mapa (contrato
  `identidad_simbolo_contract`); el núcleo encuentra el bosque del roster
  con el universo del bootloader (`identidad_simbolo_nucleo_contract`).
- Rotar el universo exige reiniciar. Una rotación en caliente correcta
  necesitaría re-indexar el estado por moneda (arena, posiciones, bosques,
  espectros) de forma atómica; queda fuera de este ADR.
