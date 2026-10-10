# ADR-0015 — El kill-switch bloquea lo que aumenta el riesgo, nunca las salidas

- **Estado**: Aceptado
- **Fecha**: 2026-10-10
- **Responsables**: Claude (ciclo 9, CL-43 y CL-44); revisión cruzada en el
  PR del ciclo 9
- **Reservado en**: `docs/HOJA_DE_RUTA_CIMIENTOS_2026-10-01.md` (ADR-0015)

## Contexto

El sistema tiene dos latches de parada total:

- `GlobalArena::kill_switch_active`, que arman el núcleo (capital ≤ 0), el
  sistema inmune del host (drawdown) y la escalada X-009 (tres OCO fallidos
  y dos cierres de emergencia fallidos).
- `OrderExecutor::kill_switch`, que el host arma junto al anterior.

Ya estaba decidido que el executor separa las dos clases de órdenes:
`check_rate_limits` (entradas, apalancamiento) respeta el latch, y
`check_exit_rate_limits` (lecturas, cancelaciones, reduce-only y piernas
protectoras) no (CL-3, CL-20). El resto del sistema no seguía esa regla:

1. **Núcleo** (FMT-232). `process_event` y `process_tick_dual` devolvían
   `None` antes de la gestión de ranuras. Con el latch armado ninguna
   posición abierta recibía SL local, trailing, break-even ni salida por
   toxicidad: el sistema se apagaba justo cuando las salidas defensivas más
   importan.
2. **Vigilante de protección del host**. Con el latch armado limpiaba la
   marca de suciedad y saltaba la auditoría: una posición desnuda no se
   re-protegía ni escalaba a cierre, aunque las piernas protectoras están
   exentas del latch en el executor.
3. **X-009 y sistema inmune**. Armaban el latch tras un cierre o un
   aplanado fallido y dejaban la posición viva sin despertar al vigilante.

## Decisión

1. El latch bloquea sólo lo que AUMENTA el riesgo: entradas nuevas,
   cotización maker y cambios de apalancamiento.
2. Las salidas siempre corren: gestión de ranuras del núcleo (el latch entra
   en `entries_blocked` y el cierre viaja por la frontera X-012), cierres
   reduce-only, cancelaciones, lecturas y piernas protectoras.
3. Toda vía que arma el latch y deja una posición viva despierta al
   vigilante (`protection_health::mark_dirty`), que la re-protege o escala
   su cierre. El vigilante audita también con el latch armado.

## Consecuencias

- Contratos de comportamiento: `close_outcome_contract.rs`
  (`kill_switch_preserves_local_stop_close_and_blocks_new_entry`,
  `kill_switch_process_event_preserves_stop_close`). Guardias sobre la
  fuente del host: `kill_switch_host_contract.rs`. El registro de vetos
  (V-LOGIC-005) apunta al contrato del núcleo.
- `train_forest` sigue abortando si el latch se arma durante la
  reproducción del tape: allí sólo se arma con capital ≤ 0, y un tape que
  liquida el capital no produce etiquetas válidas.
- Carrera conocida y aceptada: si el vigilante re-protege mientras el
  aplanado del sistema inmune cierra la posición, puede quedar una pierna
  protectora huérfana. Es reductora por construcción y la siguiente
  auditoría la purga.
- Abierto: el latch es permanente hasta reiniciar el proceso. El rearme
  con histéresis (V-LOGIC-005) no tiene prueba de runtime en el host.
