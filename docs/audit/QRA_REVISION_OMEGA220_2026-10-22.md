# QRA — revisión del candidato Omega220 antes de integración

> Actualización posterior: el anexo de `0bfd39c75` confirma corregidos en
> fuente el aumento 0.6→1 y el uso de arena después de moverla. No tratarlos
> como warnings vigentes de ese segundo corte. CI de la rama AGY no se
> certifica por esta revisión; quedan límites y un nuevo desajuste de formato.

22-Oct-2026. Revisión de root y omega220_review, sólo lectura. Corte exacto:
`4614847102e929aea494fbd959988ee368fd567b` de `agy/omega220-fees-y-filtros`,
comparado con main `cf87f8cbc6d0fe3330e460f247dd2b38e9d990f2`.
Rama activa: buscar de nuevo anclas y clasificar revisiones posteriores antes
de aplicar este informe. Este corte NO está integrado en el candidato QRA
`911a4c863`; su CI no certifica Omega220. No se cambió esa rama, no se ejecutó
Cargo, HTTP privado, entrenamiento ni operaciones.

## Evidencia por blob

| Ruta | Blob del corte |
|---|---|
| `crates/god-engine-core/src/lib.rs` | `f8b8a5b1fe2379a03301ac2910c2370a3281e5d1` |
| `crates/god-engine-core/src/reality_physics.rs` | `b5a0b203efdc9b29ada211a62ed5e75815694d36` |
| `crates/execution-engine/src/executor.rs` | `8eda549dde4d184ba90b1795ce1a6511feec6feb` |
| `crates/execution-engine/src/fixed_filter.rs` | `1daad1bca0e2b574dd7d281de1762f7db880878c` |
| `crates/execution-engine/src/client.rs` | `fe6903ad4b430398e93c2aecafae77b5899b2a69` |
| `crates/execution-engine/tests/payload_admission_contract.rs` | `3f2a8005dc4fb95e2096692a033f352a461ab4c8` |
| `crates/execution-engine/src/ws_executor.rs` | `c4dd2247cf99f27658a431a1c7e8274803daf9a5` |
| `crates/backtest-engine/src/lib.rs` | `bf34dfea5e64f9071f4b903a7767b1b0b8bf1392` |

## O220-01 — cantidad firmada distinta de la aprobada

El nuevo `push_fixed_f64` usa `format_decimal`, un formateador de presentación
que aplica `round()` (fixed_filter.rs:591–603), con fallback al f64 si devuelve
vacío. No es `validate_and_quantize` ni serializa sus unidades u128.
`count_decimals` devuelve 0 también cuando no soporta el paso (:473–483),
y los nuevos callers lo interpretan como cero decimales válidos.

Contraejemplo dentro del dominio admitido por `build_payload`: cantidad bruta
0.6, paso 1e-11. El redondeo por paso conserva `final_quantity=0.6`, pero la
precisión deducida 0 lleva `round(0.6)=1` y el query firmado contiene
`quantity=1`; el `ExecutionPayload.quantity` sigue 0.6 (executor.rs:1582–1612).
La cantidad transmitida sería 66.7 % mayor que la aprobada en ese escenario.
Reproducción por emulación numérica IEEE en Python, NO ejecución del contrato
Rust ni prueba con fondos. No se afirma que un símbolo vigente de Binance
publique ese paso: la API interna lo acepta y debe rechazar escalas no soportadas.

Corrección propuesta: precisión como Option/Result, rechazo explícito y un
único resultado de cuantización validada del que salgan cantidad reservada,
reportada y ASCII firmado. Contrato con `build_payload` real: extraer el
parámetro quantity del mismo query firmado; compararlo con la cantidad
aprobada y el presupuesto, incluidos escala no soportada, límites y paso no
potencia de diez. Un test del formateador aislado no prueba esa invariante.

El contrato previo payload_admission_contract.rs:27 espera `quantity=0.4&`;
la nueva salida de tres decimales es `quantity=0.400&`. El valor es equivalente,
pero ese assert requiere actualización legítima. Mantener comparación exacta
del valor y del presupuesto al adaptar representación. La prueba nueva de
0.007 no cubre el contraejemplo ni la escala rechazada. No hay PASS Rust
atribuible a esta revisión estática.

## O220-02 — arena movida antes de leer las tasas

`GodEngineCore::construir` mueve `arena` en el literal `Self` (lib.rs:1482)
y vuelve a usarla para `RealityPhysics::with_fees` (:1541–1542). Es un bloqueo
estático de ownership; falta confirmar el diagnóstico con el compilador del
candidato corregido. Construir la física/capturar las tasas antes de `Self`
evita usar el valor movido sin clonar Arc innecesariamente. No se editó el host.

## O220-03 — paridad aún necesita contrato del core real

El wiring por tick es una mejora concreta respecto a la física predeterminada
separada. El test nuevo instancia `RealityPhysics` en modo Optimistic y llama
sus métodos directamente: no prueba construir el core, leer arena entre ticks,
ni el débito/PnL de su ruta HyperRealistic.

`with_fees` y `sync_with_arena_fees` aceptan sólo tasas finitas no negativas
(reality_physics.rs:46–70). En negativas/ilegibles, el constructor fabrica VIP0
y el refresco conserva la anterior. El parser de PR97 preserva tasas finitas
con signo, por lo que hace falta una política explícita común de estado observado
y tratamiento económico. Esto no afirma que Binance conceda toda tasa negativa;
señala la divergencia dentro del dominio admitido en la frontera de datos.

Probar con el core real y dos observaciones rA/rB entre ticks, ingreso/cierre y
ledger: para igual nocional, la diferencia de comisión debe ser N*(rB-rA).
Aislar la nueva entrada con `allow_entries=false` al comprobar el cierre y
registrar nocional efectivo si cambia el sizing. Las identidades de símbolo,
cuenta/entorno y generación, la toma de sólo symbols.first() y los FFI con
fallback13 siguen abiertos; sanitizar INITIAL_CAPITAL no los elimina.

## Límites y relevo

WebSocket sigue siendo stub que devuelve error (ws_executor.rs:51–79).
Sus parámetros f64 son deuda futura de transporte; no se reporta una
operación WS actual que discrepe de REST. No se valida paridad de fills,
modo One-Way/Hedge, fees en otra divisa ni rentabilidad OOS por este informe.

AGY conserva la titularidad de Omega220 y su rama. Hallazgos comunicados en
COORDINACION; mantener su integración separada del cierre de PR97 y exigir
contratos y CI de su propio candidato. Los ocho blobs anteriores son trazas
acotadas, no ocho archivos completamente auditados ni nuevas filas del censo.


## Anexo — reevaluación posterior en 0bfd39c75

Corte exacto `0bfd39c75e4fb660b0afd59dc8987444c9fad320`, rama AGY activa,
posterior a los ocho blobs históricos. Sólo fetch/lectura, sin Cargo ni
edición de su rama. No está integrado en el código QRA probado de main.

| Ruta | Blob del segundo corte |
|---|---|
| `crates/god-engine-core/src/lib.rs` | `50c28a57590768355454c743979768944720e169` |
| `crates/god-engine-core/src/reality_physics.rs` | `edbcecc704985922fa4a8f4c56b4b137cf68e242` |
| `crates/god-engine-core/tests/qra_fee_sync_contract.rs` | `23ba541c453acc210385acf85171e0830aa48b15` |
| `crates/execution-engine/src/executor.rs` | `31ac0f25031bd7194b17331cf5da59f31cdcc73c` |
| `crates/execution-engine/src/fixed_filter.rs` | `ad872225b45d392ff92e0ca1e91a6770edef55c1` |
| `crates/execution-engine/tests/payload_admission_contract.rs` | `d9e6b2c2707a681a699193f2ba1884e23480ee24` |

- **O220-01 corregido en fuente:** try_count_decimals devuelve None para la
  escala no soportada, y build_payload conserva final_quantity.to_string().
  El caso0.6/paso1e-11 ya debe firmar0.6; se añadió contrato (:209–218).
  Esto evita el aumento grueso, pero conserva un float/fallback y no equivale
  a rechazar escalas ni validar el grid completo con unidades u128.
- **O220-02 corregido en fuente:** tasas capturadas antes del literal Self
  (lib.rs:1378–1379) y usadas después (:1547–1550). No mantener el diagnóstico
  de ownership antiguo como vigente en este segundo corte.
- **0.4/0.400 corregido por cambio de representación:** se recortan ceros al
  final de format_fixed_ascii (:659–666), con contrato adicional. Surge otro
  assert antiguo: qra_fixed_filter_contract.rs:67 espera buy.price_str="10.0"
  y ahora recibe"10". Fallo esperado por inspección, no ejecución Rust ni
  diferencia monetaria. Adaptar la representación sin debilitar valor/grid/presupuesto.
- **Cobertura de fees mejorada:** nuevo qra_fee_sync_contract instancia el
  core real con su física HyperRealistic y queda incluido por Unit
  god-engine-core --lib --tests. Comprueba constructor, sync explícito y
  cálculos directos. Todavía no procesa tick, entrada/cierre ni débito/PnL
  del ledger. No declarar demostrado ese recorrido de extremo a extremo.
- **Política de signo aún distinta:** las tasas negativas siguen provocando
  fallback/retención en la física mientras el dato publicado puede ser finito
  con signo. El nuevo contrato fija la retención como comportamiento (:55–62);
  no demuestra igualdad con la observación. No se afirma que un símbolo real
  tenga una rebaja determinada. Requiere política/estado explícitos comunes.

WS continúa como stub. Los seis blobs de este anexo también son trazas
acotadas, no archivos íntegramente auditados ni filas adicionales del ledger
QRA de doce rutas. Validar el candidato AGY actualizado con su propio CI.
