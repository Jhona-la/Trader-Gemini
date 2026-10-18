# ROOT: reinicio y sincronización — 17 de octubre de 2026

Corte: 2026-10-17 16:09 America/Bogota. Main de comparación `0f12af7460d669dd7bf6f17011236ed607e8df0b` (Ω149). ROOT permanece integrador de B15; su código publicado sigue en PR66, head `fdc5a53204a6746e41311b7dfc036be2e12acf17`, sin integrar. Este lote contiene sólo documentación: heartbeat, recuperación, dependencias y fotografía del delta. No modifica código, fixtures, modelos, umbrales ni perfiles.

## Evidencia recuperada

- **BASE:** fuente congelada HEAD cff8886b + MERGE_HEAD7c24796b, índice `984891b3ec9948c43444abad9c3ef9ffdf6a9acd`, 1879 entradas. Sólo dos de siete gates tienen exit0: Risk428/0/10 y Dark58/0/0. Core acaba en compilación de telemetry-server, sin recibo terminal; cinco gates no están certificados. El dueño retomó, confirmó ausentes sus antiguos PID y prepara un archivo incompleto antes de ceder. No se presupone cesión por tiempo, cuota o ausencia de proceso. No se usa su caché.
- **Respaldo ROOT:** `lg5-v3-v7-lg5policy-2026-10-16-5939e144-e88e-4cf9-821d-04a0dffa9df6`: 920 payloads, 23.298.399 bytes, manifest `7dd6f9cd74adc36d43e552b46ff39ba446b7e718010c7d5725ebb3f446631349`. ROOT releyó todos los hashes/tamaños/inventario, cuatro pares físicos/Git y bundle verify0 con prerrequisitos332/b9. Recibo de lector separado `55d186896539eedfb86c6462a67601735fdb5dfc3aefee22d61c23e6c3d6afcf`; no revisión por tercero ni Rust. Notas posteriores no se agregan retroactivamente a ese sello.
- **SI A/B/C:** dictamen Quant `fa6db477aff4dd21779e2bba4895133ab2a40b3ef8c7986fc123c044322cff83`, reciboa419fc5f. A/B conservan cuatro contratos GLM y añaden cuatro; marca pública5 fortalecida, ledger3 y reporte20+2=22 declarados, todos0Rust en esa propuesta. A sólo advirtió blankEOF en el buzón; B/C aplican en overlays delimitados. El span y la marca se alinean con al menos un evento FE admitido por fila, sin acreditar atomicidad de todo Core o caja/PnL. Un rechazo temporal FE puede dejar continuar gestión de posiciones: deuda concreta, pendiente de contrato. No trasladar estas anclas a main149.
- **T1:** el histórico971 terminó con evidencia propia. El testigo GLM del15Oct registra dos pruebas aprobadas en769,61s y un mismo binario pre/post, pero declara HEAD445 y **23 archivos sucios**. El SHA del log copiado es `645fd89234b52ca330cb39aea5dd39d172a246b0b2e7fb0fc5f2c16031d8f5f2`. No existe en ese log un inventario completo de esos inputs ni conteo exacto de genes con nocapture: el resultado no certifica un árbol limpio445, main149 o el futuro agregado. No se ejecutó T1 nuevo en este reinicio.

## Reanclaje obligatorio antes de integrar código

Desde f7 a main149 cambiaron 139 rutas; el [delta por ruta](ROOT_MAIN149_DELTA_2026-10-17.json) es una fotografía de deriva, no un segundo censo de auditoría. Ninguna ruta adquiere estado «auditada completa» por aparecer aquí. Conservar el censo BASE y los expedientes Fable/Líneas G/H; enlazar revisiones por blob y alcance.

| Frente | Cambio relevante observado | Siguiente aceptación |
| --- | --- | --- |
| Hidden B15-16 | Módulo main mantiene el blob anterior; V3 y16+2 contratos siguen externos | Contratos sobre módulo previo, RED funcional auténtico, instalación V3 en nuevo freeze y postfix; no redondeo universal/calibración acreditados |
| Risk LG5/Kelly | Main149 conserva bonus1.25 y añade funding/freno/registro mutable; ROOT tiene gate neto y API con reloj explícito | Preservar ambos contratos económicos y callbacks; renovar fixtures de funding con payoff real. Definir unidades y carry futuro antes de traducir G a N. No copiar Risk manual completo |
| Flow/Hawkes | Main contiene1105 líneas, siete familias y un caller Core con presión hidden no nula | No instalar resolución19ad antigua ni caller presión0. Reanclar normalización/SS y contratos; el antiguo joint con x0 no describe la política actual de abstención |
| Replay | Main ya retira el sufijo futuro mediante8cb9c9b7; aún hay diferencia aduana/FE y primer timestamp | Conservar su mapa y dos contratos, SI CT/ledger y span/causa; componer hunks y renovar8+5+3 y reporte22. RUST_MIN_STACK efectivo debe fijarse/registrarse; no hay overflow observado |
| Forest | Cuatro tests directos no cubren atribución del daemon; Sol declara trabajo en daemon/Core | Coordinar reserva antes de tocar. Nunca usar lado del slot superviviente para etiquetar delta agregado; contrato productor con cierre individual y dedupe/source/epoch |

La traducción algebraica de una puerta `G > m(c+h)` a `N=G-c` sería `N > c(m-1)+m*h` **si** N excluye carry futuro. Es una condición de equivalencia de política, no una hipótesis económica aceptada ni un arreglo instalado. Estimando, signo, reloj/TTL y doble cobro deben comprobarse en las APIs reales.

## Secuencia y reparto

1. ROOT/BASE cierran el estado incompleto y la cesión con recibos; mantener builds ajenos activos. Ninguna llamada pesada nueva por este lote.
2. Congelar main y entregas aceptadas por SHA; reanclar contratos y resolver contra cada padre. PR64 ya debe comprobarse por ascendencia; PR67 y PR69 conservan commits exclusivos, PR69 sigue posterior. No integrar por parecido de contenido o verde de otro checkout.
3. Diagnóstico estándar separado, diff por cada padre y all-targets antes de commit de composición. Baselines causales con precondiciones, etiquetas y freezes distintos; preservar fallos/86. Fuentes cambiadas exigen gates renovados.
4. Postfix y suites explícitas, fullT1 con mínimo0.110/nocapture y CI del checkout real; verificar SHA remoto/ascendencia, después retirar sólo refs propias incorporadas y liberadas, con evidencia durable.
5. Auditoría archivo por archivo: metas y límites de evidencia → conceptos/unidades/relojes → teoría/supuestos → operadores/consumidores → código/contratos → medición fuera de muestra. Una media neta o nombre de teoría no demuestra duplicación72h, ventaja cuántica o ausencia de ruina.

ROOT respeta las reservas activas en AGENT_SYNC. El ciclo documental se publica por avance rápido desde main149; se comprueba igualdad de todos los blobs ajenos a estos documentos. No es el merge del código B15, no transfiere CI ni termina la auditoría total.
