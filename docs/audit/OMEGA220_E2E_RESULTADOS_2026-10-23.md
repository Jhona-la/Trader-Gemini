# Omega220 E2E: recibo de integración

Estado: CÓDIGO INTEGRADO Y VERIFICADO. PR102 está en main e2867598a97699b3148657578187685ab087bfad; candidato y push real aprobaron sus nueve jobs por separado. Los dos testigos de cada revisión pasaron el cálculo independiente en Python. Este recibo sólo añade documentación y el verificador; la retirada propia se ejecuta después de su merge.

## Cambio y alcance

[PR102](https://github.com/Jhona-la/Trader-Gemini/pull/102) corrige comisiones sobre cantidad y precio físicos contabilizados, tarifas por símbolo y sesión, rebates, recibos locales consumibles y estimaciones de adopción identificadas. El OMS inicia cerrado en Testnet, exige modo hedge y reconciliación completa de tarifas, posiciones y órdenes; descarta observaciones de otro ejecutor o universo, drena tareas admitidas y conserva reservas pendientes. Publicación local y snapshots comparten un cerrojo; la tarea de entrada conserva su ejecutor hasta protección y rollback. El balance de transición valida su sesión antes de publicar.

Se preservaron AGY PR99/5605b6a66 y Sol F1094/eeae8e106, incluido su cierre documental d64a00536. El resolver D-725 conserva los escenarios registry-only sin mutar el universo global; exigir observación sin libro sigue bloqueando entradas. El ledger [de 23 rutas](OMEGA220_E2E_RUTAS_2026-10-22.tsv) vincula blobs y alcance, sin acreditar auditoría integral de esos archivos ni del repositorio.

## Evidencia separada por revisión

| Evidencia | Candidato PR102 | Push real de main |
|---|---|---|
| Head | 7715f9d6096a6fa080e5e8b4a4d8506769fa927a | e2867598a97699b3148657578187685ab087bfad |
| Base / primer padre | d64a005362d670073354638414dca5ca7dc3e4df | d64a005362d670073354638414dca5ca7dc3e4df |
| Merge / segundo padre | 42492880f30b1a014d451d4735374d6d061bdd3f | 7715f9d6096a6fa080e5e8b4a4d8506769fa927a |
| Árbol | 41cd5cb0781868fc9fba2918b247788316405734 | 41cd5cb0781868fc9fba2918b247788316405734 |
| Unit: siete jobs | [39904967784: PASS](https://github.com/Jhona-la/Trader-Gemini/actions/runs/39904967784) | [39907724231: PASS](https://github.com/Jhona-la/Trader-Gemini/actions/runs/39907724231) |
| Replay: dos jobs | [39904967770: PASS](https://github.com/Jhona-la/Trader-Gemini/actions/runs/39904967770) | [39907724237: PASS](https://github.com/Jhona-la/Trader-Gemini/actions/runs/39907724237) |
| Total jobs | 9/9 PASS | 9/9 PASS |
| Objetivos Rust únicos | 45/45 PASS | 45/45 PASS |
| Recursos / entrantes Sol | 68/68 y 7/7 PASS | 68/68 y 7/7 PASS |
| Workspace --all-targets | PASS | PASS |
| Testigos GOLDEN / Python independiente | 2/2 PASS | 2/2 PASS |

Los 45 objetivos son 42 nuevos y 3 existentes corregidos: FeeSchedule 8, core 6, sync 2, reconciliación/registro 11, flujo privado 1, host 16 y golden 1. El golden se ejecuta en la suite completa y otra vez con --exact --nocapture; sigue siendo un objetivo único. Se conservaron las suites completas y doctests existentes. Los casos objetivo no tienen fallos, ignorados ni filtros, salvo el selector exacto intencionado del testigo, que ejecuta su único caso; los experimentos manuales ignorados del repositorio no forman parte de esta certificación.

El expediente [CI por SHA](OMEGA220_E2E_CI_2026-10-23.json) conserva jobs, nombres, tiempos y logs de candidato y push real por separado. No se transfirieron resultados de otras revisiones. 0274 terminó8/9 por dos controles positivos sin entradas; se corrigió la identidad registry-only. 25dcb terminó8/9 sólo por el bit de saldo del golden. a416 fue sustituido antes de completar CI para fijar las cotizaciones del instante del cierre en su testigo, sin fallo observado.

## Recertificación contable del golden

Se mantienen una entrada, un cierre, ninguna posición terminal, margen cero y dos corridas con las ocho estadísticas bitidénticas. El oráculo privado cfg(test) contrasta snapshot y recibo reales; el rol de salida deriva del TP y las cotizaciones del cierre, sin inferirlo del fee cobrado. Rechaza una geometría o un cruce de spread ambiguos para este fixture. No calcula precios físicos ni certifica ejecución remota.

El [verificador independiente en Python](OMEGA220_E2E_VERIFY_GOLDEN.py) usa sólo la biblioteca estándar, binary64 y el orden de débitos real. Recibe el log del job con los dos testigos y el bit esperado 4652010048665245399:

```text
gh run view 39907724237 --job 120263401471 --log > logs/omega220-main-witness.log
python docs/audit/OMEGA220_E2E_VERIFY_GOLDEN.py logs/omega220-main-witness.log 4652010048665245399
```

El cálculo independiente es:

```text
Fe = (q * Pe) * rE
G  = (Pe - Px) * q             # corto de este fixture
Fx = (q * Px) * rX
C1 = C0 - Fe
C2 = C1 + (G - Fx)
net = (G - Fx) - Fe
```

En ambos testigos del candidato: q=0.0010008002200500106, Pe=59958.881905018454, Px=59587.87564818821, rE=rX=0.0005, Fe=0.030003431102247534, Fx=0.029817779530509715, G=0.3713031434756405 y net=0.3114819328428832. Con C0=1000, C2=1000.3114819328429, bits 4652010048665245399. Usar el mid en el fee de salida recupera exactamente el bit anterior 4652010048691470773 en ambos testigos; así se explica el cambio sin alterar la sonda ni seleccionar un saldo por rentabilidad. El resumen fees_est del runner no se usa como oráculo.

La misma comprobación sobre los dos testigos del push real pasó 2/2 usando esta copia del verificador. Los cuatro registros (dos del candidato y dos de main) coinciden en cantidad, precios, fees, bruto, neto, cash y bits. El expediente conserva los testigos completos y hashes del log, resultado y script.

## Recursos, límites y retirada

No hubo Cargo local: RAM libre medida 2.56GiB al inicio,2.69GiB a22:35 y3.33GiB a23:39, inferior al mínimo 4GiB del semáforo. Sin target alternativo, cambios de perfiles ni cierre de procesos ajenos. Formato/parse estático de módulos nuevos y diff-check no sustituyen compilación ni tests. El binario conserva un límite histórico del formateador por espacios; no se acredita un formato integral del host. Los 9 jobs se ejecutan en runners alojados.

Son pruebas sintéticas de software y estimaciones locales. Sin motor operativo, fondos, entrenamiento real, publicación de modelos ni promoción de entorno. Las tarifas actuales no prueban la comisión histórica de una posición adoptada; ADOPTED_ESTIMATE/order0/trade0 no son fills independientes. Los fills nativos sin ID mantienen heurísticas de solapamiento; moneda de comisión y FX requieren contratos propios. El gate de riesgo puede rechazar fricción neta negativa, aunque el ledger conserva rebates. Escenarios de evolvers/replays no adquieren automáticamente evidencia de una cuenta viva. WS sigue siendo stub; el formateador mantiene el límite previo de precisión no soportada.

El libro valida identidad de tarifa; no repara IDs persistentes ni migra posiciones al rotar el universo. El caso manual de símbolo renombrado prueba selección y fallback de fee, sin acreditar migración operativa. No se certifican liquidaciones del exchange, ROI/WR, latencia, ahorro de RAM ni la meta +100%/72h.

La retirada propia queda PENDIENTE tras el merge de este recibo: exigir HEAD integrado, cero cambios/untracked e ignorados revisados; desbloquear y retirar sólo el worktree Omega220 propio. La constancia de ejecución se añade al PR del recibo. Sin intervención propia sobre worktrees ajenos. Verify PR65 locked ACTIVE y GLM154 seguían registrados al corte; no se atribuye a esta sesión ninguna retirada de Sol.
