# QRA F0/F1 — recursos y rescate de cobertura

Cierre del código: 2026-10-23T09:56:51-05:00. [PR105](https://github.com/Jhona-la/Trader-Gemini/pull/105) está integrado en `68839f9683afdec64e4c992fdc6ce7e9d7c5441c`. El candidato y el push real de main aprobaron **10/10 por separado**. Este recibo documental no modifica sus fuentes; las corridas se atribuyen al código M, no al commit posterior de documentos.

## Identidad y evidencia

| Revisión | Identidad | CI propio |
|---|---|---|
| Base | `75e17d4635c3c12eaf377119fb2518df18b10ba1` | Historia, no transferida |
| Head candidato | `ebe5d1b6b2e78a2b0c7d3a19074e0c797d970147` | [Unit39964126655](https://github.com/Jhona-la/Trader-Gemini/actions/runs/39964126655) 8/8 + [Replay39964126563](https://github.com/Jhona-la/Trader-Gemini/actions/runs/39964126563) 2/2 |
| Merge virtual | `7b8f04140d1049f879085f785128588419f75522`, padres B/H | Árbol probado |
| Push real de main | `68839f9683afdec64e4c992fdc6ce7e9d7c5441c`, padres B/H | [Unit39968864721](https://github.com/Jhona-la/Trader-Gemini/actions/runs/39968864721) 8/8 + [Replay39968864605](https://github.com/Jhona-la/Trader-Gemini/actions/runs/39968864605) 2/2 |
| Árbol común H/V/M | `dded9f6ad0db349e956a34d8c28030267fd4897b` | Cotejo Git local y API |

El [expediente JSON](QRA_F0_F1_CI_2026-10-23.json) conserva IDs, checkout, nombres, pasos, fechas y hashes de los logs por revisión. SHA256 de sus bytes LF: `d266725534df862940295aa777bc30704944cbfc5a369114ba1a3a8dfb1491b3`. Los raw permanecen en logs ignorados y se pueden recuperar mediante los endpoints GitHub consignados; no se promete retención indefinida de GitHub. El sello Omega220 anterior conserva SHA256 `1a9f96050cdb1256280e4f1171f86279ace5fa519ec43fe6f2b760249c21dac3` y no se reescribió.

## Cambio concreto

Tres lanzadores podían iniciar Cargo sin admisión compartida de recursos; el medidor podía generar targets por worktree. Ahora usan `cargo_semaforo.sh`, target canónico y dos jobs en sus invocaciones Cargo. Rechazan las tres variables de rustflags y, en Bash, overrides CLI de target/config/jobs. Propagan errores 75/101.

Quality drena salida sin acumularla completa, ejecuta Clippy workspace una sola vez y conserva errores incluso en modo informativo. El medidor, al fallar, retira sólo sus altas: valida ruta física, common-dir, HEAD, ausencia de stage y hash raw de la copia propia. Una modificación ajena posterior conserva el árbol. No utiliza retirada forzada.

[Ledger de cinco rutas](QRA_F0_F1_RUTAS_2026-10-23.tsv), con anchor H actualizado para las cinco, incluido el workflow que recibió después el paso metadata:

| Ruta en alcance | Blob Git |
|---|---|
| `scripts/quality_gate.ps1` | `c278c5c3908817ef52703c635f530f97f22bf4bd` |
| `scripts/medir_camino_caliente.sh` | `4e7e9768f9fab27acc0384720e0e3b1695a3d698` |
| `scripts/oracle_witness.sh` | `16fa5cf31c1938c1b75f586e85fe6143898a1a3c` |
| `scripts/tests/test_qra_resource_safety.py` | `61a34540fbf3251f202ba4a70b44a422e27bafcf` |
| `.github/workflows/unit-suites.yml` | `08cbc267e2d08f6c05b6efad620bdf2552224b41` |

Los 854 archivos Rust, manifiestos/lock Cargo y `.cargo/config*` son iguales a B. La revisión cubre este cambio en cinco rutas; el [inventario histórico de2193](QRA_CONTINUACION_INVENTARIO_2026-10-23.tsv) sigue siendo una cola pendiente sobre75be, no una auditoría total.

## Rescate y pruebas

La historia de AGY PR103/f174 se conserva por el merge49f9. Se rescata sólo la cobertura completa de `telemetry-server`; los seis paquetes Sol conservan tests y doctests sin duplicarlos. GitHub cerró PR103 por ancestría; eso no acredita que sus propuestas de MXCSR/parsers estén operativas. DAZ global se excluyó por el [contrato oficial de Rust](https://doc.rust-lang.org/core/arch/x86_64/fn._mm_setcsr.html); los parsers nuevos sin consumidores permanecen sólo en historia. Os-guardian y parsers finales son iguales a B.

| Comprobación | Candidato | Push real main |
|---|---:|---:|
| Recursos: 32 F0 +23 barrido +13 semáforo +10 fail-closed +6 estados | 84/84 | 84/84 |
| Telemetry-server, suite completa | 61/61 | 61/61 |
| Seis paquetes Sol, suite completa | 50/50 | 50/50 |
| Storage integrity | 65/65 | 65/65 |
| Host OMS | 16/16 | 16/16 |
| Workspace `--all-targets` | PASS | PASS |
| Golden contable | 1 PASS,56 filtradas,2 testigos | 1 PASS,56 filtradas,2 testigos |
| Inventario `cargo tree --workspace --duplicates --locked` | 68 entradas raíz | 68 entradas raíz |

Los bloques de recursos, server, Sol, storage y Host citados no tienen fallos ni omisiones/filtros. Los doctests de server y los seis paquetes Sol se ejecutaron, con cero casos. Son 50+61 casos de cobertura de esos paquetes, no una suma global de todos los tests del proyecto. Las 68 entradas raíz representan 65 pares nombre-versión distintos, con repeticiones por contexto; es metadata, sin poda de dependencias ni cambios Cargo.

Golden conserva una sonda, una entrada y un cierre por corrida; los dos testigos coinciden, con caja final `1000.3114819328429`, bits `4652010048665245399` (`0x408f427dea3d56d7`). La réplica Python independiente del candidato reprodujo nocional/fees/cash con el mismo orden IEEE. No es prueba de formación de precios ni de liquidación real del exchange.

Validación local previa: 28 métodos completos PASS272.35s; luego seis focused PASS50.553s (tres repetidos y tres nuevos de cleanup) y un método con12 negativos de flags PASS12.416s. Son ejecuciones solapadas; la suite final tiene32 métodos. Se usaron PowerShell7.6.7 y WindowsPowerShell5.1.26100.9549 reales con sentinels Cargo. CI Ubuntu acredita al menos un intérprete PowerShell, sin trasladarle la prueba de PS5.1 local. **No hubo Cargo real local**.

## Recursos y coordinación

Censo completo de metadata del primary,2026-10-23T13:53:51.263440Z,28.99s:231309 archivos, cero pendientes/errores/reparse encontrados. Root76,312,140,672 bytes lógicos (76.31GB), target44,361,323,654, data25,689,693,842 y `.git`3,555,274,284. Volumen libre362,025,299,968. Se observó menos de300GB en esa carpeta, sin imponer un techo futuro ni medir asignación física o ahorro. El recorrido streaming sin seguir reparse no es un snapshot atómico ni suma todos los worktrees externos. No hubo borrado para obtener estas cifras.

RAM libre observada siempre por debajo de4GiB durante las decisiones de build; último corte root14:30:59Z2.34/23.38GiB visibles. Se mantuvo el umbral de admisión y no se cerraron procesos ajenos. Estos datos no garantizan por sí solos un máximo de16GB para toda la actividad del ecosistema ni dual-channel completo.

Se preservaron primary, Verify locked ACTIVE y worktrees activos Sol1097/GLM155. La transición del primary a main y retirada de la rama AGY anterior fueron observaciones de acciones ajenas, no limpieza de QRA; AGY abrió después [PR106](https://github.com/Jhona-la/Trader-Gemini/pull/106), independiente de esta evidencia. QRA comunicó sus hallazgos en [compilación/capital](https://github.com/Jhona-la/Trader-Gemini/pull/106#issuecomment-6367417134), [cache/UDS](https://github.com/Jhona-la/Trader-Gemini/pull/106#issuecomment-6367454945) y [contratos FFI](https://github.com/Jhona-la/Trader-Gemini/pull/106#issuecomment-6367483813). No se heredan verdes ni se toma posesión de su rama.

## Límites y siguiente átomo

`oracle_witness.sh` observa hashes de un glob y no certifica qué binario ejecuta un Cargo posterior. La admisión de Cargo tampoco cubre benchmarks/binarios lanzados directamente. No se midió latencia, rendimiento financiero, ahorro RAM, API autenticada, Demo, fondos ni promoción de entorno.

F2 sigue pendiente: [diseño de capital observado](QRA_F2_CAPITAL_DISENO_2026-10-23.md) y [revisión de endpoints UDS](QRA_F2_UDS_ENDPOINT_2026-10-23.md) son propuestas sin implementar. El cache delta no demuestra cuenta flat, wallet no equivale a equity y la identidad de sesión debe acompañar la observación. Los fallbacks FFI13 de M tampoco se corrigen en esta ola. La meta100%/72h continúa sin evidencia OOS neta.

El [plan por fases](QRA_CONTINUACION_PLAN_2026-10-23.md) conserva su cronología y pendientes. Reserva de código QRA liberada. Rama/worktree propios se retiran después del merge de este recibo, con HEAD integrado y árbol limpio; la constancia final se publica en PR105. No se retira trabajo ajeno activo.
