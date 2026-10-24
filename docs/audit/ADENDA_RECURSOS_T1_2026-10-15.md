# Continuidad de BASE15: recursos, lanzador T-1 y revisión de fundamentos

Fecha: 2026-10-15, America/Bogota. Responsable: Codex, chat `01a14167-dabc-75d2-a2d8-3de3ef48b48b`, con tres revisores delegados de recursos, Git y fundamentos. Corte de código: `ba445df79fa5f82402cd675af4b150f8a0398dea` (Ω132). Esta es una adenda al plan maestro del 15 de octubre y a BASE15, no un nuevo plan canónico ni una declaración de auditoría completa.

## Continuidad del recorrido archivo por archivo

BASE conserva la reserva de `docs/audit/PLAN_BASE15_2026-10-15.md`, su inventario y su cola, con integración seguida en [PR65](https://github.com/Jhona-la/Trader-Gemini/pull/65). Se consultó el plan en la rama local `codex/base-audit-2026-10-15`, commit `cff8886b5c50ff99e377b30082c2c3c814c91cf1`, blob `855889b2f99554452f1bc22d0047d375665b5d65`; no se presupone su presencia en main. La revisión cubre desde metas y supuestos hasta datos, fundamentos, representación espectral, integración, riesgo/ejecución, adaptación, plataforma y tests. Su snapshot original `b9cd773538fae9467751883eafdf3ee6ed5142f9` contiene 1.863 rutas. No se duplican sus filas ni sus estados de revisión.

El [delta de esta adenda](DELTA_RECURSOS_BASE15_2026-10-15.json), calculado con `scripts/audit_inventory.py`, enlaza ese snapshot con Ω132: **9 rutas añadidas, 0 eliminadas, 21 modificadas y 1.842 intactas**; total del destino, 1.872 rutas. Los 30 cambios de ruta deben entrar en la cola del dueño. Una ruta intacta conserva identidad del blob, no una certificación semántica. Los documentos y contratos creados después de Ω132 pertenecen al delta siguiente. Los archivos sin seguimiento se preservan y no se presentan como parte de ese snapshot.

Un archivo se cierra sólo con propósito y dominio, productores/consumidores, unidades e identidad del activo/escala, disponibilidad causal, supuestos y coste, oráculos independientes y evidencia ligada a blob/SHA. Leer un hunk o aprobar una suite no cierra el archivo entero. El avance de esta adenda es una corrección del lanzador y la confirmación de un expediente ya abierto; no se suman cientos de supuestas auditorías.

| Orden | Lote coordinado | Criterio de salida |
|---|---|---|
| 1 | BASE: metas, contabilidad, causalidad y supuestos | Equity, costes y tiempo ejecutado conciliados; dato ausente separado de cero y ruina. |
| 2 | BASE/ROOT: Ω125–132 y sus consumidores | Contratos de dominio, identidad, simetría, unidades, maduración, estado y extremos; revisión de cada delta antes de cerrar rutas. |
| 3 | Dueños de riesgo/adaptación | Cada veto tiene oráculo de admisión/rechazo; cada parámetro tiene escritor causal, consumidor, validación, persistencia y rollback. |
| 4 | Recursos: esta entrega y scripts operativos restantes | Defaults coherentes con la regla de disco; concurrencia acordada; logs y cachés con dueño/cierre verificable. |
| 5 | Dueños de la cola BASE15 | Revisar todas las rutas pendientes por dependencia y riesgo, incluidos CI, tests, generadores y documentación; conservar hallazgos abiertos. |

## Recursos observados y acciones

Muestreo de sólo lectura aproximadamente entre 16:13 y 16:16 del 15 de octubre. Los tamaños se obtuvieron por recorrido streaming, omitiendo `.git` y puntos reparse/junctions. Son longitudes lógicas de archivos: no se midieron bloques físicos ni deduplicación. Son mediciones puntuales; no predicen espacio recuperable. El recorrido del `target/` principal se limitó a 50 segundos para contener el coste.

| Recurso | Medición | Interpretación |
|---|---:|---|
| C: libre / capacidad | 225,08 / 929,69 GiB | No se observó disco lleno en este corte. |
| RAM visible / libre | 23,38 / 4,40 GiB | Dato del sistema operativo; difiere de la premisa histórica de 16 GB. Presupuestar con disponibilidad actual. |
| `target/` principal | **al menos 131,107 GiB** | Cota inferior, 136.643 archivos recorridos; no es un total exacto. |
| `data/` | 23,925 GiB | Datos, preservar. |
| `.claude/worktrees/` | 13,218 GiB | Contiene trabajos y evidencia; el tamaño no acredita abandono. |
| `.sol-r6-2026-10-15/` | 10,193 GiB | Incluye `target` de 9,756 GiB y `target-portable` de 0,321 GiB. |
| `.glm124-sol-r4-01/` | 1,812 GiB | `target` de 1,696 GiB, activo en el corte. |
| `logs/` principal | 0,775 GiB | Requiere retención acordada con los dueños de recibos. |

Otros worktrees medidos: `model-lineage-audit` 15,228 GiB; `spectral-evidence-1011` 14,829; `base-audit-2026-10-15` 9,340; `root-spectral-zero-replay-1011` 7,954; `codex-review-2026-10-07` 4,579. No se suman automáticamente a los tamaños de la raíz: son ubicaciones distintas con alcances documentados. En BASE se observaron cachés con nombres alternativos; su turno y su árbol congelado se preservan.

`git worktree list` pasó de 27 a 28 checkouts durante la auditoría; el nuevo es el worktree sparse de esta entrega, sin target. Se observaron 14 detached. SI confirmó expresamente que `spectral-evidence-1011` sigue en uso y contiene una entrega no integrada: **detached no implica abandonado**.

| Cargo PID observado | Inicio Bogotá | Trabajo / caché |
|---|---|---|
| 38360 | 15:07:33 | GLM, `backtest-engine` release/no-run, `.glm124-sol-r4-01/target`. |
| 25468 | 16:06:52 | Sol, T-1 release, `.sol-r6-2026-10-15/target-portable`. |
| 19300 | 16:10:43 | BASE, `risk-engine --jobs 1`, caché anterior `target-base15`. |

No se observó `god_engine.exe` en los muestreos. Son identidades históricas, no autorización para actuar sobre un PID futuro. No se detuvo ningún proceso ni se borraron cachés, datos, ramas o worktrees ajenos. ROOT y BASE confirmaron que los trabajos externos deben permanecer activos. No se atribuye memoria liberada ni limpieza a esta entrega.

Antes de retirar un worktree o caché: identificar dueño y PID/StartTime, acreditar fin de uso, preservar cambios/commits exclusivos y recibos ignorados fuera del checkout, verificar restauración y resolver la ruta absoluta dentro del directorio autorizado. Sólo entonces retirar el checkout y podar su registro. Los artefactos regenerables de una caché activa no son candidatos inmediatos. No migrar flags, perfiles o rutas de una ejecución en curso.

## F7-T1: corrección operativa acotada

**Causa observada:** `scripts/t1_oraculo.sh` de main, último cambio `bb2e0bfe96819979e62175dce0c1ccad92b7ce0d`, creaba `target-portable` por defecto, elegía todos los núcleos y usaba ese mismo número como jobs de compilación. Contradice el default estándar establecido por Ω132 y multiplica la presión cuando varias sesiones ejecutan el lanzador.

**Cambio para futuras invocaciones:** target estándar `<worktree>/target`, un hilo del oráculo y un job de compilación por defecto, presupuestos independientes y validación antes de crear carpetas o ejecutar Cargo. Los hilos deben ser enteros positivos. `CARGO_BUILD_JOBS` conserva la sintaxis documentada por [Cargo](https://doc.rust-lang.org/cargo/reference/config.html#buildjobs): entero no nulo con signo opcional o `default`; los negativos expresan CPUs más ese valor. Se comprueba sintaxis/no cero, no el rango de enteros del Cargo instalado; Cargo conserva su validación final. Los overrides explícitos de `CARGO_TARGET_DIR`, `CARGO_BUILD_JOBS`, `T1_THREADS` y el argumento `[hilos]` se preservan; el argumento tiene precedencia sobre `T1_THREADS`. Preservar overrides evita migrar cachés existentes y no autoriza crear targets alternativos prohibidos por la regla del proyecto. El script todavía acepta ese override: su regularización con el dueño queda fuera de esta entrega y no se afirma cumplimiento integral. La elección serial es una política conservadora de recursos compartidos, no un parámetro del modelo financiero.

El log por defecto pasa a `logs/t1_oraculo.log`, reemplazado por invocación. `T1_LOG` conserva precedencia. Se preservan el sello del árbol, su invalidación, el perfil release, los argumentos del test, el criterio económico y el código de salida. Esta corrección no aporta exclusión global: incluso con un job por proceso puede haber varias compilaciones. El turno sigue acordándose entre sesiones. No se cambia la configuración Cargo ni los runners activos.

La regresión usa un repositorio temporal y un ejecutable Cargo simulado que captura argv/entorno. Nunca compila Rust, ejecuta el motor ni mide rentabilidad. Verifica defaults, overrides, entradas inválidas sin efectos, log, sello y propagación del fallo; el [recibo](RECIBO_RECURSOS_T1_2026-10-15.json) distingue RED y GREEN. La revisión cruzada encontró y corrigió en la propuesta inicial el rechazo de jobs negativos/`default` y el tratamiento de nombres de log relativos que empiezan por guion; esos casos tienen regresión específica. Las carpetas temporales se retiran al terminar los tests. No se acredita aquí la invalidación selectiva del sello entre dos builds reales.

Otros huecos permanecen abiertos para sus dueños: `build_loop.ps1` fuerza `target-sim-3`; `run_sim_5.ps1` fuerza `target-sim-5`; ambos permiten hasta 20 reintentos. `medir_camino_caliente.sh` conserva worktrees por diseño. No se halló exclusión global ni una política de retención en los scripts inspeccionados. `LockFreeLogger` añade sin rotación en la ruta revisada; `FlightRecorder` ya tiene almacenamiento circular acotado. Estos son candidatos a lotes posteriores, no archivos íntegramente auditados.

## Confirmación matemática, sin duplicar el expediente BASE

El revisor independiente confirmó el caso ya registrado por BASE en `dark-alpha-engine/src/hidden_liquidity.rs::compute_absorption` (Ω132): con volumen 500, bid=ask=100 y ATR=100, el reparto direccional fijo `(1, 0.5)` para delta no negativo fabrica presión `+1/3` en cero. Con delta negativo infinitesimal da `−1/3`: salto lateral de `2/3`. El amortiguamiento común se cancela en la normalización.

Oráculos mínimos: `Π(V,B,A,δ) = −Π(V,A,B,−δ)` y continuidad al aproximar cero; con libro simétrico y delta cero, la simetría exige `Π=0`. Probar sólo delta cero mediante otro `if` no elimina el salto lateral. Una reparación debe justificar el reparto con flujo firmado observable o una transición dimensionalmente consistente, probar profundidad/overflow y mantener los consumidores bajo el mismo corte.

La integración Ω133 observada **sin commit** en el checkout compartido publica esta presión desde core y la suma al OBI del motor de confluencia; con peso 0,30 y OBI real cero fabrica `obi_eff=0,10`. Esta observación pertenece al working tree, no a main Ω132. El método standalone `evaluate_conformal_gate` no tenía consumidor vivo fuera de tests en la búsqueda realizada. No se afirma que su rama de veto haya ejecutado órdenes. BASE posee la reparación/contrato pendiente; esta adenda no crea otro ID ni modifica esas fuentes.

## Fundamentos que deben conservarse al evolucionar

La meta continúa siendo multiplicar la equity neta por dos cada 72 horas: `ln(2)/3 = 0,2310490602` de crecimiento logarítmico diario, equivalente a aproximadamente 25,9921% compuesto diario. Medir su distribución entre ventanas, ruina, drawdown, exposición, capacidad y costes; no sustituirla por WR ni por esperanza aritmética positiva. No hay una demostración de esa meta en esta entrega.

Los atomics configurables requieren actualización causal, estado versionado y efecto medido para acreditar adaptación. Una mezcla espectral debe conservar identidad del activo, resolución y frescura a lo largo del circuito. Las analogías físicas o cuánticas necesitan una correspondencia explícita con observables y un baseline; su nombre no establece ventaja predictiva ni justifica coste adicional en el hot path.

Para afirmar cobertura conformal hacen falta scores, actualización/calibración, objetivo de error y la garantía pertinente. Multiplicar una banda por 0,70 no establece esa garantía. Gibbs y Candès estudian adaptación online bajo cambios de distribución ([NeurIPS 2021](https://proceedings.neurips.cc/paper/2021/hash/0d441de75945e5acbc865406fc9a2559-Abstract.html)); su trabajo posterior adapta también el paso de actualización y analiza regret local ([JMLR 2024](https://jmlr.org/papers/v25/22-1218.html)). Se propone contrastar esos mecanismos con el estimando y los datos del proyecto antes de implementarlos; no se les atribuye cobertura condicional por cada activo ni rentabilidad.

Para inferencia con inspección y paradas secuenciales, las secuencias de confianza ofrecen cobertura uniforme temporal bajo sus condiciones probabilísticas ([Howard et al.](https://arxiv.org/abs/1810.08240)). Al aplicarlas hay que declarar filtración, soporte/integrabilidad, apuestas predecibles y gasto entre reinicios/búsquedas. Esta referencia no certifica las implementaciones actuales de e-values.

## Coordinación, Git y cierre

Se enviaron avisos y se recibieron acuses de ROOT (`01a116f0`), SI (`01a0f2fc`) y BASE (`01a1404c`). BASE conserva el turno Cargo y su árbol congelado; ROOT sigue en su cadena de integración, y SI conserva replay/reporte. Nuestra reserva exclusiva es el lanzador T-1, su contrato y esta adenda. ROOT solicitó una entrega por commit acotado para incorporar al tren después de BASE, evitando otras CI frías; no se crea PR paralela.

Tras fetch, el corte observado fue main local/remoto Ω132. Ω133 seguía sin commit. Había PR abiertas con checks en curso o fallos, y refs con commits exclusivos; **no todo estaba en main**. Las refs integradas restantes estaban ocupadas/reservadas, por lo que no se acreditó ninguna eliminación libre. No se hereda CI de un padre ni del encabezado histórico del maestro.

ROOT mantiene reservadas memoria, maestro y coordinación; se le entregará el vínculo de esta adenda para su actualización. El recibo separa commit local, publicación de rama e integración real. El worktree sparse se retira después de preservar los recibos y entregar el commit; la rama sólo se elimina tras confirmar ascendencia en main. No se retiran refs ajenas.

### Resultado local de F7-T1

Código probado: `314962fe1d481b3689ce3b64fcb5af74bda96447`, dos rutas. El harness final permaneció idéntico en ambos lados: **11 métodos y 25 escenarios**. Sobre el lanzador original: 21 fallos esperados, cero errores y cero omisiones, exit 1 en 235,853 s. Sobre la corrección: todas las pruebas aprobadas, exit 0 en 121,937 s. Son 21 reportes de fallo, no 21 bugs distintos. `bash -n` y `git diff --check` dieron exit 0.

El RED final selecciona una copia byte idéntica del lanzador de Ω132 mediante un runner que sólo cambia la ruta `SCRIPT` del harness; conserva nombre, aserciones y fixtures. Los ensayos preliminares se archivan como histórico: su primera versión trataba equivocadamente jobs negativos de Cargo como inválidos, corregido antes de congelar la prueba final. Ninguno de estos ensayos compiló Rust ni ejecutó el motor.

La integración en main y su validación de composición siguen con ROOT después de BASE. Esta entrega local no les atribuye aprobación ni un push que todavía no ocurrió.
