# Plan coordinado de integración y diagnóstico de GitHub — 15 de octubre de 2026

> Corte histórico: la cola siguiente corresponde al 15 de octubre. Para el estado actual, véase [Puerto F7 y GitHub del 20 de octubre](PUERTO_F7_RECURSOS_2026-10-20.md). B15-16 es un identificador propuesto pendiente del ledger de BASE; no se registra aquí como hallazgo nuevo.

Este anexo complementa PLAN_BASE15 y el maestro del día; no crea otro censo. Corte de GitHub: 2026-10-15T22:14:16.268Z. Main observado: `cd9feaa0aaa89e6b45d6363a23a4fbf365ea41d5`. No certifica una composición posterior.

## Causas verificadas

Main y PR68/69 alcanzan las aserciones Rust, pero tres fixtures de `evidence_expiry_contract.rs` observan Long donde esperan Flat (líneas 284, 353 y 437; 10 aprobadas y 3 fallidas). El candidato fixture-only ROOT `7c24796b9b5eaf57509ba528272060cc7a837e8f` conserva el oráculo Flat y adapta las precondiciones CL tras Ω130; aún requiere validación compuesta. [Job de main](https://github.com/Jhona-la/Trader-Gemini/actions/runs/38811562463/job/116685791405). No son fallos de compilación, autenticación, memoria o enlazado en los logs examinados.

Se observó un timeout TLS puntual del CLI, mientras otras consultas y descargas funcionaron. Esto no demuestra una caída general de GitHub ni credenciales inválidas. `gh auth status` agotó otra espera de 15 segundos; ese resultado tampoco identifica una causa de autenticación. El runtime auxiliar Node obtuvo exit 4 (sin sesión CLI) en tres GET; PowerShell sí autenticó. Se comprobó el mismo gh.exe en ambos procesos. La diferencia concreta es APPDATA: ausente en el hijo Node y presente como C:/Users/jhona/AppData/Roaming en PowerShell. Sin GH_CONFIG_DIR/XDG_CONFIG_HOME/HOME ni tokens de entorno en ambos, PowerShell encuentra el directorio GitHub CLI con hosts.yml/config.yml, mientras Node usa un fallback ~/.config/gh inexistente. Sólo se comprobó existencia, sin leer credenciales. Para estas operaciones se utiliza el shell que conserva la configuración o el conector; no se inicia otro login ni se modifica el perfil global. Esto explica los exit 4 de ese runtime, no todos los fallos del proyecto. El GET autenticado por PowerShell reportó main cd9 con protected=false. A las 22:30:32Z el conector devolvió una lista vacía para rulesets?includes_parents=true. Otros dos GET por CLI agotaron 18 segundos; se conservan como timeout, sin inferir permisos denegados. El conector reportó permisos de repositorio admin/maintain/push/pull/triage, que no se transfieren al CLI. Los flags allow_merge/squash/rebase quedaron desconocidos porque no venían en la respuesta. La configuración observada no aporta una exigencia de CI previa para main; no se atribuye con ello el autor ni la política exacta de un push pasado. ROOT conserva el train; no se cambia una política durante esta integración.

El hook local `pre-commit` sólo comprueba marcadores y whitespace del índice con `git diff --cached --check`; no ejecuta Rust ni accede a la red. Signing no está configurado, la identidad sí. No se imprimieron credenciales.

Los commits propios `16b02279c4b0ca79d66f6cac95ac44ddbe473077`, `ef4cd9c4c5bf31ac21367350b4a6ddc7f8cb24a2` y `2a49643529fffc27875c574bea0e1773f17ede45` existen localmente y siguen fuera de main. Son entregas pendientes, no evidencia de que fallara su creación. Usar SHA completos: `2a49` resulta ambiguo con blobs locales.

## Cola y responsables

| Entrega | Estado del corte | Acción y dueño |
| --- | --- | --- |
| [PR64](https://github.com/Jhona-la/Trader-Gemini/pull/64), dcd36d3 | Seis checks históricos verdes; conflictos en memoria/buzón | ROOT/H preservan ambos cuerpos y validan la composición |
| [PR65](https://github.com/Jhona-la/Trader-Gemini/pull/65), c3abca5 | Sin checks de ese head; conflictos en buzón/maestro | BASE termina siete gates del árbol Ω132 congelado, publica su head exacto y cede Cargo |
| [PR66](https://github.com/Jhona-la/Trader-Gemini/pull/66), 794715f | Sin checks de ese head; conflictos de planes/recibos | ROOT resuelve semánticamente, sin heredar CI de padres |
| [PR67](https://github.com/Jhona-la/Trader-Gemini/pull/67), de534db | Cinco checks históricos verdes; sin marcadores detectados | ROOT/QUANT revalidan con el padre actual; va antes de PR69 |
| [PR68](https://github.com/Jhona-la/Trader-Gemini/pull/68), 05bc2f9 | Núcleo falla los tres fixtures; entrenador verde | SI conserva su corrección causal; ROOT/BASE validan fixture CL común |
| [PR69](https://github.com/Jhona-la/Trader-Gemini/pull/69), c2fb494 | Núcleo y replay fallan los mismos fixtures | Integración posterior a PR67 y reparación CL |
| Main cd9feaa | Núcleo rojo; replay pendiente detrás del run anterior | ROOT publica sólo tras validar la nueva composición |
| F7-T1 16b02279 | Contratos stub aprobados, no integrado | ROOT conserva el pin exacto; once métodos/25 escenarios históricos y un método/cuatro casos de formatos, sin Cargo real |
| Wrappers ef4cd9c4 + 2a496435 | Siete métodos/14 escenarios stub aprobados, no integrados | Lote separado del pin T1; no seguir la ref móvil |

La detección de conflictos fue una lectura `merge-tree` contra cd9, no un merge realizado ni un check de tipos. Los checks PR68 usaron checkout c52983a con base ba445; PR69 usó a840c5d con base dfa4. Ninguno acredita el árbol main cd9. El replay pendiente de main no se cuenta como fallo.

El pin T1 es la cabeza de una cadena, no un cherry-pick aislado: `314962fe1d481b3689ce3b64fcb5af74bda96447` → `94eff4571b65668ac8d2a13c3fbddc73ef9d853a` → `16b02279c4b0ca79d66f6cac95ac44ddbe473077`. Main cd9 no contiene esos padres; el test de formatos importa el harness agregado en 314962. Integrar la cadena por ascendencia o aplicar esos tres commits en orden. Los wrappers ef4cd9c4 → 2a496435 constituyen el lote posterior separado.

## Próximos gates acordados

1. BASE completa su fuente congelada Ω132, registra resultados terminales y publica PR65; después cede explícitamente la cadena Cargo. No se cambia su fuente a mitad de prueba.
2. ROOT compone latest main, BASE y entregas aceptadas por SHA, resuelve los conflictos conservando ambos padres y revisa el diff contra cada padre. La reparación B15-16 de liquidez oculta sigue reservada por ROOT fuera del árbol BASE. Nuestro apoyo es revisión independiente; no hay otro parche paralelo instalado.
3. La nueva composición necesita contratos afectados, suites acordadas, `cargo check --workspace --all-targets` y T1 final cuando corresponda. Los verdes históricos y los stubs de wrappers mantienen su alcance.
4. Publicar y verificar el SHA remoto de main y la ascendencia de cada entrega aceptada; sólo entonces cerrar sus refs y retirar worktrees propios elegibles. No borrar ramas con commits exclusivos o reservas. Tras ese cierre, acordar la protección de main y checks requeridos con los agentes que publican; este diagnóstico no los cambia unilateralmente.

## Liquidez oculta y recursos

La causa deduplicada de liquidez oculta, con identificador propuesto B15-16 pendiente del ledger BASE, ahora es alcanzable en main Ω133 por Core→hidden→Flow. El estado anterior da presión +1/3 con libro simétrico y precio plano por multiplicar sólo un lado por 0.5. La revisión del borrador ROOT exige neutralidad, espejo, límites representables y publicación neutral sin volumen; no confunde agresor observado, absorción pasiva y retorno futuro. Los pesos suavizados propuestos son scores de una política, no un reparto físico del volumen ni una identificación de Kyle/Hasbrouck. La banda 0.70/1.50 no acredita cobertura conformal. [Revisión independiente causal y numérica](REVISION_INDEPENDIENTE_B15_16_2026-10-15.md). La revisión propia causal se entrega por separado, sin ejecutar Rust.

Snapshot 17:17:42 Bogotá: RAM libre 2.90/23.38 GiB; C libre 314.15 GiB. BASE, GLM, Sol y AGY tenían compilaciones activas. No se abre otro Cargo propio; futuros builds usan únicamente target/ estándar y un job. No se modifican cachés, modelos, perfiles, flags o procesos de otros. No se atribuyen variaciones de espacio libre a un autor sin evidencia.

Primero hubo intentos de coordinación sin respuesta; no se tomaron como ejecución. Después, la coordinación por herramienta ya obtuvo respuesta explícita de ROOT en esta ola: conserva B15-16 en exclusiva, acepta el pin T1 16b con sus padres, revisará ef4/2a496 como lote separado e integrará las entregas después de la cesión BASE. No se abrirá un PR T1 paralelo ni se conectará el agresor en paralelo. El acuerdo durable ROOT/BASE de las 22:06:12Z registra cargo_chain_released=false; la respuesta posterior confirma que el Core BASE sigue compilando. Las solicitudes anteriores sin acuse no cuentan como acuerdos.

[Detalle por SHA, run, job y conflicto](TRIAGE_PR64_69_MAIN_CD9.json). Los logs completos siguen bajo logs/ci-triage-20261015, fuera de Git. Este documento no demuestra rentabilidad ni cumplimiento de la meta económica.


## Preservación comprobada para ROOT

[Verificación independiente del archivo de continuidad](ROOT_ARCHIVE_VERIFY_RECURSOS_2026-10-15.json): 545 payloads/9.843.199 bytes; hashes de MANIFEST y COMPLETE coincidentes; cero faltantes, extras o reparse. Se cotejaron cuatro copias working con archivos físicos y cuatro blobs Git con 019e6d47; sus diferencias entre working/Git se explican por CRLF. El bundle y sus prerrequisitos quedaron cotejados contra metadata, sin ejecutar restauración ni validar un build. No incluye los borradores hidden posteriores.
