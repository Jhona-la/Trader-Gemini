# Revisión V37 desde la base — diseño de verificación y evolución causal

## Intención

Reconciliar el avance real del ecosistema y convertirlo en una secuencia pequeña de correcciones verificables, sin confundir sofisticación matemática con ventaja económica. La duplicación del capital cada 72 horas permanece como hipótesis de investigación sujeta a evidencia OOS neta, nunca como garantía ni instrucción de apalancamiento.

## Objetivos

- Incorporar en `main` local el PR #171 ya fusionado en GitHub, con contrato y workspace comprobados antes del CAS.
- Publicar un plan común recuperable por cualquier agente, con propietarios, dependencias y criterios de salida.
- Cerrar el Minor pendiente del daemon: un fallo de guardado no debe provocar reentrenamientos cada 500 ms ni perder la obligación de persistir.
- Abrir la auditoría config/lib y dejar cada hardcode o sesgo como un átomo con consumidor vivo y prueba causal.
- Conservar ramas, worktrees, overlays y evidencia mientras exista trabajo o propiedad no resueltos.

## No objetivos y antiobjetivos

- No ejecutar una reescritura masiva ni añadir teorías por nombre o complejidad aparente.
- No rehacer la persistencia `online-persistence-v1-fixes` ya validada ni promover overlays cuya puerta Clippy sigue roja.
- No cambiar V2, su fixture, comparador, ABI 144 o mínimo `0.110`; QRA decide e implementa cualquier revisión formal.
- No editar B1/A4-T2, QRA/Navier/`adversarial_gate`, legacy ni WIP de otros agentes sin handoff explícito.
- No borrar ramas, worktrees, overlays, datos, modelos, logs activos o artefactos por antigüedad/tamaño.
- No ejecutar Demo/Mainnet/Producción, leer credenciales, detener procesos ajenos ni fabricar métricas económicas.
- No declarar attestation, CI verde, ausencia de ruina o rentabilidad a partir de compilaciones aisladas.

## Estado comprobado al diseñar

- `HEAD=fb780dd46fecc3ea323586a28a299ac22ec82763`.
- PR #171 fusionado en GitHub: `origin/main=8493e48e1832a6d6505dccae70a0a4fa204fba9e`; doce checks del PR verdes y CI del merge en curso al diseñar.
- Árbol local de fusión: `4e1d6fb5f23cc20119641796dbeb6ebc0384834f`, sin conflictos. Contrato I-13 y workspace aún deben terminar antes del CAS local.
- `claude/linea-i-i13` sigue remota al fusionar; su worktree está limpio, pero sólo se retira después de CI y de preservar el trabajo.
- Otras ramas/worktrees (`agy/wave3-omega220`, `claude/linea-i-i14`, `claude/qsr-lc20r2`, `claude/qsr-lcf020`, AGENT-01) permanecen intactos hasta revisión individual y rescate.
- Persistencia smartcore/GBDT: 8/8 hardening, 6/6 persistencia, 2/2 determinismo, 229/229 crate y workspace verde en overlay. CAS de código bloqueado por Clippy baseline.
- Minor abierto: `retrain_forest_and_persist` devuelve el mismo error para entrenamiento y escritura; `pending_new_obs` sólo se reinicia cuando ambos tienen éxito, así que un fallo persistente puede reentrenar cada 500 ms.
- Auditoría económica: productor real de trades desconectado y semántica bruto/neto pendiente de contrato de dominio; ROI/PnL/WR/Sharpe/EV=`N/D`.
- V2 sigue 14/144 bajo `0.110`; B1 permanece bajo A4-T2. No existe handoff nuevo para esos alcances.
- Recursos observados: 0,89 GiB libres de 23,38 GiB y 164 GiB SSD disponibles. No se usa esa medición como admisión ni como permiso de borrado.
- Grafo existente: 19.105 nodos. Se consulta incrementalmente; no se reconstruye ni se gasta cuota en extracción redundante.

## Enfoques considerados

1. **Refactor masivo/teoría primero:** rechazado por mezcla de sujetos, riesgo de Goodhart y ausencia de atribución económica.
2. **Verificación primero (elegido):** reconciliar Git/CI/propiedad y aplicar un átomo concreto con RED→GREEN por vez. Conserva capacidad de rescate y atribución por sujeto.
3. **Sólo plan/documentación:** rechazado porque dejaría el defecto de reintentos sin corregir y no demostraría comportamiento.

## Secuencia y criterios de salida

### Fase 0 — Corte de verdad y validación de fusión

1. Terminar contrato I-13 y `cargo check --workspace --all-targets --locked` sobre `4e1d6fb5`.
2. Observar CI del merge `8493e48e` hasta resultado completo; no afirmar verde mientras esté en curso.
3. Crear commit local con padres `fb780dd4` y `8493e48e`, revisar diff contra ambos y hacer CAS con expected `fb780dd4`.
4. Retirar `claude/linea-i-i13` remota sólo tras CI verde y PR `MERGED`. Retirar worktree limpio antes de `branch -D`; cualquier suciedad exige rescate, no fuerza.
5. Registrar las demás ramas individualmente. No se elimina ninguna sin PR/ancestría, cero commits exclusivos, worktree libre, archivos rescatados y ausencia de lease/reserva.

### Fase 1 — Plan y sincronización documental

1. Revisar diseño con builder independiente y resolver cualquier Critical/Important.
2. Convertirlo en plan V37 con interfaces, comandos, resultados esperados y propietarios.
3. Actualizar heartbeat, buzón y ledger. CAS sólo de documentos propios, con índice privado y expected SHA; no push ni attestation.

### Fase 2 — Átomo backoff de persistencia

1. Escribir RED que reproduzca un fallo de guardado tras un reentrenamiento exitoso y pruebe que la implementación actual vuelve a entrenar antes de reintentar persistir.
2. Introducir error tipado `RetrainFailed`/`PersistFailed`, conservar una generación completa pendiente de guardado y reintentar sólo `save_snapshot` con backoff acotado, sin reiniciar `pending_new_obs` ni presentar persistencia completa.
3. Probar fallo permanente, recuperación posterior, ausencia de reentrenamiento duplicado y conservación de observaciones/modelo bit a bit. No usar tiempo real en tests: inyectar un calendario/cooldown determinista.
4. Ejecutar contratos de persistencia/hardening, toda `evolution-engine`, workspace y Clippy. Revisión independiente antes de cualquier CAS de código, que además sigue bloqueado por baseline.

### Fase 3 — Config/lib, hardcodes y sesgos

1. Censar valores fijos, fallbacks y conversiones de ausencia en `quantum-arena/src/config.rs`, `quantum-arena/src/lib.rs` y `src/lib.rs` con llamadores reales.
2. Seleccionar sólo un defecto libre, con hipótesis causal y sin invadir QRA/Navier/legacy. Escribir RED antes de editar producción.
3. No reemplazar un default legítimo por otro valor fijo. Derivar de evidencia o marcar ausencia `N/D`, preservando vetos justificados.

### Fase 4 — Ignición, Demo y OOS (bloqueada por propietarios y evidencia)

1. B1/A4-T2 y V2 conservan dueños. No activar esos alcances sin handoff.
2. Prerregistrar dataset/periodo, commit incumbente, fees/funding/slippage reales, partición walk-forward, multiplicidad y regla de decisión antes de cualquier medición.
3. Demo requiere `DEMO_MAINNET_AUTHORIZATION.json` con SHA/config/periodo/entorno/aprobador humano. Producción exige autorización separada y OOS favorable.
4. Reportar ROI/PnL/WR antes y después de fees, Sharpe, EV/trade, periodo y dataset. Ausencia = `N/D`. La hipótesis +100 %/72 h se marca compatible, refutada o insuficiente, nunca forzada con leverage.

### Fase 5 — Revisión, handoff y limpieza condicional

1. Revisión independiente de cada átomo. Critical/Important exige corrección con prueba; Minor queda en ledger.
2. Comparar diff contra cada padre y repetir la suite del sujeto final. No transferir PASS entre SHA, overlays, perfiles o toolchains.
3. Publicar sólo con attestation válida y hook; observar CI del mismo SHA. Sin `--no-verify` ni CI heredado.
4. Eliminar referencias únicamente cuando todas las guardas de Fase 0 estén cumplidas; conservar todo lo demás.

## Decisiones de arquitectura

- Un único `CARGO_TARGET_DIR` canónico, nightly `2026-06-30`, overlays `.agent_tx/staging/KIMI-QUANT-SR-V35/` e índice privado absoluto `.git/index_KIMI-QUANT-SR-V35`.
- Cada recibo liga sujeto, fuentes, hashes, comandos, perfil y resultado. `check` no equivale a tests ni attestation.
- Recursos: no purga para resolver RAM/SSD. Cualquier limpieza futura exige propietario, procedencia, necesidad y evidencia de regenerabilidad.
- Nuevas teorías sólo con invariante analítico, consumidor vivo, baseline, ablación y superioridad OOS. Ninguna teoría es requisito para el átomo backoff.

## Supuestos adoptados por ejecución automática

- El mandato amplio autoriza elegir verificación primero y ejecutar sin nuevas preguntas por cada documento o átomo.
- La fusión del PR #171 está autorizada por el pedido explícito de integrar avances valiosos. Otras ramas no se consideran abandonadas por antigüedad.
- La corrección del backoff se hace en overlay nuevo y no se promueve mientras Clippy esté rojo.
- No se reabren V2/B1 ni las reservas A4-T2/QRA/Navier/adversarial/legacy.
- La fecha de corte es 2026-11-05. Cualquier cambio posterior de refs exige recomponer el plan, no fingir continuidad.

## Resoluciones del builder check (vinculantes)

### 1. Sujeto inmutable por fase

- `4e1d6fb5f23cc20119641796dbeb6ebc0384834f` es un **tree**, no un commit. Fue validado mediante el overlay materializado `.agent_tx/staging/KIMI-QUANT-SR-V35/pr171-merged/subject`.
- `c4db0cdd48dc0fe031fb0fbe181ee58ff1d7a2e1` es el **merge commit** creado con padres `fb780dd4` y `8493e48e`, con ese mismo tree. El CAS local, si CI termina verde, usa ese commit y sus padres, no el tree como si fuera commit.
- Fase 2 no recompone ni mezcla `online-persistence-v1-fixes` con PR #171. El átomo backoff vive en el overlay separado `.agent_tx/staging/KIMI-QUANT-SR-V35/persist-backoff-v1/subject`, compuesto desde ese candidato previo. Toda evidencia de Fase 2 exige nueva ejecución; ningún PASS anterior se transfiere por semejanza de fuentes.

### 2. Autorización para retirar `claude/linea-i-i13`

- La autorización procede del mandato explícito del operador de fusionar avances valiosos y eliminar ramas ya fusionadas, no de un handoff de Línea I ni de una attestation de push.
- Sólo se ejecuta después de PR `MERGED`, CI verde, worktree limpio y preservación de cualquier trabajo pendiente. `git worktree remove` sin `--force` antes de `branch -D`; `git push origin --delete` es la única mutación remota y está expresamente autorizada por ese mandato.
- Si CI es rojo/cancelado, el worktree cambia o hay archivos sin rescatar, no se borra nada: se conserva y se audita. La misma regla se aplica a cualquier otra rama, sin excepción por antigüedad.

### 3. CI como puerta

- CI verde del merge `8493e48e` es obligatorio tanto para el CAS local de ese merge como para retirar la rama y continuar la integración de ese sujeto. Un resultado rojo o cancelado detiene esa línea de integración: no hay CAS local, no hay eliminación y V37 se recompone desde `fb780dd4`, sin reescribirlo ni ocultar el fallo.
- El átomo backoff puede continuar en overlay independiente, pero tampoco se promueve a `main` mientras Clippy esté rojo.

### 4. Máquina de estados del backoff

- Mientras exista `forest_persist_retry=Some(...)`, se inhiben **ambos** disparadores de reentrenamiento: el periódico por `obs_count`/`iteration_count` y el disparado por `pending_new_obs`.
- La espera entre reintentos crece exponencialmente desde 500 ms hasta un máximo de 30 s. **Sólo se acota la espera, no el número de intentos.** Se reintenta hasta que exista un snapshot válido; nunca se abandona la obligación de persistir.
- Cada reintento usa la generación completa ya entrenada. No se vuelve a entrenar si no se aceptaron observaciones nuevas; `model_generation` debe permanecer igual en ese caso.
- Las observaciones que lleguen durante el backoff se conservan. La llegada se detecta mediante `observation_revision`, una revisión monótona en memoria incrementada al aceptar cada observación, nunca mediante `len()`: el drenaje de capacidad puede reducir la cardinalidad sin borrar la obligación de aprender. Si la revisión supera la registrada al fallar el guardado, la recuperación primero guarda la generación pendiente y después entrena una nueva generación con todas las observaciones y la vuelve a guardar. No se reentrena dos veces para las observaciones ya incluidas.
- `observation_revision` se inicializa al restaurar y no se reinicia al reentrenar. No forma parte del payload `TGFOREST`, cuyo formato permanece intacto.
- `pending_new_obs` sólo se reinicia cuando el snapshot final contiene todas las observaciones vistas. Nunca se borra señal de entrenamiento por una simple recuperación de disco.
- `RetrainFailed` no crea un reintento de persistencia. `PersistFailed` lo crea o actualiza. Los tests comprueban ambos disparadores inhibidos por separado, espera acotada, recuperación sin cambio de generación cuando no hay observaciones nuevas y exactamente una nueva generación cuando sí las hay, incluido un buffer lleno cuyo `len()` disminuye tras drenar.

## Registro de ejecución y handoff (2026-11-05)

- PR #171: ambos workflows SUCCESS sobre `8493e48e` (Unit `42124750058`, Replay `42124750097`). CAS local completado: `main=c4db0cdd48dc0fe031fb0fbe181ee58ff1d7a2e1`, padres `fb780dd4`/`8493e48e`, tree `4e1d6fb5`. `claude/linea-i-i13` y su worktree fueron retirados tras verificar PR `MERGED`, CI, ancestro y worktree limpio. Las demás ramas/worktrees permanecen intactos.
- Backoff: el único sujeto válido es `.agent_tx/staging/KIMI-QUANT-SR-V35/persist-backoff-v1/subject`; `online-persistence-v1` está reemplazado. Revisión independiente agent-41: un Critical por cardinalidad tras drenaje y un Important por aislamiento del trigger pendiente, ambos corregidos con RED→GREEN. Dos Minor quedan diferidos en el ledger: contenido del snapshot no cargado explícitamente/contabilidad no extraída y el coste temporal inicialmente subestimado del próximo trigger periódico.
- Evidencia final del átomo: 234/234 en 27 suites, 0 ignored/filtered (`full-crate-revision.log`); workspace exit 0 en 3m08s (`workspace-check-revision.log`); Clippy 33 lib/40 lib-test, comparación `ADDED=[]`/`REMOVED=[]` (`clippy-revision.log`). El CAS de código sigue bloqueado por baseline roja. Rescate: `persist-backoff-v1/revision.patch`.
- Config/lib: `.agent_tx/diagnostics/config-lib-v37-census.json` registra cuatro `NO_CHANGE`, cero `OPEN_ATOM` y ningún cambio de producción. El drawdown conserva su derivación actuarial y contrato; las tasas escalares conservan su escenario explícito y precedencia de evidencia por símbolo.
- Sincronización: `.agent_tx/diagnostics/base-review-v37-status.json` es el estado máquina; `.superpowers/sdd/2026-11-05-base-review-v37/progress.md` conserva rulings y Minor. Producción NO-GO; ROI/PnL/WR/Sharpe/EV=`N/D`. No hubo push de `main`, attestation, Demo ni operación financiera.
- Reanudación: no rehacer el átomo ni transferir sus PASS a otro SHA. B1/A4-T2, QRA/Navier/`adversarial_gate` y V2 permanecen con sus dueños. Cualquier OOS exige primero dataset/periodo, commit incumbente, fees/funding/slippage reales, walk-forward, multiplicidad y regla de decisión prerregistrados; Demo exige `DEMO_MAINNET_AUTHORIZATION.json` y Producción autorización separada con OOS favorable.
