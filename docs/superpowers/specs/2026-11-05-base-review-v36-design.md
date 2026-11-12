# Revisión V36 desde la base — diseño de recuperación verificable

## Intención

Construir una ruta reproducible desde el estado real del repositorio hasta evidencia Demo/OOS neta, sin confundir sofisticación matemática con ventaja económica. La duplicación del capital cada 72 horas es una hipótesis a falsar, nunca una garantía ni una instrucción de apalancamiento.

## Objetivos

- Reconciliar `main` y su WIP sin perder trabajo ni atribuir resultados al sujeto equivocado.
- Resolver el significado del gate V2 sin bajar `0.110`, cambiar su fixture para buscar verde ni fabricar sensibilidad.
- Completar attestation y CI por SHA antes de publicar; preservar contratos fail-closed ya integrados.
- Auditar recursos, ramas, arranque, fees, configuración, aprendizaje online y persistencia con pruebas causales.
- Incorporar teorías sólo con invariante analítico, consumidor vivo, baseline, ablación y superioridad OOS.

## No objetivos y antiobjetivos

- No activar Producción ni prometer +100 %/72 h, `P(ruina)=0`, nanosegundos ni paridad Demo/Prod.
- No sustituir V2 por V3r2 por agregación de contextos, borrar fallos ni convertir `N/D` en cero.
- No adoptar WIP ajeno, invadir QRA/Navier/adversarial/legacy, borrar overlays ni purgar ramas/worktrees activos.
- No limpiar `data/`, `models/`, `.git/`, recibos, locks, índices, logs activos o artefactos de procedencia incierta.
- No añadir un catálogo de teorías, mocks, placeholders ni código sin consumidor real.

## Enfoques considerados

1. **Teoría primero:** añadir modelos avanzados y optimizar después. Rechazado: mezcla sujetos, agrava atribución y Goodhart.
2. **Verificación primero (elegido):** reconciliar evidencia y propiedad, clasificar V2, cerrar attestation y sólo entonces evolucionar por átomos RED→GREEN.
3. **Limpieza primero:** borrar artefactos grandes y ramas para acelerar el entorno. Rechazado: tamaño no demuestra que un artefacto sea regenerable ni que el trabajo esté abandonado.

## Estado comprobado

- `HEAD=a27ea15166e0833aa3f52dd3b97f1766df85bde3`; `origin/main=8277c8f18f4e88063cb8aa88e9f7c74686b45e28`; 79 commits locales exclusivos y 1 upstream por reconciliar. Refs se revalidan antes de cada CAS.
- `origin` contiene `main` y `claude/linea-i-i13`; además existe el worktree `claude/linea-i-i14` en el mismo SHA de `origin/main`.
- PR #171 sigue abierto con cinco commits exclusivos. Las otras ramas secundarias tienen worktrees: ninguna es eliminable ahora. `agy/wave3-omega220`, `claude/qsr-lc20r2` y `claude/qsr-lcf020` sólo serán candidatas tras liberar worktree y revisar archivos; `linea-i-i14`, igual.
- El último `main` con workflows relevantes observado está verde en GLM-193. El fallo GLM-192 fue `NaN == NaN` en `fr_deformacion_publicacion_contract` y ya tiene corrección verde posterior; no se reabre.
- V2 sigue rojo: 14/144 frente a `0.110`. R28 no altera ninguna de las 144 huellas respecto de su base. V3r2 no es un reemplazo válido: agrega contextos y cambia el numerador.
- Censo V2: 14 huellas distintas, 4 coordenadas acopladas `[140,141,142,143]`, 4 peticiones borradas `[13,14,15,16]` y 5 ajustes de valor `[13,14,15,16,143]`.
- Lectores verificados: Kelly 5/5, curvas 3/3, paridad 6/6; medición auxiliar 15/15. No hay justificación para “reconectar” genes por el número.
- RAM: 23,38 GiB visibles, ≈2,17 GiB disponibles. Aplicaciones de agentes/IDE dominan el working set; no se cierra ningún proceso.
- Censo de disco: comando agotó ~13 minutos sin resultado; tamaños actuales quedan `N/D`. Los valores históricos no se presentan como actuales ni autorizan borrado. No se repite el censo ciegamente.

## Decisiones de arquitectura

- Trabajar en `.agent_tx/staging/KIMI-QUANT-SR-V35/` con índice privado y merge-tree/CAS; no instalar cambios en el árbol compartido durante la revisión.
- Preservar V2 y su fallo. V3r2 puede ser diagnóstico complementario, nunca evidencia de sustitución del trinquete.
- Mantener un único sujeto por recibo: SHA, WIP, herramientas, perfil y comandos. Un `check` no equivale a tests ni a attestation.
- Una hipótesis y un contrato por átomo. Sin RED conductual no se edita producción. Sin suite requerida no hay CAS de código.
- Recursos: conservar primero propiedad y procedencia. `data/`, `models/`, recibos y trabajo no rescatado son intocables. Limpieza sólo de artefactos fríos, regenerables y verificados.

## Secuencia restante

1. Publicar este diseño, el plan, el censo V2 y la auditoría de ramas/recursos por CAS documental.
2. Ejecutar y registrar el mapa de attestation existente, sin modificar sus archivos reservados.
3. Entregar a QRA el censo V2 para decidir contrato. Sólo una desconexión demostrada habilita una corrección de producción.
4. Cerrar huecos de attestation por átomos pequeños y luego validar el mismo SHA final.
5. Auditar arranque/OMS/fees/Top 10, persistencia y aprendizaje online antes de nuevas teorías.
6. Ejecutar shadow/replay y Demo cuando esté autorizado el entorno; reportar ROI/PnL/WR antes y después de costes, Sharpe, EV/trade, periodo, dataset y commit. Ausencia = `N/D`.
7. Producción sólo con evidencia OOS y autorización humana vinculada al SHA/configuración.

## Supuestos adoptados por ejecución automática

- El mandato amplio autoriza elegir el enfoque y ejecutar incrementalmente; no se solicita una nueva aprobación por cada documento.
- QRA conserva la decisión del contrato V2 y la propiedad de Navier/adversarial/legacy. No se reasignan por antigüedad.
- Ningún resultado histórico se transfiere al SHA actual. No hay Demo, Mainnet, attestation, CI o rentabilidad certificada.

## Resoluciones obligatorias del builder check

1. **Sujeto de attestation:** sólo el árbol formado por los padres comprometidos y los documentos propios explícitamente nombrados. El WIP del checkout no es elegible ni se mezcla implícitamente. Ningún overlay de terceros entra sin manifiesto propio, propietario y revisión separada.
2. **V2 inmutable y propiedad:** fixture, comparador, ABI 144 y mínimo `0.110` no cambian. Una revisión formal del contrato sólo puede ser decidida **e implementada por QRA**. Este plan no la ejecuta ni la promueve. Una corrección de desconexión real en código no-QRA exige propietario libre y no puede alterar ese contrato.
3. **Si V2 permanece rojo:** attestation, push final y certificación económica quedan bloqueados. Se permite continuar auditoría y correcciones locales no-QRA con sus propias pruebas y sin fallos nuevos, pero attestation sigue falso y no hay publicación final ni certificación de rentabilidad.
4. **Demo/Mainnet:** es una puerta de autorización, no una prohibición eterna. Ejecutar Demo exige `DEMO_MAINNET_AUTHORIZATION.json` con SHA, configuración, periodo, entorno y autorizador humano vinculados. Producción requiere otra autorización explícita más evidencia OOS. Sin esos artefactos no se ejecuta Demo, Mainnet ni Producción y las métricas quedan `N/D`.
5. **OOS prerregistrado:** antes de cualquier medición se fijan dataset/periodo, commit incumbente, costes reales de fee/funding/slippage, partición walk-forward, multiplicidad y regla de decisión. El mínimo 72 h/20 trades es operativo, no suficiencia estadística. Se exige EV neto > 0 y crecimiento logarítmico ≥ ln(2)/72 h para declarar la hipótesis compatible; si no, resultado refutado o `N/D`. Ningún criterio se elige tras ver resultados.
