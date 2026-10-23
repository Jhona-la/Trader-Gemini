# Continuación QRA por fases — 2026-10-23

Corte 2026-10-23T01:35:52-05:00. Base 75be5d6751b0075bbf061ce9f5b2ee09a36b427f; rama codex/qra-capital-20261023 y worktree propio locked ACTIVE. El cierre Omega220 PR102/104 permanece válido para su código e286 y sus 45 objetivos; no certifica esta nueva ola. El [inventario](QRA_CONTINUACION_INVENTARIO_2026-10-23.tsv) enumera 2193 rutas del árbol base por blob. Es una cola completa de archivos, con revisión pendiente; las 23 coincidencias de Omega220 sólo heredan su alcance concreto. La fase de la tabla es clasificación mecánica, no auditoría ni prioridad científica demostrada.

## Orden y aceptación

| Fase | Trabajo | Estado y condición de cierre |
|---|---|---|
| 0 | Recursos, semáforo, perfiles, dependencias, higiene | Auditoría en curso. RAM libre observada 1.53 y después0.46GiB; Cargo/rustc ajenos activos. No Cargo ni borrado de artefactos durante uso ajeno. Sólo target canónico y debug0/strip symbols. Medir root/target/volumen por separado; cargo tree --duplicates pendiente de capacidad segura o CI. No asumir dual-channel completo de 23.38GiB visibles. |
| 1 | Ramas locales/remotas, rescate, integración y CI | Remoto main75be sólo añade ocho documentos al código e2869/9. AGY PR103 f174 tiene10/10 sobre d64, no sobre composición con e286. Sol f1095 activa y GLM154 con cambios; Verify locked ACTIVE. Ninguna ajena acreditada abandonada. No borrar avances activos. Coordinar cobertura y renovar composición antes de integrar. |
| 2 | Capital/config/API, fees, OMS, red y precisión | Siguiente átomo candidato: tres exports FFI de backtest fabrican13USD ante INITIAL_CAPITAL ausente/inválido. Pointer export omite null de volumes; mmap calcula len*32 sin checked_mul. Confirmar preflight y compatibilidad, implementar fallo explícito y probar el export real, sin convertir datos inválidos en capital. Fuera de este átomo: cambios de transporte, MXCSR/parsers AGY, workflows Sol, DMRG GLM. |
| 3 | Evolución online, Top10, persistencia y costo total | Pendiente: grafo productor-consumidor por motor, causalidad temporal, estado duradero y ahorro de recursos medido. Ingreso de wallet real ya existe en el host; escenario FFI debe declarar su capital. No introducir red dentro de un replay reproducible sin snapshot de cuenta fechado. |
| 4 | Modelos, canarios, concurrencia y Demo | Pendiente: insumo causal, dominio numérico, unidades/formas, productor y consumidor reales, baseline, costo y experimento OOS por teoría. Hipótesis MPS/Koopman/ACI y restantes del adjunto no quedan certificadas por sus nombres. No declarar ruina cero, nanosegundos o crecimiento100%/72h sin medición. |
| 5 | Visor y meta-analítica | Pendiente: linaje hasta ejecución, pérdidas/backpressure visibles y reporte con dataset/periodo, PnL/ROI/WR bruto/neto, Sharpe y EV. Telemetría fuera de hot path y almacenamiento acotado. No inventar métricas financieras a partir de tests sintéticos. |

La implementación de cada átomo sigue hipótesis → ruta de datos → contrato real → CI candidato → CI push de main → recibo → retirada propia. Se mantienen pequeños commits y se reevalúa main cuando avance. El inventario se consume archivo por archivo en esa secuencia, sin marcar miles de archivos como revisados por un grep.

## Reservas y coordinación

Root: plan/ledger y análisis de FFI en crates/backtest-engine/src/lib.rs. omega220_review: revisión adversaria RO de contrato/ABI. git_audit: ramas/CI RO. resource_safety: mediciones RO. AGY mantiene parsers/MXCSR/shard; Sol su cobertura CI; GLM DMRG. El primary compartido no recibe checkout/pull, fuentes ni cambios de índice. No se eliminan archivos o procesos ajenos.

[Coordinación AGY](https://github.com/Jhona-la/Trader-Gemini/pull/103#issuecomment-6359000192): su shard ya ejecutó112 casos (51 de los seis crates que Sol añade y61 de telemetry-server). Resolver el solapamiento conservando cobertura; una suma mecánica duplicaría jobs. Este aviso no toma posesión de sus ramas.

## Hipótesis FFI y prueba propuesta

El consumidor C recibe métricas de una cuenta ficticia cuando falta capital; también puede alcanzar creación de slices con volumes nulo y aritmética de extensión desbordada. No hay llamadores internos de esos exports; se conserva ABI para consumidores externos y no se afirma impacto live medido. La API Rust canónica ya recibe capital explícito, y el host usa fetch_account_balance con reintento.

Probar ausencia, vacío, texto, cero, negativos, NaN/Inf; positivos de diferente escala exactos; rechazo antes de mutar outputs/registries, puntero volumes nulo y extensión desbordada sin dereferenciar; control positivo del export con datos válidos hasta el motor canónico. No inferir validez de una dirección arbitraria, longitud real, terminación CStr, aliasing o inmutabilidad externa del mmap: permanecen precondiciones unsafe. El cero legado no distingue error de cero trades; debe acompañarse de diagnóstico y el límite de ABI queda explícito.

## Deuda científica y operativa

Verificar en documentación primaria los contratos reales de Binance antes de STP/GTX/reduceOnly, batchOrders, modify, cuotas, SBE y firma; ninguna instrucción genérica se aplica indiscriminadamente a Hedge/One-Way. Investigar FTZ/FMA frente a la reproducción IEEE, y postselección de estados futuros frente a leakage. Criterios para promover modelos y pasar a Producción requieren evidencia OOS, costos observados y control de posiciones vivas; este átomo no ejecuta fondos ni promueve entorno.
