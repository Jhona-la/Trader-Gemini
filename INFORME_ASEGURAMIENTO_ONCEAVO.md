# 🔬 ONCEAVO INFORME DE ASEGURAMIENTO Y CERTIFICACIÓN PROFUNDA
### Verificación integral post-todos-los-fixes: dos bugs de conexión en el feedback de autoevolución

**Fecha:** 2026-09-08 · **Método:** 2 roles senior (Quant Researcher + ML/Infra Architect). Solo documentación.
**Se agrega a:** la serie forense completa (maestro §13-22, aseguramientos 1º-10º).

---

## 🗺️ 0. Diagnóstico central

**TODO lo acumulado está VERIFIED-CORRECT** — física simétrica, kelly completo, motor continuo, OCO retry que re-firma, fill-model del simulator, rutas centralizadas. El sistema es internamente coherente y arranca sin crash. **PERO el feedback de autoevolución — que "conectamos" en el commit anterior — está roto en los DOS extremos que importan**: nadie inicializa el bus (el productor escribe al vacío) y el consumidor lee el payload en la posición equivocada (aprende "los longs ganan" en vez de calibrar probabilidad). Es el mismo patrón de siempre, un nivel más adentro: órganos que existen y compilan, circuitos secos.

---

## 🚦 1. Resumen de resolución

| Frente | Estado |
|---|---|
| ✅ VERIFIED-CORRECT | Física entrada/salida/EV/fee-maker completa; kelly pipeline completo (metrics→normalizer→breaker→bootstrap unificado); motor continuo (1 posición, consenso 14/tick, temporal_scale con fenotipo completo); OCO retry re-firma; fill-model simulator; rutas quantum-arena centralizadas; Lagged inmortales; SegQueue resuelto; arranque demo-safe |
| 🔴 Crítico | E-01 (init_global_telemetry jamás llamada — productor no-op), E-02 (payload desalineado — daemon lee is_long como ml_prob) |
| 🔴 Alto | DriftAuditor desconectado, walk-forward fitness con fee fijo divergente del EV gate, ~39 rutas data/ restantes |
| ⬜ Medios/Bajos | activity_bonus, forensics cfg(test), REJECT_COUNTERS sin exposición en vivo, feature-engine parcial, lakehouse aislado |

---

## 📊 2. Matriz de CRÍTICOS/ALTOS

| ID | Hallazgo | Ubicación | Severidad |
|---|---|---|---|
| **E-01** | **`init_global_telemetry` NUNCA se llama**: el writer global OnceLock jamás se inicializa → `write_prediction_vs_reality` es no-op perpetuo → el daemon lee un mmap que nadie escribe → **el lazo de autoevolución sigue muerto** pese al "CONECTADO" del HEAD. Fix: UNA LÍNEA al arranque del god_engine | `mmap_bus.rs:38` (definida, cero callers); `god_engine.rs` (ausente) | CRÍTICO |
| **E-02** | **Payload desalineado productor/consumidor**: productor pone `ml_prob` en `payload[0]`; daemon lee `payload[1]` como ml_prob — que es `is_long` (0.0/1.0). El Shadow Forest aprende "los longs ganan" en vez de calibrar probabilidad ML | `mmap_bus.rs:56` vs `online_daemon.rs:112` | CRÍTICO |
| **E-03** | **DriftAuditor instanciado y descartado**: `_drift_auditor` sin ninguna llamada — el sistema opera sin detección de drift (exactamente el modo de fallo backtest→live) | `god_engine.rs:1090` | ALTO |
| **E-04** | **Walk-forward fitness con fricción fija 0.0008**: divergente del EV gate real (~0.0012+ con física) — sesga promociones hacia genomas sobre-apostados | `online_daemon.rs:358,435` | ALTO |
| **E-05** | **~39 rutas data/ restantes** fuera de paths.rs (telemetry-server, data-pipeline, symbol_manager, dashboard) — solo quantum-arena fue migrada | múltiple | MEDIO-ALTO |
| **E-06** | **EV gate slip aditivo vs física max()**: el gate usa `slip_floor + latency_slip` (aditivo) pero la física usa `max(impacto+latency, floor)` — el gate es más conservador (dirección segura) pero diverge ~floor bps | `risk-engine/lib.rs:552-564` vs `reality_physics.rs:70` | MEDIO |

---

## 📈 3. Los 10 literales más dañinos restantes

| # | file:line | Literal | Impacto |
|---|---|---|---|
| 1 | lib.rs:1219 (god) | Pesos fusión `0.40/0.35/0.25` | El corazón de la decisión con pesos no evolutivos |
| 2 | lib.rs:588-592 (risk) | Conf mínima `0.78/0.62` | Anclado al bootstrap de $13 |
| 3 | lib.rs:644,700 (risk) | Fee-impact `0.035` micro | 3.5% tolerado — letal compuesto |
| 4 | lib.rs:625 (risk) | Cushion `0.98` | Margen al 98% → una vela adversa liquida |
| 5 | online_daemon:358,435 | Fee fijo `0.0008` + capital `13.0` | Fitness divergente del EV gate |
| 6 | lib.rs:1620 (god) | Max pos `50000` + floor `5.0` | Cap hardcodeado |
| 7 | lib.rs:739-740 (god) | Timeouts `1.8M+12.6M·s` | Derivados a mano, no del gen |
| 8 | lib.rs:1339+ (god) | Cooldowns `45s/20s/180s/60s` | Frenan momentum legítimo |
| 9 | lib.rs:1084 (god) | ATR_1m `atr·7.746` | Asume tick exacto del feed |
| 10 | online_daemon:111 | Literales `12/30` duplicados | Sin const compartida con productor |

---

## 🎯 4. Hoja de ruta

1. **E-01 (UNA LÍNEA)**: `init_global_telemetry(&data_join("telemetry.mmap"))` al arranque del god_engine.
2. **E-02 (una línea)**: daemon leer `payload[0]` en vez de `payload[1]`, o productor mover ml_prob a `payload[1]`.
3. **E-03 DriftAuditor**: llamar `audit_execution()` en el loop de telemetría.
4. **E-04**: walk-forward usa la misma fórmula del EV gate para fricción.
5. **Certificación overnight** (el sistema está listo una vez E-01/E-02 cierren el circuito).

*Se agrega sin sustraer contenido. Referencias a main @ 0a7d67fa.*
