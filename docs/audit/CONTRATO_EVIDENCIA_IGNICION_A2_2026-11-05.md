# FASE A-A2 — CONTRATO DE EVIDENCIA DE IGNICIÓN (diseño ejecutable, hueco #1 de A1)

> PLAN MAESTRO DE EVIDENCIA, Fase A, paso A2. Cierra el BLOQUEANTE #1 de
> `IGNICION_A1_2026-11-05.md`: `darwin_approved` se auto-aprueba tras un
> entrenamiento NO-OP (GE:1716 + BOOT:287-294). Diseño GLM; implementación
> mecánica para AGY/QRA (dueños del host) o próxima sesión GLM.

## 1. Principio

La aprobación de Darwin deja de ser un timer y pasa a ser una FUNCIÓN de
evidencia medible producida por el arranque. Fuente única de verdad: un
struct nuevo en `god-engine-core` (crate neutro, sin tocar el bin).

## 2. Núcleo: `crates/god-engine-core/src/evidencia_ignicion.rs`

```rust
pub struct EvidenciaIgnicion {
    pub klines_warmup_por_simbolo: u32,   // BOOT cuenta lo que cargó de verdad
    pub venue_warmup: Venue,              // Testnet | Mainnet (del discriminador GE:1586)
    pub epochs_entrenamiento_efectivas: u32, // BOOT:288 hoy las ignora; contarlas
    pub metrica_entrenamiento: Option<f64>,  // loss/IC del modelo calentado; None = no-op
    pub oms_reconciled: bool,             // del snapshot conjunto (OMS:619-629), NO del papel
    pub es_cuenta_viva: bool,             // Testnet o Mainnet; false = papel local
}

pub enum VeredictoDarwin { Aprobado, Rechazado(&'static str) }

impl EvidenciaIgnicion {
    /// Política ÚNICA de aprobación — sustituye al timer de 120 ticks.
    pub fn aprueba_darwin(&self) -> VeredictoDarwin {
        // 1. Warmup suficiente: >= 500 klines/símbolo (hoy 1000 hardcode en
        //    BOOT:255 — medir y reportar; el umbral es config, no literal mágico).
        // 2. Entrenamiento REAL: epochs_efectivas >= 1 Y metrica.is_some().
        //    Si el training sigue siendo no-op => RECHAZADO("training_no_op")
        //    — el sistema NO se habilita a sí mismo.
        // 3. Cuenta viva (Testnet/Mainnet): oms_reconciled debe venir del
        //    snapshot conjunto (true sólo vía apply_snapshot).
        //    Papel local: oms_reconciled se EXIGE false aquí y el veredicto
        //    lo dice explícito — el papel nunca "reconcilia".
        // 4. Venue del warmup == venue del ejecutor (paridad; BOOT:28-35 ya
        //    lo garantiza — el contrato lo CERTIFICA, no lo supone).
    }
}
```

## 3. Puntos de integración (host — dueños AGY/QRA)

| Punto | Hoy | Con el contrato |
|---|---|---|
| GE:1716 | `darwin_approved.store(true)` incondicional tras no-op | `store(matches!(ev.aprueba_darwin(), Aprobado))` + log del veredicto con la razón |
| BOOT:287-294 | no-op devuelve `Some(())` | devolver métricas reales (aunque sea kline-count + epochs); el no-op pasa a ser RECHAZADO visible |
| ORCH:74-83 | admite tras 120 ticks + darwin | sin cambio: ya exige darwin; ahora darwin significa algo |
| OMS:199 | papel: ready=TRUE | sin cambio en vivo; el contrato lee oms_reconciled y exige es_cuenta_viva coherente |

## 4. Contrato de tests (mínimo, en `tests/evidencia_ignicion_contract.rs`)

1. `training_no_op_rechaza`: epochs=0, metrica=None => Rechazado("training_no_op") aunque todo lo demás esté perfecto. **Es el test que mata al BLOQUEANTE #1.**
2. `papel_nunca_reconcilia`: es_cuenta_viva=false, oms_reconciled=true => Rechazado (incoherencia detectada).
3. `cuenta_viva_reconciliada_con_entrenamiento_aprueba`: happy path Testnet.
4. `warmup_insuficiente_rechaza`: klines < umbral (config) => Rechazado con la razón.
5. Fall-closed: NaN en metrica => Rechazado; struct con campos imposibles => Rechazado, jamás panic.

## 5. Regla de compatibilidad con A3 (la corrida Demo)

Mientras el entrenamiento real no exista, el arranque en Testnet con el
contrato activo quedará en fase pre-Darwin — CORRECTO: la corrida A3
del plan debe ejecutarse con evidencia real, no con un timer. Si el
consejo quiere la corrida A3 ANTES de tener training real, la opción
honesta es una puerta explícita `TG_IGNICION_SIN_DARWIN=1` (auditable en
log, nunca silenciosa) — decisión del consejo, no un default.

## 6. Fuera de este átomo

Hueco #2 (EV neto post-fees como puerta DemoVerify→ProductionMainnet,
ORCH:95-103) y #3 (breaker Prod→Demo): requieren la métrica viva de A3;
su diseño sigue al de este contrato.
