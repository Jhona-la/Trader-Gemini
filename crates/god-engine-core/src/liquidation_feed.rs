//! P-4 (PREDICTORES / MOTOR UNIVERSAL) — FEED DE LIQUIDACIONES.
//!
//! QUÉ: canal global de severidad de liquidaciones. Los eventos
//! `!forceOrder` del WS (y cualquier otro productor de cascadas, p.ej. el
//! sniffer de Hyperliquid) depositan aquí su severidad normalizada; el
//! camino per-tick del core la consume UNA vez (swap) y la inyecta como
//! `dex_severity` a `update_macro_features` — el `ExponentialDecayTensor`
//! (dark_alpha) la decae con su constante genómica.
//!
//! POR QUÉ: `dex_severity` llevaba 0.0 constante desde el origen (sin
//! productor): la dimensión [9] del vector universal está zerificada a
//! ambos lados por paridad (el trainer no tiene historia de
//! liquidaciones), pero el TENSOR 54D de la NN, el registry y el futuro
//! asiento "Ente del Mercado" del consejo SÍ consumen la señal viva.
//!
//! SEMÁNTICA EVENT-DRIVEN: `bump` deposita max(pendiente, sev) — nunca
//! acumula niveles; el consumo es swap(0): cada evento alimenta un solo
//! apply_event del tensor de decaimiento. Severidad normalizada [0,1]:
//! |notional liquidado| / umbral de cascada (≈$1M), log-escalada.
use quantum_arena::atomic_float::AtomicF64;
use std::sync::atomic::Ordering;

static PENDING_SEVERITY: AtomicF64 = AtomicF64::new(0.0);

/// Deposita la severidad de un evento de liquidación (clamp [0,1]).
/// Llamado por el parser de `!forceOrder` del WS y por los sniffers DEX.
/// CAS-loop por si dos hilos depositan a la vez: gana el mayor.
pub fn bump(severity: f64) {
    if severity.is_finite() && severity > 0.0 {
        let sev = severity.min(1.0);
        let _ = PENDING_SEVERITY.fetch_update(
            Ordering::Relaxed,
            Ordering::Relaxed,
            |cur| Some(if cur >= sev { cur } else { sev }),
        );
    }
}

/// Consume la severidad pendiente (una sola lectura por evento).
#[inline]
pub fn take_pending() -> f64 {
    PENDING_SEVERITY.swap(0.0, Ordering::Relaxed)
}

/// Lee la severidad pendiente SIN consumirla — para observadores
/// (payload del consejo, telemetría) que no deben robarle el evento al
/// camino per-tick.
#[inline]
pub fn peek_pending() -> f64 {
    PENDING_SEVERITY.load(Ordering::Relaxed)
}

/// Normaliza un notional liquidado (USD) a severidad [0,1]:
/// log-escala contra umbral de cascada de $1M — $10k≈0.33, $100k≈0.67,
/// $1M=1.0 (cap).
pub fn severity_from_notional(notional_usd: f64) -> f64 {
    if !notional_usd.is_finite() || notional_usd <= 0.0 {
        return 0.0;
    }
    (notional_usd.ln() / 1_000_000f64.ln()).clamp(0.0, 1.0)
}
