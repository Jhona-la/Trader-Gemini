//! SALUD DE LA PROTECCIÓN (B1.1/B1.3) — puente global stream⇒watchdog.
//!
//! PROBLEMA: los brackets TP/SL son órdenes ALGO cuyo ciclo de vida es
//! invisible para el order_registry (no generan ORDER_TRADE_UPDATE hasta
//! dispararse). Si una pierna se cancela, expira o es rechazada por el
//! matching engine (ALGO_UPDATE: CANCELED/EXPIRED/REJECTED), la posición
//! queda DESNUDA y el motor no se entera hasta el siguiente arranque.
//!
//! PUENTE: mismo patrón que feed_health. El user-stream marca
//! `protection_dirty()` al recibir un ALGO_UPDATE terminal; el watchdog de
//! posición desnuda del motor lee la bandera y re-bracketea en su próximo
//! ciclo (encuesta rápida de 5s cuando está sucia, 60s en reposo). El
//! propio watchdog la limpia tras auditar contra el exchange.

use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};

static PROTECTION_DIRTY: AtomicBool = AtomicBool::new(false);
/// Contador de eventos ALGO_UPDATE terminales vistos (telemetría forense).
static TERMINAL_EVENTS_SEEN: AtomicU64 = AtomicU64::new(0);

#[inline(always)]
pub fn mark_dirty() {
    PROTECTION_DIRTY.store(true, Ordering::Release);
    TERMINAL_EVENTS_SEEN.fetch_add(1, Ordering::Relaxed);
}

#[inline(always)]
pub fn clear_dirty() {
    PROTECTION_DIRTY.store(false, Ordering::Release);
}

#[inline(always)]
pub fn is_dirty() -> bool {
    PROTECTION_DIRTY.load(Ordering::Acquire)
}

#[inline(always)]
pub fn terminal_events_seen() -> u64 {
    TERMINAL_EVENTS_SEEN.load(Ordering::Relaxed)
}

/// Compatibility boundary for bracket-placement errors, not a general log
/// scanner. Recognize the client's structured envelope or anchored legacy
/// displays only. Incidental numbers in transport text are not evidence.
/// Unknown execution (-1000/-1006/-1007) and AMBIGUOUS remain inconclusive.
/// A true result says the request was rejected, NOT that protection is
/// permanently impossible. Callers still need to reconcile current coverage.
/// Future work: carry a typed, endpoint-correlated outcome through the client
/// instead of flattening status/provenance into a string (FMT-213).
pub fn is_exchange_rejection(msg: &str) -> bool {
    let msg = msg.trim();
    if msg.starts_with("AMBIGUOUS") {
        return false;
    }
    #[derive(serde::Deserialize)]
    struct ErrorEnvelope {
        code: i64,
        #[serde(rename = "msg")]
        _message: String,
    }
    let json = msg.strip_prefix("Binance API Error: ").unwrap_or(msg);
    let code = if json.starts_with('{') {
        serde_json::from_str::<ErrorEnvelope>(json)
            .ok()
            .map(|e| e.code)
    } else if let Some(rest) = msg.strip_prefix("REJECTED code=") {
        anchored_code(rest, " msg=")
    } else if let Some(rest) = msg.strip_prefix("code=") {
        anchored_code(rest, ":")
    } else if let Some(rest) = msg.strip_prefix("reject: (") {
        anchored_code(rest, ")")
    } else {
        anchored_code(msg, ":")
    };
    matches!(code, Some(code) if (-9999..=-1000).contains(&code)
        && !matches!(code, -1000 | -1006 | -1007))
}

fn anchored_code(text: &str, delimiter: &str) -> Option<i64> {
    let code = text.get(..5)?;
    let bytes = code.as_bytes();
    if bytes[0] != b'-'
        || bytes[1] == b'0'
        || !bytes[1..].iter().all(u8::is_ascii_digit)
        || !text.get(5..)?.starts_with(delimiter)
    {
        return None;
    }
    code.parse().ok()
}

/// B3.5b — RECHAZOS DE BRACKET por símbolo (evidencia para el escalado).
static BRACKET_REJECTIONS: std::sync::LazyLock<
    std::sync::Mutex<std::collections::HashMap<String, u64>>,
> = std::sync::LazyLock::new(|| std::sync::Mutex::new(std::collections::HashMap::new()));

/// Anota un intento de colocación de bracket RECHAZADO por el exchange
/// (los fallos de transporte se ignoran). Devuelve true si anotó.
pub fn note_rejection(symbol: &str, err: &str) -> bool {
    if !is_exchange_rejection(err) {
        return false;
    }
    if let Ok(mut m) = BRACKET_REJECTIONS.lock() {
        *m.entry(symbol.to_string()).or_insert(0) += 1;
        return true;
    }
    false
}

/// Rechazos acumulados del símbolo (para comparar entre auditorías).
pub fn rejections_of(symbol: &str) -> u64 {
    BRACKET_REJECTIONS
        .lock()
        .ok()
        .and_then(|m| m.get(symbol).copied())
        .unwrap_or(0)
}

/// CL-44b (ADR-0015) — APLANADO Y AUDITORÍA NO SE SOLAPAN.
///
/// El vigilante corre también con el kill-switch armado (CL-44). Un aplanado
/// total cancela primero las piernas de cada símbolo y cierra después; esas
/// cancelaciones llegan por el stream como ALGO_UPDATE terminales y despiertan
/// al vigilante, que veía la posición aún abierta y le colocaba piernas
/// nuevas justo antes del cierre. En el apagado el proceso salía con ellas
/// vivas sobre una posición plana.
///
/// Protocolo de apretón de manos (Dekker con `SeqCst`): el vigilante anuncia
/// su auditoría y después mira si hay aplanado; el aplanado se anuncia y
/// después espera a que no quede auditoría en curso. O el vigilante ve el
/// aplanado y no audita, o el aplanado ve la auditoría y la espera.
static FLATTENS_ACTIVE: AtomicU64 = AtomicU64::new(0);
static AUDITS_ACTIVE: AtomicU64 = AtomicU64::new(0);

/// Aplanado en curso mientras viva; al soltarse el vigilante vuelve a auditar.
#[must_use = "el aplanado sólo está anunciado mientras vive la guarda"]
pub struct FlattenGuard(());

impl Drop for FlattenGuard {
    fn drop(&mut self) {
        FLATTENS_ACTIVE.fetch_sub(1, Ordering::SeqCst);
    }
}

/// Anuncia un aplanado total. Quien lo llama espera después a que
/// `audits_in_progress()` llegue a 0 antes de cancelar piernas.
pub fn begin_flatten() -> FlattenGuard {
    FLATTENS_ACTIVE.fetch_add(1, Ordering::SeqCst);
    FlattenGuard(())
}

#[inline(always)]
pub fn is_flattening() -> bool {
    FLATTENS_ACTIVE.load(Ordering::SeqCst) > 0
}

/// Auditoría del vigilante en curso mientras viva.
#[must_use = "la auditoría sólo está anunciada mientras vive la guarda"]
pub struct AuditGuard(());

impl Drop for AuditGuard {
    fn drop(&mut self) {
        AUDITS_ACTIVE.fetch_sub(1, Ordering::SeqCst);
    }
}

/// `None` mientras haya un aplanado en curso: el vigilante no audita (ni
/// coloca piernas) y deja la bandera sucia para auditar al terminar.
pub fn try_begin_audit() -> Option<AuditGuard> {
    AUDITS_ACTIVE.fetch_add(1, Ordering::SeqCst);
    if FLATTENS_ACTIVE.load(Ordering::SeqCst) > 0 {
        AUDITS_ACTIVE.fetch_sub(1, Ordering::SeqCst);
        return None;
    }
    Some(AuditGuard(()))
}

#[inline(always)]
pub fn audits_in_progress() -> u64 {
    AUDITS_ACTIVE.load(Ordering::SeqCst)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// B3.5b — las formas reales de rechazo del exchange SÍ clasifican.
    #[test]
    fn b3_5b_rechazos_reales_del_exchange() {
        for msg in [
            "-2019: Order is rejected.",
            "Binance API Error: {\"code\":-4164,\"msg\":\"Order's notional must be no smaller than 5.0\"}",
            "reject: (-1013) Filter failure: PRICE_FILTER",
            "code=-2021: Order would immediately trigger",
        ] {
            assert!(
                is_exchange_rejection(msg),
                "debe clasificar como rechazo del exchange: {msg}"
            );
        }
    }

    /// B3.5b — los textos de TRANSPORTE no clasifican aunque traigan guiones
    /// numéricos (timestamps con offset, rangos con unidades, errores del SO).
    #[test]
    fn b3_5b_fallos_de_transporte_no_cuentan() {
        for msg in [
            "error sending request for url (https://fapi.binance.com/fapi/v1/algoOrder?symbol=BTCUSDT)",
            "Network Error: tcp connect error: Connection refused (os error 10061)",
            "timeout at 2026-09-15T10:30:45.123-0500 retrying",
            "AMBIGUOUS: Network Error: handshake 1200-4800ms elapsed",
            "latency spike 5000-60000ms window",
            "seq mismatch -1757955600000 vs -1757955601000",
            "HTTP_429_RATE_LIMITED retry_after=60",
            "err -2019",
            "trailing -4182.",
            "",
            "-",
            "short",
        ] {
            assert!(
                !is_exchange_rejection(msg),
                "fallo de transporte mal clasificado como rechazo: {msg}"
            );
        }
    }

    /// B3.5b — el diario de evidencia sólo anota rechazos del exchange.
    #[test]
    fn b3_5b_note_rejection_filtra_transporte() {
        assert!(!note_rejection("TESTUSDT", "error sending request"));
        assert_eq!(rejections_of("TESTUSDT"), 0);
        assert!(note_rejection("TESTUSDT", "-2019: rejected"));
        assert_eq!(rejections_of("TESTUSDT"), 1);
    }

    /// CL-44b — el aplanado y la auditoría se excluyen: con un aplanado
    /// anunciado no empieza ninguna auditoría, y una auditoría anunciada
    /// es visible para el aplanado que llega después.
    #[test]
    fn cl44b_aplanado_y_auditoria_no_se_solapan() {
        assert!(!is_flattening());
        let auditoria = try_begin_audit().expect("sin aplanado se audita");
        assert_eq!(audits_in_progress(), 1);
        let aplanado = begin_flatten();
        assert!(is_flattening());
        assert_eq!(audits_in_progress(), 1, "el aplanado ve la auditoría en curso y la espera");
        drop(auditoria);
        assert_eq!(audits_in_progress(), 0);
        assert!(try_begin_audit().is_none(), "con aplanado en curso no empieza ninguna auditoría");
        assert_eq!(audits_in_progress(), 0, "la auditoría rechazada no queda anunciada");
        drop(aplanado);
        assert!(!is_flattening());
        assert!(try_begin_audit().is_some(), "terminado el aplanado se vuelve a auditar");
        assert_eq!(audits_in_progress(), 0);
    }
}
