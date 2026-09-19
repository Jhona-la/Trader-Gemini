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

/// B3.5b — CLASIFICACIÓN DE ERRORES DE COLOCACIÓN: distingue un RECHAZO
/// del exchange (contiene un código de error Binance, p.ej. "-2019",
/// "-2021", "-4164") de un fallo de transporte ("error sending request",
/// timeouts). El escalado anti-desnudez sólo debe disparar con evidencia
/// POSITIVA de rechazo: un blip de red de 3 ciclos no cierra posiciones
/// sanas a mercado.
///
/// FIRMA DEL CÓDIGO (endurecida tras falsos positivos medidos): un código
/// Binance es un entero NEGATIVO de exactamente 4 dígitos sin cero a la
/// izquierda (rango documentado -1000..-9999), delimitado por un carácter
/// NO alfanumérico (`:`, `,`, `)`, espacio, comilla, fin…). Con eso quedan
/// fuera los patrones numéricos habituales de los textos de red:
///
/// ```text
/// "…45.123-0500"  offset de zona horaria  → cero inicial      ✗
/// "…123-05:00"    RFC 3339 con offset     → 2 dígitos         ✗
/// "2026-09-15"    fecha ISO               → 2 dígitos         ✗
/// "…-1757955…"    timestamp Unix          → 5º dígito es dígito ✗
/// "1200-4800ms"   rango con unidad        → unidad alfanum.   ✗
/// "-2019: …"      Display de Binance      → delimitador ':'   ✓
/// "code":-4164,   cuerpo JSON de Binance  → delimitador ','   ✓
/// ```
pub fn is_exchange_rejection(msg: &str) -> bool {
    let b = msg.as_bytes();
    if b.len() < 5 {
        return false;
    }
    for i in 0..=b.len() - 5 {
        if b[i] != b'-' || b[i + 1] == b'0' || !b[i + 1].is_ascii_digit() {
            continue;
        }
        // Exactamente 4 dígitos (1000..=9999): el tercero y el cuarto
        // deben ser dígitos y el quinto carácter NO puede serlo (un run
        // más largo es un timestamp o un rango, no un código).
        if !b[i + 2].is_ascii_digit() || !b[i + 3].is_ascii_digit() || !b[i + 4].is_ascii_digit() {
            continue;
        }
        if let Some(&next) = b.get(i + 5) {
            // Una letra inmediata ("−4800ms", "-4164x") delata una magnitud
            // con unidad, no un código de error del exchange.
            if next.is_ascii_alphanumeric() {
                continue;
            }
        }
        return true;
    }
    false
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
    }
    true
}

/// Rechazos acumulados del símbolo (para comparar entre auditorías).
pub fn rejections_of(symbol: &str) -> u64 {
    BRACKET_REJECTIONS
        .lock()
        .ok()
        .and_then(|m| m.get(symbol).copied())
        .unwrap_or(0)
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
            "err -2019",
            "trailing -4182.",
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
}
