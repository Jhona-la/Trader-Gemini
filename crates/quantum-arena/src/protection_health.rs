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
pub fn is_exchange_rejection(msg: &str) -> bool {
    let b = msg.as_bytes();
    if b.len() < 5 {
        return false;
    }
    for i in 0..=b.len() - 5 {
        if b[i] == b'-'
            && b[i + 1..i + 5].iter().all(u8::is_ascii_digit)
        {
            return true;
        }
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
