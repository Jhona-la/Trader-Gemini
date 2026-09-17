//! SALUD DEL FEED (X-010 / REHAB-4) — puente global lector⇒motor⇒inmune.
//!
//! PROBLEMA: el watchdog del WS vive en el hilo lector (sin acceso al arena);
//! la métrica de latencia vive en el arena. Ante MUERTE SILENCIOSA del feed
//! (5s sin datos), el watchdog reconectaba pero la latencia quedaba congelada
//! en el último latido sano — el interlock y el sistema inmune jamás se
//! enteraban (hallazgo X-010, verificado).
//!
//! PUENTE: un átomo global. El watchdog marca `stalled`; cualquier evento
//! vivo del loop unificado lo limpia. El core y el inmune leen
//! `is_stalled() || last_ws_latency_ms > umbral`.

use std::sync::atomic::{AtomicBool, Ordering};

static WATCHDOG_STALLED: AtomicBool = AtomicBool::new(false);

#[inline(always)]
pub fn stall() {
    WATCHDOG_STALLED.store(true, Ordering::Release);
}

#[inline(always)]
pub fn clear() {
    WATCHDOG_STALLED.store(false, Ordering::Release);
}

#[inline(always)]
pub fn is_stalled() -> bool {
    WATCHDOG_STALLED.load(Ordering::Acquire)
}
