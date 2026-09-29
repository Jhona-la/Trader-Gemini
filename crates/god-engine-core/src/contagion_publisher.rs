//! PUBLICADOR DE CONTAGIO HAWKES (Ola XLV·H).
//!
//! Conector final de la cadena Hawkes: lee los timestamps de ticks de las
//! monedas activas del arena, ejecuta `contagion_matrix` + `contagion_roles`
//! (feature-engine), y escribe cada moneda's `net_role` al registry como
//! `hawkes_contagion_net_role` — que el modulador XLV·G lee para descontar
//! la convicción de símbolos SEGUIDORES.
//!
//! Contrato:
//! - Variable: timestamps de los ticks recientes (ring) de cada moneda con
//!   masa espectral.
//! - Coste: O(N² · n_local) cada `PERIOD_TICKS` ticks — amortizado, no
//!   en el hot path.
//! - Contorno: <2 monedas con datos → no-op (matriz 1×1 no dice nada).

use quantum_arena::GlobalArena;
use std::sync::atomic::Ordering;

/// Publica los roles de contagio de todas las monedas activas al registry.
/// Llamar periódicamente (no en cada tick — es O(N²)).
pub fn publish_contagion_roles(arena: &GlobalArena) {
    // Recolectar series de timestamps de monedas con actividad espectral
    let mut series: Vec<Vec<u64>> = Vec::new();
    let mut coin_ids: Vec<usize> = Vec::new();
    let mut spans: Vec<u64> = Vec::new();

    for (id, coin) in arena.coins.iter().enumerate() {
        // Sólo monedas con energía espectral (campo caliente)
        let fisher = coin.spectral_fisher.load(Ordering::Relaxed);
        if fisher < 0.0 {
            continue; // frío, sin masa
        }
        let ticks = coin.tick_ring.snapshot_recent(256);
        if ticks.len() < 20 {
            continue; // sin suficientes eventos para el kernel
        }
        let ts: Vec<u64> = ticks.iter().map(|t| t.timestamp).collect();
        let span = ts.last().unwrap().saturating_sub(ts[0]);
        if span == 0 {
            continue;
        }
        series.push(ts);
        spans.push(span);
        coin_ids.push(id);
    }

    if series.len() < 2 {
        return; // matriz 1×1 no dice nada de estructura
    }

    // Rejilla de lags: 100ms a 30s (escala del scalp al swing corto)
    let lag_grid: Vec<u64> = vec![200, 500, 1_000, 5_000, 15_000, 30_000];

    let matrix = match feature_engine::hawkes_cross::contagion_matrix(
        &series, &spans, &lag_grid,
    ) {
        Some(m) => m,
        None => return,
    };

    let roles = match feature_engine::hawkes_cross::contagion_roles(&matrix) {
        Some(r) => r,
        None => return,
    };

    // Escribir cada moneda's net_role al registry
    for (idx, &coin_id) in coin_ids.iter().enumerate() {
        let net_role = roles[idx].net_role;
        if let Some(symbol) = quantum_arena::symbol_registry::try_symbol(coin_id) {
            arena.registry.set_for_coin(
                coin_id,
                "hawkes_contagion_net_role",
                net_role,
            );
            let _ = symbol; // symbol disponible para telemetría scoped si se necesita
        }
    }

    // XLVI·C (T05) — Hodge sobre el MISMO campo de contagio: fracción de la
    // energía del flujo que ninguna jerarquía líder→seguidor explica (cámara
    // de eco). Observable GLOBAL: estructura del universo, no de un activo.
    // No veta nada todavía — la constante de acoplamiento exige medición en
    // vivo (doctrina Fisher/D-754: publicar la evidencia antes de gobernar).
    if let Some(curl) = risk_engine::hodge::hodge_curl_share(&matrix) {
        arena
            .registry
            .set("hawkes_contagion_curl_share", curl);
    }
}
