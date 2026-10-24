//! LG-33b (Línea G, ola 10; hallazgo LG10-INFRA-02, confirmado por un
//! escéptico): la pata de drawdown del Auditor Interno comparaba dos magnitudes
//! distintas. La afirmación interna era la caída ACTUAL del capital respecto a la
//! base (0 en cuanto la cuenta está sobre la base) y la referencia independiente
//! es la caída MÁXIMA pico-valle del canal. Tras cualquier caída de más del 20 %
//! desde un pico, con la cuenta otra vez sobre la base, la discrepancia quedaba
//! fija por encima del umbral 0,20 y el consejo vetaba la moneda; el máximo sólo
//! sale del diario tras 4 096 fills nuevos y el veto impide abrir: absorbente.
//!
//! La afirmación interna pasa a ser la caída máxima pico-valle del diario
//! interno del auditor (los cierres tal como los contabilizó el núcleo): misma
//! definición a ambos lados.

use audit_engine::auditor_interno::{ExchangeFillRecord, MIN_FILLS_SOMBRA};
use god_engine_core::GodEngineCore;
use quantum_arena::GlobalArena;
use std::sync::atomic::Ordering;

fn cierre(i: usize, pnl: f64) -> ExchangeFillRecord {
    ExchangeFillRecord {
        timestamp_ns: 1_700_000_000_000_000_000 + i as u64 * 1_000_000_000,
        symbol_id: 0,
        is_buyer: true,
        price: 50_000.0,
        qty: 0.001,
        fee_usd: 0.0,
        realized_pnl: pnl,
        is_closed_position: true,
    }
}

/// Seis ganancias de +1,5 (13 → 22), una racha −2, −2, −1 (22 → 17: caída
/// pico-valle 5/22 ≈ 0,227) y once cierres pequeños alternados.
fn camino() -> Vec<f64> {
    let mut v = vec![1.5; 6];
    v.extend([-2.0, -2.0, -1.0]);
    for i in 0..11 {
        v.push(if i % 2 == 0 { 0.01 } else { -0.01 });
    }
    assert!(v.len() >= MIN_FILLS_SOMBRA);
    v
}

#[test]
fn lg33b_caida_pasada_con_la_cuenta_sobre_la_base_no_veta() {
    let arena = GlobalArena::build_in_own_stack(13.0);
    let mut core = GodEngineCore::new(arena);
    // El mismo camino en el diario interno (núcleo) y en el canal independiente.
    for (i, pnl) in camino().into_iter().enumerate() {
        core.auditor_interno.record_fill(cierre(i, pnl));
        core.registrar_fill_independiente_host(cierre(i, pnl));
    }
    // La cuenta terminó por encima de la base (13 + 7,0 = 20).
    core.arena.unified_capital.store(20.0, Ordering::Relaxed);
    let indep = core.auditor_interno.compute_independent_metrics_from_host();
    assert!(
        indep.calculated_max_drawdown > 0.20,
        "precondición: caída pico-valle del canal > 0,20 ({})",
        indep.calculated_max_drawdown
    );
    // Afirmación de win rate honesta: la del propio canal.
    core.publicar_discrepancia_auditor(0, indep.calculated_win_rate);
    let disc = core
        .arena
        .registry
        .get_for_coin_or(0, "internal_discrepancy", f64::NAN);
    assert!(disc.is_finite(), "con 20 cierres hay evidencia: se publica ({disc})");
    assert!(
        disc < 0.20,
        "mismo camino a ambos lados: sin veto (disc {disc}, dd independiente {})",
        indep.calculated_max_drawdown
    );
}

/// La pata sigue detectando una subestimación real: el canal independiente cae
/// más de lo que el núcleo contabilizó.
#[test]
fn lg33b_la_subestimacion_real_del_drawdown_sigue_vetando() {
    let arena = GlobalArena::build_in_own_stack(13.0);
    let mut core = GodEngineCore::new(arena);
    for (i, pnl) in camino().into_iter().enumerate() {
        // El núcleo no vio la racha perdedora: contabilizó cierres neutros.
        let interno = if (6..9).contains(&i) { 0.0 } else { pnl };
        core.auditor_interno.record_fill(cierre(i, interno));
        core.registrar_fill_independiente_host(cierre(i, pnl));
    }
    core.arena.unified_capital.store(20.0, Ordering::Relaxed);
    let indep = core.auditor_interno.compute_independent_metrics_from_host();
    core.publicar_discrepancia_auditor(0, indep.calculated_win_rate);
    let disc = core
        .arena
        .registry
        .get_for_coin_or(0, "internal_discrepancy", f64::NAN);
    assert!(disc > 0.20, "el núcleo afirma caer menos que la realidad: veta ({disc})");
}
