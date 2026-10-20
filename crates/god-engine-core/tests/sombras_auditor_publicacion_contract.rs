//! SOL (2026-10-19, censo C-F7b §3-2): PUBLICACIÓN DE LAS SOMBRAS DEL
//! AUDITOR EN EL REGISTRO (por moneda) y su conexión con el flujo real.
//!
//! Certifica que `GodEngineCore::publicar_sombras_auditor` — llamado por el
//! cierre de posición del host, justo tras la puerta de evidencia de
//! `internal_discrepancy` —
//! - no publica nada sin evidencia independiente suficiente (clave ausente,
//!   jamás cero inventado);
//! - publica `sombra_wr_disc` / `sombra_wr_disc_ee` / `sombra_wr_disc_n`
//!   con los valores exactos del contrato matemático de audit-engine;
//! - publica `sombra_friccion_real_menos_modelada_bp` consistente con la
//!   MISMA fricción modelo que usa el gate (`friccion_modelada_rt_bp`), no
//!   con una constante: el test lee el modelo del núcleo y verifica la
//!   relación, sin asumir el valor del config;
//! - aísla por moneda (la moneda sin canal no hereda la de al lado).

use audit_engine::auditor_interno::ExchangeFillRecord;
use god_engine_core::GodEngineCore;
use quantum_arena::GlobalArena;
use std::sync::atomic::Ordering;

/// Cierre del canal host: nocional 50 (50 000 × 0,001); fee 0,025 ⇒ 5 bp.
fn cierre(ts_ns: u64, pnl: f64) -> ExchangeFillRecord {
    ExchangeFillRecord {
        timestamp_ns: ts_ns,
        symbol_id: 0,
        is_buyer: false,
        price: 50_000.0,
        qty: 0.001,
        fee_usd: 0.025,
        realized_pnl: pnl,
        is_closed_position: true,
    }
}

#[test]
fn sin_evidencia_no_se_publica_ninguna_sombra() {
    let arena = GlobalArena::build_in_own_stack(13.0);
    let core = GodEngineCore::new(arena);
    core.publicar_sombras_auditor(0, 0.9);
    let nan = f64::NAN;
    for clave in [
        "sombra_wr_disc",
        "sombra_wr_disc_ee",
        "sombra_wr_disc_n",
        "sombra_friccion_real_menos_modelada_bp",
        "sombra_friccion_ee_bp",
    ] {
        assert!(
            core.arena
                .registry
                .get_for_coin_strict_or(0, clave, nan)
                .is_nan(),
            "{clave}: sin canal la clave debe estar ausente (payload NaN), no inventar 0"
        );
    }
}

#[test]
fn con_evidencia_la_sombra_wr_se_publica_exacta() {
    let arena = GlobalArena::build_in_own_stack(13.0);
    let mut core = GodEngineCore::new(arena);
    for i in 0..24u64 {
        let pnl = if i % 2 == 0 { 0.5 } else { -0.2 };
        core.registrar_fill_independiente_host(cierre(1_700_000_000_000_000_000 + i * 1_000_000_000, pnl));
    }
    core.publicar_sombras_auditor(0, 0.9);
    let nan = f64::NAN;
    let disc = core.arena.registry.get_for_coin_strict_or(0, "sombra_wr_disc", nan);
    let ee = core.arena.registry.get_for_coin_strict_or(0, "sombra_wr_disc_ee", nan);
    let n = core.arena.registry.get_for_coin_strict_or(0, "sombra_wr_disc_n", nan);
    assert!((disc - 0.4).abs() < 1e-12, "interno 0,9 vs real 0,5 ⇒ 0,4 exacto, obtuvo {disc}");
    assert!(ee.abs() < 1e-12, "alternados ⇒ EE 0, obtuvo {ee}");
    assert!((n - 24.0).abs() < 1e-12, "n = 24 cierres, obtuvo {n}");
}

#[test]
fn sombra_friccion_consistente_con_el_modelo_del_gate() {
    let arena = GlobalArena::build_in_own_stack(13.0);
    let mut core = GodEngineCore::new(arena);
    core.arena.coins[0].current_price.store(100.0, Ordering::Relaxed);
    core.arena.coins[0].current_atr.store(0.5, Ordering::Relaxed);

    // El modelo del gate con este config (taker + slip + ATR + latencia).
    let modelo_bp = core
        .friccion_modelada_rt_bp(0)
        .expect("con precio y ATR sanos el gate tiene fricción modelo");

    for i in 0..24u64 {
        core.registrar_fill_independiente_host(cierre(1_700_000_000_000_000_000 + i * 1_000_000_000, if i % 2 == 0 { 0.5 } else { -0.2 }));
    }
    core.publicar_sombras_auditor(0, 0.9);

    let nan = f64::NAN;
    let real_menos_modelada =
        core.arena.registry.get_for_coin_strict_or(0, "sombra_friccion_real_menos_modelada_bp", nan);
    // fee real 5 bp (0,025/50×10⁴) contra el modelo: la relación, no una constante.
    assert!(
        (real_menos_modelada - (5.0 - modelo_bp)).abs() < 1e-9,
        "5 bp reales − {modelo_bp} bp modelo = {}, obtuvo {real_menos_modelada}",
        5.0 - modelo_bp
    );
    let ee = core.arena.registry.get_for_coin_strict_or(0, "sombra_friccion_ee_bp", nan);
    assert!(ee.is_finite() && ee >= 0.0, "EE finito y no negativo, obtuvo {ee}");
}

#[test]
fn sombra_aislada_por_moneda() {
    let arena = GlobalArena::build_in_own_stack(13.0);
    let mut core = GodEngineCore::new(arena);
    for i in 0..24u64 {
        let pnl = if i % 2 == 0 { 0.5 } else { -0.2 };
        // Todos al coin 0 explícitamente.
        let mut f = cierre(1_700_000_000_000_000_000 + i * 1_000_000_000, pnl);
        f.symbol_id = 0;
        core.registrar_fill_independiente_host(f);
    }
    core.publicar_sombras_auditor(0, 0.9);
    let nan = f64::NAN;
    assert!(
        (core.arena.registry.get_for_coin_strict_or(0, "sombra_wr_disc", nan) - 0.4).abs() < 1e-12,
        "la moneda 0 publica su sombra"
    );
    assert!(
        core.arena
            .registry
            .get_for_coin_strict_or(1, "sombra_wr_disc", nan)
            .is_nan(),
        "la moneda 1 (sin canal) no hereda la sombra de la 0"
    );
}
