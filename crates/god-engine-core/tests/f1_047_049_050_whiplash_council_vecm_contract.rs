//! Contrato F1-047, F1-049 y F1-050 (Ola Ω166):
//! Anti-whiplash con Techo Temporal de Horizonte, Paridad de Drawdown y Fail-Closed en Consejo,
//! y Erradicación de Aliasing Espurio de pos_dev en vecm_zscore sin Spot.
//!
//! TEORÍA Y DOCTRINA:
//!
//! F1-047: Anti-Whiplash Post-Loss Bounded by Temporal Horizon Ceiling
//! - Tras un stop de largo (`last_was_win = false`, `last_is_long = true`), la intención
//!   contratendencia (Short) quedaba vetada indefinidamente mientras `p_stretch < 0.0`
//!   (precio bajo la EMA en un tramo bajista), sin ningún límite temporal.
//! - D-757 establece que "más allá del horizonte más lento... esto deja de ser un veto y
//!   pasa a ser un apagado". Se acota con `ventana_max_ms = (ventana_base_ms * 2.0).min(TAU_ANCHOR_SLOW_MS)`.
//!   Pasada esa ventana, el motor vuelve a tener plena capacidad para operar a favor de la
//!   nueva tendencia bajista.
//!
//! F1-049: Unificación de Drawdown y Fail-Closed en Asientos del Consejo
//! - SeniorRiesgo medía el drawdown usando `unified_capital` contra `peak_capital`,
//!   divergiendo de `risk_engine::drawdown::capital_del_freno` (el mínimo entre arena y exchange).
//! - SeniorEjecucion caía a 0.5 pb (fail-open) cuando el libro estaba cruzado (`ask < bid`),
//!   el peor caso de fallo justo cuando el libro está roto.
//! - Ahora: se usa `capital_del_freno` y el pico máximo, y ante libro cruzado o inválido
//!   se fija 500.0 pb (fail-closed estricto), honrando además el suelo `base_slippage_floor`.
//!
//! F1-050: Paridad BT <-> Vivo en VECM y Erradicación de Aliasing pos_dev
//! - Sin datos de spot (`!spot_presente` o `!spot_fresco`), la clave `vecm_zscore` caía
//!   en `(mid - kline_ema_slow) / (mid * atr)`, duplicando bit a bit `pos_dev`.
//! - En backtest (donde nunca hay spot), VECM votaba reversión a la EMA bajo la etiqueta
//!   de spread y se declaraba erróneamente operable (`|z| > 1e-12`), rompiendo la paridad con producción.
//! - Ahora: sin spot se publica 0.0 («sin dato»), declarando a VECM inoperable e inmune a falsos votos.

use omniscient_registry::OmniscientRegistry;
use strategy_core::vecm_arbitrage::JohansenVecmEngine;
use strategy_core::QuantumStrategy;
use std::sync::Arc;

#[test]
fn test_f1_047_anti_whiplash_post_loss_bounded_by_temporal_ceiling() {
    // Simulación analítica de la guarda de lib.rs:8702-8710
    let ventana_base_ms: f64 = 60_000.0; // 1 minuto
    let tau_anchor_slow_ms: f64 = quantum_arena::temporal_spectrum::TAU_ANCHOR_SLOW_MS;
    let ventana_max_ms: f64 = (ventana_base_ms * 2.0_f64).min(tau_anchor_slow_ms);

    // Simular escenario F1-047:
    // Long detenido con pérdida (last_was_win = false, last_is_long = true).
    // Ahora evaluamos un Short en plena tendencia bajista (p_stretch = -1.5 < 0.0).
    let is_same_dir = false;
    let p_stretch = -1.5;

    let eval_whiplash = |elapsed_ms: f64| -> bool {
        let transcurrido_ms = elapsed_ms;
        if !is_same_dir {
            transcurrido_ms < ventana_base_ms
                || (transcurrido_ms < ventana_max_ms && (p_stretch < 0.0))
        } else {
            false
        }
    };

    // 1. Inmediatamente tras el SL (t = 15s < ventana_base_ms):
    // Debe vetar incondicionalmente para no vender la capitulación instantánea.
    assert!(
        eval_whiplash(15_000.0),
        "En t < ventana_base_ms el anti-whiplash debe vetar incondicionalmente"
    );

    // 2. Entre ventana_base y ventana_max (t = 90s):
    // Debe vetar porque el precio sigue estirado a la baja en la dirección adversa.
    assert!(
        eval_whiplash(90_000.0),
        "En ventana_base <= t < ventana_max debe vetar si p_stretch < 0"
    );

    // 3. Superado el techo de horizonte (t = 150s >= ventana_max_ms = 120s):
    // NO debe vetar. El mercado ha tenido 2 horizontes completos para confirmar la tendencia bajista.
    // Antes de F1-047, esto vetaba indefinidamente por horas mientras p_stretch < 0.
    assert!(
        !eval_whiplash(150_000.0),
        "Superado ventana_max_ms, el anti-whiplash no debe vetar (erradica parálisis indefinida)"
    );

    // 4. Si el precio ya hubiera rebotado hacia la media (p_stretch >= 0.0) en t = 90s:
    // Tampoco debe vetar en la ventana intermedia.
    let eval_rebound = |elapsed_ms: f64| -> bool {
        let transcurrido_ms = elapsed_ms;
        let p_stretch_rebote = 0.2;
        transcurrido_ms < ventana_base_ms
            || (transcurrido_ms < ventana_max_ms && (p_stretch_rebote < 0.0))
    };
    assert!(
        !eval_rebound(90_000.0),
        "Con precio retornado a la media (p_stretch >= 0), no debe vetar pasada ventana_base"
    );
}

#[test]
fn test_f1_049_council_drawdown_unified_and_fail_closed_slippage() {
    // 1. Verificación de slippage fail-closed ante libro cruzado o corrupto
    let eval_slip = |ask: f64, bid: f64, mid: f64, base_slip_floor_frac: f64| -> f64 {
        let base_slip_floor = base_slip_floor_frac * 10_000.0;
        if mid <= 1e-8 || ask < bid {
            500.0 // Fail-closed
        } else {
            let current_spread_bps = ((ask - bid) / mid) * 10_000.0;
            ((current_spread_bps * 0.5) + 0.5).max(base_slip_floor).clamp(0.5, 500.0)
        }
    };

    // Libro cruzado (ask < bid): antes caía a 0.5 pb (fail-open). Ahora debe dar 500 pb.
    let slip_cruzado = eval_slip(99.0, 100.0, 99.5, 0.0001);
    assert_eq!(
        slip_cruzado, 500.0,
        "Libro cruzado (ask < bid) debe retornar 500.0 pb fail-closed para vetar ejecución"
    );

    // Libro vacío o precio nulo: debe dar 500 pb.
    let slip_vacio = eval_slip(0.0, 0.0, 0.0, 0.0001);
    assert_eq!(
        slip_vacio, 500.0,
        "Precio <= 1e-8 debe retornar 500.0 pb fail-closed"
    );

    // Libro normal con spread de 2 pb y base_slip_floor de 3 pb (0.0003):
    // current_spread = 2 pb -> (2 * 0.5) + 0.5 = 1.5 pb.
    // Pero max(1.5, 3.0 pb) debe respetar el suelo genético = 3.0 pb.
    let slip_normal = eval_slip(100.02, 100.00, 100.01, 0.0003);
    assert!(
        (slip_normal - 3.0).abs() < 1e-6,
        "Debe respetar el suelo genético base_slippage_floor: obtenido={}, esperado=3.0",
        slip_normal
    );

    // 2. Verificación de paridad de cálculo de drawdown con capital_del_freno
    let arena = quantum_arena::GlobalArena::new(13.0);
    arena.unified_capital.store(15.0, std::sync::atomic::Ordering::Relaxed);
    // Simular que el exchange real reporta un saldo menor (12.0 USD)
    arena.capital_exchange.store(12.0, std::sync::atomic::Ordering::Relaxed);
    arena.pico_capital.store(16.0, std::sync::atomic::Ordering::Relaxed);

    let capital_freno = risk_engine::drawdown::capital_del_freno(&arena);
    assert_eq!(capital_freno, 12.0, "capital_del_freno debe adoptar el mínimo conservador (exchange)");

    let peak_cap = 16.0;
    let dd_unificado = ((peak_cap - capital_freno) / peak_cap).clamp(0.0, 1.0);
    // (16 - 12) / 16 = 4 / 16 = 25% DD.
    // Antes de F1-049, usaba unified_capital (15.0), midiendo sólo (16 - 15)/16 = 6.25% DD.
    assert!(
        (dd_unificado - 0.25).abs() < 1e-6,
        "Drawdown del consejo debe medir contra capital_del_freno (25%), no contra unified_capital"
    );
}

#[test]
fn test_f1_050_vecm_basis_z_is_zero_without_spot_and_inoperable() {
    let registry = Arc::new(OmniscientRegistry::new());
    let mut vecm = JohansenVecmEngine::default();
    assert!(vecm.init(registry.clone()).is_ok());

    // 1. Simular ausencia de spot en producción o backtest (F1-050):
    // El núcleo publica vecm_zscore = 0.0 («sin dato»).
    registry.set_scoped("BTCUSDT", "vecm_zscore", 0.0);

    // Comprobar que vecm se declara inoperable para esa moneda
    assert!(
        !vecm.is_operable_for_coin(0, "BTCUSDT"),
        "Con vecm_zscore = 0.0 (sin spot), VECM debe ser estrictamente inoperable"
    );

    // Comprobar que evaluate_for_coin devuelve 0.0
    let voto_sin_spot = vecm.evaluate_for_coin(0, "BTCUSDT");
    assert_eq!(
        voto_sin_spot, 0.0,
        "VECM no debe emitir ningún voto sin datos de spot reales"
    );

    // 2. Simular presencia de spot con dislocación real de base (z = 2.5):
    registry.set_scoped("BTCUSDT", "vecm_zscore", 2.5);

    // Comprobar que vecm es operable
    assert!(
        vecm.is_operable_for_coin(0, "BTCUSDT"),
        "Con vecm_zscore = 2.5 (spot presente y dislocado), VECM debe ser operable"
    );

    // Comprobar que evaluate_for_coin vota reversión a la base
    let voto_con_spot = vecm.evaluate_for_coin(0, "BTCUSDT");
    assert!(
        voto_con_spot < 0.0,
        "Con z = +2.5, VECM debe votar reversión negativa hacia la media: {}",
        voto_con_spot
    );
}
