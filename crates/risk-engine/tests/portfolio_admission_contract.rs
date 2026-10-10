use quantum_arena::GlobalArena;
use risk_engine::{orchestrator::PortfolioOrchestrator, regime::MarketRegime};
use std::sync::atomic::Ordering;

#[test]
fn unknown_capital_cannot_authorize_margin() {
    let arena = GlobalArena::build_in_own_stack(100.0);
    for bad in [f64::NAN, f64::INFINITY, f64::NEG_INFINITY, 0.0, -1.0] {
        arena.unified_capital.store(bad, Ordering::Relaxed);
        assert!(
            !PortfolioOrchestrator::new(&arena).allow_trade(false, 1.0, MarketRegime::Range, 5.0),
            "capital={bad}"
        );
    }
}

#[test]
fn unknown_drawdown_budget_cannot_authorize_margin() {
    let arena = GlobalArena::build_in_own_stack(100.0);
    for bad in [f64::NAN, f64::INFINITY, f64::NEG_INFINITY, -0.1, 1.1] {
        arena
            .config
            .global_max_drawdown
            .store(bad, Ordering::Relaxed);
        assert!(
            !PortfolioOrchestrator::new(&arena).allow_trade(true, 1.0, MarketRegime::Range, 5.0),
            "budget={bad}"
        );
    }
}

#[test]
fn valid_margin_boundary_and_crash_policy_are_preserved() {
    let arena = GlobalArena::build_in_own_stack(100.0);
    arena
        .config
        .global_max_drawdown
        .store(0.1, Ordering::Relaxed);
    // D-744 (fusión PR #5): el colchón de margen ya no se deriva del gen de
    // drawdown — tiene gen propio. Se fija 0.90 para reproducir la frontera
    // que este contrato audita.
    arena
        .config
        .margin_cushion_pct
        .store(0.90, Ordering::Relaxed);
    let guard = PortfolioOrchestrator::new(&arena);
    assert!(guard.allow_trade(true, 90.0, MarketRegime::Range, 5.0));
    assert!(!guard.allow_trade(true, 90.01, MarketRegime::Range, 5.0));
    assert!(!guard.allow_trade(true, 1.0, MarketRegime::Crash, 5.0));
    assert!(guard.allow_trade(false, 1.0, MarketRegime::Crash, 5.0));
}

#[test]
fn invalid_open_position_margin_cannot_hide_exposure() {
    let arena = GlobalArena::build_in_own_stack(100.0);
    let position = arena.coins[0].positions.slots()[0];
    position.is_open.store(true, Ordering::Release);
    for side in [false, true] {
        position.is_long.store(side, Ordering::Relaxed);
        for bad in [f64::NAN, f64::NEG_INFINITY, -1.0] {
            position.margin_used.store(bad, Ordering::Relaxed);
            assert!(
                !PortfolioOrchestrator::new(&arena).allow_trade(true, 1.0, MarketRegime::Range, 5.0),
                "margin={bad},side={side}"
            );
        }
    }
}

#[test]
fn finite_margins_are_summed_across_spectral_slots_and_sides() {
    let arena = GlobalArena::build_in_own_stack(100.0);
    arena
        .config
        .global_max_drawdown
        .store(0.1, Ordering::Relaxed);
    // D-744 (fusión PR #5): el colchón de margen ya no se deriva del gen de
    // drawdown — tiene gen propio. Se fija 0.90 para reproducir la frontera
    // que este contrato audita.
    arena
        .config
        .margin_cushion_pct
        .store(0.90, Ordering::Relaxed);
    for (i, margin) in [40.0, 30.0].into_iter().enumerate() {
        let position = arena.coins[0].positions.slots()[i];
        position.is_long.store(i == 0, Ordering::Relaxed);
        position.margin_used.store(margin, Ordering::Relaxed);
        position.is_open.store(true, Ordering::Release);
    }
    let guard = PortfolioOrchestrator::new(&arena);
    assert!(guard.allow_trade(true, 20.0, MarketRegime::Range, 5.0));
    assert!(!guard.allow_trade(false, 20.01, MarketRegime::Range, 5.0));
}

/// Ola XLIV — la presión de crash sobre los largos sólo la ejerce una caída:
/// una moneda con crash-ness alta y marea ALCISTA (subida intensa) no puede
/// recortar el margen de los largos del resto de la cartera.
#[test]
fn crash_pressure_only_from_adverse_tide() {
    let arena = GlobalArena::build_in_own_stack(100.0);
    arena
        .config
        .global_max_drawdown
        .store(0.1, Ordering::Relaxed);
    arena
        .config
        .margin_cushion_pct
        .store(0.90, Ordering::Relaxed);
    let coin = &arena.coins[1];
    coin.spectral_crash_flux.store(0.60, Ordering::Relaxed);

    // Subida intensa (marea > 0): sin presión, la frontera de 90 se mantiene.
    coin.spectral_coherence.store(0.70, Ordering::Relaxed);
    let guard = PortfolioOrchestrator::new(&arena);
    assert!(guard.allow_trade(true, 90.0, MarketRegime::Range, 5.0));

    // Caída (marea < 0): el colchón pierde 0,25·0,60 = 15 pp ⇒ tope 75.
    coin.spectral_coherence.store(-0.70, Ordering::Relaxed);
    assert!(!guard.allow_trade(true, 90.0, MarketRegime::Range, 5.0));
    assert!(guard.allow_trade(true, 74.0, MarketRegime::Range, 5.0));
    assert!(!guard.allow_trade(true, 76.0, MarketRegime::Range, 5.0));
    // Los cortos no pagan presión de crash.
    assert!(guard.allow_trade(false, 89.0, MarketRegime::Range, 5.0));
}

/// AGY-AUD-P31: El símplex continuo de régimen de mercado contrae suavemente el margen
/// admisible y ejerce veto de crash sistémico sin saltos discretos ni colisiones.
#[test]
fn continuous_regime_simplex_contracts_margin_smoothly() {
    let arena = GlobalArena::build_in_own_stack(100.0);
    arena
        .config
        .global_max_drawdown
        .store(0.1, Ordering::Relaxed);
    arena
        .config
        .margin_cushion_pct
        .store(0.90, Ordering::Relaxed);
    let guard = PortfolioOrchestrator::new(&arena);

    // Baseline: con simplex en estado por defecto (p_range=1.0, resto 0), tope es 90.0
    assert!(guard.allow_trade(true, 90.0, MarketRegime::Range, 5.0));
    assert!(!guard.allow_trade(true, 90.01, MarketRegime::Range, 5.0));

    // 1. Presión continua de Crash: p_crash = 0.40 contrae el colchón por 0.25 * 0.40 = 0.10 (10 pp)
    // El tope baja de 90 a 80.
    arena.regime_p_crash.store(0.40, Ordering::Relaxed);
    assert!(guard.allow_trade(true, 80.0, MarketRegime::Range, 5.0));
    assert!(!guard.allow_trade(true, 80.01, MarketRegime::Range, 5.0));
    // G0-1: Verificación de fusión suave — incluso si el enum MAP discreto es Crash (p_crash fue argmax),
    // el sistema no colapsa a X=0 si p_crash < 0.90, sino que modula continuamente con el tope de 80.0.
    assert!(guard.allow_trade(true, 80.0, MarketRegime::Crash, 5.0));
    assert!(!guard.allow_trade(true, 80.01, MarketRegime::Crash, 5.0));

    // Los cortos no son penalizados por p_crash (siguen en 90.0)
    assert!(guard.allow_trade(false, 90.0, MarketRegime::Range, 5.0));

    // 2. Presión continua de Squeeze para cortos: p_bull = 0.60 contrae el colchón de cortos por 0.25 * 0.60 = 0.15 (15 pp)
    // El tope para cortos baja de 90 a 75.
    arena.regime_p_bull.store(0.60, Ordering::Relaxed);
    assert!(guard.allow_trade(false, 75.0, MarketRegime::Range, 5.0));
    assert!(!guard.allow_trade(false, 75.01, MarketRegime::Range, 5.0));

    // 3. Veto de Crash sistémico continuo: si p_crash >= 0.90, las compras quedan absolutamente vetadas
    // incluso si el enum discreto está en Range o Crash.
    arena.regime_p_crash.store(0.92, Ordering::Relaxed);
    assert!(!guard.allow_trade(true, 1.0, MarketRegime::Range, 5.0));
    assert!(!guard.allow_trade(true, 1.0, MarketRegime::Crash, 5.0));
    // Los cortos permanecen permitidos (pueden surfear la caída con margen acotado por p_bull)
    assert!(guard.allow_trade(false, 75.0, MarketRegime::Range, 5.0));
}

