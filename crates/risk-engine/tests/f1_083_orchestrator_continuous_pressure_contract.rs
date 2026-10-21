//! CONTRATO FORMAL DE MANIFOLD CONTINUO C1 DE PRESIÓN SISTÉMICA Y ERRADICACIÓN
//! DEL ACANTILADO DEL 76% EN EL ORQUESTADOR (OLA Ω187, FICHA F1-083, D-641 / P2)
//!
//! Este contrato formaliza y blinda:
//! 1. Continuidad C0 y C1 en la frontera lineal/Hermite P = 0.80 (sin escalón, derivada continua).
//! 2. Aterrizaje horizontal suave C1 en P = 0.90 con valor 0.0 y derivada 0.0.
//! 3. Monotonía decreciente estricta a lo largo de todo el dominio P in [0.0, 1.0].
//! 4. Erradicación de la orden suicida de ~$9.85 USD en cuentas micro de $13.00 USD ante p_crash = 0.89.
//! 5. Simetría direccional exacta entre largos (p_crash) y cortos (p_bull).
//! 6. Eliminación de código muerto (.max(0.05)) y alcance de cero genuino a P >= 0.90.

use quantum_arena::GlobalArena;
use risk_engine::{orchestrator::PortfolioOrchestrator, regime::MarketRegime};
use std::sync::atomic::Ordering::Relaxed;

fn arena_micro_13_usd() -> std::sync::Arc<GlobalArena> {
    let arena = GlobalArena::build_in_own_stack(13.0);
    arena.config.margin_cushion_pct.store(0.90, Relaxed);
    arena.config.global_max_drawdown.store(0.20, Relaxed);
    arena
}

fn arena_standard_100_usd() -> std::sync::Arc<GlobalArena> {
    let arena = GlobalArena::build_in_own_stack(100.0);
    arena.config.margin_cushion_pct.store(0.90, Relaxed);
    arena.config.global_max_drawdown.store(0.20, Relaxed);
    arena
}

#[test]
fn test_f1_083_c1_continuity_across_linear_hermite_boundary() {
    let cushion = 0.90;
    let p_coin = 0.0;

    // 1. Verificación de continuidad C0: no hay salto macroscópico en P = 0.80
    let e_left = PortfolioOrchestrator::continuous_exposure_limit(cushion, p_coin, 0.799999);
    let e_at = PortfolioOrchestrator::continuous_exposure_limit(cushion, p_coin, 0.80);
    let e_right = PortfolioOrchestrator::continuous_exposure_limit(cushion, p_coin, 0.800001);

    assert!(
        (e_left - e_at).abs() < 1e-5,
        "Discontinuidad izquierda en P=0.80: left={e_left}, at={e_at}"
    );
    assert!(
        (e_right - e_at).abs() < 1e-5,
        "Discontinuidad derecha en P=0.80: right={e_right}, at={e_at}"
    );

    // 2. Verificación de continuidad C1: la derivada numérica coincide con -0.25 a ambos lados
    let h = 1e-6;
    let slope_left = (PortfolioOrchestrator::continuous_exposure_limit(cushion, p_coin, 0.80)
        - PortfolioOrchestrator::continuous_exposure_limit(cushion, p_coin, 0.80 - h))
        / h;
    let slope_right = (PortfolioOrchestrator::continuous_exposure_limit(cushion, p_coin, 0.80 + h)
        - PortfolioOrchestrator::continuous_exposure_limit(cushion, p_coin, 0.80))
        / h;

    assert!(
        (slope_left - (-0.25)).abs() < 1e-3,
        "Pendiente izquierda en P=0.80 ({slope_left}) debe aproximar -0.25"
    );
    assert!(
        (slope_right - (-0.25)).abs() < 1e-3,
        "Pendiente derecha en P=0.80 ({slope_right}) debe aproximar -0.25"
    );
}

#[test]
fn test_f1_083_c1_horizontal_landing_at_veto_point() {
    let cushion = 0.90;
    let p_coin = 0.0;

    // En P = 0.90, el límite debe aterrizar con precisión en 0.0
    let e_veto = PortfolioOrchestrator::continuous_exposure_limit(cushion, p_coin, 0.90);
    assert_eq!(e_veto, 0.0, "En P=0.90 la exposición debe ser exactamente 0.0");

    // En P = 0.899, la exposición es suave y microscópica (~0.02% del capital),
    // no el 67.5% discontinuo del código anterior
    let e_pre_veto = PortfolioOrchestrator::continuous_exposure_limit(cushion, p_coin, 0.899);
    assert!(
        e_pre_veto < 0.001,
        "A P=0.899 la exposición ({e_pre_veto}) debe ser infinitesimal, erradicando el acantilado"
    );

    // Derivada horizontal en P=0.90: dE/dP -> 0.0
    let h = 1e-6;
    let slope_to_veto = (PortfolioOrchestrator::continuous_exposure_limit(cushion, p_coin, 0.90)
        - PortfolioOrchestrator::continuous_exposure_limit(cushion, p_coin, 0.90 - h))
        / h;
    assert!(
        slope_to_veto.abs() < 1e-3,
        "La derivada en aterrizaje P=0.90 ({slope_to_veto}) debe ser suave (tangente horizontal)"
    );
}

#[test]
fn test_f1_083_strict_monotonicity_across_entire_domain() {
    let cushion = 0.98; // Cuenta micro
    let p_coin = 0.0;

    let mut prev_e = PortfolioOrchestrator::continuous_exposure_limit(cushion, p_coin, 0.0);
    assert_eq!(prev_e, cushion);

    for i in 1..=1000 {
        let p = i as f64 / 1000.0;
        let e = PortfolioOrchestrator::continuous_exposure_limit(cushion, p_coin, p);

        assert!(
            e.is_finite(),
            "Exposición no finita en p={p}"
        );
        assert!(
            e >= 0.0,
            "Exposición negativa ({e}) en p={p}"
        );
        assert!(
            e <= prev_e + 1e-12,
            "Monotonía violada en p={p}: e={e} > prev_e={prev_e}"
        );
        prev_e = e;
    }
}

#[test]
fn test_f1_083_micro_account_eradication_of_9_85_usd_suicide_order() {
    let arena = arena_micro_13_usd();
    let orchestrator = PortfolioOrchestrator::new(&arena);

    // 1. En calma (p_crash = 0.0): margen admisible pleno de micro (~$12.74 USD)
    arena.regime_p_crash.store(0.0, Relaxed);
    let headroom_calma = orchestrator.available_margin_headroom(true, 5.0);
    assert!(
        (headroom_calma - (13.0 * 0.98)).abs() < 1e-6,
        "En calma, micro account de $13 tiene 98% de colchón (~$12.74): {headroom_calma}"
    );
    assert!(orchestrator.allow_trade(true, 1.0, MarketRegime::Range, 5.0));

    // 2. A p_crash = 0.85: el colchón se contrae continuamente a ~$5.02 USD
    arena.regime_p_crash.store(0.85, Relaxed);
    let headroom_85 = orchestrator.available_margin_headroom(true, 5.0);
    assert!(
        headroom_85 < 6.0 && headroom_85 > 4.5,
        "A p=0.85, el margen debe contraerse a ~$5.02 USD: {headroom_85}"
    );

    // 3. A p_crash = 0.88: el colchón se contrae a ~$1.04 USD
    arena.regime_p_crash.store(0.88, Relaxed);
    let headroom_88 = orchestrator.available_margin_headroom(true, 5.0);
    assert!(
        headroom_88 < 1.50 && headroom_88 > 0.80,
        "A p=0.88, el margen debe contraerse a ~$1.04 USD: {headroom_88}"
    );

    // 4. A p_crash = 0.89: el margen libre se reduce a ~$0.28 USD
    // En el código anterior autorizaba $9.85 USD (75.75% del capital) a p=0.89.
    // Con F1-083, una orden de Binance con nocional mínimo $5 a 5x lev (margen $1.00 USD)
    // es RECHAZADA SUAVEMENTE por falta de headroom sin esperar al escalón discreto de 0.90!
    arena.regime_p_crash.store(0.89, Relaxed);
    let headroom_89 = orchestrator.available_margin_headroom(true, 5.0);
    assert!(
        headroom_89 < 0.50,
        "A p=0.89, margen libre ({headroom_89}) debe ser menor a $0.50 USD, impidiendo trades grandes"
    );
    assert!(
        !orchestrator.allow_trade(true, 1.0, MarketRegime::Range, 5.0),
        "A p=0.89, una orden de $1.00 margen debe ser rechazada por headroom insuficiente, erradicando la orden suicida"
    );

    // 5. A p_crash = 0.90: margen libre es exactamente 0.00 USD
    arena.regime_p_crash.store(0.90, Relaxed);
    let headroom_90 = orchestrator.available_margin_headroom(true, 5.0);
    assert_eq!(
        headroom_90, 0.0,
        "A p=0.90, margen libre debe ser exactamente 0.00 USD"
    );
    assert!(!orchestrator.allow_trade(true, 0.01, MarketRegime::Range, 5.0));
}

#[test]
fn test_f1_083_directional_symmetry_long_vs_short() {
    let arena = arena_standard_100_usd();
    let orchestrator = PortfolioOrchestrator::new(&arena);

    for p in [0.0, 0.40, 0.60, 0.80, 0.85, 0.88, 0.90, 0.95] {
        // Long con p_crash = p
        arena.regime_p_crash.store(p, Relaxed);
        arena.regime_p_bull.store(0.0, Relaxed);
        let headroom_long = orchestrator.available_margin_headroom(true, 5.0);

        // Short con p_bull = p
        arena.regime_p_crash.store(0.0, Relaxed);
        arena.regime_p_bull.store(p, Relaxed);
        let headroom_short = orchestrator.available_margin_headroom(false, 5.0);

        assert!(
            (headroom_long - headroom_short).abs() < 1e-9,
            "Asimetría direccional a p={p}: long={headroom_long}, short={headroom_short}"
        );
    }
}

#[test]
fn test_f1_083_zero_dead_literals_headroom_genuine_zero() {
    let arena = arena_standard_100_usd();
    let orchestrator = PortfolioOrchestrator::new(&arena);

    // Para p_crash >= 0.90, el headroom debe llegar a 0.0 genuino sin piso fantasma de .max(0.05)
    for p in [0.90, 0.92, 0.95, 1.0, 1.5] {
        arena.regime_p_crash.store(p, Relaxed);
        let headroom = orchestrator.available_margin_headroom(true, 5.0);
        assert_eq!(
            headroom, 0.0,
            "A p={p}, el headroom debe ser 0.0 sin residuos de piso muerto: {headroom}"
        );
        assert!(
            !orchestrator.allow_trade(true, 0.001, MarketRegime::Range, 5.0),
            "A p={p}, ningún trade largo debe ser admisible"
        );
    }
}
