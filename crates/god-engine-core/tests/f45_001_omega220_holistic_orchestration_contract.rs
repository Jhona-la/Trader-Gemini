//! # Contrato Formal F45-001: Orquestación Holística Omega220 y Grafo Vivo Cuántico
//!
//! Valida la integración holística de las 10 capas físico-cuánticas en el grafo vivo (§45):
//! 1. `test_c1_holistic_living_graph_end_to_end_synthesis`: Síntesis end-to-end de los 10 motores físico-cuánticos.
//! 2. `test_c2_tri_slot_coordinated_dispatch_with_ville_ruin_bound`: Despacho coordinado tri-slot y cota de Ville.
//! 3. `test_c3_conformal_hard_veto_flattens_decisions_safely`: Aplanado seguro ante veto o evaporación de liquidez.
//! 4. `test_c4_zero_panic_rb_p1_and_stack_stability_under_extreme_dynamics`: Cero pánicos y estabilidad determinista.

use god_engine_core::GodEngineCore;
use quantum_arena::GlobalArena;
use strategy_core::AssetColorState;

#[test]
fn test_c1_holistic_living_graph_end_to_end_synthesis() {
    let arena = GlobalArena::build_in_own_stack(1_000.0);
    let mut core = GodEngineCore::new(arena);
    let coin_id = 0;

    let bid = 99.95;
    let ask = 100.05;
    let bid_qty = 100.0;
    let ask_qty = 40.0;
    let event_time_ms = 1_700_000_000_000;

    let triad = [
        AssetColorState {
            normalized_return: 0.015,
            order_flow_imbalance: 0.60,
            relative_weight: 0.50,
        },
        AssetColorState {
            normalized_return: -0.010,
            order_flow_imbalance: -0.40,
            relative_weight: 0.30,
        },
        AssetColorState {
            normalized_return: 0.002,
            order_flow_imbalance: 0.05,
            relative_weight: 0.20,
        },
    ];

    let capital_available_usd = 1_000.0;
    let max_leverage = 5.0;
    let risk_fraction = 0.02;

    // Calentamiento inicial del grafo vivo
    let _ = core.evaluate_omniscient_living_graph(
        coin_id,
        bid,
        ask,
        bid_qty,
        ask_qty,
        event_time_ms,
        &triad,
        capital_available_usd,
        max_leverage,
        risk_fraction,
    );

    // Evaluación con dinámica temporal sucesiva
    let report = core.evaluate_omniscient_living_graph(
        coin_id,
        bid,
        ask,
        bid_qty,
        ask_qty,
        event_time_ms + 100,
        &triad,
        capital_available_usd,
        max_leverage,
        risk_fraction,
    );

    // 1. Operador de Volatilidad Continua CWT (Fase 34)
    assert!(report.volatility_state.effective_volatility >= 0.0);
    assert!(report.volatility_state.continuous_hurst >= 0.0);

    // 2. Shockwave Riemann-Burgers (Fase 35)
    assert!(report.shockwave_report.shock_velocity_s.is_finite());
    assert!(report.shockwave_report.shock_layer_width > 0.0);

    // 3. Gauge SU(3) Cromodinámico Yang-Mills (Fase 37)
    let force_sum: f64 = report.su3_gauge_report.color_forces.iter().sum();
    assert!(force_sum.abs() < 1e-10);
    assert!(report.su3_gauge_report.yang_mills_action >= 0.0);

    // 4. Mean Field Games HJB+FPK (Fase 38)
    assert!((0.0..=1.0).contains(&report.mfg_report.crowd_herding_risk));

    // 5. Transporte Óptimo Monge-Kantorovich Wasserstein (Fase 40)
    assert!(report.wasserstein_report.w1_distance >= 0.0);
    assert!(report.wasserstein_report.transport_entropy >= 0.0);
    assert!((0.0..=1.0).contains(&report.wasserstein_report.liquidity_evaporation_risk));

    // 6. Entropía SRB Kolmogorov-Sinai y Lyapunov (Fase 41)
    assert!(report.srb_entropy_report.kolmogorov_sinai_entropy >= 0.0);
    assert!(report.srb_entropy_report.predictability_modulator > 0.0);

    // 7. Red Tensorial MPS y Optimización Variacional DMRG (Fase 42)
    assert!(report.mps_dmrg_report.entanglement_entropy >= 0.0);
    let sum_weights: f64 = report.mps_dmrg_report.asset_weights.iter().sum();
    assert!((sum_weights - 1.0).abs() < 1e-3);

    // 8. Hidrodinámica 2D Navier-Stokes Kraichnan (Fase 43)
    assert!((0.0..=1.0).contains(&report.navier_stokes_report.total_enstrophy));
    assert!(report.navier_stokes_report.kraichnan_spectral_slope >= 0.5);
    assert!((0.0..=1.0).contains(&report.navier_stokes_report.inverse_cascade_intensity));

    // 9. Geometría Espectral No Conmutativa de Connes (Fase 44)
    assert!(report.connes_spectral_report.connes_distance >= 0.0);
    assert!((-1.0..=1.0).contains(&report.connes_spectral_report.chiral_index));
    assert!(report.connes_spectral_report.spectral_action >= 0.0);

    // 10. Convicción Tensorial Holística y Escala de Slot
    assert!((0.0..=1.0).contains(&report.holistic_quantum_psi));
    assert!(
        report.holistic_slot_scale >= 0.10 && report.holistic_slot_scale <= 1.25,
        "Escala de slot fuera de rango: {}",
        report.holistic_slot_scale
    );
    assert!(report.ville_ruin_bound_satisfied);

    // 11. Colector Hiperbólico de Poincaré-Hodge (Fase 46)
    assert!(report.poincare_hodge_report.geodesic_distance_to_center >= 0.0);
    assert!(report
        .poincare_hodge_report
        .symplectic_area_vorticity
        .is_finite());

    // 12. Enrutamiento Cuántico por Integral de Caminos de Feynman (Fase 47)
    assert!(report.feynman_routing_report.optimal_slice_fraction > 0.0);
    assert!((0.0..=1.0).contains(&report.feynman_routing_report.constructive_coherence));

    // 13. Metamorfosis Topológica del Grafo Vivo Autónomo (Fase 48)
    assert!(report.metamorphosis_report.active_edge_count > 0);
    assert!(report.metamorphosis_report.spectral_radius_bound <= 1.0);

    // 14. Flujo Geodésico Cuántico y Cuantización BRST (Fase 49)
    assert!(report.geodesic_phase_space_report.hamiltonian_energy >= 0.0);
    assert!((0.0..=1.0).contains(&report.geodesic_phase_space_report.ghost_evaporation_ratio));
    assert!(report.geodesic_phase_space_report.slot_scaling_factor >= 0.10);

    // 15. Volatilidad Rugosa Fraccionaria de Riemann-Liouville (Fase 50)
    assert!(report.fractional_rough_vol_report.instantaneous_vol >= 0.0);
    assert!((0.02..=0.48).contains(&report.fractional_rough_vol_report.adaptive_hurst_estimate));
    assert!(
        report
            .fractional_rough_vol_report
            .signal_resonance_modulator
            >= 0.10
    );

    // 16. Gobernanza Conformal y Barrera FTAP de No-Arbitraje Cuántico (Fase 51)
    assert!(
        report
            .quantum_no_arbitrage_report
            .conformal_quantile_threshold
            > 0.0
    );
    assert!((0.0..=1.0).contains(&report.quantum_no_arbitrage_report.conformal_risk_multiplier));
    assert!(!report.quantum_no_arbitrage_report.hard_veto_induced);

    // 17. Homología Persistente Vietoris-Rips y TDA (Fase 52)
    assert!(report.vietoris_rips_report.mean_betti_0 >= 0.0);
    assert!(report.vietoris_rips_report.mean_betti_1 >= 0.0);
    assert!((0.0..=1.0).contains(&report.vietoris_rips_report.persistent_entropy));
    assert!(report.vietoris_rips_report.slot_scaling_factor >= 0.10);

    // 18. Flujo del Grupo de Renormalización de Wilson (Fase 53)
    assert!(report
        .wilson_rg_report
        .min_distance_to_fixed_point
        .is_finite());
    assert!(report.wilson_rg_report.signal_conviction_modulator >= 0.10);

    // 19. Cuantización Topológica de Chern-Simons y Fase de Berry (Fase 54)
    assert!(report.chern_simons_report.berry_phase.is_finite());
    assert!(!report.chern_simons_report.topological_veto_induced);
    assert!(report.chern_simons_report.slot_scaling_factor >= 0.10);
}

#[test]
fn test_c2_tri_slot_coordinated_dispatch_with_ville_ruin_bound() {
    let arena = GlobalArena::build_in_own_stack(500.0);
    let mut core = GodEngineCore::new(arena);
    let coin_id = 0;

    let bid = 50.00;
    let ask = 50.02; // Spread ajustado
    let bid_qty = 250.0;
    let ask_qty = 15.0; // Desequilibrio alcista
    let event_time_ms = 1_000_000;

    let triad = [
        AssetColorState {
            normalized_return: 0.03,
            order_flow_imbalance: 0.85,
            relative_weight: 0.60,
        },
        AssetColorState {
            normalized_return: -0.01,
            order_flow_imbalance: -0.50,
            relative_weight: 0.25,
        },
        AssetColorState {
            normalized_return: 0.00,
            order_flow_imbalance: 0.00,
            relative_weight: 0.15,
        },
    ];

    let capital_available_usd = 500.0;
    let max_leverage = 3.0;
    let risk_fraction = 0.03;

    let report = core.evaluate_omniscient_living_graph(
        coin_id,
        bid,
        ask,
        bid_qty,
        ask_qty,
        event_time_ms,
        &triad,
        capital_available_usd,
        max_leverage,
        risk_fraction,
    );

    if report.elastic_entry_granted {
        let total_notional: f64 = report
            .coordinated_emissions
            .slots
            .iter()
            .filter_map(|s| s.as_ref())
            .map(|p| p.price * p.quantity)
            .sum();
        assert!(
            total_notional <= capital_available_usd * max_leverage + 1.0,
            "Apalancamiento excedido: {} > {}",
            total_notional,
            capital_available_usd * max_leverage
        );
    }
    assert!(report.ville_ruin_bound_satisfied);
}

#[test]
fn test_c3_conformal_hard_veto_flattens_decisions_safely() {
    let arena = GlobalArena::build_in_own_stack(100.0);
    let mut core = GodEngineCore::new(arena);
    let coin_id = 0;

    // Spread abismal (1000 bps)
    let bid = 90.0;
    let ask = 100.0;
    let bid_qty = 1.0;
    let ask_qty = 1.0;
    let event_time_ms = 2_000_000;

    let neutral_triad = [
        AssetColorState::default(),
        AssetColorState::default(),
        AssetColorState::default(),
    ];

    let report = core.evaluate_omniscient_living_graph(
        coin_id,
        bid,
        ask,
        bid_qty,
        ask_qty,
        event_time_ms,
        &neutral_triad,
        100.0,
        1.0,
        0.01,
    );

    // Debe aplanarse completamente
    assert!(!report.elastic_entry_granted);
    for decision in &report.multi_slot_evaluation.decisions {
        assert_eq!(decision.signal, strategy_core::SignalType::Flat);
        assert_eq!(decision.net_confidence, 0.0);
        assert!(decision.vetoed);
    }
    for slot_order in &report.coordinated_emissions.slots {
        assert!(slot_order.is_none());
    }
}

#[test]
fn test_c4_zero_panic_rb_p1_and_stack_stability_under_extreme_dynamics() {
    let arena = GlobalArena::build_in_own_stack(2_000.0);
    let mut core = GodEngineCore::new(arena);
    let coin_id = 0;

    let triad = [
        AssetColorState {
            normalized_return: 0.01,
            order_flow_imbalance: 0.3,
            relative_weight: 0.4,
        },
        AssetColorState {
            normalized_return: -0.01,
            order_flow_imbalance: -0.2,
            relative_weight: 0.3,
        },
        AssetColorState {
            normalized_return: 0.0,
            order_flow_imbalance: 0.1,
            relative_weight: 0.3,
        },
    ];

    // 500 iteraciones deterministas con oscilación
    for step in 1..=500 {
        let ts = 10_000_000 + step * 50;
        let p_mid = 100.0 + ((step as f64) * 0.05).sin() * 5.0;
        let sp = 0.02 + ((step as f64) * 0.02).cos().abs() * 0.08;
        let bid = p_mid - sp * 0.5;
        let ask = p_mid + sp * 0.5;
        let b_qty = 50.0 + ((step as f64) * 0.03).sin() * 30.0;
        let a_qty = 50.0 + ((step as f64) * 0.03).cos() * 30.0;

        let rep = core.evaluate_omniscient_living_graph(
            coin_id, bid, ask, b_qty, a_qty, ts, &triad, 1_000.0, 3.0, 0.02,
        );

        assert!(rep.mid_price > 0.0);
        assert!(rep.holistic_quantum_psi.is_finite());
        assert!(rep.holistic_slot_scale.is_finite());
    }
}
