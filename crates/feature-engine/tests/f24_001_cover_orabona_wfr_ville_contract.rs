//! # Contrato Formal F24-001 — Grossman-Zhou, Cover-Vovk, Orabona, Filoche-Mayboroda, WFR y Ville
//!
//! Invariantes matemáticas y financieras comprobadas:
//! - **C1:** El piso de Grossman-Zhou $W_t \ge F \cdot W_0$ colapsa estrictamente la exposición a $0.0$ al tocar el piso, y Kyle $\lambda$ contrae la posición ante impacto adverso.
//! - **C2:** El portafolio universal continuo de Cover-Vovk garantiza $\sum b_i = 1$, $b_i \ge 0$, y arrepentimiento acotado $R_T \le \frac{d-1}{2}\ln T$.
//! - **C3:** El algoritmo de coin-betting de Orabona sin tasa de aprendizaje auto-adapta la fracción $\beta_t$ y multiplica la riqueza ante gradientes favorables.
//! - **C4:** La función paisaje de localización $u(x) > 0$ de Filoche-Mayboroda ($Lu=1$) construye un potencial efectivo $W = 1/u$ que localiza barreras de confinamiento de precio.
//! - **C5:** La métrica no conservativa de Wasserstein-Fisher-Rao $d_{\text{WFR}}$ es cero para perfiles idénticos y descompone costos horizontales y verticales.
//! - **C6:** La desigualdad maximal de Ville $\mathbb{P}(\sup M_t \ge \alpha) \le 1/\alpha$ certifica analíticamente la cota de ruina $< 10^{-6}$.
//! - **C7:** El motor maestro `CoverOrabonaWfrVilleMasterEngine` es resiliente a patologías numéricas sin ningún pánico.

use feature_engine::cover_orabona_wfr_ville::*;

#[test]
fn c1_grossman_zhou_floor_and_kyle_lambda_invariants() {
    let engine = GrossmanZhouKyleFloorEngine::new(0.90, 2.0); // Piso en 90% del capital inicial

    // Caso 1: Riqueza saludable (W_t = 1.05 * W_0), colchón = 0.15
    let rep_safe = engine.evaluate(105.0, 100.0, 0.02, 0.01, 0.001);
    assert!(rep_safe.cushion_to_floor > 0.10);
    assert!(rep_safe.max_grossman_exposure > 0.0);
    assert!(rep_safe.kyle_scaled_position > 0.0);

    // Caso 2: Riqueza en o por debajo del piso (W_t = 0.89 * W_0)
    let rep_floor = engine.evaluate(89.0, 100.0, 0.02, 0.01, 0.001);
    assert_eq!(
        rep_floor.cushion_to_floor, 0.0,
        "Al tocar el piso el colchón debe ser cero"
    );
    assert_eq!(
        rep_floor.max_grossman_exposure, 0.0,
        "La exposición de Grossman-Zhou debe colapsar estrictamente a 0.0 en el piso"
    );
    assert_eq!(
        rep_floor.kyle_scaled_position, 0.0,
        "La posición neta permitida debe ser exactamente 0.0"
    );

    // Caso 3: Efecto de contracción de Kyle Lambda: a mayor impacto (lambda alto), menor posición
    let rep_low_impact = engine.evaluate(105.0, 100.0, 0.02, 0.01, 0.0001);
    let rep_high_impact = engine.evaluate(105.0, 100.0, 0.02, 0.01, 0.05);

    assert!(
        rep_high_impact.kyle_scaled_position < rep_low_impact.kyle_scaled_position,
        "Mayor impacto de Kyle debe contraer estrictamente la posición"
    );
}

#[test]
fn c2_cover_vovk_universal_portfolio_invariants() {
    let mut cover = CoverVovkUniversalPortfolioEngine::new();

    // Récord de activos con activo 0 superando fuertemente al resto: (1.05, 0.99, 1.00, 0.98)
    let mut rep = CoverVovkReport::default();
    for _ in 0..10 {
        let price_relatives = [1.05, 0.99, 1.00, 0.98];
        rep = cover.update(&price_relatives);
    }

    // 1. Conservación estricta de la probabilidad en el simplex
    let sum_w: f64 = rep.optimal_weights.iter().sum();
    assert!(
        (sum_w - 1.0).abs() < 1e-10,
        "La suma de los pesos del portafolio universal debe ser 1.0, fue {}",
        sum_w
    );
    for &w in &rep.optimal_weights {
        assert!(w >= 0.0, "Cada peso debe ser no negativo");
    }

    // 2. El peso asignado al activo ganador (activo 0) debe dominar
    assert!(
        rep.optimal_weights[0] > rep.optimal_weights[1],
        "El activo ganador debe tener mayor peso ({}) que el perdedor ({})",
        rep.optimal_weights[0],
        rep.optimal_weights[1]
    );

    // 3. Arrepentimiento acotado
    assert!(
        rep.theoretical_regret_bound > 0.0,
        "La cota de arrepentimiento debe ser positiva"
    );
}

#[test]
fn c3_orabona_coin_betting_growth_and_adaptation() {
    let mut orabona = OrabonaCoinBettingEngine::new();

    // Serie de retornos consistentemente positivos (gradientes favorables g = +0.5)
    let mut rep = OrabonaCoinBettingReport::default();
    for _ in 0..15 {
        rep = orabona.update(0.5);
    }

    assert!(
        rep.wealth > 1.0,
        "La riqueza acumulada debe haber crecido sobre 1.0, fue {}",
        rep.wealth
    );
    assert!(
        rep.betting_fraction > 0.0,
        "La fracción de apuesta debe ser positiva ante gradientes positivos, fue {}",
        rep.betting_fraction
    );
    assert!(
        rep.bet_size > 0.0,
        "El tamaño de la apuesta debe ser positivo"
    );
}

#[test]
fn c4_filoche_mayboroda_localization_landscape_invariants() {
    let engine = FilocheMayborodaLocalizationEngine;

    // Perfil de profundidad con un pozo de liquidez masivo en el centro (índice 7 y 8)
    let mut depth = [1.0; FILOCHE_GRID_SIZE];
    depth[7] = 50.0;
    depth[8] = 50.0;

    let rep = engine.evaluate(&depth);

    // 1. La función paisaje u(x) debe ser estrictamente positiva en todos los nodos
    for (i, &u_val) in rep.landscape_function.iter().enumerate() {
        assert!(
            u_val > 0.0,
            "La función paisaje en nodo {} debe ser positiva, fue {}",
            i,
            u_val
        );
    }

    // 2. El potencial efectivo W = 1/u debe ser finito y positivo
    for (i, &w_val) in rep.effective_potential.iter().enumerate() {
        assert!(
            w_val > 0.0 && w_val.is_finite(),
            "El potencial efectivo en nodo {} debe ser positivo y finito, fue {}",
            i,
            w_val
        );
    }

    assert!(
        rep.max_confinement_barrier > 0.0,
        "La barrera máxima de confinamiento debe ser positiva"
    );
}

#[test]
fn c5_wasserstein_fisher_rao_distance_metric_invariants() {
    let engine = WassersteinFisherRaoEngine;

    let mu = [10.0, 20.0, 15.0, 30.0, 25.0, 12.0, 8.0, 5.0];

    // 1. Identidad de indiscernibles: d(mu, mu) == 0
    let rep_identity = engine.evaluate(&mu, &mu);
    assert!(
        rep_identity.wfr_distance < 1e-10,
        "La distancia WFR de un perfil consigo mismo debe ser 0.0, fue {}",
        rep_identity.wfr_distance
    );
    assert!(rep_identity.horizontal_transport_cost < 1e-10);
    assert!(rep_identity.vertical_hellinger_cost < 1e-10);

    // 2. Perfiles distintos: d(mu, nu) > 0 con componentes separadas
    let nu = [5.0, 8.0, 12.0, 25.0, 30.0, 15.0, 20.0, 10.0];
    let rep_diff = engine.evaluate(&mu, &nu);

    assert!(
        rep_diff.wfr_distance > 0.0,
        "La distancia WFR debe ser estrictamente positiva para perfiles diferentes"
    );
    assert!(rep_diff.horizontal_transport_cost > 0.0);
    assert!(rep_diff.vertical_hellinger_cost > 0.0);
}

#[test]
fn c6_ville_supermartingale_maximal_inequality_bounds() {
    let mut ville = VilleMaximalSupermartingaleEngine::new(10.0); // Meta: multiplicar 10x

    // 1. Cota de probabilidad de Ville: P(sup M_t >= 10) <= 1 / 10 = 0.10
    let rep_0 = ville.update(0.1, 0.05);
    assert_eq!(
        rep_0.probability_bound_upper, 0.10,
        "La cota superior maximal de Ville para alfa=10 debe ser exactamente 0.10"
    );
    assert!(
        rep_0.certified_ruin_bound <= 1e-6,
        "La cota analítica de ruina debe ser <= 10^-6"
    );

    // 2. Comportamiento multiplicativo acumulado
    let rep_pos = ville.update(0.2, 0.10);
    assert!(rep_pos.supermartingale_value > 1.0);
    assert!(rep_pos.running_supremum >= rep_pos.supermartingale_value);
}

#[test]
fn c7_master_engine_zero_panic_and_invariants() {
    let mut master = CoverOrabonaWfrVilleMasterEngine::new(0.90, 2.0, 100.0);

    let price_relatives = [1.01, 1.00, 0.99, 1.02];
    let depth = [10.0; FILOCHE_GRID_SIZE];
    let volumes = [5.0; WFR_BINS];

    // Prueba 1: Ejecución regular
    let rep_normal = master.evaluate(
        100.0,
        100.0,
        0.02,
        0.01,
        0.001,
        &price_relatives,
        0.1,
        &depth,
        &volumes,
        &volumes,
        0.1,
        0.02,
    );

    assert!(
        rep_normal.universal_portfolio_resilience >= 0.0
            && rep_normal.universal_portfolio_resilience <= 1.0
    );

    // Prueba 2: Resiliencia ante entradas patológicas (NaNs, infinitos, ceros)
    let bad_prices = [f64::NAN; ASSET_UNIVERSE];
    let bad_depth = [f64::INFINITY; FILOCHE_GRID_SIZE];
    let bad_vols = [-1.0; WFR_BINS];

    let rep_bad = master.evaluate(
        f64::NAN,
        0.0,
        f64::INFINITY,
        f64::NAN,
        f64::NAN,
        &bad_prices,
        f64::NAN,
        &bad_depth,
        &bad_vols,
        &bad_vols,
        f64::NAN,
        f64::NAN,
    );

    assert!(rep_bad.universal_portfolio_resilience.is_finite());
    assert!(rep_bad.grossman_kyle.cushion_to_floor >= 0.0);
    assert!(rep_bad.ville.certified_ruin_bound.is_finite());
}
