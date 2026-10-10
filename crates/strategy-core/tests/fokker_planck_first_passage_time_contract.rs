use strategy_core::vecm_arbitrage::{ContinuousOrnsteinUhlenbeckSde, JohansenVecmEngine};

#[test]
fn absorption_probability_boundary_conditions() {
    let sde = ContinuousOrnsteinUhlenbeckSde::new(0.2, 10.0, 0.5);
    let a = 8.0;
    let b = 12.0;

    // Condición en barrera inferior
    let p_at_a = sde.fokker_planck_absorption_probability(a, a, b);
    assert_eq!(p_at_a, 0.0, "P(hit b before a | x0 = a) debe ser 0.0");

    // Condición en barrera superior
    let p_at_b = sde.fokker_planck_absorption_probability(b, a, b);
    assert_eq!(p_at_b, 1.0, "P(hit b before a | x0 = b) debe ser 1.0");

    // Puntos fuera del intervalo
    assert_eq!(sde.fokker_planck_absorption_probability(7.0, a, b), 0.0);
    assert_eq!(sde.fokker_planck_absorption_probability(13.0, a, b), 1.0);

    // Simetría central alrededor de mu
    let p_mid = sde.fokker_planck_absorption_probability(10.0, a, b);
    assert!(
        (p_mid - 0.5).abs() < 1e-6,
        "En un intervalo simétrico alrededor de mu, la probabilidad en mu debe ser ~0.5 (GL10 error ~2.3e-8): {p_mid}"
    );
}

#[test]
fn absorption_probability_strict_monotonicity() {
    let sde = ContinuousOrnsteinUhlenbeckSde::new(0.3, 100.0, 2.0);
    let a = 90.0;
    let b = 110.0;

    let mut prev_p = -1.0;
    for step in 0..=20 {
        let x0 = a + (step as f64) * 1.0;
        let p = sde.fokker_planck_absorption_probability(x0, a, b);
        assert!(p.is_finite(), "P debe ser finita para x0 = {x0}");
        assert!(
            p >= prev_p,
            "P debe ser monótonamente no-decreciente: x0={x0}, p={p}, prev={prev_p}"
        );
        if step > 0 && step < 20 {
            assert!(
                p > prev_p,
                "P debe ser estrictamente creciente en el interior del intervalo: x0={x0}"
            );
        }
        prev_p = p;
    }
}

#[test]
fn brownian_motion_martingale_lever_rule_limit() {
    // Cuando theta = 0, el proceso es un paseo browniano puro sin deriva
    let sde_zero_theta = ContinuousOrnsteinUhlenbeckSde::new(0.0, 50.0, 1.0);
    let a = 10.0;
    let b = 30.0;

    for &x0 in &[12.0, 15.0, 20.0, 25.0, 28.0] {
        let p_ou = sde_zero_theta.fokker_planck_absorption_probability(x0, a, b);
        let p_lever = (x0 - a) / (b - a);
        assert!(
            (p_ou - p_lever).abs() < 1e-6,
            "Con theta=0, P debe coincidir con la regla de la palanca: x0={x0}, p_ou={p_ou}, p_lever={p_lever}"
        );
    }
}

#[test]
fn mean_reversion_amplifies_fokker_planck_win_probability() {
    let mu = 100.0;
    let sde = ContinuousOrnsteinUhlenbeckSde::new(0.4, mu, 1.5);

    // Escenario 1: Sobrecompra (x0 = 103.0 > mu). Abrimos Short.
    // Target Take Profit en 100.0 (hacia la media), Stop Loss adverso en 104.5.
    let x0_short = 103.0;
    let target_short = 100.0;
    let stop_short = 104.5;

    let p_win_short = sde.fokker_planck_win_probability(x0_short, false, target_short, stop_short);
    // En la regla de la palanca neutral (sin reversión a la media):
    // Distancia al target = 3.0, Distancia al stop = 1.5. Distancia total = 4.5.
    // P_neutral = (stop - x0) / (stop - target) = 1.5 / 4.5 ≈ 0.3333.
    let p_neutral_short = (stop_short - x0_short) / (stop_short - target_short);

    assert!(
        p_win_short > p_neutral_short,
        "La fuerza de atracción hacia la media debe elevar P_win del short: p_ou={p_win_short} > p_neutral={p_neutral_short}"
    );

    // Escenario 2: Sobreventa (x0 = 97.0 < mu). Abrimos Long.
    // Target Take Profit en 100.0 (hacia la media), Stop Loss adverso en 95.5.
    let x0_long = 97.0;
    let target_long = 100.0;
    let stop_long = 95.5;

    let p_win_long = sde.fokker_planck_win_probability(x0_long, true, target_long, stop_long);
    let p_neutral_long = (x0_long - stop_long) / (target_long - stop_long);

    assert!(
        p_win_long > p_neutral_long,
        "La fuerza de atracción hacia la media debe elevar P_win del long: p_ou={p_win_long} > p_neutral={p_neutral_long}"
    );
}

#[test]
fn expected_first_hitting_time_properties() {
    let mu = 50.0;
    let theta = 0.25;
    let sde = ContinuousOrnsteinUhlenbeckSde::new(theta, mu, 1.0);

    // Tiempo de x0 a x0 es 0
    assert_eq!(sde.expected_first_hitting_time_seconds(50.0, 50.0), 0.0);

    // Tiempo convergiendo hacia la media desde 55.0 a 50.0 debe ser finito y positivo
    let t_to_mu = sde.expected_first_hitting_time_seconds(55.0, 50.0);
    assert!(t_to_mu.is_finite() && t_to_mu > 0.0);
    // El orden de magnitud de la vida media es ln(2)/theta ≈ 2.77 s
    assert!(t_to_mu > 1.0 && t_to_mu < 30.0);

    // Tiempo escalando alejándose de la media (de 55.0 a 60.0) debe ser mayor
    let t_away = sde.expected_first_hitting_time_seconds(55.0, 60.0);
    assert!(
        t_away > t_to_mu,
        "Ir contra el pozo de potencial debe tomar más tiempo que converger: away={t_away} > to_mu={t_to_mu}"
    );

    // Con theta = 0 el tiempo esperado tiende a infinito
    let sde_flat = ContinuousOrnsteinUhlenbeckSde::new(0.0, mu, 1.0);
    assert!(sde_flat.expected_first_hitting_time_seconds(55.0, 50.0).is_infinite());
}

#[test]
fn stationary_and_transition_densities_gaussian_consistency() {
    let sde = ContinuousOrnsteinUhlenbeckSde::new(0.5, 10.0, 1.0);

    // Densidad estacionaria en la media es el máximo
    let p_mu = sde.stationary_density(10.0);
    let p_off = sde.stationary_density(12.0);
    assert!(p_mu > 0.0);
    assert!(p_off > 0.0);
    assert!(p_mu > p_off, "Densidad en la media debe ser maximal");

    // Densidad de transición para dt -> 0 se concentra en x0
    let p_trans_same = sde.transition_density(10.0, 10.0, 0.01);
    let p_trans_far = sde.transition_density(15.0, 10.0, 0.01);
    assert!(p_trans_same > p_trans_far * 100.0);

    // Densidad de transición para dt grande converge a la densidad estacionaria
    let p_trans_long_time = sde.transition_density(12.0, 10.0, 100.0);
    assert!(
        (p_trans_long_time - p_off).abs() < 1e-4,
        "Para dt grande, la densidad de transición debe converger a la estacionaria"
    );
}

#[test]
fn fail_closed_nan_inf_immunity() {
    let sde = ContinuousOrnsteinUhlenbeckSde::new(0.2, 10.0, 0.5);

    assert_eq!(
        sde.fokker_planck_absorption_probability(f64::NAN, 8.0, 12.0),
        0.5
    );
    assert_eq!(
        sde.fokker_planck_absorption_probability(10.0, f64::NAN, 12.0),
        0.5
    );
    assert_eq!(
        sde.fokker_planck_absorption_probability(10.0, 8.0, f64::INFINITY),
        0.5
    );

    assert_eq!(sde.stationary_density(f64::NAN), 0.0);
    assert_eq!(sde.transition_density(f64::NAN, 10.0, 1.0), 0.0);
    assert!(sde.expected_first_hitting_time_seconds(f64::NAN, 10.0).is_infinite());
}

#[test]
fn johansen_vecm_engine_fokker_planck_integration() {
    let mut vecm = JohansenVecmEngine::new(0.2, 1.0);

    // Alimentar precios con timestamps físicos
    let mut ts = 1_000_000u64;
    for i in 0..50 {
        ts += 1000;
        let p_a = 100.0 + (if i % 2 == 0 { 0.5 } else { -0.5 });
        let p_b = 100.0;
        let z = vecm.update_with_timestamp(p_a, p_b, ts);
        assert!(z.is_finite());
    }

    // Calcular probabilidad de absorción sobre el spread usando la API de Johansen
    let p_abs = vecm.fokker_planck_absorption_probability(0.02, -0.05, 0.05);
    assert!(p_abs.is_finite() && (0.0..=1.0).contains(&p_abs));

    // Tiempo esperado de reversión
    let t_rev = vecm.expected_mean_reversion_time_seconds(0.05);
    assert!(t_rev.is_finite() && t_rev > 0.0);
}
