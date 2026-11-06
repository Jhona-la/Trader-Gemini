//! 🧪 CONTRATOS FORMALES DE VERIFICACIÓN: MOTOR CONTINUO DE PORTFOLIO ESPECTRAL
//!
//! # FASE 31 (§31) — OLA OMEGA220: DE SILICIO A LA CÚSPIDE CUÁNTICA
//!
//! Suites de verificación determinista para `ContinuousSpectralPortfolioEngine`:
//! - C1: Normalización exacta en el símplex de Hilbert ($\sum w_i = 1.0 \pm 10^{-6}$).
//! - C2: Ausencia de cortes rígidos discretos: evalúa $N > 10$ activos simultáneamente.
//! - C3: Monitoreo del Índice Herfindahl-Hirschman (HHI) y número efectivo de apuestas $N_{\text{eff}}$.
//! - C4: Techo máximo de peso individual (`max_single_asset_weight`) respetado rigurosamente.
//! - C5: Conservación del capital total asignado en USD e inmunidad a valores no finitos.

use quantum_arena::continuous_spectral_portfolio::{
    ContinuousPortfolioParams, ContinuousSpectralPortfolioEngine,
};

#[test]
fn c1_simplex_normalization_and_partition_of_unity() {
    let engine = ContinuousSpectralPortfolioEngine::new();
    let n = 16;
    let liquidity = vec![1.5; n];
    let free_entropy = vec![0.5; n];
    let behavioral = vec![0.2; n];
    let capital = 1000.0;

    let report = engine.compute_allocations(&liquidity, &free_entropy, &behavioral, capital);

    assert_eq!(report.evaluated_count, n);
    let mut sum_w = 0.0;
    for i in 0..n {
        let w = report.allocations[i].continuous_weight;
        assert!(w >= 0.0, "Los pesos en el símplex deben ser no negativos");
        sum_w += w;
    }

    assert!(
        (sum_w - 1.0).abs() < 1e-6,
        "La suma de pesos debe cumplir la partición de la unidad (sum = {})",
        sum_w
    );
}

#[test]
fn c2_evaluation_beyond_rigid_ten_asset_cutoff() {
    let engine = ContinuousSpectralPortfolioEngine::new();
    // 24 activos simultáneos (superando ampliamente el límite heredado de 10)
    let n = 24;
    let mut liquidity = Vec::with_capacity(n);
    let mut free_entropy = Vec::with_capacity(n);
    let mut behavioral = Vec::with_capacity(n);

    for i in 0..n {
        liquidity.push(1.0 + (i as f64) * 0.1);
        free_entropy.push(0.1 * (i as f64));
        behavioral.push(if i % 2 == 0 { 0.3 } else { -0.2 });
    }

    let report = engine.compute_allocations(&liquidity, &free_entropy, &behavioral, 500.0);

    assert_eq!(report.evaluated_count, 24);
    assert!(
        report.active_assets_count > 10,
        "El motor debe admitir más de 10 activos si su masa espectral es viable, activos: {}",
        report.active_assets_count
    );

    // Los activos con mayores índices deben tener mayor peso
    let w_first = report.allocations[0].continuous_weight;
    let w_last = report.allocations[n - 1].continuous_weight;
    assert!(
        w_last > w_first,
        "Activo con mayor score espectral debe recibir mayor ponderación en el símplex ({} > {})",
        w_last,
        w_first
    );
}

#[test]
fn c3_herfindahl_index_and_effective_bets() {
    let engine = ContinuousSpectralPortfolioEngine::new();
    let n = 10;

    // Caso uniforme: 10 activos con scores idénticos
    let liq_uniform = vec![2.0; n];
    let ent_uniform = vec![0.0; n];
    let beh_uniform = vec![0.0; n];

    let report_uniform =
        engine.compute_allocations(&liq_uniform, &ent_uniform, &beh_uniform, 100.0);

    // En distribución uniforme, HHI = 1/N = 0.10, N_eff = 10
    assert!(
        (report_uniform.herfindahl_index - 0.10).abs() < 1e-3,
        "HHI uniforme debe ser ~0.10, obtenido: {}",
        report_uniform.herfindahl_index
    );
    assert!(
        (report_uniform.effective_bets_count - 10.0).abs() < 0.2,
        "N_eff uniforme debe ser ~10, obtenido: {}",
        report_uniform.effective_bets_count
    );

    // Caso concentrado: un activo domina masivamente
    let mut liq_concentrated = vec![0.1; n];
    liq_concentrated[0] = 50.0;

    let report_concentrated =
        engine.compute_allocations(&liq_concentrated, &ent_uniform, &beh_uniform, 100.0);

    assert!(
        report_concentrated.herfindahl_index > report_uniform.herfindahl_index,
        "HHI concentrado debe ser mayor que el uniforme"
    );
    assert!(
        report_concentrated.effective_bets_count < report_uniform.effective_bets_count,
        "N_eff concentrado debe ser menor que el uniforme"
    );
}

#[test]
fn c4_max_single_asset_cap_enforcement() {
    let engine = ContinuousSpectralPortfolioEngine::with_params(ContinuousPortfolioParams {
        temperature_beta: 5.0,
        min_weight_threshold: 0.01,
        max_single_asset_weight: 0.25, // Máximo 25% por activo
        free_entropy_weight: 0.30,
        behavioral_weight: 0.30,
    });

    let n = 8;
    let mut liquidity = vec![0.01; n];
    liquidity[0] = 100.0; // Dominancia extrema
    let free_entropy = vec![0.0; n];
    let behavioral = vec![0.0; n];

    let report = engine.compute_allocations(&liquidity, &free_entropy, &behavioral, 200.0);

    for i in 0..n {
        let w = report.allocations[i].continuous_weight;
        assert!(
            w <= 0.2501,
            "Ningún activo debe superar el cap máximo del 25%, activo {} tiene peso: {}",
            i,
            w
        );
    }
}

#[test]
fn c5_capital_conservation_and_nan_resilience() {
    let engine = ContinuousSpectralPortfolioEngine::new();
    let _n = 5;
    let liquidity = vec![1.0, 2.0, 3.0, 4.0, 5.0];
    let free_entropy = vec![0.1, 0.2, 0.3, 0.4, 0.5];
    let behavioral = vec![0.1, -0.1, 0.2, -0.2, 0.0];
    let total_capital = 777.77;

    let report = engine.compute_allocations(&liquidity, &free_entropy, &behavioral, total_capital);

    assert!(
        (report.total_allocated_capital_usd - total_capital).abs() < 1e-4,
        "El capital asignado debe conservar exactamente el capital total provisto ({} vs {})",
        report.total_allocated_capital_usd,
        total_capital
    );

    // Entrada corrupta con NaN/Inf
    let report_corrupt = engine.compute_allocations(
        &[f64::NAN, f64::INFINITY],
        &[-50.0, f64::NAN],
        &[f64::NAN, 1.0],
        -100.0,
    );

    assert_eq!(report_corrupt.evaluated_count, 0);
    assert_eq!(report_corrupt.total_allocated_capital_usd, 0.0);
}
