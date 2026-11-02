//! # F12-002: Contratos Formales de Haces Celulares de Grothendieck, Cohomología $H^1$ & Atiyah-Singer
//!
//! Certifica los 6 invariantes algebraicos, espectrales y topológicos de `grothendieck_sheaves`:
//! - **C1:** Sección global armónica exacta ($x \in H^0(\mathcal{G}; \mathcal{F}) = \ker \delta_0 \implies \mathcal{E}_{\text{sheaf}}(x) = 0$).
//! - **C2:** Identidad exacta de Dirichlet ($x^T \Delta_{\mathcal{F}} x \equiv \|\delta_0 x\|_2^2$), gradiente restaurador de arbitraje y contracción por difusión de calor.
//! - **C3:** Detección de obstrucción cohomológica $H^1$ y `TopologicalHolonomyTrap` bajo holonomía cíclica frustrada ($\rho_{01}\rho_{12}\rho_{20} \ne 1$).
//! - **C4:** Invariancia topológica de Euler-Poincaré / McKean-Singer: $\text{Ind}(D) = \dim H^0 - \dim H^1 \equiv N_V - N_E$.
//! - **C5:** Antisimetría quiral de modos cero y acotamiento estricto de `topological_chirality_bias` en $[-1, 1]$.
//! - **C6:** Robustez total (cero pánicos, salidas 100% finitas) ante entradas corruptas (`NaN`, `±Inf`, lazos propios, pesos negativos o vacíos).

use feature_engine::{CellularSheafEngine, SheafRestrictionEdge, SheafTopologicalRegime};

#[test]
fn test_c1_exact_global_section_zero_frustration_energy() {
    let engine = CellularSheafEngine::default();
    // Triángulo consistente:
    // e01: 1.0 * x0 = 2.0 * x1
    // e12: 1.0 * x1 = 3.0 * x2
    // e20: 6.0 * x2 = 1.0 * x0
    // Solución armónica exacta: x = [6.0, 3.0, 1.0]
    let edges = [
        SheafRestrictionEdge::new(0, 1, 1.0, 2.0, 1.0),
        SheafRestrictionEdge::new(1, 2, 1.0, 3.0, 1.0),
        SheafRestrictionEdge::new(2, 0, 6.0, 1.0, 1.0),
    ];
    let stalks = [6.0, 3.0, 1.0];

    let report = engine.analyze_sheaf(&stalks, &edges);
    assert_eq!(report.n_vertices, 3);
    assert_eq!(report.n_edges, 3);
    assert!(report.frustration_energy < 1e-12);
    assert!(report.cycle_holonomy_obstruction < 1e-12);
    assert_eq!(report.regime, SheafTopologicalRegime::HarmonicGlobalSection);
    assert!(report.atiyah_singer.dim_h0_global_sections >= 1);

    for &force in report.harmonic_restoration_force.iter().take(3) {
        assert!(force.abs() < 1e-12);
    }
}

#[test]
fn test_c2_cohomological_arbitrage_dislocation_and_restoration_gradient() {
    let engine = CellularSheafEngine::default();
    let edges = [
        SheafRestrictionEdge::new(0, 1, 1.0, 1.0, 1.0),
        SheafRestrictionEdge::new(1, 2, 1.0, 1.0, 1.0),
        SheafRestrictionEdge::new(2, 0, 1.0, 1.0, 1.0),
    ];
    // Dislocación respecto a la sección constante: activo 0 sobrevalorado (+5.0), activo 2 subvalorado (-2.0)
    let stalks = [5.0, 1.5, -2.0];

    let ws = CellularSheafEngine::build_sheaf_laplacian_and_coboundary(3, &stalks, &edges);
    let mut quadratic_form = 0.0_f64;
    for (u, &xu) in stalks.iter().enumerate() {
        for (v, &xv) in stalks.iter().enumerate() {
            quadratic_form += xu * ws.laplacian[u][v] * xv;
        }
    }

    let report = engine.analyze_sheaf(&stalks, &edges);
    // Identidad exacta de Dirichlet: x^T \Delta_F x == ||\delta_0 x||_2^2
    assert!((quadratic_form - report.frustration_energy).abs() < 1e-10);
    assert_eq!(
        report.regime,
        SheafTopologicalRegime::CohomologicalArbitrageDislocation
    );
    assert_eq!(report.most_overvalued_vertex, 0);
    assert_eq!(report.most_undervalued_vertex, 2);
    assert!(report.harmonic_restoration_force[0] < 0.0);
    assert!(report.harmonic_restoration_force[2] > 0.0);

    // La sección difundida por flujo de calor debe reducir estrictamente la energía de frustración
    let post_diffusion = engine.analyze_sheaf(&report.diffused_harmonic_section[..3], &edges);
    assert!(post_diffusion.frustration_energy < report.frustration_energy);
}

#[test]
fn test_c3_topological_holonomy_trap_frustrated_cycle() {
    let engine = CellularSheafEngine::default();
    // Ciclo frustrado donde el producto de holonomía es 2.0 != 1.0 (ln(2) ≈ 0.693147)
    let edges = [
        SheafRestrictionEdge::new(0, 1, 2.0, 1.0, 1.0),
        SheafRestrictionEdge::new(1, 2, 1.0, 1.0, 1.0),
        SheafRestrictionEdge::new(2, 0, 1.0, 1.0, 1.0),
    ];
    let stalks = [1.0, 2.0, 2.0];

    let report = engine.analyze_sheaf(&stalks, &edges);
    let expected_holonomy = 2.0_f64.ln();
    assert!((report.cycle_holonomy_obstruction - expected_holonomy).abs() < 1e-9);
    assert_eq!(report.atiyah_singer.dim_h0_global_sections, 0);
    assert!(report.spectral_frustration_gap > 0.05);
    assert_eq!(
        report.regime,
        SheafTopologicalRegime::TopologicalHolonomyTrap
    );
}

#[test]
fn test_c4_atiyah_singer_euler_poincare_index_invariance() {
    let engine = CellularSheafEngine::default();
    // Grafo K4 completo (4 vértices, 6 aristas): Ind(D) = dim H^0 - dim H^1 = 4 - 6 = -2
    let edges_k4 = [
        SheafRestrictionEdge::new(0, 1, 1.0, 1.0, 1.0),
        SheafRestrictionEdge::new(0, 2, 1.0, 1.0, 1.0),
        SheafRestrictionEdge::new(0, 3, 1.0, 1.0, 1.0),
        SheafRestrictionEdge::new(1, 2, 1.0, 1.0, 1.0),
        SheafRestrictionEdge::new(1, 3, 1.0, 1.0, 1.0),
        SheafRestrictionEdge::new(2, 3, 1.0, 1.0, 1.0),
    ];
    let stalks_k4 = [2.0, 2.0, 2.0, 2.0];

    let rep_consistent = engine.analyze_sheaf(&stalks_k4, &edges_k4);
    assert_eq!(rep_consistent.atiyah_singer.dim_h0_global_sections, 1);
    assert_eq!(rep_consistent.atiyah_singer.dim_h1_cohomology, 3);
    assert_eq!(rep_consistent.atiyah_singer.analytical_euler_index, 4 - 6);

    // Incluso bajo frustración total (dim H^0 = 0, dim H^1 = 2), el índice topológico permanece invariante en -2
    let edges_frustrated = [
        SheafRestrictionEdge::new(0, 1, 2.0, 1.0, 1.0),
        SheafRestrictionEdge::new(0, 2, 3.0, 1.0, 1.0),
        SheafRestrictionEdge::new(0, 3, 5.0, 1.0, 1.0),
        SheafRestrictionEdge::new(1, 2, 1.0, 1.0, 1.0),
        SheafRestrictionEdge::new(1, 3, 1.0, 1.0, 1.0),
        SheafRestrictionEdge::new(2, 3, 1.0, 1.0, 1.0),
    ];
    let rep_frustrated = engine.analyze_sheaf(&stalks_k4, &edges_frustrated);
    assert_eq!(rep_frustrated.atiyah_singer.dim_h0_global_sections, 0);
    assert_eq!(rep_frustrated.atiyah_singer.dim_h1_cohomology, 2);
    assert_eq!(rep_frustrated.atiyah_singer.analytical_euler_index, 4 - 6);
}

#[test]
fn test_c5_chiral_zero_mode_asymmetry_and_topological_bias() {
    let engine = CellularSheafEngine::default();
    let edges = [
        SheafRestrictionEdge::new(0, 1, 1.0, 1.0, 1.0),
        SheafRestrictionEdge::new(1, 2, 1.0, 1.0, 1.0),
    ];
    let pos_stalks = [3.0, 3.0, 3.0];
    let neg_stalks = [-3.0, -3.0, -3.0];

    let rep_pos = engine.analyze_sheaf(&pos_stalks, &edges);
    let rep_neg = engine.analyze_sheaf(&neg_stalks, &edges);

    assert!(rep_pos.atiyah_singer.topological_chirality_bias >= -1.0);
    assert!(rep_pos.atiyah_singer.topological_chirality_bias <= 1.0);
    assert!(
        (rep_pos.atiyah_singer.chiral_zero_mode_index
            + rep_neg.atiyah_singer.chiral_zero_mode_index)
            .abs()
            < 1e-9
    );
    assert!(
        (rep_pos.atiyah_singer.topological_chirality_bias
            + rep_neg.atiyah_singer.topological_chirality_bias)
            .abs()
            < 1e-9
    );
}

#[test]
fn test_c6_adversarial_nan_inf_and_empty_inputs_zero_panic() {
    let engine = CellularSheafEngine::default();
    let empty_rep = engine.analyze_sheaf(&[], &[]);
    assert_eq!(empty_rep.n_vertices, 0);
    assert_eq!(empty_rep.n_edges, 0);

    let bad_edges = [
        SheafRestrictionEdge::new(0, 0, 1.0, 1.0, 1.0), // self-loop ignorado
        SheafRestrictionEdge::new(0, 99, 1.0, 1.0, 1.0), // fuera de rango ignorado
        SheafRestrictionEdge::new(0, 1, f64::NAN, f64::INFINITY, -5.0), // peso negativo ignorado
        SheafRestrictionEdge::new(0, 1, f64::NAN, 1.0, 1.0),
    ];
    let bad_stalks = [f64::NAN, f64::INFINITY, f64::NEG_INFINITY];

    let rep = engine.analyze_sheaf(&bad_stalks, &bad_edges);
    assert!(rep.frustration_energy.is_finite());
    assert!(rep.spectral_frustration_gap.is_finite());
    assert!(rep.cycle_holonomy_obstruction.is_finite());
    assert!(rep.atiyah_singer.topological_chirality_bias.is_finite());
}
