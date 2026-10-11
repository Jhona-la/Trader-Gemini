//! LF-2 (Línea F, revisión de composición de C-2/Ω48 con MG05) — la escala a
//! la que se publica el IC de grupo (`qo_613_rho_tau`).
//!
//! La malla `SPECTRUM_SCALES_MS` es geométrica (razón 4). El nodo más cercano
//! a una τ debe elegirse en ln τ: en distancia lineal la frontera entre los
//! nodos k y k+1 cae en 2,5·τ_k en vez de 2·τ_k (su punto medio geométrico), y
//! una τ de 40 s (2,33·τ_17) se asignaba al nodo de 17,2 s, fuera de la banda
//! operable [30 s, 12 h] que asume `FAMILIA_VETO_GRUPO`.
use god_engine_core::evidence_publication::escala_espectral_mas_cercana;
use quantum_arena::temporal_spectrum::SPECTRUM_SCALES_MS;

#[test]
fn lf2_el_nodo_mas_cercano_se_mide_en_ln_tau() {
    // 40 s está más cerca, en ln τ, de 68,7 s (nodo 18) que de 17,2 s (nodo 17).
    assert_eq!(escala_espectral_mas_cercana(40_000.0), Some(18));
    // 30 s (1,75·τ_17) sigue en el nodo 17 con cualquiera de las dos métricas.
    assert_eq!(escala_espectral_mas_cercana(30_000.0), Some(17));
}

#[test]
fn lf2_la_frontera_entre_nodos_es_su_punto_medio_geometrico() {
    for k in 0..SPECTRUM_SCALES_MS.len() - 1 {
        let medio = (SPECTRUM_SCALES_MS[k] * SPECTRUM_SCALES_MS[k + 1]).sqrt();
        assert_eq!(escala_espectral_mas_cercana(medio * 0.99), Some(k), "justo bajo el medio de {k}");
        assert_eq!(escala_espectral_mas_cercana(medio * 1.01), Some(k + 1), "justo sobre el medio de {k}");
    }
}

#[test]
fn lf2_tau_invalida_no_tiene_nodo() {
    for t in [0.0, -1.0, f64::NAN, f64::INFINITY] {
        assert_eq!(escala_espectral_mas_cercana(t), None, "τ = {t}");
    }
}
