//! Contrato de la Curva de Page (Fase 81) con exposición real.
//!
//! El motor recibe la exposición bruta en unidades de equity (|nocional_i| / equity), no pesos
//! normalizados: un libro plano no radia entropía y toda la equity es isla. Con exposición
//! abierta, la radiación crece con Λ·σ. Lo que no se puede medir (peso, volatilidad o
//! correlación desconocidos) no se lee como cero: veta o no acredita isla. En o bajo el piso de
//! Grossman-Zhou no queda capacidad y veta.
//!
//! Valores con la configuración por defecto (piso 5 USD, σ_ref 0,02, área mínima 0,15).

use risk_engine::{PageCurveIslandConfig, PageCurveIslandRiskEngine, MAX_PORTFOLIO_ASSETS};

const TOL: f64 = 1e-9;

fn motor() -> PageCurveIslandRiskEngine {
    PageCurveIslandRiskEngine::new(PageCurveIslandConfig::default())
}

fn cerca(a: f64, b: f64) -> bool {
    (a - b).abs() <= TOL
}

#[test]
fn c1_libro_plano_de_13_usd_no_radia_y_no_veta() {
    // En el motor anterior este libro plano radiaba ln(1 + 0,5/0,02) = ln26 > S_BH sólo por la
    // volatilidad (post-Page), y lo salvaba únicamente que la ρ desconocida se leía como 0.
    let plano = [0.0; MAX_PORTFOLIO_ASSETS];
    // Volatilidad alta y correlación desconocida: sin exposición no intervienen.
    let r = motor().evaluate(13.0, &plano, 0.5, f64::NAN);

    assert!(cerca(
        r.bekenstein_hawking_entropy,
        (1.0_f64 + 13.0 / 5.0).ln()
    ));
    assert_eq!(r.radiation_entropy, 0.0, "un libro plano no radia");
    assert!(!r.post_page_transition);
    assert!(
        r.quantum_island_present,
        "sin exposición toda la equity es isla"
    );
    assert!(!r.island_veto_induced);
    assert!(cerca(r.slot_scale_factor, 1.25));
}

#[test]
fn c2_equity_en_el_piso_veta_aun_con_libro_plano() {
    let plano = [0.0; MAX_PORTFOLIO_ASSETS];
    let en_piso = motor().evaluate(5.0, &plano, 0.05, 0.0);
    assert!(en_piso.island_veto_induced, "en el piso no queda capacidad");
    assert_eq!(en_piso.slot_scale_factor, 0.0);

    let bajo_piso = motor().evaluate(4.0, &plano, 0.05, 0.0);
    assert!(bajo_piso.island_veto_induced);

    let sobre_piso = motor().evaluate(5.01, &plano, 0.05, 0.0);
    assert!(!sobre_piso.island_veto_induced);
}

#[test]
fn c3_volatilidad_desconocida_con_exposicion_abierta_veta() {
    // Un slot abierto de 0,4× equity (riesgo 0,02 × apalancamiento 20).
    let un_slot = [0.4, 0.0, 0.0, 0.0];
    for vol in [f64::NAN, f64::INFINITY, f64::NEG_INFINITY, -0.01] {
        let r = motor().evaluate(13.0, &un_slot, vol, 1.0);
        assert!(
            r.island_veto_induced,
            "volatilidad {vol} con exposición debe vetar"
        );
        assert_eq!(r.slot_scale_factor, 0.0);
        assert!(r.radiation_entropy.is_finite());
    }

    // Sin exposición, la volatilidad desconocida no veta.
    let plano = [0.0; MAX_PORTFOLIO_ASSETS];
    let r = motor().evaluate(13.0, &plano, f64::NAN, 1.0);
    assert!(!r.island_veto_induced);
}

#[test]
fn c4_un_slot_radia_en_proporcion_a_su_exposicion() {
    let un_slot = [0.4, 0.0, 0.0, 0.0];
    let r = motor().evaluate(13.0, &un_slot, 0.01, 1.0);

    // Un solo activo: Shannon 0, término térmico ln(1 + 0,4·0,01/0,02) = ln1,2.
    assert!(cerca(r.radiation_entropy, 1.2_f64.ln()));
    assert!(!r.post_page_transition);
    assert!(
        !r.quantum_island_present,
        "ρ = 1 con el propio instrumento: sin isla"
    );
    assert!(!r.island_veto_induced);
    let s_bh = 3.6_f64.ln();
    let esperado = 1.20 * (s_bh - 1.2_f64.ln()) / (s_bh + 0.1);
    assert!(cerca(r.slot_scale_factor, esperado));
}

#[test]
fn c5_cerca_del_piso_la_misma_exposicion_veta_con_mas_volatilidad() {
    // E = 6, piso 5: S_BH = ln2,2. Un slot de 0,4×.
    let un_slot = [0.4, 0.0, 0.0, 0.0];

    // σ = 0,07: ln(1 + 1,4) = ln2,4 > ln2,2, post-Page sin isla ⇒ veto.
    let alta = motor().evaluate(6.0, &un_slot, 0.07, 1.0);
    assert!(alta.post_page_transition);
    assert!(alta.island_veto_induced);

    // σ = 0,05: ln2 < ln2,2 ⇒ sin veto.
    let baja = motor().evaluate(6.0, &un_slot, 0.05, 1.0);
    assert!(!baja.post_page_transition);
    assert!(!baja.island_veto_induced);
}

#[test]
fn c6_peso_desconocido_o_negativo_es_exposicion_desconocida() {
    let casos = [
        [0.25, 0.25, f64::NAN, 0.25],
        [0.25, f64::INFINITY, 0.25, 0.25],
        [0.25, 0.25, -0.10, 0.25],
        [f64::MAX, f64::MAX, 0.0, 0.0],
    ];
    for pesos in casos {
        let r = motor().evaluate(1000.0, &pesos, 0.05, 0.20);
        assert!(r.island_veto_induced, "pesos {pesos:?} deben vetar");
        assert!(
            !r.quantum_island_present,
            "exposición desconocida no acredita isla"
        );
        assert!(r.radiation_entropy.is_finite());
        assert!(r.page_entropy.is_finite());
    }
}

#[test]
fn c7_carga_termica_no_certificable_veta() {
    // Exposición finita pero tan grande que Λ·σ/σ_ref desborda.
    let enorme = [1.0e308, 0.0, 0.0, 0.0];
    let r = motor().evaluate(1000.0, &enorme, 10.0, 0.0);
    assert!(r.island_veto_induced);
    assert!(r.radiation_entropy.is_finite());
}

#[test]
fn c8_la_radiacion_es_extensiva_en_la_exposicion() {
    // Misma composición (Shannon ln4), exposición doble: el término térmico pasa de
    // ln(1 + 2,5) a ln(1 + 5).
    let simple = motor().evaluate(1000.0, &[0.25; MAX_PORTFOLIO_ASSETS], 0.05, 0.20);
    let doble = motor().evaluate(1000.0, &[0.5; MAX_PORTFOLIO_ASSETS], 0.05, 0.20);

    assert!(cerca(simple.radiation_entropy, 4.0_f64.ln() + 3.5_f64.ln()));
    assert!(cerca(doble.radiation_entropy, 4.0_f64.ln() + 6.0_f64.ln()));
    assert!(cerca(
        doble.radiation_entropy - simple.radiation_entropy,
        6.0_f64.ln() - 3.5_f64.ln()
    ));
}

#[test]
fn c9_correlacion_desconocida_con_exposicion_no_acredita_isla() {
    let pesos = [0.25; MAX_PORTFOLIO_ASSETS];
    let conocida = motor().evaluate(1000.0, &pesos, 0.05, 0.20);
    assert!(conocida.quantum_island_present);

    let desconocida = motor().evaluate(1000.0, &pesos, 0.05, f64::NAN);
    assert!(
        !desconocida.quantum_island_present,
        "ρ desconocida no es diversificación perfecta"
    );
}

#[test]
fn c10_la_radiacion_no_baja_al_subir_una_volatilidad_conocida() {
    // E = 6 (S_BH = ln2,2), Λ = 2,5. Antes, una volatilidad conocida ≤ 1e-4 saltaba a 0,01:
    // σ = 0 vetaba (ln2,25 > ln2,2) y σ = 2e-4 no.
    let pesos = [2.5, 0.0, 0.0, 0.0];
    let nula = motor().evaluate(6.0, &pesos, 0.0, 1.0);
    assert!(
        !nula.island_veto_induced,
        "volatilidad conocida nula no veta"
    );

    let mut previa = f64::NEG_INFINITY;
    for k in 0..=200 {
        let vol = 1e-4 * k as f64;
        let r = motor().evaluate(6.0, &pesos, vol, 1.0);
        assert!(
            r.radiation_entropy >= previa,
            "σ = {vol}: radiación {} < {previa}",
            r.radiation_entropy
        );
        previa = r.radiation_entropy;
    }
}
