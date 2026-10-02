//! Contratos LXXII — inflado de cola (cópula t) en el veto same-bet.
//!
//! El consumo de λ̂ (V-RISK-002, etapa 3) debe ser: (a) BIT-EXACT sin
//! medición (D-754 — el manifest ausente no cambia nada), (b) monótono en
//! λ̂, (c) la identidad de composición sobre el complemento, (d) el caso
//! FP/FN que justifica la ola: un grupo con dependencia de cola medida
//! debe vetar un tamaño que la ρ̄ lineal admite — el stop-out conjunto
//! subestimado es exactamente el falso-negativo que LXXI midió.

use risk_engine::correlation_guard::{inflar_cola, lambda_grupo_max, veto_por_riesgo_cramer_lundberg};

#[test]
fn lxxii_sin_medicion_es_bit_a_bit_el_legado() {
    // None ⇒ idéntico bit a bit (continuidad D-754)
    for rho in [-0.5_f64, 0.0, 0.3, 0.77, 0.99, 1.0, -1.0] {
        assert_eq!(inflar_cola(rho, None).to_bits(), rho.to_bits());
        // λ = 0 (medición nula de cola) ⇒ también idéntico
        assert_eq!(inflar_cola(rho, Some(0.0)).to_bits(), rho.to_bits());
    }
}

#[test]
fn lxxii_monotono_en_lambda_y_acotado() {
    let rho = 0.6_f64;
    let mut previo = inflar_cola(rho, Some(0.0));
    for lambda in [0.05, 0.1, 0.3, 0.5, 0.8, 1.0] {
        let actual = inflar_cola(rho, Some(lambda));
        assert!(
            actual >= previo,
            "λ={lambda}: {actual} < {previo} — el inflado no es monótono"
        );
        assert!((-1.0..=1.0).contains(&actual));
        previo = actual;
    }
    // λ=1 ⇒ colas perfectas ⇒ ρ=1 (no hay independencia que preservar)
    assert!((inflar_cola(rho, Some(1.0)) - 1.0).abs() < 1e-12);
}

#[test]
fn lxxii_identidad_composicion_sobre_el_complemento() {
    // (1−ρ_final) = (1−ρ)·(1−λ) — la tres etapas (base, curl², λ̂)
    // conmutan como factores de la independencia restante.
    for rho in [0.0_f64, 0.3, 0.6, 0.9] {
        for lambda in [0.1, 0.32, 0.51, 0.9] {
            let rho_final = inflar_cola(rho, Some(lambda));
            let complemento = (1.0 - rho_final) - (1.0 - rho) * (1.0 - lambda);
            assert!(
                complemento.abs() < 1e-15,
                "ρ={rho} λ={lambda}: |{complemento}| ≥ 1e-15"
            );
        }
    }
}

#[test]
fn lxxii_falso_negativo_el_veto_caza_lo_que_rho_lineal_admite() {
    // El caso que JUSTIFICA la ola: grupo same-bet de k=4 miembros con
    // riesgo r cada uno y ρ̄ lineal 0.3 — la agregación legada ADMITE el
    // tamaño; con λ̂=0.51 (BTC-SOL medido) la ρ de cola 0.657 hace que
    // el MISMO veto RECHACE. Ese hueco era un falso negativo real.
    let riesgos = vec![0.01_f64; 5]; // 4 miembros + candidata
    let tope = 0.04;
    let rho_lineal = 0.3;
    let rho_cola = inflar_cola(rho_lineal, Some(0.51));

    let admite_lineal = !veto_por_riesgo_cramer_lundberg(
        &riesgos, Some(rho_lineal), tope, None, 0.05,
    );
    let veta_cola = veto_por_riesgo_cramer_lundberg(
        &riesgos, Some(rho_cola), tope, None, 0.05,
    );
    assert!(
        admite_lineal && veta_cola,
        "lineal={} cola={}: el inflado debe cruzar la frontera del veto en el caso medido",
        !admite_lineal, !veta_cola
    );
}

/// El store resuelve pares contra el universo dinámico y degrada a
/// fuera-de-roster sin romper. SECUENCIAL a propósito: update_dynamic_
/// universe y el OnceLock del store son estado de PROCESO — dos tests
/// paralelos se pisarían el universo mutuamente.
mod store {
    use risk_engine::copulas_store;

    #[test]
    fn manifiesto_resuelve_filtra_y_rechaza_invalidos() {
        // Fase 1: manifiesto válido con un par fuera del universo
        quantum_arena::symbols::update_dynamic_universe(vec![
            "ZZQ1USDT".into(),
            "ZZQ2USDT".into(),
        ]);
        let json = r#"{"horizonte_ms": 300000, "pares": [
    {"a": "ZZQ1USDT", "b": "ZZQ2USDT", "rho": 0.7, "nu": 4.0, "lambda": 0.4, "n": 100},
    {"a": "ZZQ3USDT", "b": "ZZQ2USDT", "rho": 0.6, "nu": 5.0, "lambda": 0.3, "n": 100}
  ]}"#;
        let n = copulas_store::cargar_pares_desde_json(json).expect("manifest válido");
        assert_eq!(n, 1, "sólo el par con ambos símbolos en el universo aplica");
        // get_coin_id es posicional: ZZQ1=0, ZZQ2=1
        assert_eq!(copulas_store::lambda_entre(0, 1), Some(0.4));
        assert_eq!(copulas_store::lambda_entre(1, 0), Some(0.4)); // simétrico
        assert_eq!(copulas_store::lambda_entre(0, 5), None); // par sin medición
        let fuera = copulas_store::pares_fuera_de_roster();
        assert!(
            fuera.iter().any(|p| p.contains("ZZQ3USDT")),
            "el símbolo muerto debe ser visible, no silencioso: {fuera:?}"
        );

        // Fase 1b (LXXIII): λ̂ del grupo = máx sobre TODOS los pares —
        // el par miembro-miembro (1,2) λ=0.4 es el único medido y domina
        // aunque la candidata (0) sólo lo comparte por pertenecer al grupo.
        // (El manifest de arriba ya cargó el par ZZQ1-ZZQ2=0.4.)
        use risk_engine::correlation_guard::lambda_grupo_max;
        // candidato-miembro medido (ZZQ1-ZZQ2 = ids 0,1)
        assert_eq!(lambda_grupo_max(0, &[1]), Some(0.4));
        // CASO LXXIII: candidata 3 SIN pares medidos con nadie, pero sus
        // miembros 0 y 1 tienen el par interno medido — la cola interna
        // del grupo cuenta aunque la candidata no la tenga.
        assert_eq!(lambda_grupo_max(3, &[0, 1]), Some(0.4));
        // grupos sin ningún par medido ⇒ None ⇒ bit-exact legado
        assert_eq!(lambda_grupo_max(3, &[1]), None);
        assert_eq!(lambda_grupo_max(0, &[3]), None);
        assert_eq!(lambda_grupo_max(3, &[]), None);

        // Fase 2: λ fuera de [0,1] rechaza EL PARSEO aunque el store ya
        // esté inicializado (parsear corre ANTES de tocar el OnceLock).
        let mal = r#"{"pares": [
    {"a": "ZZQ9USDT", "b": "ZZQ8USDT", "lambda": 1.4, "n": 10}
  ]}"#;
        assert!(copulas_store::cargar_pares_desde_json(mal).is_err());
        let nan = r#"{"pares": [
    {"a": "ZZQ9USDT", "b": "ZZQ8USDT", "lambda": NaN, "n": 10}
  ]}"#;
        assert!(copulas_store::cargar_pares_desde_json(nan).is_err());
    }
}
