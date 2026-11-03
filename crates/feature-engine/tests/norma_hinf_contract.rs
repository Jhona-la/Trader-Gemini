//! Contratos de la norma H∞ escalar (GLM 185, mandato §12.4) con SOLUCIÓN
//! CONOCIDA: |b|/√(1−a²) exacta para el sistema contractivo, y las dos
//! ganancias del peor caso del filtro de precio (proceso y medición).
use feature_engine::norma_hinf::{GananciaHinfEmpirica, ganancias_hinf_filtro, norma_hinf_escalar};

#[test]
fn forma_cerrada_es_exacta() {
    // a=0.6, b=0.8: ‖G‖∞ = 0.8/√(1−0.36) = 0.8/0.8 = 1.0 EXACTO.
    assert_eq!(norma_hinf_escalar(0.6, 0.8).expect("contractivo"), 1.0);
    // a=0 ⇒ sin memoria: ‖G‖∞ = |b| puro (energía instantánea).
    assert_eq!(norma_hinf_escalar(0.0, 2.5).expect("contractivo"), 2.5);
    assert_eq!(norma_hinf_escalar(0.0, -2.5).expect("contractivo"), 2.5);
    // a=0.5, b=1: 1/√(1−0.25) = 2/√3.
    let esperado = 2.0 / 3.0f64.sqrt();
    assert!((norma_hinf_escalar(0.5, 1.0).expect("contractivo") - esperado).abs() < 1e-12);
    // Monotonía: más memoria (|a|→1) ⇒ más ganancia del peor caso.
    let g_baja = norma_hinf_escalar(0.2, 1.0).expect("contractivo");
    let g_alta = norma_hinf_escalar(0.9, 1.0).expect("contractivo");
    assert!(g_baja < g_alta);
}

#[test]
fn no_contractivo_es_error_honesto() {
    // |a| ≥ 1: integrador — la energía del peor caso no tiene cota.
    assert!(norma_hinf_escalar(1.0, 1.0).is_err());
    assert!(norma_hinf_escalar(-1.0, 1.0).is_err());
    assert!(norma_hinf_escalar(1.5, 0.3).is_err());
    // Degenerados: sin invención.
    assert!(norma_hinf_escalar(f64::NAN, 1.0).is_err());
    assert!(norma_hinf_escalar(0.5, f64::INFINITY).is_err());
}

#[test]
fn filtro_de_precio_tiene_dos_ganancias_del_peor_caso() {
    // K=0.5 ⇒ a=0.5: proceso = 0.5/√0.75, medición = 0.5/√0.75 (simetría).
    let g = ganancias_hinf_filtro(0.5).expect("K válido");
    let denominador = 0.75f64.sqrt();
    assert!((g.proceso - 0.5 / denominador).abs() < 1e-12);
    assert!((g.medicion - 0.5 / denominador).abs() < 1e-12);
    assert_eq!(g.a, 0.5);
    // K pequeño (filtro lento, a→1): el numerador del canal de PROCESO es
    // a≈1 y el de medición es K≈0 ⇒ domina el proceso (CI cazó las
    // aserciones invertidas de la primera versión — la matemática manda).
    let lento = ganancias_hinf_filtro(0.1).expect("K válido");
    assert!(lento.proceso > lento.medicion);
    // K grande (filtro agresivo, a→0): numerador de proceso a≈0 y el de
    // medición K≈1 ⇒ domina la medición.
    let agresivo = ganancias_hinf_filtro(0.9).expect("K válido");
    assert!(agresivo.medicion > agresivo.proceso);
    // Fuera de (0,1): estimador muerto o divergente — Err.
    assert!(ganancias_hinf_filtro(0.0).is_err());
    assert!(ganancias_hinf_filtro(1.0).is_err());
    assert!(ganancias_hinf_filtro(1.5).is_err());
}

#[test]
fn empirica_cota_el_peor_caso_observado() {
    // Sistema con ganancia verdadera |b|/√(1−a²): a=0.5, b=1 ⇒ ≈1.1547.
    // La empírica sobre el sistema REAL (energía salida/entrada de una
    // realización) debe quedarse POR DEBAJO de la forma cerrada (el peor
    // caso analítico es un supremo sobre TODAS las entradas, la muestra
    // ve una sola) y ser positiva una vez llena la ventana.
    let mut emp = GananciaHinfEmpirica::try_new(64).expect("ventana válida");
    // Entrada determinista mezclada (no la peor): rampa + alternancia.
    let mut e = 0.0f64;
    let (a, b) = (0.5f64, 1.0f64);
    let mut entrada_prev = 0.0f64;
    for k in 0..200u32 {
        let u = if k % 2 == 0 { 0.7 } else { -0.4 };
        let du = u - entrada_prev;
        entrada_prev = u;
        e = a * e + b * du;
        emp.push(e, du).expect("muestras finitas");
    }
    let peor = emp.peor_ganancia().expect("ventana llena");
    let exacta = norma_hinf_escalar(a, b).expect("contractivo");
    assert!(peor > 0.0, "ganancia empírica positiva, dio {peor}");
    assert!(
        peor <= exacta * (1.0 + 1e-9),
        "la muestra ({peor}) no supera el peor caso analítico ({exacta})"
    );
}

#[test]
fn empirica_rechaza_lo_invalido() {
    assert!(GananciaHinfEmpirica::try_new(4).is_err());
    let mut emp = GananciaHinfEmpirica::try_new(8).expect("ventana válida");
    assert!(
        emp.peor_ganancia().is_none(),
        "sin ventana llena no hay ganancia"
    );
    assert!(emp.push(f64::NAN, 1.0).is_err());
    assert!(emp.push(1.0, f64::INFINITY).is_err());
    // Estado intacto tras los Err.
    assert_eq!(emp.llenos(), 0);
}
