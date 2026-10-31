//! CONTRATO σz STREAMING (GLM-175, §10 firmas): el observable de la
//! medición GLM-163 promovido a módulo — z del cross S¹² contra los
//! momentos EXACTOS del nulo de máxima entropía.
//!
//! QUÉ: `FirmasMaxEnt` (anillo O(1) push, medir O(L)) publica
//! z(S¹²) por ventana; la medición 163 documentó la separación
//! transversal σz 0,47→1,79 (10 tapes × 4.000 ventanas).
//! POR QUÉ: hasta ahora el z sólo existía en el vehículo de medición;
//! el observable de régimen necesita vivir en el motor.
//! CÓMO: (1) paridad streaming-batch con el nulo de enumeración bruta;
//! (2) semántica del anillo (últimas L); (3) anti-simetría z(S²¹)=−z(S¹²)
//! recalculada; (4) ventana incompleta y nulo degenerado; (5) saturación.

use feature_engine::firma_maxent::FirmasMaxEnt;
use feature_engine::path_signatures::firma_nivel2_incrementos;

/// Los mismos pesos/momentos del contrato del 163 (copiados deliberadamente:
/// Los mismos pesos/momentos del contrato del 163 (copiados deliberadamente:
/// el diseño experimental del nulo vive en los tests, no en el módulo).
fn momentos_nulo(xs: &[f64], ys: &[f64]) -> (f64, f64) {
    let n = xs.len() as f64;
    let (sx, sy): (f64, f64) = (xs.iter().sum(), ys.iter().sum());
    let (x_bar, v_bar_cero) = (sx / n, 0.0);
    let _ = v_bar_cero;
    let sxx = xs.iter().map(|&a| (a - x_bar) * (a - x_bar)).sum::<f64>();
    let mut v = Vec::with_capacity(xs.len());
    let mut sufijo = sy;
    for &y in ys {
        sufijo -= y;
        v.push(sufijo + 0.5 * y);
    }
    let sv: f64 = v.iter().sum();
    let v_bar = sv / n;
    let svv = v.iter().map(|&a| (a - v_bar) * (a - v_bar)).sum::<f64>();
    (x_bar * sv, sxx * svv / (n - 1.0))
}

fn ventana_generica(n: usize) -> (Vec<f64>, Vec<f64>) {
    ventana_con_semilla(n, 0x9E37_79B9_7F4A_7C15)
}

fn ventana_con_semilla(n: usize, semilla_inicial: u64) -> (Vec<f64>, Vec<f64>) {
    // dt NO simétricos (Σy = 1) y retornos genéricos — el caso del feed.
    let mut xs = Vec::with_capacity(n);
    let mut ys = Vec::with_capacity(n);
    let mut semilla = semilla_inicial;
    let mut rng = move || {
        semilla ^= semilla >> 12;
        semilla ^= semilla << 25;
        semilla ^= semilla >> 27;
        semilla.wrapping_mul(0x2545_F491_4F6C_DD1D)
    };
    let mut acumulado = 0.0_f64;
    for i in 0..n {
        xs.push(-0.3 + 0.6 * ((rng() % 10_000) as f64 / 10_000.0));
        let fraccion = if i == n - 1 { 1.0 - acumulado } else { 1.0 / (3.0 * n as f64) };
        acumulado += fraccion;
        ys.push(fraccion);
    }
    (xs, ys)
}

#[test]
fn sz1_paridad_streaming_batch_con_nulo_cerrado() {
    // L=6 genérica: el z del streaming == z de la ventana batch contra
    // los momentos cerrados (que a su vez están anclados por enumeración
    // bruta de 720 perms en el contrato del 163).
    let (xs, ys) = ventana_generica(6);
    let mut f = FirmasMaxEnt::new(6).expect("ventana válida");
    assert!(f.medir().is_none(), "ventana incompleta: no mide");
    for (&x, &y) in xs.iter().zip(ys.iter()) {
        f.push(x, y);
    }
    let m = f.medir().expect("ventana llena mide");
    assert_eq!(m.n, 6);

    let incs: Vec<(f64, f64)> = xs.iter().zip(ys.iter()).map(|(&a, &b)| (a, b)).collect();
    let s12 = firma_nivel2_incrementos(&incs).level2[0][1];
    let (e, var) = momentos_nulo(&xs, &ys);
    let z_batch = (s12 - e) / var.sqrt();
    assert!(
        (m.z12 - z_batch).abs() <= 1e-12,
        "streaming {m_z} vs batch {z_batch}",
        m_z = m.z12
    );
    // sigma_x coherente con la muestra.
    let nf = 6.0_f64;
    let x_bar = xs.iter().sum::<f64>() / nf;
    let sxx = xs.iter().map(|&a| (a - x_bar) * (a - x_bar)).sum::<f64>();
    assert!((m.sigma_x - (sxx / (nf - 1.0)).sqrt()).abs() <= 1e-12);
}

#[test]
fn sz2_el_anillo_cubre_las_ultimas_l() {
    // Tras 2L pushes, medir() es el z de la SEGUNDA ventana — no de la
    // primera ni de la unión.
    let (xs1, ys1) = ventana_con_semilla(6, 0x9E37_79B9_7F4A_7C15);
    let (xs2, ys2) = ventana_con_semilla(6, 0xA24B_A8C5_F1D3_6E20);
    let mut f = FirmasMaxEnt::new(6).expect("ventana válida");
    for i in 0..6 {
        f.push(xs1[i], ys1[i]);
    }
    for i in 0..6 {
        f.push(xs2[i], ys2[i]);
    }
    let m = f.medir().expect("llena");
    let incs2: Vec<(f64, f64)> = xs2.iter().zip(ys2.iter()).map(|(&a, &b)| (a, b)).collect();
    let s12_2 = firma_nivel2_incrementos(&incs2).level2[0][1];
    let (e2, var2) = momentos_nulo(&xs2, &ys2);
    let z2 = (s12_2 - e2) / var2.sqrt();
    assert!((m.z12 - z2).abs() <= 1e-12, "el anillo debe cubrir las últimas L");
    // Y ≠ z de la primera ventana (genérica distinta).
    let (e1, var1) = momentos_nulo(&xs1, &ys1);
    let incs1: Vec<(f64, f64)> = xs1.iter().zip(ys1.iter()).map(|(&a, &b)| (a, b)).collect();
    let z1 = (firma_nivel2_incrementos(&incs1).level2[0][1] - e1) / var1.sqrt();
    assert!((m.z12 - z1).abs() > 1e-6, "ventanas genéricas distintas ⇒ z distinto");
}

#[test]
fn sz3_antisimetria_exacta_verificada_por_el_kernel() {
    // z(S²¹) recalculado con prefijos == −z(S¹²) exacto (identidad del
    // 163: con Σy=1, S¹²+S²¹=S¹S², E₁₂+E₂₁=S¹, Var₁₂=Var₂₁).
    let (xs, ys) = ventana_generica(8);
    let mut f = FirmasMaxEnt::new(8).expect("ventana válida");
    for (&x, &y) in xs.iter().zip(ys.iter()) {
        f.push(x, y);
    }
    let m = f.medir().expect("mide");
    let z21 = f.medir_z21().expect("z21 mide");
    assert!(
        (m.z12 + z21).abs() <= 1e-9,
        "z12 {z12} + z21 {z21} debe ser 0",
        z12 = m.z12
    );
}

#[test]
fn sz4_ventana_incompleta_y_nulo_degenerado() {
    // Incompleta: None hasta L pushes.
    let mut f = FirmasMaxEnt::new(4).expect("ventana válida");
    f.push(0.1, 0.25);
    assert!(f.medir().is_none());
    // Degenerado: retornos constantes ⇒ Sxx=0 ⇒ Var=0 ⇒ None (z no medible).
    let mut g = FirmasMaxEnt::new(4).expect("ventana válida");
    for i in 0..4 {
        g.push(0.10, 0.25 * (i as f64 + 1.0) / 4.0);
    }
    assert!(g.medir().is_none(), "retornos constantes: z indefinido");
    // Constructor: ventana < 2 rechazado.
    assert!(FirmasMaxEnt::new(1).is_err());
    assert!(FirmasMaxEnt::new(0).is_err());
}

#[test]
fn sz5_estabilidad_en_valores_extremos() {
    // Incrementos extremos: el z o es finito o None — jamás NaN/Inf.
    let mut f = FirmasMaxEnt::new(4).expect("ventana válida");
    for &(x, y) in &[(1e300, 0.25), (-1e300, 0.25), (1e-300, 0.25), (f64::MAX, 0.25)] {
        f.push(x, y);
    }
    match f.medir() {
        Some(m) => assert!(m.z12.is_finite(), "z debe ser finito o None"),
        None => {}
    }
    // Con NaN en la entrada el observable no publica basura.
    let mut h = FirmasMaxEnt::new(4).expect("ventana válida");
    for &(x, y) in &[(0.1, 0.25), (f64::NAN, 0.25), (-0.2, 0.25), (0.05, 0.25)] {
        h.push(x, y);
    }
    match h.medir() {
        Some(m) => assert!(m.z12.is_finite(), "NaN de entrada no debe producir z"),
        None => {}
    }
}
