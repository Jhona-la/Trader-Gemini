//! Contratos de la geometría Fisher-Rao (GLM-191, mandato §3.4). Toda
//! la numerología fue VERIFICADA A MANO (python, mismo LCG, mismas
//! recurrencias EWMA) antes de escribir el núcleo — incluida una
//! calibración del nulo que DESCARTÓ un diseño (warmup corto: la
//! convergencia de la escala lenta contamina la deformación con valores
//! comparables a un cambio de régimen real; medido, no supuesto).
use feature_engine::fisher_rao::{
    DeformacionFisherRao, PERIODO_LENTO, PERIODO_RAPIDO, WARMUP, dist_fisher_rao,
};
use std::f64::consts::SQRT_2;

/// LCG determinista del contrato (idéntico al de los contratos 189/190):
/// Knuth 64-bit, uniforme por los 53 bits altos, Box–Muller par fijo.
struct Lcg(u64);

impl Lcg {
    fn unif(&mut self) -> f64 {
        self.0 = self
            .0
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        (self.0 >> 11) as f64 / (1u64 << 53) as f64
    }
    fn gauss(&mut self) -> f64 {
        let u1 = self.unif();
        let u2 = self.unif().max(1e-300);
        (2.0 * std::f64::consts::PI * u1).cos() * (-2.0 * u2.ln()).sqrt()
    }
}

#[test]
fn identidades_de_la_forma_cerrada() {
    // Identidad de indiscernibles: EXACTO (arcosh(1) = 0).
    assert_eq!(dist_fisher_rao(10.0, 2.0, 10.0, 2.0).expect("válido"), 0.0);
    // Simetría.
    let ab = dist_fisher_rao(10.0, 2.0, 20.0, 3.0).expect("válido");
    let ba = dist_fisher_rao(20.0, 3.0, 10.0, 2.0).expect("válido");
    assert_eq!(ab, ba);
    // Caso σ puro: la geodésica vertical da √2·|ln(σ₂/σ₁)| EXACTO
    // (verificado a mano a 2e-16 — la identidad algebraica cierra).
    for r in [1.5f64, 2.0, 4.0, 0.25] {
        let d = dist_fisher_rao(0.0, 1.0, 0.0, r).expect("válido");
        let esperado = SQRT_2 * r.ln().abs();
        assert!(
            (d - esperado).abs() < 1e-12,
            "σ puro r={r}: {d} ≈ {esperado}"
        );
    }
    // La geodésica es el mínimo: la distancia ≤ la integral métrica por
    // el camino RECTO en (μ,σ) — cota superior numérica del diseño.
    let (mu1, s1, mu2, s2) = (10.0f64, 2.0, -5.0, 3.0);
    let d = dist_fisher_rao(mu1, s1, mu2, s2).expect("válido");
    let n = 200_000usize;
    let mut largo = 0.0f64;
    for i in 0..n {
        let t0 = i as f64 / n as f64;
        let t1 = (i + 1) as f64 / n as f64;
        let s_mid = s1 + (s2 - s1) * (t0 + t1) / 2.0;
        let dm = mu2 - mu1;
        let ds = s2 - s1;
        let paso = (t1 - t0) as f64;
        let ds2 = (dm * dm) / (s_mid * s_mid) + 2.0 * (ds * ds) / (s_mid * s_mid);
        largo += ds2.sqrt() * paso;
    }
    assert!(
        d <= largo + 1e-9,
        "geodésica {d} ≤ camino recto {largo} (verificado a mano: 4.297 ≤ 6.109)"
    );
}

#[test]
fn falla_cerrado_sin_ceros_fabricados() {
    assert!(dist_fisher_rao(0.0, -1.0, 0.0, 1.0).is_err());
    assert!(dist_fisher_rao(0.0, 0.0, 0.0, 1.0).is_err());
    assert!(dist_fisher_rao(f64::NAN, 1.0, 0.0, 1.0).is_err());
    assert!(dist_fisher_rao(0.0, f64::INFINITY, 0.0, 1.0).is_err());
    // La escala EWMA respeta su período (α = 2/(P+1)) y el streaming
    // hereda el fall-closed: serie constante ⇒ var 0 ⇒ None (ausencia
    // honesta, jamás σ=0 publicable).
    let mut d = DeformacionFisherRao::new();
    for _ in 0..(WARMUP + 8) {
        assert_eq!(d.push(0.01), None, "serie constante: var degenerada ⇒ None");
    }
    assert_eq!(PERIODO_RAPIDO, 64);
    assert_eq!(PERIODO_LENTO, 512);
}

#[test]
fn el_nulo_caminante_es_acotado_y_documentado() {
    // 8 caminantes LCG deterministas (retornos iid σ=0.004, 1024 pasos):
    // calibración medida a mano — p50 ∈ [0.11, 0.15], MÁXIMO 0.372.
    // El contrato certifica que el nulo del núcleo REPRODUCE esa cota
    // con margen: nada supera 0.40 en ruido puro.
    let mut maximo = 0.0f64;
    for semilla in [42u64, 1, 2, 3, 4, 5, 6, 7] {
        let mut rng = Lcg(semilla);
        let mut d = DeformacionFisherRao::new();
        let mut max_semilla = 0.0f64;
        for _ in 0..1024 {
            if let Some(v) = d.push(rng.gauss() * 0.004) {
                max_semilla = max_semilla.max(v);
            }
        }
        assert!(
            max_semilla < 0.40,
            "semilla {semilla}: el nulo reproduce la cota medida (max {max_semilla} < 0.40; medido a mano 0.372)"
        );
        maximo = maximo.max(max_semilla);
    }
    // Orden de magnitud de la calibración (el número fino vive en la
    // doc del módulo): el nulo NO es cero — es la cola de estimación.
    assert!(maximo > 0.20, "calibración viva: max nulo {maximo} > 0.20");
}

#[test]
fn el_cambio_de_regimen_deforma_y_el_drift_tambien() {
    // Shift σ×2 en t=768 (semilla 42, medido a mano): pico post 0.69 —
    // ≥1.9× el máximo del nulo (0.372). El estadístico RESPONDE a la
    // física del §3.4: la distribución de corto plazo se aleja de la de
    // fondo cuando el régimen cambia.
    let mut rng = Lcg(42);
    let mut d = DeformacionFisherRao::new();
    let mut max_estacionario = 0.0f64;
    let mut max_post = 0.0f64;
    for i in 0..1024u32 {
        let sig = if i >= 768 { 0.008 } else { 0.004 };
        if let Some(v) = d.push(rng.gauss() * sig) {
            if i >= 768 {
                max_post = max_post.max(v);
            } else {
                max_estacionario = max_estacionario.max(v);
            }
        }
    }
    assert!(
        max_post > 0.60,
        "σ×2 deforma: pico {max_post} > 0.60 (medido a mano 0.69)"
    );
    assert!(
        max_post > max_estacionario,
        "el pico del cambio supera al tramo estacionario ({max_estacionario})"
    );
    // Drift de media +1σ (misma semilla, medido a mano: pico 0.73): la
    // geometría también ve el desplazamiento de la media, no sólo la
    // varianza.
    let mut rng2 = Lcg(42);
    let mut d2 = DeformacionFisherRao::new();
    let mut max_drift = 0.0f64;
    for i in 0..1024u32 {
        let x = rng2.gauss() * 0.004 + if i >= 768 { 0.004 } else { 0.0 };
        if let Some(v) = d2.push(x) {
            if i >= 768 {
                max_drift = max_drift.max(v);
            }
        }
    }
    assert!(
        max_drift > 0.65,
        "drift +1σ deforma: pico {max_drift} > 0.65 (medido a mano 0.73)"
    );
}
