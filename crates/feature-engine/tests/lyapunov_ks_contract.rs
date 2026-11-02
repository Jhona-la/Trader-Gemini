//! Contratos del dual ergódico de Koopman (GLM 181, mandato §12.3):
//! espectro de Lyapunov EN VIVO — λ_max = máx ln|λ_i| por muestra,
//! h_KS = Σ_{ln|λ|>0} ln|λ| y horizonte de predictibilidad 1/λ_max.
//!
//! Solución conocida (patrón cf2, mismos generadores del contrato GLM 149):
//! el oscilador x_{t+1} = ρ·R(θ)·x_t tiene DOS exponentes de Lyapunov, ambos
//! ln ρ ⇒ λ_max = ln ρ y h_KS = 2·ln ρ (el par e^{±iθ}·ρ suma dos veces).
//! ρ < 1: espectro contractivo ⇒ h_KS = 0 y horizonte INFINITO (nada
//! destruye la trayectoria); ρ > 1: expansión medible y horizonte finito.
use feature_engine::koopman_dmd::StreamingDmd;

/// LCG determinista (Knuth MMIX) — el mismo del patrón cf2 del crate.
struct Lcg(u64);
impl Lcg {
    fn next_f64(&mut self) -> f64 {
        self.0 = self
            .0
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        ((self.0 >> 11) as f64 / (1u64 << 53) as f64) * 2.0 - 1.0
    }
}

fn serie_oscilador(n: usize, rho: f64, theta: f64, ruido: f64, semilla: u64) -> Vec<f64> {
    let mut rng = Lcg(semilla);
    let (mut x, mut y) = (1.0, 0.0);
    let (ct, st) = (theta.cos(), theta.sin());
    (0..n)
        .map(|_| {
            let nx = rho * (ct * x - st * y);
            let ny = rho * (st * x + ct * y);
            x = nx;
            y = ny;
            x + ruido * rng.next_f64()
        })
        .collect()
}

fn espectro_de(
    n: usize,
    rho: f64,
    theta: f64,
    semilla: u64,
) -> feature_engine::koopman_dmd::EspectroKoopman {
    let serie = serie_oscilador(n, rho, theta, 0.0, semilla);
    let mut dmd = StreamingDmd::new();
    for &x in &serie {
        dmd.push(x);
    }
    dmd.analizar().expect("espectro estimable")
}

#[test]
fn espectro_contractivo_no_genera_informacion_ni_horizonte_finito() {
    let e = espectro_de(600, 0.90, 0.6, 42);
    let ln_rho = 0.90_f64.ln();
    assert!(
        (e.lambda_max - ln_rho).abs() < 0.02,
        "λ_max {} ≈ ln ρ {ln_rho}",
        e.lambda_max
    );
    // Ningún ln|λ| > 0 ⇒ h_KS EXACTAMENTE 0, no «casi 0».
    assert_eq!(
        e.entropia_ks, 0.0,
        "contractivo: sin creación de información"
    );
    assert!(
        e.horizonte_muestras.is_infinite(),
        "sin expansión ⇒ horizonte infinito, dio {}",
        e.horizonte_muestras
    );
}

#[test]
fn oscilador_neutro_esta_en_la_frontera_de_predictibilidad() {
    let e = espectro_de(600, 1.0, 0.6, 7);
    // |λ| = 1 ⇒ ln|λ| ≈ 0: frontera exacta entre orden y caos.
    assert!(
        e.lambda_max.abs() < 0.01,
        "frontera: λ_max ≈ 0, dio {}",
        e.lambda_max
    );
    assert!(e.entropia_ks < 0.02, "h_KS ≈ 0, dio {}", e.entropia_ks);
    // λ_max ≤ 0 (o numéricamente 0) ⇒ horizonte INFINITO o enorme; si el
    // ajuste da una pizca positiva, el horizonte debe ser grande.
    assert!(
        e.horizonte_muestras.is_infinite() || e.horizonte_muestras > 100.0,
        "frontera ⇒ horizonte ∞ o >100 muestras, dio {}",
        e.horizonte_muestras
    );
}

#[test]
fn oscilador_expansivo_da_lyapunov_y_ks_exactos() {
    let rho = 1.05_f64;
    let e = espectro_de(600, rho, 0.6, 11);
    let ln_rho = rho.ln();
    // λ_max = ln ρ (un miembro del par domina el máximo).
    assert!(
        (e.lambda_max - ln_rho).abs() < 0.02,
        "λ_max {} ≈ ln ρ {ln_rho}",
        e.lambda_max
    );
    // h_KS = 2·ln ρ: el par e^{±iθ}·ρ aporta DOS exponentes positivos.
    assert!(
        (e.entropia_ks - 2.0 * ln_rho).abs() < 0.06,
        "h_KS {} ≈ 2·ln ρ {}",
        e.entropia_ks,
        2.0 * ln_rho
    );
    // Horizonte = 1/λ_max (escala e-folding del error).
    let esperado = 1.0 / ln_rho;
    assert!(
        (e.horizonte_muestras - esperado).abs() / esperado < 0.05,
        "horizonte {} ≈ 1/λ_max {esperado}",
        e.horizonte_muestras
    );
    // Invariante estructural: h_KS nunca es menor que λ_max.
    assert!(
        e.entropia_ks >= e.lambda_max,
        "h_KS {} ≥ λ_max {}",
        e.entropia_ks,
        e.lambda_max
    );
}

#[test]
fn la_entropia_ordena_los_regimenes() {
    // Monotonía física: contracción < frontera < expansión.
    let e_contr = espectro_de(400, 0.90, 0.8, 1);
    let e_neutro = espectro_de(400, 1.0, 0.8, 2);
    let e_expan = espectro_de(400, 1.05, 0.8, 3);
    assert!(e_contr.entropia_ks <= e_neutro.entropia_ks);
    assert!(e_neutro.entropia_ks < e_expan.entropia_ks);
    assert!(e_contr.horizonte_muestras > e_expan.horizonte_muestras);
}
