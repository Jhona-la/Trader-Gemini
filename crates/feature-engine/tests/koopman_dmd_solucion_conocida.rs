//! Contratos del núcleo Koopman/DMD streaming (GLM 149) con SOLUCIÓN
//! CONOCIDA — patrón cf2 del crate: la dinámica sintética tiene autovalores
//! exactos y el motor debe recuperarlos dentro de tolerancia, o declarar
//! ausencia (CL-15), jamás inventarlos.
use feature_engine::koopman_dmd::{DmdError, KOOPMAN_M, StreamingDmd};

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

/// Oscilador amortiguado 2D x_{t+1} = ρ·R(θ)·x_t + ruido: la serie escalar
/// es la primera coordenada; su embedding Hankel de orden ≥ 2 captura la
/// dinámica con autovalores ρ·e^{±iθ}.
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

/// La señal debe ser RICA: sin offset dominante que haga a G mal
/// condicionada. El oscilador puro cumple.
fn alimentar(serie: &[f64]) -> StreamingDmd {
    let mut dmd = StreamingDmd::new();
    for &x in serie {
        dmd.push(x);
    }
    dmd
}

#[test]
fn koopman_recupera_el_autovalor_del_oscilador_amortiguado() {
    let rho = 0.97_f64;
    let theta = 0.6_f64; // ≈ periodo de ~10.5 muestras por vuelta completa
    let serie = serie_oscilador(600, rho, theta, 0.0, 42);
    let espectro = alimentar(&serie).analizar().expect("espectro estimable");
    assert!(
        (espectro.modulo_dominante - rho).abs() < 1e-3,
        "|λ| dominante {} ≈ ρ {rho}",
        espectro.modulo_dominante
    );
    let angulo_esperado = theta.rem_euclid(std::f64::consts::PI);
    let dif = (espectro.angulo_rad - angulo_esperado).abs().min(
        (espectro.angulo_rad - std::f64::consts::PI - angulo_esperado).abs(),
    );
    assert!(dif < 1e-2, "ángulo {} ≈ θ {theta}", espectro.angulo_rad);
}

#[test]
fn koopman_recupera_la_frecuencia_con_ruido_moderado() {
    let rho = 0.95_f64;
    let theta = 1.1_f64;
    let serie = serie_oscilador(2000, rho, theta, 0.05, 7);
    let espectro = alimentar(&serie).analizar().expect("espectro estimable");
    assert!(
        (espectro.modulo_dominante - rho).abs() < 0.03,
        "con ruido ρ̂ {} ≈ ρ {rho}",
        espectro.modulo_dominante
    );
    let angulo_esperado = theta.rem_euclid(std::f64::consts::PI);
    assert!(
        (espectro.angulo_rad - angulo_esperado).abs() < 0.05,
        "con ruido θ̂ {} ≈ θ {theta}",
        espectro.angulo_rad
    );
}

#[test]
fn koopman_detecta_la_bifurcacion_al_cruzar_el_circulo_unidad() {
    // Régimen 1: amortiguado (ρ=0.9). Régimen 2: expansivo (ρ=1.05) —
    // bifurcación a mitad del stream: |λ| cruza 1.0.
    let mut serie = serie_oscilador(400, 0.90, 0.8, 0.0, 1);
    serie.extend(serie_oscilador(400, 1.05, 0.8, 0.0, 2));
    let mut dmd = StreamingDmd::new();
    let mut antes: Option<f64> = None;
    for (i, &x) in serie.iter().enumerate() {
        dmd.push(x);
        if i == 399 {
            antes = dmd.analizar().ok().map(|e| e.bifurcacion_max);
        }
    }
    let despues = dmd.analizar().ok().map(|e| e.bifurcacion_max);
    let antes = antes.expect("tramo amortiguado estimable");
    let despues = despues.expect("tramo expansivo estimable");
    assert!(antes < 1.0, "régimen amortiguado: máx|λ| {antes} < 1");
    assert!(despues > 1.0, "régimen expansivo: máx|λ| {despues} > 1 — bifurcación detectada");
}

#[test]
fn koopman_warmup_insuficiente_y_stream_constante_son_ausencia_honesta() {
    let mut dmd = StreamingDmd::new();
    for i in 0..(2 * KOOPMAN_M - 1) {
        dmd.push(i as f64 * 0.01);
    }
    match dmd.analizar() {
        Err(DmdError::WarmupInsuficiente { validas, requeridas }) => {
            assert_eq!(requeridas, (2 * KOOPMAN_M) as u64);
            assert!(validas < requeridas);
        }
        otro => panic!("warmup debe ser Err, llegó {otro:?}"),
    }
    // Stream constante: con la regularización Tikhonov el operador queda
    // bien definido y el autovalor dominante es 1.0 — el INVARIANTE Koopman
    // de una dinámica constante (λ=1 es la respuesta matemática correcta,
    // no un número inventado).
    let mut plano = StreamingDmd::new();
    for _ in 0..500 {
        plano.push(0.42);
    }
    let e = plano.analizar().expect("stream constante estimable con ridge");
    assert!(
        (e.modulo_dominante - 1.0).abs() < 1e-3,
        "constante ⇒ λ=1 (invariante), llegó {}",
        e.modulo_dominante
    );
    // Las 7 raíces espurias del ridge son MÚLTIPLES (λ≈0, multiplicidad 7):
    // Durand–Kerner converge en raíces m-múltiples a error ~ε^(1/m)·radio —
    // el límite TEÓRICO del estimador deja el segundo modo ≲1% del radio,
    // dos órdenes bajo el dominante. Tolerancia documentada, no arbitraria.
    assert!(
        e.segundo_modulo < 0.01,
        "modos espurios del ridge ≲1%% del radio, llegó {}",
        e.segundo_modulo
    );
}

#[test]
fn koopman_rechaza_no_finitos_sin_contaminar_el_operador() {
    let mut dmd = StreamingDmd::new();
    let serie = serie_oscilador(400, 0.96, 0.7, 0.0, 11);
    for &x in &serie {
        dmd.push(x);
    }
    let limpio = dmd.analizar().expect("limpio estimable");
    // Inyectar NaN/Inf: se rechazan visiblemente, el espectro NO cambia.
    dmd.push(f64::NAN);
    dmd.push(f64::INFINITY);
    dmd.push(f64::NEG_INFINITY);
    assert_eq!(dmd.rechazadas(), 3, "no-finitos contados, no absorbidos");
    let tras = dmd.analizar().expect("sigue estimable");
    assert!((tras.modulo_dominante - limpio.modulo_dominante).abs() < 1e-12);
    assert_eq!(tras.muestras, limpio.muestras, "muestras válidas sin cambiar");
}

#[test]
fn koopman_factor_de_olvido_validado_en_constructor() {
    assert!(StreamingDmd::try_new_with_forget(0.995).is_ok());
    assert!(StreamingDmd::try_new_with_forget(0.0).is_err());
    assert!(StreamingDmd::try_new_with_forget(1.5).is_err());
    assert!(StreamingDmd::try_new_with_forget(f64::NAN).is_err());
}

/// El eig no-simétrico (Hessenberg+QR) sobre una matriz compañera con
/// raíces conocidas: (x−0.9)(x+0.7)(x²−0.04) = autovalores exactos
/// {0.9, −0.7, 0.2, −0.2}. Verifica el núcleo algebraico por dentro… vía
/// el contrato público con la dinámica que genera esa recurrencia:
/// x_{t+4} = c1·x_{t+3} + c2·x_{t+2} + c3·x_{t+1} + c4·x_t.
#[test]
fn koopman_recupera_la_recurrencia_lineal_de_orden_4() {
    // (x−0.9)(x+0.7)(x²−0.04) = (x²−0.2x−0.63)(x²−0.04) = x⁴ − 0.2x³ − 0.67x² + 0.008x + 0.0252
    // ⇒ x_{t+4} = 0.2x_{t+3} + 0.67x_{t+2} − 0.008x_{t+1} − 0.0252x_t.
    let (c1, c2, c3, c4) = (0.2_f64, 0.67_f64, -0.008_f64, -0.0252_f64);
    let mut s = [1.0f64, 0.3, -0.2, 0.05];
    let mut serie: Vec<f64> = s.to_vec();
    for _ in 0..800 {
        let x4 = c1 * s[3] + c2 * s[2] + c3 * s[1] + c4 * s[0];
        s = [s[1], s[2], s[3], x4];
        serie.push(x4);
    }
    let espectro = match alimentar(&serie).analizar() {
        Ok(e) => e,
        Err(err) => panic!("recurrencia orden 4 estimable: {err:?}"),
    };
    // El modo dominante de la recurrencia es 0.9.
    assert!(
        (espectro.modulo_dominante - 0.9).abs() < 1e-3,
        "λ dominante {} ≈ 0.9",
        espectro.modulo_dominante
    );
    // El segundo modo es 0.7 (el negativo aparece como |−0.7|).
    assert!(
        (espectro.segundo_modulo - 0.7).abs() < 0.02,
        "segundo |λ| {} ≈ 0.7",
        espectro.segundo_modulo
    );
}
