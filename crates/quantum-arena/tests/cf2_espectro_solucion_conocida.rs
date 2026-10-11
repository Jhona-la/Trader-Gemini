//! C-F2.3a (Línea C): el espectro temporal frente a procesos con solución
//! conocida (sólo API pública de `TemporalSpectrum`).
//!
//! Prueba de solución conocida de la fase C-F2 (ver
//! `docs/audit/HALLAZGOS_C-F2_2026-10-11.md`). Dos procesos con forma cerrada,
//! a un evento por segundo de reloj del exchange:
//!
//! - Ornstein-Uhlenbeck exacto en log-precio, vida media 30 s y desviación
//!   estacionaria s: para dos bloques adyacentes de duración L,
//!   corr(r₁, r₂) = −(1 − a)/2 con a = e^{−θL}, y por Sheppard
//!   E[sign(r₁r₂)] = (2/π)·asin ρ. El retorno del bloque siguiente es
//!   predecible con correlación √((1 − a)/2) > 0, pero la señal de momento
//!   (desviación respecto de la EWMA) lo predice con signo negativo.
//! - Caminata aleatoria gaussiana: la señal publicada al abrir un bloque es
//!   función del pasado y el retorno del bloque es un incremento futuro
//!   independiente, así que el IC verdadero es 0 en toda escala y ninguna
//!   escala tiene habilidad que validar.
//!
//! Con estos valores se miden la persistencia CL-30, el IC prequential por
//! escala, la fusión, la τ dominante y la τ operativa unificada.
//!
//! Convención: un test que documenta un defecto confirmado lleva
//! `#[ignore = "OPEN F2.x-nn (dueño): causa"]` con el assert intacto. Falla
//! hoy por la causa del hallazgo y se activa (se quita el `ignore`) con el
//! arreglo. Los tests sin `ignore` son regresiones de la teoría y deben pasar.

use quantum_arena::evalues::EProceso;
use quantum_arena::temporal_spectrum::{TemporalSpectrum, SPECTRUM_SCALES_MS};

struct Lcg(u64);
impl Lcg {
    fn u(&mut self) -> f64 {
        self.0 = self
            .0
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        ((self.0 >> 11) as f64 + 0.5) / (1u64 << 53) as f64
    }
    fn n(&mut self) -> f64 {
        let (u1, u2) = (self.u(), self.u());
        (-2.0 * u1.ln()).sqrt() * (std::f64::consts::TAU * u2).cos()
    }
}

const T0: u64 = 1_700_000_000_000;
/// Un evento por segundo de reloj del exchange.
const DT: u64 = 1_000;
const P0: f64 = 100.0;

/// OU exacto discreto en log-precio, vida media `hl_s` y desviación
/// estacionaria `s`, arrancado en la distribución estacionaria.
fn ou(seed: u64, n: usize, hl_s: f64, s: f64) -> Vec<f64> {
    let mut g = Lcg(seed);
    let a1 = (-(std::f64::consts::LN_2 / hl_s) * (DT as f64 / 1000.0)).exp();
    let k = s * (1.0 - a1 * a1).sqrt();
    let mut x = s * g.n();
    (0..n)
        .map(|_| {
            x = a1 * x + k * g.n();
            x
        })
        .collect()
}

/// Caminata aleatoria gaussiana en log-precio.
fn rw(seed: u64, n: usize, sd: f64) -> Vec<f64> {
    let mut g = Lcg(seed);
    let mut x = 0.0;
    (0..n)
        .map(|_| {
            x += sd * g.n();
            x
        })
        .collect()
}

/// Duración del bloque CL-30 de la escala: el primer evento con
/// ts − t0 ≥ τ (69 s para la escala 18 a 1 Hz).
fn bloque_s(escala: usize) -> f64 {
    (SPECTRUM_SCALES_MS[escala] / DT as f64).ceil() * DT as f64 / 1000.0
}

/// CL-30 en OU: bloques adyacentes (el que cierra arma el siguiente en el
/// mismo evento), ρ = −(1 − a)/2 y Sheppard (2/π)·asin ρ.
fn persistencia_esperada_ou(hl_s: f64, escala: usize) -> f64 {
    let a = (-(std::f64::consts::LN_2 / hl_s) * bloque_s(escala)).exp();
    std::f64::consts::FRAC_2_PI * (-(1.0 - a) / 2.0).asin()
}

fn alimentar(spec: &mut TemporalSpectrum, xs: &[f64], i: usize) {
    spec.update(P0 * xs[i].exp(), T0 + i as u64 * DT);
}

// Valor verdadero (Sheppard sobre OU, bloques de 18, 69 y 275 s): −0,1088,
// −0,2609 y −0,3327. Media temporal desde el evento 14 000, con ≈ 3 220, 840 y
// 210 bloques. Tolerancia 4·√((1 − p²)/m) = 0,070, 0,133 y 0,26; el SE por
// camino de una simulación exacta a nivel de bloque (4 000 caminos) es 0,017,
// 0,030 y 0,058, así que son ≥ 4 SE.
// Además se recupera la vida media invirtiendo la forma cerrada en las escalas
// 17 y 18 (la inversión sólo es válida con p ∈ (−1/3, 0)):
// θ̂_L = −ln(1 + 2·sin(π·p/2))/L, combinadas con pesos 1/Var del método delta
// evaluados en la p verdadera. SE(θ̂) ≈ 0,0038 s⁻¹ frente a θ = 0,0231 s⁻¹;
// se exige |θ̂ − ln2/30| < 4 SE (vida media entre ≈ 18 s y 88 s).
#[test]
fn cf2_ou_persistencia_sigue_la_forma_cerrada() {
    let hl = 30.0f64;
    let xs = ou(0xC0F2_0001, 72_000, hl, 0.002);
    let mut spec = TemporalSpectrum::new();
    let escalas = [17usize, 18, 19];
    let mut acum = [0.0f64; 3];
    let mut n = 0usize;
    let desde = 14_000usize;
    for i in 0..xs.len() {
        alimentar(&mut spec, &xs, i);
        if i >= desde {
            for (j, &k) in escalas.iter().enumerate() {
                acum[j] += spec.scales[k].persistence;
            }
            n += 1;
        }
    }
    let medidas: Vec<f64> = acum.iter().map(|a| a / n as f64).collect();
    for (j, &k) in escalas.iter().enumerate() {
        let esperada = persistencia_esperada_ou(hl, k);
        let bloques = (xs.len() - desde) as f64 / bloque_s(k);
        let se = ((1.0 - esperada * esperada) / bloques).sqrt();
        let medida = medidas[j];
        assert!(
            (medida - esperada).abs() < 4.0 * se,
            "escala {k}: persistencia medida {medida:.4}, forma cerrada {esperada:.4}, 4 SE = {:.4}",
            4.0 * se
        );
    }

    // Vida media recuperada (escalas 17 y 18).
    let theta = std::f64::consts::LN_2 / hl;
    let mut num = 0.0f64;
    let mut den = 0.0f64;
    let mut thetas = Vec::new();
    for (j, &k) in escalas.iter().take(2).enumerate() {
        let l = bloque_s(k);
        let p_verdad = persistencia_esperada_ou(hl, k);
        let bloques = (xs.len() - desde) as f64 / l;
        let se_p = ((1.0 - p_verdad * p_verdad) / bloques).sqrt();
        let rho_v = (std::f64::consts::FRAC_PI_2 * p_verdad).sin();
        let deriv = std::f64::consts::PI * (std::f64::consts::FRAC_PI_2 * p_verdad).cos()
            / (l * (1.0 + 2.0 * rho_v));
        let var_theta = (deriv * se_p).powi(2);
        let base = 1.0 + 2.0 * (std::f64::consts::FRAC_PI_2 * medidas[j]).sin();
        assert!(
            base > 0.0,
            "escala {k}: persistencia {:.4} fuera del dominio de inversión (−1/3, 0)",
            medidas[j]
        );
        let theta_k = -base.ln() / l;
        thetas.push(theta_k);
        num += theta_k / var_theta;
        den += 1.0 / var_theta;
    }
    let theta_hat = num / den;
    let se_theta = (1.0 / den).sqrt();
    assert!(
        (theta_hat - theta).abs() < 4.0 * se_theta,
        "θ̂ = {theta_hat:.5} s⁻¹ (vida media {:.1} s; por escala {thetas:?}), verdad {theta:.5} (30 s), 4 SE = {:.5}",
        std::f64::consts::LN_2 / theta_hat,
        4.0 * se_theta
    );
    println!(
        "persistencias {medidas:?}; θ̂ = {theta_hat:.5} (vida media {:.1} s)",
        std::f64::consts::LN_2 / theta_hat
    );
}

// Valor verdadero: en OU la desviación del precio respecto de su EWMA a la
// escala τ tiene correlación −√((1−a)/2)·√(θτ/(1+θτ)) = −0,4945 con el retorno
// del bloque de 69 s, y corr(tanh(cZ), Z) = κ ≈ 0,950; con el factor ≈ 0,977
// de que s₀ es la señal del evento anterior, IC ≈ −0,46. La media temporal de
// ≈ 840 bloques tiene SE ≈ 0,03; la ventana (−0,65, −0,30) deja > 5 SE a cada
// lado y cubre la incertidumbre de κ. El signo negativo es analítico: el
// momento predice al revés en un proceso con reversión (el «OU ⇒ IC > 0» del
// plan no vale para la señal de momento).
#[test]
fn cf2_ou_ic_de_la_senal_de_momento_es_negativo() {
    let xs = ou(0xC0F2_0002, 72_000, 30.0, 0.002);
    let mut spec = TemporalSpectrum::new();
    let (mut suma, mut n) = (0.0f64, 0usize);
    for i in 0..xs.len() {
        alimentar(&mut spec, &xs, i);
        if i >= 14_000 {
            if let Some(ic) = spec.scales[18].habilidad_medida() {
                suma += ic;
                n += 1;
            }
        }
    }
    assert!(n > 50_000, "la escala 18 no maduró ({n})");
    let ic = suma / n as f64;
    println!("IC medio del momento a 69 s en OU: {ic:.4} (n = {n})");
    assert!(
        ic < -0.30 && ic > -0.65,
        "IC del momento en OU a 69 s = {ic:.3}; forma cerrada ≈ −0,46"
    );
}

// Valor verdadero: IC = 0 en toda escala (E[s₀·r] = 0 exacto en caminata
// aleatoria). Media temporal desde el evento 7 200 con ≈ 15 000, 3 600 y 939
// bloques en las escalas 16, 17 y 18: SE ≈ 0,009, 0,017 y 0,033. Tolerancias
// a ≥ 4 SE: 0,10, 0,10 y 0,14.
#[test]
fn cf2_rw_ic_por_escala_es_nulo() {
    let xs = rw(0xC0F2_0003, 72_000, 1e-4);
    let mut spec = TemporalSpectrum::new();
    let escalas = [16usize, 17, 18];
    let tol = [0.10f64, 0.10, 0.14];
    let mut suma = [0.0f64; 3];
    let mut n = [0usize; 3];
    for i in 0..xs.len() {
        alimentar(&mut spec, &xs, i);
        if i >= 7_200 {
            for (j, &k) in escalas.iter().enumerate() {
                if let Some(ic) = spec.scales[k].habilidad_medida() {
                    suma[j] += ic;
                    n[j] += 1;
                }
            }
        }
    }
    let mut medias = [0.0f64; 3];
    for (j, &k) in escalas.iter().enumerate() {
        assert!(n[j] > 30_000, "escala {k} sin IC maduro ({})", n[j]);
        let ic = suma[j] / n[j] as f64;
        medias[j] = ic;
        assert!(
            ic.abs() < tol[j],
            "escala {k}: IC medio {ic:.3} en caminata aleatoria; verdad 0, tolerancia {}",
            tol[j]
        );
    }
    println!("IC medios en caminata aleatoria (escalas 16, 17, 18): {medias:?}");
}

// Contrato (fila 1 del plan): un OU es predecible a 69 s con correlación
// √((1−a)/2) = 0,63 > 0 (mejor predictor lineal −(1−a)·x, a = 0,203); una
// fusión que se presenta como pronóstico direccional y firma cada escala por
// su habilidad recupera IC > 0. Se exige IC > 3/√m (3 SE bajo H0 con
// m ≈ 939 pares no solapados) y basta con el signo: la magnitud de hoy no
// está verificada en forma cerrada.
// Hoy: fused_score suma señales de momento con pesos 1/vol sin el signo de la
// habilidad, y todas las escalas maduras tienen IC < 0 en OU.
#[test]
#[ignore = "OPEN F2.3-03 (Línea C, C-F3 fusión): la fusión ignora el signo de la habilidad; en OU su IC es negativo"]
fn cf2_ou_la_fusion_firmada_predice_el_bloque_siguiente() {
    let xs = ou(0xC0F2_0004, 72_000, 30.0, 0.002);
    let mut spec = TemporalSpectrum::new();
    let h = 69usize; // bloque de la escala 18 a 1 Hz
    let (mut sfr, mut sff, mut srr, mut m) = (0.0f64, 0.0f64, 0.0f64, 0usize);
    for i in 0..xs.len() - h {
        alimentar(&mut spec, &xs, i);
        if i >= 7_200 && (i - 7_200) % h == 0 {
            let f = spec.fused_score;
            let r = xs[i + h] - xs[i];
            sfr += f * r;
            sff += f * f;
            srr += r * r;
            m += 1;
        }
    }
    let ic = sfr / (sff * srr).sqrt();
    let umbral = 3.0 / (m as f64).sqrt();
    assert!(
        ic > umbral,
        "IC de la fusión en OU = {ic:.3} con m = {m}; se exige > {umbral:.3} (predictibilidad conocida 0,63)"
    );
}

// Contrato de diseño del siguiente paso de la fila 1 (corrección del
// escéptico REFUTAR: no es una solución conocida ni un contrato vigente). Si la
// meta es un único vector de pesos, la «coherencia» que lee el consejo
// (spectral_coherence, pesos observables con 1/σ_dev) y la del campo
// (spectral_field().global_coherence, pesos espectrales sin 1/σ_dev y con
// |persistencia|) deben coincidir: |Δ| < 1e-9 en todo evento, sin tolerancia
// estadística. Hoy difieren aunque todas las |persistencias| fueran iguales,
// porque sólo uno de los dos pesos divide por la vol de desviación.
#[test]
#[ignore = "OPEN F2.3-04 (Línea C, C-F3): dos vectores de pesos (1/σ_dev frente a |persistencia|) bajo el mismo nombre"]
fn cf2_contrato_peso_unico_fila1() {
    let xs = rw(0xC0F2_0005, 21_600, 1e-4);
    let mut spec = TemporalSpectrum::new();
    let mut peor = 0.0f64;
    let mut eventos = 0usize;
    let mut distintos = 0usize;
    for i in 0..xs.len() {
        alimentar(&mut spec, &xs, i);
        if i >= 3_600 {
            let consejo = spec.spectral_coherence(true);
            let campo = spec.spectral_field(true).global_coherence;
            let d = (consejo - campo).abs();
            peor = peor.max(d);
            eventos += 1;
            if d >= 1e-9 {
                distintos += 1;
            }
        }
    }
    assert!(
        peor < 1e-9,
        "dos spectral_coherence distintas: |Δ| máx = {peor:.4}, distintas en {distintos}/{eventos} eventos"
    );
}

// Valor verdadero: en caminata aleatoria ninguna escala tiene habilidad. Si la
// τ unificada sólo deja el centroide cuando una escala tiene habilidad
// validada por Ville a α = 0,05, el número de semillas X con τ unificada ≠
// centroide es como mucho Binomial(20; 0,05): se exige X ≤ 3
// (P(X ≥ 4) = 0,016).
// Hoy: tau_operativa_unificada sólo exige habilidad_en(dominante) > 0. Sin
// habilidad validada la dominante es el respaldo argmax |w·s| en la banda
// [30 s, 12 h], normalmente la escala 18 (68,7 s) y a veces la 19, con IC
// maduro de signo aleatorio en 4 h: X ≈ Binomial(20; ≈ 0,5).
#[test]
#[ignore = "OPEN F2.3-01 (Línea C, C-F3 τ dominante): la τ unificada sale del centroide con el signo de un IC sin validar"]
fn cf2_rw_sin_habilidad_validada_la_tau_unificada_es_el_centroide() {
    let mut discrepancias = 0usize;
    let mut detalle = Vec::new();
    for k in 0..20u64 {
        let xs = rw(0xC0F2_0600 + k, 14_400, 1e-4);
        let mut spec = TemporalSpectrum::new();
        for i in 0..xs.len() {
            alimentar(&mut spec, &xs, i);
        }
        let unificada = spec.tau_operativa_unificada();
        let centroide = spec.continuous_resonant_tau_ms();
        if (unificada - centroide).abs() > 1e-9 * centroide.abs().max(1.0) {
            discrepancias += 1;
            detalle.push((
                k,
                unificada / 1000.0,
                spec.habilidad_en(spec.dominant_tau_ms),
            ));
        }
    }
    // Binomial(20, 0,05): P(X ≥ 4) ≈ 0,016.
    assert!(
        discrepancias <= 3,
        "{discrepancias}/20 semillas usan una τ sin habilidad validada; (semilla, τ unificada en s, IC en la dominante): {detalle:?}"
    );
}

/// Validación bilateral reconstruida en el test con la API pública: por cada
/// cierre de bloque de la escala se forma el par (s₀, r) que puntúa el
/// espectro —s₀ es la señal publicada antes del evento que armó el bloque y r
/// el retorno que expone `ultimo_bloque_maduro`— y se alimentan dos
/// e-procesos de Ville, uno por cada signo del IC. La réplica del IC con
/// olvido 1/64 se compara con `habilidad_medida` para asegurar que el par es
/// el mismo que usa el código.
struct ValidacionBilateral {
    a_favor: EProceso,
    en_contra: EProceso,
    s0: f64,
    ws: f64,
    wr: f64,
    wsr: f64,
    n: u64,
    validada: bool,
}

impl ValidacionBilateral {
    fn new() -> Self {
        Self {
            a_favor: EProceso::new(),
            en_contra: EProceso::new(),
            s0: 0.0,
            ws: 0.0,
            wr: 0.0,
            wsr: 0.0,
            n: 0,
            validada: false,
        }
    }
    fn ic_replica(&self) -> Option<f64> {
        if self.n < 30 {
            return None;
        }
        let den = self.ws * self.wr;
        if !(den.is_finite() && den > 0.0) {
            return None;
        }
        Some((self.wsr / den.sqrt()).clamp(-1.0, 1.0))
    }
}

/// Escalas de la banda operable [30 s, 12 h] y familia bilateral (2 signos ×
/// 5 escalas) del umbral de Ville M/α.
const BANDA: std::ops::RangeInclusive<usize> = 18..=22;
const FAMILIA_BILATERAL: usize = 2 * 5;

// Valor verdadero: en OU (vida media 30 s) todas las escalas maduras tienen
// |IC| grande (≈ 0,47 a 69 s y ≈ 0,62 a 275 s) con signo negativo. Corrección
// de los escépticos: el oráculo ya no es el argmax de |IC| sobre IC ruidosos
// sin validar, sino el argmax de |IC| ENTRE las escalas cuya habilidad está
// validada de forma bilateral por Ville (familia 2 × 5, umbral 200), que es
// lo que haría la τ dominante arreglada. La escala 18 se valida en ≈ 3,6 h y
// la 19 en ≈ 10 h; las 20-22 no llegan en 12 h. Cada 600 eventos desde las 6 h
// se exige dominant_tau_ms = τ del argmax en ≥ 90 % de las comprobaciones con
// alguna escala validada (≥ 30 casos de 36). Se informa también la pertenencia
// de la dominante al conjunto validado (contrato (a) de COMPLETITUD).
// Hoy: la selección sólo admite IC > 0 y todas son negativas, así que manda el
// respaldo argmax |w·s|, que mide amplitud y no habilidad.
#[test]
#[ignore = "OPEN F2.3-03 (Línea C, C-F3): la τ dominante sólo admite IC > 0; en OU cae al respaldo de amplitud"]
fn cf2_ou_tau_dominante_sigue_la_habilidad_absoluta() {
    let xs = ou(0xC0F2_0007, 43_200, 30.0, 0.002);
    let mut spec = TemporalSpectrum::new();
    let mut val: Vec<ValidacionBilateral> = (0..32).map(|_| ValidacionBilateral::new()).collect();
    let (mut casos, mut aciertos, mut pertenece) = (0usize, 0usize, 0usize);
    let mut primera_validacion = [None::<usize>; 32];
    let mut elegidas = Vec::new();
    for i in 0..xs.len() {
        let previas: Vec<f64> = BANDA.map(|k| spec.scales[k].signal).collect();
        let ts = T0 + i as u64 * DT;
        alimentar(&mut spec, &xs, i);
        for (j, k) in BANDA.enumerate() {
            if let Some((t, r)) = spec.ultimo_bloque_maduro(k) {
                if t == ts {
                    let v = &mut val[k];
                    let s0 = v.s0;
                    v.a_favor.observar(s0, r);
                    v.en_contra.observar(-s0, r);
                    v.ws += (s0 * s0 - v.ws) / 64.0;
                    v.wr += (r * r - v.wr) / 64.0;
                    v.wsr += (s0 * r - v.wsr) / 64.0;
                    v.n += 1;
                    if !v.validada
                        && (v.a_favor.significativo_familia(FAMILIA_BILATERAL)
                            || v.en_contra.significativo_familia(FAMILIA_BILATERAL))
                    {
                        // Tiempo de parada de Ville: validada desde el primer cruce.
                        v.validada = true;
                        primera_validacion[k] = Some(i);
                    }
                    // El bloque que nace ahora se puntuará con la señal previa.
                    v.s0 = previas[j];
                    // La réplica debe ser el IC del propio espectro.
                    let replica = v.ic_replica();
                    let codigo = spec.scales[k].habilidad_medida();
                    match (replica, codigo) {
                        (Some(a), Some(b)) => assert!(
                            (a - b).abs() < 1e-9,
                            "réplica del IC de la escala {k} = {a}, código = {b}"
                        ),
                        (None, None) => {}
                        otro => panic!(
                            "réplica y código discrepan en madurez en la escala {k}: {otro:?}"
                        ),
                    }
                }
            }
        }
        if i >= 21_600 && i % 600 == 0 {
            let mut mejor: Option<(usize, f64)> = None;
            let mut validadas = Vec::new();
            for k in BANDA {
                if !val[k].validada {
                    continue;
                }
                if let Some(ic) = spec.scales[k].habilidad_medida() {
                    validadas.push(k);
                    if mejor.map_or(true, |(_, b)| ic.abs() > b) {
                        mejor = Some((k, ic.abs()));
                    }
                }
            }
            if let Some((k, _)) = mejor {
                casos += 1;
                let dom = spec.dominant_tau_ms;
                let tau_k = SPECTRUM_SCALES_MS[k];
                if (dom - tau_k).abs() < 1e-6 * tau_k {
                    aciertos += 1;
                }
                if validadas
                    .iter()
                    .any(|&v| (dom - SPECTRUM_SCALES_MS[v]).abs() < 1e-6 * SPECTRUM_SCALES_MS[v])
                {
                    pertenece += 1;
                }
                elegidas.push((i, k, (dom / 1000.0).round()));
            }
        }
    }
    let validacion_h: Vec<(usize, f64)> = BANDA
        .filter_map(|k| primera_validacion[k].map(|i| (k, i as f64 / 3600.0)))
        .collect();
    assert!(
        casos >= 30,
        "pocos casos evaluables: {casos}; validación (escala, h): {validacion_h:?}"
    );
    assert!(
        aciertos as f64 >= 0.9 * casos as f64,
        "τ dominante = argmax |IC| validado sólo en {aciertos}/{casos}; dentro del conjunto validado en {pertenece}/{casos}; validación (escala, h): {validacion_h:?}; (evento, escala oráculo, dominante en s): {elegidas:?}"
    );
}
