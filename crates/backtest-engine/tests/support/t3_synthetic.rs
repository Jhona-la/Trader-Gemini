//! Synthetic T-3 model and reference readers, shared with the integration test.
//! Can be checked without Cargo: rustc --edition=2024 --test this_file -o tests.exe

/// Fricción de ida y vuelta de referencia para los techos: dos piernas taker
/// del spec del fixture (0,05 % cada una). El motor paga además la horquilla
/// del genoma; el techo es por tanto optimista en ese término.
pub const FRICCION_RT: f64 = 2.0 * 0.0005;

/// Geometría de referencia del bracket de los techos (RR 2,25, stop 0,5 %).
pub const SL_REF: f64 = 0.005;
pub const RR_REF: f64 = 2.25;

/// Un peldaño de la escalera.
#[derive(Clone, Copy, Debug)]
pub struct Peldano {
    /// Persistencia del signo: P(s_t = s_{t−1}).
    pub a: f64,
    /// Deriva por vela en fracción (0,001 = 0,1 %).
    pub d: f64,
    /// Desviación del ruido por vela en fracción.
    pub sigma: f64,
    /// Velas de 1 min.
    pub n: usize,
}

fn validar_parametros(p: Peldano) {
    assert!(
        p.a.is_finite() && (0.0..=1.0).contains(&p.a),
        "a must be a probability"
    );
    assert!(
        p.d.is_finite() && p.d >= 0.0,
        "d must be finite and nonnegative"
    );
    assert!(
        p.sigma.is_finite() && p.sigma > 0.0,
        "sigma must be finite and positive"
    );
}

#[test]
#[should_panic(expected = "sigma must be finite and positive")]
fn reject_degenerate_noise_instead_of_nan_posterior() {
    serie_persistente_v2(Peldano {
        a: 0.9,
        d: 0.0,
        sigma: 0.0,
        n: 2,
    });
}

#[test]
fn absent_followup_is_not_perfect_calibration_or_zero_participation() {
    let p = Peldano {
        a: 0.9,
        d: 0.002,
        sigma: 0.0025,
        n: 1,
    };
    let mut s = serie_persistente_v2(p);
    let b = serie_persistente_v2(p);
    s.closes.extend(b.closes);
    s.highs.extend(b.highs);
    s.lows.extend(b.lows);
    s.vols.extend(b.vols);
    s.estado.extend(b.estado);
    s.tramos.push((1, p));
    let t = techo(&s);
    assert!(t.brier_filtro.is_nan());
    assert!(t.cobertura_neta.iter().all(|v| v.is_nan()));
}

/// Serie sintética con su estado oculto.
pub struct Serie {
    pub closes: Vec<f64>,
    pub highs: Vec<f64>,
    pub lows: Vec<f64>,
    pub vols: Vec<f64>,
    /// Estado oculto por vela (+1/−1).
    pub estado: Vec<i8>,
    /// Parámetros con los que se generó (los del PRIMER tramo si es compuesta).
    pub peldano: Peldano,
    pub generador: Generador,
    /// Tramos (índice de la primera vela, parámetros): uno en una serie
    /// simple, dos en una compuesta. El filtro del techo conoce los de cada
    /// tramo y se reinicia en la unión, como el generador (M-04).
    pub tramos: Vec<(usize, Peldano)>,
}

/// Sal del generador: a en los bits 0..19, d·10⁸ en 20..37, n desde 44; la
/// v2 invierte el bit 40, libre (G-05), así que ninguna serie v2 comparte
/// semilla con una v1 de cualquier peldaño de la escalera.
pub fn sal_de(p: Peldano, g: Generador) -> u64 {
    let base = ((p.a * 1e6) as u64) ^ (((p.d * 1e8) as u64) << 20) ^ ((p.n as u64) << 44);
    match g {
        Generador::V1Uniforme => base,
        Generador::V2Gauss => base ^ (1u64 << 40),
    }
}

/// Versión del generador.
///
/// * `V1Uniforme` (olas 1 y 2): ruido uniforme acotado y mechas y volumen
///   construidos con el mismo |u| que el ruido. El estado oculto se
///   decodifica del OHLCV (H1-01 de la Línea G): se conserva sólo para
///   reproducir las tablas publicadas.
/// * `V2Gauss` (ola 3): ruido gaussiano (Box–Muller sobre el mismo LCG) y
///   sorteos INDEPENDIENTES para las dos mechas (half-normal de escala σ
///   sobre el cuerpo de la vela) y el volumen (lognormal). Ningún dato de la
///   vela revela el estado con certeza; lo que se puede saber de él lo dice
///   el filtro de Bayes ([`filtro_bayes`]).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Generador {
    V1Uniforme,
    V2Gauss,
}

impl Generador {
    pub fn etiqueta(self) -> &'static str {
        match self {
            Generador::V1Uniforme => "v1",
            Generador::V2Gauss => "v2",
        }
    }
}

/// `T3_GENERADOR` = `v1` (por defecto: reproduce las tablas de las olas 1 y 2)
/// o `v2` (ola 3). Lo leen todas las mediciones.
pub fn generador_env() -> Generador {
    match std::env::var("T3_GENERADOR").ok().as_deref().map(str::trim) {
        Some("v2") | Some("V2") => Generador::V2Gauss,
        _ => Generador::V1Uniforme,
    }
}

/// LCG de 64 bits (misma familia que el T-2) con normales por Box–Muller.
struct Lcg {
    seed: u64,
    cache: Option<f64>,
}

impl Lcg {
    fn nuevo(sal: u64) -> Self {
        Self {
            seed: 0x2545F4914F6CDD1Du64 ^ sal.wrapping_mul(0x9E3779B97F4A7C15),
            cache: None,
        }
    }

    /// Uniforme en [0, 1).
    fn uniforme(&mut self) -> f64 {
        self.seed = self
            .seed
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        (self.seed >> 11) as f64 / (1u64 << 53) as f64
    }

    /// N(0, 1) por Box–Muller: dos uniformes dan dos normales independientes;
    /// la segunda se guarda para la llamada siguiente.
    fn normal(&mut self) -> f64 {
        if let Some(z) = self.cache.take() {
            return z;
        }
        // 1 − u ∈ (0, 1]: nunca ln 0.
        let u1 = 1.0 - self.uniforme();
        let u2 = self.uniforme();
        let radio = (-2.0 * u1.ln()).sqrt();
        let (seno, coseno) = (2.0 * std::f64::consts::PI * u2).sin_cos();
        self.cache = Some(radio * seno);
        radio * coseno
    }
}

/// Serie del peldaño con el generador pedido.
pub fn serie_de(p: Peldano, g: Generador) -> Serie {
    match g {
        Generador::V1Uniforme => serie_persistente(p),
        Generador::V2Gauss => serie_persistente_v2(p),
    }
}

/// Generador v2 (ola 3). La cadena oculta y la deriva son las de la v1; el
/// ruido del cierre es N(0, σ²) y las mechas y el volumen salen de sorteos
/// propios, independientes del ruido y del estado. La sal lleva un bit de
/// versión: una serie v2 nunca comparte el ruido con la v1 del mismo peldaño.
pub fn serie_persistente_v2(p: Peldano) -> Serie {
    validar_parametros(p);
    let n = p.n;
    let mut closes = Vec::with_capacity(n);
    let mut highs = Vec::with_capacity(n);
    let mut lows = Vec::with_capacity(n);
    let mut vols = Vec::with_capacity(n);
    let mut estado = Vec::with_capacity(n);
    let mut rng = Lcg::nuevo(sal_de(p, Generador::V2Gauss));
    let mut ln_p = 60_000.0f64.ln();
    let mut s: i8 = 1;
    for _ in 0..n {
        if rng.uniforme() > p.a {
            s = -s;
        }
        let apertura = ln_p.exp();
        ln_p += f64::from(s) * p.d + p.sigma * rng.normal();
        let cierre = ln_p.exp();
        let mecha_alta = rng.normal().abs() * p.sigma;
        let mecha_baja = rng.normal().abs() * p.sigma;
        highs.push(apertura.max(cierre) * (1.0 + mecha_alta));
        lows.push(apertura.min(cierre) * (1.0 - mecha_baja));
        vols.push(1_400.0 * (0.30 * rng.normal()).exp());
        closes.push(cierre);
        estado.push(s);
    }
    Serie {
        closes,
        highs,
        lows,
        vols,
        estado,
        peldano: p,
        generador: Generador::V2Gauss,
        tramos: vec![(0, p)],
    }
}

/// Generador v1 (olas 1 y 2), determinista (LCG de 64 bits, misma familia que
/// el T-2) con una sal por peldaño para que dos peldaños distintos no
/// compartan el ruido. Se conserva bit a bit para reproducir las tablas
/// publicadas; ver [`Generador`] por su límite (H1-01).
pub fn serie_persistente(p: Peldano) -> Serie {
    validar_parametros(p);
    let n = p.n;
    let mut closes = Vec::with_capacity(n);
    let mut highs = Vec::with_capacity(n);
    let mut lows = Vec::with_capacity(n);
    let mut vols = Vec::with_capacity(n);
    let mut estado = Vec::with_capacity(n);
    let sal = ((p.a * 1e6) as u64) ^ (((p.d * 1e8) as u64) << 20) ^ ((n as u64) << 44);
    let mut seed = 0x2545F4914F6CDD1Du64 ^ sal.wrapping_mul(0x9E3779B97F4A7C15);
    let mut uniforme = move || {
        seed = seed
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        (seed >> 11) as f64 / (1u64 << 53) as f64
    };
    // Uniforme centrada en (−0,5, 0,5): σ = 1/√12.
    let escala_ruido = p.sigma / (1.0 / 12f64.sqrt());
    let mut ln_p = 60_000.0f64.ln();
    let mut s: i8 = 1;
    for _ in 0..n {
        if uniforme() > p.a {
            s = -s;
        }
        let u = uniforme() - 0.5;
        let ruido = u * escala_ruido;
        ln_p += f64::from(s) * p.d + ruido;
        let px = ln_p.exp();
        closes.push(px);
        highs.push(px * (1.0 + 0.0015 + u.abs() * 0.0020));
        lows.push(px * (1.0 - 0.0015 - u.abs() * 0.0020));
        vols.push(900.0 + u.abs() * 2_000.0);
        estado.push(s);
    }
    Serie {
        closes,
        highs,
        lows,
        vols,
        estado,
        peldano: p,
        generador: Generador::V1Uniforme,
        tramos: vec![(0, p)],
    }
}

/// Reference-reader diagnostics, without the engine. The legacy type name
/// `Techo` is retained for callers; none of these columns is a universal
/// upper bound for a selective strategy with a different holding policy.
#[derive(Clone, Copy, Debug, Default)]
pub struct Techo {
    /// Persistencia medida del estado oculto.
    pub a_medida: f64,
    /// P(signo del retorno de la vela = estado oculto): cuánto deja ver el ruido.
    pub lectura: f64,
    /// Retorno medio neto de fricción a 1, 5 y 20 velas siguiendo al oráculo
    /// (privileged latent-state reader, with the expected future direction).
    pub oraculo: [f64; 3],
    /// Lo mismo siguiendo al signo de la vela anterior (el proxy del flujo).
    pub proxy: [f64; 3],
    /// Always-active direction from the close-only HMM posterior. Known
    /// DGP parameters and segment schedule are diagnostic privileged inputs.
    pub filtro: [f64; 3],
    pub filtro_neto: [f64; 3],
    pub cobertura_neta: [f64; 3],
    /// Average R across all complete reference brackets, including expirations.
    pub bracket_ev_r: [f64; 3],
    pub bracket_vencidos: [f64; 3],
    /// P(TP antes que SL) del bracket de referencia siguiendo al oráculo.
    pub p_tp_oraculo: f64,
    /// P(TP antes que SL) siguiendo al proxy.
    pub p_tp_proxy: f64,
    /// P(TP antes que SL) siguiendo al filtro.
    pub p_tp_filtro: f64,
    /// Fracción de brackets que vencen (ni TP ni SL en 60 velas), oráculo.
    pub vencidos_oraculo: f64,
    /// Puntuación de Brier de la p del filtro para el signo del estado
    /// siguiente (0,25 = no sabe nada; 0 = lo sabe todo).
    pub brier_filtro: f64,
    /// Media de |2p − 1| del filtro: cuánto se atreve a decidir.
    pub decision_filtro: f64,
    /// Fracción de velas que, SOLAS, decodifican el estado (una verosimilitud
    /// vale 0): la fuga H1-01 del soporte acotado de la v1; 0 en la v2.
    pub decodificable: f64,
    /// Fracción de velas en que el posterior acumulado queda seguro
    /// (≥ 0,999 o ≤ 0,001). In v1 this can result from bounded-support leakage.
    pub seguro: f64,
}

/// Filtro de Bayes: el paso hacia delante del HMM de dos estados sobre los
/// retornos de cierre. Conoce (a, d, σ) y la familia del ruido del generador
/// (la del [`Generador`] de la serie). This is a posterior for closed-bar
/// returns under that DGP, not a bound on the engine's PnL. V1 also leaks
/// information through joint OHLCV observations not consumed by this filter.
/// The OHLC replay bridge remains noncausal within each bar (H1-01 iii).
pub fn filtro_bayes(serie: &Serie) -> Filtro {
    let n = serie.closes.len();
    let mut siguiente = Vec::with_capacity(n);
    let mut posterior = Vec::with_capacity(n);
    let mut excluye = Vec::with_capacity(n);
    let mut parametros = Vec::with_capacity(n);
    let veros = |p: Peldano, r: f64, s: f64| -> f64 {
        match serie.generador {
            Generador::V2Gauss => {
                let z = (r - s * p.d) / p.sigma.max(1e-12);
                (-0.5 * z * z).exp()
            }
            Generador::V1Uniforme => {
                let semi = 0.5 * p.sigma * 12f64.sqrt();
                if (r - s * p.d).abs() <= semi + 1e-12 {
                    1.0
                } else {
                    0.0
                }
            }
        }
    };
    for (i, &(inicio, p)) in serie.tramos.iter().enumerate() {
        validar_parametros(p);
        let fin = serie.tramos.get(i + 1).map(|t| t.0).unwrap_or(n).min(n);
        // Cada tramo parte de s = +1 ANTES de su primer sorteo de cambio
        // (así arranca el generador y así continúa la serie compuesta).
        let mut pi = 1.0f64;
        let mut prev = if inicio == 0 {
            60_000.0f64.ln()
        } else {
            serie.closes[inicio - 1].ln()
        };
        for t in inicio..fin {
            let ln_c = serie.closes[t].ln();
            let r = ln_c - prev;
            prev = ln_c;
            let pred = p.a * pi + (1.0 - p.a) * (1.0 - pi);
            let (vp, vm) = (veros(p, r, 1.0), veros(p, r, -1.0));
            excluye.push(serie.generador == Generador::V1Uniforme && ((vp == 0.0) != (vm == 0.0)));
            pi = if serie.generador == Generador::V2Gauss && pred > 0.0 && pred < 1.0 {
                let log_odds = pred.ln() - (-pred).ln_1p() + 2.0 * p.d * r / p.sigma.powi(2);
                if log_odds >= 0.0 {
                    1.0 / (1.0 + (-log_odds).exp())
                } else {
                    let e = log_odds.exp();
                    e / (1.0 + e)
                }
            } else {
                let lp = pred * vp;
                let lm = (1.0 - pred) * vm;
                if lp + lm > 0.0 { lp / (lp + lm) } else { pred }
            };
            posterior.push(pi);
            siguiente.push(p.a * pi + (1.0 - p.a) * (1.0 - pi));
            parametros.push(p);
        }
    }
    Filtro {
        siguiente,
        posterior,
        excluye,
        parametros,
    }
}

/// Salida del filtro, por vela t.
pub struct Filtro {
    /// P(s_{t+1} = +1 | r_1..r_t): lo que decide la operación siguiente.
    pub siguiente: Vec<f64>,
    /// P(s_t = +1 | r_1..r_t).
    pub posterior: Vec<f64>,
    /// La vela t SOLA excluye un estado (una verosimilitud vale 0): el canal
    /// (i) de H1-01, el soporte acotado; imposible con ruido gaussiano.
    pub excluye: Vec<bool>,
    /// Parámetros del tramo al que pertenece cada vela.
    pub parametros: Vec<Peldano>,
}

/// Esperanza BRUTA del retorno acumulado a `k` velas para quien cree que
/// P(s_{t+1} = +1) = q: d·(2q − 1)·Σ_{j<k} (2a − 1)^j.
pub fn esperanza_k(q: f64, p: Peldano, k: usize) -> f64 {
    let rho = 2.0 * p.a - 1.0;
    let geometrica: f64 = (0..k).map(|j| rho.powi(j as i32)).sum();
    p.d * (2.0 * q - 1.0) * geometrica
}

/// Dirección de un lector a `k` velas: el signo de su esperanza bruta
/// cuando ésta paga la fricción de ida y vuelta, y 0 (se abstiene) si no
/// (G-02: el techo neto es el del lector que no opera lo que no paga). Con
/// d = 0 gives zero expected return. q = 0.5 also abstains; a = 0.5 does
/// not force an arbitrary externally supplied q to equal 0.5.
pub fn direccion_neta(q: f64, p: Peldano, k: usize) -> f64 {
    let e = esperanza_k(q, p, k);
    if e.abs() <= FRICCION_RT {
        0.0
    } else {
        e.signum()
    }
}

/// Dirección «siempre dentro»: el signo de la esperanza bruta (0 sólo si es
/// exactamente nula). Es el óptimo por vela de quien opera todas las velas.
pub fn direccion_bruta(q: f64, p: Peldano, k: usize) -> f64 {
    let e = esperanza_k(q, p, k);
    if e.abs() < 1e-15 { 0.0 } else { e.signum() }
}

/// Retorno medio neto de fricción a `k` velas siguiendo a `direccion`, y la
/// fracción de velas en que el lector operó. Sin ninguna operación devuelve
/// NaN (no 0: «sin muestras» no es un retorno; M-11).
fn retorno_medio(serie: &Serie, direccion: impl Fn(usize) -> f64, k: usize) -> (f64, f64) {
    let n = serie.closes.len();
    if n <= k + 1 {
        return (f64::NAN, f64::NAN);
    }
    let mut suma = 0.0;
    let mut cuenta = 0usize;
    let mut elegibles = 0usize;
    for t in 1..(n - k) {
        if serie
            .tramos
            .iter()
            .any(|&(inicio, _)| inicio > t && inicio <= t + k)
        {
            continue;
        }
        elegibles += 1;
        let dir = direccion(t);
        if dir == 0.0 {
            continue;
        }
        suma += dir * (serie.closes[t + k].ln() - serie.closes[t].ln());
        cuenta += 1;
    }
    let opera = if elegibles > 0 {
        cuenta as f64 / elegibles as f64
    } else {
        f64::NAN
    };
    if cuenta == 0 {
        (f64::NAN, 0.0)
    } else {
        (suma / cuenta as f64 - FRICCION_RT, opera)
    }
}

fn retorno_neto(serie: &Serie, direccion: impl Fn(usize) -> f64, k: usize) -> f64 {
    retorno_medio(serie, direccion, k).0
}

/// Primer toque del bracket (SL_REF, RR_REF·SL_REF) abierto al cierre de `t`
/// en la dirección `dir`, resuelto con máximos y mínimos hasta `horizonte`
/// velas; devuelve (tp, sl, vencido).
fn primer_toque(serie: &Serie, t: usize, dir: f64, horizonte: usize) -> (bool, bool, bool) {
    let entrada = serie.closes[t];
    let tp = entrada * (1.0 + dir * RR_REF * SL_REF);
    let sl = entrada * (1.0 - dir * SL_REF);
    let n = serie.closes.len();
    for j in (t + 1)..(t + 1 + horizonte).min(n) {
        let (hi, lo) = (serie.highs[j], serie.lows[j]);
        let toca_sl = if dir > 0.0 { lo <= sl } else { hi >= sl };
        let toca_tp = if dir > 0.0 { hi >= tp } else { lo <= tp };
        if toca_sl {
            return (false, true, false);
        }
        if toca_tp {
            return (true, false, false);
        }
    }
    (false, false, true)
}

#[derive(Debug)]
struct Bracket {
    n: usize,
    p_tp_resuelto: f64,
    vencidos: f64,
    ev_r: f64,
}

/// Full follow-up only; crossing segment boundaries and right-censored
/// entries are excluded. Reference barrier fills assume exact barrier prices.
/// Expirations settle at the horizon close. Both barriers in a bar => SL first.
fn bracket(serie: &Serie, direccion: impl Fn(usize) -> f64, horizonte: usize) -> Bracket {
    let n = serie.closes.len();
    let (mut tps, mut sls, mut vencidos, mut total) = (0usize, 0usize, 0usize, 0usize);
    let mut suma_r = 0.0;
    for t in 1..n.saturating_sub(horizonte) {
        if serie
            .tramos
            .iter()
            .any(|&(inicio, _)| inicio > t && inicio <= t + horizonte)
        {
            continue;
        }
        let dir = direccion(t);
        if dir == 0.0 {
            continue;
        }
        let (tp, sl, v) = primer_toque(serie, t, dir, horizonte);
        tps += tp as usize;
        sls += sl as usize;
        vencidos += v as usize;
        total += 1;
        suma_r += if tp {
            RR_REF
        } else if sl {
            -1.0
        } else {
            dir * (serie.closes[t + horizonte] / serie.closes[t] - 1.0) / SL_REF
        } - FRICCION_RT / SL_REF;
    }
    let resueltos = tps + sls;
    Bracket {
        n: total,
        p_tp_resuelto: if resueltos > 0 {
            tps as f64 / resueltos as f64
        } else {
            f64::NAN
        },
        vencidos: if total > 0 {
            vencidos as f64 / total as f64
        } else {
            f64::NAN
        },
        ev_r: if total > 0 {
            suma_r / total as f64
        } else {
            f64::NAN
        },
    }
}

fn p_primer_toque(serie: &Serie, direccion: impl Fn(usize) -> f64) -> (f64, f64) {
    let b = bracket(serie, direccion, 60);
    (b.p_tp_resuelto, b.vencidos)
}

pub fn techo(serie: &Serie) -> Techo {
    let n = serie.closes.len();
    let mut iguales = 0usize;
    let mut lecturas = 0usize;
    for t in 1..n {
        iguales += (serie.estado[t] == serie.estado[t - 1]) as usize;
        let signo = (serie.closes[t] - serie.closes[t - 1]).signum();
        lecturas += (signo == f64::from(serie.estado[t])) as usize;
    }
    let oraculo = |t: usize| {
        let p = serie
            .tramos
            .iter()
            .rev()
            .find(|&&(inicio, _)| inicio <= t)
            .unwrap()
            .1;
        direccion_bruta(if serie.estado[t] > 0 { p.a } else { 1.0 - p.a }, p, 1)
    };
    let proxy = |t: usize| (serie.closes[t] - serie.closes[t - 1]).signum();
    let (p_tp_oraculo, vencidos_oraculo) = p_primer_toque(serie, oraculo);
    let (p_tp_proxy, _) = p_primer_toque(serie, proxy);
    let f = filtro_bayes(serie);
    let (q, posterior) = (&f.siguiente, &f.posterior);
    let params = &f.parametros;
    let filtro_k = |k: usize| move |t: usize| direccion_bruta(q[t], params[t], k);
    let filtro_neto_k = |k: usize| move |t: usize| direccion_neta(q[t], params[t], k);
    let netos = [1, 5, 20].map(|k| retorno_medio(serie, filtro_neto_k(k), k));
    let (p_tp_filtro, _) = p_primer_toque(serie, filtro_k(1));
    let brackets = [
        bracket(serie, oraculo, 60),
        bracket(serie, proxy, 60),
        bracket(serie, filtro_k(1), 60),
    ];
    let mut brier_n = 0usize;
    let (mut brier, mut decision, mut seguras, mut excluyen) = (0.0f64, 0.0f64, 0usize, 0usize);
    for t in 0..n {
        decision += (2.0 * q[t] - 1.0).abs();
        if posterior[t] >= 0.999 || posterior[t] <= 0.001 {
            seguras += 1;
        }
        excluyen += f.excluye[t] as usize;
        if t + 1 < n && !serie.tramos.iter().any(|&(inicio, _)| inicio == t + 1) {
            brier_n += 1;
            let y = if serie.estado[t + 1] > 0 { 1.0 } else { 0.0 };
            brier += (q[t] - y) * (q[t] - y);
        }
    }
    Techo {
        a_medida: iguales as f64 / (n - 1).max(1) as f64,
        lectura: lecturas as f64 / (n - 1).max(1) as f64,
        oraculo: [
            retorno_neto(serie, oraculo, 1),
            retorno_neto(serie, oraculo, 5),
            retorno_neto(serie, oraculo, 20),
        ],
        proxy: [
            retorno_neto(serie, proxy, 1),
            retorno_neto(serie, proxy, 5),
            retorno_neto(serie, proxy, 20),
        ],
        filtro: [
            retorno_neto(serie, filtro_k(1), 1),
            retorno_neto(serie, filtro_k(5), 5),
            retorno_neto(serie, filtro_k(20), 20),
        ],
        filtro_neto: netos.map(|r| r.0),
        cobertura_neta: netos.map(|r| r.1),
        bracket_ev_r: brackets.each_ref().map(|b| b.ev_r),
        bracket_vencidos: brackets.each_ref().map(|b| b.vencidos),
        p_tp_oraculo,
        p_tp_proxy,
        p_tp_filtro,
        vencidos_oraculo,
        brier_filtro: if brier_n > 0 {
            brier / brier_n as f64
        } else {
            f64::NAN
        },
        decision_filtro: decision / n.max(1) as f64,
        decodificable: excluyen as f64 / n.max(1) as f64,
        seguro: seguras as f64 / n.max(1) as f64,
    }
}

/// R conditional on resolution at a barrier. Not the unconditional EV when
/// brackets can expire; use `Techo.bracket_ev_r` for that measurement.
pub fn ev_en_r(p_tp: f64) -> f64 {
    p_tp * RR_REF - (1.0 - p_tp) - FRICCION_RT / SL_REF
}

#[test]
fn reference_direction_and_fee_boundary_are_exact() {
    let p = Peldano {
        a: 0.1,
        d: 0.002,
        sigma: 0.0025,
        n: 10,
    };
    assert_eq!(direccion_bruta(0.1, p, 5), -1.0);
    let p = Peldano { d: 0.002, ..p };
    assert_eq!(direccion_neta(0.75, p, 1), 0.0); // e == 10 bp fee
    assert_eq!(direccion_neta(0.76, p, 1), 1.0);
    assert_eq!(direccion_neta(0.24, p, 1), -1.0);
    assert_eq!(direccion_bruta(0.9, Peldano { d: 0.0, ..p }, 5), 0.0);
}

#[test]
fn reference_bracket_counts_expirations_and_excludes_censoring() {
    let p = Peldano {
        a: 0.9,
        d: 0.002,
        sigma: 0.0025,
        n: 5,
    };
    let flat = Serie {
        closes: vec![100.0; 5],
        highs: vec![100.0; 5],
        lows: vec![100.0; 5],
        vols: vec![1.0; 5],
        estado: vec![1; 5],
        peldano: p,
        generador: Generador::V2Gauss,
        tramos: vec![(0, p)],
    };
    let b = bracket(&flat, |_| 1.0, 2);
    assert_eq!(b.n, 2);
    assert_eq!(b.vencidos, 1.0);
    assert!(b.p_tp_resuelto.is_nan());
    assert!((b.ev_r + FRICCION_RT / SL_REF).abs() < 1e-14);
    assert!(bracket(&flat, |_| 0.0, 2).ev_r.is_nan());
    assert_eq!(bracket(&flat, |_| 1.0, 60).n, 0);
    let mut hit = flat;
    hit.highs[2] = 102.0;
    assert!(
        (bracket(&hit, |t| if t == 1 { 1.0 } else { 0.0 }, 2).ev_r
            - (RR_REF - FRICCION_RT / SL_REF))
            .abs()
            < 1e-14
    );
    hit.lows[2] = 99.0;
    assert!(
        (bracket(&hit, |t| if t == 1 { 1.0 } else { 0.0 }, 2).ev_r - (-1.0 - FRICCION_RT / SL_REF))
            .abs()
            < 1e-14
    );
}

#[test]
fn segment_reset_and_prefix_causality() {
    let p = Peldano {
        a: 0.9,
        d: 0.002,
        sigma: 0.0025,
        n: 200,
    };
    let a = serie_persistente_v2(p);
    let mut compound = serie_persistente_v2(p);
    let second = serie_persistente_v2(Peldano { a: 0.1, ..p });
    let ratio = compound.closes.last().unwrap() / 60000.0;
    compound.tramos.push((p.n, second.peldano));
    compound
        .closes
        .extend(second.closes.iter().map(|v| v * ratio));
    compound
        .highs
        .extend(second.highs.iter().map(|v| v * ratio));
    compound.lows.extend(second.lows.iter().map(|v| v * ratio));
    compound.vols.extend(second.vols.iter());
    compound.estado.extend(second.estado.iter());
    let fa = filtro_bayes(&a);
    let fc = filtro_bayes(&compound);
    let fb = filtro_bayes(&second);
    assert_eq!(fa.posterior, fc.posterior[..p.n]);
    for (x, y) in fb.posterior.iter().zip(&fc.posterior[p.n..]) {
        assert!((x - y).abs() < 1e-10);
    }
    // Changing a future observation cannot change an earlier posterior.
    compound.closes[350] *= 1.01;
    assert_eq!(
        fc.posterior[..350],
        filtro_bayes(&compound).posterior[..350]
    );
}

#[test]
fn gaussian_extreme_evidence_does_not_fall_back_to_prior() {
    let p = Peldano {
        a: 0.9,
        d: 0.002,
        sigma: 0.0025,
        n: 2,
    };
    let mut s = serie_persistente_v2(p);
    s.closes[0] = 60000.0 * 0.2f64.exp();
    assert!(filtro_bayes(&s).posterior[0] > 0.999999);
}

#[test]
fn gaussian_filter_matches_enumerated_hidden_paths() {
    let p = Peldano {
        a: 0.73,
        d: 0.001,
        sigma: 0.0025,
        n: 3,
    };
    let s = serie_persistente_v2(p);
    let f = filtro_bayes(&s);
    for last in 0..p.n {
        let (mut positive, mut total) = (0.0, 0.0);
        for mask in 0..(1usize << (last + 1)) {
            let (mut previous_state, mut previous_price, mut weight) = (1i8, 60000.0f64, 1.0);
            let mut state = 1;
            for t in 0..=last {
                state = if mask & (1 << t) == 0 { 1 } else { -1 };
                weight *= if state == previous_state {
                    p.a
                } else {
                    1.0 - p.a
                };
                let r = s.closes[t].ln() - previous_price.ln();
                let z = (r - f64::from(state) * p.d) / p.sigma;
                weight *= (-0.5 * z * z).exp();
                previous_state = state;
                previous_price = s.closes[t];
            }
            total += weight;
            if state > 0 {
                positive += weight;
            }
        }
        assert!((f.posterior[last] - positive / total).abs() < 1e-12);
    }
}

/// El generador hace lo que dice: persistencia medida ≈ a, y la ventaja del
/// oráculo crece con a. Es una propiedad de la serie, no del motor.
#[test]
fn t3_la_serie_tiene_la_ventaja_que_declara() {
    let n = 20_000;
    let nulo = techo(&serie_persistente(Peldano {
        a: 0.5,
        d: 0.0,
        sigma: 0.0025,
        n,
    }));
    assert!(
        (nulo.a_medida - 0.5).abs() < 0.02,
        "persistencia medida {:.3} lejos de 0,5",
        nulo.a_medida
    );
    assert!(
        nulo.oraculo[1].is_nan() && nulo.p_tp_oraculo.is_nan(),
        "sin deriva el oráculo no debería tener ventaja: 5v={:+.2e} p_tp={:.3}",
        nulo.oraculo[1],
        nulo.p_tp_oraculo
    );
    let mut anterior = f64::NEG_INFINITY;
    for a in [0.6, 0.8, 0.95] {
        let t = techo(&serie_persistente(Peldano {
            a,
            d: 0.0010,
            sigma: 0.0025,
            n,
        }));
        assert!(
            (t.a_medida - a).abs() < 0.02,
            "a={a}: persistencia medida {:.3}",
            t.a_medida
        );
        assert!(
            t.oraculo[1] > anterior,
            "a={a}: la ventaja a 5 velas ({:+.2e}) no crece con la persistencia ({:+.2e})",
            t.oraculo[1],
            anterior
        );
        anterior = t.oraculo[1];
    }
}

/// El generador v2 hace lo que dice: persistencia medida ≈ a, el lector
/// causal (filtro) gana con la persistencia y se abstiene en el nulo.
#[test]
fn t3_v2_la_serie_tiene_la_ventaja_que_declara() {
    let n = 20_000;
    let nulo = techo(&serie_persistente_v2(Peldano {
        a: 0.5,
        d: 0.0,
        sigma: 0.0025,
        n,
    }));
    assert!(
        (nulo.a_medida - 0.5).abs() < 0.02,
        "v2: persistencia medida {:.3} lejos de 0,5",
        nulo.a_medida
    );
    assert!(
        nulo.filtro[1].is_nan() && nulo.decision_filtro < 1e-9,
        "v2: en el nulo el filtro debe abstenerse: 5v={:+.2e} decisión={:.3}",
        nulo.filtro[1],
        nulo.decision_filtro
    );
    assert!(
        nulo.oraculo[1].is_nan() && nulo.p_tp_oraculo.is_nan(),
        "v2: sin deriva el oráculo no debería tener ventaja: 5v={:+.2e} p_tp={:.3}",
        nulo.oraculo[1],
        nulo.p_tp_oraculo
    );
    let mut anterior = f64::NEG_INFINITY;
    for a in [0.6, 0.8, 0.95] {
        let t = techo(&serie_persistente_v2(Peldano {
            a,
            d: 0.0010,
            sigma: 0.0025,
            n,
        }));
        assert!(
            (t.a_medida - a).abs() < 0.02,
            "v2 a={a}: persistencia medida {:.3}",
            t.a_medida
        );
        assert!(
            t.filtro[1] > anterior,
            "v2 a={a}: la ventaja causal a 5 velas ({:+.2e}) no crece con la persistencia ({:+.2e})",
            t.filtro[1],
            anterior
        );
        anterior = t.filtro[1];
    }
}

/// El filtro es el lector causal óptimo y está calibrado; en la v2 nada de la
/// vela decodifica el estado (H1-01 cerrado), en la v1 sí (la fuga medida).
#[test]
fn t3_v2_filtro_y_control_de_fuga_conjunta() {
    let p = Peldano {
        a: 0.90,
        d: 0.0020,
        sigma: 0.0025,
        n: 20_000,
    };
    let v1 = serie_persistente(p);
    let v2 = serie_persistente_v2(p);
    let t1 = techo(&v1);
    let t2 = techo(&v2);
    // La fuga H1-01: en la v1 el posterior salta a certeza en una fracción
    // grande de velas; en la v2 nunca (ruido gaussiano, soporte infinito).
    assert!(
        t1.decodificable > 0.30,
        "v1: la fuga H1-01 debería dejar decodificar ≥ 30 % de las velas, mide {:.3}",
        t1.decodificable
    );
    assert!(
        t2.decodificable == 0.0,
        "v2: ninguna vela sola debería decodificar el estado, mide {:.3}",
        t2.decodificable
    );
    // Causal ≤ no causal; óptimo ≥ ingenuo.
    assert!(
        t2.filtro[0] <= t2.oraculo[0] + 1e-9,
        "v2: el filtro ({:+.2e}) no puede superar al oráculo ({:+.2e})",
        t2.filtro[0],
        t2.oraculo[0]
    );
    assert!(
        t2.filtro[1] >= t2.proxy[1] - 2e-5,
        "v2: el filtro a 5 velas ({:+.2e}) debería ganar al proxy ({:+.2e})",
        t2.filtro[1],
        t2.proxy[1]
    );
    assert!(
        t2.brier_filtro < 0.25 && t2.brier_filtro > 0.0,
        "v2: Brier del filtro fuera de (0, 0,25): {:.4}",
        t2.brier_filtro
    );
    // Calibración por cuantiles de q: |media de q − frecuencia| < 0,05.
    let q = filtro_bayes(&v2).siguiente;
    let mut cubos = [(0usize, 0.0f64, 0.0f64); 5];
    for t in 0..(v2.estado.len() - 1) {
        let c = ((q[t] * 5.0) as usize).min(4);
        cubos[c].0 += 1;
        cubos[c].1 += q[t];
        cubos[c].2 += if v2.estado[t + 1] > 0 { 1.0 } else { 0.0 };
    }
    for (i, (n, sq, sy)) in cubos.iter().enumerate() {
        if *n < 200 {
            continue;
        }
        let (media_q, frec) = (sq / *n as f64, sy / *n as f64);
        assert!(
            (media_q - frec).abs() < 0.05,
            "v2: cubo {i} mal calibrado: media q {:.3}, frecuencia {:.3} (n={n})",
            media_q,
            frec
        );
    }
    // Independencia de mechas y volumen frente al estado en la v2.
    let corr = |x: &dyn Fn(usize) -> f64| -> f64 {
        let n = v2.estado.len() as f64;
        let (mut sx, mut sy, mut sxx, mut syy, mut sxy) = (0.0, 0.0, 0.0, 0.0, 0.0);
        for t in 0..v2.estado.len() {
            let (xi, yi) = (x(t), f64::from(v2.estado[t]));
            sx += xi;
            sy += yi;
            sxx += xi * xi;
            syy += yi * yi;
            sxy += xi * yi;
        }
        let cov = sxy / n - (sx / n) * (sy / n);
        let vx = sxx / n - (sx / n) * (sx / n);
        let vy = syy / n - (sy / n) * (sy / n);
        cov / (vx * vy).sqrt().max(1e-18)
    };
    let mecha_alta = |t: usize| {
        v2.highs[t] / v2.closes[t].max(if t == 0 { 60_000.0 } else { v2.closes[t - 1] }) - 1.0
    };
    let volumen = |t: usize| v2.vols[t];
    let c_m = corr(&mecha_alta);
    let c_v = corr(&volumen);
    assert!(
        c_m.abs() < 0.03 && c_v.abs() < 0.03,
        "v2: mecha o volumen correlacionan con el estado: {:.3} / {:.3}",
        c_m,
        c_v
    );
    let corr_noise_volume = |serie: &Serie| {
        let mut x = Vec::new();
        let mut y = Vec::new();
        for t in 1..serie.closes.len() {
            let r = serie.closes[t].ln() - serie.closes[t - 1].ln();
            x.push((r - f64::from(serie.estado[t]) * p.d).abs());
            y.push(serie.vols[t]);
        }
        let n = x.len() as f64;
        let mx = x.iter().sum::<f64>() / n;
        let my = y.iter().sum::<f64>() / n;
        let cov = x
            .iter()
            .zip(&y)
            .map(|(a, b)| (a - mx) * (b - my))
            .sum::<f64>();
        let vx = x.iter().map(|a| (a - mx).powi(2)).sum::<f64>();
        let vy = y.iter().map(|b| (b - my).powi(2)).sum::<f64>();
        cov / (vx * vy).sqrt()
    };
    assert!(
        corr_noise_volume(&v1) > 0.99,
        "positive control must detect v1 leak"
    );
    // Fixed-seed smoke bound: 5/sqrt(n), not a universal independence proof.
    assert!(corr_noise_volume(&v2).abs() < 5.0 / (p.n as f64).sqrt());
    println!(
        "[T3-V2] a=0,9 d=0,2 %: v1 decodificable={:.3} seguro={:.3} filtro(5v)={:+.1} pb oráculo={:+.1} pb | v2 decodificable={:.3} seguro={:.3} filtro(1/5/20)={:+.1}/{:+.1}/{:+.1} pb proxy={:+.1}/{:+.1}/{:+.1} pb oráculo={:+.1}/{:+.1}/{:+.1} pb brier={:.4} decisión={:.3} corr(mecha,estado)={:+.3} corr(vol,estado)={:+.3}",
        t1.decodificable,
        t1.seguro,
        t1.filtro[1] * 1e4,
        t1.oraculo[1] * 1e4,
        t2.decodificable,
        t2.seguro,
        t2.filtro[0] * 1e4,
        t2.filtro[1] * 1e4,
        t2.filtro[2] * 1e4,
        t2.proxy[0] * 1e4,
        t2.proxy[1] * 1e4,
        t2.proxy[2] * 1e4,
        t2.oraculo[0] * 1e4,
        t2.oraculo[1] * 1e4,
        t2.oraculo[2] * 1e4,
        t2.brier_filtro,
        t2.decision_filtro,
        c_m,
        c_v
    );
}
