//! CL-97 (C-F3, F3.7-02, v2): prueba de reversión de la base a la escala de
//! decisión.
//!
//! StatArb vota la reversión de la base futuro-spot. Sólo debe votar si la
//! base revierte a la escala a la que se decide (τ*, la escala dominante de
//! la moneda). Esta prueba lo contrasta a partir de los pares de la base,
//! y la decisión se toma con su nivel, no con una estimación puntual de la
//! vida media.
//!
//! Por qué no la SDE OU de antes (CL-84, revisión adversarial): un
//! Dickey-Fuller evento a evento confunde el ruido de tick y la asincronía
//! entre el futuro y el spot con reversión; su crítico depende de la edad del
//! estimador (tamaño 0,44 al primer minuto, 0,02 en régimen), y comparar la
//! vida media estimada con 2τ* no muerde. Dos caminatas independientes
//! pasaban la guarda en 40 de 40 semillas.
//!
//! Hipótesis. Con la base b como un OU de reversión θ, H0: θ ≤ θ0 = ln2/(2τ_k),
//! es decir, vida media ≥ 2τ_k. Rechazar H0 demuestra t½ < 2τ_k. Se contrasta
//! en el nodo k de la malla espectral más grande con τ_k ≤ τ*: así, rechazar
//! implica t½ < 2τ_k ≤ 2τ*, la condición que la guarda del lector exige.
//!
//! Muestreo. Por nodo, una rejilla de celdas de Δ = τ_k/8. La muestra de una
//! celda es el último par de la celda; una celda sin pares es un dato ausente:
//! corta la cadena de retardos y sólo hace decaer las sumas (como los huecos
//! de CL-87). Las sumas olvidan con λ = e^{−1/128} por celda: memoria de
//! 16·τ_k.
//!
//! Estimador. Con las muestras b_t de celdas consecutivas, regresión
//! Δb_t = a − c·b_{t−1} + u por variables instrumentales con instrumento
//! b_{t−2}: ĉ = −S_zy/S_zx con z = b_{t−2}, x = b_{t−1}, y = Δb_t (sumas
//! centradas y ponderadas). El ruido de medida de la base (tick, horquilla)
//! hace u de media móvil de orden 1; con b_{t−2} el instrumento no
//! correlaciona con u, y por eso el ruido no fabrica reversión (con mínimos
//! cuadrados sí la fabrica). Error típico HAC de retardo 1:
//! var(ĉ) = (γ0·Σw²z̃² + 2γ1·Σw_t w_{t−1} z̃_t z̃_{t−1})/S_zx². Corrección de
//! sesgo del AR(1) (Kendall): ĉ_bc = ĉ − (1 + 3ρ̂)/n_ef, con ρ̂ = 1 − ĉ y
//! n_ef = (Σw)²/Σw². En la frontera, c0 = 1 − e^{−θ0·Δ} = 1 − 2^{−1/16}.
//! Estadístico t = (ĉ_bc − c0)/se; se rechaza H0 si t > `CRITICO`.
//!
//! Crítico. 1,75, calibrado por Monte Carlo en la frontera
//! (`tests/medicion_critico_reversion.rs`, N = 10 000 por configuración):
//! nodos 17 a 19, pares cada 0,25 a 4 s con marcas irregulares, ruido de
//! medida de 0 a 1 desviación típica de la base y edades de 0,55 a 4
//! memorias. El cuantil 95 máximo de t es 1,76 y el tamaño máximo 5,1 %,
//! los dos con ruido de 1 σ; con ruido ≤ ½ σ el tamaño no pasa de 4,7 %.
//! En el prototipo, sin la corrección de sesgo el cuantil subía a 2,2 y
//! dependía de la edad.
//!
//! Madurez. Un nodo sólo opina con Σw ≥ (1 − e^{−1/2})/(1 − λ), la masa de
//! media memoria de celdas consecutivas (8·τ_k de pares). Antes, abstención.
//!
//! Lo que no hace: no estima la vida media (sólo la acota), no corrige la
//! repetición de miradas en el tiempo (el nivel es por decisión: en cada
//! instante la probabilidad de pasar la guarda bajo H0 es ≤ 5 %), y con una
//! reversión mucho más rápida que la rejilla (t½ ≪ Δ) el instrumento se
//! debilita y la prueba se abstiene: a la escala de decisión esa base es
//! ruido blanco.

use quantum_arena::temporal_spectrum::SPECTRUM_SCALES_MS;

/// Primer y último nodo con prueba: τ_17 = 17,2 s y τ_22 = 4,9 h. La τ de
/// decisión vive en la banda [30 s, 12 h], así que el nodo más grande con
/// τ_k ≤ τ* está siempre entre ellos.
pub const NODO_MIN: usize = 17;
pub const NODO_MAX: usize = 22;

/// Celdas de la rejilla por τ_k (Δ = τ_k/8).
pub const CELDAS_POR_TAU: f64 = 8.0;
/// Memoria de las sumas, en τ_k (128 celdas).
pub const MEMORIA_EN_TAU: f64 = 16.0;
/// Memoria en celdas.
const MEMORIA_CELDAS: f64 = CELDAS_POR_TAU * MEMORIA_EN_TAU;
/// Crítico del estadístico t (cola superior, nivel 5 %).
pub const CRITICO: f64 = 1.75;

/// c en la frontera de H0: 1 − e^{−θ0·Δ}, con θ0·Δ = ln2/(2·8).
fn c_frontera() -> f64 {
    -(-std::f64::consts::LN_2 / (2.0 * CELDAS_POR_TAU)).exp_m1()
}

/// Masa mínima para opinar: la de media memoria de celdas consecutivas.
fn masa_madura() -> f64 {
    let lambda = (-1.0 / MEMORIA_CELDAS).exp();
    -(-0.5f64).exp_m1() / (1.0 - lambda)
}

/// Sumas ponderadas de un nodo. Las de primer orden pesan w_t (λ por celda);
/// las `w2*` pesan w_t² y las `w11*` w_t·w_{t−1} (λ² por celda).
#[derive(Debug, Clone, Copy, Default)]
struct Sumas {
    w: f64,
    z: f64,
    x: f64,
    y: f64,
    zz: f64,
    zx: f64,
    zy: f64,
    xx: f64,
    xy: f64,
    yy: f64,
    // Retardo 1 del residuo: Σ w_t·(·)_t·(·)_{t−1}.
    yy1: f64,
    yx1: f64,
    xy1: f64,
    xx1: f64,
    w2: f64,
    w2z: f64,
    w2zz: f64,
    w11: f64,
    w11z: f64,
    w11z1: f64,
    w11zz1: f64,
}

impl Sumas {
    /// Decaimiento de `n` celdas: λ^n en las de primer orden, λ^{2n} en las
    /// de segundo.
    fn decaer(&mut self, n: f64) {
        let l1 = (-n / MEMORIA_CELDAS).exp();
        let l2 = l1 * l1;
        for v in [
            &mut self.w,
            &mut self.z,
            &mut self.x,
            &mut self.y,
            &mut self.zz,
            &mut self.zx,
            &mut self.zy,
            &mut self.xx,
            &mut self.xy,
            &mut self.yy,
            &mut self.yy1,
            &mut self.yx1,
            &mut self.xy1,
            &mut self.xx1,
        ] {
            *v *= l1;
        }
        for v in [
            &mut self.w2,
            &mut self.w2z,
            &mut self.w2zz,
            &mut self.w11,
            &mut self.w11z,
            &mut self.w11z1,
            &mut self.w11zz1,
        ] {
            *v *= l2;
        }
    }

    /// Suma la fila (z, x, y) con peso 1; `previa` es la fila de la celda
    /// anterior, que tras decaer pesa λ.
    fn sumar(&mut self, (z, x, y): (f64, f64, f64), previa: Option<(f64, f64, f64)>) {
        self.w += 1.0;
        self.z += z;
        self.x += x;
        self.y += y;
        self.zz += z * z;
        self.zx += z * x;
        self.zy += z * y;
        self.xx += x * x;
        self.xy += x * y;
        self.yy += y * y;
        self.w2 += 1.0;
        self.w2z += z;
        self.w2zz += z * z;
        if let Some((zp, xp, yp)) = previa {
            let lambda = (-1.0 / MEMORIA_CELDAS).exp();
            self.yy1 += y * yp;
            self.yx1 += y * xp;
            self.xy1 += x * yp;
            self.xx1 += x * xp;
            self.w11 += lambda;
            self.w11z += lambda * z;
            self.w11z1 += lambda * zp;
            self.w11zz1 += lambda * z * zp;
        }
    }
}

/// Estado de la prueba en un nodo tras la última celda cerrada.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct EstadoNodo {
    /// Estadístico t; −∞ si no se puede calcular (instrumento nulo,
    /// varianza no positiva).
    pub t: f64,
    /// Masa Σw y si llega a la de madurez.
    pub masa: f64,
    pub madura: bool,
    /// Media y desviación típica de la base en la memoria del nodo
    /// (respecto al origen de la prueba).
    pub media: f64,
    pub sigma: f64,
}

#[derive(Debug, Clone)]
struct Nodo {
    delta_ms: f64,
    /// Celda abierta y último valor de la base en ella.
    celda: Option<u64>,
    valor: f64,
    /// Muestras de las dos celdas anteriores, si son consecutivas.
    b1: Option<f64>,
    b2: Option<f64>,
    /// Fila de la celda anterior (para el retardo 1).
    previa: Option<(f64, f64, f64)>,
    s: Sumas,
    estado: Option<EstadoNodo>,
}

impl Nodo {
    fn new(tau_ms: f64) -> Self {
        Self {
            delta_ms: tau_ms / CELDAS_POR_TAU,
            celda: None,
            valor: 0.0,
            b1: None,
            b2: None,
            previa: None,
            s: Sumas::default(),
            estado: None,
        }
    }

    fn observar(&mut self, ts_ms: u64, b: f64) {
        let j = (ts_ms as f64 / self.delta_ms).floor() as u64;
        match self.celda {
            None => {
                self.celda = Some(j);
                self.valor = b;
            }
            Some(c) if j < c => {}
            Some(c) if j == c => self.valor = b,
            Some(c) => {
                self.cerrar();
                let vacias = j - c - 1;
                if vacias > 0 {
                    // Celdas sin pares: sin dato. Cortan la cadena de
                    // retardos y sólo hacen decaer las sumas.
                    self.s.decaer(vacias as f64);
                    self.b1 = None;
                    self.b2 = None;
                    self.previa = None;
                    self.estado = self.calcular();
                }
                self.celda = Some(j);
                self.valor = b;
            }
        }
    }

    /// Cierra la celda abierta: un paso de olvido y su fila, si las dos
    /// celdas anteriores tienen muestra.
    fn cerrar(&mut self) {
        self.s.decaer(1.0);
        let fila = match (self.b2, self.b1) {
            (Some(b2), Some(b1)) => Some((b2, b1, self.valor - b1)),
            _ => None,
        };
        if let Some(f) = fila {
            self.s.sumar(f, self.previa);
        }
        self.previa = fila;
        self.b2 = self.b1;
        self.b1 = Some(self.valor);
        self.estado = self.calcular();
    }

    fn calcular(&self) -> Option<EstadoNodo> {
        let s = &self.s;
        let w = s.w;
        if !(w > 0.0) || !(s.w2 > 0.0) {
            return None;
        }
        let (zb, xb, yb) = (s.z / w, s.x / w, s.y / w);
        let sxx = (s.xx - w * xb * xb).max(0.0);
        let media = xb;
        let sigma = (sxx / w).sqrt();
        let masa = w;
        let madura = w >= masa_madura();
        let szx = s.zx - w * zb * xb;
        let szy = s.zy - w * zb * yb;
        let t = (|| {
            if !(szx.abs() > 0.0) {
                return f64::NEG_INFINITY;
            }
            let c = -szy / szx;
            let syy = s.yy - w * yb * yb;
            let sxy = s.xy - w * xb * yb;
            let g0 = (syy + 2.0 * c * sxy + c * c * sxx) / w;
            let syy1 = s.yy1 - w * yb * yb;
            let syx1 = s.yx1 - w * yb * xb;
            let sxy1 = s.xy1 - w * xb * yb;
            let sxx1 = s.xx1 - w * xb * xb;
            let g1 = (syy1 + c * syx1 + c * sxy1 + c * c * sxx1) / w;
            let szz2 = s.w2zz - 2.0 * zb * s.w2z + zb * zb * s.w2;
            let szz11 = s.w11zz1 - zb * (s.w11z + s.w11z1) + zb * zb * s.w11;
            let var = (g0 * szz2 + 2.0 * g1 * szz11) / (szx * szx);
            if !(var > 0.0) || !var.is_finite() || !(g0 > 0.0) {
                return f64::NEG_INFINITY;
            }
            let n_ef = w * w / s.w2;
            let c_bc = c - (1.0 + 3.0 * (1.0 - c)) / n_ef;
            let t = (c_bc - c_frontera()) / var.sqrt();
            if t.is_finite() {
                t
            } else {
                f64::NEG_INFINITY
            }
        })();
        Some(EstadoNodo {
            t,
            masa,
            madura,
            media,
            sigma,
        })
    }
}

/// Veredicto de la prueba a una τ de decisión.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Veredicto {
    /// τ_k del nodo contrastado (NaN si la τ de decisión queda bajo la
    /// malla).
    pub tau_nodo_ms: f64,
    /// Reversión demostrada: nodo maduro y t > `CRITICO`.
    pub demostrada: bool,
    /// Estadístico t del nodo (−∞ sin estado).
    pub t: f64,
    /// Último valor de la base y su media y desviación típica en la memoria
    /// del nodo (NaN sin estado).
    pub base: f64,
    pub media: f64,
    pub sigma: f64,
}

/// Prueba de reversión de una serie de pares, en los nodos 17 a 22.
#[derive(Debug, Clone)]
pub struct PruebaReversion {
    nodos: Vec<Nodo>,
    /// Primer valor de la base: las sumas se llevan respecto a él para no
    /// perder precisión al centrar.
    origen: Option<f64>,
    /// Último par (marca, base).
    ultimo: Option<(u64, f64)>,
    pares: u64,
}

impl Default for PruebaReversion {
    fn default() -> Self {
        Self::new()
    }
}

impl PruebaReversion {
    pub fn new() -> Self {
        Self {
            nodos: (NODO_MIN..=NODO_MAX)
                .map(|k| Nodo::new(SPECTRUM_SCALES_MS[k]))
                .collect(),
            origen: None,
            ultimo: None,
            pares: 0,
        }
    }

    /// Pares observados.
    pub fn pares(&self) -> u64 {
        self.pares
    }

    /// Observa el par síncrono de marca `ts_ms` con base `b`. Una base no
    /// finita o una marca anterior a la del último par se descartan.
    pub fn observar(&mut self, ts_ms: u64, b: f64) {
        if !b.is_finite() {
            return;
        }
        if matches!(self.ultimo, Some((t0, _)) if ts_ms < t0) {
            return;
        }
        let o = *self.origen.get_or_insert(b);
        for n in self.nodos.iter_mut() {
            n.observar(ts_ms, b - o);
        }
        self.ultimo = Some((ts_ms, b));
        self.pares += 1;
    }

    /// Índice del nodo de decisión: el más grande con τ_k ≤ τ*.
    fn nodo_de(tau_decision_ms: f64) -> Option<usize> {
        if !tau_decision_ms.is_finite() {
            return None;
        }
        (NODO_MIN..=NODO_MAX)
            .rev()
            .find(|&k| SPECTRUM_SCALES_MS[k] <= tau_decision_ms)
            .map(|k| k - NODO_MIN)
    }

    /// Estado del nodo k de la malla (17 a 22), si tiene.
    pub fn estado_nodo(&self, k: usize) -> Option<EstadoNodo> {
        if !(NODO_MIN..=NODO_MAX).contains(&k) {
            return None;
        }
        self.nodos[k - NODO_MIN].estado
    }

    /// Veredicto a la τ de decisión `tau_decision_ms`. `None` si aún no hay
    /// ningún par.
    pub fn veredicto(&self, tau_decision_ms: f64) -> Option<Veredicto> {
        let (_, b) = self.ultimo?;
        let o = self.origen.unwrap_or(0.0);
        let Some(i) = Self::nodo_de(tau_decision_ms) else {
            return Some(Veredicto {
                tau_nodo_ms: f64::NAN,
                demostrada: false,
                t: f64::NEG_INFINITY,
                base: b,
                media: f64::NAN,
                sigma: f64::NAN,
            });
        };
        let tau_nodo_ms = SPECTRUM_SCALES_MS[NODO_MIN + i];
        Some(match self.nodos[i].estado {
            Some(e) => Veredicto {
                tau_nodo_ms,
                demostrada: e.madura && e.t > CRITICO,
                t: e.t,
                base: b,
                media: e.media + o,
                sigma: e.sigma,
            },
            None => Veredicto {
                tau_nodo_ms,
                demostrada: false,
                t: f64::NEG_INFINITY,
                base: b,
                media: f64::NAN,
                sigma: f64::NAN,
            },
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Verdad: c0 = 1 − 2^{−1/16} y la masa de madurez es la de 64 celdas
    /// consecutivas.
    #[test]
    fn constantes_de_la_frontera() {
        assert!((c_frontera() - (1.0 - 2f64.powf(-1.0 / 16.0))).abs() < 1e-15);
        let lambda = (-1.0f64 / 128.0).exp();
        let masa64: f64 = (0..64).map(|i| lambda.powi(i)).sum();
        assert!((masa_madura() - masa64).abs() < 1e-9, "{} {}", masa_madura(), masa64);
    }

    /// Verdad: las sumas ponderadas son las de la definición: una fila por
    /// celda con muestra en las dos anteriores, peso λ^{edad en celdas}.
    /// Se comparan con la suma directa sobre la lista de filas, con huecos.
    #[test]
    fn sumas_igual_que_la_definicion() {
        let mut n = Nodo::new(SPECTRUM_SCALES_MS[17]);
        let d = n.delta_ms;
        // Celdas con muestra (índice de celda → valor); 7 y 8 vacías.
        let celdas: Vec<(u64, f64)> = (0..20u64)
            .filter(|c| *c != 7 && *c != 8)
            .map(|c| (c, ((c * 37) % 11) as f64 * 1e-4))
            .collect();
        for &(c, v) in &celdas {
            // Dos pares por celda: cuenta el último.
            n.observar(((c as f64 + 0.2) * d) as u64, v + 1.0);
            n.observar(((c as f64 + 0.7) * d) as u64, v);
        }
        // Cierra la celda 19 con un par en la 20.
        n.observar((20.5 * d) as u64, 0.0);
        let lambda = (-1.0f64 / 128.0).exp();
        let ultima = 19u64;
        let mut w = 0.0;
        let mut zx = 0.0;
        let mut w11 = 0.0;
        let mut yy1 = 0.0;
        let mut prev: Option<(u64, f64, f64, f64)> = None;
        let val = |c: u64| celdas.iter().find(|x| x.0 == c).map(|x| x.1);
        for c in 2..=ultima {
            let fila = match (val(c), val(c - 1), val(c - 2)) {
                (Some(b), Some(b1), Some(b2)) => Some((b2, b1, b - b1)),
                _ => None,
            };
            if let Some((z, x, y)) = fila {
                let peso = lambda.powi((ultima - c) as i32);
                w += peso;
                zx += peso * z * x;
                if let Some((cp, _, xp, yp)) = prev {
                    if cp + 1 == c {
                        w11 += peso * lambda.powi((ultima - cp) as i32);
                        yy1 += peso * y * yp;
                        let _ = xp;
                    }
                }
                prev = Some((c, z, x, y));
            }
        }
        assert!((n.s.w - w).abs() < 1e-12, "{} {}", n.s.w, w);
        assert!((n.s.zx - zx).abs() < 1e-15, "{} {}", n.s.zx, zx);
        assert!((n.s.w11 - w11).abs() < 1e-12, "{} {}", n.s.w11, w11);
        assert!((n.s.yy1 - yy1).abs() < 1e-15, "{} {}", n.s.yy1, yy1);
    }

    /// Verdad: un nodo sin masa de media memoria no demuestra nada, por
    /// fuerte que sea la reversión.
    #[test]
    fn sin_madurez_no_hay_demostracion() {
        let mut p = PruebaReversion::new();
        let tau = SPECTRUM_SCALES_MS[17];
        let d = tau / 8.0;
        // Alterna ±1 por celda (reversión extrema) durante 60 celdas.
        for c in 0..60u64 {
            p.observar((c as f64 * d + 1.0) as u64, if c % 2 == 0 { 1e-4 } else { -1e-4 });
        }
        let v = p.veredicto(tau).expect("hay pares");
        assert!(!v.demostrada, "{v:?}");
    }

    /// Verdad: la τ de decisión elige el nodo más grande con τ_k ≤ τ*; bajo
    /// la malla no hay nodo y no hay demostración.
    #[test]
    fn el_nodo_de_decision() {
        assert_eq!(PruebaReversion::nodo_de(SPECTRUM_SCALES_MS[18]), Some(1));
        assert_eq!(PruebaReversion::nodo_de(SPECTRUM_SCALES_MS[18] * 3.9), Some(1));
        assert_eq!(PruebaReversion::nodo_de(30_000.0), Some(0));
        assert_eq!(PruebaReversion::nodo_de(43_200_000.0), Some(5));
        assert_eq!(PruebaReversion::nodo_de(10_000.0), None);
        assert_eq!(PruebaReversion::nodo_de(f64::NAN), None);
        let mut p = PruebaReversion::new();
        p.observar(1_000, 1e-4);
        let v = p.veredicto(10_000.0).expect("hay pares");
        assert!(!v.demostrada && v.tau_nodo_ms.is_nan());
    }

    /// Verdad: bases no finitas y marcas que retroceden no cuentan.
    #[test]
    fn entradas_invalidas() {
        let mut p = PruebaReversion::new();
        assert!(p.veredicto(60_000.0).is_none());
        p.observar(5_000, f64::NAN);
        assert_eq!(p.pares(), 0);
        p.observar(5_000, 1e-4);
        p.observar(4_000, 2e-4);
        assert_eq!(p.pares(), 1);
        assert_eq!(p.ultimo, Some((5_000, 1e-4)));
    }
}
