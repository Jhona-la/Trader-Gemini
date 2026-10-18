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
//! 16·τ_k. Una fila exige tres celdas seguidas con pares (EST-6 de la
//! revisión del PR #60): con pares de Poisson por debajo de 0,615 por segundo,
//! el nodo 17 (Δ = 2,1 s) no madura nunca. En vivo lo evita el sondeo REST
//! del spot de 1 s.
//!
//! Estimador. Con las muestras b_t de celdas consecutivas, regresión
//! Δb_t = a − c·b_{t−1} + u por variables instrumentales con instrumento
//! b_{t−2}: ĉ = −S_zy/S_zx con z = b_{t−2}, x = b_{t−1}, y = Δb_t (sumas
//! centradas y ponderadas). El ruido de medida de la base (tick, horquilla)
//! hace u de media móvil de orden 1; con b_{t−2} el instrumento no
//! correlaciona con u, y por eso el ruido no fabrica reversión (con mínimos
//! cuadrados sí la fabrica). Eso vale con un ruido de medida de hasta una
//! desviación típica de la base; con más, el instrumento se debilita y el
//! tamaño en la frontera sube (EST-2 de la revisión del PR #60: 6,2-7,1 % con
//! 2 σ, 7,3 % con 3 σ, 8,4-8,9 % con 10 σ y 8,8 % con ruido blanco puro).
//! Desde C-F5b el veredicto exige además la prueba a la escala de decisión,
//! y en esa frontera lo votado revierte como mucho 0,29/(1 + r²) en τ_k:
//! con ruido fuerte la prueba larga lo deja en su H0 y el veredicto no vota.
//! Error típico HAC de
//! retardo 1: var(ĉ) = (γ0·Σw²z̃² + 2γ1·Σw_t w_{t−1} z̃_t z̃_{t−1})/S_zx².
//! Corrección de sesgo del AR(1) (Kendall): ĉ_bc = ĉ − (1 + 3ρ̂)/n_ef, con
//! ρ̂ = 1 − ĉ y n_ef = (Σw)²/Σw². En la frontera,
//! c0 = 1 − e^{−θ0·Δ} = 1 − 2^{−1/16}.
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
//! Madurez. La prueba corta sólo opina con Σw ≥ (1 − e^{−1/2})/(1 − λ), la
//! masa de media memoria de celdas consecutivas (8·τ_k de pares); el
//! veredicto, además, con la prueba a la escala de decisión madura (33·τ_k,
//! ver abajo). Antes, abstención.
//!
//! Lo que no hace: no estima la vida media (sólo la acota), no corrige la
//! repetición de miradas en el tiempo (el nivel es por decisión: en cada
//! instante la probabilidad de pasar la guarda bajo H0 es ≤ 5 %), y con una
//! reversión mucho más rápida que la rejilla (t½ ≪ Δ) el instrumento se
//! debilita: a la escala de decisión esa base es ruido blanco. «Se abstiene»
//! sólo es cierto con poco ruido de medida; con más de una desviación típica
//! el tamaño de esta prueba sola sube al 7-10 % (ver arriba, EST-2).
//!
//! Prueba a la escala de decisión (C-F5b, EST-3 de la revisión del PR #60;
//! estimador de C-F5e). La prueba de retardos cortos extrapola de Δ y 2Δ a
//! τ_k suponiendo un OU único. Con una base de dos escalas (un OU rápido más
//! otro lento o una caminata) demuestra «t½ < 2τ_k» por la escala rápida
//! aunque la desviación que se vota apenas revierta en τ_k: en el caso del
//! revisor (OU de 2 celdas más OU de 1 676 celdas con 10,4 veces su
//! desviación) votaba en el 10-21 % de los instantes con un 15-19 % de
//! reversión en τ_k. La segunda prueba (`Largo`) mide en τ_k lo que se vota:
//! la desviación observada d = b − m respecto a la media del nodo (ruido de
//! medida incluido). Fila x = d_{t−L}, y = b_t − (2b_{t−L−1} − b_{t−L−2}) con
//! L = 8 celdas, y R̂ = −Σw·x·y/Σw·x² es la fracción de la desviación votada
//! que revierte en τ_k (cota inferior; ver `Largo`). H0: R ≤ ½. Error típico
//! sándwich con núcleo uniforme de retardo L; memoria de 64·τ_k (512
//! celdas), madurez a media memoria (unas 266 celdas seguidas). Estadístico
//! t = (R̂ − ½ − `SESGO_LARGO`/n)·Σw·x²/√V, con n = min(n_ef, n_x); rechazo
//! si t > `CRITICO_LARGO` = z_{0,95}. El veredicto exige las dos pruebas: es
//! una prueba de intersección-unión, y su nivel no pasa del de cada una.
//!
//! Por qué cambió en C-F5e (revisión adversarial de C-F5b). El ĉ de C-F5b,
//! por variables instrumentales, medía la reversión de la parte persistente
//! de la base: el ruido de medida no lo movía, pero lo votado revierte
//! (1 − ρ^L)/(1 + r²) con ruido de r desviaciones típicas, y con r ≥ 1 la
//! prueba votaba (44-52 % de los instantes con t½ = τ_k/5) desviaciones que
//! revertían menos de la mitad. Su corrección de muestra finita por n_ef
//! tampoco cubría la volatilidad por rachas (5,2-9,8 % de tamaño), y sus
//! sumas de cuarto orden respecto al origen perdían precisión al alejarse
//! la base de él.
//!
//! Frontera. Para un OU de vida media t½ (celdas) con ruido r, el estimando
//! estacionario vale ½ en t½ = 7,76 celdas sin ruido, 7,01 con r = ¼ y 5,23
//! con r = ½; con r ≥ 0,8 ningún OU único llega a ½ (con r = 1 el máximo es
//! 0,41) y todos están en H0. Hay una segunda raíz, por debajo de
//! 0,57-0,75 celdas: la extrapolación de orden 2 falla con una reversión mucho más
//! rápida que la rejilla y R̂ cae aunque lo votado revierta casi entero; es
//! pérdida de potencia, no de nivel, y esas configuraciones no entran en la
//! calibración del tamaño.
//!
//! Calibración (prototipo numpy del 2026-10-18, el mismo estimador fila a
//! fila; 2 000 a 5 000 trayectorias por configuración): nodos 17 y 18, pares
//! cada 0,25 a 4 s, edades de 270 a 2 048 celdas (n_ef satura en ≈ 1 023).
//! H0: la frontera superior con r = 0, ¼ y ½ (15 configuraciones), la misma
//! con volatilidad por rachas (log-volatilidad de desviación típica ½ y 1,
//! vida media de 50 y 500 celdas, y dos estados de Markov con volatilidad
//! ×4 y ×8 en el alto: 20) y t½ = τ_k/5 con ruido de 1 a 10 desviaciones típicas (12). Con
//! `SESGO_LARGO` = 19 y el crítico asintótico el tamaño máximo de la prueba
//! larga sola es 4,96 % (rachas de Markov ×4, 2 048 celdas; error de
//! muestreo 0,3 %): 4,5 % en la frontera sin ruido, 2,5 % con r = ½ y
//! ≤ 0,33 % con r ≥ 1. El del veredicto, 4,4 %. Un núcleo de retardo L + 2
//! (g es una media móvil de orden L + 2) necesita además un exceso del
//! crítico (45/n con el mismo sesgo) y tiene menos potencia. Medición en Rust:
//! `tests/medicion_critico_reversion.rs`.
//!
//! Resultado (prototipo). Con t½ = τ_k/5 sin ruido el veredicto rechaza el
//! 70 % de las trayectorias a 270 celdas, el 89 % a 384 y el 97 % a 512; con
//! ruido de ½ σ, del 16 % al 81 % entre 270 y 2 048 celdas (lo votado
//! revierte 0,79). En el caso del revisor vota ≤ 0,9 %. Con bases de dos
//! escalas en la frontera (la desviación media revierte la mitad en τ_k)
//! vota hasta el 16 % de los instantes en las edades jóvenes, y lo votado
//! revierte de 0,49 a 0,74 (todas las edades juntas): la prueba mide lo que
//! revierte la desviación en la ventana, y una ventana en la que la
//! componente lenta está quieta sí revierte. Ese es el contrato de
//! decisión: se vota lo que revierte al menos la mitad en τ_k; el tamaño
//! sobre trayectorias de esas bases no está acotado al 5 %. La prueba corta
//! añade poco: en un OU, rechazar la larga implica t½ < τ_k y por tanto la
//! H1 de la corta; en lo medido le quita al veredicto ≤ 6 puntos de
//! potencia. Coste: la prueba larga madura más tarde (unas 266 celdas,
//! 33·τ_k).

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
    // CL-110 (EST-1): masa y momentos de primer orden de las filas CON fila
    // previa (las únicas que entran en las sumas de retardo 1), para
    // centrarlas con lo que llevan y no con la masa total.
    l1: f64,
    l1y: f64,
    l1yp: f64,
    l1x: f64,
    l1xp: f64,
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
            &mut self.l1,
            &mut self.l1y,
            &mut self.l1yp,
            &mut self.l1x,
            &mut self.l1xp,
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
            self.l1 += 1.0;
            self.l1y += y;
            self.l1yp += yp;
            self.l1x += x;
            self.l1xp += xp;
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
    /// C-F5b (EST-3): la prueba a la escala de decisión, sobre las mismas
    /// celdas.
    largo: Largo,
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
            largo: Largo::default(),
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
                    self.largo.vaciar(vacias);
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
        self.largo.cerrar(self.valor, self.estado.map(|e| e.media));
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
            // CL-110 (EST-1): Σ(a_t − ā)(b_{t−1} − b̄) sobre las filas con
            // fila previa = Σ a·b₋₁ − b̄·Σa − ā·Σb₋₁ + ā·b̄·Σ1, con las sumas
            // de ESAS filas (antes, w·ā·b̄ con la masa total: t dejaba de ser
            // invariante a desplazar la base cuando había celdas vacías).
            let cruz = |ab1: f64, sa: f64, sb1: f64, ma: f64, mb: f64| {
                ab1 - mb * sa - ma * sb1 + ma * mb * s.l1
            };
            let syy1 = cruz(s.yy1, s.l1y, s.l1yp, yb, yb);
            let syx1 = cruz(s.yx1, s.l1y, s.l1xp, yb, xb);
            let sxy1 = cruz(s.xy1, s.l1x, s.l1yp, xb, yb);
            let sxx1 = cruz(s.xx1, s.l1x, s.l1xp, xb, xb);
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

/// C-F5b (EST-3): horizonte de la prueba a la escala de decisión, en celdas
/// (8 celdas = τ_k).
pub const HORIZONTE_CELDAS: usize = 8;
/// Memoria de la prueba a la escala de decisión, en τ_k (512 celdas).
pub const MEMORIA_LARGA_EN_TAU: f64 = 64.0;
const MEMORIA_LARGA_CELDAS: f64 = CELDAS_POR_TAU * MEMORIA_LARGA_EN_TAU;
/// Frontera de H0 de la prueba a la escala de decisión: la desviación
/// votada revierte menos de la mitad en τ_k.
pub const FRACCION_FRONTERA: f64 = 0.5;
/// Corrección de muestra finita del estadístico de la prueba a la escala de
/// decisión, por unidad de tamaño efectivo: t se calcula con
/// R̂ − ½ − SESGO_LARGO/n (n = `EstadoLargo::n`). Es la menor que deja el
/// tamaño ≤ 5 % en las 47 configuraciones de H0 del Monte Carlo con el
/// crítico asintótico (ver la cabecera).
pub const SESGO_LARGO: f64 = 19.0;
/// Crítico del estadístico t de la prueba a la escala de decisión (cola
/// superior, nivel 5 %): z_{0,95}. C-F5e: con la corrección por n el
/// tamaño ya queda ≤ 5 % y el exceso de muestra finita del crítico de
/// C-F5b (100/n_ef) sobra.
pub const CRITICO_LARGO: f64 = 1.644_853_626_951_472_2;

/// Masa mínima de la prueba a la escala de decisión: media memoria.
fn masa_madura_larga() -> f64 {
    let lambda = (-1.0 / MEMORIA_LARGA_CELDAS).exp();
    -(-0.5f64).exp_m1() / (1.0 - lambda)
}

/// Muestras que necesita una fila: las celdas t, t−L, t−L−1 y t−L−2.
const VENTANA: usize = HORIZONTE_CELDAS + 3;

/// C-F5b (EST-3 de la revisión del PR #60) y C-F5e: prueba de reversión a la
/// escala de decisión.
///
/// Estimando (C-F5e). Lo que vota el lector es la desviación OBSERVADA de
/// la base respecto a la media del nodo, d = b − m (m es `EstadoNodo::media`,
/// memoria de 16·τ_k), ruido de medida incluido. Lo que esa desviación
/// revierte en τ_k es R = −E[d·Δ]/E[d²], con Δ el cambio de la base limpia
/// en τ_k. La fila de la celda t es x = d_{t−L}, lo que se votó, e
/// y = b_t − (2·b_{t−L−1} − b_{t−L−2}): el valor en τ_k menos la
/// extrapolación lineal de b_{t−L} desde las dos celdas anteriores, con
/// L = `HORIZONTE_CELDAS`. y no lleva el ruido de x, y
/// R̂ = −Σw·x·y/Σw·x² (regresión por el origen) estima R menos lo que
/// pierde la extrapolación: para una mezcla de OU, γ(0) se reconstruye como
/// 2γ(1) − γ(2) = Σ v_i (1 − (1 − ρ_i)²) ≤ γ(0), así que R̂ es conservador
/// y pierde (1 − ρ)² (0,007 en la frontera; 0,12 con t½ = τ_k/5). Sólo exige
/// muestra en esas cuatro celdas y media en t − L; los huecos de en medio no
/// cortan la fila.
///
/// Por el origen y sobre d, no centrado en la propia muestra: centrar x en
/// su media de la memoria larga quitaba la parte de d que no revierte (la
/// deriva lenta y la caminata), y con bases de dos escalas la prueba votaba
/// en el 15-22 % de los instantes con la desviación media en la frontera.
/// x y y son diferencias, así que las sumas no dependen del origen ni
/// pierden precisión al alejarse la base de él.
///
/// Hasta C-F5b se contrastaba ĉ = −S̃zy/S̃zx (z = b_{t−L−1}, y = b_t − x),
/// la reversión de la parte persistente: el ruido no la mueve, pero lo
/// votado, con ruido de una desviación típica o más, revertía menos de la
/// mitad aunque la prueba lo diera por demostrado. La alternativa de usar el
/// instrumento sólo en el denominador, −S̃zy/S̃xx, también es conservadora
/// pero pierde el factor ρ(Δ) entero: con t½ = τ_k/5 la potencia caía del
/// 95 % al 21 %.
///
/// Varianza. Sándwich con núcleo uniforme de retardo L sobre la influencia
/// g_t = x_t·y_t + R̂·x_t², con pesos w_t·w_s:
/// V = Σ_t Σ_{|t−s| ≤ L} w_t w_s g_t g_s y var(R̂) = V/(Σw·x²)². Con
/// φ = [x·y, x²], g = αᵀφ con α = [1, R̂], y V = αᵀMα con
/// M = Σ_t Σ_{|t−s| ≤ L} w_t w_s φ_t φ_sᵀ. g es una media móvil de orden
/// L + 2 (las filas t y t − L − 2 comparten una celda); el núcleo de retardo
/// L la cubre casi entera y la calibración absorbe el resto (con L + 2 el
/// tamaño no mejora y la potencia baja; ver la cabecera).
///
/// Tamaño efectivo (C-F5e, heterocedasticidad). La corrección de muestra
/// finita va por n = min(n_ef, n_x), con n_ef = (Σw)²/Σw² y
/// n_x = 3·(Σw·x²)²/Σw²·x⁴: con volatilidad por rachas la energía de lo
/// votado se concentra en las rachas altas, n_x baja y el estadístico tiene
/// colas más pesadas de lo que n_ef supone. Con d gaussiana homocedástica
/// n_x ≈ n_ef.
#[derive(Debug, Clone, Default)]
struct Largo {
    /// Muestras de las últimas L + 3 celdas (0 = la última cerrada) y su
    /// desviación respecto a la media del nodo al cerrarlas.
    muestras: [Option<f64>; VENTANA],
    desviaciones: [Option<f64>; VENTANA],
    /// φ = [x·y, x²] de las filas de las L celdas anteriores (0 = la de la
    /// anterior).
    filas: [Option<[f64; 2]>; HORIZONTE_CELDAS],
    w: f64,
    xx: f64,
    xy: f64,
    w2: f64,
    /// Σ_t Σ_{|t−s| ≤ L} w_t w_s φ_t φ_sᵀ.
    m: [[f64; 2]; 2],
    /// Σ_t w_t² x_t⁴.
    x4: f64,
    estado: Option<EstadoLargo>,
}

/// Estado de la prueba a la escala de decisión en un nodo.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct EstadoLargo {
    /// Estadístico t; −∞ si no se puede calcular.
    pub t: f64,
    /// C-F5e: R̂, lo que revierte en τ_k la desviación votada (cota
    /// inferior; sin corregir).
    pub reversion: f64,
    /// Masa Σw y si llega a la de madurez (media memoria larga).
    pub masa: f64,
    pub madura: bool,
    /// Filas efectivas (Σw)²/Σw².
    pub n_ef: f64,
    /// C-F5e: tamaño efectivo de la corrección de muestra finita,
    /// min(n_ef, n_x).
    pub n: f64,
}

impl Largo {
    fn decaer(&mut self, n: f64) {
        let l1 = (-n / MEMORIA_LARGA_CELDAS).exp();
        let l2 = l1 * l1;
        for v in [&mut self.w, &mut self.xx, &mut self.xy] {
            *v *= l1;
        }
        for v in [&mut self.w2, &mut self.x4] {
            *v *= l2;
        }
        for fila in self.m.iter_mut() {
            for v in fila.iter_mut() {
                *v *= l2;
            }
        }
    }

    /// Celdas vacías: sin muestra ni fila; sólo decaen las sumas.
    fn vaciar(&mut self, n: u64) {
        let ventana = VENTANA as u64;
        for _ in 0..n.min(ventana) {
            self.decaer(1.0);
            self.muestras.rotate_right(1);
            self.muestras[0] = None;
            self.desviaciones.rotate_right(1);
            self.desviaciones[0] = None;
            self.filas.rotate_right(1);
            self.filas[0] = None;
        }
        if n > ventana {
            self.decaer((n - ventana) as f64);
        }
        self.estado = self.calcular();
    }

    /// Cierra una celda con muestra `b` y media del nodo `media` (las dos
    /// respecto al origen de la prueba).
    fn cerrar(&mut self, b: f64, media: Option<f64>) {
        self.decaer(1.0);
        self.muestras.rotate_right(1);
        self.muestras[0] = Some(b);
        self.desviaciones.rotate_right(1);
        self.desviaciones[0] = media.map(|m| b - m).filter(|d| d.is_finite());
        let l = HORIZONTE_CELDAS;
        let fila = match (self.desviaciones[l], self.muestras[l + 1], self.muestras[l + 2]) {
            (Some(x), Some(z), Some(z2)) => Some((x, b - (2.0 * z - z2))),
            _ => None,
        };
        let phi = fila.map(|(x, y)| [x * y, x * x]);
        if let (Some((x, y)), Some(p)) = (fila, phi) {
            self.w += 1.0;
            self.xx += x * x;
            self.xy += x * y;
            self.w2 += 1.0;
            self.x4 += p[1] * p[1];
            // Retardos 1..L: w_t·w_{t−j} = λ^j·w_t², el par (t, t−j) y su
            // simétrico. a = Σ_j λ^j φ_{t−j}.
            let lambda = (-1.0 / MEMORIA_LARGA_CELDAS).exp();
            let mut a = [0.0; 2];
            let mut lj = 1.0;
            for f in self.filas {
                lj *= lambda;
                if let Some(q) = f {
                    a[0] += lj * q[0];
                    a[1] += lj * q[1];
                }
            }
            for i in 0..2 {
                for j in 0..2 {
                    self.m[i][j] += p[i] * p[j] + p[i] * a[j] + a[i] * p[j];
                }
            }
        }
        self.filas.rotate_right(1);
        self.filas[0] = phi;
        self.estado = self.calcular();
    }

    fn calcular(&self) -> Option<EstadoLargo> {
        let w = self.w;
        if !(w > 0.0) || !(self.w2 > 0.0) {
            return None;
        }
        let madura = w >= masa_madura_larga();
        let n_ef = w * w / self.w2;
        let (t, reversion, n) = if self.xx > 0.0 {
            let r = -self.xy / self.xx;
            let v = self.m[0][0] + 2.0 * r * self.m[0][1] + r * r * self.m[1][1];
            let n_x = 3.0 * self.xx * self.xx / self.x4;
            let n = if n_x > 0.0 && n_x.is_finite() { n_ef.min(n_x) } else { n_ef };
            let t = (r - FRACCION_FRONTERA - SESGO_LARGO / n) * self.xx / v.sqrt();
            let t = if v > 0.0 && t.is_finite() { t } else { f64::NEG_INFINITY };
            (t, r, n)
        } else {
            (f64::NEG_INFINITY, f64::NAN, n_ef)
        };
        Some(EstadoLargo {
            t,
            reversion,
            masa: w,
            madura,
            n_ef,
            n,
        })
    }
}

/// Veredicto de la prueba a una τ de decisión.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Veredicto {
    /// τ_k del nodo contrastado (NaN si la τ de decisión queda bajo la
    /// malla).
    pub tau_nodo_ms: f64,
    /// Reversión demostrada: las dos pruebas del nodo maduras y sobre su
    /// crítico (t > `CRITICO` y t_largo > `CRITICO_LARGO`).
    pub demostrada: bool,
    /// Estadístico t del nodo (−∞ sin estado).
    pub t: f64,
    /// C-F5b: estadístico t de la prueba a la escala de decisión (−∞ sin
    /// estado).
    pub t_largo: f64,
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

    /// Estado de la prueba a la escala de decisión del nodo k (17 a 22), si
    /// tiene.
    pub fn estado_largo(&self, k: usize) -> Option<EstadoLargo> {
        if !(NODO_MIN..=NODO_MAX).contains(&k) {
            return None;
        }
        self.nodos[k - NODO_MIN].largo.estado
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
                t_largo: f64::NEG_INFINITY,
                base: b,
                media: f64::NAN,
                sigma: f64::NAN,
            });
        };
        let tau_nodo_ms = SPECTRUM_SCALES_MS[NODO_MIN + i];
        let (largo_ok, t_largo) = match self.nodos[i].largo.estado {
            Some(l) => (l.madura && l.t > CRITICO_LARGO, l.t),
            None => (false, f64::NEG_INFINITY),
        };
        Some(match self.nodos[i].estado {
            Some(e) => Veredicto {
                tau_nodo_ms,
                demostrada: e.madura && e.t > CRITICO && largo_ok,
                t: e.t,
                t_largo,
                base: b,
                media: e.media + o,
                sigma: e.sigma,
            },
            None => Veredicto {
                tau_nodo_ms,
                demostrada: false,
                t: f64::NEG_INFINITY,
                t_largo,
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

    /// CL-110 (EST-1 de la revisión del PR #60). Verdad: el estadístico t no
    /// cambia si a toda la base se le suma una constante, también con
    /// celdas vacías. Las sumas de retardo 1 sólo llevan las filas con fila
    /// previa; centrarlas con la masa total rompía la invariancia, y la
    /// varianza HAC encogía o quedaba ≤ 0 (t = −∞) al alejarse la base del
    /// primer par.
    #[test]
    fn t_invariante_a_desplazar_la_base() {
        let tau = SPECTRUM_SCALES_MS[17];
        let d = tau / 8.0;
        let mut semilla = 12345u64;
        let mut u = || {
            semilla = semilla.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
            ((semilla >> 11) as f64 + 0.5) / (1u64 << 53) as f64
        };
        // OU por celdas (ρ = 0,8) con una celda vacía de cada siete.
        let mut x = 0.0f64;
        let serie: Vec<(u64, f64)> = (0..400u64)
            .filter_map(|c| {
                let n = (-2.0 * u().ln()).sqrt() * (2.0 * std::f64::consts::PI * u()).cos();
                x = 0.8 * x + 0.6 * n;
                (c % 7 != 3).then_some((c, x))
            })
            .collect();
        let t_con = |k: f64| {
            let mut n = Nodo::new(tau);
            for &(c, v) in &serie {
                n.observar(((c as f64 + 0.5) * d) as u64, v + k);
            }
            n.estado.expect("hay estado").t
        };
        let t0 = t_con(0.0);
        for k in [5.0, 10.0, 20.0] {
            let tk = t_con(k);
            assert!(
                t0.is_finite() && (tk - t0).abs() <= 1e-6 * t0.abs().max(1.0),
                "desplazar {k}: t = {tk} frente a {t0}"
            );
        }
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
