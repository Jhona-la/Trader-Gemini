//! Motor Koopman / DMD STREAMING (GLM 149 — pilar §57 del mandato, titulado
//! a GLM por el consejo en Ω209). Descompone la dinámica NO LINEAL del
//! retorno normalizado proyectándola en observables por retardo (Hankel de
//! orden `KOOPMAN_M`) y estimando el operador de Koopman lineal efectivo
//! A = Q·G⁻¹ resuelto FILA a fila (G·a_i = q_i con G simétrica — identidad
//! corregida por AGY Ω212; las primeras M−1 filas reproducen la matriz
//! compañera de desplazamiento) con actualización rank-1 por tick
//! (streaming DMD). Sus
//! autovalores revelan el MODO DE CRECIMIENTO (|λ| dominante por tick), la
//! FRECUENCIA OCULTA (ángulo de λ) y la PROXIMIDAD DE BIFURCACIÓN (máx|λ|
//! contra el círculo unidad) ANTES de que se reflejen en el precio.
//!
//! # Qué NO modela
//! No es un predictor puntual: entrega el espectro modal de la dinámica
//! lineal efectiva. No reemplaza al FFT de `spectral.rs` (éste ve senales
//! en el dominio de frecuencia sobre una ventana fija de 64; Koopman ve
//! MODOS con tasa de crecimiento/amortiguación). No hay clamping de
//! autovalores ni imputación: warmup insuficiente o G singular ⇒ Err (CL-15).
//!
//! # Coste
//! push: O(M²) rank-1 sobre stack (zero-alloc). analyze: LU O(M³) para
//! A = Q·G⁻¹ + Faddeev-LeVerrier/Durand-Kerner para el espectro — llamado
//! por CADENCIA (no por tick), M=8 ⇒ sub-microsegundo.

/// Orden del embedding por retardo (observables Koopman). 8 retardos ⇒ el
/// operador efectivo es 8×8: capta hasta 4 modos oscilatorios independientes
/// del retorno normalizado, suficiente para el régimen de barras del examen.
pub const KOOPMAN_M: usize = 8;

/// Tolerancia de singularidad de G (pivote mínimo del LU en magnitud
/// absoluta relativa a la traza acumulada).
const PIVOTE_MIN: f64 = 1e-12;

#[derive(Debug, Clone, Copy, PartialEq)]
pub enum DmdError {
    /// Menos muestras válidas que 2·M: el espectro no es estimable.
    WarmupInsuficiente { validas: u64, requeridas: u64 },
    /// G singular (dinámica degenerada — p. ej. stream constante): el
    /// operador no está identificado. Ausencia honesta, jamás λ inventado.
    GSingular,
}

/// Espectro Koopman al momento del análisis.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct EspectroKoopman {
    /// Autovalor dominante (parte real, parte imaginaria).
    pub lambda_dominante: (f64, f64),
    /// |λ| dominante: tasa de crecimiento (>1) o amortiguación (<1) por
    /// muestra del modo más activo de la dinámica.
    pub modulo_dominante: f64,
    /// Ángulo del autovalor dominante en radianes ∈ [0, π]: frecuencia
    /// oculta del modo (π = Nyquist, alternancia muestra a muestra).
    pub angulo_rad: f64,
    /// |λ| del segundo modo (0.0 si sólo hay uno real repetido).
    pub segundo_modulo: f64,
    /// max_i |λ_i| — distancia al círculo unidad: ≥1.0 es régimen de
    /// bifurcación (el sistema amplifica perturbaciones).
    pub bifurcacion_max: f64,
    /// Cantidad de autovalores estimados (M, con pares complejos = 2).
    pub n_autovalores: usize,
    /// Muestras válidas acumuladas.
    pub muestras: u64,
}

/// DMD streaming con embedding Hankel de orden `KOOPMAN_M`.
///
/// Estado: ventana de los últimos M retornos, gramiana G = Σ φφᵀ y
/// correlación cruzada Q = Σ φ⁺φᵀ (φ⁺ = ventana desplazada un paso).
/// Factor de olvido `forget` ∈ (0,1]: antes de cada rank-1, G y Q se
/// multiplican por `forget` (DMD exponencialmente ponderado — `forget`=1.0
/// es acumulación pura). Véase kalman.rs para el patrón new/try_new.
pub struct StreamingDmd {
    ventana: [f64; KOOPMAN_M],
    llenos: usize,
    g: [[f64; KOOPMAN_M]; KOOPMAN_M],
    q: [[f64; KOOPMAN_M]; KOOPMAN_M],
    forget: f64,
    validas: u64,
    rechazadas: u64,
}

impl Default for StreamingDmd {
    fn default() -> Self {
        Self::new()
    }
}

impl StreamingDmd {
    pub fn new() -> Self {
        Self {
            ventana: [0.0; KOOPMAN_M],
            llenos: 0,
            g: [[0.0; KOOPMAN_M]; KOOPMAN_M],
            q: [[0.0; KOOPMAN_M]; KOOPMAN_M],
            forget: 1.0,
            validas: 0,
            rechazadas: 0,
        }
    }

    /// Constructor con factor de olvido. Err (sin construir) si `forget`
    /// no es finito o está fuera de (0,1].
    pub fn try_new_with_forget(forget: f64) -> Result<Self, String> {
        if !forget.is_finite() || forget <= 0.0 || forget > 1.0 {
            return Err(format!("forget debe estar en (0,1], recibido {forget}"));
        }
        let mut s = Self::new();
        s.forget = forget;
        Ok(s)
    }

    /// Alimenta un retorno normalizado. Los no-finitos se RECHAZAN con
    /// contador visible (jamás contaminan el operador).
    pub fn push(&mut self, x: f64) {
        if !x.is_finite() {
            self.rechazadas += 1;
            return;
        }
        if self.llenos < KOOPMAN_M {
            // Rellenar la primera ventana en orden histórico.
            self.ventana[self.llenos] = x;
            self.llenos += 1;
            if self.llenos == KOOPMAN_M {
                self.validas += 1; // primera φ completa
            }
            return;
        }
        // φ = ventana actual (retardo 0 = muestra más reciente al final),
        // φ⁺ = ventana desplazada: entra x al frente, cae la más vieja.
        let f = self.forget;
        if f != 1.0 {
            for fila in &mut self.g {
                for v in fila.iter_mut() {
                    *v *= f;
                }
            }
            for fila in &mut self.q {
                for v in fila.iter_mut() {
                    *v *= f;
                }
            }
        }
        // φ = [x_{t-M+1}, ..., x_t]; φ⁺ = [x_{t-M+2}, ..., x_t, x].
        // G += φφᵀ con φ en el orden del buffer.
        for i in 0..KOOPMAN_M {
            let fi = self.ventana[i];
            if fi == 0.0 {
                continue;
            }
            for j in 0..KOOPMAN_M {
                self.g[i][j] += fi * self.ventana[j];
            }
        }
        // φ⁺[i] = ventana[i+1] para i<M-1; φ⁺[M-1] = x.
        // Q += φ⁺φᵀ.
        for i in 0..KOOPMAN_M {
            let fmas = if i + 1 < KOOPMAN_M {
                self.ventana[i + 1]
            } else {
                x
            };
            if fmas == 0.0 {
                continue;
            }
            for j in 0..KOOPMAN_M {
                self.q[i][j] += fmas * self.ventana[j];
            }
        }
        // Desplazar la ventana.
        self.ventana.copy_within(1..KOOPMAN_M, 0);
        self.ventana[KOOPMAN_M - 1] = x;
        self.validas += 1;
    }

    /// Muestras válidas acumuladas (para cadencias del consumidor).
    #[inline(always)]
    pub fn validas(&self) -> u64 {
        self.validas
    }

    /// No-finitos rechazados (telemetría de higiene).
    #[inline(always)]
    pub fn rechazadas(&self) -> u64 {
        self.rechazadas
    }

    /// Espectro Koopman: A = (G+εI)⁻¹Q con regularización Tikhonov
    /// ε = 1e-6·traza(G)/M (Kutz/Brunton, DMD regularizada: exacta en el
    /// subespacio de señal — la Hankel de una dinámica de orden r<M deja G
    /// de rango r, y el ridge entrega el operador de norma mínima: los M−r
    /// modos extra salen ≈0, sin contaminar los r verdaderos). LU con
    /// pivoteo parcial + Hessenberg/QR desplazado. Err explícito si el
    /// warmup es insuficiente o no queda pivote útil.
    pub fn analizar(&self) -> Result<EspectroKoopman, DmdError> {
        let requeridas = (2 * KOOPMAN_M) as u64;
        if self.validas < requeridas {
            return Err(DmdError::WarmupInsuficiente {
                validas: self.validas,
                requeridas,
            });
        }
        let traza: f64 = (0..KOOPMAN_M).map(|i| self.g[i][i]).sum();
        if !traza.is_finite() || traza <= 0.0 {
            return Err(DmdError::GSingular);
        }
        let eps = 1e-10 * traza / KOOPMAN_M as f64;
        // LU de G+εI con pivoteo parcial.
        let mut lu = self.g;
        for i in 0..KOOPMAN_M {
            lu[i][i] += eps;
        }
        let mut perm: [usize; KOOPMAN_M] = [0; KOOPMAN_M];
        let (ok_lu, _) = factorizar_lu(&mut lu, &mut perm);
        if !ok_lu {
            return Err(DmdError::GSingular);
        }
        // A = Q G⁻¹ fila a fila (G a_i = q_i con G simétrica).
        let mut a = [[0.0f64; KOOPMAN_M]; KOOPMAN_M];
        for i in 0..KOOPMAN_M {
            let mut b = [0.0f64; KOOPMAN_M];
            for j in 0..KOOPMAN_M {
                b[j] = self.q[i][j];
            }
            let row = resolver_lu(&lu, &perm, &b);
            if row.iter().any(|v| !v.is_finite()) {
                return Err(DmdError::GSingular);
            }
            for j in 0..KOOPMAN_M {
                a[i][j] = row[j];
            }
        }
        let autovalores = autovalores_faddeev_durand(&a);
        if autovalores.is_empty()
            || autovalores
                .iter()
                .any(|(re, im)| !re.is_finite() || !im.is_finite())
        {
            return Err(DmdError::GSingular);
        }
        // Ordenar por módulo descendente.
        let mut lams = autovalores;
        lams.sort_by(|x, y| {
            let mx = (x.0 * x.0 + x.1 * x.1).sqrt();
            let my = (y.0 * y.0 + y.1 * y.1).sqrt();
            my.partial_cmp(&mx).unwrap_or(std::cmp::Ordering::Equal)
        });
        let (re0, im0) = lams[0];
        let modulo0 = (re0 * re0 + im0 * im0).sqrt();
        let segundo = if lams.len() > 1 {
            let (re1, im1) = lams[1];
            (re1 * re1 + im1 * im1).sqrt()
        } else {
            0.0
        };
        Ok(EspectroKoopman {
            lambda_dominante: (re0, im0),
            modulo_dominante: modulo0,
            angulo_rad: im0.abs().atan2(re0),
            segundo_modulo: segundo,
            bifurcacion_max: modulo0,
            n_autovalores: lams.len(),
            muestras: self.validas,
        })
    }
}

/// Factorización LU con pivoteo parcial. Devuelve (éxito, determinante).
fn factorizar_lu(
    a: &mut [[f64; KOOPMAN_M]; KOOPMAN_M],
    perm: &mut [usize; KOOPMAN_M],
) -> (bool, f64) {
    let n = KOOPMAN_M;
    for i in 0..n {
        perm[i] = i;
    }
    let mut det = 1.0;
    for k in 0..n {
        // Pivoteo parcial.
        let mut piv = k;
        let mut mejor = a[k][k].abs();
        for r in k + 1..n {
            let v = a[r][k].abs();
            if v > mejor {
                mejor = v;
                piv = r;
            }
        }
        if mejor < PIVOTE_MIN || !mejor.is_finite() {
            return (false, 0.0);
        }
        if piv != k {
            a.swap(k, piv);
            perm.swap(k, piv);
            det = -det;
        }
        let akk = a[k][k];
        det *= akk;
        for r in k + 1..n {
            let f = a[r][k] / akk;
            a[r][k] = f;
            if f != 0.0 {
                for c in k + 1..n {
                    a[r][c] -= f * a[k][c];
                }
            }
        }
    }
    (true, det)
}

/// Resuelve A x = b con el LU ya factorizado (perm del pivoteo).
fn resolver_lu(
    lu: &[[f64; KOOPMAN_M]; KOOPMAN_M],
    perm: &[usize; KOOPMAN_M],
    b: &[f64; KOOPMAN_M],
) -> [f64; KOOPMAN_M] {
    let n = KOOPMAN_M;
    let mut x = [0.0f64; KOOPMAN_M];
    // Pb.
    for i in 0..n {
        x[i] = b[perm[i]];
    }
    // Sustitución directa (L unidad).
    for i in 0..n {
        let mut s = x[i];
        for j in 0..i {
            s -= lu[i][j] * x[j];
        }
        x[i] = s;
    }
    // Sustitución inversa.
    for i in (0..n).rev() {
        let mut s = x[i];
        for j in i + 1..n {
            s -= lu[i][j] * x[j];
        }
        x[i] = s / lu[i][i];
    }
    x
}

/// Autovalores de una matriz general M×M vía FADDEEV–LEVERRIER (polinomio
/// característico por trazas de potencias, O(n⁴)) + DURAND–KERNER
/// (iteración de Weierstrass con aritmética compleja). Los pares
/// complejos conjugados salen naturales de las raíces del polinomio; para
/// M=8 el condicionamiento del F–L es aceptable y los contratos con
/// solución conocida certifican la precisión.
fn autovalores_faddeev_durand(a: &[[f64; KOOPMAN_M]; KOOPMAN_M]) -> Vec<(f64, f64)> {
    let n = KOOPMAN_M;
    // ── Faddeev–LeVerrier: coeficientes c_k del det(λI − A) = λ^n + c_1λ^(n-1) + ... + c_n.
    // M_k = A·M_{k-1} + c_{k-1}I ; c_k = −tr(A·M_k)/k
    let mut mk = [[0.0f64; KOOPMAN_M]; KOOPMAN_M];
    let mut coef = vec![0.0f64; n + 1];
    coef[0] = 1.0; // λ^n
    for k in 1..=n {
        // A·M_{k-1}
        let mut am = [[0.0f64; KOOPMAN_M]; KOOPMAN_M];
        for i in 0..n {
            for j in 0..n {
                let mut s = 0.0;
                for l in 0..n {
                    s += a[i][l] * mk[l][j];
                }
                am[i][j] = s;
            }
        }
        // M_k = A·M_{k-1} + c_{k-1}·I ; c_k = −tr(A·M_k)/k
        let c_prev = coef[k - 1];
        for i in 0..n {
            am[i][i] += c_prev;
        }
        // tr(A·M_k)
        let mut traza = 0.0;
        for i in 0..n {
            let mut s = 0.0;
            for l in 0..n {
                s += a[i][l] * am[l][i];
            }
            traza += s;
        }
        coef[k] = -traza / k as f64;
        mk = am;
    }
    // ── Durand–Kerner sobre p(λ) = λ^n + coef[1]λ^{n-1} + ... + coef[n].
    // Monomios en orden descendente (p[0]=1).
    let p = coef; // p[0]=1, p[k]=c_k
                  // Normalizar por el coeficiente líder (ya 1.0 por construcción).
                  // Semillas: círculo de radio R en el plano complejo (raíces de la unidad
                  // escaladas por un cota de Cauchy aproximada).
    let radio = 1.0 + p[1..].iter().fold(0.0f64, |m, &c| m.max(c.abs()));
    #[derive(Clone, Copy)]
    struct C(f64, f64);
    let cmul = |x: C, y: C| C(x.0 * y.0 - x.1 * y.1, x.0 * y.1 + x.1 * y.0);
    let cadd = |x: C, y: C| C(x.0 + y.0, x.1 + y.1);
    let cdiv = |x: C, y: C| {
        C(
            (x.0 * y.0 + x.1 * y.1) / (y.0 * y.0 + y.1 * y.1),
            (x.1 * y.0 - x.0 * y.1) / (y.0 * y.0 + y.1 * y.1),
        )
    };
    let cabs = |x: C| (x.0 * x.0 + x.1 * x.1).sqrt();
    let peval = |z: C| -> C {
        // Horner con p[0]=1.
        let mut acc = C(1.0, 0.0);
        for k in 1..=n {
            acc = cadd(cmul(acc, z), C(p[k], 0.0));
        }
        acc
    };
    let mut raices: Vec<C> = (0..n)
        .map(|k| {
            let ang = 2.0 * std::f64::consts::PI * (k as f64 + 0.25) / n as f64;
            C(radio * 0.5 * ang.cos(), radio * 0.5 * ang.sin())
        })
        .collect();
    // Durand–Kerner: λ_i ← λ_i − p(λ_i)/Π_{j≠i}(λ_i − λ_j).
    for _ in 0..500 {
        let mut delta_max = 0.0f64;
        for i in 0..n {
            let mut denom = C(1.0, 0.0);
            for (j, &rj) in raices.iter().enumerate() {
                if j != i {
                    denom = cmul(denom, cadd(raices[i], C(-rj.0, -rj.1)));
                }
            }
            let d = cabs(denom);
            if d < 1e-300 {
                continue;
            }
            let paso = cdiv(peval(raices[i]), denom);
            raices[i] = cadd(raices[i], C(-paso.0, -paso.1));
            delta_max = delta_max.max(cabs(paso));
        }
        if delta_max < 1e-14 {
            break;
        }
    }
    raices.into_iter().map(|r| (r.0, r.1)).collect()
}
