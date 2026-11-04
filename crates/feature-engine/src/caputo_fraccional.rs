//! Derivada fraccional de Caputo en streaming (mandato §11.7, GLM-187).
//!
//! «Reemplaza las derivadas ordinarias por derivadas fraccionales de
//! Caputo (D_t^α) para fundir la memoria de corto, mediano y largo plazo
//! en un único operador espectral continuo.»
//!
//! Verificación previa (norma GLM-186: grep de la matemática ANTES de
//! escribir): Caputo = 0 apariciones en main; `rough_ergodic.rs` (AGY
//! Fase 7) aproxima el NÚCLEO fraccionario de volatilidad por factores
//! Markovianos (Abi Jaber–El Euch) pero NO implementa el OPERADOR
//! D_t^α f(t) — hueco real, no duplicación.
//!
//! **Forma**: la derivada de Caputo de orden α ∈ (0,1] de f es
//! D^α f(t) = (1/Γ(1−α)) ∫₀ᵗ f'(s) (t−s)^(−α) ds — la derivada ORDINARIA
//! primero, luego una convolución con memoria de ley de potencia. Su
//! forma discreta exacta por diferencias es la Grünwald–Letnikov:
//!
//! D^α f[n] = Σ_{k=0}^{W−1} w_k(α) · (f[n−k] − f[n−k−1]),
//! w_0 = 1, w_k = w_{k−1} · (k − α) / (k + 1)  →  w_k ~ k^{−(1+α)}
//!
//! Propiedades certificadas en el contrato:
//! - α = 1 ⇒ w = [1, 0, 0, ...]: la diferencia finita ORDINARIA EXACTA.
//! - α → 0 ⇒ w_k → 1: D^0 f = f (el operador degenera a identidad).
//! - Cada muestra antigua contribuye con peso ~k^{−(1+α)} > 0: la cola
//!   pesada FUNDE corto/medio/largo plazo en un solo operador — α es la
//!   perilla de la memoria (α pequeño = memoria larga).
//! - O(1) por muestra con pesos precomputados en un ring buffer.

/// Derivada fraccional de Caputo en streaming vía Grünwald–Letnikov
/// truncada a `ventana` muestras. Sin asignaciones tras la construcción.
pub struct CaputoStreaming {
    pesos: Vec<f64>,
    /// Ring de diferencias (f[n] − f[n−1]) más recientes primero.
    diffs: Vec<f64>,
    cabeza: usize,
    llenos: usize,
    anterior: f64,
    calientes: bool,
    alfa: f64,
}

impl CaputoStreaming {
    /// α ∈ (0,1] y ventana ≥ 4. Sin constructor panic (RB-P1): el Err se
    /// maneja antes de arrancar el consumidor.
    pub fn try_new(alfa: f64, ventana: usize) -> Result<Self, &'static str> {
        if !alfa.is_finite() || alfa <= 0.0 || alfa > 1.0 {
            return Err("α de Caputo debe estar en (0,1]");
        }
        if ventana < 4 {
            return Err("ventana >= 4 para memoria significativa");
        }
        Ok(Self::construir(alfa, ventana))
    }

    /// Constructor infalible (GLM 188): α=0.5, W=64 — la configuración del
    /// operador VIVO en `StatefulEngine`. α=0.5 es la frontera browniana
    /// (H=0.5: ni persistencia ni anti-persistencia — la memoria neutral);
    /// W=64 alinea el ring con la cadencia de publicación %64. Existe para
    /// que el consumidor de engine tenga una vía SIN sitio de pánico (el
    /// patrón `FirmasMaxEnt::ventana64` del GLM 175); las constantes son
    /// estáticamente válidas (α∈(0,1], W≥4) y la equivalencia exacta con
    /// `try_new(0.5, 64)` está certificada en el contrato del módulo.
    pub fn caputo_neutral64() -> Self {
        Self::construir(0.5, 64)
    }

    /// Construcción pura (sin validación): la validación vive en `try_new`
    /// y las constantes de `caputo_neutral64` son válidas por inspección.
    fn construir(alfa: f64, ventana: usize) -> Self {
        // Pesos GL: w_0 = 1; w_k = w_{k−1}·(k−α)/(k+1). La recursión es
        // estable (positivos, decrecientes) para α ∈ (0,1].
        let mut pesos = vec![0.0f64; ventana];
        pesos[0] = 1.0;
        for k in 1..ventana {
            pesos[k] = pesos[k - 1] * (k as f64 - alfa) / (k as f64 + 1.0);
        }
        Self {
            pesos,
            diffs: vec![0.0; ventana],
            cabeza: 0,
            llenos: 0,
            anterior: f64::NAN,
            calientes: false,
            alfa,
        }
    }

    /// α del operador (la perilla de la memoria).
    pub fn alfa(&self) -> f64 {
        self.alfa
    }

    /// Ingresa una muestra de la señal; devuelve la derivada fraccional
    /// D^α f en la muestra actual, o None durante el warmup (primera
    /// muestra: sin diferencia previa) o si la entrada no es finita
    /// (ausencia honesta, la muestra se DESCARTA — jamás se inyecta cero).
    pub fn push(&mut self, muestra: f64) -> Option<f64> {
        if !muestra.is_finite() {
            return None;
        }
        if !self.calientes {
            self.anterior = muestra;
            self.calientes = true;
            return None;
        }
        let diff = muestra - self.anterior;
        self.anterior = muestra;
        if !diff.is_finite() {
            return None;
        }
        // Ring: cabeza avanza HACIA ATRÁS al escribir (cabeza = más
        // reciente), y el rezago k se lee en (cabeza+k) % n.
        let n = self.diffs.len();
        self.cabeza = (self.cabeza + n - 1) % n;
        self.diffs[self.cabeza] = diff;
        if self.llenos < n {
            self.llenos += 1;
        }
        let mut suma = 0.0f64;
        for k in 0..self.llenos {
            suma += self.pesos[k] * self.diffs[(self.cabeza + k) % n];
        }
        if !suma.is_finite() {
            return None;
        }
        Some(suma)
    }

    /// Muestras de diferencia acumuladas en el ring.
    pub fn llenos(&self) -> usize {
        self.llenos
    }

    /// El peso del rezago k (expuesto para auditoría del contrato).
    pub fn peso(&self, k: usize) -> Option<f64> {
        self.pesos.get(k).copied()
    }
}
