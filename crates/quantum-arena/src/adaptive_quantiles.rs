use std::f64;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};

/// Capacidad de la ventana deslizante acotada para estimación de cuantiles (F1-079 / C-F2.2).
/// 256 muestras de tipo `f64` consumen exactamente 2 048 bytes (2 KiB), residiendo íntegramente
/// en L1 cache sin alocación en heap (`zero-alloc`).
pub const CAPACIDAD_VENTANA_CUANTIL: usize = 256;

/// Mínimo número de muestras observadas para considerar madura la estimación del cuantil empírico.
/// Evita que puertas de entrada vivas tomen decisiones críticas basadas en pocas observaciones de ruido (F1-079).
pub const MUESTRAS_MADURAS_CUANTIL: usize = 30;

/// Estimador heurístico P² del cuantil de todas las observaciones aceptadas.
/// Mantiene cinco marcadores: memoria y trabajo O(1) por observación.
/// No incorpora olvido temporal, ventana móvil ni intervalos de confianza.
/// Actualizar un cuantil no calibra por sí solo un umbral predictivo.
#[derive(Debug, Clone)]
pub struct P2Quantile {
    pub p: f64,       // Percentil objetivo (e.g., 0.80 para P80)
    pub count: u64,   // Número total de observaciones
    pub q: [f64; 5],  // Alturas de los 5 marcadores
    pub n: [i64; 5],  // Posiciones actuales de los 5 marcadores
    pub np: [f64; 5], // Posiciones deseadas de los 5 marcadores
    pub dn: [f64; 5], // Incrementos deseados por cada observación
    pub initialized: bool,
}

impl P2Quantile {
    /// Adaptador histórico: recorta p a [0.01,0.99], incluidos infinitos.
    /// NaN se rechaza al construir; antes congelaba silenciosamente el ajuste.
    /// Para validar una probabilidad sin modificarla, usar `try_new`.
    ///
    /// # Panics
    /// Si p es NaN. Los consumidores configurables deben usar `try_new`.
    pub fn new(p: f64) -> Self {
        Self::try_new(p.clamp(0.01, 0.99)).expect("quantile probability must not be NaN")
    }

    /// Construye sin recortar p: el dominio de P² es 0 < p < 1.
    /// Los extremos requieren estimadores de mínimo/máximo, no este marcador.
    pub fn try_new(p: f64) -> Result<Self, &'static str> {
        if !p.is_finite() || p <= 0.0 || p >= 1.0 {
            return Err("quantile probability must be finite and strictly between 0 and 1");
        }
        Ok(Self {
            p,
            count: 0,
            q: [0.0; 5],
            n: [1, 2, 3, 4, 5],
            np: [
                1.0,
                1.0 + 2.0 * p,
                1.0 + 4.0 * p,
                3.0 + 2.0 * p,
                5.0,
            ],
            dn: [
                0.0,
                p / 2.0,
                p,
                (1.0 + p) / 2.0,
                1.0,
            ],
            initialized: false,
        })
    }

    #[inline(always)]
    pub fn update(&mut self, x: f64) {
        if !x.is_finite() {
            return;
        }

        if !self.initialized {
            if self.count < 5 {
                self.q[self.count as usize] = x;
                self.count += 1;
                if self.count == 5 {
                    self.q
                        .sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
                    self.initialized = true;
                }
                return;
            }
        }

        self.count += 1;

        // Encontrar k tal que q[k] <= x < q[k+1]
        let k = if x < self.q[0] {
            self.q[0] = x;
            0
        } else if x >= self.q[4] {
            self.q[4] = x;
            3
        } else {
            let mut found = 0;
            for i in 0..4 {
                if self.q[i] <= x && x < self.q[i + 1] {
                    found = i;
                    break;
                }
            }
            found
        };

        // Incrementar posiciones a la derecha de k
        for i in (k + 1)..5 {
            self.n[i] += 1;
        }

        // Actualizar posiciones deseadas
        for i in 0..5 {
            self.np[i] += self.dn[i];
        }

        // Ajustar alturas de marcadores 1, 2 y 3 si es necesario
        for i in 1..4 {
            let d = self.np[i] - self.n[i] as f64;

            if (d >= 1.0 && self.n[i + 1] - self.n[i] > 1)
                || (d <= -1.0 && self.n[i - 1] - self.n[i] < -1)
            {
                let d_sign = if d > 0.0 { 1 } else { -1 };
                let q_est = self.parabolic_interpolation(i, d_sign as f64);

                if self.q[i - 1] < q_est && q_est < self.q[i + 1] {
                    self.q[i] = q_est;
                } else {
                    self.q[i] = self.linear_interpolation(i, d_sign as f64);
                }

                self.n[i] += d_sign;
            }
        }
    }

    #[inline(always)]
    fn parabolic_interpolation(&self, i: usize, d: f64) -> f64 {
        let n_i = self.n[i] as f64;
        let n_ip1 = self.n[i + 1] as f64;
        let n_im1 = self.n[i - 1] as f64;

        let q_i = self.q[i];
        let q_ip1 = self.q[i + 1];
        let q_im1 = self.q[i - 1];

        let predict = |lower: f64, center: f64, upper: f64| {
            center + (d / (n_ip1 - n_im1))
                * ((n_i - n_im1 + d) * (upper - center) / (n_ip1 - n_i)
                    + (n_ip1 - n_i - d) * (center - lower) / (n_i - n_im1))
        };
        let estimate = predict(q_im1, q_i, q_ip1);
        if estimate.is_finite() {
            return estimate;
        }
        // P² is homogeneous in marker heights. Normalize only if intermediate
        // arithmetic overflowed, retaining ordinary-range numerical behavior.
        // A true overshoot is still rejected by the neighboring-marker check.
        let scale = q_im1.abs().max(q_i.abs()).max(q_ip1.abs());
        predict(q_im1 / scale, q_i / scale, q_ip1 / scale) * scale
    }

    #[inline(always)]
    fn linear_interpolation(&self, i: usize, d: f64) -> f64 {
        let idx_next = if d > 0.0 { i + 1 } else { i - 1 };
        let n_diff = (self.n[idx_next] - self.n[i]) as f64;
        let q_diff = self.q[idx_next] - self.q[i];

        if q_diff.is_finite() {
            self.q[i] + d * (q_diff / n_diff)
        } else {
            // Opposite-sign finite endpoints can have an infinite difference.
            // The move is a convex combination (0 < weight < 1), so the
            // result remains between those endpoints without that subtraction.
            let weight = d / n_diff;
            (1.0 - weight) * self.q[i] + weight * self.q[idx_next]
        }
    }

    #[inline(always)]
    pub fn value(&self) -> f64 {
        if !self.initialized {
            if self.count == 0 {
                return 0.0;
            }
            let mut temp = self.q;
            let temp = &mut temp[..self.count as usize];
            temp.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
            let idx = ((self.count as f64 - 1.0) * self.p) as usize;
            return temp[idx.min(temp.len() - 1)];
        }
        self.q[2]
    }
}

/// Estimador de cuantil con memoria acotada (ventana deslizante / ring buffer de tamaño `N`).
///
/// Incorpora olvido temporal estricto (F1-079 / C-F2.2): tras `N` observaciones, cualquier régimen
/// anterior queda 100 % descartado, eliminando la congelación por edad de sesión que sufría el estimador P².
/// La memoria reside enteramente en el stack/inline (`[f64; N]`), garantizando cero alocación en heap (`zero-alloc`).
/// La evaluación utiliza `select_nth_unstable_by` (introselect) con complejidad promedio O(N) e invocado de forma
/// perezosa con caché lock-free (`AtomicU64` / `AtomicBool`), manteniendo la estructura segura para hilos (`Send + Sync`).
pub struct RingQuantile<const N: usize = CAPACIDAD_VENTANA_CUANTIL> {
    pub p: f64,
    pub buffer: [f64; N],
    pub head: usize,
    pub len: usize,
    pub total_observed: u64,
    pub cached_value_bits: AtomicU64,
    pub dirty: AtomicBool,
    pub initialized: bool,
}

impl<const N: usize> RingQuantile<N> {
    /// Construye un estimador de cuantil en ventana deslizante para el percentil `p` ∈ (0, 1).
    /// Si `p` no es válido o es NaN, recurre de forma segura a 0.50 (mediana) sin entrar en pánico (RB-P1).
    pub fn new(p: f64) -> Self {
        let p_valido = if p.is_finite() && p > 0.0 && p < 1.0 {
            p
        } else {
            0.50
        };
        Self {
            p: p_valido,
            buffer: [0.0; N],
            head: 0,
            len: 0,
            total_observed: 0,
            cached_value_bits: AtomicU64::new(0.0f64.to_bits()),
            dirty: AtomicBool::new(false),
            initialized: false,
        }
    }

    /// Valida que la probabilidad `p` sea finita y pertenezca al intervalo abierto (0, 1).
    pub fn try_new(p: f64) -> Result<Self, &'static str> {
        if !p.is_finite() || p <= 0.0 || p >= 1.0 {
            return Err("quantile probability must be finite and strictly between 0 and 1");
        }
        Ok(Self {
            p,
            buffer: [0.0; N],
            head: 0,
            len: 0,
            total_observed: 0,
            cached_value_bits: AtomicU64::new(0.0f64.to_bits()),
            dirty: AtomicBool::new(false),
            initialized: false,
        })
    }

    /// Actualiza el buffer circular con una nueva observación. Muestras NaN o Infinitas se descartan sin efectos secundarios.
    #[inline(always)]
    pub fn update(&mut self, x: f64) {
        if !x.is_finite() {
            return;
        }
        self.buffer[self.head] = x;
        self.head = (self.head + 1) % N;
        if self.len < N {
            self.len += 1;
        }
        self.total_observed = self.total_observed.saturating_add(1);
        self.initialized = self.len >= MUESTRAS_MADURAS_CUANTIL;
        self.dirty.store(true, Ordering::Release);
    }

    /// Retorna el valor del cuantil actual. Si hay nuevas observaciones pendientes, recomputa
    /// el orden estadístico mediante selección parcial O(N) sin alocar en heap.
    #[inline(always)]
    pub fn value(&self) -> f64 {
        if self.dirty.load(Ordering::Acquire) {
            if self.len == 0 {
                return 0.0;
            }
            let mut scratch = [0.0f64; N];
            scratch[..self.len].copy_from_slice(&self.buffer[..self.len]);
            let idx = ((self.len as f64 - 1.0) * self.p).round() as usize;
            let idx = idx.min(self.len - 1);
            let (_, median, _) = scratch[..self.len].select_nth_unstable_by(idx, |a, b| {
                a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal)
            });
            let val = *median;
            self.cached_value_bits.store(val.to_bits(), Ordering::Release);
            self.dirty.store(false, Ordering::Release);
            val
        } else {
            f64::from_bits(self.cached_value_bits.load(Ordering::Relaxed))
        }
    }

    /// Indica si el estimador ha acumulado suficientes observaciones (>= MUESTRAS_MADURAS_CUANTIL).
    #[inline(always)]
    pub fn is_initialized(&self) -> bool {
        self.initialized
    }

    /// Retorna el total histórico de observaciones válidas recibidas.
    #[inline(always)]
    pub fn count(&self) -> u64 {
        self.total_observed
    }

    /// Retorna la cantidad actual de muestras válidas en la ventana deslizante (máximo `N`).
    #[inline(always)]
    pub fn window_len(&self) -> usize {
        self.len
    }
}

impl<const N: usize> Clone for RingQuantile<N> {
    fn clone(&self) -> Self {
        Self {
            p: self.p,
            buffer: self.buffer,
            head: self.head,
            len: self.len,
            total_observed: self.total_observed,
            cached_value_bits: AtomicU64::new(self.cached_value_bits.load(Ordering::Relaxed)),
            dirty: AtomicBool::new(self.dirty.load(Ordering::Relaxed)),
            initialized: self.initialized,
        }
    }
}

impl<const N: usize> Default for RingQuantile<N> {
    fn default() -> Self {
        Self::new(0.50)
    }
}

impl<const N: usize> std::fmt::Debug for RingQuantile<N> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("RingQuantile")
            .field("p", &self.p)
            .field("head", &self.head)
            .field("len", &self.len)
            .field("total_observed", &self.total_observed)
            .field("value", &self.value())
            .field("initialized", &self.initialized)
            .finish()
    }
}

/// Motor de cuantiles adaptativos con ventana acotada y olvido temporal perfecto (F1-079).
///
/// Mantiene el percentil 80 de |OFI|, percentil 80 de |OBI| y percentil 85 de |composite_score|.
/// Las muestras maduras (>=30) activan el cuantil empírico reciente de 256 muestras sin hardcodes.
/// Mientras no haya madurez, utiliza el fallback derivado de la curva del genoma / arena.config.
///
/// `atr_p50` ha sido extirpado al no tener consumidores en el grafo de trading vivo (Fallo Tipo 2).
#[derive(Debug, Clone)]
pub struct AdaptiveQuantileEngine {
    pub ofi_p80: RingQuantile<CAPACIDAD_VENTANA_CUANTIL>,
    pub obi_p80: RingQuantile<CAPACIDAD_VENTANA_CUANTIL>,
    pub holistic_p85: RingQuantile<CAPACIDAD_VENTANA_CUANTIL>,
    pub fallback_ofi: f64,
    pub fallback_obi: f64,
    pub fallback_holistic: f64,
}

impl AdaptiveQuantileEngine {
    /// Construye el motor con percentiles canónicos (80/80/85) y respaldos base.
    pub fn new() -> Self {
        Self {
            ofi_p80: RingQuantile::new(0.80),
            obi_p80: RingQuantile::new(0.80),
            holistic_p85: RingQuantile::new(0.85),
            fallback_ofi: 0.15,
            fallback_obi: 0.15,
            fallback_holistic: 0.40,
        }
    }

    /// Construye el motor con fallbacks derivados explícitamente de la configuración genómica.
    pub fn with_fallbacks(fallback_ofi: f64, fallback_obi: f64, fallback_holistic: f64) -> Self {
        Self {
            ofi_p80: RingQuantile::new(0.80),
            obi_p80: RingQuantile::new(0.80),
            holistic_p85: RingQuantile::new(0.85),
            fallback_ofi,
            fallback_obi,
            fallback_holistic,
        }
    }

    /// Sincroniza los respaldos fríos desde los genes y curvas del genoma.
    pub fn configurar_fallbacks(&mut self, fallback_ofi: f64, fallback_obi: f64, fallback_holistic: f64) {
        if fallback_ofi.is_finite() && fallback_ofi > 0.0 {
            self.fallback_ofi = fallback_ofi;
        }
        if fallback_obi.is_finite() && fallback_obi > 0.0 {
            self.fallback_obi = fallback_obi;
        }
        if fallback_holistic.is_finite() && fallback_holistic > 0.0 {
            self.fallback_holistic = fallback_holistic;
        }
    }

    /// Actualiza los tres estimadores de magnitud (|OFI|, |OBI|, |score|) en O(1) memoria.
    /// El parámetro `_atr_pct` se preserva por retrocompatibilidad de firma con el núcleo.
    #[inline(always)]
    pub fn update(&mut self, ofi: f64, obi: f64, holistic: f64, _atr_pct: f64) {
        self.ofi_p80.update(ofi.abs());
        self.obi_p80.update(obi.abs());
        self.holistic_p85.update(holistic.abs());
    }

    /// Actualización directa de los tres canales de cuantiles adaptativos.
    #[inline(always)]
    pub fn update_3(&mut self, ofi: f64, obi: f64, holistic: f64) {
        self.ofi_p80.update(ofi.abs());
        self.obi_p80.update(obi.abs());
        self.holistic_p85.update(holistic.abs());
    }

    #[inline(always)]
    pub fn dynamic_ofi_threshold(&self) -> f64 {
        if self.ofi_p80.initialized {
            self.ofi_p80.value().max(f64::MIN_POSITIVE)
        } else {
            self.fallback_ofi
        }
    }

    #[inline(always)]
    pub fn dynamic_obi_threshold(&self) -> f64 {
        if self.obi_p80.initialized {
            self.obi_p80.value().max(f64::MIN_POSITIVE)
        } else {
            self.fallback_obi
        }
    }

    #[inline(always)]
    pub fn dynamic_holistic_threshold(&self) -> f64 {
        if self.holistic_p85.initialized {
            self.holistic_p85.value().max(f64::MIN_POSITIVE)
        } else {
            self.fallback_holistic
        }
    }

    /// Umbral OFI con respaldo frío explícito derivado del genoma / config (F1-079).
    #[inline(always)]
    pub fn dynamic_ofi_threshold_con_fallback(&self, fallback: f64) -> f64 {
        if self.ofi_p80.initialized {
            self.ofi_p80.value().max(f64::MIN_POSITIVE)
        } else {
            fallback
        }
    }

    /// Umbral OBI con respaldo frío explícito derivado de la curva del genoma obi_threshold_at_tau (F1-079).
    #[inline(always)]
    pub fn dynamic_obi_threshold_con_fallback(&self, fallback: f64) -> f64 {
        if self.obi_p80.initialized {
            self.obi_p80.value().max(f64::MIN_POSITIVE)
        } else {
            fallback
        }
    }

    /// Umbral de score holístico con respaldo frío explícito derivado del genoma tech_threshold (F1-079).
    #[inline(always)]
    pub fn dynamic_holistic_threshold_con_fallback(&self, fallback: f64) -> f64 {
        if self.holistic_p85.initialized {
            self.holistic_p85.value().max(f64::MIN_POSITIVE)
        } else {
            fallback
        }
    }
}

impl Default for AdaptiveQuantileEngine {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_p2_quantile_convergence() {
        let mut p50 = P2Quantile::new(0.50);
        for i in 1..=100 {
            p50.update(i as f64);
        }
        assert!(p50.initialized);
        let val = p50.value();
        // El percentil 50 de 1..100 debe estar en el rango [40, 60]
        assert!(
            val >= 40.0 && val <= 60.0,
            "P50 converge cerca de 50: {}",
            val
        );
    }

    #[test]
    fn test_p2_quantile_nan_and_uninitialized_value() {
        let mut p80 = P2Quantile::new(0.80);
        assert_eq!(p80.value(), 0.0);
        p80.update(f64::NAN);
        p80.update(f64::INFINITY);
        assert!(!p80.initialized);
        assert_eq!(p80.count, 0);

        // 3 samples (still uninitialized)
        p80.update(10.0);
        p80.update(20.0);
        p80.update(30.0);
        assert!(!p80.initialized);
        assert!(p80.value() >= 10.0 && p80.value() <= 30.0);
    }

    #[test]
    fn test_adaptive_quantile_engine_initialization_and_update() {
        let mut engine = AdaptiveQuantileEngine::new();
        // Fase fría: antes de alcanzar MUESTRAS_MADURAS_CUANTIL (30), devuelve el fallback
        assert_eq!(engine.dynamic_ofi_threshold(), 0.15);
        assert_eq!(engine.dynamic_obi_threshold(), 0.15);
        assert_eq!(engine.dynamic_holistic_threshold(), 0.40);
        assert!(!engine.ofi_p80.initialized);

        // Muestreo inicial inmaduro (10 muestras < 30)
        for i in 0..10 {
            engine.update(i as f64 * 0.1, 0.2, 0.5, 0.01);
        }
        assert!(!engine.ofi_p80.initialized, "10 muestras no deben marcar inicializado");
        assert_eq!(engine.dynamic_ofi_threshold(), 0.15);
        assert_eq!(engine.dynamic_obi_threshold(), 0.15);

        // Verificación de fallback frío explícito derivado del genoma
        assert_eq!(engine.dynamic_obi_threshold_con_fallback(0.28), 0.28);
        assert_eq!(engine.dynamic_ofi_threshold_con_fallback(0.22), 0.22);
        assert_eq!(engine.dynamic_holistic_threshold_con_fallback(0.45), 0.45);

        // Completar madurez (hasta 35 muestras >= 30)
        for i in 10..35 {
            engine.update(i as f64 * 0.1, 0.2, 0.5, 0.01);
        }

        assert!(engine.ofi_p80.initialized, ">= 30 muestras deben marcar inicializado");
        assert!(engine.dynamic_ofi_threshold() > 0.0);
        assert!(engine.dynamic_obi_threshold() > 0.0);
        assert!(engine.dynamic_holistic_threshold() > 0.0);

        // Cuando está maduro, dynamic_*_threshold_con_fallback devuelve el cuantil empírico, ignorando el fallback
        let medido = engine.dynamic_obi_threshold();
        assert_eq!(engine.dynamic_obi_threshold_con_fallback(0.99), medido);
    }

    #[test]
    fn test_adaptive_quantile_engine_nan_immunity() {
        let mut engine = AdaptiveQuantileEngine::default();
        for _ in 0..20 {
            engine.update(f64::NAN, f64::INFINITY, f64::NEG_INFINITY, f64::NAN);
        }
        assert_eq!(engine.dynamic_ofi_threshold(), 0.15);
        assert_eq!(engine.dynamic_obi_threshold(), 0.15);
        assert_eq!(engine.dynamic_holistic_threshold(), 0.40);
    }
}
