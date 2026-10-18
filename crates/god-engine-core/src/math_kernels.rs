use std::f64;
/// Welford's Online Algorithm for computing variance and standard deviation in O(1).
/// Used for Z-Scores, Bollinger Bands, and running volatility.
#[derive(Debug, Clone, Default)]
pub struct WelfordVariance {
    pub count: f64,
    pub mean: f64,
    pub m2: f64,
}

impl WelfordVariance {
    #[inline(always)]
    pub fn new() -> Self {
        Self {
            count: 0.0,
            mean: 0.0,
            m2: 0.0,
        }
    }

    #[inline(always)]
    pub fn update(&mut self, new_value: f64) {
        if !new_value.is_finite() {
            return;
        }
        if self.count >= 2000.0 {
            // Decaimiento proporcional de M2 para mantener la ventana móvil estable (#567)
            self.m2 *= 1999.0 / 2000.0;
            self.count = 1999.0;
        }
        self.count += 1.0;
        let delta = new_value - self.mean;
        self.mean += delta / self.count;
        let delta2 = new_value - self.mean;
        self.m2 += delta * delta2;
    }

    #[inline(always)]
    pub fn variance(&self) -> f64 {
        if self.count < 2.0 {
            return 0.0;
        }
        (self.m2 / (self.count - 1.0)).max(0.0)
    }

    #[inline(always)]
    pub fn std_dev(&self) -> f64 {
        self.variance().sqrt()
    }

    #[inline(always)]
    pub fn remove(&mut self, old_value: f64) {
        if self.count <= 1.0 {
            self.count = 0.0;
            self.mean = 0.0;
            self.m2 = 0.0;
            return;
        }
        let delta = old_value - self.mean;
        self.count -= 1.0;
        self.mean -= delta / self.count;
        let delta2 = old_value - self.mean;
        self.m2 -= delta * delta2;
    }
}

/// Kahan summation algorithm to reduce floating-point error accumulation in O(1).
/// Used for Volume Delta, cumulative PnL, and other large running sums.
#[derive(Debug, Clone, Default)]
pub struct KahanSummation {
    pub sum: f64,
    pub c: f64,
}

impl KahanSummation {
    #[inline(always)]
    pub fn new() -> Self {
        Self { sum: 0.0, c: 0.0 }
    }

    #[inline(always)]
    pub fn add(&mut self, value: f64) {
        if !value.is_finite() {
            return;
        }
        let y = value - self.c;
        let t = self.sum + y;
        self.c = (t - self.sum) - y;
        self.sum = t;
    }

    #[inline(always)]
    pub fn get_sum(&self) -> f64 {
        self.sum
    }
}

/// QO-M0.7 (auditoría matemática) — RAZÓN DE AMIHUD (iliquidez), no Kyle.
/// Lo que calcula es Σ|Δp| / Σv: la razón de iliquidez de Amihud (2002),
/// correcta para lo que hace. El Kyle-λ REAL es el coeficiente de la
/// regresión Δp = λ·v + ε (cov/var del volumen firmado) — atribución
/// corregida; si algún consumidor exige Kyle de verdad, implementar OLS
/// rodante. Devuelve ln(1+λ) para domar colas pesadas.
#[derive(Debug, Clone, Default)]
pub struct AmihudIlliquidity {
    pub delta_p_kahan: KahanSummation,
    pub delta_v_kahan: KahanSummation,
    pub last_price: f64,
}

impl AmihudIlliquidity {
    #[inline(always)]
    pub fn new() -> Self {
        Self {
            delta_p_kahan: KahanSummation::new(),
            delta_v_kahan: KahanSummation::new(),
            last_price: 0.0,
        }
    }

    #[inline(always)]
    pub fn update(&mut self, current_price: f64, volume: f64) -> f64 {
        // FIX #687: Validar finitud de precio y volumen
        if !current_price.is_finite() || !volume.is_finite() || volume < 0.0 {
            return 0.0;
        }

        if self.last_price != 0.0 {
            let delta_p = (current_price - self.last_price).abs();
            self.delta_p_kahan.add(delta_p);
            self.delta_v_kahan.add(volume);
        }
        self.last_price = current_price;

        let dv = self.delta_v_kahan.get_sum();
        if dv > 0.0 {
            let lambda = self.delta_p_kahan.get_sum() / dv;
            if lambda.is_finite() {
                (1.0 + lambda).ln()
            } else {
                0.0
            }
        } else {
            0.0
        }
    }
}

/// Continuous VPIN (Volume-Synchronized Probability of Informed Trading) in O(1)
#[derive(Debug, Clone, Default)]
pub struct ContinuousVPIN {
    pub buy_volume: f64,
    pub sell_volume: f64,
    pub bucket_size: f64,
    /// VPIN-FIX: EWMA del notional por tick, para calibrar el bucket al
    /// reloj de volumen (N trades por bucket) en vez de un dólar fijo.
    pub ewma_tick_notional: f64,
    /// D-712: dólar de construcción, piso ESTRICTO del bucket. Antes el piso y
    /// el valor vivo eran el mismo campo, de modo que el bucket no podía bajar.
    pub initial_bucket_size: f64,
}

impl ContinuousVPIN {
    /// Trades objetivo por bucket — práctica estándar del VPIN de
    /// Easley/López de Prado: el bucket debe agregar decenas de trades para
    /// que |buy−sell|/total tenga significado. Con un dólar fijo y ticks de
    /// nocional grande, cada bucket contiene UN trade y el VPIN satura a 1.0
    /// permanente — exactamente el veto fantasma del Senior Causal que
    /// bloqueaba el 100% de las señales (diag 2026-09-07).
    const TICKS_PER_BUCKET: f64 = 100.0;

    pub fn new(bucket_size: f64) -> Self {
        Self {
            buy_volume: 0.0,
            sell_volume: 0.0,
            bucket_size,
            ewma_tick_notional: 0.0,
            initial_bucket_size: bucket_size,
        }
    }

    #[inline(always)]
    pub fn update(&mut self, volume: f64, is_buyer_maker: bool) -> f64 {
        if !volume.is_finite() || volume < 0.0 {
            return 0.0;
        }
        // VPIN-FIX — reloj de volumen auto-calibrado: el bucket sigue al
        // tamaño típico del tick (EWMA) para contener ~TICKS_PER_BUCKET
        // trades, con el dólar original como piso. Mecánica:
        // con el bucket fijo de $10k y ticks de $9k-$450k (BTC), cada bucket
        // = 1 trade => vpin = 1.0 constante => Senior Causal veta todo.
        if self.ewma_tick_notional <= 0.0 {
            self.ewma_tick_notional = volume.max(1.0);
        } else {
            self.ewma_tick_notional = 0.98 * self.ewma_tick_notional + 0.02 * volume;
        }
        // D-712 (DÉCIMA OLA · auditoría integral): EL BUCKET SIGUE AL VOLUMEN
        // VIVO, NO A SU MÁXIMO HISTÓRICO.
        //
        // `bucket_size` sólo se actualizaba HACIA ARRIBA: un único episodio de
        // nocionales grandes (apertura de Nueva York, cascada de liquidaciones,
        // vela de noticia) lo fijaba en el máximo de la sesión y ahí se quedaba.
        // A partir de ese momento el reloj de volumen deja de cerrar buckets, el
        // desequilibrio se acumula sobre una ventana cada vez más larga y el VPIN
        // se aplana justo DESPUÉS del evento que debía detectar — con el veto
        // causal y la telemetría de riesgo leyendo esa medida aplanada.
        //
        // El reloj de volumen de Easley/López de Prado sigue al volumen típico
        // vivo: `bucket_size` acompaña a `calibrated` en ambas direcciones, con
        // el dólar de construcción como piso estricto (no como valor mutable).
        let calibrated = self.ewma_tick_notional * Self::TICKS_PER_BUCKET;
        self.bucket_size = calibrated.max(self.initial_bucket_size);
        if is_buyer_maker {
            self.sell_volume += volume;
        } else {
            self.buy_volume += volume;
        }

        // Decay to keep within bucket context (EWMA style decay for O(1) rolling VPIN)
        let mut total_vol = self.buy_volume + self.sell_volume;
        if total_vol > self.bucket_size {
            let ratio = self.bucket_size / total_vol;
            self.buy_volume *= ratio;
            self.sell_volume *= ratio;
            total_vol = self.bucket_size; // FIX #568: Denominador exacto tras reescalado
        }

        if total_vol > 0.0 {
            (self.buy_volume - self.sell_volume).abs() / total_vol
        } else {
            0.0
        }
    }

    #[inline(always)]
    pub fn current_vpin(&self) -> f64 {
        let total = self.buy_volume + self.sell_volume;
        if total > 0.0 {
            (self.buy_volume - self.sell_volume).abs() / total
        } else {
            0.0
        }
    }
}

/// Recursive O(1) SMA
#[derive(Debug, Clone, Default)]
pub struct RecursiveSMA {
    pub sum: f64,
    pub count: usize,
    pub window: usize,
}

impl RecursiveSMA {
    #[inline(always)]
    pub fn new(window: usize) -> Self {
        Self {
            sum: 0.0,
            count: 0,
            window,
        }
    }

    #[inline(always)]
    pub fn update(&mut self, new_val: f64, old_val: f64) -> f64 {
        if !new_val.is_finite() || !old_val.is_finite() {
            return if self.count > 0 {
                self.sum / (self.count as f64)
            } else {
                0.0
            };
        }
        if self.count < self.window {
            self.count += 1;
            self.sum += new_val;
        } else {
            self.sum = self.sum + new_val - old_val;
        }
        self.sum / (self.count as f64)
    }
}

/// Recursive O(1) EMA
#[derive(Debug, Clone, Default)]
pub struct RecursiveEMA {
    pub ema: f64,
    pub alpha: f64,
    pub initialized: bool,
}

impl RecursiveEMA {
    #[inline(always)]
    pub fn new(window: usize) -> Self {
        Self {
            ema: 0.0,
            alpha: 2.0 / (window as f64 + 1.0),
            initialized: false,
        }
    }

    #[inline(always)]
    pub fn update(&mut self, new_val: f64) -> f64 {
        if !new_val.is_finite() {
            return self.ema;
        }
        if !self.initialized {
            self.ema = new_val;
            self.initialized = true;
        } else {
            self.ema = (new_val - self.ema) * self.alpha + self.ema;
        }
        self.ema
    }
}

/// Shannon Entropy O(1) Approximation for market noise measurement
#[derive(Debug, Clone)]
pub struct ShannonEntropy {
    bins: [f64; 10], // Simple 10-bin histogram approximation
    total_count: f64,
    /// D-713: dispersión viva de la propia entrada, para que los bins midan
    /// la FORMA de la distribución y no un rango fijo en tanto por uno.
    disp: WelfordVariance,
}

impl Default for ShannonEntropy {
    fn default() -> Self {
        Self::new()
    }
}

impl ShannonEntropy {
    pub fn new() -> Self {
        Self {
            bins: [0.0; 10],
            total_count: 0.0,
            disp: WelfordVariance::new(),
        }
    }

    #[inline(always)]
    pub fn update(&mut self, norm_return: f64) -> f64 {
        if !norm_return.is_finite() {
            return 0.0;
        }
        // FIX #569: Decaimiento exponencial (0.999) para mantener sensibilidad a regímenes vivos
        let decay = 0.999;
        self.total_count *= decay;
        for b in self.bins.iter_mut() {
            *b *= decay;
        }

        // D-713 (DÉCIMA OLA · auditoría integral): LOS BINS SE DIMENSIONAN CON
        // LA DISPERSIÓN MEDIDA, NO CON UN RANGO LITERAL.
        //
        // El mapeo anterior, `5 + 100·r`, repartía diez bins sobre ±5 % POR
        // EVENTO: con retornos entre eventos del orden de 1e-5 a 1e-4, toda la
        // masa caía en el bin 5, p = 1 y la entropía valía −1·ln(1) = 0 de forma
        // permanente. La dimensión 8 del vector ML y el registro
        // `shannon_entropy` eran una constante cero disfrazada de medida de
        // ruido: el GBDT no podía partir por ella y cualquier consumidor que
        // modulase por entropía modulaba por una constante.
        //
        // Ahora el bin sale del z de la propia serie (Welford en línea) repartido
        // sobre ±z95, el mismo criterio de cobertura que usa el resto del motor.
        // Hasta tener dispersión medible (σ = 0, arranque) la entrada cae al bin
        // central, que es lo correcto: sin variación medida no hay información.
        self.disp.update(norm_return);
        let sigma = self.disp.std_dev();
        let z = if sigma > 0.0 {
            (norm_return - self.disp.mean) / sigma
        } else {
            0.0
        };
        let span = crate::diffusion::Z95;
        let bin_idx = (((z + span) / (2.0 * span)) * 10.0).clamp(0.0, 9.99) as usize;
        self.bins[bin_idx] += 1.0;
        self.total_count += 1.0;

        self.current()
    }

    /// Máxima entropía teórica para 10 bins en nats: ln(10.0).
    pub const MAX_SHANNON_NATS_10BINS: f64 = 2.302_585_092_994_046;

    /// Entropía normalizada en [0, 1] (D-1 / R7-R2-D-1).
    /// El gemelo de feature-engine normaliza por log2(num_bins).
    /// Aquí normalizamos por ln(10) para calibración uniforme en [0, 1].
    #[inline(always)]
    pub fn current(&self) -> f64 {
        let nats = self.current_nats();
        (nats / Self::MAX_SHANNON_NATS_10BINS).clamp(0.0, 1.0)
    }

    /// Entropía en nats puros sin normalizar [0, ln(10)].
    #[inline(always)]
    pub fn current_nats(&self) -> f64 {
        let mut entropy = 0.0;
        if self.total_count > 0.0 {
            for &count in self.bins.iter() {
                if count > 0.0 {
                    let p = count / self.total_count;
                    entropy -= p * p.ln();
                }
            }
        }
        entropy
    }
}

/// EXPONENTE DE HURST DEL CAMINO DE DECISIÓN (D-615 — DÉCIMA OLA).
///
/// # El sesgo que se elimina
///
/// La implementación anterior era R/S de UNA SOLA ESCALA:
/// `H = ln(R/S) / ln(N)` con `N = 256`. Es el estimador clásico de Mandelbrot
/// y Wallis **sin la corrección de Anis–Lloyd**, y arrastra un sesgo conocido:
/// para movimiento browniano fraccional `E[R/S] ≈ (πN/2)^H`, no `N^H`. De ahí
///
/// ```text
/// H_medido = ln(√(πN/2)) / ln N = 0,5 + 0,5·ln(π/2)/ln N
/// ```
///
/// que con `N = 256` da **0,5407** analíticamente y **0,524** medido sobre
/// 400 realizaciones de un paseo aleatorio puro.
///
/// El motor clasifica «tendencia» a partir de `H ≥ 0,52` y «anti-persistente»
/// por debajo de `0,48`. **Un paseo aleatorio sin ninguna estructura cruzaba
/// el umbral de tendencia**, de modo que el sistema percibía dirección donde
/// sólo había ruido — y lo hacía de forma sistemática, no ocasional.
///
/// # El estimador adoptado
///
/// DFA sobre ventanas de muestras, con hasta siete escalas (ver
/// `feature_engine::hurst_dfa`). Requiere justificar reloj, datos ausentes y
/// modelo de escalamiento; no es insesgado en toda muestra ni elimina toda
/// tendencia. R² mide ajuste descriptivo, no prueba existencia de una ley.
///
/// Durante el calentamiento devuelve 0,50 —la hipótesis nula honesta— en lugar
/// de una estimación sesgada: es preferible no opinar a opinar mal.
#[derive(Debug, Clone)]
pub struct RecursiveHurst {
    dfa: feature_engine::hurst_dfa::HurstDfa,
}

impl Default for RecursiveHurst {
    fn default() -> Self {
        Self::new()
    }
}

impl RecursiveHurst {
    pub fn new() -> Self {
        Self {
            dfa: feature_engine::hurst_dfa::HurstDfa::new(),
        }
    }

    #[inline(always)]
    pub fn update(&mut self, price: f64) -> f64 {
        self.dfa.update(price);
        self.current()
    }

    /// Exponente actual sin mutar estado.
    ///
    /// El umbral R²=0,85 es política descriptiva, no significación estadística.
    /// El DFA rechaza pendientes fuera del modelo estacionario adoptado;
    /// devolver 0,5 por falta de evidencia no certifica ausencia de memoria.
    #[inline(always)]
    pub fn current(&self) -> f64 {
        self.dfa.hurst_or_neutral(0.85)
    }

    /// R² log-log en [0,1] si el ajuste es admisible. El nombre se conserva
    /// por compatibilidad: NO es probabilidad calibrada de acierto/cobertura.
    #[inline(always)]
    pub fn confidence(&self) -> f64 {
        if self.dfa.is_valid { self.dfa.r_squared } else { 0.0 }
    }

    /// Retornos acumulados por el estimador.
    #[inline(always)]
    pub fn samples(&self) -> usize {
        self.dfa.samples()
    }
}

/// O(1) Second Derivative of Order Book Imbalance (Liquidity Acceleration)
#[derive(Debug, Clone, Default)]
pub struct ObiAcceleration {
    pub prev_obi: f64,
    pub prev_obi_velocity: f64,
    pub accel: f64,
}

impl ObiAcceleration {
    #[inline(always)]
    pub fn new() -> Self {
        Self {
            prev_obi: 0.0,
            prev_obi_velocity: 0.0,
            accel: 0.0,
        }
    }

    #[inline(always)]
    pub fn update(&mut self, current_obi: f64) -> f64 {
        let current_velocity = current_obi - self.prev_obi;
        let acceleration = current_velocity - self.prev_obi_velocity;
        self.prev_obi = current_obi;
        self.prev_obi_velocity = current_velocity;
        self.accel = acceleration;
        acceleration
    }
}

/// O(1) nonnegative impulse accumulator with exponential elapsed-time decay.
/// This scalar is not a probability, a tensor decomposition or calibrated risk.
#[derive(Debug, Clone)]
pub struct ExponentialDecayTensor {
    pub current_severity: f64,
    pub decay_lambda: f64,
    pub last_timestamp_ms: u64,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DecayError {
    InvalidHalfLife,
    InvalidRate,
    InvalidState,
    InvalidImpulse,
    OutOfOrder,
    Overflow,
}

impl ExponentialDecayTensor {
    /// Compatibility constructor, total (regla 6: sin pánico). Con una semivida
    /// que no da una tasa finita y positiva devuelve un acumulador inerte: su
    /// tasa es NaN, así que `try_apply_event` responde `InvalidRate` y el nivel
    /// se queda en 0, es decir, no opina. Los llamadores operativos pasan
    /// constantes; `try_new` conserva el motivo para parámetros de ejecución.
    #[inline(always)]
    pub fn new(half_life_ms: f64) -> Self {
        Self::try_new(half_life_ms).unwrap_or(Self {
            current_severity: 0.0,
            decay_lambda: f64::NAN,
            last_timestamp_ms: 0,
        })
    }

    pub fn try_new(half_life_ms: f64) -> Result<Self, DecayError> {
        if !half_life_ms.is_finite() || half_life_ms <= 0.0 {
            return Err(DecayError::InvalidHalfLife);
        }
        let decay_lambda = std::f64::consts::LN_2 / half_life_ms;
        if !decay_lambda.is_finite() || decay_lambda <= 0.0 {
            return Err(DecayError::InvalidRate);
        }
        Ok(Self {
            current_severity: 0.0,
            decay_lambda,
            last_timestamp_ms: 0,
        })
    }

    /// Compatibility wrapper: invalid input is a no-op. Checked callers should
    /// use try_apply_event to retain the reason. Equal-time impulses are additive;
    /// timestamps alone are NOT event identities or a deduplication mechanism.
    #[inline(always)]
    pub fn apply_event(&mut self, event_severity: f64, timestamp_ms: u64) {
        let _ = self.try_apply_event(event_severity, timestamp_ms);
    }

    /// Reject out-of-order events instead of moving their information to now.
    /// Compute the entire transition before committing either value or clock.
    pub fn try_apply_event(&mut self, event_severity: f64, timestamp_ms: u64) -> Result<(), DecayError> {
        if !self.decay_lambda.is_finite() || self.decay_lambda <= 0.0 {
            return Err(DecayError::InvalidRate);
        }
        if !self.current_severity.is_finite() || self.current_severity < 0.0 {
            return Err(DecayError::InvalidState);
        }
        if !event_severity.is_finite() || event_severity < 0.0 {
            return Err(DecayError::InvalidImpulse);
        }
        if timestamp_ms < self.last_timestamp_ms { return Err(DecayError::OutOfOrder); }
        let dt = (timestamp_ms - self.last_timestamp_ms) as f64;
        let next = self.current_severity * (-self.decay_lambda * dt).exp() + event_severity;
        if !next.is_finite() { return Err(DecayError::Overflow); }
        self.current_severity = next;
        self.last_timestamp_ms = timestamp_ms;
        Ok(())
    }

    /// Compatibility wrapper. On an invalid clock/state it returns the unchanged
    /// level; use try_decay_to when the distinction is required for admission.
    #[inline(always)]
    pub fn decay_to(&mut self, current_timestamp_ms: u64) -> f64 {
        let _ = self.try_decay_to(current_timestamp_ms);
        self.current_severity
    }

    pub fn try_decay_to(&mut self, timestamp_ms: u64) -> Result<f64, DecayError> {
        self.try_apply_event(0.0, timestamp_ms)?;
        Ok(self.current_severity)
    }
}
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_welford_variance() {
        let mut welford = WelfordVariance::new();
        welford.update(10.0);
        welford.update(12.0);
        welford.update(14.0);
        welford.update(16.0);
        welford.update(18.0);

        let mean = welford.mean;
        let std_dev = welford.std_dev();

        // Mean of 10, 12, 14, 16, 18 is 14
        assert!((mean - 14.0).abs() < 1e-6);

        // Variance of sample is sum((x - mean)^2) / (n - 1)
        // 16 + 4 + 0 + 4 + 16 = 40. 40 / 4 = 10
        // Std Dev = sqrt(10) = 3.162277...
        assert!((std_dev - 10.0_f64.sqrt()).abs() < 1e-6);

        // Test rolling removal
        welford.remove(10.0);
        assert!((welford.mean - 15.0).abs() < 1e-6); // Mean of 12, 14, 16, 18
    }

    #[test]
    fn test_kahan_summation() {
        let mut kahan = KahanSummation::new();
        // Add 1.0 ten million times
        for _ in 0..10_000_000 {
            kahan.add(1.0);
        }
        // Then add a very small number
        kahan.add(1e-10);

        assert!((kahan.get_sum() - 10_000_000.0000000001).abs() < 1e-10);
    }

    #[test]
    fn test_continuous_vpin_update_and_nan_immunity() {
        let mut vpin = ContinuousVPIN::new(1000.0);
        let score1 = vpin.update(500.0, false); // Buy
        assert!(score1 > 0.0);

        let score2 = vpin.update(500.0, true); // Sell balanced
        assert_eq!(score2, 0.0);

        // NaN volume ignored
        let score_nan = vpin.update(f64::NAN, false);
        assert_eq!(score_nan, 0.0);
    }

    #[test]
    fn test_recursive_sma_update_and_nan_immunity() {
        let mut sma = RecursiveSMA::new(3);
        assert_eq!(sma.update(10.0, 0.0), 10.0);
        assert_eq!(sma.update(20.0, 0.0), 15.0);
        assert_eq!(sma.update(30.0, 0.0), 20.0);
        // Window full (size 3), old_val 10 removed, new_val 40 added: (20 + 30 + 40) / 3 = 30
        assert_eq!(sma.update(40.0, 10.0), 30.0);

        // NaN input returns current average safely
        let safe_avg = sma.update(f64::NAN, 0.0);
        assert_eq!(safe_avg, 30.0);
    }

    #[test]
    fn test_exponential_decay_tensor_event_and_decay() {
        let mut tensor = ExponentialDecayTensor::new(1000.0); // 1000ms half life
        tensor.apply_event(100.0, 1000);
        assert_eq!(tensor.current_severity, 100.0);

        // Decay 1000ms (1 half life) -> severity should be ~50.0
        let decayed = tensor.decay_to(2000);
        assert!((decayed - 50.0).abs() < 1e-3);
    }

    #[test]
    fn ret24_semivida_invalida_no_entra_en_panico_y_no_acumula() {
        // Regla 6: `new` no puede abortar el proceso. Con una semivida que no
        // da una tasa finita y positiva, el acumulador queda inerte: rechaza
        // todo impulso con InvalidRate y su nivel se queda en 0.
        for semivida in [0.0, -1.0, f64::NAN, f64::INFINITY, f64::NEG_INFINITY, 1e-320] {
            let mut t = ExponentialDecayTensor::new(semivida);
            assert!(ExponentialDecayTensor::try_new(semivida).is_err(), "semivida {semivida}");
            assert_eq!(t.try_apply_event(1.0, 1000), Err(DecayError::InvalidRate), "semivida {semivida}");
            t.apply_event(5.0, 2000);
            assert_eq!(t.current_severity, 0.0, "semivida {semivida}");
            assert_eq!(t.last_timestamp_ms, 0, "semivida {semivida}");
            assert_eq!(t.decay_to(3000), 0.0, "semivida {semivida}");
            assert_eq!(t.try_decay_to(3000), Err(DecayError::InvalidRate), "semivida {semivida}");
        }
    }

    #[test]
    fn d712_el_bucket_del_vpin_vuelve_a_bajar_tras_una_rafaga() {
        let mut v = ContinuousVPIN::new(10_000.0);
        for i in 0..1000 {
            v.update(10.0, i % 2 == 0);
        }
        let tranquilo = v.bucket_size;
        for i in 0..20 {
            v.update(500_000.0, i % 2 == 0);
        }
        let en_rafaga = v.bucket_size;
        assert!(en_rafaga > tranquilo, "el bucket debe crecer con la ráfaga");
        for i in 0..1000 {
            v.update(10.0, i % 2 == 0);
        }
        let despues = v.bucket_size;
        assert!(
            despues < en_rafaga * 0.1,
            "tras la ráfaga el bucket debe volver al volumen vivo: {despues} frente a {en_rafaga}"
        );
        assert!(despues >= 10_000.0, "el dólar de construcción es piso estricto: {despues}");
    }


    #[test]
    fn d713_la_entropia_mide_la_forma_y_no_es_cero_constante() {
        let mut e = ShannonEntropy::new();
        // Serie gaussiana de sigma 5e-5: el rango literal anterior la metía
        // entera en un bin y devolvía 0,0.
        let mut x: f64 = 0.0;
        let mut ultimo = 0.0;
        for i in 0..10_000 {
            // Generador determinista con forma de campana (suma de uniformes).
            let i = i as u64;
            let u = (i.wrapping_mul(7919) % 1000) as f64 / 1000.0 - 0.5;
            let v = (i.wrapping_mul(104_729) % 997) as f64 / 997.0 - 0.5;
            let w = (i.wrapping_mul(15_485_863) % 991) as f64 / 991.0 - 0.5;
            x = (u + v + w) * 5e-5;
            ultimo = e.update(x);
        }
        assert!(ultimo > 0.65, "una distribución extendida sobre diez bins debe dar entropía normalizada alta: {ultimo}");
        assert!(ultimo <= 1.0 + 1e-9, "la entropía normalizada no puede superar 1.0: {ultimo}");
        assert!(e.current_nats() > 1.5, "los nats sin normalizar deben superar 1.5: {}", e.current_nats());
        assert!(e.current_nats() <= (10.0f64).ln() + 1e-9, "los nats no pueden superar ln(10): {}", e.current_nats());
        let _ = x;
    }

}
