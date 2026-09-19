use std::f64;
use std::simd::prelude::*;
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

/// True Dynamic Kelly Sizing for exponential compounding
#[derive(Debug, Clone)]
pub struct DynamicKelly {
    pub win_rate_welford: WelfordVariance,
    pub win_size_welford: WelfordVariance,
    pub loss_size_welford: WelfordVariance,
    pub kelly_multiplier: f64,
}

impl DynamicKelly {
    pub fn new(multiplier: f64) -> Self {
        Self {
            win_rate_welford: WelfordVariance::new(),
            win_size_welford: WelfordVariance::new(),
            loss_size_welford: WelfordVariance::new(),
            kelly_multiplier: multiplier,
        }
    }

    #[inline(always)]
    pub fn update(&mut self, is_win: bool, pnl_pct: f64) {
        self.win_rate_welford.update(if is_win { 1.0 } else { 0.0 });
        if is_win {
            self.win_size_welford.update(pnl_pct.abs());
        } else {
            self.loss_size_welford.update(pnl_pct.abs());
        }
    }

    #[inline(always)]
    pub fn sizing_fraction(&self) -> f64 {
        // FIX #687: Sanitizar parámetros de Kelly
        let wr = if self.win_rate_welford.mean.is_finite() {
            self.win_rate_welford.mean.clamp(0.0, 1.0)
        } else {
            0.5
        };
        let avg_win = if self.win_size_welford.mean.is_finite() && self.win_size_welford.mean > 0.0
        {
            self.win_size_welford.mean
        } else {
            0.001
        };
        let avg_loss =
            if self.loss_size_welford.mean.is_finite() && self.loss_size_welford.mean > 0.0 {
                self.loss_size_welford.mean
            } else {
                0.001
            };
        let mult = if self.kelly_multiplier.is_finite() && self.kelly_multiplier > 0.0 {
            self.kelly_multiplier
        } else {
            0.5
        };

        // If not enough data, return a safe base default
        if self.win_rate_welford.count < 5.0 || avg_loss == 0.0 {
            return 0.10;
        }

        let r = avg_win / avg_loss;
        // Kelly Formula: K = W - ((1 - W) / R)
        let kelly = wr - ((1.0 - wr) / r);
        let adjusted_kelly = kelly * mult;

        // QO-M0.2 (auditoría matemática): el clamp inferior 0.01 FORZABA
        // una apuesta del 1% con edge NEGATIVO (Kelly<0 = el sistema dice
        // NO apostar). Sin edge ⇒ 0: la fracción multiplica tamaños, no
        // fabrica convicción. El fallback no-finito también baja a 0.
        if adjusted_kelly.is_finite() {
            adjusted_kelly.clamp(0.0, 1.0)
        } else {
            0.0
        }
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

    #[inline(always)]
    pub fn current(&self) -> f64 {
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
/// DFA sobre agregaciones temporales reales, con regresión de `ln F(s)` sobre
/// `ln s` en siete escalas (ver `feature_engine::hurst_dfa`). Insesgado frente
/// al paseo aleatorio, robusto a tendencias no estacionarias y capaz de
/// declarar cuándo NO hay ley de potencias (`r_squared`), cosa que el
/// estimador anterior ni siquiera podía expresar.
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
    /// El umbral de `r²` exige que la serie SIGA efectivamente una ley de
    /// potencias antes de emitir un valor distinto de 0,5. Sin él, el
    /// consumidor trataría cualquier pendiente de regresión como un régimen
    /// de mercado — que es la clase de error que esta corrección persigue.
    #[inline(always)]
    pub fn current(&self) -> f64 {
        self.dfa.hurst_or_neutral(0.85)
    }

    /// Bondad del ajuste log-log en [0,1]. Permite a los consumidores modular
    /// su convicción por la calidad de la medición en lugar de tratar el
    /// exponente como un número siempre significativo.
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

// FFI Kelly Fraction & Stats Calculation

#[inline(always)]
pub fn compute_kelly_fraction(
    p: f64,
    b: f64,
    apply_mult: bool,
    kelly_mult: f64,
    stress_score: f64,
    max_exposure: f64,
) -> f64 {
    if !p.is_finite() || !b.is_finite() || b <= 0.0 || p < 0.0 || p > 1.0 {
        return 0.0;
    }
    let q = 1.0 - p;
    let kelly = (p * b - q) / b;
    if !apply_mult {
        return if kelly.is_finite() {
            kelly.max(0.0).min(max_exposure)
        } else {
            0.0
        };
    }
    let mut mult = if kelly_mult.is_finite() && kelly_mult > 0.0 {
        kelly_mult
    } else {
        1.0
    };
    if stress_score < 90.0 {
        mult = 0.125;
    }
    let mut fractional_kelly = kelly * mult;
    if !fractional_kelly.is_finite() || fractional_kelly < 0.0 {
        fractional_kelly = 0.0;
    }
    if fractional_kelly > max_exposure {
        fractional_kelly = max_exposure;
    }
    fractional_kelly
}

#[inline(always)]
pub fn extract_kelly_stats(pnl_array: &[f64], is_win_array: &[bool]) -> (f64, f64) {
    let n = pnl_array.len() as f64;
    if n == 0.0 {
        return (0.5, 1.0);
    }
    let mut wins = 0.0;
    let mut losses = 0.0;
    let mut sum_wins = 0.0;
    let mut sum_losses = 0.0;
    for i in 0..pnl_array.len() {
        if is_win_array[i] {
            wins += 1.0;
            sum_wins += pnl_array[i];
        } else {
            losses += 1.0;
            sum_losses += pnl_array[i].abs();
        }
    }
    // QO-M0.3 (auditoría matemática): wins=0 ⇒ p = 0/n = 0 (cero victorias
    // es INFORMACIÓN: edge malo), no una moneda justa fabricada (0.5). Con
    // p=0 el Kelly downstream colapsa a 0 — el comportamiento correcto.
    let p = wins / n;
    let avg_win = if wins > 0.0 { sum_wins / wins } else { 0.01 };
    let avg_loss = if losses > 0.0 {
        sum_losses / losses
    } else {
        0.01
    };
    let b = if avg_loss > 0.0 {
        avg_win / avg_loss
    } else {
        1.0
    };
    (p, b)
}

#[inline(always)]
pub fn compute_cvar(loss_history: &[f64], confidence_level: f64) -> f64 {
    if loss_history.is_empty() {
        return 0.0;
    }
    let mut sorted_losses = loss_history.to_vec();
    // Sort in descending order (largest losses first)
    sorted_losses.sort_by(|a, b| b.partial_cmp(a).unwrap_or(std::cmp::Ordering::Equal));

    let n = sorted_losses.len();
    let cutoff_idx = ((1.0 - confidence_level) * n as f64).floor() as usize;
    let cutoff_idx = cutoff_idx.max(1);

    let mut sum = 0.0;
    for i in 0..cutoff_idx {
        sum += sorted_losses[i];
    }
    sum / (cutoff_idx as f64)
}

// =========================================================
// VECTORIZED TECHNICAL INDICATORS
// =========================================================

#[inline(always)]
pub fn compute_ema_vectorized(data: &[f64], period: usize, out: &mut [f64]) {
    let n = data.len();
    if n == 0 || period == 0 || out.len() != n {
        return;
    }
    let k = 2.0 / (period as f64 + 1.0);
    out[0] = data[0];
    for i in 1..n {
        out[i] = data[i] * k + out[i - 1] * (1.0 - k);
    }
}

#[inline(always)]
pub fn compute_rsi_vectorized(data: &[f64], period: usize, out: &mut [f64]) {
    let n = data.len();
    if n < period || period == 0 || out.len() != n {
        for i in 0..n {
            out[i] = 50.0;
        } // Default safe value
        return;
    }

    let mut gain = 0.0;
    let mut loss = 0.0;

    // Seed first window
    for i in 1..period {
        let diff = data[i] - data[i - 1];
        if diff > 0.0 {
            gain += diff;
        } else {
            loss -= diff;
        }
    }

    gain /= period as f64;
    loss /= period as f64;

    // Fill until period with 50.0 to prevent artifacting
    for i in 0..period {
        out[i] = 50.0;
    }

    if loss == 0.0 {
        out[period - 1] = 100.0;
    } else {
        let rs = gain / loss;
        out[period - 1] = 100.0 - (100.0 / (1.0 + rs));
    }

    // Smoothed Wilders moving average
    for i in period..n {
        let diff = data[i] - data[i - 1];
        if diff > 0.0 {
            gain = (gain * (period as f64 - 1.0) + diff) / period as f64;
            loss = (loss * (period as f64 - 1.0)) / period as f64;
        } else {
            gain = (gain * (period as f64 - 1.0)) / period as f64;
            loss = (loss * (period as f64 - 1.0) - diff) / period as f64;
        }
        if loss == 0.0 {
            out[i] = 100.0;
        } else {
            let rs = gain / loss;
            out[i] = 100.0 - (100.0 / (1.0 + rs));
        }
    }
}

#[inline(always)]
pub fn compute_bollinger_bands(
    data: &[f64],
    period: usize,
    std_dev_mult: f64,
    out_up: &mut [f64],
    out_mid: &mut [f64],
    out_low: &mut [f64],
) {
    let n = data.len();
    if n < period || period == 0 {
        for i in 0..n {
            out_mid[i] = data[i];
            out_up[i] = data[i];
            out_low[i] = data[i];
        }
        return;
    }

    for i in 0..period - 1 {
        out_mid[i] = data[i];
        out_up[i] = data[i];
        out_low[i] = data[i];
    }

    let window = period as f64;
    for i in (period - 1)..n {
        let mut sum = 0.0;
        for j in 0..period {
            sum += data[i - j];
        }
        let mean = sum / window;

        let mut variance = 0.0;
        for j in 0..period {
            let diff = data[i - j] - mean;
            variance += diff * diff;
        }
        let std_dev = (variance / window).sqrt();

        out_mid[i] = mean;
        out_up[i] = mean + std_dev_mult * std_dev;
        out_low[i] = mean - std_dev_mult * std_dev;
    }
}

#[inline(always)]
pub fn compute_macd(
    data: &[f64],
    fast_period: usize,
    slow_period: usize,
    signal_period: usize,
    out_macd: &mut [f64],
    out_signal: &mut [f64],
    out_hist: &mut [f64],
) {
    let n = data.len();
    if n == 0 {
        return;
    }

    let mut fast_ema = vec![0.0; n];
    let mut slow_ema = vec![0.0; n];

    compute_ema_vectorized(data, fast_period, &mut fast_ema);
    compute_ema_vectorized(data, slow_period, &mut slow_ema);

    for i in 0..n {
        out_macd[i] = fast_ema[i] - slow_ema[i];
    }

    compute_ema_vectorized(out_macd, signal_period, out_signal);

    for i in 0..n {
        out_hist[i] = out_macd[i] - out_signal[i];
    }
}

// =====================================================================
// MACHINE LEARNING INFERENCE KERNELS (Nano-Latency)
// =====================================================================

pub fn predict_rf(
    x: &[f64],
    children_left: &[i64],
    children_right: &[i64],
    feature: &[i64],
    threshold: &[f64],
    value: &[f64],
    tree_offsets: &[i64],
) -> f64 {
    let n_trees = tree_offsets.len().saturating_sub(1);
    if n_trees == 0 {
        return 0.0;
    }
    let mut total_prob = 0.0;

    for i in 0..n_trees {
        let mut node = tree_offsets[i] as usize;
        while children_left[node] != -1 {
            let f_idx = feature[node] as usize;
            if x[f_idx] <= threshold[node] {
                node = children_left[node] as usize;
            } else {
                node = children_right[node] as usize;
            }
        }
        total_prob += value[node];
    }
    total_prob / (n_trees as f64)
}

pub fn predict_gb(
    x: &[f64],
    children_left: &[i64],
    children_right: &[i64],
    feature: &[i64],
    threshold: &[f64],
    value: &[f64],
    tree_offsets: &[i64],
    init_score: f64,
    learning_rate: f64,
) -> f64 {
    let n_trees = tree_offsets.len().saturating_sub(1);
    let mut score = init_score;

    for i in 0..n_trees {
        let mut node = tree_offsets[i] as usize;
        while children_left[node] != -1 {
            let f_idx = feature[node] as usize;
            if x[f_idx] <= threshold[node] {
                node = children_left[node] as usize;
            } else {
                node = children_right[node] as usize;
            }
        }
        score += learning_rate * value[node];
    }

    // Sigmoid
    if score >= 0.0 {
        1.0 / (1.0 + (-score).exp())
    } else {
        let exp_s = score.exp();
        exp_s / (1.0 + exp_s)
    }
}

pub fn fused_compute_step(
    closes: &[f64],
    volumes: &[f64],
    portfolio_state: &[f64; 3], // [has_pos, pnl_norm, dur_norm]
    gene_params: &[f64; 2],     // [sl_norm, tp_norm]
    brain_weights: &[f64; 100], // 25 * 4 = 100 flattened
    l2_state: &[f64; 2],        // [ofi, microprice_divergence]
    window: usize,
    out_scores: &mut [f64; 4],
) {
    let n = closes.len();
    if n < 30 {
        out_scores.fill(0.0);
        return;
    }

    let mut state_tensor = [0.0f64; 25];

    // 1A. Market Data (20 Features)
    // Returns (5)
    for i in 0..window {
        let idx = n - window + i;
        let val = (closes[idx] - closes[idx - 1]) / closes[idx - 1];
        state_tensor[i] = val;
    }

    // Volatility (5)
    let mut vol_sum = 0.0;
    for i in (n - 20)..n {
        vol_sum += volumes[i];
    }
    let mut mean_vol = vol_sum / 20.0;
    if mean_vol < 1e-8 {
        mean_vol = 1.0;
    }

    for i in 0..window {
        let idx = n - window + i;
        state_tensor[5 + i] = volumes[idx] / mean_vol;
    }

    // Momentum / Custom (5)
    for i in 0..window {
        let idx = n - window + i;
        let mom = if idx >= 2 {
            (closes[idx] / closes[idx - 2]) - 1.0
        } else {
            0.0
        };
        state_tensor[10 + i] = mom;
        state_tensor[15 + i] = 0.0;
    }

    // Inject L2 Data
    state_tensor[18] = l2_state[0];
    state_tensor[19] = l2_state[1];

    // 2. Add Portfolio & Gene (5 Features)
    state_tensor[20] = portfolio_state[0];
    state_tensor[21] = portfolio_state[1];
    state_tensor[22] = portfolio_state[2];
    state_tensor[23] = gene_params[0];
    state_tensor[24] = gene_params[1];

    // 3. Neural Inference Dot Product (SIMD AVX-512 / AVX2 optimized)
    for act in 0..4 {
        let base_idx = act * 25;

        // 3 x 8 = 24 elements using SIMD
        let st_0 = f64x8::from_slice(&state_tensor[0..8]);
        let bw_0 = f64x8::from_slice(&brain_weights[base_idx..base_idx + 8]);
        let mut sim_sum = st_0 * bw_0;

        let st_1 = f64x8::from_slice(&state_tensor[8..16]);
        let bw_1 = f64x8::from_slice(&brain_weights[base_idx + 8..base_idx + 16]);
        sim_sum += st_1 * bw_1;

        let st_2 = f64x8::from_slice(&state_tensor[16..24]);
        let bw_2 = f64x8::from_slice(&brain_weights[base_idx + 16..base_idx + 24]);
        sim_sum += st_2 * bw_2;

        let mut score = sim_sum.reduce_sum();

        // 1 remaining element
        score += state_tensor[24] * brain_weights[base_idx + 24];

        out_scores[act] = score;
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

/// O(1) Funding Rate Elasticity (∂FundingRate / ∂Price)
#[derive(Debug, Clone, Default)]
pub struct FundingRateElasticity {
    pub prev_funding_rate: f64,
    pub prev_price: f64,
}

impl FundingRateElasticity {
    #[inline(always)]
    pub fn new() -> Self {
        Self {
            prev_funding_rate: 0.0,
            prev_price: 0.0,
        }
    }

    #[inline(always)]
    pub fn update(&mut self, funding_rate: f64, price: f64) -> f64 {
        // QO-M0.4 (auditoría matemática): prev_price se sobrescribía ANTES
        // del guard — la primera llamada emitía basura (delta_p=0 ⇒ guard
        // muerto) y la condición usaba el precio NUEVO como denominador.
        // Orden correcto: capturar previos, actualizar, dividir por el
        // precio ANTERIOR.
        let delta_fr = funding_rate - self.prev_funding_rate;
        let delta_p = price - self.prev_price;
        let prev_price = self.prev_price;

        self.prev_funding_rate = funding_rate;
        self.prev_price = price;

        if delta_p.abs() > f64::EPSILON && prev_price > 0.0 {
            let pct_delta_p = delta_p / prev_price;
            if pct_delta_p.abs() > f64::EPSILON {
                return delta_fr / pct_delta_p;
            }
        }
        0.0
    }
}

/// O(1) Exponential Decay Tensor for MEV/RBF Severity (Dark Alpha)
#[derive(Debug, Clone)]
pub struct ExponentialDecayTensor {
    pub current_severity: f64,
    pub decay_lambda: f64,
    pub last_timestamp_ms: u64,
}

impl ExponentialDecayTensor {
    #[inline(always)]
    pub fn new(half_life_ms: f64) -> Self {
        let decay_lambda = std::f64::consts::LN_2 / half_life_ms;
        Self {
            current_severity: 0.0,
            decay_lambda,
            last_timestamp_ms: 0,
        }
    }

    #[inline(always)]
    pub fn apply_event(&mut self, event_severity: f64, timestamp_ms: u64) {
        self.decay_to(timestamp_ms);
        self.current_severity += event_severity;
        self.last_timestamp_ms = timestamp_ms;
    }

    #[inline(always)]
    pub fn decay_to(&mut self, current_timestamp_ms: u64) -> f64 {
        if current_timestamp_ms > self.last_timestamp_ms {
            let dt = (current_timestamp_ms - self.last_timestamp_ms) as f64;
            let decay_factor = (-self.decay_lambda * dt).exp();
            self.current_severity *= decay_factor;
            self.last_timestamp_ms = current_timestamp_ms;
        }
        self.current_severity
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
        assert!(ultimo > 1.5, "una distribución extendida sobre diez bins debe dar entropía alta: {ultimo}");
        assert!(ultimo <= (10.0f64).ln() + 1e-9, "la entropía no puede superar ln(10): {ultimo}");
        let _ = x;
    }

}
