//! ESPECTRO TEMPORAL CONTINUO (F8) — un solo sistema, todas las escalas.
//!
//! PROBLEMA (directriz del operador): el binario scalp/swing es un CORTE
//! ARBITRARIO del espectro temporal que contamina genoma, motor, posiciones
//! y 66 archivos. El mercado no tiene "dos modos": tiene estructuras en
//! TODAS las escalas simultáneamente.
//!
//! DISEÑO FÍSICO-HONESTO (fronteras reales, no marketing):
//!   - Cota inferior ~1ms: el inter-arribo de ticks de Binance futures y el
//!     RTT de red. Por debajo, "horizonte" solo existe DENTRO del motor
//!     (procesamos cada evento a cadencia de nanosegundo — eso ya ocurre).
//!   - Cota superior ~2 años: los ciclos macro relevantes para futuros
//!     USDT-M (halving ≈ 4y queda como contexto macro, no como horizonte
//!     operable con data de 2019+).
//!   - Entre ambas: ESPECTRO CONTINUO log-espaciado base-4: 32 escalas de
//!     1ms a 68_719_476_736ms (≈2.18 años) sin huecos ni bandas prohibidas.
//!
//! QUÉ calcula CADA TICK (todo junto, nanosegundo a nanosegundo de proceso):
//!   Por escala τ_i: precio EWMA(τ_i), volatilidad EWMA(τ_i), momentum
//!   z-scoreado, señal tanh(z) ∈ [-1,1] y persistencia (acuerdo de signo).
//!   Fusión: score espectral = Σ w_i·señal_i con w_i ∝ 1/vol_i — PARIDAD DE
//!   RIESGO entre escalas (cada horizonte aporta según su Sharpe potencial
//!   inverso a su ruido). Nada de "scalp manda aquí, swing allá".
//!
//! COSTE: O(S)=32 escalas × ~6 FLOPs = ~120 FLOPs/tick — despreciable frente
//! al proceso del evento本身.

//! # CÓMO LEER SUS VALORES (guía operativa)
//!
//! * `persistence` por escala: 0.5 = RUIDO puro (signo aleatorio, H=0.5);
//!   →1 = tendencia que se auto-confirma (deja correr); →0 = reversión
//!   perfecta (asegurar pronto). Es EL dial de régimen del motor: todos los
//!   lerp espectrales (S-2/S-3) lo usan como t∈[0,1].
//! * `fused_score` alto = las escalas QUE SABEN (persistencia alta) están
//!   alineadas direccionalmente; alto con persistencias bajas = ruido
//!   promediado — el peso suelo 5% evita que una escala impredecible domine.
//! * `dominant_tau_ms`: 30 s→12 h es la BANDA OPERATIVA; τ corta = micro
//!   impulso (brackets estrechos, trailing rápido), τ larga = tendencia de
//!   banda (respiración amplia). El espectro OBSERVA más allá de la banda,
//!   pero la DECISIÓN jamás sale de ella (C-05).

/// Escalas del espectro: 10^-6 ms * 4^i para i∈0..32 → 1 ns (10^-6 ms) … ≈146.15 años (4.61*10^12 ms).
/// Log-espaciadas base 4 (≈4.15 escalas/década): resolución uniforme en
/// log(τ), cubriendo desde microestructura en nanosegundos hasta tendencias seculares de más de 100 años.
pub const SPECTRUM_SCALES_MS: [f64; 32] = [
    1.0e-6,                   // 1 ns
    4.0e-6,                   // 4 ns
    1.6e-5,                   // 16 ns
    6.4e-5,                   // 64 ns
    2.56e-4,                  // 256 ns
    1.024e-3,                 // ~1.02 µs
    4.096e-3,                 // ~4.10 µs
    1.6384e-2,                // ~16.38 µs
    6.5536e-2,                // ~65.54 µs
    0.262144,                 // ~262.14 µs
    1.048576,                 // ~1.05 ms
    4.194304,                 // ~4.19 ms
    16.777216,                // ~16.78 ms
    67.108864,                // ~67.11 ms
    268.435456,               // ~268.44 ms
    1_073.741824,             // ~1.07 s
    4_294.967296,             // ~4.29 s
    17_179.869184,            // ~17.18 s
    68_719.476736,            // ~1.15 min
    274_877.906944,           // ~4.58 min
    1_099_511.627776,         // ~18.33 min
    4_398_046.511104,         // ~1.22 h
    17_592_186.044416,        // ~4.89 h
    70_368_744.177664,        // ~19.55 h
    281_474_976.710656,       // ~3.26 d
    1_125_899_906.842624,     // ~13.03 d
    4_503_599_627.370496,     // ~52.12 d
    18_014_398_509.481984,    // ~208.5 d
    72_057_594_037.927936,    // ~2.28 años
    288_230_376_151.711744,   // ~9.13 años
    1_152_921_504_606.846976, // ~36.54 años
    4_611_686_018_427.387904, // ~146.15 años (>100 años)
];

/// τ de anclaje histórica (compat): la banda rápida ≈ 30s, la extendida ≈ 12h.
/// El continuo las reemplaza; quedan solo como puntos de conversión del
/// genoma legacy — ningún código decide por pertenecer a una banda.
pub const TAU_ANCHOR_FAST_MS: f64 = 30_000.0;
pub const TAU_ANCHOR_SLOW_MS: f64 = 43_200_000.0;

/// D-638b (DÉCIMA OLA) — MAPEO ÚNICO DEL HORIZONTE OPERATIVO.
///
/// Tras ampliar el espectro a 1 ns–146 años convivían TRES conversiones del
/// horizonte de una intención:
///   · `risk-engine::horizon_tau_ms` interpolaba sobre los EXTREMOS del
///     espectro: con el rango nuevo, `temporal_scale = 0,05` daba ~9 ns y
///     0,95 daba ~17 años, de modo que el gate de TP/SL rechazaba por «no
///     operable» o dimensionaba stops de décadas;
///   · el bloque de Kelly interpolaba entre 10 s y 24 h;
///   · la matriz de apalancamiento invertía otra fórmula sobre 10 s–24 h.
///
/// El espectro de OBSERVACIÓN (dónde el sistema mira) y la banda de OPERACIÓN
/// (dónde mantiene posiciones) son objetos distintos: observar a 1 ns tiene
/// sentido, mantener una posición un nanosegundo no. La banda de operación de
/// referencia son las anclas que ya definen la semántica del gen
/// `temporal_scale` (s = 0 ↔ 30 s, s = 1 ↔ 12 h) y en las que `apply_to_arena`
/// evalúa las curvas: no se introduce ningún extremo nuevo.
///
/// La duración que declara la señal, si existe, tiene prioridad: es
/// información de mercado, no un parámetro.
#[inline]
pub fn tau_from_temporal_scale(s: f64) -> f64 {
    let s = if s.is_finite() {
        s.clamp(0.0, 1.0)
    } else {
        0.5
    };
    let (lo, hi) = (TAU_ANCHOR_FAST_MS.ln(), TAU_ANCHOR_SLOW_MS.ln());
    (lo + s * (hi - lo)).exp()
}

/// Inversa de `tau_from_temporal_scale`, acotada a [0, 1].
#[inline]
pub fn temporal_scale_from_tau(tau_ms: f64) -> f64 {
    if !tau_ms.is_finite() || tau_ms <= 0.0 {
        return 0.5;
    }
    let (lo, hi) = (TAU_ANCHOR_FAST_MS.ln(), TAU_ANCHOR_SLOW_MS.ln());
    ((tau_ms.ln() - lo) / (hi - lo)).clamp(0.0, 1.0)
}

/// Horizonte operativo de una intención, en milisegundos. FUENTE ÚNICA.
#[inline]
pub fn operating_tau_ms(expected_duration_ms: u64, temporal_scale: f64) -> f64 {
    if expected_duration_ms > 0 {
        expected_duration_ms as f64
    } else {
        tau_from_temporal_scale(temporal_scale)
    }
}

#[derive(Debug, Clone, Copy, Default)]
pub struct ScaleState {
    pub tau_ms: f64,
    pub ewma_price: f64,
    /// EWMA de |desviación a ESTA escala| — el denominador estadísticamente
    /// correcto del z: caminata aleatoria ⇒ z ~ O(1); deriva ⇒ z crece.
    pub ewma_dev_vol: f64,
    pub momentum_z: f64,
    pub signal: f64,      // tanh(z): opinión direccional ∈ [-1,1]
    pub persistence: f64, // EWMA de sign(dev)·sign(prev_dev) — autocorrelación de sorpresas
    /// Factor adaptativo epigenético por escala armónica (0.20..3.00, inicial 1.0)
    pub epigenetic_gain: f64,
    prev_dev: f64,
}

pub struct TemporalSpectrum {
    pub scales: [ScaleState; 32],
    last_ts_ms: u64,
    /// Score espectral fusionado (paridad de riesgo 1/vol) ∈ ~[-1,1].
    pub fused_score: f64,
    /// Escala dominante (mayor |w·señal|) en ms — información, no decisión.
    /// C-05 (INFORME 14, FASE 0): acotada a la banda operativa
    /// [TAU_ANCHOR_FAST_MS, TAU_ANCHOR_SLOW_MS] — ver `update`.
    pub dominant_tau_ms: f64,
}

impl Default for TemporalSpectrum {
    fn default() -> Self {
        Self::new()
    }
}

impl TemporalSpectrum {
    pub fn new() -> Self {
        let mut scales = [ScaleState::default(); 32];
        for (i, s) in scales.iter_mut().enumerate() {
            s.tau_ms = SPECTRUM_SCALES_MS[i];
            s.epigenetic_gain = 1.0;
        }
        Self {
            scales,
            last_ts_ms: 0,
            fused_score: 0.0,
            dominant_tau_ms: 0.0,
        }
    }

    /// Actualiza TODAS las escalas con un tick. Cadencia: cada evento del
    /// motor (nanosegundo-a-nanosegundo en proceso). O(19).
    pub fn update(&mut self, price: f64, ts_ms: u64) {
        if !price.is_finite() || price <= 0.0 {
            return;
        }
        if self.last_ts_ms == 0 {
            // Primer tick: inicializar EWMAs al precio observado.
            for s in self.scales.iter_mut() {
                s.ewma_price = price;
                s.ewma_dev_vol = 1e-7; // vol de desviación semilla (evita z=∞)
                s.signal = 0.0;
            }
            self.last_ts_ms = ts_ms;
            return;
        }
        // Idempotencia parcial de X-035: dt=0 (mismo evento por dos caminos,
        // p.ej. process_event→dual) NO re-pesa; ts hacia atrás se descarta.
        // GAPS grandes siguen siendo ceguera conocida (X-035: reset explícito
        // pendiente).
        if ts_ms <= self.last_ts_ms {
            return;
        }
        let dt = (ts_ms - self.last_ts_ms) as f64;
        self.last_ts_ms = ts_ms;

        // Relajación homeostática continua de las 32 escalas epigenéticas hacia 1.0 (tau_homeo = 30 min):
        // Erradica la histeresis no-ergódica donde ganancias infladas o penalizadas se petrificaban sin disipación.
        let homeo_decay = (-dt / 1_800_000.0).exp();

        let mut w_sum = 0.0;
        let mut w_sig_sum = 0.0;
        let mut best_contrib = 0.0f64;
        let mut dominant = 0.0f64;

        for s in self.scales.iter_mut() {
            s.epigenetic_gain = 1.0 + (s.epigenetic_gain - 1.0) * homeo_decay;
            // α de la escala para el dt transcurrido: el horizonte τ_i define
            // cuánto pesa ESTE tick en esa escala. Continuo en dt y τ.
            let alpha = 1.0 - (-dt / s.tau_ms).exp();
            let prev_ewma = s.ewma_price;
            if prev_ewma <= 0.0 {
                s.ewma_price = price;
                continue;
            }
            // Sorpresa a esta escala: cuánto se aparta el precio del consenso
            // EWMA(τ_i) ANTES de absorber este tick.
            let dev = (price - prev_ewma) / prev_ewma;
            s.ewma_price += alpha * (price - prev_ewma);
            // La vol DE LA DESVIACIÓN (no del retorno por tick): es el único
            // denominador que mantiene z ~ O(1) ante caminata aleatoria en
            // TODAS las escalas — de lo contrario la deriva √N espuria satura
            // tanh con convicción de mentira (bug que el test de ruido cazó).
            s.ewma_dev_vol += alpha * (dev.abs() - s.ewma_dev_vol);

            let z = if s.ewma_dev_vol > 1e-12 {
                dev / s.ewma_dev_vol
            } else {
                0.0
            };
            // Persistencia (F4.10-ready): autocorrelación de signos de las
            // sorpresas CONSECUTIVAS de esta escala. Ruido blanco ⇒ ±1 al 50%
            // ⇒ EWMA→0; tendencia sostenida ⇒ misma firma ⇒ →+1; reversión a
            // la media ⇒ →−1. (La versión previa comparaba dev consigo misma
            // post-update — trivialmente +1; el test de ruido la cazó.)
            let agree = (dev * s.prev_dev).signum()
                * (if dev.abs() > 1e-12 && s.prev_dev.abs() > 1e-12 {
                    1.0
                } else {
                    0.0
                });
            s.persistence += alpha * (agree - s.persistence);
            s.prev_dev = dev;
            s.momentum_z = z;
            s.signal = z.clamp(-5.0, 5.0).tanh();

            // CERT-M3-H01 — FUSIÓN POR CONTENIDO INFORMATIVO (|Hurst−0.5|).
            //
            // La paridad-de-riesgo anterior (w ∝ 1/ewma_dev_vol) degeneraba:
            // la vol de sorpresa de las escalas lentas es sistemáticamente
            // menor ⇒ SIEMPRE pesaban más (el sesgo que el propio comentario
            // C-05 documentaba abajo para la τ dominante, replicado aquí en
            // la fusión que consumen arbitración/consejo/teleonomía).
            //
            // DERIVACIÓN: cada escala ya entrega su señal z-normalizada
            // (comparables entre sí). Bajo H0 (martingala) TODAS aportan ruido
            // idéntico — el peso correcto es el contenido de información de
            // cada escala, y `persistence` ∈ [0,1] (EMA de persistencia de
            // signo de la desviación) es su medida directa: el análogo
            // discreto de |Hurst − 0.5| para procesos fraccionalmente
            // integrados. persistence=0.5 ⇒ puro ruido ⇒ peso suelo (5%,
            // conserva diversificación del promedio de ensamble); 1.0 ⇒
            // tendencia pura ⇒ peso pleno.
            let gain = if s.epigenetic_gain.is_finite() && s.epigenetic_gain > 0.0 {
                s.epigenetic_gain
            } else {
                1.0
            };
            let w = (s.persistence.abs() * 2.0 * gain).clamp(0.02, 3.0);
            w_sum += w;
            let contrib = w * s.signal;
            w_sig_sum += contrib;
            if contrib.abs() > best_contrib.abs() {
                best_contrib = contrib;
                dominant = s.tau_ms;
            }
        }
        self.fused_score = if w_sum > 1e-12 {
            (w_sig_sum / w_sum).clamp(-1.0, 1.0)
        } else {
            // H0-correcto: sin información medible, promedio uniforme de las
            // señales (ruido promediado, varianza ↓ por CLT) — jamás 0 plano.
            let n = self.scales.len() as f64;
            (self.scales.iter().map(|s| s.signal).sum::<f64>() / n).clamp(-1.0, 1.0)
        };
        // C-05 (INFORME 14, FASE 0) — τ DEGENERADA. La fusión por paridad de
        // riesgo (w ∝ 1/ewma_dev_vol) degenera: la vol de sorpresa de las
        // escalas lentas es sistemáticamente menor, así que SIEMPRE pesan más
        // y la escala dominante cruda queda pegada al extremo lento del
        // espectro — escala 31 ≈ 146 años (verificado en vivo:
        // data/position_journal.jsonl con tau_ms = 4611686018427 en 2/3 de
        // las entradas), llevando a HorizonCurve.eval a extrapolar brackets
        // absurdos (+65%/−32%).
        //
        // FIX: la τ que sale del espectro hacia la DECISIÓN se acota al
        // espectro físico Y a la banda operativa de las anclas [30s, 12h].
        // El espectro de OBSERVACIÓN sigue completo (las 32 escalas siguen
        // alimentando fused_score/señales): el espectro puede VER más allá
        // de la banda, pero la DECISIÓN opera en la banda.
        self.dominant_tau_ms = dominant
            .clamp(SPECTRUM_SCALES_MS[0], SPECTRUM_SCALES_MS[31])
            .clamp(TAU_ANCHOR_FAST_MS, TAU_ANCHOR_SLOW_MS);
    }

    /// Señal de la escala más cercana a τ (interpolación log-lineal entre
    /// escalas vecinas — el espectro es CONTINUO, no una lista discreta).
    pub fn signal_at(&self, tau_ms: f64) -> f64 {
        if tau_ms <= 0.0 {
            return 0.0;
        }
        let ln_tau = tau_ms.max(1e-6).ln();
        let ln_min = (1e-6_f64).ln();
        let step = 4f64.ln();
        let idx_f = (ln_tau - ln_min) / step;
        let i0 = idx_f.floor().clamp(0.0, 30.0) as usize;
        let i1 = (i0 + 1).min(31);
        let frac = (idx_f - i0 as f64).clamp(0.0, 1.0);
        let s0 = self.scales[i0].signal;
        let s1 = self.scales[i1].signal;
        s0 * (1.0 - frac) + s1 * frac
    }

    /// Persistencia interpolada log-linealmente a τ (como signal_at — el
    /// espectro es función continua en TODAS sus observables).
    pub fn persistence_at(&self, tau_ms: f64) -> f64 {
        if tau_ms <= 0.0 {
            return 0.0;
        }
        let ln_tau = tau_ms.max(1e-6).ln();
        let ln_min = (1e-6_f64).ln();
        let step = 4f64.ln();
        let idx_f = (ln_tau - ln_min) / step;
        let i0 = idx_f.floor().clamp(0.0, 30.0) as usize;
        let i1 = (i0 + 1).min(31);
        let frac = (idx_f - i0 as f64).clamp(0.0, 1.0);
        self.scales[i0].persistence * (1.0 - frac) + self.scales[i1].persistence * frac
    }

    /// Z-score de momentum e interpolación continua de desviación a escala τ.
    /// Cuantifica analíticamente la posición de fase del precio relativo a la media de la escala.
    #[inline]
    pub fn momentum_z_at(&self, tau_ms: f64) -> f64 {
        if tau_ms <= 0.0 {
            return 0.0;
        }
        let ln_tau = tau_ms.max(1e-6).ln();
        let ln_min = (1e-6_f64).ln();
        let step = 4f64.ln();
        let idx_f = (ln_tau - ln_min) / step;
        let i0 = idx_f.floor().clamp(0.0, 30.0) as usize;
        let i1 = (i0 + 1).min(31);
        let frac = (idx_f - i0 as f64).clamp(0.0, 1.0);
        self.scales[i0].momentum_z * (1.0 - frac) + self.scales[i1].momentum_z * frac
    }

    /// Ganancia epigenética adaptativa interpolada log-linealmente a escala tau.
    #[inline]
    pub fn scale_gain_at(&self, tau_ms: f64) -> f64 {
        if tau_ms <= 0.0 {
            return 1.0;
        }
        let ln_tau = tau_ms.max(1e-6).ln();
        let ln_min = (1e-6_f64).ln();
        let step = 4f64.ln();
        let idx_f = (ln_tau - ln_min) / step;
        let i0 = idx_f.floor().clamp(0.0, 30.0) as usize;
        let i1 = (i0 + 1).min(31);
        let frac = (idx_f - i0 as f64).clamp(0.0, 1.0);
        let g0 = if self.scales[i0].epigenetic_gain > 0.0 {
            self.scales[i0].epigenetic_gain
        } else {
            1.0
        };
        let g1 = if self.scales[i1].epigenetic_gain > 0.0 {
            self.scales[i1].epigenetic_gain
        } else {
            1.0
        };
        g0 * (1.0 - frac) + g1 * frac
    }

    /// Retroalimentación Epigenética Adaptativa Multivariante Espectral por Escala.
    ///
    /// Modula continuamente la ganancia informacional (`epigenetic_gain`) de las 32 escalas
    /// basándose en los resultados reales de trading en la longitud de onda `tau_trade_ms`.
    ///
    /// - Escalas en resonancia con un trade exitoso reciben amplificación epigenética.
    /// - Escalas en resonancia con un trade perdedor son amortiguadas defensivamente.
    pub fn apply_epigenetic_outcome(&mut self, tau_trade_ms: f64, is_win: bool, pnl_pct: f64) {
        if !tau_trade_ms.is_finite() || tau_trade_ms <= 0.0 {
            return;
        }
        let ln_trade = tau_trade_ms.max(1e-6).ln();
        for s in self.scales.iter_mut() {
            let ln_scale = s.tau_ms.max(1e-6).ln();
            let delta_ln = (ln_scale - ln_trade).abs();
            // Núcleo de resonancia espectral gaussiano con ancho sigma = 0.8
            let kernel = (-0.5 * (delta_ln / 0.8).powi(2)).exp();
            if kernel > 0.05 {
                if is_win {
                    let boost = 0.06 * kernel * (1.0 + (pnl_pct.max(0.0) * 20.0).min(2.0));
                    s.epigenetic_gain = (s.epigenetic_gain + boost).clamp(0.20, 3.00);
                } else {
                    let penalty = 0.10 * kernel * (1.0 + ((-pnl_pct).max(0.0) * 15.0).min(2.0));
                    s.epigenetic_gain = (s.epigenetic_gain - penalty).clamp(0.20, 3.00);
                }
            }
        }
    }

    /// Snapshot compacto para modelos/telemetría: 32 señales + fusión.
    pub fn signals_vector(&self) -> ([f32; 32], f32) {
        let mut v = [0.0f32; 32];
        for (i, s) in self.scales.iter().enumerate() {
            v[i] = s.signal as f32;
        }
        (v, self.fused_score as f32)
    }
}

/// Estado y caracterización completa del Campo Multivariante Continuo Temporal Espectral.
/// Analiza e integra formalmente TODAS las 32 escalas espectrales (desde 1 ns hasta 146.15 años).
#[derive(Debug, Clone, Copy, Default)]
pub struct SpectralFieldState {
    /// Masa o energía informacional total integrada sobre las 32 escalas.
    pub total_energy: f64,
    /// Wavelength o escala resonante central continua τ* (centro de masa espectral en ms).
    pub resonant_tau_ms: f64,
    /// Dispersión o ancho de banda espectral σ_ln(τ) (en unidades logarítmicas naturales).
    pub spectral_bandwidth: f64,
    /// Entropía espectral de Shannon normalizada ∈ [0.0, 1.0] (0 = láser armónico, 1 = ruido blanco térmico).
    pub spectral_entropy: f64,
    /// Gradiente o inclinación espectral continua ∂s/∂ln(τ) (flujo de fase entre micro y macro).
    pub spectral_tilt: f64,
    /// Coherencia armónica de fase global evaluada en TODAS las 32 partes espectrales ∈ [-1.0, 1.0].
    pub global_coherence: f64,
    /// Proporción de confluencia: fracción de escalas con alineación favorable ∈ [0.0, 1.0].
    pub confluence_ratio: f64,
}

impl TemporalSpectrum {
    /// Consenso de precio continuo EWMA interpolado log-linealmente a τ.
    #[inline]
    pub fn ewma_price_at(&self, tau_ms: f64) -> f64 {
        if tau_ms <= 0.0 {
            return 0.0;
        }
        let ln_tau = tau_ms.max(1e-6).ln();
        let ln_min = (1e-6_f64).ln();
        let step = 4f64.ln();
        let idx_f = (ln_tau - ln_min) / step;
        let i0 = idx_f.floor().clamp(0.0, 30.0) as usize;
        let i1 = (i0 + 1).min(31);
        let frac = (idx_f - i0 as f64).clamp(0.0, 1.0);
        self.scales[i0].ewma_price * (1.0 - frac) + self.scales[i1].ewma_price * frac
    }

    /// Volatilidad de sorpresa continua interpolada log-linealmente a τ.
    #[inline]
    pub fn volatility_at(&self, tau_ms: f64) -> f64 {
        if tau_ms <= 0.0 {
            return 0.0;
        }
        let ln_tau = tau_ms.max(1e-6).ln();
        let ln_min = (1e-6_f64).ln();
        let step = 4f64.ln();
        let idx_f = (ln_tau - ln_min) / step;
        let i0 = idx_f.floor().clamp(0.0, 30.0) as usize;
        let i1 = (i0 + 1).min(31);
        let frac = (idx_f - i0 as f64).clamp(0.0, 1.0);
        self.scales[i0].ewma_dev_vol * (1.0 - frac) + self.scales[i1].ewma_dev_vol * frac
    }

    /// Desviación continua normalizada del precio respecto al consenso a escala τ.
    #[inline]
    pub fn deviation_at(&self, tau_ms: f64, price: f64) -> f64 {
        let p_ewma = self.ewma_price_at(tau_ms);
        if p_ewma > 1e-12 {
            (price - p_ewma) / p_ewma
        } else {
            0.0
        }
    }

    /// Exponente de Hurst continuo H(τ) ∈ [0.0, 1.0] evaluado analíticamente a cualquier escala τ.
    #[inline]
    pub fn hurst_at(&self, tau_ms: f64) -> f64 {
        let p = self.persistence_at(tau_ms);
        ((p + 1.0) * 0.5).clamp(0.0, 1.0)
    }

    /// Densidad de energía informacional espectral continua E(τ) = w(τ) · |s(τ)| a escala τ.
    #[inline]
    pub fn continuous_energy_density(&self, tau_ms: f64) -> f64 {
        let sig = self.signal_at(tau_ms);
        let p = self.persistence_at(tau_ms);
        let w = ((p - 0.5) * 2.0).max(0.05);
        w * sig.abs()
    }

    /// Gradiente o derivada espectral local ∂s/∂ln(τ) evaluada por diferencias finitas continuas.
    #[inline]
    pub fn spectral_gradient_at(&self, tau_ms: f64) -> f64 {
        let tau_plus = tau_ms * 2.0;
        let tau_minus = (tau_ms * 0.5).max(1e-6);
        let s_plus = self.signal_at(tau_plus);
        let s_minus = self.signal_at(tau_minus);
        (s_plus - s_minus) / (2.0 * 2.0_f64.ln())
    }

    /// Resonancia de fase armónica entre dos frecuencias temporales continuas τ_fast y τ_slow.
    /// Retorna en [-1.0, 1.0]: +1.0 = en fase perfecta, -1.0 = oposición de fase destructiva.
    #[inline]
    pub fn phase_resonance(&self, tau_fast_ms: f64, tau_slow_ms: f64) -> f64 {
        let s_fast = self.signal_at(tau_fast_ms);
        let s_slow = self.signal_at(tau_slow_ms);
        (s_fast * s_slow).clamp(-1.0, 1.0)
    }

    /// Estado espectral continuo interpolado completo ScaleState a escala τ.
    pub fn state_at(&self, tau_ms: f64) -> ScaleState {
        ScaleState {
            tau_ms,
            ewma_price: self.ewma_price_at(tau_ms),
            ewma_dev_vol: self.volatility_at(tau_ms),
            momentum_z: self.momentum_z_at(tau_ms),
            signal: self.signal_at(tau_ms),
            persistence: self.persistence_at(tau_ms),
            epigenetic_gain: self.scale_gain_at(tau_ms),
            prev_dev: 0.0,
        }
    }

    /// Caracterización cuántica e integral del Campo Multivariante Continuo Temporal Espectral.
    /// Comprende y unifica TODAS Y CADA UNA de las 32 partes espectrales (desde 1 ns hasta 146.15 años).
    pub fn spectral_field(&self, is_long: bool) -> SpectralFieldState {
        let sign = if is_long { 1.0 } else { -1.0 };
        let mut total_w = 0.0;
        let mut total_energy = 0.0;
        let mut weighted_ln_tau = 0.0;
        let mut coherent_signal_sum = 0.0;
        let mut aligned_scales_count = 0.0;

        let mut weights = [0.0f64; 32];
        let mut ln_taus = [0.0f64; 32];

        // 1. Integración armónica de las 32 partes espectrales
        for (i, s) in self.scales.iter().enumerate() {
            let ln_t = s.tau_ms.max(1e-6).ln();
            ln_taus[i] = ln_t;
            let w = (s.persistence.abs() * 2.0).clamp(0.05, 1.0);
            weights[i] = w;
            total_w += w;

            let energy_i = w * s.signal.abs();
            total_energy += energy_i;
            weighted_ln_tau += energy_i * ln_t;

            let align = s.signal * sign;
            coherent_signal_sum += w * align;
            if align > 0.05 {
                aligned_scales_count += 1.0;
            }
        }

        // 2. Centro de masa espectral continuo (Escala resonante τ*)
        let ln_tau_star = if total_energy > 1e-12 {
            weighted_ln_tau / total_energy
        } else {
            (TAU_ANCHOR_FAST_MS.ln() + TAU_ANCHOR_SLOW_MS.ln()) * 0.5
        };
        let resonant_tau_ms = ln_tau_star.exp();

        // 3. Dispersión espectral (ancho de banda) y Entropía de Shannon
        let mut var_ln_tau = 0.0;
        let mut entropy = 0.0;
        let ln_32 = 32.0f64.ln();

        for i in 0..32 {
            let p_i = if total_energy > 1e-12 {
                (weights[i] * self.scales[i].signal.abs()) / total_energy
            } else {
                1.0 / 32.0
            };
            if p_i > 1e-15 {
                entropy -= p_i * p_i.ln();
            }
            let diff = ln_taus[i] - ln_tau_star;
            var_ln_tau += p_i * diff * diff;
        }
        let spectral_bandwidth = var_ln_tau.sqrt();
        let spectral_entropy = (entropy / ln_32).clamp(0.0, 1.0);

        // 4. Inclinación / Gradiente Espectral Continuo ∂s/∂ln(τ) (Regresión lineal sobre las 32 escalas)
        let mean_ln_t = ln_taus.iter().sum::<f64>() / 32.0;
        let mean_s = self.scales.iter().map(|s| s.signal).sum::<f64>() / 32.0;
        let mut cov_ts = 0.0;
        let mut var_t = 0.0;
        for i in 0..32 {
            let dt = ln_taus[i] - mean_ln_t;
            let ds = self.scales[i].signal - mean_s;
            cov_ts += dt * ds;
            var_t += dt * dt;
        }
        let spectral_tilt = if var_t > 1e-12 { cov_ts / var_t } else { 0.0 };

        // 5. Coherencia armónica de fase global
        let global_coherence = if total_w > 1e-12 {
            (coherent_signal_sum / total_w).clamp(-1.0, 1.0)
        } else {
            0.0
        };

        let confluence_ratio = aligned_scales_count / 32.0;

        SpectralFieldState {
            total_energy,
            resonant_tau_ms,
            spectral_bandwidth,
            spectral_entropy,
            spectral_tilt,
            global_coherence,
            confluence_ratio,
        }
    }

    /// Coherencia Espectral Multivariante: evalúa el grado de alineación armónica
    /// de TODAS las 32 escalas espectrales en una dirección dada.
    /// Retorna un valor en [-1.0, 1.0]:
    /// +1.0 = resonancia armónica plena (todas las 32 partes confirman la dirección).
    /// -1.0 = contradicción armónica severa (el espectro empuja en contra).
    #[inline]
    pub fn spectral_coherence(&self, is_long: bool) -> f64 {
        let sign = if is_long { 1.0 } else { -1.0 };
        let mut total_w = 0.0;
        let mut coherent_sig = 0.0;
        for s in &self.scales {
            let w = (s.persistence.abs() * 2.0).clamp(0.05, 1.0);
            total_w += w;
            coherent_sig += w * (s.signal * sign);
        }
        if total_w > 1e-12 {
            (coherent_sig / total_w).clamp(-1.0, 1.0)
        } else {
            0.0
        }
    }

    /// Entropía espectral de Shannon normalizada ∈ [0.0, 1.0].
    pub fn spectral_entropy(&self) -> f64 {
        self.spectral_field(true).spectral_entropy
    }

    /// Wavelength o escala resonante continua central τ* (ms).
    pub fn spectral_resonance_tau(&self) -> f64 {
        self.spectral_field(true).resonant_tau_ms
    }

    /// Proyección armónica continua con filtro gaussiano logarítmico alrededor de tau_center.
    /// Sin cortes discretos de slice: el kernel abarca todo el continuo espectral de 32 escalas.
    pub fn continuous_band_projection(&self, center_tau_ms: f64, bandwidth_octaves: f64) -> f64 {
        let ln_center = center_tau_ms.max(1e-6).ln();
        let sigma = bandwidth_octaves * 2.0_f64.ln();
        let mut w_sum = 0.0;
        let mut w_sig = 0.0;
        for s in &self.scales {
            let ln_t = s.tau_ms.max(1e-6).ln();
            let dist = (ln_t - ln_center) / sigma;
            let kernel = (-0.5 * dist * dist).exp();
            let w = (s.persistence.abs() * 2.0).clamp(0.05, 1.0) * kernel;
            w_sum += w;
            w_sig += w * s.signal;
        }
        if w_sum > 1e-12 {
            (w_sig / w_sum).clamp(-1.0, 1.0)
        } else {
            0.0
        }
    }

    /// Banda táctica rápida (~1 minuto) mediante kernel gaussiano continuo.
    #[inline]
    pub fn tactical_score(&self) -> f64 {
        self.continuous_band_projection(60_000.0, 2.0)
    }

    /// Banda intermedia (~4 horas) mediante kernel gaussiano continuo.
    #[inline]
    pub fn swing_score(&self) -> f64 {
        self.continuous_band_projection(14_400_000.0, 2.5)
    }

    /// Banda macro / secular (~30 días) mediante kernel gaussiano continuo.
    #[inline]
    pub fn secular_score(&self) -> f64 {
        self.continuous_band_projection(2_592_000_000.0, 3.0)
    }

    /// Escala resonante continua universal tau* (centroide logarítmico del espectro de 32 partes).
    pub fn continuous_resonant_tau_ms(&self) -> f64 {
        let mut total_e = 0.0;
        let mut weighted_ln = 0.0;
        for s in &self.scales {
            let gain = if s.epigenetic_gain.is_finite() && s.epigenetic_gain > 0.0 {
                s.epigenetic_gain
            } else {
                1.0
            };
            let w = (s.persistence.abs() * 2.0 * gain).clamp(0.02, 3.0);
            let e = w * s.signal.abs();
            total_e += e;
            weighted_ln += e * s.tau_ms.max(1e-6).ln();
        }
        if total_e > 1e-12 {
            (weighted_ln / total_e).exp().clamp(TAU_ANCHOR_FAST_MS, TAU_ANCHOR_SLOW_MS)
        } else {
            self.dominant_tau_ms.max(30_000.0)
        }
    }

    /// Escala resonante continua de alta frecuencia (modo reactivo del espectro continuo sin cortes fijos).
    pub fn micro_resonant_tau_ms(&self) -> f64 {
        let pivot_ln = self.continuous_resonant_tau_ms().ln();
        let mut total_e = 0.0;
        let mut weighted_ln = 0.0;
        for s in &self.scales {
            let ln_tau = s.tau_ms.max(1e-6).ln();
            let weight_fast = if ln_tau <= pivot_ln {
                1.0
            } else {
                (-0.5 * (ln_tau - pivot_ln).powi(2)).exp()
            };
            let gain = if s.epigenetic_gain.is_finite() && s.epigenetic_gain > 0.0 {
                s.epigenetic_gain
            } else {
                1.0
            };
            let w = (s.persistence.abs() * 2.0 * gain).clamp(0.02, 3.0) * weight_fast;
            let e = w * s.signal.abs();
            total_e += e;
            weighted_ln += e * ln_tau;
        }
        if total_e > 1e-12 {
            (weighted_ln / total_e).exp().clamp(1_000.0, TAU_ANCHOR_SLOW_MS)
        } else {
            30_000.0
        }
    }

    /// Escala resonante continua de baja frecuencia (modo portador del espectro continuo sin cortes fijos).
    pub fn macro_resonant_tau_ms(&self) -> f64 {
        let pivot_ln = self.continuous_resonant_tau_ms().ln();
        let mut total_e = 0.0;
        let mut weighted_ln = 0.0;
        for s in &self.scales {
            let ln_tau = s.tau_ms.max(1e-6).ln();
            let weight_slow = if ln_tau >= pivot_ln {
                1.0
            } else {
                (-0.5 * (pivot_ln - ln_tau).powi(2)).exp()
            };
            let gain = if s.epigenetic_gain.is_finite() && s.epigenetic_gain > 0.0 {
                s.epigenetic_gain
            } else {
                1.0
            };
            let w = (s.persistence.abs() * 2.0 * gain).clamp(0.02, 3.0) * weight_slow;
            let e = w * s.signal.abs();
            total_e += e;
            weighted_ln += e * ln_tau;
        }
        if total_e > 1e-12 {
            (weighted_ln / total_e).exp().clamp(TAU_ANCHOR_FAST_MS, TAU_ANCHOR_SLOW_MS)
        } else {
            14_400_000.0
        }
    }
}

/// CURVAS DE PARÁMETROS DEL GENOMA EN FUNCIÓN DEL HORIZONTE (F8):
/// param(τ) = exp(a + b·ln τ_ms). El genoma evoluciona (a,b) por familia —
/// la PENDIENTE b define cómo escala el parámetro con el horizonte: una
/// decisión continua, no dos valores sueltos por bucket.
#[derive(Debug, Clone, Copy, serde::Serialize, serde::Deserialize)]
pub struct HorizonCurve {
    pub a: f64,
    pub b: f64,
}

impl HorizonCurve {
    pub fn eval(&self, tau_ms: f64) -> f64 {
        (self.a + self.b * tau_ms.max(1e-6).ln()).exp()
    }

    /// Ajusta (a,b) para pasar EXACTAMENTE por dos puntos ancla — usado para
    /// convertir genomas legacy sin cambiar su comportamiento en las bandas
    /// históricas (migración sin trauma).
    pub fn through_two_points(tau1_ms: f64, v1: f64, tau2_ms: f64, v2: f64) -> Self {
        let l1 = tau1_ms.max(1e-6).ln();
        let l2 = tau2_ms.max(1e-6).ln();
        let b = if (l2 - l1).abs() > 1e-9 {
            (v2.max(1e-12).ln() - v1.max(1e-12).ln()) / (l2 - l1)
        } else {
            0.0
        };
        let a = v1.max(1e-12).ln() - b * l1;
        Self { a, b }
    }

    /// Curva plana (constante v en todo el espectro, b = 0).
    pub fn flat(v: f64) -> Self {
        Self {
            a: v.max(1e-12).ln(),
            b: 0.0,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// D-638b: la conversión τ ↔ s es la misma en ambos sentidos.
    #[test]
    fn d638b_tau_y_escala_temporal_son_inversas() {
        for i in 0..=20 {
            let s = i as f64 / 20.0;
            let back = temporal_scale_from_tau(tau_from_temporal_scale(s));
            assert!((back - s).abs() < 1e-9, "s={s} -> {back}");
        }
    }

    /// D-638b: la banda de operación son las anclas del gen, no los extremos
    /// del espectro de observación (1 ns–146 años).
    #[test]
    fn d638b_la_banda_operativa_son_las_anclas_no_el_espectro() {
        assert!((tau_from_temporal_scale(0.0) - TAU_ANCHOR_FAST_MS).abs() < 1e-6);
        assert!((tau_from_temporal_scale(1.0) - TAU_ANCHOR_SLOW_MS).abs() < 1e-3);
        for i in 0..=20 {
            let tau = tau_from_temporal_scale(i as f64 / 20.0);
            assert!(
                tau >= TAU_ANCHOR_FAST_MS - 1e-6 && tau <= TAU_ANCHOR_SLOW_MS + 1e-3,
                "s={} produjo tau={tau} ms, fuera de la banda operativa",
                i as f64 / 20.0
            );
        }
    }

    /// La duración declarada por la señal tiene prioridad sobre el gen.
    #[test]
    fn d638b_la_duracion_declarada_tiene_prioridad() {
        assert_eq!(operating_tau_ms(90_000, 0.9), 90_000.0);
        assert!((operating_tau_ms(0, 0.0) - TAU_ANCHOR_FAST_MS).abs() < 1e-6);
    }

    #[test]
    fn escalas_cubren_el_espectro_sin_huecos() {
        assert_eq!(
            SPECTRUM_SCALES_MS[0], 1.0e-6,
            "cota inferior: 1ns = 10^-6 ms (física del reloj de CPU)"
        );
        assert!(
            SPECTRUM_SCALES_MS[31] > 3_150_000_000_000.0,
            "cota superior > 100 años"
        );
        // Log-espaciado exacto base 4: sin bandas prohibidas.
        for i in 1..32 {
            let ratio = SPECTRUM_SCALES_MS[i] / SPECTRUM_SCALES_MS[i - 1];
            assert!(
                (ratio - 4.0).abs() < 1e-6,
                "escala {} no es ×4 la anterior",
                i
            );
        }
        // Las anclas históricas viven DENTRO del espectro (no en los bordes).
        assert!(SPECTRUM_SCALES_MS[0] < TAU_ANCHOR_FAST_MS);
        assert!(TAU_ANCHOR_SLOW_MS < SPECTRUM_SCALES_MS[31]);
    }

    #[test]
    fn tendencia_lenta_detectada_por_las_escalas_largas() {
        let mut spec = TemporalSpectrum::new();
        let mut t = 1_700_000_000_000u64;
        // Rampa alcista durante 30 días a ticks de 1min: las escalas largas
        // deben opinar positivo y la fusión ser > 0.
        for i in 0..43_200u64 {
            let price = 60_000.0 * (1.0 + 0.00005 * i as f64);
            spec.update(price, t);
            t += 60_000;
        }
        assert!(
            spec.fused_score > 0.05,
            "rampa de 30d debe dar score positivo, dio {}",
            spec.fused_score
        );
        let (v, _) = spec.signals_vector();
        let slow_avg = (v[24] + v[25] + v[26]) / 3.0;
        assert!(slow_avg > 0.0, "escalas lentas deben ver la tendencia");
    }

    #[test]
    fn ruido_puro_no_genera_conviccion_persistente() {
        let mut spec = TemporalSpectrum::new();
        let mut seed: u64 = 42;
        let mut t = 1_700_000_000_000u64;
        let mut fused_sum = 0.0f64;
        let mut persistence_sum = 0.0f64;
        let n_measure = 5_000u64;
        for i in 0..50_000 {
            seed = seed.wrapping_mul(6364136223846793005).wrapping_add(1);
            let noise = ((seed >> 33) as f64 / u32::MAX as f64) - 0.5;
            spec.update(60_000.0 * (1.0 + noise * 0.001), t);
            t += 100;
            // Propiedad ESTADÍSTICA: en ruido blanco la fusión no debe tener SESGO temporal persistente.
            if i >= 50_000 - n_measure {
                fused_sum += spec.fused_score;
                persistence_sum += spec.scales[18].persistence;
            }
        }
        let mean_fused = fused_sum / n_measure as f64;
        let mean_persist = persistence_sum / n_measure as f64;
        assert!(
            mean_fused.abs() < 0.15,
            "ruido sin deriva ⇒ media temporal de fusión ≈ 0, dio {mean_fused}"
        );
        assert!(
            mean_persist.abs() < 0.3,
            "persistencia en ruido debe ser débil, dio {mean_persist}"
        );
    }

    #[test]
    fn signal_at_es_continua_entre_escalas() {
        let mut spec = TemporalSpectrum::new();
        let mut t = 1_700_000_000_000u64;
        for i in 0..10_000u64 {
            spec.update(60_000.0 * (1.0 + 0.00001 * i as f64), t);
            t += 50;
        }
        let s_exact = spec.signal_at(65_536.0);
        let s_near = spec.signal_at(66_000.0);
        assert!(
            (s_exact - s_near).abs() < 0.15,
            "continuidad: {s_exact} vs {s_near}"
        );
    }

    #[test]
    fn curva_horizonte_pasa_por_los_anchos_legacy() {
        let curve =
            HorizonCurve::through_two_points(TAU_ANCHOR_FAST_MS, 0.01, TAU_ANCHOR_SLOW_MS, 0.05);
        assert!((curve.eval(TAU_ANCHOR_FAST_MS) - 0.01).abs() < 1e-9);
        assert!((curve.eval(TAU_ANCHOR_SLOW_MS) - 0.05).abs() < 1e-9);
        let mid = curve.eval((TAU_ANCHOR_FAST_MS * TAU_ANCHOR_SLOW_MS).sqrt());
        assert!(mid > 0.01 && mid < 0.05);
        assert!(curve.b > 0.0, "TP crece con horizonte: pendiente positiva");
    }

    /// C-05 (INFORME 14, FASE 0): la τ dominante que sale hacia la DECISIÓN
    /// queda SIEMPRE dentro de la banda operativa [30s, 12h], aunque la
    /// fusión 1/vol degenerada corone a una escala ultra-lenta del espectro
    /// (el espectro puede VER más allá de la banda; la decisión no). Antes:
    /// dominant_tau_ms = 4.61e12 ms (≈146 años) en producción.
    #[test]
    fn c05_dominant_tau_opera_dentro_de_la_banda_operativa() {
        let mut spec = TemporalSpectrum::new();
        let mut t = 1_700_000_000_000u64;
        // Rampa alcista de 60 días a ticks de 1 min: el régimen donde la
        // paridad 1/vol degenera hacia las escalas más lentas.
        for i in 0..86_400u64 {
            spec.update(60_000.0 * (1.0 + 0.00005 * i as f64), t);
            t += 60_000;
        }
        assert!(
            spec.dominant_tau_ms >= TAU_ANCHOR_FAST_MS,
            "τ dominante {} por debajo de la banda rápida",
            spec.dominant_tau_ms
        );
        assert!(
            spec.dominant_tau_ms <= TAU_ANCHOR_SLOW_MS,
            "τ dominante {} por encima de la banda lenta (degeneración 1/vol)",
            spec.dominant_tau_ms
        );

        // Sin contribuciones todavía (arranque): default de banda, no 0 ni
        // 146 años.
        let mut spec2 = TemporalSpectrum::new();
        spec2.update(60_000.0, 1_700_000_000_000);
        spec2.update(60_001.0, 1_700_000_001_000);
        assert!(spec2.dominant_tau_ms >= TAU_ANCHOR_FAST_MS);
        assert!(spec2.dominant_tau_ms <= TAU_ANCHOR_SLOW_MS);
    }
}
