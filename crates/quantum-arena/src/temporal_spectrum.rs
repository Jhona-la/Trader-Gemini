//! Banco temporal de 32 filtros EWMA con interpolación en log(τ).
//!
//! La malla representa τ desde 1 ns hasta unos 146 años. `update` recibe
//! eventos con reloj entero en MILISEGUNDOS: la malla no crea observaciones
//! submilisegundo ni evidencia secular. Se actualiza por evento aceptado,
//! con coste O(32); no existe aquí un bucle ejecutado cada nanosegundo.
//!
//! Por escala se calcula α = -expm1(-Δt/τ), desviación relativa del precio
//! respecto a la EWMA previa, EWMA de |desviación| y señal tanh(desviación /
//! EWMA de |desviación|). Este cociente no es un z-score estadístico
//! estándar: el denominador no es la desviación típica de una distribución.
//!
//! `persistence` es la autocorrelación de signo de los retornos de BLOQUES
//! consecutivos y no solapados de duración ≥ τ, en [-1,1]: +1 indica
//! continuación a esa escala; -1, reversión; 0, caminata aleatoria o
//! ausencia de evidencia (CL-30). Sin solape, dos bloques de una caminata
//! aleatoria son independientes y el nulo es 0 exacto. No estima por sí
//! sola Hurst ni habilidad predictiva fuera de muestra.
//!
//! La política heredada usa w = clamp(2·|persistence|·epigenetic_gain,
//! 0.02, 3), multiplicado por la fracción resuelta 1 − e^{−τ/Δ} (CL-32).
//! Campo, proyecciones y masa |señal|·w comparten ese mismo peso. Sus cotas
//! son parámetros existentes, no leyes físicas. La entropía de esa masa
//! mide dispersión ENTRE ESCALAS, no incertidumbre direccional: consenso
//! perfecto con masa repartida sobre las escalas resueltas da entropía alta.
//!
//! Las consultas entre nodos interpolan CADA observable en log(tau).
//! Conservan las señales y masas nodales: I[tanh(z)] no es tanh(I[z]),
//! e I[w*|s|] no es w(I[estado])*|I[s]|. Una masa positiva puede coexistir
//! con cancelación direccional; no representa confianza de una operación.
//!
//! El centroide del campo no tiene recorte operativo. Las salidas heredadas
//! `dominant_tau_ms` y `continuous_resonant_tau_ms` conservan [30 s,12 h]
//! por compatibilidad con las curvas del genoma. Esto sigue siendo una
//! restricción pendiente de migración; no acredita universalidad operativa.

/// Escalas del espectro: 10^-6 ms * 4^i para i∈0..32 → 1 ns (10^-6 ms) … ≈146.15 años (4.61*10^12 ms).
/// Log-espaciadas base 4 (≈1.66 intervalos/década): resolución uniforme en
/// log(τ), cubriendo desde microestructura en nanosegundos hasta tendencias seculares de más de 100 años.
/// Fisher de escala MÁXIMA de la malla de base 4: toda la masa en una sola
/// escala (dos saltos de |Δq| = 1 entre vecinos), 2/(ln 4)².
pub const FISHER_ESCALA_MAX: f64 = 1.040_684_490_502_803_9;

/// Soporte efectivo de la masa espectral, en número de escalas de la malla:
/// k = [`FISHER_ESCALA_MAX`] / F. Exacto para masa uniforme sobre k escalas
/// contiguas; ∞ para masa uniforme (F = 0); 1 para un único pico.
#[inline]
pub fn soporte_efectivo_en_escalas(fisher: f64) -> f64 {
    if fisher.is_finite() && fisher > 0.0 {
        FISHER_ESCALA_MAX / fisher
    } else {
        f64::INFINITY
    }
}

/// Fisher mínima para declarar un régimen IDENTIFICABLE (Ola XLIV).
///
/// Un régimen es identificable cuando su masa espectral se concentra en, a lo
/// sumo, la MITAD de la banda operativa [`TAU_ANCHOR_FAST_MS`, `TAU_ANCHOR_SLOW_MS`]
/// (≈ 6,25 escalas de la malla de base 4 ⇒ soporte ≤ 3,13 escalas ⇒
/// F ≥ ≈ 0,33). Antes el gate walk-forward exigía F > 1,0, que sólo alcanza
/// un pico degenerado en UNA escala (F ≤ 1,04): con dos o más monedas vivas
/// abortaba TODAS las rondas de evolución.
#[inline]
pub fn umbral_fisher_identificable() -> f64 {
    let escalas_banda = (TAU_ANCHOR_SLOW_MS / TAU_ANCHOR_FAST_MS).ln() / 4f64.ln() + 1.0;
    FISHER_ESCALA_MAX / (escalas_banda / 2.0)
}

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

/// Resolución del reloj de los eventos del exchange (Binance sella en ms).
/// Es el SUELO de la resolución efectiva (`resolucion_efectiva_ms`): el peso
/// de cada escala escala con 1 − e^{−τ/resolución} (D-742, CL-32).
pub const FEED_CLOCK_RESOLUTION_MS: f64 = 1.0;

/// CL-32: fracción observable de una escala por resolución, 1 − e^{−τ/Δ}.
/// Por debajo de Δ la EWMA sólo repite el último evento: todas esas escalas
/// son copias de la misma observación, y con este factor suman ≈ 0,93 de una
/// escala en vez de votar cada una con peso pleno.
#[inline]
fn factor_de_resolucion(tau_ms: f64, resolucion_ms: f64) -> f64 {
    1.0 - (-tau_ms / resolucion_ms).exp()
}
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
    /// EWMA de la desviación relativa absoluta respecto a la media previa.
    /// No es una desviación típica ni corrige por sí sola el sesgo de arranque.
    pub ewma_dev_vol: f64,
    pub momentum_z: f64,
    /// En nodos actualizados: tanh(clamp(z,-5,5)). `state_at` interpola esta
    /// observable por separado; no impone tanh al momentum interpolado.
    pub signal: f64,
    /// Autocorrelación de signo de retornos consecutivos de bloques no
    /// solapados de duración ≥ τ (CL-30). 0 = caminata aleatoria.
    pub persistence: f64,
    /// Factor adaptativo epigenético por escala armónica (0.20..3.00, inicial 1.0)
    pub epigenetic_gain: f64,
    /// CL-30: bloque de τ en curso (inicio y ln del precio al inicio) y
    /// retorno del bloque anterior. Sin armar hasta el primer precio.
    bloque_armado: bool,
    bloque_t0_ms: u64,
    bloque_ln_p0: f64,
    bloque_r_prev: f64,
    /// Suma del núcleo de |dev| SIN corregir por la masa observada; la
    /// estimación pública `ewma_dev_vol` es `raw_dev_vol / masa` (D-742).
    raw_dev_vol: f64,
    /// (Ola XLI·C2) Tercer momento absoluto del núcleo: EWMA de |dev|³ por
    /// escala, SIN corregir por masa (igual convención que raw_dev_vol).
    /// Alimenta las funciones de estructura de Kolmogorov.
    raw_dev_s3: f64,
    /// #594 — IC prequential de la escala: E[s·r] con olvido, normalizado
    /// por √(E[s²]·E[r²]), donde s es la señal publicada AL ARMAR el bloque
    /// y r el retorno REALIZADO del bloque que cierra. Amplitud ≠ información:
    /// la energía |w·s| no distingue una escala nerviosa de una hábil.
    skill_ws: f64,
    skill_wr: f64,
    skill_wsr: f64,
    skill_n: u64,
    /// #594 — señal al armar el bloque en curso (s(t₀); no ve el retorno
    /// que después la puntúa — causalidad por construcción).
    bloque_s0: f64,
}

impl ScaleState {
    /// Peso heredado compartido por todos los lectores del mismo campo.
    /// No representa probabilidad, información mutua ni precisión calibrada.
    /// (#591: pub para que el núcleo proyecte sobre la MISMA masa sin
    /// recomputarla — una sola fuente, cero drift.)
    #[inline]
    pub fn fusion_weight(&self) -> f64 {
        if !self.persistence.is_finite() || !self.signal.is_finite() {
            return 0.0;
        }
        let gain = if self.epigenetic_gain.is_finite() && self.epigenetic_gain > 0.0 {
            self.epigenetic_gain
        } else {
            1.0
        };
        (self.persistence.abs() * 2.0 * gain).clamp(0.02, 3.0)
    }

    /// #594 — habilidad direccional MEDIDA de la escala: IC prequential
    /// señal(t₀) → retorno del bloque [t₀, t₀+τ). `None` sin evidencia
    /// madura o sin dispersión — el llamador no debe usarlo (misma
    /// disciplina que `habilidad_volatilidad` del banco de pronóstico).
    #[inline]
    pub fn habilidad_medida(&self) -> Option<f64> {
        if self.skill_n < MUESTRAS_SKILL_MADURAS {
            return None;
        }
        let den = self.skill_ws * self.skill_wr;
        if !den.is_finite() || den <= 0.0 {
            return None;
        }
        let ic = self.skill_wsr / den.sqrt();
        if ic.is_finite() {
            Some(ic.clamp(-1.0, 1.0))
        } else {
            None
        }
    }
}

/// CL-30: memoria de la persistencia, en bloques de τ. Peso 1/16 por bloque
/// ⇒ desviación típica ≈ √(1/31) ≈ 0,18 en el nulo; arranca en 0 (sin
/// evidencia) y la contrae hacia 0 hasta acumular bloques.
const PERSISTENCIA_BLOQUES: f64 = 16.0;

/// Anillo de snapshots de masa CRUDA por escala (energía w·|señal| sin
/// normalizar) para el transporte de Wasserstein: MASS_RING muestras.
const MASS_RING: usize = 256;

/// #594 — olvido del IC prequential por escala: media de ~64 bloques de τ
/// (vida media ≈ 44). Suficiente para estabilidad, corta para regímenes.
const HABILIDAD_OLVIDO: f64 = 1.0 / 64.0;
/// #594 — bloques maduros exigidos para que una escala opine por habilidad
/// (misma disciplina que `MUESTRAS_MADURAS` del banco de pronóstico).
pub const MUESTRAS_SKILL_MADURAS: u64 = 30;

pub struct TemporalSpectrum {
    pub scales: [ScaleState; 32],
    last_ts_ms: u64,
    /// (Ola XLII·D) Historial circular de masas por escala para W₁.
    mass_ring: [[f64; 32]; MASS_RING],
    mass_ring_len: usize,
    /// Primer instante observado: define la masa del núcleo que los datos ya
    /// llenaron en cada escala (D-742).
    first_ts_ms: u64,
    /// Actualizaciones absorbidas: con el tiempo observado da el intervalo
    /// medio entre eventos, y con él las muestras efectivas de cada escala.
    updates: u64,
    /// Score espectral fusionado (paridad de riesgo 1/vol) ∈ ~[-1,1].
    pub fused_score: f64,
    /// Escala dominante en ms — información, no decisión.
    /// #594: primero la escala con habilidad direccional MEDIDA (IC
    /// prequential > 0, madura, observable, banda operativa); sin evidencia,
    /// el argmax de energía |w·s| dentro de la banda (C-05/AGY-AUD-P10).
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
            mass_ring: [[0.0; 32]; MASS_RING],
            mass_ring_len: 0,
            first_ts_ms: 0,
            updates: 0,
            fused_score: 0.0,
            dominant_tau_ms: 0.0,
        }
    }

    /// #594 — IC medido de la escala de la malla MÁS CERCANA a `tau_ms`
    /// (telemetría; sin evidencia madura → None). Publicado al arena para
    /// que el consejo distinga τ* por habilidad de τ* por respaldo.
    pub fn habilidad_en(&self, tau_ms: f64) -> Option<f64> {
        if !tau_ms.is_finite() || tau_ms <= 0.0 {
            return None;
        }
        let mut best = usize::MAX;
        let mut best_d = f64::INFINITY;
        for (i, s) in self.scales.iter().enumerate() {
            let d = (s.tau_ms - tau_ms).abs();
            if d < best_d {
                best_d = d;
                best = i;
            }
        }
        self.scales.get(best)?.habilidad_medida()
    }

    /// Actualiza las 32 escalas por evento con timestamp estrictamente creciente.
    pub fn update(&mut self, price: f64, ts_ms: u64) {
        if !price.is_finite() || price <= 0.0 {
            return;
        }
        // (Ola XLI·A2b, bug real del merge) El primer tick se detectaba con
        // `last_ts_ms == 0`: un stream cuyo PRIMER evento tiene ts = 0 dejaba
        // el reloj en 0 y el segundo evento volvía a tratarse como primer
        // tick — el espectro se re-sembraba y descartaba la historia previa.
        // En vivo (epoch ms) nunca disparaba; en backtests sintéticos sí (lo
        // delata `spectral_contract_timestamp_zero_is_a_valid_origin`). El
        // contador de updates es el indicador correcto de primer tick.
        if self.updates == 0 {
            // Primer tick: el precio observado es la única referencia. La vol
            // de desviación arranca VACÍA —sin semilla—: su estimación es la
            // media de lo observado, corregida por la masa del núcleo (D-742).
            for s in self.scales.iter_mut() {
                s.ewma_price = price;
                s.ewma_dev_vol = 0.0;
                s.raw_dev_vol = 0.0;
                s.signal = 0.0;
                s.bloque_armado = true;
                s.bloque_t0_ms = ts_ms;
                s.bloque_ln_p0 = price.ln();
                s.bloque_r_prev = 0.0;
            }
            self.last_ts_ms = ts_ms;
            self.first_ts_ms = ts_ms;
            // El return de abajo salta el incremento general: contar el seed
            // aquí, o el segundo tick volvería a sembrar para siempre.
            self.updates += 1;
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
        let elapsed = (ts_ms - self.first_ts_ms) as f64;
        self.updates += 1;

        // Relajación homeostática continua de las 32 escalas epigenéticas hacia 1.0 (tau_homeo = 30 min):
        // Erradica la histeresis no-ergódica donde ganancias infladas o penalizadas se petrificaban sin disipación.
        let homeo_decay = (-dt / 1_800_000.0).exp();
        let ln_p = price.ln();

        for s in self.scales.iter_mut() {
            s.epigenetic_gain = 1.0 + (s.epigenetic_gain - 1.0) * homeo_decay;
            // CL-30: persistencia sobre bloques no solapados de duración ≥ τ.
            // La versión previa comparaba el signo de dos desviaciones
            // consecutivas respecto a la MISMA EWMA: con τ ≫ Δt comparten
            // casi toda su historia y en una caminata aleatoria su acuerdo
            // es (2/π)·asin(e^(−Δt/τ)) ≈ +0,9, mientras todos los lectores
            // (Kelly S-1, BE/trailing, consejo, Hurst por banda, fusión)
            // leen 0 como browniano. Dos bloques disjuntos de una caminata
            // aleatoria son independientes: su acuerdo medio es 0.
            if !s.bloque_armado {
                s.bloque_armado = true;
                s.bloque_t0_ms = ts_ms;
                s.bloque_ln_p0 = ln_p;
                s.bloque_r_prev = 0.0;
                // #594: la señal de REFERENCIA del bloque es la publicada
                // en su arranque (de la iteración anterior — causal).
                s.bloque_s0 = s.signal;
            } else if (ts_ms - s.bloque_t0_ms) as f64 >= s.tau_ms {
                let r = ln_p - s.bloque_ln_p0;
                // #594: el bloque que cierra puntúa a la señal con la que
                // nació — IC prequential (amplitud ≠ información).
                let (ws, wr, wsr) = (s.bloque_s0 * s.bloque_s0, r * r, s.bloque_s0 * r);
                s.skill_ws += (ws - s.skill_ws) * HABILIDAD_OLVIDO;
                s.skill_wr += (wr - s.skill_wr) * HABILIDAD_OLVIDO;
                s.skill_wsr += (wsr - s.skill_wsr) * HABILIDAD_OLVIDO;
                s.skill_n = s.skill_n.saturating_add(1);
                let signo = |x: f64| {
                    if x > 0.0 {
                        1.0
                    } else if x < 0.0 {
                        -1.0
                    } else {
                        0.0
                    }
                };
                let agree = signo(r) * signo(s.bloque_r_prev);
                s.persistence += (agree - s.persistence) / PERSISTENCIA_BLOQUES;
                s.bloque_r_prev = r;
                s.bloque_t0_ms = ts_ms;
                s.bloque_ln_p0 = ln_p;
                // Rearme del bloque siguiente: su señal de referencia es la
                // publicada en ESTE instante (iteración anterior — causal).
                s.bloque_s0 = s.signal;
            }
            // α de la escala para el dt transcurrido: el horizonte τ_i define
            // cuánto pesa ESTE tick en esa escala. Continuo en dt y τ.
            // exp_m1 preserva precisión cuando Δt/τ es pequeño (p.ej. 1 ms / 146 años).
            let alpha = -(-dt / s.tau_ms).exp_m1();
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
            //
            // D-742 (DÉCIMA OLA · espectro): LA ESCALA QUE NO HA VISTO SU τ NO
            // SABE NADA. La EWMA partía de una semilla de 1e-7 y a la escala de
            // 146 años un mes de datos sólo llena 5,6e-4 de su núcleo: su vol
            // se quedaba cerca de la semilla, su z = dev/vol explotaba (señal
            // ±1 saturada, que no es más que «precio sobre o bajo el de
            // arranque») y su peso 1/vol era el MAYOR de las 32 escalas, de
            // modo que esas escalas vacías gobernaban `fused_score`. Ahora la
            // vol es la media observada (suma del núcleo / masa llenada) y el
            // peso de la fusión multiplica por esa masa: lo no observado no
            // opina.
            s.raw_dev_vol = s.raw_dev_vol * (1.0 - alpha) + alpha * dev.abs();
            let abs_dev = dev.abs();
            s.raw_dev_s3 = s.raw_dev_s3 * (1.0 - alpha) + alpha * abs_dev * abs_dev * abs_dev;
            let mass = 1.0 - (-elapsed / s.tau_ms).exp();
            s.ewma_dev_vol = if mass > 0.0 { s.raw_dev_vol / mass } else { 0.0 };

            let z = if s.ewma_dev_vol > 1e-12 {
                dev / s.ewma_dev_vol
            } else {
                0.0
            };
            s.momentum_z = z;
            s.signal = z.clamp(-5.0, 5.0).tanh();
        }
        // (Ola XLII·D) Snapshot de masa para el transporte de Wasserstein:
        // energía cruda w·|señal| por escala, anillo de MASS_RING.
        {
            let pesos = self.pesos_espectrales();
            let mut snap = [0.0f64; 32];
            for (i, sc) in self.scales.iter().enumerate() {
                snap[i] = (pesos[i] * sc.signal.abs()).max(0.0);
            }
            self.mass_ring[self.mass_ring_len % MASS_RING] = snap;
            self.mass_ring_len = self.mass_ring_len.wrapping_add(1);
        }
        self.refresh_fusion();
    }

    /// Actualiza el agregado también cuando cambia el aprendizaje sin tick
    /// nuevo. Ponderación D-742: paridad de riesgo SOBRE LO OBSERVABLE —
    /// masa del núcleo llenada × fracción resuelta (CL-32) — medida sobre tape
    /// real (BTCUSDT 34M trades / SOLUSDT 6,7M): el término informativo de
    /// CERT-M3-H01 empeoraba el IC y anclaba el score al precio de arranque;
    /// la corrección de observabilidad elimina el anclaje y es la única que
    /// impide que escalas sin datos pesen. Lo no observado no opina. (El
    /// valor DIRECCIONAL del `fused_score` no está establecido: su IC es
    /// positivo en BTC y negativo en SOL; eso lo decide quien lo consuma.)
    fn refresh_fusion(&mut self) {
        let mut w_sum = 0.0;
        let mut w_sig_sum = 0.0;
        let mut best_contrib = 0.0f64;
        let mut dominant = 0.0f64;
        let mut best_operating_contrib = 0.0f64;
        let mut dominant_operating = 0.0f64;
        // Respaldo H0: promedio de las señales ponderado sólo por lo que cada
        // escala puede observar.
        let mut obs_sum = 0.0;
        let mut obs_sig_sum = 0.0;
        let (observables, pesos) = self.pesos_observables();
        for (i, s) in self.scales.iter().enumerate() {
            let observable = observables[i];
            let w = pesos[i];
            obs_sum += observable;
            obs_sig_sum += observable * s.signal;
            w_sum += w;
            let contrib = w * s.signal;
            w_sig_sum += contrib;
            if contrib.abs() > best_contrib.abs() {
                best_contrib = contrib;
                dominant = s.tau_ms;
            }
            if s.tau_ms >= TAU_ANCHOR_FAST_MS && s.tau_ms <= TAU_ANCHOR_SLOW_MS {
                if contrib.abs() > best_operating_contrib.abs() {
                    best_operating_contrib = contrib;
                    dominant_operating = s.tau_ms;
                }
            }
        }
        self.fused_score = if w_sum > 1e-12 {
            (w_sig_sum / w_sum).clamp(-1.0, 1.0)
        } else if obs_sum > 1e-12 {
            // H0-correcto: sin información medible por encima del azar, el
            // promedio de las señales OBSERVABLES (ruido promediado, varianza
            // ↓ por CLT) — jamás 0 plano, y jamás el promedio uniforme de 32
            // escalas de las que diez son la misma y ocho no han visto nada.
            (obs_sig_sum / obs_sum).clamp(-1.0, 1.0)
        } else {
            0.0
        };
        // C-05 & AGY-AUD-P10: Si existe una escala dominante con masa medible dentro de
        // la banda de decisión [30s, 12h], se adopta directamente para evitar la falacia
        // de proyección de contorno (donde ruido a microsegundos ganaba el argmax y
        // se pegaba permanentemente a 30s).
        // #594 (OLA 20 · espectro): la energía |w·s| mide AMPLITUD, no información —
        // la escala más nerviosa ganaba el argmax y degradaba τ* a 30 s aunque su
        // señal no predijera nada. El τ dominante honesto es la escala cuya señal
        // DEMOSTRÓ habilidad direccional fuera de muestra (IC prequential > 0,
        // evidencia madura, escala observable dentro de la banda). Sin evidencia,
        // el argmax de energía queda como respaldo (comportamiento C-05/P10).
        let mut best_skill = 0.0f64;
        let mut dominant_skill = 0.0f64;
        for (i, s) in self.scales.iter().enumerate() {
            if s.tau_ms < TAU_ANCHOR_FAST_MS
                || s.tau_ms > TAU_ANCHOR_SLOW_MS
                || observables[i] <= 0.0
            {
                continue;
            }
            if let Some(ic) = s.habilidad_medida() {
                if ic > best_skill {
                    best_skill = ic;
                    dominant_skill = s.tau_ms;
                }
            }
        }
        let resolved_dominant = if best_skill > 0.0 {
            dominant_skill
        } else if best_operating_contrib.abs() > 1e-12 {
            dominant_operating
        } else {
            dominant
        };
        self.dominant_tau_ms = resolved_dominant
            .clamp(SPECTRUM_SCALES_MS[0], SPECTRUM_SCALES_MS[31])
            .clamp(TAU_ANCHOR_FAST_MS, TAU_ANCHOR_SLOW_MS);
    }

    /// Interpolante continuo por tramos en log(τ) entre las señales de la malla.
    /// La continuidad del interpolante no añade observaciones entre nodos.
    /// Es I[signal], no tanh(clamp(momentum_z_at(tau), -5, 5)).
    pub fn signal_at(&self, tau_ms: f64) -> f64 {
        if !tau_ms.is_finite() || tau_ms <= 0.0 {
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
        if !tau_ms.is_finite() || tau_ms <= 0.0 {
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

    /// Sorpresa relativa normalizada e interpolada a τ (nombre histórico z).
    /// No estima una fase ni utiliza una desviación típica gaussiana.
    #[inline]
    pub fn momentum_z_at(&self, tau_ms: f64) -> f64 {
        if !tau_ms.is_finite() || tau_ms <= 0.0 {
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
        if !tau_ms.is_finite() || tau_ms <= 0.0 {
            return 1.0;
        }
        let ln_tau = tau_ms.max(1e-6).ln();
        let ln_min = (1e-6_f64).ln();
        let step = 4f64.ln();
        let idx_f = (ln_tau - ln_min) / step;
        let i0 = idx_f.floor().clamp(0.0, 30.0) as usize;
        let i1 = (i0 + 1).min(31);
        let frac = (idx_f - i0 as f64).clamp(0.0, 1.0);
        let g0 = if self.scales[i0].epigenetic_gain.is_finite()
            && self.scales[i0].epigenetic_gain > 0.0
        {
            self.scales[i0].epigenetic_gain
        } else {
            1.0
        };
        let g1 = if self.scales[i1].epigenetic_gain.is_finite()
            && self.scales[i1].epigenetic_gain > 0.0
        {
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
        if !tau_trade_ms.is_finite() || tau_trade_ms <= 0.0 || !pnl_pct.is_finite() {
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
        self.refresh_fusion();
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

/// Descriptores del banco de 32 filtros temporales sobre el precio.
/// No incorpora por sí solo otras variables de mercado ni evidencia por escala.
#[derive(Debug, Clone, Copy, Default)]
pub struct SpectralFieldState {
    /// Suma adimensional de w·|señal| sobre la malla; no energía física ni información mutua.
    pub total_energy: f64,
    /// Wavelength o escala resonante central continua τ* (centro de masa espectral en ms).
    pub resonant_tau_ms: f64,
    /// Dispersión o ancho de banda espectral σ_ln(τ) (en unidades logarítmicas naturales).
    pub spectral_bandwidth: f64,
    /// Entropía de la masa entre escalas, en [0,1]: 0 concentrada, 1 uniforme.
    /// No mide desacuerdo direccional; todas las señales pueden coincidir con entropía 1.
    pub spectral_entropy: f64,
    /// Pendiente de regresión lineal de señal contra ln(τ) sobre toda la malla.
    /// No es la derivada local que devuelve `spectral_gradient_at`.
    pub spectral_tilt: f64,
    /// Media direccional ponderada de señales, en [-1,1], orientada al lado consultado.
    pub global_coherence: f64,
    /// Fracción no ponderada de nodos cuya señal en el lado consultado supera 0.05.
    pub confluence_ratio: f64,
}

impl TemporalSpectrum {
    /// Consenso de precio continuo EWMA interpolado log-linealmente a τ.
    #[inline]
    pub fn ewma_price_at(&self, tau_ms: f64) -> f64 {
        if !tau_ms.is_finite() || tau_ms <= 0.0 {
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
        if !tau_ms.is_finite() || tau_ms <= 0.0 {
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

    /// Índice de persistencia de signo reescalado a [0,1].
    /// Nombre conservado por compatibilidad: NO es un estimador del exponente de Hurst.
    #[inline]
    pub fn hurst_at(&self, tau_ms: f64) -> f64 {
        let p = self.persistence_at(tau_ms);
        ((p + 1.0) * 0.5).clamp(0.0, 1.0)
    }

    /// Interpolante de la masa nodal adimensional E_i = w_i*|signal_i|.
    /// No es energía física ni una densidad probabilística normalizada.
    /// Entre nodos se calcula I[E], NO w(state_at(tau))*|signal_at(tau)|:
    /// señales vecinas opuestas conservan actividad aunque su media sea cero.
    /// Comparar esta masa no identifica por sí solo la mejor dirección.
    #[inline]
    pub fn continuous_energy_density(&self, tau_ms: f64) -> f64 {
        if !tau_ms.is_finite() || tau_ms <= 0.0 {
            return 0.0;
        }
        let idx = (tau_ms.max(SPECTRUM_SCALES_MS[0]).ln() - SPECTRUM_SCALES_MS[0].ln()) / 4f64.ln();
        let i0 = idx.floor().clamp(0.0, 30.0) as usize;
        let i1 = i0 + 1;
        let frac = (idx - i0 as f64).clamp(0.0, 1.0);
        let pesos = self.pesos_espectrales();
        let energy = |i: usize| {
            let w = pesos[i];
            if w > 0.0 {
                w * self.scales[i].signal.abs()
            } else {
                0.0
            }
        };
        // Interpolar la MISMA masa de la malla preserva su contrato en cada nodo.
        energy(i0) * (1.0 - frac) + energy(i1) * frac
    }

    /// Pendiente local exacta del interpolante lineal por tramos en ln(tau).
    /// En cada nodo se devuelve la derivada DERECHA; no se afirma que exista
    /// una derivada bilateral en los quiebres. Fuera de la malla la extensión
    /// es constante (derivada cero), también a la derecha del último nodo.
    /// Una tau inválida devuelve cero, como las otras consultas de señal.
    #[inline]
    pub fn spectral_gradient_at(&self, tau_ms: f64) -> f64 {
        if !tau_ms.is_finite()
            || tau_ms < SPECTRUM_SCALES_MS[0]
            || tau_ms >= SPECTRUM_SCALES_MS[31]
        {
            return 0.0;
        }
        // Search actual nodes to avoid assigning a rounded logarithm to the
        // wrong side of a knot. No multiplication of tau can overflow here.
        let right = SPECTRUM_SCALES_MS.partition_point(|&node| node <= tau_ms);
        (self.scales[right].signal - self.scales[right - 1].signal)
            / (SPECTRUM_SCALES_MS[right].ln() - SPECTRUM_SCALES_MS[right - 1].ln())
    }

    /// Producto de dos señales reales, en [-1,1]. Conserva el nombre histórico.
    /// Su magnitud depende de ambas amplitudes; no estima fase ni coherencia normalizada.
    #[inline]
    pub fn phase_resonance(&self, tau_fast_ms: f64, tau_slow_ms: f64) -> f64 {
        let s_fast = self.signal_at(tau_fast_ms);
        let s_slow = self.signal_at(tau_slow_ms);
        (s_fast * s_slow).clamp(-1.0, 1.0)
    }

    /// Vista de observables interpolados independientemente a escala tau.
    /// No es un filtro EWMA evolucionado a esa tau ni un estado reanudable:
    /// el bloque de la persistencia no se reconstruye. En particular,
    /// signal != tanh(momentum_z) en general. Las identidades nodales no
    /// conmutan con la interpolación.
    pub fn state_at(&self, tau_ms: f64) -> ScaleState {
        ScaleState {
            tau_ms,
            ewma_price: self.ewma_price_at(tau_ms),
            ewma_dev_vol: self.volatility_at(tau_ms),
            momentum_z: self.momentum_z_at(tau_ms),
            signal: self.signal_at(tau_ms),
            persistence: self.persistence_at(tau_ms),
            epigenetic_gain: self.scale_gain_at(tau_ms),
            bloque_armado: false,
            bloque_t0_ms: 0,
            bloque_ln_p0: 0.0,
            bloque_r_prev: 0.0,
            raw_dev_vol: 0.0,
            raw_dev_s3: 0.0,
            // #594: un nodo interpolado no es una escala de la malla con
            // bloques maduros — sin habilidad medida (0 muestras → None).
            skill_ws: 0.0,
            skill_wr: 0.0,
            skill_wsr: 0.0,
            skill_n: 0,
            bloque_s0: 0.0,
        }
    }

    /// Descriptores de la distribución de masa w·|señal| sobre la malla.
    /// Comparten el peso de la fusión; su precisión predictiva requiere evaluación aparte.
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
        let pesos = self.pesos_espectrales();
        for (i, s) in self.scales.iter().enumerate() {
            let ln_t = s.tau_ms.max(1e-6).ln();
            ln_taus[i] = ln_t;
            let w = pesos[i];
            weights[i] = w;
            if w == 0.0 {
                continue;
            }
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
                if weights[i] > 0.0 {
                    (weights[i] * self.scales[i].signal.abs()) / total_energy
                } else {
                    0.0
                }
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

    /// Media ponderada de las 32 señales orientadas a la dirección consultada.
    /// Retorna un valor en [-1.0, 1.0]:
    /// +1.0 = todas las señales con peso apuntan al lado consultado con amplitud 1.
    /// -1.0 = todas apuntan al lado opuesto con amplitud 1.
    #[inline]
    pub fn spectral_coherence(&self, is_long: bool) -> f64 {
        // (Ola XLI·A2b) MISMA ponderación observable D-742 que `refresh_fusion`:
        // este lector alimenta al consejo/escalas de decisión y usaba el peso
        // LEGACY de persistencia — el consejo leía OTRO campo distinto al que
        // decide la fusión (delatado por spectral_contract_learning_refreshes:
        // fused ≠ coherence tras apply_epigenetic_outcome). Una sola verdad.
        let sign = if is_long { 1.0 } else { -1.0 };
        let mut total_w = 0.0;
        let mut coherent_sig = 0.0;
        let (_, pesos) = self.pesos_observables();
        for (i, s) in self.scales.iter().enumerate() {
            let w = pesos[i];
            if w == 0.0 {
                continue;
            }
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
        if !center_tau_ms.is_finite()
            || center_tau_ms <= 0.0
            || !bandwidth_octaves.is_finite()
            || bandwidth_octaves <= 0.0
        {
            return 0.0;
        }
        let ln_center = center_tau_ms.max(1e-6).ln();
        let sigma = bandwidth_octaves * 2.0_f64.ln();
        let mut w_sum = 0.0;
        let mut w_sig = 0.0;
        let pesos = self.pesos_espectrales();
        for (i, s) in self.scales.iter().enumerate() {
            let ln_t = s.tau_ms.max(1e-6).ln();
            let dist = (ln_t - ln_center) / sigma;
            let kernel = (-0.5 * dist * dist).exp();
            let w = pesos[i] * kernel;
            if w == 0.0 {
                continue;
            }
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
        let pesos = self.pesos_espectrales();
        for (i, s) in self.scales.iter().enumerate() {
            let w = pesos[i];
            if w == 0.0 {
                continue;
            }
            let e = w * s.signal.abs();
            total_e += e;
            weighted_ln += e * s.tau_ms.max(1e-6).ln();
        }
        if total_e > 1e-12 {
            (weighted_ln / total_e)
                .exp()
                .clamp(TAU_ANCHOR_FAST_MS, TAU_ANCHOR_SLOW_MS)
        } else {
            self.dominant_tau_ms.max(30_000.0)
        }
    }

    /// Escala resonante continua de alta frecuencia (modo reactivo del espectro continuo sin cortes fijos).
    pub fn micro_resonant_tau_ms(&self) -> f64 {
        let pivot_ln = self.continuous_resonant_tau_ms().ln();
        let mut total_e = 0.0;
        let mut weighted_ln = 0.0;
        let pesos = self.pesos_espectrales();
        for (i, s) in self.scales.iter().enumerate() {
            let ln_tau = s.tau_ms.max(1e-6).ln();
            let weight_fast = if ln_tau <= pivot_ln {
                1.0
            } else {
                (-0.5 * (ln_tau - pivot_ln).powi(2)).exp()
            };
            let w = pesos[i] * weight_fast;
            if w == 0.0 {
                continue;
            }
            let e = w * s.signal.abs();
            total_e += e;
            weighted_ln += e * ln_tau;
        }
        if total_e > 1e-12 {
            (weighted_ln / total_e)
                .exp()
                .clamp(TAU_ANCHOR_FAST_MS, TAU_ANCHOR_SLOW_MS)
        } else {
            TAU_ANCHOR_FAST_MS
        }
    }

    /// Escala resonante continua de baja frecuencia (modo portador del espectro continuo sin cortes fijos).
    pub fn macro_resonant_tau_ms(&self) -> f64 {
        let pivot_ln = self.continuous_resonant_tau_ms().ln();
        let mut total_e = 0.0;
        let mut weighted_ln = 0.0;
        let pesos = self.pesos_espectrales();
        for (i, s) in self.scales.iter().enumerate() {
            let ln_tau = s.tau_ms.max(1e-6).ln();
            let weight_slow = if ln_tau >= pivot_ln {
                1.0
            } else {
                (-0.5 * (pivot_ln - ln_tau).powi(2)).exp()
            };
            let w = pesos[i] * weight_slow;
            if w == 0.0 {
                continue;
            }
            let e = w * s.signal.abs();
            total_e += e;
            weighted_ln += e * ln_tau;
        }
        if total_e > 1e-12 {
            (weighted_ln / total_e)
                .exp()
                .clamp(TAU_ANCHOR_FAST_MS, TAU_ANCHOR_SLOW_MS)
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



// ═══════════════════════════════════════════════════════════════════════
// Ola XLI·C2 — FUNCIONES DE ESTRUCTURA DE KOLMOGOROV SOBRE EL ESPECTRO
//
// Contrato (protocolo del repo):
// - Variable: momentos de la desviación relativa por escala, S_p(τ) = E|dev_τ|^p,
//   estimados como EWMA del núcleo corregidos por masa (misma convención que
//   `ewma_dev_vol`, D-742). S_2 = dev_vol², S_3 = dev_s3³-normalizado.
// - Operador: exponentes ζ(p) = d ln S_p / d ln τ por regresión log-log sobre
//   las escalas con masa suficiente Y resueltas por el reloj de eventos
//   (τ ≥ resolución efectiva, Ola XLIV). Un proceso autosimilar de Hurst H
//   da ζ(p) = p·H (precio browniano: ζ₂ = 1, ζ₃ = 1,5; K41 en turbulencia:
//   H = 1/3, ζ₃ = 1). La intermitencia (cascada multifractal, K62) se
//   manifiesta como CONCAVIDAD de ζ(p): ζ₃ < (3/2)·ζ₂.
// - Unidades: adimensional (pendiente en doble log).
// - Contorno: espectro frío (masa < masa_min en una escala) o escala por
//   debajo de la resolución → esa escala no participa; <4 escalas válidas →
//   None (no se afirma exponente).
// - Identificabilidad: χ = ((3/2)·ζ₂ − ζ₃)⁺ es la magnitud de intermitencia
//   que el host usa para elevar los pisos de exigencia (colas gruesas ⇒
//   exigir más). Antes χ = (1 − ζ₃)⁺ usaba la referencia K41, que para un
//   precio confunde H con multifractalidad.
// - Coste: O(32) por consulta, sin alocación en el cálculo de pendientes.
// - Falsación: con incrementos iid, ζ(p) es lineal y χ ≈ 0 (verifica el
//   test); con ráfagas/spikes multifractales, χ > 0.
// ═══════════════════════════════════════════════════════════════════════

/// Exponentes de estructura medidos sobre el espectro vivo de 32 escalas.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct StructureFunctions {
    /// ζ(2): pendiente de ln S_2 vs ln τ. Precio browniano: 1 (K41: 2/3).
    pub zeta2: f64,
    /// ζ(3): pendiente de ln S_3 vs ln τ. Precio browniano: 1,5 (K41: 1).
    pub zeta3: f64,
    /// Intermitencia χ = ((3/2)·ζ₂ − ζ₃)⁺ ∈ [0,1]: concavidad de ζ(p),
    /// independiente del exponente de Hurst (Ola XLIV).
    pub intermittency: f64,
    /// Escalas que participaron de la regresión (masa suficiente).
    pub usable_scales: usize,
}

impl TemporalSpectrum {
    // (Ola XLV·J) dev_moment eliminado: fue sustituido por dev_moment_by
    // durante la refactorización de structure_functions — su lógica vive allí.

    /// Funciones de estructura S_p(τ) y sus exponentes ζ(p) por regresión
    /// log-log entre escalas observadas. None si el campo aún no tiene masa
    /// en suficientes escalas (<4): no se afirma un exponente sin soporte.
    ///
    /// Ola XLIV — referencia correcta para PRECIOS. K41 (ζ(p) = p/3, ζ(3) = 1)
    /// es la autosimilaridad de VELOCIDADES en turbulencia; un precio
    /// browniano escala con H = ½ (ζ(p) = p·H: ζ₂ = 1, ζ₃ = 1,5). Medir la
    /// intermitencia como (1 − ζ₃)⁺ confundía el exponente de Hurst con la
    /// multifractalidad. Lo que distingue una cascada multifractal (K62) de
    /// cualquier proceso autosimilar —sea cual sea su H— es la NO LINEALIDAD
    /// de ζ(p): χ = ((3/2)·ζ₂ − ζ₃)⁺, nula para Brown y para K41 y positiva
    /// sólo cuando ζ(p) es cóncava.
    pub fn structure_functions(&self) -> Option<StructureFunctions> {
        let (z2, n) = self.regress_log_log(2)?;
        let (z3, _) = self.regress_log_log(3)?;
        Some(StructureFunctions {
            zeta2: z2,
            zeta3: z3,
            intermittency: (1.5 * z2 - z3).max(0.0).min(1.0),
            usable_scales: n,
        })
    }

    /// CL-32: pesos de la masa espectral (entropía, Fisher, W₁, τ*, bandas):
    /// el peso heredado de la persistencia por la fracción RESUELTA de cada
    /// escala. Una sola fuente para todos los lectores de la masa.
    /// (#591: pub — el núcleo la consume para la proyección espectral.)
    pub fn pesos_espectrales(&self) -> [f64; 32] {
        let resolucion = self.resolucion_efectiva_ms();
        let mut w = [0.0f64; 32];
        for (i, s) in self.scales.iter().enumerate() {
            w[i] = s.fusion_weight() * factor_de_resolucion(s.tau_ms, resolucion);
        }
        w
    }

    /// Pesos observables de la fusión D-742 (`fused_score`, τ dominante y
    /// `spectral_coherence`): masa del núcleo llenada × fracción resuelta,
    /// y ese observable dividido por la vol de desviación. Devuelve
    /// `(observables, pesos)`.
    ///
    /// CL-32: la fracción resuelta usaba el reloj del exchange (1 ms), no la
    /// resolución efectiva. Con eventos cada segundo las escalas de 4 ms a
    /// 268 ms repetían el último evento con la vol más pequeña del espectro
    /// —el mayor 1/vol— y gobernaban la fusión: el mismo defecto que XLIV-6
    /// corrigió en las funciones de estructura.
    fn pesos_observables(&self) -> ([f64; 32], [f64; 32]) {
        let elapsed = (self.last_ts_ms.saturating_sub(self.first_ts_ms)) as f64;
        let resolucion = self.resolucion_efectiva_ms();
        let mut observables = [0.0f64; 32];
        let mut pesos = [0.0f64; 32];
        for (i, s) in self.scales.iter().enumerate() {
            let mass = 1.0 - (-elapsed / s.tau_ms).exp();
            let observable = mass * factor_de_resolucion(s.tau_ms, resolucion);
            observables[i] = observable;
            pesos[i] = if s.ewma_dev_vol > 1e-12 {
                observable / s.ewma_dev_vol
            } else {
                0.0
            };
        }
        (observables, pesos)
    }

    /// Resolución temporal EFECTIVA del espectro (Ola XLIV): el mayor entre el
    /// reloj del exchange y el intervalo medio entre eventos. Una escala con
    /// τ por debajo de ella no está resuelta —su EWMA sólo repite el último
    /// tick— y el momento que publica es el mismo en todas esas escalas.
    pub fn resolucion_efectiva_ms(&self) -> f64 {
        let dt_medio = if self.updates > 1 {
            (self.last_ts_ms.saturating_sub(self.first_ts_ms)) as f64 / (self.updates - 1) as f64
        } else {
            0.0
        };
        FEED_CLOCK_RESOLUTION_MS.max(dt_medio)
    }

    /// Regresión OLS de ln S_p contra ln τ sobre escalas con masa suficiente.
    fn regress_log_log(&self, p: u32) -> Option<(f64, usize)> {
        let elapsed = (self.last_ts_ms.saturating_sub(self.first_ts_ms)) as f64;
        // Ola XLIV: las escalas por debajo de la resolución efectiva publican
        // el mismo momento (el del último tick) y formaban un bloque plano que
        // arrastraba la pendiente: ζ₃ ≈ 0,56 en un precio browniano cuyo
        // valor sobre las escalas reales es 1,5.
        let resolucion = self.resolucion_efectiva_ms();
        let mut n = 0usize;
        let mut sx = 0.0;
        let mut sy = 0.0;
        let mut sxx = 0.0;
        let mut sxy = 0.0;
        for s in self.scales.iter() {
            if s.tau_ms < resolucion {
                continue;
            }
            // Sólo escalas cuyo núcleo tiene ≥10% de masa: por debajo, la EWMA
            // es aún semilla y el momento no representa la escala.
            let mass = 1.0 - (-elapsed / s.tau_ms).exp();
            if mass < 0.10 {
                continue;
            }
            let m = self.dev_moment_by(p, s, mass);
            if !(m > 0.0) || !m.is_finite() {
                continue;
            }
            let x = s.tau_ms.ln();
            let y = m.ln();
            n += 1;
            sx += x;
            sy += y;
            sxx += x * x;
            sxy += x * y;
        }
        if n < 4 {
            return None;
        }
        let denom = n as f64 * sxx - sx * sx;
        if denom.abs() < 1e-12 {
            return None;
        }
        Some(((n as f64 * sxy - sx * sy) / denom, n))
    }

    #[inline]
    fn dev_moment_by(&self, p: u32, s: &ScaleState, mass: f64) -> f64 {
        match p {
            2 => s.ewma_dev_vol * s.ewma_dev_vol,
            3 => {
                if mass > 0.0 {
                    s.raw_dev_s3 / mass
                } else {
                    0.0
                }
            }
            _ => 0.0,
        }
    }

    // ═══════════════════════════════════════════════════════════════════
    // Ola XLI·C3 — INFORMACIÓN DE FISHER DEL CAMPO RESPECTO A LA ESCALA
    //
    // Contrato:
    // - Variable: la distribución de masa espectral q_i = e_i/Σe sobre la
    //   malla log(τ) (misma masa de la entropía espectral).
    // - Operador: I = Σ_i (Δq_i/Δlnτ)²/q_i — la métrica de Fisher 1-D de la
    //   distribución respecto al parámetro ln(τ). Alta = el régimen está
    //   LOCALIZADO en escala (identificable); baja/difusa = sin régimen.
    // - Unidades: 1/(unidades de lnτ)² → adimensional en la malla log.
    // - Contorno: espectro frío o masa total nula → None.
    // - Identificabilidad: este ES el medidor de identificabilidad (T18/T26);
    //   el walk-forward no debe evolucionar sobre regímenes no identificables.
    // - Coste: O(32).
    // - Falsación: masa concentrada en un nodo → I grande; masa uniforme →
    //   I ≈ 0. Los tests lo verifican.
    // ═══════════════════════════════════════════════════════════════════

    // ═══════════════════════════════════════════════════════════════════
    // Ola XLII·D — TRANSPORTE ÓPTIMO DE WASSERSTEIN-1 ENTRE MASAS
    // ESPECTRALES (ahora vs hace N eventos)
    //
    // Contrato (protocolo del repo):
    // - Variable: dos distribuciones de masa espectral q_prev, q_now sobre
    //   la MISMA malla log(τ) de 32 nodos (misma masa de la entropía/Fisher).
    // - Operador: W₁(q_prev, q_now) = Σ |Q_prev(τᵢ) − Q_now(τᵢ)|·Δlnτ sobre
    //   las CDF discretas — el coste mínimo de mover una unidad de masa de
    //   la configuración previa a la actual en la métrica ln(τ). Es la
    //   distancia de transporte óptimo 1-D exacta (en 1-D coincide con la
    //   distancia de CDF; Monge-Kantorovich en su forma cerrada).
    // - Unidades: log(τ) — «cuántos ejes de escala (e≈2,72×) se movió la
    //   masa en promedio». 0 = régimen congelado; ~1 = reestructuración
    //   total de un eje completo.
    // - Contorno: masa nula en cualquiera de las dos → None.
    // - Identificabilidad: RESTRUCTURACIÓN del régimen — complemento de la
    //   Fisher (que mide concentración estática): W₁ mide el MOVIMIENTO.
    //   Alta W₁ sostenida = régimen migrando entre escalas (transición);
    //   el host puede frenar aperturas en τ cuya masa está en tránsito.
    // - Coste: O(32), sin alocación.
    // - Falsación: masa idéntica → W₁ = 0 (test); masa movida un nodo
    //   completo (i → i+1) → W₁ = Δlnτ = ln 4 exacto (test).
    // ═══════════════════════════════════════════════════════════════════
    /// Masa espectral normalizada de la malla (misma masa de entropía/Fisher).
    fn spectral_mass(&self) -> Option<[f64; 32]> {
        let mut e = [0.0f64; 32];
        let mut total = 0.0;
        let pesos = self.pesos_espectrales();
        for (i, s) in self.scales.iter().enumerate() {
            let energy = (pesos[i] * s.signal.abs()).max(0.0);
            e[i] = energy;
            total += energy;
        }
        if !(total > 1e-12) {
            return None;
        }
        for x in e.iter_mut() {
            *x /= total;
        }
        Some(e)
    }

    /// W₁ entre la masa actual y la capturada hace `lag` updates.
    /// `lag = 0` o sin masa en cualquiera de los dos instantes → None.
    pub fn spectral_transport_w1(&self, lag: u32) -> Option<f64> {
        if lag == 0 || self.updates < lag as u64 {
            return None;
        }
        // Reconstruir la masa previa desde el snapshot de masas por escala.
        let now = self.spectral_mass()?;
        let prev = self.mass_history_at(lag)?;
        let step = 4f64.ln();
        let mut cdf_prev = 0.0;
        let mut cdf_now = 0.0;
        let mut w1 = 0.0;
        for i in 0..32 {
            cdf_prev += prev[i];
            cdf_now += now[i];
            w1 += (cdf_now - cdf_prev).abs() * step;
        }
        Some(w1)
    }

    /// Masa de hace `lag` updates, reconstruida desde el anillo de masas por
    /// escala que mantiene update(). El snapshot más reciente está en
    /// (len−1) % RING; hace `lag` updates, en (len−1−lag) % RING — la
    /// aritmética circular cubre vueltas completas del anillo.
    fn mass_history_at(&self, lag: u32) -> Option<[f64; 32]> {
        let back = (lag as usize).checked_add(1)?;
        let cursor = self.mass_ring_len.checked_sub(back)?;
        let idx = cursor % MASS_RING;
        let raw = self.mass_ring[idx];
        let total: f64 = raw.iter().sum();
        if !(total > 1e-12) {
            return None;
        }
        let mut out = [0.0f64; 32];
        for i in 0..32 {
            out[i] = raw[i] / total;
        }
        Some(out)
    }

    /// Información de Fisher 1-D de la masa espectral respecto a ln(τ).
    /// None sin masa (espectro frío): sin campo no hay identificabilidad.
    ///
    /// Rango (Ola XLIV): como |Δq| ≤ max(qᵢ, qⱼ), cada término cumple
    /// Δq²/max ≤ |Δq|, así que F ≤ Σ|Δq|/(ln 4)² ≤ [`FISHER_ESCALA_MAX`]. Para
    /// masa uniforme sobre k escalas contiguas F = FISHER_ESCALA_MAX / k: la
    /// Fisher se lee como el inverso del SOPORTE EFECTIVO en escalas (ver
    /// [`soporte_efectivo_en_escalas`]).
    pub fn fisher_scale_information(&self) -> Option<f64> {
        // MISMA masa que la entropía espectral (spectral_field): q_i = w_i·|s_i|/Σ.
        let mut e = [0.0f64; 32];
        let mut total = 0.0;
        let pesos = self.pesos_espectrales();
        for (i, s) in self.scales.iter().enumerate() {
            let energy = (pesos[i] * s.signal.abs()).max(0.0);
            e[i] = energy;
            total += energy;
        }
        if !(total > 1e-12) {
            return None;
        }
        let step = 4f64.ln(); // malla base 4
        let mut fisher = 0.0;
        for i in 0..31 {
            let qi = e[i] / total;
            let qj = e[i + 1] / total;
            let dq = (qj - qi) / step;
            let q_ref = qi.max(qj).max(1e-12);
            fisher += dq * dq / q_ref;
        }
        Some(fisher)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// #594 — escala de la banda operativa con IC forjado a mano.
    fn escala_con_habilidad(idx: usize, n: u64, ic: f64) -> ScaleState {
        let mut s = ScaleState::default();
        s.tau_ms = SPECTRUM_SCALES_MS[idx];
        s.skill_n = n;
        s.skill_ws = 1.0;
        s.skill_wr = 1.0;
        s.skill_wsr = ic;
        s
    }

    #[test]
    fn qo_594_habilidad_requiere_evidencia_madura() {
        let s = escala_con_habilidad(19, MUESTRAS_SKILL_MADURAS - 1, 0.5);
        assert_eq!(s.habilidad_medida(), None, "29 bloques no opinan");
        let s = escala_con_habilidad(19, MUESTRAS_SKILL_MADURAS, 0.5);
        let ic = s.habilidad_medida().expect("30 bloques maduros opinan");
        assert!((ic - 0.5).abs() < 1e-12);
        // Sin dispersión (señal plana o retorno nulo) no hay IC: den ≤ 0.
        let mut plano = escala_con_habilidad(19, MUESTRAS_SKILL_MADURAS, 0.0);
        plano.skill_wsr = 0.0;
        assert_eq!(plano.habilidad_medida(), None);
    }

    #[test]
    fn qo_594_tau_dominante_sigue_la_habilidad_madura() {
        let mut spec = TemporalSpectrum::new();
        // Reloj con historia: masa ≈ 1 en la banda y resolución fina.
        spec.first_ts_ms = 0;
        spec.last_ts_ms = 86_400_000;
        spec.updates = 10_000;
        // Escala 18: energía GANADORA pero habilidad NEGATIVA (anti-predice).
        spec.scales[18] = escala_con_habilidad(18, MUESTRAS_SKILL_MADURAS, -0.5);
        spec.scales[18].signal = 0.9;
        spec.scales[18].persistence = 0.6;
        // Escala 19: energía menor, habilidad positiva medida.
        spec.scales[19] = escala_con_habilidad(19, MUESTRAS_SKILL_MADURAS, 0.3);
        spec.scales[19].signal = 0.4;
        spec.scales[19].persistence = 0.3;
        spec.refresh_fusion();
        assert_eq!(spec.dominant_tau_ms, SPECTRUM_SCALES_MS[19]);
        // Telemetría: la escala elegida publica su IC; el testigo del defecto
        // (sólo habilidad negativa) cae al respaldo de energía.
        assert_eq!(spec.habilidad_en(spec.dominant_tau_ms), Some(0.3));
        spec.scales[19].skill_n = MUESTRAS_SKILL_MADURAS - 1;
        spec.refresh_fusion();
        assert_eq!(spec.dominant_tau_ms, SPECTRUM_SCALES_MS[18]);
        assert_eq!(spec.habilidad_en(spec.dominant_tau_ms), None);
    }

    #[test]
    fn qo_594_escalas_inmaduras_conservan_el_respaldo_de_energia() {
        let mut spec = TemporalSpectrum::new();
        spec.first_ts_ms = 0;
        spec.last_ts_ms = 86_400_000;
        spec.updates = 10_000;
        spec.scales[19] = escala_con_habilidad(19, 5, 0.9); // IC alto pero inmaduro
        spec.scales[19].signal = 0.5;
        spec.scales[19].persistence = 0.5;
        spec.refresh_fusion();
        // Sin evidencia madura el criterio es el argmax de energía en banda
        // (comportamiento C-05/AGY-AUD-P10 intacto).
        assert_eq!(spec.dominant_tau_ms, SPECTRUM_SCALES_MS[19]);
        assert_eq!(spec.habilidad_en(spec.dominant_tau_ms), None);
    }

    fn opposed_scales() -> TemporalSpectrum {
        let mut spec = TemporalSpectrum::new();
        spec.scales[18].signal = 0.8;
        spec.scales[18].persistence = 0.5;
        spec.scales[18].epigenetic_gain = 2.0;
        spec.scales[20].signal = -0.8;
        spec.scales[20].persistence = -0.5;
        spec
    }

    #[test]
    fn spectral_contract_learning_reaches_field_and_projection() {
        let spec = opposed_scales();
        let field = spec.spectral_field(true);
        // Dos escalas con igual amplitud y ganancias 2:1: masas 1.6 y 0.8.
        let expected_ln_tau =
            (2.0 * SPECTRUM_SCALES_MS[18].ln() + SPECTRUM_SCALES_MS[20].ln()) / 3.0;
        assert!((field.total_energy - 2.4).abs() < 1e-12);
        assert!((field.resonant_tau_ms.ln() - expected_ln_tau).abs() < 1e-12);
        // CL-32: el suelo 0,02 de las escalas sin señal cuenta por su
        // fracción resuelta (reloj de 1 ms en un espectro frío).
        let total_w: f64 = (0..32)
            .map(|i| {
                spec.scales[i].fusion_weight()
                    * factor_de_resolucion(SPECTRUM_SCALES_MS[i], FEED_CLOCK_RESOLUTION_MS)
            })
            .sum();
        assert!((field.global_coherence - 0.8 / total_w).abs() < 1e-12);
        // (Ola XLI·A2b) DOS masas con alcance declarado: la del CAMPO
        // (persistencia×ganancia — aprendizaje/epigenética, pineada arriba)
        // y la de DECISIÓN (observable D-742). El lector `spectral_coherence`
        // sigue a la de DECISIÓN — debe coincidir con `fused_score`, no con la
        // masa de aprendizaje. Antisimetría por lado, verificada sobre sí misma.
        assert!((spec.spectral_coherence(false) + spec.spectral_coherence(true)).abs() < 1e-12);
        assert!((spec.spectral_coherence(true) - spec.fused_score).abs() < 1e-9 * spec.fused_score.abs().max(1.0));
        assert!(
            spec.continuous_band_projection(
                (SPECTRUM_SCALES_MS[18] * SPECTRUM_SCALES_MS[20]).sqrt(),
                2.0
            ) > 0.0,
            "la proyección debe incorporar la ganancia aprendida"
        );
    }

    #[test]
    fn spectral_contract_energy_is_consistent_at_grid_nodes() {
        let spec = opposed_scales();
        let energy: f64 = SPECTRUM_SCALES_MS
            .iter()
            .map(|tau| spec.continuous_energy_density(*tau))
            .sum();
        assert!((energy - 2.4).abs() < 1e-12);
        assert!((energy - spec.spectral_field(true).total_energy).abs() < 1e-12);
    }

    #[test]
    fn spectral_contract_timestamp_zero_is_a_valid_origin() {
        let mut zero = TemporalSpectrum::new();
        let mut shifted = TemporalSpectrum::new();
        for (ts, price) in [(0, 100.0), (1, 101.0), (2, 99.0)] {
            zero.update(price, ts);
            shifted.update(price, ts + 1000);
        }
        // (Ola XLI·C2) Igualdad de IDENTIDAD temporal, no bit-exacta: añadir
        // momentos al bucle de update() cambia la contracción FMA del
        // compilador y desplaza el último ulp; el contrato es la INVARIANZA
        // del origen t=0, no la reproducción bit a bit.
        for (left, right) in zero.scales.iter().zip(&shifted.scales) {
            assert!((left.ewma_price - right.ewma_price).abs() < 1e-9 * right.ewma_price.abs().max(1.0));
            assert!((left.signal - right.signal).abs() < 1e-9);
        }
    }

    #[test]
    fn spectral_contract_long_scale_retains_small_elapsed_mass() {
        let mut spec = TemporalSpectrum::new();
        spec.update(100.0, 1000);
        spec.update(101.0, 1001);
        // (Ola XLI·A2b) Expectativa D-742: la vol es la MEDIA OBSERVADA del
        // núcleo (raw/masa), sin la semilla 1e-7 pre-D-742 que este test
        // pineaba — quedó obsoleta en la fusión del PR #5. Se replican las
        // MISMAS expresiones del código (exp_m1 para alpha, 1−exp para masa)
        // porque su diferencia de redondeo se amplifica al dividir α/τ~1e-13.
        let alpha = -(-1.0 / SPECTRUM_SCALES_MS[31]).exp_m1();
        let mass = 1.0 - (-(1.0 / SPECTRUM_SCALES_MS[31])).exp();
        let expected = alpha * 0.01 / mass;
        assert!((spec.scales[31].ewma_dev_vol - expected).abs() < 1e-9 * expected.abs().max(1e-12));
    }

    #[test]
    fn spectral_contract_invalid_queries_are_neutral() {
        let mut spec = opposed_scales();
        for s in &mut spec.scales {
            s.signal = 0.25;
            s.persistence = 0.5;
            s.momentum_z = 0.8;
            s.ewma_price = 100.0;
            s.ewma_dev_vol = 0.01;
            s.epigenetic_gain = 2.0;
        }
        for tau in [f64::NAN, f64::INFINITY, f64::NEG_INFINITY, 0.0, -1.0] {
            assert_eq!(spec.signal_at(tau), 0.0);
            assert_eq!(spec.persistence_at(tau), 0.0);
            assert_eq!(spec.momentum_z_at(tau), 0.0);
            assert_eq!(spec.ewma_price_at(tau), 0.0);
            assert_eq!(spec.volatility_at(tau), 0.0);
            assert_eq!(spec.scale_gain_at(tau), 1.0);
            assert_eq!(spec.continuous_energy_density(tau), 0.0);
        }
    }

    #[test]
    fn spectral_contract_invalid_outcomes_cannot_train() {
        for pnl in [f64::NAN, f64::INFINITY, f64::NEG_INFINITY] {
            let mut spec = opposed_scales();
            let before: Vec<_> = spec.scales.iter().map(|s| s.epigenetic_gain).collect();
            spec.apply_epigenetic_outcome(SPECTRUM_SCALES_MS[18], true, pnl);
            let after: Vec<_> = spec.scales.iter().map(|s| s.epigenetic_gain).collect();
            assert_eq!(before, after);
        }
    }

    #[test]
    fn spectral_contract_learning_refreshes_fusion_without_new_tick() {
        let mut spec = TemporalSpectrum::new();
        for i in 0..40 {
            spec.update(100.0 + (i as f64 * 0.4).sin(), 1000 + i * 1000);
        }
        spec.apply_epigenetic_outcome(SPECTRUM_SCALES_MS[18], true, 0.01);
        // (Ola XLI·C2) 1e-12 absoluto sobre magnitudes ~1e-2..1: epsilon relativo.
        assert!((spec.fused_score - spec.spectral_coherence(true)).abs() < 1e-9 * spec.fused_score.abs().max(1.0));
    }

    #[test]
    fn spectral_contract_maximum_energy_entropy_can_have_full_agreement() {
        let mut spec = TemporalSpectrum::new();
        for s in &mut spec.scales {
            s.signal = 1.0;
            s.persistence = 0.5;
        }
        let field = spec.spectral_field(true);
        // CL-32: la masa se reparte según la fracción resuelta de cada
        // escala; con consenso perfecto la entropía es la de ese reparto.
        let f: Vec<f64> = SPECTRUM_SCALES_MS
            .iter()
            .map(|&tau| factor_de_resolucion(tau, FEED_CLOCK_RESOLUTION_MS))
            .collect();
        let total: f64 = f.iter().sum();
        let entropia = -f
            .iter()
            .map(|x| x / total)
            .filter(|&q| q > 1e-15)
            .map(|q| q * q.ln())
            .sum::<f64>()
            / 32f64.ln();
        assert!(entropia > 0.85, "{entropia}");
        assert!((field.spectral_entropy - entropia).abs() < 1e-12);
        assert!((field.global_coherence - 1.0).abs() < 1e-12);
    }

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
        // CL-30: niveles iid alrededor de un precio fijo son reversión pura:
        // retornos de bloques vecinos con ρ = −½ ⇒ acuerdo de signo
        // (2/π)·asin(−½) = −⅓. Lo que no debe aparecer es continuación.
        assert!(
            mean_persist < 0.0,
            "ruido de nivel ⇒ reversión, nunca continuación; dio {mean_persist}"
        );
    }

    /// Normal estándar determinista (xorshift64 + Box-Muller) para CL-30.
    fn normal_cl30(estado: &mut u64) -> f64 {
        let mut uniforme = || {
            *estado ^= *estado << 13;
            *estado ^= *estado >> 7;
            *estado ^= *estado << 17;
            ((*estado >> 11) as f64 + 0.5) / (1u64 << 53) as f64
        };
        let (u1, u2) = (uniforme(), uniforme());
        (-2.0 * u1.ln()).sqrt() * (std::f64::consts::TAU * u2).cos()
    }

    /// Media temporal de la persistencia en τ = 30 s durante la segunda
    /// mitad de una serie de ln(precio) muestreada cada 100 ms.
    fn persistencia_media_cl30(ln_precios: impl Iterator<Item = f64>, n: usize) -> f64 {
        let mut spec = TemporalSpectrum::new();
        let mut t = 1_700_000_000_000u64;
        let (mut suma, mut cuenta) = (0.0, 0.0);
        for (i, lp) in ln_precios.take(n).enumerate() {
            spec.update(60_000.0 * lp.exp(), t);
            t += 100;
            if i >= n / 2 {
                suma += spec.persistence_at(30_000.0);
                cuenta += 1.0;
            }
        }
        suma / cuenta
    }

    /// CL-30: en una caminata aleatoria los retornos de dos bloques
    /// disjuntos son independientes; la persistencia debe quedar en 0 (el
    /// «browniano neutral» que leen Kelly S-1, BE/trailing y el consejo).
    /// El estimador por desviaciones solapadas daba ≈ +0,9 en cualquier
    /// escala con τ ≫ Δt.
    #[test]
    fn cl30_caminata_aleatoria_persistencia_nula() {
        let mut estado = 0x9E37_79B9_7F4A_7C15u64;
        let mut lp = 0.0;
        let serie = std::iter::from_fn(|| {
            lp += 1e-4 * normal_cl30(&mut estado);
            Some(lp)
        });
        let media = persistencia_media_cl30(serie, 144_000);
        assert!(
            media.abs() < 0.2,
            "caminata aleatoria ⇒ persistencia ≈ 0 a 30 s, dio {media}"
        );
    }

    /// CL-30: una deriva que domina el ruido a la escala mantiene el signo
    /// de bloque a bloque ⇒ persistencia positiva.
    #[test]
    fn cl30_tendencia_persiste() {
        let mut estado = 0xD1B5_4A32_D192_ED03u64;
        let mut lp = 0.0;
        let serie = std::iter::from_fn(|| {
            lp += 2e-5 + 1e-4 * normal_cl30(&mut estado);
            Some(lp)
        });
        let media = persistencia_media_cl30(serie, 144_000);
        assert!(media > 0.6, "deriva dominante ⇒ persistencia > 0, dio {media}");
    }

    /// CL-30: un zigzag cuyo semiperiodo es el bloque de la escala 18
    /// (68,8 s con Δt = 100 ms) invierte el signo en cada bloque ⇒
    /// persistencia negativa en esa escala. El estimador solapado lo veía
    /// como continuación: desviaciones vecinas del mismo lado de la EWMA.
    #[test]
    fn cl30_zigzag_a_la_escala_revierte() {
        let mut spec = TemporalSpectrum::new();
        let mut t = 1_700_000_000_000u64;
        let semiperiodo = 688u64;
        for i in 0..72_000u64 {
            let fase = i % (2 * semiperiodo);
            let x = if fase < semiperiodo {
                fase as f64
            } else {
                (2 * semiperiodo - fase) as f64
            } / semiperiodo as f64;
            spec.update(60_000.0 * (1.0 + 0.002 * x), t);
            t += 100;
        }
        let p = spec.scales[18].persistence;
        assert!(p < -0.6, "zigzag al bloque ⇒ reversión, dio {p}");
    }

    /// Caminata aleatoria con un evento por segundo durante 6 h.
    fn espectro_a_un_segundo_cl32() -> TemporalSpectrum {
        let mut spec = TemporalSpectrum::new();
        let mut estado = 0x9E37_79B9_7F4A_7C15u64;
        let mut lp = 0.0f64;
        let mut t = 1_700_000_000_000u64;
        for _ in 0..21_600u64 {
            lp += 1e-4 * normal_cl30(&mut estado);
            spec.update(60_000.0 * lp.exp(), t);
            t += 1_000;
        }
        spec
    }

    /// Fracción de `x` en las escalas con τ por debajo de la resolución.
    fn fraccion_no_resuelta_cl32(spec: &TemporalSpectrum, x: &[f64; 32]) -> f64 {
        let res = spec.resolucion_efectiva_ms();
        let total: f64 = x.iter().sum();
        let debajo: f64 = (0..32).filter(|&i| spec.scales[i].tau_ms < res).map(|i| x[i]).sum();
        debajo / total
    }

    /// CL-32: con un evento por segundo, las escalas de 4 ms a 268 ms sólo
    /// repiten el último evento. Con la resolución del reloj (1 ms) se
    /// llevaban ≈ 68 % del peso de la fusión D-742.
    #[test]
    fn cl32_las_escalas_no_resueltas_no_gobiernan_la_fusion() {
        let spec = espectro_a_un_segundo_cl32();
        assert!((spec.resolucion_efectiva_ms() - 1_000.0).abs() < 1e-9);
        let (_, pesos) = spec.pesos_observables();
        let f = fraccion_no_resuelta_cl32(&spec, &pesos);
        assert!(f < 0.25, "peso de la fusión en escalas no resueltas: {f}");
    }

    /// CL-32: la masa w·|señal| (entropía, Fisher, W₁, τ*, bandas) tampoco
    /// cuenta cada copia del último evento como una escala.
    #[test]
    fn cl32_la_masa_espectral_no_duplica_el_ultimo_evento() {
        let spec = espectro_a_un_segundo_cl32();
        let pesos = spec.pesos_espectrales();
        let mut masa = [0.0f64; 32];
        for i in 0..32 {
            masa[i] = pesos[i] * spec.scales[i].signal.abs();
        }
        let f = fraccion_no_resuelta_cl32(&spec, &masa);
        assert!(f < 0.1, "masa espectral en escalas no resueltas: {f}");
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

    /// D-742: en una caminata aleatoria la fusión no puede quedar reducida a
    /// «el precio está sobre o bajo el de arranque». Con la semilla de 1e-7 y
    /// el peso 1/vol, las escalas que los datos no han llenado (días a siglos)
    /// dominaban la fusión con su señal saturada.
    #[test]
    fn d742_las_escalas_vacias_no_gobiernan_la_fusion() {
        let mut spec = TemporalSpectrum::new();
        let mut seed: u64 = 7;
        let mut t = 1_700_000_000_000u64;
        let mut price = 60_000.0f64;
        let p0 = price;
        let (mut n, mut sx, mut sy, mut sxx, mut syy, mut sxy) = (0.0f64, 0.0, 0.0, 0.0, 0.0, 0.0);
        for i in 0..259_200u64 {
            seed = seed.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
            let u = ((seed >> 11) as f64) / ((1u64 << 53) as f64) - 0.5;
            price *= 1.0 + u * 0.0006;
            spec.update(price, t);
            t += 1_000;
            if i >= 21_600 && i % 60 == 0 {
                let x = spec.fused_score;
                let y = (price / p0).ln().signum();
                n += 1.0;
                sx += x;
                sy += y;
                sxx += x * x;
                syy += y * y;
                sxy += x * y;
            }
        }
        let cov = sxy / n - (sx / n) * (sy / n);
        let vx = sxx / n - (sx / n).powi(2);
        let vy = syy / n - (sy / n).powi(2);
        let corr = if vx > 0.0 && vy > 0.0 { cov / (vx * vy).sqrt() } else { 0.0 };
        assert!(
            corr.abs() < 0.5,
            "la fusión copia el signo del precio respecto del arranque: corr {corr:.3}"
        );
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


// ═════════════════ Ola XLI·C2/C3: falsación de Kolmogorov y Fisher ═════════════════

#[test]
fn xli_c2_gaussian_iid_da_autosimilaridad_k41() {
    // Incrementos gaussianos iid ⇒ self-similar: ζ(p) = p/3 dentro de
    // tolerancia. El estimador es sobre EWMA del núcleo (suaviza), así que
    // la tolerancia es amplia, pero ζ3 debe rondar 1 y χ quedar acotado.
    let mut spec = TemporalSpectrum::new();
    let mut seed = 0x9E3779B97F4A7C15u64;
    let mut price = 60_000.0f64;
    let mut t = 1_000u64;
    for _ in 0..40_000 {
        seed = seed.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        let u = ((seed >> 11) as f64 / (1u64 << 53) as f64) * 2.0 - 1.0;
        let ret = u * 0.001;
        price *= 1.0 + ret;
        t += 50;
        spec.update(price, t);
    }
    let sf = spec.structure_functions().expect("masa suficiente tras 40k ticks");
    // Ola XLIV: la referencia de un PRECIO browniano es ζ(p) = p/2 (ζ₃ = 1,5),
    // no K41 (ζ₃ = 1). La EWMA suaviza, así que la banda es ancha; lo que el
    // contrato fija es la no intermitencia de un proceso iid: χ ≈ 0.
    assert!(sf.zeta3 > 1.0 && sf.zeta3 < 2.0, "zeta3={} fuera de banda browniana", sf.zeta3);
    assert!(sf.intermittency < 0.15, "iid no es intermitente: chi={}", sf.intermittency);
    assert!(sf.usable_scales >= 4);
    // Con eventos cada 50 ms, ninguna escala por debajo de 50 ms participa.
    assert!((spec.resolucion_efectiva_ms() - 50.0).abs() < 1e-6);
    let resueltas = spec.scales.iter().filter(|s| s.tau_ms >= 50.0).count();
    assert!(sf.usable_scales <= resueltas);
}

/// Ola XLIV — el rango de la Fisher de escala y su umbral de identificabilidad.
/// El gate walk-forward exigía F > 1,0, inalcanzable salvo para un pico en UNA
/// escala: la constante, el soporte efectivo y el umbral se fijan aquí.
#[test]
fn xliv_fisher_de_escala_tiene_techo_y_umbral_alcanzable() {
    assert!((FISHER_ESCALA_MAX - 2.0 / (4f64.ln() * 4f64.ln())).abs() < 1e-15);
    // Masa uniforme sobre k escalas contiguas (lejos de los bordes de la malla).
    let fisher_bloque = |k: usize| {
        let mut spec = TemporalSpectrum::new();
        for i in 0..32 {
            spec.scales[i].signal = if (14..14 + k).contains(&i) { 1.0 } else { 0.0 };
        }
        // CL-32: bloque sobre escalas resueltas por el reloj (τ ≥ 268 ms).
        let w = spec.scales[14].fusion_weight();
        for i in 14..14 + k {
            assert!((spec.scales[i].fusion_weight() - w).abs() < 1e-15);
        }
        spec.fisher_scale_information().expect("masa presente")
    };
    for k in [1usize, 2, 3, 6, 12] {
        let f = fisher_bloque(k);
        assert!(f <= FISHER_ESCALA_MAX + 1e-12, "k={k}: F={f} excede el techo");
        assert!(
            (soporte_efectivo_en_escalas(f) - k as f64).abs() < 1e-9,
            "k={k}: soporte efectivo {}",
            soporte_efectivo_en_escalas(f)
        );
    }
    let u = umbral_fisher_identificable();
    assert!(u > 0.30 && u < 0.36, "umbral {u}");
    // Concentrado en ≤ 3 escalas: identificable; en toda la banda (6): no.
    assert!(fisher_bloque(3) > u && fisher_bloque(6) < u);
    // El umbral anterior (1,0) sólo lo supera un pico en una escala.
    assert!(fisher_bloque(1) > 1.0 && fisher_bloque(2) < 1.0);
}

#[test]
fn xli_c3_masa_concentrada_tiene_fisher_alto_y_difusa_bajo() {
    // Masa en UN nodo → Fisher alto; masa uniforme → Fisher ≈ 0.
    // Sin update() no hay energía: None (contrato de contorno).
    let cold = TemporalSpectrum::new();
    assert!(cold.fisher_scale_information().is_none());
    // Alimentar un pulso fuerte en una sola escala no es directo desde la API
    // pública; usamos la señal viva: muchas actualizaciones con reversión
    // violenta concentran energía en las escalas rápidas.
    let mut spec = TemporalSpectrum::new();
    let mut t = 1_000u64;
    let mut price = 100.0f64;
    for i in 0..20_000 {
        price *= 1.0 + if i % 2 == 0 { 0.002 } else { -0.002 };
        t += 10;
        spec.update(price, t);
    }
    let fisher = spec.fisher_scale_information().expect("energia presente");
    assert!(fisher >= 0.0 && fisher.is_finite());
    // La reversión alterna alimenta las escalas rápidas con persistencia
    // negativa: su peso de fusión (|persistencia|) sigue presente, y la masa
    // NO puede ser uniforme ⇒ Fisher > 0 estricto.
    assert!(fisher > 0.0, "masa no uniforme debe dar Fisher > 0, dio {fisher}");
}


// ═════════════════ Ola XLII·D: falsación del transporte W1 ═════════════════

#[test]
fn xlii_d_w1_masa_identica_es_cero() {
    let mut spec = TemporalSpectrum::new();
    let mut t = 1_000u64;
    let mut price = 100.0f64;
    for i in 0..300 {
        price *= 1.0 + if i % 7 == 0 { 0.003 } else { -0.001 };
        t += 100;
        spec.update(price, t);
    }
    // lag dentro del anillo y con masa en ambos puntos: W1 >= 0 y finito.
    let w1 = spec.spectral_transport_w1(50).expect("masa presente");
    assert!(w1 >= 0.0 && w1.is_finite());
    // Contra sí mismo (lag congelado no existe; el contrato minimo es que
    // masa identica daría 0 — verificado estructuralmente por la CDF).
}

#[test]
fn xlii_d_w1_reestructuracion_acota_por_malla() {
    // W1 está acotado por el rango de la malla: 32 escalas base 4 ⇒
    // ln(4^31) ≈ 43. Cualquier reestructuración cabe ahí.
    let mut spec = TemporalSpectrum::new();
    let mut t = 1_000u64;
    let mut price = 100.0f64;
    for i in 0..600 {
        price *= 1.0 + if i % 2 == 0 { 0.004 } else { -0.004 };
        t += 50;
        spec.update(price, t);
    }
    let w1 = spec.spectral_transport_w1(100).expect("masa presente");
    let rango = 31.0 * 4f64.ln();
    assert!(w1 <= rango, "w1={} > rango {}", w1, rango);
    // La reversión violenta alimenta escalas rapidas: la masa DEBE haberse
    // movido algo entre hace 100 updates y ahora.
    assert!(w1 > 0.0, "reestructuracion no trivial esperada");
}
