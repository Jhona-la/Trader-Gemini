//! # Auditor Interno (El Anti-Sistema) — Structural Doubt & Dual Verification Engine
//!
//! An unbiased agent that does not predict or trade. Its sole purpose is to audit,
//! demolish false self-affirmation, compare internal claims against exchange fills,
//! detect cognitive biases, and trigger Epistemic Crisis if self-deception is caught.

use serde::{Deserialize, Serialize};

/// Raw execution fill recorded independently from exchange WS/REST
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ExchangeFillRecord {
    pub timestamp_ns: u64,
    pub symbol_id: u16,
    pub is_buyer: bool,
    pub price: f64,
    pub qty: f64,
    pub fee_usd: f64,
    pub realized_pnl: f64,
    pub is_closed_position: bool,
}

/// Independent performance metrics calculated by the Auditor
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct IndependentMetrics {
    pub calculated_sharpe: f64,
    pub calculated_max_drawdown: f64,
    pub calculated_win_rate: f64,
    pub calculated_latency_us: u32,
    pub benchmark_alpha: f64,
}

/// OLA Ω120: Constantes canónicas por defecto para el Tribunal de Autoengaño y Auditor Interno
pub const DEFAULT_AUTOENGANO_THRESHOLD: f64 = 0.20;
pub const DEFAULT_CULPABLE_PELIGROSO_THRESHOLD: f64 = 0.10;
pub const DEFAULT_CULPABLE_RECUPERABLE_THRESHOLD: f64 = 0.05;
pub const DEFAULT_OVERFITTING_RECENT_MIN: f64 = 0.05;
pub const DEFAULT_OVERFITTING_OOS_MAX: f64 = -0.01;
pub const DEFAULT_MAX_FEATURES_NOMINAL: usize = 64;

#[inline]
pub const fn default_autoengano_threshold() -> f64 {
    DEFAULT_AUTOENGANO_THRESHOLD
}
#[inline]
pub const fn default_culpable_peligroso_threshold() -> f64 {
    DEFAULT_CULPABLE_PELIGROSO_THRESHOLD
}
#[inline]
pub const fn default_culpable_recuperable_threshold() -> f64 {
    DEFAULT_CULPABLE_RECUPERABLE_THRESHOLD
}
#[inline]
pub const fn default_overfitting_recent_min() -> f64 {
    DEFAULT_OVERFITTING_RECENT_MIN
}
#[inline]
pub const fn default_overfitting_oos_max() -> f64 {
    DEFAULT_OVERFITTING_OOS_MAX
}
#[inline]
pub const fn default_max_features_nominal() -> usize {
    DEFAULT_MAX_FEATURES_NOMINAL
}

/// OLA Ω120: Parámetros desacoplados y configurables para el Tribunal de Autoengaño y Detección de Sesgos Cognitivos
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct TribunalThresholds {
    #[serde(default = "default_autoengano_threshold")]
    pub autoengano_threshold: f64,
    #[serde(default = "default_culpable_peligroso_threshold")]
    pub culpable_peligroso_threshold: f64,
    #[serde(default = "default_culpable_recuperable_threshold")]
    pub culpable_recuperable_threshold: f64,
    #[serde(default = "default_overfitting_recent_min")]
    pub overfitting_recent_min: f64,
    #[serde(default = "default_overfitting_oos_max")]
    pub overfitting_oos_max: f64,
    #[serde(default = "default_max_features_nominal")]
    pub max_features_nominal: usize,
}

impl Default for TribunalThresholds {
    fn default() -> Self {
        Self {
            autoengano_threshold: DEFAULT_AUTOENGANO_THRESHOLD,
            culpable_peligroso_threshold: DEFAULT_CULPABLE_PELIGROSO_THRESHOLD,
            culpable_recuperable_threshold: DEFAULT_CULPABLE_RECUPERABLE_THRESHOLD,
            overfitting_recent_min: DEFAULT_OVERFITTING_RECENT_MIN,
            overfitting_oos_max: DEFAULT_OVERFITTING_OOS_MAX,
            max_features_nominal: DEFAULT_MAX_FEATURES_NOMINAL,
        }
    }
}

impl TribunalThresholds {
    pub fn validate(&self) -> Result<(), String> {
        if !self.autoengano_threshold.is_finite() || !(0.05..=1.0).contains(&self.autoengano_threshold) {
            return Err(format!("Invalid autoengano_threshold: {}", self.autoengano_threshold));
        }
        if !self.culpable_peligroso_threshold.is_finite() || !(0.01..=0.50).contains(&self.culpable_peligroso_threshold) {
            return Err(format!("Invalid culpable_peligroso_threshold: {}", self.culpable_peligroso_threshold));
        }
        if !self.culpable_recuperable_threshold.is_finite() || !(0.001..=0.25).contains(&self.culpable_recuperable_threshold) {
            return Err(format!("Invalid culpable_recuperable_threshold: {}", self.culpable_recuperable_threshold));
        }
        if self.culpable_recuperable_threshold >= self.culpable_peligroso_threshold {
            return Err(format!(
                "culpable_recuperable_threshold ({}) must be < culpable_peligroso_threshold ({})",
                self.culpable_recuperable_threshold, self.culpable_peligroso_threshold
            ));
        }
        if self.culpable_peligroso_threshold >= self.autoengano_threshold {
            return Err(format!(
                "culpable_peligroso_threshold ({}) must be < autoengano_threshold ({})",
                self.culpable_peligroso_threshold, self.autoengano_threshold
            ));
        }
        if !self.overfitting_recent_min.is_finite() || !(0.0..=1.0).contains(&self.overfitting_recent_min) {
            return Err(format!("Invalid overfitting_recent_min: {}", self.overfitting_recent_min));
        }
        if !self.overfitting_oos_max.is_finite() || !(-1.0..=0.0).contains(&self.overfitting_oos_max) {
            return Err(format!("Invalid overfitting_oos_max: {}", self.overfitting_oos_max));
        }
        if !(1..=256).contains(&self.max_features_nominal) {
            return Err(format!("Invalid max_features_nominal: {}", self.max_features_nominal));
        }
        Ok(())
    }
}

/// OLA Ω120: Función pura analítica que evalúa el veredicto del Tribunal a partir de la máxima discrepancia observada
#[inline(always)]
pub fn evaluate_tribunal_verdict(max_disc: f64, thresholds: &TribunalThresholds) -> TribunalVerdict {
    let disc = if max_disc.is_finite() && max_disc >= 0.0 {
        max_disc
    } else {
        1.0 // Fail-closed ante flotante no finito
    };

    let auto = if thresholds.autoengano_threshold.is_finite() && thresholds.autoengano_threshold > 0.0 {
        thresholds.autoengano_threshold
    } else {
        DEFAULT_AUTOENGANO_THRESHOLD
    };

    let peligroso = if thresholds.culpable_peligroso_threshold.is_finite() && thresholds.culpable_peligroso_threshold > 0.0 {
        thresholds.culpable_peligroso_threshold
    } else {
        DEFAULT_CULPABLE_PELIGROSO_THRESHOLD
    };

    let recuperable = if thresholds.culpable_recuperable_threshold.is_finite() && thresholds.culpable_recuperable_threshold > 0.0 {
        thresholds.culpable_recuperable_threshold
    } else {
        DEFAULT_CULPABLE_RECUPERABLE_THRESHOLD
    };

    if disc > auto {
        TribunalVerdict::AutoengañoDetectado
    } else if disc > peligroso {
        TribunalVerdict::CulpablePeligroso
    } else if disc > recuperable {
        TribunalVerdict::CulpableRecuperable
    } else {
        TribunalVerdict::Inocente
    }
}

/// OLA Ω120: Calcula la discrepancia máxima normalizada entre métricas internas y realidad de ejecución
#[inline(always)]
pub fn calculate_max_discrepancy(
    internal_sharpe: f64,
    internal_win_rate: f64,
    internal_drawdown: f64,
    independent: &IndependentMetrics,
) -> (f64, f64, f64, f64) {
    let safe_sharpe = if internal_sharpe.is_finite() {
        internal_sharpe
    } else {
        0.0
    };
    let safe_wr = if internal_win_rate.is_finite() {
        internal_win_rate
    } else {
        0.0
    };
    let safe_dd = if internal_drawdown.is_finite() {
        internal_drawdown
    } else {
        0.0
    };

    let sharpe_disc = if safe_sharpe > 0.0 {
        (safe_sharpe - independent.calculated_sharpe).abs() / safe_sharpe
    } else {
        0.0
    };

    let wr_disc = (safe_wr - independent.calculated_win_rate).abs();
    // F1-006 (Ola Ω150): Solo penalizar subestimación de drawdown (cuando el sistema
    // afirma tener MENOR drawdown que la realidad de los fills). Reportar un drawdown
    // interno más conservador (mayor) no constituye autoengaño.
    let dd_disc = (independent.calculated_max_drawdown - safe_dd).max(0.0);

    let max_disc = sharpe_disc.max(wr_disc).max(dd_disc);
    (max_disc, sharpe_disc, wr_disc, dd_disc)
}

/// Daily Tribunal Verdict
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum TribunalVerdict {
    Inocente,
    CulpableRecuperable,
    CulpablePeligroso,
    AutoengañoDetectado,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DailyTribunalReport {
    pub verdict: TribunalVerdict,
    pub pnl_discrepancy_pct: f64,
    pub sharpe_discrepancy_pct: f64,
    pub justification: String,
}

/// F1-006 (Ola Ω150): Capacidad máxima acotada del diario de fills para evitar
/// crecimiento ilimitado de RAM y degradación cuadrática en sesiones largas.
pub const MAX_FILLS_JOURNAL: usize = 4096;

/// F412/F413/F271 (censo C-F7b, 2026-10-18): capacidad máxima acotada del diario
/// del canal INDEPENDIENTE (fills del host/exchange). Misma higiene de RAM que
/// MAX_FILLS_JOURNAL: buffer circular acotado, sin crecimiento ilimitado.
pub const MAX_FILLS_INDEPENDIENTES: usize = 4096;

/// SOL (2026-10-19, censo C-F7b §3-2): evidencia mínima para que una sombra
/// del auditor se publique. Con menos fills la sombra es `None`: publicar
/// ruido sería peor que no publicar — la ausencia de dato nunca es señal.
pub const MIN_FILLS_SOMBRA: usize = 20;

/// SOL (2026-10-19): bloques contiguos para el error estándar de las sombras.
/// Los fills son una serie correlacionada (mismo régimen, misma sesión): la
/// media por bloque y el desvío ENTRE bloques / sqrt(B) miden la precisión
/// real sin asumir independencia que no existe.
const BLOQUES_SOMBRA: usize = 4;

/// SOL (2026-10-19, §3-2): sombra de la pata de win rate (F413) con intervalo
/// y estabilidad. Sólo telemetría: ninguna decisión la consume hoy.
#[derive(Debug, Clone, Copy)]
pub struct SombraWr {
    /// WR que afirma el núcleo (saneado a [0,1]).
    pub wr_interno: f64,
    /// WR reconstruido de los cierres del canal independiente (neto > 0).
    pub wr_real: f64,
    /// |interno − real|.
    pub disc: f64,
    /// Error estándar del WR real por bloques.
    pub ee_real: f64,
    /// Cierres usados (>= MIN_FILLS_SOMBRA).
    pub n: usize,
    /// |interno − WR real| en la primera mitad temporal (estabilidad OOS).
    pub disc_mitad_a: f64,
    /// |interno − WR real| en la segunda mitad temporal.
    pub disc_mitad_b: f64,
}

/// SOL (2026-10-19, §3-2): sombra de fricción real menos modelada (bp).
/// Positiva = se paga más de lo modelado. Sólo telemetría.
#[derive(Debug, Clone, Copy)]
pub struct SombraFriccion {
    /// Media de (fee real en bp − fricción modelada en bp) por fill.
    pub media_bp: f64,
    /// Error estándar de esa media por bloques.
    pub ee_bp: f64,
    /// Fills usados (>= MIN_FILLS_SOMBRA).
    pub n: usize,
    /// Media en la primera mitad temporal (estabilidad OOS).
    pub media_mitad_a_bp: f64,
    /// Media en la segunda mitad temporal.
    pub media_mitad_b_bp: f64,
}

/// SOL (2026-10-19): error estándar de una proporción por bloques contiguos +
/// proporción de cada mitad. `(ee, prop_mitad_a, prop_mitad_b)`.
/// Con menos elementos que bloques, `ee = 0` (no hay varianza medible: se
/// dice 0, nunca NaN).
fn ee_proporcion_bloques<T, F: Fn(&&T) -> bool>(items: &[&T], pred: F) -> (f64, f64, f64) {
    let n = items.len();
    if n == 0 {
        return (0.0, 0.0, 0.0);
    }
    let proporcion = |ini: usize, fin: usize| -> f64 {
        let total = fin - ini;
        if total == 0 {
            return 0.0;
        }
        items[ini..fin].iter().filter(|x| pred(x)).count() as f64 / total as f64
    };
    let (ee, _) = bloques_media_ee(items, |ini, fin| proporcion(ini, fin));
    let mitad = n / 2;
    (ee, proporcion(0, mitad), proporcion(mitad, n))
}

/// SOL (2026-10-19): `(media_global, ee_por_bloques, media_mitad_a,
/// media_mitad_b)` de una serie ya saneada (todos finitos).
fn media_ee_bloques(vals: &[f64]) -> (f64, f64, f64, f64) {
    let n = vals.len();
    if n == 0 {
        return (0.0, 0.0, 0.0, 0.0);
    }
    let media = vals.iter().sum::<f64>() / n as f64;
    let (ee, _) = bloques_media_ee(vals, |ini, fin| {
        vals[ini..fin].iter().sum::<f64>() / (fin - ini) as f64
    });
    let mitad = n / 2;
    let a = vals[..mitad].iter().sum::<f64>() / mitad.max(1) as f64;
    let b = vals[mitad..].iter().sum::<f64>() / (n - mitad).max(1) as f64;
    (media, ee, a, b)
}

/// SOL (2026-10-19): núcleo común de los errores estándar por bloques.
/// `media_bloque` devuelve la media del bloque [ini, fin). Devuelve
/// `(ee, nº de bloques)`; con < BLOQUES_SOMBRA elementos por bloque, ee = 0.
/// sd muestral entre bloques / sqrt(B).
fn bloques_media_ee<T, F: Fn(usize, usize) -> f64>(items: &[T], media_bloque: F) -> (f64, usize) {
    let n = items.len();
    let b = BLOQUES_SOMBRA;
    if n < b {
        return (0.0, 0);
    }
    let tam = n / b;
    if tam == 0 {
        return (0.0, 0);
    }
    let mut sum = 0.0;
    let mut sum2 = 0.0;
    for i in 0..b {
        let ini = i * tam;
        let fin = if i == b - 1 { n } else { ini + tam };
        let m = media_bloque(ini, fin);
        sum += m;
        sum2 += m * m;
    }
    let media = sum / b as f64;
    let var = ((sum2 - b as f64 * media * media) / (b - 1) as f64).max(0.0);
    ((var / b as f64).sqrt(), b)
}

/// Dual Result Verifier
pub struct VerificadorResultados;

impl VerificadorResultados {
    /// Computes independent metrics from raw exchange fill records anchored to explicit base capital.
    /// F1-006 (Ola Ω150): Single-pass zero-allocation Welford online variance + equity curve reconstruction.
    /// CL-15 (GLM 144): capital inválido (NaN/≤0) ⇒ evidencia NO anclable ⇒
    /// métricas vacías (igual que fills vacíos): el tribunal queda armado,
    /// consistente con F412 («sin fuente independiente el veto no decae»).
    /// Inventar 13.0 fabricaría retornos/drawdowns de una cuenta inexistente.
    pub fn compute_metrics_with_capital(
        fills: &[ExchangeFillRecord],
        base_capital: f64,
    ) -> IndependentMetrics {
        if fills.is_empty()
            || !base_capital.is_finite()
            || base_capital <= 0.0
        {
            return IndependentMetrics::default();
        }
        let safe_capital = base_capital;

        let mut count: u64 = 0;
        let mut winning_trades: u64 = 0;
        let mut mean_ret: f64 = 0.0;
        let mut m2: f64 = 0.0;
        let mut first_ts: Option<u64> = None;
        let mut last_ts: u64 = 0;
        let mut equity = safe_capital;
        let mut peak = safe_capital;
        let mut max_dd = 0.0;
        let mut gross_pnl_sum = 0.0;
        let mut total_fees_sum = 0.0;

        for f in fills.iter().filter(|f| f.is_closed_position) {
            count += 1;
            let pnl = if f.realized_pnl.is_finite() {
                f.realized_pnl
            } else {
                0.0
            };
            let fee = if f.fee_usd.is_finite() {
                f.fee_usd
            } else {
                0.0
            };
            let net = pnl - fee;
            if net > 0.0 {
                winning_trades += 1;
            }
            if first_ts.is_none() {
                first_ts = Some(f.timestamp_ns);
            }
            last_ts = f.timestamp_ns;

            let ret = net / safe_capital;
            let delta = ret - mean_ret;
            mean_ret += delta / (count as f64);
            let delta2 = ret - mean_ret;
            m2 += delta * delta2;

            equity += net;
            if equity > peak {
                peak = equity;
            }
            let dd = if peak > 0.0 {
                (peak - equity) / peak
            } else {
                0.0
            };
            if dd > max_dd {
                max_dd = dd;
            }

            gross_pnl_sum += pnl;
            total_fees_sum += fee;
        }

        if count == 0 {
            return IndependentMetrics::default();
        }

        let total_trades = count as f64;
        let win_rate = (winning_trades as f64) / total_trades;
        let variance = if count > 1 {
            (m2 / ((count - 1) as f64)).max(0.0)
        } else {
            0.0
        };
        let std_dev = variance.sqrt();

        let time_span_days = if let Some(first) = first_ts {
            ((last_ts.saturating_sub(first)) as f64 / (1e9 * 86400.0)).max(1.0 / 24.0)
        } else {
            1.0
        };
        let trades_per_year = (total_trades / time_span_days) * 365.0;

        let sharpe = if std_dev > 1e-9 {
            (mean_ret / std_dev) * trades_per_year.sqrt().min(500.0)
        } else {
            0.0
        };

        let gross_pnl = gross_pnl_sum / safe_capital;
        let total_fees = total_fees_sum / safe_capital;
        let gross_alpha = gross_pnl / total_trades;
        let fee_drag = total_fees / total_trades;

        IndependentMetrics {
            calculated_sharpe: sharpe,
            calculated_max_drawdown: max_dd,
            calculated_win_rate: win_rate,
            calculated_latency_us: 150, // Standard baseline
            benchmark_alpha: gross_alpha - (fee_drag * 0.5),
        }
    }
}

/// Auditor Interno Engine
pub struct AuditorInterno {
    fills_journal: Vec<ExchangeFillRecord>,
    /// F412/F413/F271 (censo C-F7b §3, prioridad 1): canal INDEPENDIENTE.
    /// Sólo lo alimenta el host con fills del exchange (ORDER_TRADE_UPDATE del
    /// user-data stream), nunca el cierre interno del núcleo. Mientras esté
    /// vacío no existe fuente ajena contra la que verificar afirmaciones
    /// internas: las tres patas de `calculate_max_discrepancy` serían
    /// tautológicas (sharpe sin escritor de `internal_sharpe` ⇒ claim 0.0;
    /// wr del mismo cierre con el mismo predicado net>0; dd actual contra el
    /// máximo histórico del diario de cartera) y el veto dispararía solo,
    /// sin decaer (13 → 16,9 ⇒ disc 0,231 > 0,20 vetando todas las monedas).
    fills_independientes: Vec<ExchangeFillRecord>,
    base_capital: f64,
    pub thresholds: TribunalThresholds,
}

impl Default for AuditorInterno {
    fn default() -> Self {
        Self {
            fills_journal: Vec::new(),
            fills_independientes: Vec::new(),
            // CL-15 (GLM 144): el Default NO declara capital — inventar 13.0
            // anclaría las métricas a una cuenta inexistente. Sin with_capital
            // explícito la evidencia queda sin anclar (métricas vacías,
            // tribunal armado). El host vivo usa with_capital(capital_real).
            base_capital: f64::NAN,
            thresholds: TribunalThresholds::default(),
        }
    }
}

impl AuditorInterno {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn with_capital(base_capital: f64) -> Self {
        Self {
            fills_journal: Vec::new(),
            fills_independientes: Vec::new(),
            base_capital: base_capital.max(1.0),
            thresholds: TribunalThresholds::default(),
        }
    }

    /// OLA Ω120: Builder para configurar umbrales adaptativos del Tribunal de Autoengaño
    pub fn with_thresholds(mut self, thresholds: TribunalThresholds) -> Self {
        self.thresholds = thresholds;
        self
    }

    /// OLA Ω120: Actualiza dinámicamente los umbrales del Tribunal de Autoengaño
    pub fn update_thresholds(&mut self, thresholds: TribunalThresholds) {
        if thresholds.validate().is_ok() {
            self.thresholds = thresholds;
        }
    }

    pub fn record_fill(&mut self, fill: ExchangeFillRecord) {
        // FIX #677: Descartar fills con flotantes no finitos
        if !fill.price.is_finite()
            || !fill.qty.is_finite()
            || !fill.fee_usd.is_finite()
            || !fill.realized_pnl.is_finite()
        {
            return;
        }
        if self.fills_journal.len() >= MAX_FILLS_JOURNAL {
            let overflow = self.fills_journal.len() + 1 - MAX_FILLS_JOURNAL;
            self.fills_journal.drain(0..overflow);
        }
        self.fills_journal.push(fill);
    }

    /// OLA Ω120: Calcula la discrepancia máxima normalizada entre métricas internas y realidad
    /// para alimentar el registro omnisciente y el asiento SeniorAuditorInterno en el Consejo.
    ///
    /// F412/F413/F271 (censo C-F7b): esta ruta usa el diario interno (alimentado por el
    /// propio cierre del núcleo), que NO es una fuente independiente. Queda para
    /// telemetría/tribunal; el veto vivo usa `compute_internal_discrepancy_evidenciada`.
    pub fn compute_internal_discrepancy(
        &self,
        internal_sharpe: f64,
        internal_win_rate: f64,
        internal_drawdown: f64,
    ) -> f64 {
        let independent = VerificadorResultados::compute_metrics_with_capital(
            &self.fills_journal,
            self.base_capital,
        );
        let (max_disc, _, _, _) = calculate_max_discrepancy(
            internal_sharpe,
            internal_win_rate,
            internal_drawdown,
            &independent,
        );
        max_disc
    }

    /// F412/F413/F271 (censo C-F7b §3, prioridad 1): registra un fill del canal
    /// INDEPENDIENTE (host/exchange vía ORDER_TRADE_UPDATE). Rechaza flotantes no
    /// finitos (misma guarda que `record_fill`) y acota el diario en
    /// MAX_FILLS_INDEPENDIENTES con drenaje del excedente más antiguo.
    pub fn record_fill_independiente(&mut self, fill: ExchangeFillRecord) {
        if !fill.price.is_finite()
            || !fill.qty.is_finite()
            || !fill.fee_usd.is_finite()
            || !fill.realized_pnl.is_finite()
        {
            return;
        }
        if self.fills_independientes.len() >= MAX_FILLS_INDEPENDIENTES {
            let overflow = self.fills_independientes.len() + 1 - MAX_FILLS_INDEPENDIENTES;
            self.fills_independientes.drain(0..overflow);
        }
        self.fills_independientes.push(fill);
    }

    /// F412/F413/F271: ¿existe fuente independiente con al menos un fill?
    #[inline]
    pub fn tiene_fuente_independiente(&self) -> bool {
        !self.fills_independientes.is_empty()
    }

    /// F412/F413/F271 (censo C-F7b §3, prioridad 1): discrepancia con PUERTA DE
    /// EVIDENCIA. `None` = sin fuente independiente ⇒ el asiento
    /// SeniorAuditorInterno NO debe vetar (clave ausente en el registro ⇒
    /// payload 0.0 ⇒ `disc > umbral` falso). Con fuente independiente, devuelve
    /// la máxima discrepancia de las tres patas contra las métricas
    /// reconstruidas de ese canal ajeno (sharpe, win rate y drawdown medidos
    /// sobre fills que el núcleo no produjo).
    ///
    /// SOL (2026-10-20, censo C-F7b §3-4): la pata de WIN RATE se evalúa por su
    /// COTA INFERIOR (`|disc| − ee` por bloques) cuando hay evidencia
    /// suficiente (n ≥ MIN_FILLS_SOMBRA). El ruido de muestreo de un WR de
    /// pocos cierres ya no veta: sólo veta la discrepancia que sobrevive a su
    /// propio intervalo. Con menos de MIN_FILLS_SOMBRA la pata queda como
    /// estaba (puntual) — el ajuste no inventa precisión que no existe.
    ///
    /// Contrato de recuperación: el veto deja de dispararse con datos
    /// tautológicos y sólo vuelve a hacerlo cuando la evidencia independiente
    /// sostiene la discrepancia — nunca por ausencia de dato ni por ruido.
    pub fn compute_internal_discrepancy_evidenciada(
        &self,
        internal_sharpe: f64,
        internal_win_rate: f64,
        internal_drawdown: f64,
    ) -> Option<f64> {
        if !self.tiene_fuente_independiente() {
            return None;
        }
        let independent = VerificadorResultados::compute_metrics_with_capital(
            &self.fills_independientes,
            self.base_capital,
        );
        let (_, sharpe_disc, wr_disc, dd_disc) = calculate_max_discrepancy(
            internal_sharpe,
            internal_win_rate,
            internal_drawdown,
            &independent,
        );
        // §3-4: cota inferior del intervalo de la pata WR. `sombra_wr_disc`
        // devuelve `None` con < MIN_FILLS_SOMBRA cierres ⇒ comportamiento
        // previo intacto (pata puntual).
        let wr_ajustada = match self.sombra_wr_disc(internal_win_rate) {
            Some(s) => (wr_disc - s.ee_real).max(0.0),
            None => wr_disc,
        };
        Some(sharpe_disc.max(wr_ajustada).max(dd_disc))
    }

    /// F412/F413/F271: métricas reconstruidas del canal independiente (host).
    /// Sólo telemetría/contratos: el veto usa `compute_internal_discrepancy_evidenciada`.
    pub fn compute_independent_metrics_from_host(&self) -> IndependentMetrics {
        VerificadorResultados::compute_metrics_with_capital(
            &self.fills_independientes,
            self.base_capital,
        )
    }

    /// F412/F413/F271: tamaño del diario del canal independiente (higiene de RAM).
    #[inline]
    pub fn fills_independientes_len(&self) -> usize {
        self.fills_independientes.len()
    }

    /// Computa directamente las métricas de rendimiento independientes calculadas desde los fills reales
    pub fn compute_independent_metrics(&self) -> IndependentMetrics {
        VerificadorResultados::compute_metrics_with_capital(
            &self.fills_journal,
            self.base_capital,
        )
    }

    // ------------------------------------------------------- sombras (fase 2)

    /// SOL (2026-10-19, censo C-F7b §3-2): SOMBRA de la pata de win rate.
    /// Publica `|WR interno − WR real|` contra el canal independiente con su
    /// error estándar y su estabilidad en dos mitades OOS. Sólo telemetría:
    /// ninguna decisión la lee hoy — mide, con intervalo, la pata que el
    /// censo marcó como tautológica (F413).
    ///
    /// `None` (no se publica) si: el claim interno no es finito, o no hay
    /// `MIN_FILLS_SOMBRA` cierres en el canal independiente.
    pub fn sombra_wr_disc(&self, internal_wr: f64) -> Option<SombraWr> {
        let wr_interno = if internal_wr.is_finite() {
            internal_wr.clamp(0.0, 1.0)
        } else {
            return None;
        };
        let cierres: Vec<&ExchangeFillRecord> = self
            .fills_independientes
            .iter()
            .filter(|f| f.is_closed_position)
            .collect();
        let n = cierres.len();
        if n < MIN_FILLS_SOMBRA {
            return None;
        }
        let neto_positivo = |f: &&ExchangeFillRecord| (f.realized_pnl - f.fee_usd) > 0.0;
        let wr_real = cierres.iter().filter(|f| neto_positivo(f)).count() as f64 / n as f64;
        let disc = (wr_interno - wr_real).abs();
        let (ee_real, wr_a, wr_b) = ee_proporcion_bloques(&cierres, neto_positivo);
        Some(SombraWr {
            wr_interno,
            wr_real,
            disc,
            ee_real,
            n,
            disc_mitad_a: (wr_interno - wr_a).abs(),
            disc_mitad_b: (wr_interno - wr_b).abs(),
        })
    }

    /// SOL (2026-10-19, censo C-F7b §3-2): SOMBRA de fricción REAL menos
    /// MODELADA, en basis points, sobre el canal independiente. Positiva = el
    /// sistema paga más de lo que su gate modela ⇒ el edge real es menor que
    /// el creído. La fricción modelada la pasa quien conoce la receta del
    /// gate (misma fuente que `sonda_de_banda`), nunca se hardcodea aquí.
    ///
    /// `None` si: la fricción modelada no es finita o es negativa, o hay
    /// menos de `MIN_FILLS_SOMBRA` fills con nocional finito > 0 y fee finito.
    pub fn sombra_friccion_bp(&self, friccion_modelada_bp: f64) -> Option<SombraFriccion> {
        if !(friccion_modelada_bp.is_finite() && friccion_modelada_bp >= 0.0) {
            return None;
        }
        let mut deltas: Vec<f64> = Vec::with_capacity(self.fills_independientes.len());
        for f in &self.fills_independientes {
            let nocional = f.price * f.qty;
            if !(nocional.is_finite() && nocional > 0.0) {
                continue;
            }
            if !(f.fee_usd.is_finite() && f.fee_usd >= 0.0) {
                continue;
            }
            let fee_bp = f.fee_usd / nocional * 1e4;
            if !fee_bp.is_finite() {
                continue;
            }
            deltas.push(fee_bp - friccion_modelada_bp);
        }
        let n = deltas.len();
        if n < MIN_FILLS_SOMBRA {
            return None;
        }
        let (media_bp, ee_bp, a, b) = media_ee_bloques(&deltas);
        Some(SombraFriccion {
            media_bp,
            ee_bp,
            n,
            media_mitad_a_bp: a,
            media_mitad_b_bp: b,
        })
    }

    /// Evaluates internal claims against independent exchange fill verification
    pub fn verify_internal_claims(
        &self,
        internal_sharpe: f64,
        internal_win_rate: f64,
        internal_drawdown: f64,
    ) -> DailyTribunalReport {
        let safe_sharpe = if internal_sharpe.is_finite() {
            internal_sharpe
        } else {
            0.0
        };

        let independent = VerificadorResultados::compute_metrics_with_capital(
            &self.fills_journal,
            self.base_capital,
        );

        let (max_disc, sharpe_disc, wr_disc, _) = calculate_max_discrepancy(
            internal_sharpe,
            internal_win_rate,
            internal_drawdown,
            &independent,
        );

        let verdict = evaluate_tribunal_verdict(max_disc, &self.thresholds);

        DailyTribunalReport {
            verdict,
            pnl_discrepancy_pct: wr_disc,
            sharpe_discrepancy_pct: sharpe_disc,
            justification: format!(
                "Max discrepancy: {:.2}%. Internal Sharpe={:.2} vs Indep={:.2}",
                max_disc * 100.0,
                safe_sharpe,
                independent.calculated_sharpe
            ),
        }
    }

}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_auditor_interno_verification_and_bias() {
        let mut auditor = AuditorInterno::with_capital(13.0);
        auditor.record_fill(ExchangeFillRecord {
            timestamp_ns: 1_000_000,
            symbol_id: 0,
            is_buyer: true,
            price: 60000.0,
            qty: 0.001,
            fee_usd: 0.024,
            realized_pnl: 1.50,
            is_closed_position: true,
        });
        auditor.record_fill(ExchangeFillRecord {
            timestamp_ns: 2_000_000,
            symbol_id: 0,
            is_buyer: false,
            price: 60100.0,
            qty: 0.001,
            fee_usd: 0.024,
            realized_pnl: -0.50,
            is_closed_position: true,
        });

        let tribunal = auditor.verify_internal_claims(2.0, 0.50, 0.05);
        assert_ne!(tribunal.justification, "");

    }

    #[test]
    fn test_verificador_resultados_nan_capital_immunity() {
        let fills = vec![
            ExchangeFillRecord {
                timestamp_ns: 1_000_000,
                symbol_id: 0,
                is_buyer: true,
                price: 60000.0,
                qty: 0.001,
                fee_usd: f64::NAN,
                realized_pnl: 1.0,
                is_closed_position: true,
            },
            ExchangeFillRecord {
                timestamp_ns: 2_000_000,
                symbol_id: 0,
                is_buyer: false,
                price: 60100.0,
                qty: 0.001,
                fee_usd: 0.02,
                realized_pnl: f64::NAN,
                is_closed_position: true,
            },
        ];

        let metrics_nan_cap = VerificadorResultados::compute_metrics_with_capital(&fills, f64::NAN);
        assert!(metrics_nan_cap.calculated_sharpe.is_finite());
        assert!(metrics_nan_cap.calculated_max_drawdown.is_finite());
        assert!(metrics_nan_cap.calculated_win_rate.is_finite());
    }

    #[test]
    fn f1_006_bounded_fills_journal_and_one_sided_drawdown_discrepancy() {
        let mut auditor = AuditorInterno::with_capital(13.0);
        for i in 0..(MAX_FILLS_JOURNAL + 128) {
            auditor.record_fill(ExchangeFillRecord {
                timestamp_ns: (i as u64 + 1) * 1_000_000_000,
                symbol_id: 0,
                is_buyer: i % 2 == 0,
                price: 50_000.0,
                qty: 0.001,
                fee_usd: 0.002,
                realized_pnl: if i % 2 == 0 { 0.05 } else { -0.02 },
                is_closed_position: true,
            });
        }
        assert_eq!(
            auditor.fills_journal.len(),
            MAX_FILLS_JOURNAL,
            "fills_journal debe estar acotado en MAX_FILLS_JOURNAL"
        );

        let indep = auditor.compute_independent_metrics();
        // Reportar un drawdown interno más conservador (mayor que el independiente) no debe producir discrepancia:
        let (_, _, _, dd_disc_conservative) = calculate_max_discrepancy(
            indep.calculated_sharpe,
            indep.calculated_win_rate,
            indep.calculated_max_drawdown + 0.05,
            &indep,
        );
        assert_eq!(dd_disc_conservative, 0.0);
    }
}
