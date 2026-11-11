//! M1 — Registro de cada operación (ledger) para evidencia financiera.
//!
//! # QUÉ
//! Un asiento por operación con `gross_pnl`, `fees`, `funding`, `slippage` y
//! `net_pnl`, y agregados por periodo. Es la fuente de datos que debe alimentar
//! el panel de PnL; hoy el panel imprime `N/D` porque no existe una ruta de
//! datos viva (hallazgo `SOL-PANEL-ND-20261110`).
//!
//! # POR QUÉ
//! Un importe que no se midió NO es cero. Convertir la ausencia en `0.0`
//! fabrica una pérdida de comisiones (`gross=0, fees=15, net=-15`) o un
//! beneficio sin costes (`fees=0`). Convención heredada de F1-099 y de la
//! Task 2 de V43: `NaN` = dato no medido (N/D) en su propio campo; `0.0`
//! medido sigue siendo `0.0`. El neto se deriva de los cuatro importes
//! medidos, nunca se inventa.
//!
//! # CÓMO
//! `net = gross - fees - funding - slippage`. Un valor negativo en `fees`,
//! `funding` o `slippage` (funding recibido, rebate) es dato medido válido:
//! no se prohíbe por heurística. Si algún componente falta o la resta
//! desborda, el neto queda `NaN` y los componentes medidos se conservan.
//! Los agregados son fail-closed: un asiento sin medir contamina SÓLO su
//! campo agregado con `NaN`, en vez de publicar una suma parcial que parezca
//! un total medido.
//!
//! # INVARIANTE (cerrado)
//! Los campos de `TradeLedgerEntry` son privados y el ÚNICO constructor es
//! `from_measured`, que deriva el neto. No hay `Deserialize` ni literal de
//! struct reachable desde fuera: no se puede inyectar un asiento con un neto
//! inconsistente que luego `totals()` publique como beneficio medido.
//!
//! # CAPITAL Y CABLE
//! `roi_pct(wallet_capital)` recibe el balance REAL de la wallet en Demo o
//! Producción desde el llamador. Aquí no hay capital fijo, ni balances de
//! ejemplo, ni constantes de periodicidad. Sin capital medido, o con neto
//! ausente, el ROI es `N/D` (`NaN`), y la división por cero está guardada.
//! Este módulo NO deriva `Serialize`: `serde_json` convertiría el `NaN` en
//! `null` y un `Number(null) || 0` del panel imprimiría `0.00`, que es justo
//! la cifra inventada que se prohíbe. Para el cable (paso 3) se usa
//! `wire_amounts()`, donde `None` representa N/D de forma explícita.
//!
//! # LÍMITES
//! Este módulo registra y agrega; NO conecta aún el panel ni persiste en
//! disco (pasos 3 y 2 del plan M1). Sin datos de operaciones reales, todos
//! los agregados son `N/D`; los importes sintéticos de las pruebas son sólo
//! para comportamiento y no constituyen evidencia financiera ni OOS.

/// Importes de una operación para el cable: `None` = N/D (dato no medido).
/// Es la representación explícita que evita `null` → `0.00` en el panel.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct WireAmounts {
    pub gross_pnl: Option<f64>,
    pub fees: Option<f64>,
    pub funding: Option<f64>,
    pub slippage: Option<f64>,
    pub net_pnl: Option<f64>,
}

/// Una operación cerrada, con los cinco importes en la moneda de la wallet.
/// Campos privados: construir sólo mediante `from_measured`.
#[derive(Debug, Clone, PartialEq)]
pub struct TradeLedgerEntry {
    timestamp_ns: u64,
    symbol: String,
    gross_pnl: f64,
    fees: f64,
    funding: f64,
    slippage: f64,
    net_pnl: f64,
}

impl TradeLedgerEntry {
    /// Construye un asiento a partir de importes MEDIDOS. No admite valores
    /// por defecto: cada campo ausente llega como `NaN` y permanece `NaN`.
    pub fn from_measured(
        timestamp_ns: u64,
        symbol: impl Into<String>,
        gross_pnl: f64,
        fees: f64,
        funding: f64,
        slippage: f64,
    ) -> Self {
        let gross = finite_or_absent(gross_pnl);
        let fee = finite_or_absent(fees);
        let fund = finite_or_absent(funding);
        let slip = finite_or_absent(slippage);

        let net_candidate = gross - fee - fund - slip;
        let net_valid = gross.is_finite()
            && fee.is_finite()
            && fund.is_finite()
            && slip.is_finite()
            && net_candidate.is_finite();
        let net = if net_valid { net_candidate } else { f64::NAN };

        Self {
            timestamp_ns,
            symbol: symbol.into(),
            gross_pnl: gross,
            fees: fee,
            funding: fund,
            slippage: slip,
            net_pnl: net,
        }
    }

    pub fn timestamp_ns(&self) -> u64 {
        self.timestamp_ns
    }

    pub fn symbol(&self) -> &str {
        &self.symbol
    }

    pub fn gross_pnl(&self) -> f64 {
        self.gross_pnl
    }

    pub fn fees(&self) -> f64 {
        self.fees
    }

    pub fn funding(&self) -> f64 {
        self.funding
    }

    pub fn slippage(&self) -> f64 {
        self.slippage
    }

    pub fn net_pnl(&self) -> f64 {
        self.net_pnl
    }

    pub fn gross_is_measured(&self) -> bool {
        self.gross_pnl.is_finite()
    }

    pub fn net_is_measured(&self) -> bool {
        self.net_pnl.is_finite()
    }

    /// Representación explícita para el cable: N/D es `None`, nunca `null`
    /// interpretado como cero ni un `NaN` serializado de forma opaca.
    pub fn wire_amounts(&self) -> WireAmounts {
        WireAmounts {
            gross_pnl: measured_opt(self.gross_pnl),
            fees: measured_opt(self.fees),
            funding: measured_opt(self.funding),
            slippage: measured_opt(self.slippage),
            net_pnl: measured_opt(self.net_pnl),
        }
    }
}

/// Agregado de un periodo. `operations == 0` ⇒ todos los importes `NaN` (N/D).
/// Es un valor de SALIDA: no alimenta el ledger, por lo que sus campos pueden
/// leerse sin riesgo de inyección.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct LedgerTotals {
    pub operations: usize,
    pub gross_pnl: f64,
    pub fees: f64,
    pub funding: f64,
    pub slippage: f64,
    pub net_pnl: f64,
}

impl LedgerTotals {
    fn absent() -> Self {
        Self {
            operations: 0,
            gross_pnl: f64::NAN,
            fees: f64::NAN,
            funding: f64::NAN,
            slippage: f64::NAN,
            net_pnl: f64::NAN,
        }
    }

    /// Igual que `wire_amounts()` del asiento, pero para el agregado.
    pub fn wire_amounts(&self) -> WireAmounts {
        WireAmounts {
            gross_pnl: measured_opt(self.gross_pnl),
            fees: measured_opt(self.fees),
            funding: measured_opt(self.funding),
            slippage: measured_opt(self.slippage),
            net_pnl: measured_opt(self.net_pnl),
        }
    }
}

/// Ledger en memoria del periodo. Sin capacidad fija ni ceros por defecto.
#[derive(Debug, Default)]
pub struct TradeLedger {
    entries: Vec<TradeLedgerEntry>,
}

impl TradeLedger {
    pub fn new() -> Self {
        Self {
            entries: Vec::new(),
        }
    }

    pub fn append(&mut self, entry: TradeLedgerEntry) {
        self.entries.push(entry);
    }

    pub fn len(&self) -> usize {
        self.entries.len()
    }

    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }

    pub fn entries(&self) -> &[TradeLedgerEntry] {
        &self.entries
    }

    /// Agregados fail-closed por campo. Un asiento con un campo ausente deja
    /// ese agregado en `NaN`; no se publica una suma parcial como si fuera
    /// un total medido. El desborde de la suma también se convierte en `NaN`.
    pub fn totals(&self) -> LedgerTotals {
        if self.entries.is_empty() {
            return LedgerTotals::absent();
        }

        let mut gross = 0.0f64;
        let mut fees = 0.0f64;
        let mut funding = 0.0f64;
        let mut slippage = 0.0f64;
        let mut net = 0.0f64;

        for entry in &self.entries {
            gross += entry.gross_pnl;
            fees += entry.fees;
            funding += entry.funding;
            slippage += entry.slippage;
            net += entry.net_pnl;
        }

        LedgerTotals {
            operations: self.entries.len(),
            gross_pnl: finite_or_absent(gross),
            fees: finite_or_absent(fees),
            funding: finite_or_absent(funding),
            slippage: finite_or_absent(slippage),
            net_pnl: finite_or_absent(net),
        }
    }

    /// ROI neto sobre el capital REAL de la wallet, en porcentaje.
    /// Guarda contra capital ausente, no finito, nulo o negativo (división
    /// por cero) y contra neto ausente. No hay capital fijo en este módulo.
    pub fn roi_pct(&self, wallet_capital: f64) -> f64 {
        let net = self.totals().net_pnl;
        if !wallet_capital.is_finite() || wallet_capital <= 0.0 || !net.is_finite() {
            return f64::NAN;
        }
        finite_or_absent(net / wallet_capital * 100.0)
    }
}

/// Un valor finito (incluido el cero medido) se conserva; `NaN`/`±Inf`
/// se publica como ausencia explícita.
fn finite_or_absent(value: f64) -> f64 {
    if value.is_finite() { value } else { f64::NAN }
}

fn measured_opt(value: f64) -> Option<f64> {
    if value.is_finite() { Some(value) } else { None }
}

// ===========================================================================
// M1 PASO 2 — Informe financiero: ROI / PnL / WR ANTES y DESPUES de fees,
// con PERIODO y DATOS USADOS. Sin datos reales => N/D, nunca cifras inventadas.
// ===========================================================================

/// Periodo cerrado `[start_ns, end_ns]` en nanosegundos. Lo aporta el llamador:
/// aquí no hay ventana por defecto ni constantes de tiempo. Un periodo con
/// `end_ns < start_ns` es inválido y deja TODAS las métricas en N/D.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ReportPeriod {
    start_ns: u64,
    end_ns: u64,
}

impl ReportPeriod {
    pub fn new(start_ns: u64, end_ns: u64) -> Self {
        Self { start_ns, end_ns }
    }

    pub fn start_ns(&self) -> u64 {
        self.start_ns
    }

    pub fn end_ns(&self) -> u64 {
        self.end_ns
    }

    pub fn is_valid(&self) -> bool {
        self.end_ns >= self.start_ns
    }

    pub fn contains(&self, timestamp_ns: u64) -> bool {
        self.start_ns <= timestamp_ns && timestamp_ns <= self.end_ns
    }
}

/// Procedencia de los datos del informe. Sólo `Real` autoriza a publicar
/// cifras; `Unavailable` obliga a N/D en todas las métricas. Es el llamador
/// (Demo/Producción) quien declara la procedencia y el dataset usado.
#[derive(Debug, Clone, PartialEq)]
pub enum DataProvenance {
    /// Datos reales medidos de la wallet, con el identificador del dataset.
    Real { dataset: String },
    /// Sin datos reales: no se publica ninguna cifra.
    Unavailable,
}

impl DataProvenance {
    pub fn is_real(&self) -> bool {
        match self {
            DataProvenance::Real { .. } => true,
            DataProvenance::Unavailable => false,
        }
    }

    pub fn dataset(&self) -> Option<&str> {
        match self {
            DataProvenance::Real { dataset } => Some(dataset.as_str()),
            DataProvenance::Unavailable => None,
        }
    }
}

/// Métricas calculadas sobre un conjunto de asientos. Comportamiento puro y
/// determinista; NO constituye evidencia financiera por sí mismo: el informe
/// (`FinancialReport`) sólo las publica cuando hay datos reales declarados.
/// Convención explícita (sin heurística): `before_fees` = bruto, es decir SIN
/// ningún coste; `after_fees` = neto, es decir DESPUÉS de `fees + funding +
/// slippage` (los tres costes registrados por operación). Así `WR` puede caer
/// al pasar de bruto a neto, que es justo lo que el informe debe medir.
/// Unidades: PnL y costes en la moneda de la wallet; ROI en porcentaje;
/// `win_rate_*` como fracción en [0, 1]. `NaN` = N/D por campo.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct LedgerMetrics {
    rows: usize,
    pnl_before_fees: f64,
    pnl_after_fees: f64,
    fees: f64,
    funding: f64,
    slippage: f64,
    roi_before_fees_pct: f64,
    roi_after_fees_pct: f64,
    win_rate_before_fees: f64,
    win_rate_after_fees: f64,
}

impl LedgerMetrics {
    /// Cálculo puro sobre asientos YA medidos. Sin capital fijo: `wallet_capital`
    /// es el balance REAL de la wallet en Demo/Producción, aportado por el
    /// llamador. Sin capital medido, o con capital nulo/negativo, los ROI son
    /// N/D (guarda de división por cero). Sin asientos, todo es N/D.
    pub fn compute(entries: &[TradeLedgerEntry], wallet_capital: f64) -> Self {
        // @BEGIN_COMPUTE@
        if entries.is_empty() {
            return Self::absent();
        }
        let capital = finite_or_absent(wallet_capital);
        let pnl_before_fees = sum_checked(entries.iter().map(|e| e.gross_pnl()));
        let pnl_after_fees = sum_checked(entries.iter().map(|e| e.net_pnl()));
        let fees = sum_checked(entries.iter().map(|e| e.fees()));
        let funding = sum_checked(entries.iter().map(|e| e.funding()));
        let slippage = sum_checked(entries.iter().map(|e| e.slippage()));
        Self {
            rows: entries.len(),
            pnl_before_fees,
            pnl_after_fees,
            fees,
            funding,
            slippage,
            roi_before_fees_pct: roi_checked(pnl_before_fees, capital),
            roi_after_fees_pct: roi_checked(pnl_after_fees, capital),
            win_rate_before_fees: win_rate_checked(entries, |e| e.gross_pnl()),
            win_rate_after_fees: win_rate_checked(entries, |e| e.net_pnl()),
        }
        // @END_COMPUTE@
    }

    fn absent() -> Self {
        Self {
            rows: 0,
            pnl_before_fees: f64::NAN,
            pnl_after_fees: f64::NAN,
            fees: f64::NAN,
            funding: f64::NAN,
            slippage: f64::NAN,
            roi_before_fees_pct: f64::NAN,
            roi_after_fees_pct: f64::NAN,
            win_rate_before_fees: f64::NAN,
            win_rate_after_fees: f64::NAN,
        }
    }

    pub fn rows(&self) -> usize {
        self.rows
    }

    pub fn pnl_before_fees(&self) -> f64 {
        self.pnl_before_fees
    }

    pub fn pnl_after_fees(&self) -> f64 {
        self.pnl_after_fees
    }

    pub fn fees(&self) -> f64 {
        self.fees
    }

    pub fn funding(&self) -> f64 {
        self.funding
    }

    pub fn slippage(&self) -> f64 {
        self.slippage
    }

    pub fn roi_before_fees_pct(&self) -> f64 {
        self.roi_before_fees_pct
    }

    pub fn roi_after_fees_pct(&self) -> f64 {
        self.roi_after_fees_pct
    }

    pub fn win_rate_before_fees(&self) -> f64 {
        self.win_rate_before_fees
    }

    pub fn win_rate_after_fees(&self) -> f64 {
        self.win_rate_after_fees
    }
}

/// Informe de un periodo. Lleva SIEMPRE el periodo y los datos usados
/// (dataset, filas, capital). Sin datos reales, todas las métricas son N/D.
#[derive(Debug, Clone, PartialEq)]
pub struct FinancialReport {
    period: ReportPeriod,
    dataset: Option<String>,
    rows: usize,
    wallet_capital: f64,
    metrics: LedgerMetrics,
}

impl FinancialReport {
    /// Construye el informe del periodo usando SÓLO los asientos cuyo
    /// `timestamp_ns` cae dentro de `[start_ns, end_ns]`: mezclar operaciones
    /// de fuera del periodo produciría cifras que no corresponden a los datos
    /// declarados. Sin procedencia real, periodo inválido, capital no medido
    /// o periodo sin operaciones => N/D en todas las métricas.
    pub fn build(
        ledger: &TradeLedger,
        period: ReportPeriod,
        wallet_capital: f64,
        provenance: DataProvenance,
    ) -> Self {
        // @BEGIN_BUILD@
        let inside: Vec<TradeLedgerEntry> = if period.is_valid() {
            ledger
                .entries()
                .iter()
                .filter(|e| period.contains(e.timestamp_ns()))
                .cloned()
                .collect()
        } else {
            Vec::new()
        };

        let capital = finite_or_absent(wallet_capital);
        // Disponibilidad independiente: que falte el capital deja SÓLO los ROI
        // en N/D (roi_checked), no descarta PnL ni WR medidos. No se exige
        // capital aquí para no publicar N/D donde sí hay datos reales.
        let publicable = provenance.is_real() && period.is_valid() && !inside.is_empty();

        let metrics = if publicable {
            LedgerMetrics::compute(&inside, capital)
        } else {
            LedgerMetrics::absent()
        };

        // Un periodo inválido NO es un dato medido: no se publican filas ni
        // capital como si el informe fuera válido (evita `0 filas` y un
        // capital real en un informe rechazado).
        let usable_period = provenance.is_real() && period.is_valid();
        Self {
            period,
            dataset: provenance.dataset().map(str::to_string),
            rows: if usable_period { inside.len() } else { 0 },
            wallet_capital: if usable_period { capital } else { f64::NAN },
            metrics,
        }
        // @END_BUILD@
    }

    pub fn period(&self) -> ReportPeriod {
        self.period
    }

    /// Identificador del dataset usado, o `None` si no hay datos reales.
    pub fn dataset(&self) -> Option<&str> {
        self.dataset.as_deref()
    }

    /// Filas usadas del periodo. `0` con procedencia real significa "periodo
    /// sin operaciones" (dato medido); con procedencia ausente significa N/D.
    pub fn rows(&self) -> usize {
        self.rows
    }

    /// Capital real usado (`NaN` = no medido). Nunca un valor fijo.
    pub fn wallet_capital(&self) -> f64 {
        self.wallet_capital
    }

    pub fn metrics(&self) -> &LedgerMetrics {
        &self.metrics
    }

    /// `true` sólo si hay métricas calculadas a partir de datos reales.
    pub fn has_evidence(&self) -> bool {
        self.dataset.is_some() && self.metrics.rows() > 0
    }

    /// Representación para el cable: `None` = N/D explícito. Evita que un
    /// `null` se interprete como `0.00` en el panel (paso 3).
    pub fn wire(&self) -> ReportWire {
        let m = &self.metrics;
        ReportWire {
            period_start_ns: self.period.start_ns(),
            period_end_ns: self.period.end_ns(),
            dataset: self.dataset.clone(),
            // Un periodo inválido no es un dato medido: `rows` también es N/D.
            rows: if self.dataset.is_some() && self.period.is_valid() {
                Some(self.rows)
            } else {
                None
            },
            wallet_capital: measured_opt(self.wallet_capital),
            pnl_before_fees: measured_opt(m.pnl_before_fees()),
            pnl_after_fees: measured_opt(m.pnl_after_fees()),
            fees: measured_opt(m.fees()),
            funding: measured_opt(m.funding()),
            slippage: measured_opt(m.slippage()),
            roi_before_fees_pct: measured_opt(m.roi_before_fees_pct()),
            roi_after_fees_pct: measured_opt(m.roi_after_fees_pct()),
            win_rate_before_fees: measured_opt(m.win_rate_before_fees()),
            win_rate_after_fees: measured_opt(m.win_rate_after_fees()),
        }
    }
}

/// Informe para el cable. Cada métrica es `Option<f64>`: `None` = N/D,
/// `Some(v)` = valor medido (incluido un cero real).
#[derive(Debug, Clone, PartialEq)]
pub struct ReportWire {
    pub period_start_ns: u64,
    pub period_end_ns: u64,
    pub dataset: Option<String>,
    pub rows: Option<usize>,
    pub wallet_capital: Option<f64>,
    pub pnl_before_fees: Option<f64>,
    pub pnl_after_fees: Option<f64>,
    pub fees: Option<f64>,
    pub funding: Option<f64>,
    pub slippage: Option<f64>,
    pub roi_before_fees_pct: Option<f64>,
    pub roi_after_fees_pct: Option<f64>,
    pub win_rate_before_fees: Option<f64>,
    pub win_rate_after_fees: Option<f64>,
}

/// Suma fail-closed: un componente ausente o un desborde dejan el campo en
/// N/D en vez de publicar una suma parcial como si fuera un total medido.
fn sum_checked<I: IntoIterator<Item = f64>>(values: I) -> f64 {
    let mut acc = 0.0f64;
    for value in values {
        if !value.is_finite() {
            return f64::NAN;
        }
        acc += value;
        if !acc.is_finite() {
            return f64::NAN;
        }
    }
    acc
}

/// ROI con guarda explícita de división por cero y de capital no medido.
fn roi_checked(pnl: f64, capital: f64) -> f64 {
    if !pnl.is_finite() || !capital.is_finite() || capital <= 0.0 {
        return f64::NAN;
    }
    finite_or_absent(pnl / capital * 100.0)
}

/// Win rate como fracción en [0, 1]. Sin asientos o con algún valor ausente
/// => N/D (no se cuentan como derrotas ni se inventa un 0.0).
fn win_rate_checked<F: Fn(&TradeLedgerEntry) -> f64>(entries: &[TradeLedgerEntry], pick: F) -> f64 {
    if entries.is_empty() {
        return f64::NAN;
    }
    let mut wins = 0usize;
    for entry in entries {
        let value = pick(entry);
        if !value.is_finite() {
            return f64::NAN;
        }
        if value > 0.0 {
            wins += 1;
        }
    }
    finite_or_absent(wins as f64 / entries.len() as f64)
}

// ===========================================================================
// M1 PASO 3 — Conexión del panel de PnL al registro de operaciones.
//
// El `causal_report` del paquete mmap sigue siendo `None` en los archivos
// preparados por otros (no se tocan). Este conector da al panel una ruta de
// datos propia y explícita: un JSONL de operaciones REALES, con periodo,
// dataset y capital REAL de la wallet declarados por el operador mediante
// variables de entorno. Sin alguno de ellos => TODO "N/D"; nunca se fabrica
// un 0.00, un capital por defecto ni una procedencia inventada.
// ===========================================================================

/// Líneas ya formateadas del panel [1] "Pre & Post Fees".
/// Cada campo es una cadena lista para imprimir: `N/D` = dato no medido.
#[derive(Debug, Clone, PartialEq)]
pub struct PnlPanelLines {
    pub gross_pnl: String,
    pub fees: String,
    pub funding: String,
    pub slippage: String,
    pub net_pnl: String,
    pub roi_before_fees_pct: String,
    pub roi_after_fees_pct: String,
    pub win_rate_before_fees: String,
    pub win_rate_after_fees: String,
    pub period: String,
    pub dataset: String,
    pub rows: String,
}

impl PnlPanelLines {
    /// Sin datos reales verificables: N/D en TODOS los campos.
    pub fn unavailable() -> Self {
        let nd = || "N/D".to_string();
        Self {
            gross_pnl: nd(),
            fees: nd(),
            funding: nd(),
            slippage: nd(),
            net_pnl: nd(),
            roi_before_fees_pct: nd(),
            roi_after_fees_pct: nd(),
            win_rate_before_fees: nd(),
            win_rate_after_fees: nd(),
            period: nd(),
            dataset: nd(),
            rows: nd(),
        }
    }
}

/// `None` o valor no finito => `N/D`. Un cero medido se imprime como `0.00`.
fn money(value: Option<f64>) -> String {
    match value {
        Some(v) if v.is_finite() => format!("{:.2}", v),
        _ => "N/D".to_string(),
    }
}

/// Win rate (fracción [0,1]) a porcentaje con una decimal.
fn percent(value: Option<f64>) -> String {
    match value {
        Some(v) if v.is_finite() => format!("{:.1}%", v * 100.0),
        _ => "N/D".to_string(),
    }
}

/// ROI ya viene en porcentaje; se muestra con dos decimales.
fn roi(value: Option<f64>) -> String {
    match value {
        Some(v) if v.is_finite() => format!("{:.2}", v),
        _ => "N/D".to_string(),
    }
}

/// Formatea un informe para el panel. Sólo `wire()` (Option<f64>) decide
/// qué es N/D; aquí no se inventa ningún valor ni se convierte `null` en 0.
pub fn pnl_panel_lines(report: &FinancialReport) -> PnlPanelLines {
    // @BEGIN_FORMAT@
    let w = report.wire();
    // Sin procedencia real declarada no hay informe: N/D en los 12 campos,
    // incluido el periodo (presentar un periodo sin datos sugeriría un
    // informe que no existe). Con procedencia real, periodo y filas son
    // metadatos medidos aunque las métricas estén ausentes.
    if w.dataset.is_none() {
        return PnlPanelLines::unavailable();
    }
    PnlPanelLines {
        gross_pnl: money(w.pnl_before_fees),
        fees: money(w.fees),
        funding: money(w.funding),
        slippage: money(w.slippage),
        net_pnl: money(w.pnl_after_fees),
        roi_before_fees_pct: roi(w.roi_before_fees_pct),
        roi_after_fees_pct: roi(w.roi_after_fees_pct),
        win_rate_before_fees: percent(w.win_rate_before_fees),
        win_rate_after_fees: percent(w.win_rate_after_fees),
        period: format!("{}..{} ns", w.period_start_ns, w.period_end_ns),
        dataset: match w.dataset {
            Some(d) => d,
            None => "N/D".to_string(),
        },
        rows: match w.rows {
            Some(n) => n.to_string(),
            None => "N/D".to_string(),
        },
    }
    // @END_FORMAT@
}

/// Parsea un JSONL de operaciones: una línea por operación con
/// `timestamp_ns`, `symbol`, `gross_pnl`, `fees`, `funding`, `slippage`.
/// Un campo ausente o no finito queda como ausencia (`NaN`) en el asiento,
/// no como cero. Una línea malformada => panel completo N/D (fail-closed).
/// Sin `unwrap`/`expect`/`panic`: todos los fallos devuelven N/D.
pub fn pnl_panel_from_ledger_jsonl(
    text: &str,
    period: ReportPeriod,
    wallet_capital: f64,
    provenance: DataProvenance,
) -> PnlPanelLines {
    // @BEGIN_PARSE@
    let mut ledger = TradeLedger::new();
    for line in text.lines() {
        if line.trim().is_empty() {
            continue;
        }
        let row: serde_json::Value = match serde_json::from_str(line) {
            Ok(v) => v,
            Err(_) => return PnlPanelLines::unavailable(),
        };
        let timestamp_ns = match row.get("timestamp_ns").and_then(|v| v.as_u64()) {
            Some(t) => t,
            None => return PnlPanelLines::unavailable(),
        };
        let symbol = match row.get("symbol").and_then(|v| v.as_str()) {
            Some(s) if !s.is_empty() => s.to_string(),
            _ => return PnlPanelLines::unavailable(),
        };
        ledger.append(TradeLedgerEntry::from_measured(
            timestamp_ns,
            symbol,
            json_finite(&row, "gross_pnl"),
            json_finite(&row, "fees"),
            json_finite(&row, "funding"),
            json_finite(&row, "slippage"),
        ));
    }
    let report = FinancialReport::build(&ledger, period, wallet_capital, provenance);
    pnl_panel_lines(&report)
    // @END_PARSE@
}

/// Ausente o no finito => `NaN` (N/D). Un cero medido se conserva.
fn json_finite(row: &serde_json::Value, key: &str) -> f64 {
    match row.get(key).and_then(|v| v.as_f64()) {
        Some(v) if v.is_finite() => v,
        _ => f64::NAN,
    }
}

fn env_nonempty(name: &str) -> Option<String> {
    match std::env::var(name) {
        Ok(v) if !v.trim().is_empty() => Some(v),
        _ => None,
    }
}

fn env_u64(name: &str) -> Option<u64> {
    match env_nonempty(name) {
        Some(v) => match v.trim().parse::<u64>() {
            Ok(n) => Some(n),
            Err(_) => None,
        },
        None => None,
    }
}

/// Fuente de datos del panel, declarada por el operador en Demo/Producción.
/// Variables: `TG_M1_LEDGER_JSONL` (ruta), `TG_M1_PERIOD_START_NS`,
/// `TG_M1_PERIOD_END_NS`, `TG_M1_DATASET`, `TG_M1_DATA_REAL`
/// (`demo`|`production`) y `TG_M1_WALLET_CAPITAL` (balance REAL de la wallet).
/// Falta alguna o la procedencia no es real => N/D total. No hay ruta por
/// defecto, ni capital fijo, ni dataset inventado.
pub fn pnl_panel_from_env() -> PnlPanelLines {
    let path = match env_nonempty("TG_M1_LEDGER_JSONL") {
        Some(p) => p,
        None => return PnlPanelLines::unavailable(),
    };
    let dataset = match env_nonempty("TG_M1_DATASET") {
        Some(d) => d,
        None => return PnlPanelLines::unavailable(),
    };
    let real = match env_nonempty("TG_M1_DATA_REAL") {
        Some(s) if s == "demo" || s == "production" => DataProvenance::Real { dataset },
        _ => return PnlPanelLines::unavailable(),
    };
    let start = match env_u64("TG_M1_PERIOD_START_NS") {
        Some(v) => v,
        None => return PnlPanelLines::unavailable(),
    };
    let end = match env_u64("TG_M1_PERIOD_END_NS") {
        Some(v) => v,
        None => return PnlPanelLines::unavailable(),
    };
    let capital = match env_nonempty("TG_M1_WALLET_CAPITAL") {
        Some(v) => match v.trim().parse::<f64>() {
            Ok(c) if c.is_finite() && c > 0.0 => c,
            _ => return PnlPanelLines::unavailable(),
        },
        None => return PnlPanelLines::unavailable(),
    };
    let text = match std::fs::read_to_string(&path) {
        Ok(t) => t,
        Err(_) => return PnlPanelLines::unavailable(),
    };
    pnl_panel_from_ledger_jsonl(&text, ReportPeriod::new(start, end), capital, real)
}
