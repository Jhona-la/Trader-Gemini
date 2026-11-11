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
