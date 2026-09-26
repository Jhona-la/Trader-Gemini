//! Bounded income evidence and unit-aware summaries, independent of account I/O.
//! An exhausted numbered page is an observation of this API traversal, not a
//! provider snapshot, retention guarantee, fill ledger or independent trade sample.
use crate::order_types::IncomeEntry;
use std::collections::{BTreeMap, HashMap};
use std::future::Future;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum IncomeCoverage {
    PageExhausted,
    PageBudgetExceeded,
    NoProgress,
    Simulated,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum IncomeEvidenceError {
    InvalidRange,
    InvalidPageBudget,
    InvalidPageSize,
    InvalidPage,
    InvalidTimestamp,
    UnsupportedFilter,
    InvalidRecord,
    MixedAssets,
    OutOfRange,
    OversizedPage,
    ConflictingRecord,
    NonFiniteAggregate,
    CountOverflow,
    Incomplete(IncomeCoverage),
    Transport(String),
}

#[derive(Debug)]
pub struct IncomeWindow {
    pub start_ms: u64,
    pub end_ms: u64,
    pub pages_read: u32,
    pub coverage: IncomeCoverage,
    pub entries: Vec<IncomeEntry>,
}

impl IncomeWindow {
    pub fn into_exhausted_entries(self) -> Result<Vec<IncomeEntry>, IncomeEvidenceError> {
        if self.coverage == IncomeCoverage::PageExhausted {
            Ok(self.entries)
        } else {
            Err(IncomeEvidenceError::Incomplete(self.coverage))
        }
    }
}

fn validate_range(start_ms: u64, end_ms: u64) -> Result<(), IncomeEvidenceError> {
    if start_ms > end_ms || end_ms > i64::MAX as u64 {
        Err(IncomeEvidenceError::InvalidRange)
    } else {
        Ok(())
    }
}

/// Canonical unsigned query for the signed endpoint. incomeType is a scalar;
/// lists/query delimiters are rejected instead of inventing an API extension.
pub fn income_page_query(
    start_ms: u64,
    end_ms: u64,
    page: u32,
    limit: u32,
    income_types: &[&str],
    timestamp: u64,
) -> Result<String, IncomeEvidenceError> {
    validate_range(start_ms, end_ms)?;
    if page == 0 {
        return Err(IncomeEvidenceError::InvalidPage);
    }
    if !(1..=1000).contains(&limit) {
        return Err(IncomeEvidenceError::InvalidPageSize);
    }
    if timestamp == 0 || timestamp > i64::MAX as u64 {
        return Err(IncomeEvidenceError::InvalidTimestamp);
    }
    if income_types.len() > 1
        || income_types.first().is_some_and(|value| {
            value.is_empty()
                || !value
                    .bytes()
                    .all(|b| b.is_ascii_uppercase() || b.is_ascii_digit() || b == b'_')
        })
    {
        return Err(IncomeEvidenceError::UnsupportedFilter);
    }
    let mut query = format!(
        "startTime={start_ms}&endTime={end_ms}&page={page}&limit={limit}&timestamp={timestamp}"
    );
    if let Some(kind) = income_types.first() {
        query.push_str("&incomeType=");
        query.push_str(kind);
    }
    Ok(query)
}

/// Exact visible identity, intentionally not an assertion of global tranId
/// uniqueness. Same visible identity with another amount is a conflict.
/// If the provider revises identity fields, reconciliation is still needed.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
struct IncomeIdentity {
    asset: String,
    symbol: String,
    kind: String,
    transaction: u64,
    time: u64,
    trade: String,
}

fn identity(e: &IncomeEntry) -> Result<IncomeIdentity, IncomeEvidenceError> {
    if !e.income.is_finite()
        || e.asset.trim().is_empty()
        || e.income_type.trim().is_empty()
        || e.tran_id == 0
        || e.time == 0
    {
        return Err(IncomeEvidenceError::InvalidRecord);
    }
    Ok(IncomeIdentity {
        asset: e.asset.clone(),
        symbol: e.symbol.clone(),
        kind: e.income_type.clone(),
        transaction: e.tran_id,
        time: e.time,
        trade: e.trade_id.clone(),
    })
}

/// Page size and page budget bound calls/storage; neither establishes an
/// economic threshold. All pages use the caller's fixed [start_ms,end_ms].
/// The fetcher receives a 1-based page number, never a lossy timestamp cursor.
pub async fn collect_income_window<F, Fut>(
    start_ms: u64,
    end_ms: u64,
    max_pages: u32,
    page_size: u32,
    mut fetch: F,
) -> Result<IncomeWindow, IncomeEvidenceError>
where
    F: FnMut(u32) -> Fut,
    Fut: Future<Output = Result<Vec<IncomeEntry>, String>>,
{
    validate_range(start_ms, end_ms)?;
    if max_pages == 0 {
        return Err(IncomeEvidenceError::InvalidPageBudget);
    }
    if !(1..=1000).contains(&page_size) {
        return Err(IncomeEvidenceError::InvalidPageSize);
    }
    let mut window = IncomeWindow {
        start_ms,
        end_ms,
        pages_read: 0,
        coverage: IncomeCoverage::PageBudgetExceeded,
        entries: Vec::new(),
    };
    let mut seen = HashMap::<IncomeIdentity, f64>::new();
    for page in 1..=max_pages {
        let rows = fetch(page).await.map_err(IncomeEvidenceError::Transport)?;
        window.pages_read = page;
        let count = rows.len();
        if count > page_size as usize {
            return Err(IncomeEvidenceError::OversizedPage);
        }
        let before = window.entries.len();
        for row in rows {
            let key = identity(&row)?;
            if row.time < start_ms || row.time > end_ms {
                return Err(IncomeEvidenceError::OutOfRange);
            }
            if let Some(previous) = seen.get(&key) {
                if *previous != row.income {
                    return Err(IncomeEvidenceError::ConflictingRecord);
                }
            } else {
                seen.insert(key, row.income);
                window.entries.push(row);
            }
        }
        // A repeated nonempty page is not proof that the range is exhausted.
        if count > 0 && window.entries.len() == before {
            window.coverage = IncomeCoverage::NoProgress;
            break;
        }
        if count < page_size as usize {
            window.coverage = IncomeCoverage::PageExhausted;
            break;
        }
    }
    Ok(window)
}

/// A unit guard, not FX conversion. Empty evidence has no currency.
/// Repeated/conflicting identities must already have been checked by collection.
pub fn single_income_asset(entries: &[IncomeEntry]) -> Result<Option<&str>, IncomeEvidenceError> {
    let mut asset = None;
    for entry in entries {
        identity(entry)?;
        match asset {
            Some(previous) if previous != entry.asset => {
                return Err(IncomeEvidenceError::MixedAssets)
            }
            None => asset = Some(entry.asset.as_str()),
            _ => {}
        }
    }
    Ok(asset)
}

#[derive(Debug, Default, Clone)]
pub struct IncomeTotals {
    pub realized_pnl: f64,
    pub commission: f64,
    pub funding: f64,
    /// All other cash-flow classes; NOT assumed to be trading profit.
    pub other_income: f64,
    pub pnl_rows: u64,
    pub positive_pnl_rows: u64,
    pub other_rows: u64,
}

#[derive(Debug, Default)]
pub struct FeeCurrencyPartition {
    /// Only rows used by the legacy fee policy, each symbol in one asset.
    /// This is not economic eligibility or independent-trade evidence.
    pub same_asset_by_symbol: BTreeMap<String, Vec<IncomeEntry>>,
    pub rejected_by_symbol: BTreeMap<String, IncomeEvidenceError>,
}

/// Isolate missing FX evidence by affected symbol. A global transfer or an
/// unrelated symbol's fee asset must not suppress every other symbol's review.
/// Other classes remain excluded by the legacy policy, not declared irrelevant.
pub fn partition_legacy_fee_evidence(entries: &[IncomeEntry]) -> FeeCurrencyPartition {
    let mut groups = BTreeMap::<String, Vec<IncomeEntry>>::new();
    for entry in entries {
        if !entry.symbol.is_empty()
            && matches!(
                entry.income_type.as_str(),
                "REALIZED_PNL" | "COMMISSION" | "FUNDING_FEE"
            )
        {
            groups
                .entry(entry.symbol.clone())
                .or_default()
                .push(entry.clone());
        }
    }
    let mut result = FeeCurrencyPartition::default();
    for (symbol, rows) in groups {
        match single_income_asset(&rows) {
            Ok(_) => {
                result.same_asset_by_symbol.insert(symbol, rows);
            }
            Err(reason) => {
                result.rejected_by_symbol.insert(symbol, reason);
            }
        }
    }
    result
}

fn add(a: f64, b: f64) -> Result<f64, IncomeEvidenceError> {
    let value = a + b;
    if value.is_finite() {
        Ok(value)
    } else {
        Err(IncomeEvidenceError::NonFiniteAggregate)
    }
}

impl IncomeTotals {
    fn record(&mut self, e: &IncomeEntry) -> Result<(), IncomeEvidenceError> {
        match e.income_type.as_str() {
            "REALIZED_PNL" => {
                self.realized_pnl = add(self.realized_pnl, e.income)?;
                self.pnl_rows = self
                    .pnl_rows
                    .checked_add(1)
                    .ok_or(IncomeEvidenceError::CountOverflow)?;
                if e.income > 0.0 {
                    self.positive_pnl_rows = self
                        .positive_pnl_rows
                        .checked_add(1)
                        .ok_or(IncomeEvidenceError::CountOverflow)?;
                }
            }
            "COMMISSION" => self.commission = add(self.commission, e.income)?,
            "FUNDING_FEE" => self.funding = add(self.funding, e.income)?,
            _ => {
                self.other_income = add(self.other_income, e.income)?;
                self.other_rows = self
                    .other_rows
                    .checked_add(1)
                    .ok_or(IncomeEvidenceError::CountOverflow)?;
            }
        }
        Ok(())
    }

    /// Subtotal of three selected classes in ONE asset, not complete equity PnL.
    pub fn selected_net(&self) -> Result<f64, IncomeEvidenceError> {
        add(add(self.realized_pnl, self.commission)?, self.funding)
    }

    /// Row-level positive fraction, not trade win rate; absent with no PnL rows.
    pub fn positive_row_fraction(&self) -> Option<f64> {
        if self.pnl_rows == 0 {
            None
        } else {
            Some(self.positive_pnl_rows as f64 / self.pnl_rows as f64)
        }
    }

    /// Signed (commission+funding)/abs(realized). Undefined at zero denominator;
    /// no monetary epsilon. Does not estimate friction per trade or per turnover.
    pub fn cost_to_abs_realized_ratio(&self) -> Option<f64> {
        if !self.realized_pnl.is_finite() || self.realized_pnl == 0.0 {
            return None;
        }
        let ratio = add(self.commission, self.funding).ok()? / self.realized_pnl.abs();
        ratio.is_finite().then_some(ratio)
    }
}

#[derive(Debug, Default)]
pub struct IncomeAggregation {
    pub by_symbol: BTreeMap<(String, String), IncomeTotals>,
    pub by_asset: BTreeMap<String, IncomeTotals>,
}

/// Consumes collected rows; not a second deduplicator. Currency is always part
/// of the aggregation key. No global sum of heterogeneous assets is exposed.
pub fn aggregate_income_by_asset(
    entries: &[IncomeEntry],
) -> Result<IncomeAggregation, IncomeEvidenceError> {
    let mut result = IncomeAggregation::default();
    for e in entries {
        identity(e)?;
        result
            .by_symbol
            .entry((e.asset.clone(), e.symbol.clone()))
            .or_default()
            .record(e)?;
        result
            .by_asset
            .entry(e.asset.clone())
            .or_default()
            .record(e)?;
    }
    for totals in result.by_symbol.values().chain(result.by_asset.values()) {
        totals.selected_net()?;
    }
    Ok(result)
}

/// Validate a reporting lookback rather than overflow/wrap timestamps or silently
/// clamp to the epoch. A representable request is not a retention guarantee.
pub fn income_lookback_start(now_ms: u64, days: u64) -> Result<u64, IncomeEvidenceError> {
    if days == 0 {
        return Err(IncomeEvidenceError::InvalidRange);
    }
    let span = days
        .checked_mul(86_400_000)
        .ok_or(IncomeEvidenceError::InvalidRange)?;
    now_ms
        .checked_sub(span)
        .ok_or(IncomeEvidenceError::InvalidRange)
}

// ═══════════════════════════════════════════════════════════════════════
// FMT-285 (OLA XL) — COBERTURA POR SÍMBOLO Y CUARENTENA RECUPERABLE
//
// Hoja de ruta §13.2 del informe XXXIX: «Incorporar intervalo/cobertura por
// símbolo y una política de cuarentena recuperable. Una muestra inválida no
// debe desaparecer ni convertirse en rentabilidad cero.»
//
// Antes (FMT-282/284): `collect_income_window` abortaba TODA la ventana al
// primer `InvalidRecord`, y la cobertura era un único estado global. Un solo
// registro corrupto suprimía la evidencia de todos los demás símbolos, y un
// símbolo sin filas era indistinguible de un símbolo cuya ventana se truncó.
//
// Qué garantiza esta partición:
//   · COBERTURA POR SÍMBOLO: el intervalo [min_time, max_time] de las filas
//     ACEPTADAS de cada símbolo, como observación de este recorrido — no una
//     prueba de retención del proveedor ni un censo completo.
//   · CUARENTENA RECUPERABLE: las filas que violan el contrato de identidad
//     se apartan CON su motivo. `recoverable` distingue lo que una re-lectura
//     puede reparar (registro malformado por transporte) de lo que exige
//     conciliación (identidad visible repetida con importe distinto).
//   · NINGUNA FILA DESAPARECE NI SE ANOTA A CERO: aceptadas + cuarentenadas
//     == entradas; el caller decide qué política aplica a cada partición.
// ═══════════════════════════════════════════════════════════════════════

/// Intervalo observado de un símbolo en ESTE recorrido: mín/máx tiempo de sus
/// filas aceptadas. No es cobertura de retención ni ausencia universal fuera
/// de él; es lo que esta ventana puede atestiguar por símbolo.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct SymbolInterval {
    pub first_time_ms: u64,
    pub last_time_ms: u64,
    pub rows: u64,
}

/// Una fila apartada, con el motivo del contrato que violó y si una re-lectura
/// puede repararla. El importe se conserva íntegro: nunca se anota a cero.
#[derive(Debug, Clone)]
pub struct QuarantinedEntry {
    pub entry: IncomeEntry,
    pub reason: QuarantineReason,
    /// true: reintentar la lectura puede repararla (malformación de transporte).
    /// false: exige conciliación (conflicto de identidad con importe distinto).
    pub recoverable: bool,
}

/// Motivos de cuarentena. `ConflictingIdentity` NO es recuperable por re-lectura.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum QuarantineReason {
    InvalidRecord,
    ConflictingIdentity,
}

/// Resultado de particionar una tanda por el contrato de identidad.
#[derive(Debug, Default)]
pub struct PartitionedIncome {
    pub accepted: Vec<IncomeEntry>,
    pub quarantined: Vec<QuarantinedEntry>,
    /// Repeticiones EXACTAS de una identidad ya aceptada (misma fila leída
    /// dos veces): descartadas sin cuarentena, contadas para conservación.
    pub exact_duplicates_dropped: u64,
    /// Intervalo observado por símbolo (filas aceptadas, símbolo vacío incluido
    /// como clave tal cual: la tanda no inventa símbolos que el registro no trae).
    pub interval_by_symbol: BTreeMap<String, SymbolInterval>,
}

impl PartitionedIncome {
    /// Inventario de conservación: nada se pierde ni se anota a cero.
    /// aceptadas + cuarentena + duplicados exactos == filas de entrada.
    pub fn accounted_rows(&self) -> u64 {
        (self.accepted.len() + self.quarantined.len()) as u64
            + self.exact_duplicates_dropped
    }

    /// Símbolos con filas en cuarentena: su intervalo observado está DEBILITADO
    /// y debe reportarse junto al motivo, no como cobertura limpia.
    pub fn symbols_with_quarantine(&self) -> BTreeMap<String, usize> {
        let mut out = BTreeMap::new();
        for q in &self.quarantined {
            *out.entry(q.entry.symbol.clone()).or_insert(0) += 1;
        }
        out
    }
}

/// Particiona una tanda ya recogida (sin transporte): filas válidas → aceptadas
/// + cobertura por símbolo; filas que violan el contrato → cuarentena con
/// motivo. A diferencia de `collect_income_window`, un registro inválido no
/// aborta la tanda. La deduplicación por identidad visible se aplica igual:
/// la repetición idéntica se descarta (ya contada), la conflictiva se aparta.
pub fn partition_income(entries: Vec<IncomeEntry>) -> PartitionedIncome {
    let mut out = PartitionedIncome::default();
    let mut seen = HashMap::<IncomeIdentity, f64>::new();
    for entry in entries {
        let key = match identity(&entry) {
            Ok(key) => key,
            Err(_) => {
                out.quarantined.push(QuarantinedEntry {
                    entry,
                    reason: QuarantineReason::InvalidRecord,
                    recoverable: true,
                });
                continue;
            }
        };
        if let Some(previous) = seen.get(&key) {
            if *previous != entry.income {
                out.quarantined.push(QuarantinedEntry {
                    entry,
                    reason: QuarantineReason::ConflictingIdentity,
                    recoverable: false,
                });
                continue;
            }
            // Repetición idéntica de una identidad ya aceptada: descartada
            // (ya está contada); no es cuarentena ni conflicto.
            out.exact_duplicates_dropped += 1;
            continue;
        }
        seen.insert(key.clone(), entry.income);
        let interval = out
            .interval_by_symbol
            .entry(entry.symbol.clone())
            .or_insert(SymbolInterval {
                first_time_ms: entry.time,
                last_time_ms: entry.time,
                rows: 0,
            });
        interval.first_time_ms = interval.first_time_ms.min(entry.time);
        interval.last_time_ms = interval.last_time_ms.max(entry.time);
        interval.rows += 1;
        out.accepted.push(entry);
    }
    out
}
