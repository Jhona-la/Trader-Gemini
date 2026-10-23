use std::collections::HashSet;
use std::fmt;

/// Observed rates for one symbol, expressed as signed fractions of notional.
/// Zero fees and rebates are evidence; neither is replaced with a default.
#[derive(Clone, Debug, PartialEq)]
pub struct SymbolFees {
    pub symbol: String,
    pub maker: f64,
    pub taker: f64,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum FeeScheduleError {
    EmptySymbol { index: usize },
    NonFiniteRate { symbol: String },
    DuplicateSymbol { symbol: String },
}

impl fmt::Display for FeeScheduleError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::EmptySymbol { index } => write!(f, "empty fee symbol at slot {index}"),
            Self::NonFiniteRate { symbol } => write!(f, "nonfinite fee rate for {symbol}"),
            Self::DuplicateSymbol { symbol } => write!(f, "duplicate fee symbol {symbol}"),
        }
    }
}

impl std::error::Error for FeeScheduleError {}

/// One immutable publication, ordered exactly like the active universe.
///
/// `generation` identifies the publisher's session. Publishing/clearing this
/// book and deciding which session may publish remain the caller's duty.
#[derive(Clone, Debug)]
pub struct FeeSchedule {
    generation: u64,
    rows: Vec<SymbolFees>,
}

impl FeeSchedule {
    pub fn new(generation: u64, rows: Vec<SymbolFees>) -> Result<Self, FeeScheduleError> {
        let mut symbols = HashSet::with_capacity(rows.len());
        for (index, row) in rows.iter().enumerate() {
            if row.symbol.is_empty() {
                return Err(FeeScheduleError::EmptySymbol { index });
            }
            if !row.maker.is_finite() || !row.taker.is_finite() {
                return Err(FeeScheduleError::NonFiniteRate {
                    symbol: row.symbol.clone(),
                });
            }
            if !symbols.insert(row.symbol.to_ascii_uppercase()) {
                return Err(FeeScheduleError::DuplicateSymbol {
                    symbol: row.symbol.clone(),
                });
            }
        }
        Ok(Self { generation, rows })
    }

    pub fn generation(&self) -> u64 {
        self.generation
    }

    pub fn rows(&self) -> &[SymbolFees] {
        &self.rows
    }

    /// Direct slot lookup without searching/cloning the universe or its rows.
    /// A reordered or missing symbol invalidates this slot until republished.
    #[inline]
    pub fn rates_for(&self, coin_id: usize) -> Option<(f64, f64)> {
        let row = self.rows.get(coin_id)?;
        crate::symbol_registry::slot_es_simbolo(coin_id, &row.symbol)
            .then_some((row.maker, row.taker))
    }
}
