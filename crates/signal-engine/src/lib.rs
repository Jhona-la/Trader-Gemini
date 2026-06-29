pub mod scalp;
pub mod swing;

#[derive(Debug, Clone, Copy, PartialEq)]
pub enum SignalType {
    Long,
    Short,
    Flat,
}

/// Intención de orden. 
/// Es evaluada por el risk-engine para convertirse (o no) en una orden real.
#[derive(Debug, Clone, Copy)]
pub struct SignalIntent {
    pub signal: SignalType,
    pub confidence: f64,
}

impl SignalIntent {
    #[inline(always)]
    pub fn flat() -> Self {
        Self {
            signal: SignalType::Flat,
            confidence: 0.0,
        }
    }
}
