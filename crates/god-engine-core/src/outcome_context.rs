//! Scope of LOCAL close estimates, not proof of exchange settlement.
//! Market horizons remain continuous; simulation provenance is not a strategy.

#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum OutcomeContext {
    /// Research/backtest/paper evaluation: learn in the instance, never publish
    /// close outcomes to shared training files, the live bus or trauma storage.
    #[default]
    IsolatedSimulation,
    /// Host connected to an execution provider. Entry confirmation is a
    /// minimum eligibility condition, NOT evidence that the exit was filled.
    /// Full settlement attribution still belongs to an execution ledger.
    ExchangeLocalEstimate,
}

impl OutcomeContext {
    pub fn permits_local_learning(self, entry_confirmed: bool) -> bool {
        self == Self::IsolatedSimulation || entry_confirmed
    }

    pub fn permits_shared_estimate_outputs(self, entry_confirmed: bool) -> bool {
        self == Self::ExchangeLocalEstimate && entry_confirmed
    }
}
