//! Identity for one local proposal, not proof of an exchange fill or a durable ledger.
use quantum_arena::{position::PositionTransitionError, symbol_registry, GlobalArena};
use std::sync::atomic::Ordering;

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct EntryReservation {
    pub coin_id: usize,
    pub slot: usize,
    pub generation: u64,
    pub symbol: String,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ReservationError {
    InvalidCoin,
    InvalidSlot,
    SymbolMismatch,
    Position(PositionTransitionError),
}

impl EntryReservation {
    fn position<'a>(
        &self,
        arena: &'a GlobalArena,
    ) -> Result<&'a quantum_arena::position::Position, ReservationError> {
        let coin = arena
            .coins
            .get(self.coin_id)
            .ok_or(ReservationError::InvalidCoin)?;
        let slot = coin
            .positions
            .slots()
            .get(self.slot)
            .copied()
            .ok_or(ReservationError::InvalidSlot)?;
        if !symbol_registry::try_symbol(self.coin_id)
            .is_some_and(|s| s.eq_ignore_ascii_case(&self.symbol))
        {
            return Err(ReservationError::SymbolMismatch);
        }
        Ok(slot)
    }

    /// Release only this rejected reservation, at most once. No exchange I/O.
    /// Slot transition is serialized; portfolio atomics are not a durable transaction.
    pub fn cancel(&self, arena: &GlobalArena) -> Result<(), ReservationError> {
        let (_, _, _, margin, fee) = self
            .position(arena)?
            .cancel_unconfirmed_generation(self.generation)
            .map_err(ReservationError::Position)?;
        if margin > 0.0 {
            arena.used_margin.fetch_sub(margin, Ordering::AcqRel);
        }
        if fee > 0.0 {
            arena.unified_capital.fetch_add(fee, Ordering::AcqRel);
        }
        Ok(())
    }

    /// Bind caller-supplied confirmation to the exact slot, not a fixed legacy slot.
    pub fn confirm(&self, arena: &GlobalArena) -> Result<(), ReservationError> {
        self.position(arena)?
            .confirm_generation(self.generation)
            .map_err(ReservationError::Position)
    }
}
