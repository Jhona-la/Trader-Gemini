//! Identity for one local proposal, not proof of an exchange fill or a durable ledger.
use quantum_arena::dimensionable::ClaveReserva;
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
        if !symbol_registry::slot_es_simbolo(self.coin_id, &self.symbol) {
            return Err(ReservationError::SymbolMismatch);
        }
        Ok(slot)
    }

    /// CAPITAL-UNICO V4H: la clave de esta reserva en el libro del dimensionable.
    pub fn clave(&self) -> ClaveReserva {
        ClaveReserva { coin_id: self.coin_id, slot: self.slot, generation: self.generation }
    }

    /// Release only this rejected reservation, at most once. No exchange I/O.
    /// Slot transition is serialized; portfolio atomics are not a durable transaction.
    ///
    /// CAPITAL-UNICO V4G (INV-C): la comision pasa por el libro del dimensionable, bajo la
    /// puerta de publicacion: con el dimensionable derivado se QUITA la reserva (no se suma
    /// dinero sobre un saldo que quiza ya la incluia); sin activar, se devuelve como siempre.
    /// Quien llama no tiene la puerta tomada (el nucleo compensa con su propia guarda).
    pub fn cancel(&self, arena: &GlobalArena) -> Result<(), ReservationError> {
        let (_, _, _, margin, fee) = self
            .position(arena)?
            .cancel_unconfirmed_generation(self.generation)
            .map_err(ReservationError::Position)?;
        if margin > 0.0 {
            arena.used_margin.fetch_sub(margin, Ordering::AcqRel);
        }
        arena
            .config
            .entry_publication_gate
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .cancelar(arena, self.clave(), fee);
        Ok(())
    }

    /// CL-41b: la reserva retiene el margen que retendrá el exchange con el
    /// apalancamiento de envío; `used_margin` recibe la diferencia (CL-41c:
    /// bajo el cerrojo de la ranura y antes de publicarla). Así el margen
    /// libre de la siguiente entrada es el del exchange.
    pub fn reajustar_margen(&self, arena: &GlobalArena, nuevo_margen: f64) -> Result<(), ReservationError> {
        self.position(arena)?
            .reajustar_margen_generation(self.generation, nuevo_margen, &arena.used_margin)
            .map_err(ReservationError::Position)?;
        Ok(())
    }

    /// CL-45: confirma la reserva con la cantidad que el exchange ejecutó.
    /// Con `Some(q)` y q menor que la reservada, la ranura se escala a q y la
    /// comisión de entrada no pagada vuelve al capital. Con `None` (paper, o
    /// una ruta cuya respuesta no trae la ejecución) es `confirm`.
    pub fn confirmar_llenado(
        &self,
        arena: &GlobalArena,
        ejecutada: Option<f64>,
    ) -> Result<(), ReservationError> {
        self.confirmar_llenado_con_marca(arena, ejecutada, None)
    }

    /// CAPITAL-UNICO V4G: como [`confirmar_llenado`](Self::confirmar_llenado), con la marca del
    /// exchange (`T`) del ultimo fill. Con el dimensionable derivado la reserva queda reducida a
    /// la comision de lo ejecutado y la quita el primer saldo aceptado con marca igual o mayor
    /// (ya la incluye: no hay doble resta). Sin activar, la parte no pagada vuelve al capital,
    /// como antes.
    pub fn confirmar_llenado_con_marca(
        &self,
        arena: &GlobalArena,
        ejecutada: Option<f64>,
        marca_fill: Option<u64>,
    ) -> Result<(), ReservationError> {
        let devuelta = match ejecutada {
            None => {
                self.position(arena)?
                    .confirm_generation(self.generation)
                    .map_err(ReservationError::Position)?;
                0.0
            }
            Some(q) => self
                .position(arena)?
                .confirmar_llenado_generation(self.generation, q, &arena.used_margin)
                .map_err(ReservationError::Position)?,
        };
        arena
            .config
            .entry_publication_gate
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .ejecutar(arena, self.clave(), devuelta, marca_fill);
        Ok(())
    }

    /// Bind caller-supplied confirmation to the exact slot, not a fixed legacy slot.
    pub fn confirm(&self, arena: &GlobalArena) -> Result<(), ReservationError> {
        self.confirmar_llenado_con_marca(arena, None, None)
    }
}
