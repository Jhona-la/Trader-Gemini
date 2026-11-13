//! CAPITAL-UNICO V4G: el saldo dimensionable (`GlobalArena::unified_capital`, lo que lee Risk)
//! deja de ser un acumulador en el arena vivo y se DERIVA:
//!
//! dimensionable = ultimo saldo de la cuenta aceptado por el puente (era y sesion vigentes)
//!                 - comisiones de las reservas que el exchange aun no refleja.
//!
//! El libro vive dentro de la puerta que ya serializa la publicacion de entradas
//! (`QuantumConfig::entry_publication_gate`): el nucleo ya la tiene tomada cuando reserva, asi que
//! no hay candado nuevo en el camino de los ticks. Cada operacion vuelve a calcular el valor entero
//! (no por incrementos): sin reservas es exactamente el saldo aceptado.
//!
//! - Reservar anade la reserva; cancelar la QUITA (no suma dinero).
//! - Ejecutar baja la reserva a la comision de lo ejecutado y la marca con la `T` del exchange del
//!   ultimo fill. Un saldo aceptado con marca igual o mayor ya la incluye y la quita (sin doble
//!   resta). Con la marca del fill desconocida, la quita el primer saldo con marca posterior a la
//!   vigente al confirmar.
//! - El saldo solo lo entrega el publicador del puente ([`LibroDimensionable::saldo_aceptado`]).
//!
//! Sin activar ([`LibroDimensionable::activar`]: backtest, replay, simuladores, pruebas) el arena
//! sigue siendo el acumulador de siempre: las mismas sumas atomicas que antes.

use crate::state::GlobalArena;
use std::sync::atomic::Ordering;

/// Identidad de una reserva: la ranura exacta y su generacion.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ClaveReserva {
    pub coin_id: usize,
    pub slot: usize,
    pub generation: u64,
}

#[derive(Debug, Clone, Copy, PartialEq)]
enum Estado {
    /// Sin ejecucion conocida: el exchange no la refleja.
    Viva,
    /// Ejecutada. `marca`: `T` del exchange del ultimo fill. `tras`: marca del saldo vigente al
    /// confirmar (solo cuenta si `marca` es desconocida).
    Ejecutada { marca: Option<u64>, tras: Option<u64> },
}

#[derive(Debug, Clone, Copy, PartialEq)]
struct Reserva {
    clave: ClaveReserva,
    comision: f64,
    estado: Estado,
}

impl Reserva {
    fn reflejada(&self, marca_saldo: Option<u64>) -> bool {
        match (self.estado, marca_saldo) {
            (Estado::Viva, _) | (_, None) => false,
            (Estado::Ejecutada { marca: Some(t), .. }, Some(m)) => m >= t,
            (Estado::Ejecutada { marca: None, tras: Some(p) }, Some(m)) => m > p,
            (Estado::Ejecutada { marca: None, tras: None }, Some(_)) => true,
        }
    }
}

/// El libro del dimensionable. Se usa siempre bajo `entry_publication_gate`.
#[derive(Debug, Default)]
pub struct LibroDimensionable {
    derivado: bool,
    saldo: Option<f64>,
    marca: Option<u64>,
    reservas: Vec<Reserva>,
}

impl LibroDimensionable {
    /// El host lo activa en el arena vivo antes de vincular el publicador del puente. Desde ahi el
    /// cierre local y la reconciliacion dejan de escribir el campo (lo trae el saldo del exchange).
    pub fn activar(&mut self, arena: &GlobalArena) {
        self.derivado = true;
        arena.config.dimensionable_derivado.store(true, Ordering::Release);
    }

    pub fn derivado(&self) -> bool {
        self.derivado
    }

    /// Ultimo saldo aceptado y su marca.
    pub fn saldo(&self) -> Option<(f64, Option<u64>)> {
        self.saldo.map(|s| (s, self.marca))
    }

    /// Comisiones reservadas que el exchange aun no refleja.
    pub fn reservado(&self) -> f64 {
        self.reservas.iter().map(|r| r.comision).sum()
    }

    pub fn reservas(&self) -> usize {
        self.reservas.len()
    }

    /// saldo aceptado - reservado, si hay saldo.
    pub fn dimensionable(&self) -> Option<f64> {
        self.saldo.map(|s| s - self.reservado())
    }

    fn publicar(&self, arena: &GlobalArena) {
        if let Some(d) = self.dimensionable() {
            arena.unified_capital.store(d, Ordering::Relaxed);
        }
    }

    /// El publicador del puente: saldo aceptado (era y sesion vigentes) con su marca del exchange.
    /// Quita las ejecuciones que ese saldo ya incluye y publica el dimensionable.
    pub fn saldo_aceptado(&mut self, arena: &GlobalArena, saldo: f64, marca: Option<u64>) {
        self.saldo = Some(saldo);
        self.marca = marca;
        self.reservas.retain(|r| !r.reflejada(marca));
        if self.derivado {
            self.publicar(arena);
        }
    }

    /// Reserva de una entrada (el nucleo, al abrir la ranura).
    pub fn reservar(&mut self, arena: &GlobalArena, clave: ClaveReserva, comision: f64) {
        if !self.derivado {
            arena.unified_capital.fetch_add(-comision, Ordering::Relaxed);
            return;
        }
        if comision.is_finite() && comision != 0.0 {
            self.reservas.push(Reserva { clave, comision, estado: Estado::Viva });
        }
        self.publicar(arena);
    }

    /// Cancelacion sin ejecucion: quita la reserva, no suma dinero.
    pub fn cancelar(&mut self, arena: &GlobalArena, clave: ClaveReserva, comision: f64) {
        if !self.derivado {
            if comision.is_finite() && comision != 0.0 {
                arena.unified_capital.fetch_add(comision, Ordering::AcqRel);
            }
            return;
        }
        if let Some(i) = self.reservas.iter().position(|r| r.clave == clave && r.estado == Estado::Viva) {
            self.reservas.remove(i);
        }
        self.publicar(arena);
    }

    /// Ejecucion confirmada: `devuelta` es la comision de lo NO ejecutado; `marca_fill`, la `T`
    /// del exchange del ultimo fill.
    pub fn ejecutar(&mut self, arena: &GlobalArena, clave: ClaveReserva, devuelta: f64, marca_fill: Option<u64>) {
        if !self.derivado {
            if devuelta.is_finite() && devuelta != 0.0 {
                arena.unified_capital.fetch_add(devuelta, Ordering::AcqRel);
            }
            return;
        }
        let marca_saldo = self.marca;
        if let Some(i) = self.reservas.iter().position(|r| r.clave == clave && r.estado == Estado::Viva) {
            let r = &mut self.reservas[i];
            if devuelta.is_finite() {
                r.comision -= devuelta;
            }
            r.estado = Estado::Ejecutada { marca: marca_fill, tras: marca_saldo };
            if r.reflejada(marca_saldo) && marca_fill.is_some() {
                self.reservas.remove(i);
            }
        }
        self.publicar(arena);
    }
}
