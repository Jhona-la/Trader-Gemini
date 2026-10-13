//! D4 (ADR-0016, LC-F4-19) — LA SONDA ES EXPLORACIÓN PRESUPUESTADA.
//!
//! Una moneda sin historial suficiente no tiene evidencia con la que juzgar
//! el valor esperado de una orden. Sus primeras órdenes son SONDAS: se abren
//! para medir, no porque haya ventaja demostrada. ADR-0016 (D4) decidió su
//! presupuesto:
//!
//! - una sola sonda abierta en toda la cartera: abrir varias a la vez
//!   multiplica un riesgo que nadie ha medido;
//! - tamaño: la orden mínima del símbolo (D-750);
//! - como máximo [`CIERRES_DE_SONDA`] por moneda;
//! - sus pérdidas cuentan en el drawdown de D3, sin contabilidad aparte.
//!
//! Antes había dos puertas, las dos por moneda: la fase de sonda del riesgo
//! (`trade_count < 5`, exención del EV) y la sonda del núcleo
//! (`arranque_frio_sin_roster`: sin modelo, `trade_count == 0` y sin
//! posición en la moneda, exención del ML-GATE). Ninguna miraba la cartera,
//! así que cada moneda podía abrir su sonda a la vez. Y como `trade_count`
//! no se persiste, tras cada reinicio todas volvían a la fase de sonda.
//! Las dos llaman ahora a la misma función, [`admite_sonda`]:
//! - el veto del riesgo decide: rechaza con `sonda_ocupada` antes de
//!   dimensionar;
//! - la exención del ML-GATE en el núcleo ([`sonda_sin_modelo`]) la consulta
//!   también, aunque es redundante por construcción: con una sonda abierta el
//!   riesgo ya rechazó y el núcleo nunca llega al ML-GATE con una orden viva.
//!   Se conserva para que la exención no pueda contradecir a D4 si el orden
//!   de las puertas cambia.
//!
//! Reinicio. Una moneda en fase de sonda es la que tiene menos de
//! [`CIERRES_DE_SONDA`] cierres en este proceso. Tras un reinicio todas
//! vuelven a 0, así que una posición adoptada del exchange cuenta como
//! sonda abierta. La cartera no abre otra sonda hasta que esa posición
//! cierra. La regla de «una sola» se conserva sin estado persistido. Lo
//! que sí se pierde al reiniciar es el recuento por moneda; persistirlo es
//! del estado aprendido (C-F8).

use quantum_arena::GlobalArena;
use std::sync::atomic::Ordering;

/// Cierres por debajo de los cuales una moneda sigue en fase de sonda.
/// ADR-0016 D4: «máximo 5 por moneda (vigente)»; es el umbral que ya usaba
/// la exención del EV.
pub const CIERRES_DE_SONDA: usize = 5;

/// ¿La moneda está en fase de sonda? Una moneda fuera del arena no lo está:
/// no puede abrir nada.
#[inline]
pub fn en_fase_de_sonda(arena: &GlobalArena, coin_id: usize) -> bool {
    arena
        .coins
        .get(coin_id)
        .is_some_and(|c| c.metrics.trade_count.load(Ordering::Relaxed) < CIERRES_DE_SONDA)
}

/// ¿Hay una sonda abierta en la cartera? Lo es cualquier posición abierta en
/// una moneda que sigue en fase de sonda, también si se adoptó del exchange
/// tras un reinicio.
#[inline]
pub fn hay_sonda_abierta(arena: &GlobalArena) -> bool {
    arena.coins.iter().any(|c| {
        c.metrics.trade_count.load(Ordering::Relaxed) < CIERRES_DE_SONDA
            && c.positions.is_any_open()
    })
}

/// ¿Puede la moneda abrir una orden ahora? Fuera de la fase de sonda,
/// siempre (lo deciden los demás vetos). En fase de sonda, sólo si no hay
/// ya una sonda abierta en la cartera.
#[inline]
pub fn admite_sonda(arena: &GlobalArena, coin_id: usize) -> bool {
    !en_fase_de_sonda(arena, coin_id) || !hay_sonda_abierta(arena)
}

/// Exención del ML-GATE para la primera orden de una moneda SIN modelo del
/// roster (Ola XLI·D1): con 0 cierres, esa orden es la sonda que genera la
/// evidencia que el gate exige; sin ella el motor no arrancaría. Con modelo
/// decide el lift del bosque, no la sonda. Y sólo si la cartera admite una
/// sonda ([`admite_sonda`], la misma función del veto del riesgo).
#[inline]
pub fn sonda_sin_modelo(arena: &GlobalArena, coin_id: usize, tiene_modelo_roster: bool) -> bool {
    !tiene_modelo_roster
        && arena
            .coins
            .get(coin_id)
            .is_some_and(|c| c.metrics.trade_count.load(Ordering::Relaxed) == 0)
        && admite_sonda(arena, coin_id)
}
