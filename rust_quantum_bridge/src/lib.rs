use pyo3::prelude::*;

mod memory;
mod engine_loop;
mod math;
mod tensors;
mod binance_ws;
mod binance_rest;

use memory::{QuantumQueue, QuantumStateStore};
use engine_loop::QuantumEngine;
use math::QuantumMath;
use tensors::QuantumTensors;

/// A Python module implemented in Rust.
#[pymodule]
fn quantum_bridge(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<QuantumQueue>()?;
    m.add_class::<QuantumEngine>()?;
    m.add_class::<QuantumStateStore>()?;
    m.add_class::<QuantumMath>()?;
    m.add_class::<QuantumTensors>()?;
    m.add_function(wrap_pyfunction!(binance_ws::start_binance_websocket, m)?)?;
    m.add_function(wrap_pyfunction!(binance_ws::get_arena_price, m)?)?;
    m.add_function(wrap_pyfunction!(binance_rest::place_order_sync, m)?)?;
    Ok(())
}
