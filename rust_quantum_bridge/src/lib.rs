use pyo3::prelude::*;

mod memory;
mod engine_loop;

use memory::{QuantumQueue, QuantumStateStore};
use engine_loop::QuantumEngine;

/// A Python module implemented in Rust.
#[pymodule]
fn quantum_bridge(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<QuantumQueue>()?;
    m.add_class::<QuantumEngine>()?;
    m.add_class::<QuantumStateStore>()?;
    Ok(())
}
