use pyo3::prelude::*;
use pyo3::types::PyDict;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;
use tokio::runtime::Runtime;

#[pyclass]
pub struct QuantumEngine {
    is_running: Arc<AtomicBool>,
    runtime: Runtime,
}

#[pymethods]
impl QuantumEngine {
    #[new]
    pub fn new() -> PyResult<Self> {
        let runtime = tokio::runtime::Builder::new_multi_thread()
            .worker_threads(4) // Optimization for 8-core CPU
            .enable_all()
            .build()
            .unwrap();

        Ok(QuantumEngine {
            is_running: Arc::new(AtomicBool::new(false)),
            runtime,
        })
    }

    pub fn start(&self) -> PyResult<()> {
        self.is_running.store(true, Ordering::SeqCst);
        Ok(())
    }

    pub fn stop(&self) -> PyResult<()> {
        self.is_running.store(false, Ordering::SeqCst);
        Ok(())
    }
    
    pub fn is_running(&self) -> PyResult<bool> {
        Ok(self.is_running.load(Ordering::SeqCst))
    }
}
