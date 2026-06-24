use pyo3::prelude::*;
use crossbeam::queue::ArrayQueue;
use std::sync::Arc;

/// A lock-free, bounded priority queue optimized for nanosecond latencies.
/// This replaces Python's queue.PriorityQueue which suffers from GIL contention.
#[pyclass]
pub struct QuantumQueue {
    high_priority: Arc<ArrayQueue<PyObject>>,
    normal_priority: Arc<ArrayQueue<PyObject>>,
    low_priority: Arc<ArrayQueue<PyObject>>,
}

#[pymethods]
impl QuantumQueue {
    #[new]
    pub fn new(capacity: usize) -> Self {
        QuantumQueue {
            high_priority: Arc::new(ArrayQueue::new(capacity / 4)),
            normal_priority: Arc::new(ArrayQueue::new(capacity / 2)),
            low_priority: Arc::new(ArrayQueue::new(capacity / 4)),
        }
    }

    pub fn put(&self, item: PyObject, priority: i32) -> PyResult<()> {
        let q = match priority {
            0..=1 => &self.high_priority,
            2..=5 => &self.normal_priority,
            _ => &self.low_priority,
        };
        
        let _ = q.push(item); // Ignore if full to match bounded behavior
        Ok(())
    }

    pub fn get(&self, _py: Python) -> PyResult<Option<PyObject>> {
        if let Some(item) = self.high_priority.pop() {
            return Ok(Some(item));
        }
        if let Some(item) = self.normal_priority.pop() {
            return Ok(Some(item));
        }
        if let Some(item) = self.low_priority.pop() {
            return Ok(Some(item));
        }
        Ok(None)
    }

    pub fn qsize(&self) -> PyResult<usize> {
        Ok(self.high_priority.len() + self.normal_priority.len() + self.low_priority.len())
    }
}

use std::collections::HashMap;
use parking_lot::RwLock;

/// FFI Native State Segregation
#[pyclass]
pub struct QuantumStateStore {
    store: Arc<RwLock<HashMap<String, HashMap<String, f64>>>>,
}

#[pymethods]
impl QuantumStateStore {
    #[new]
    pub fn new() -> Self {
        QuantumStateStore {
            store: Arc::new(RwLock::new(HashMap::new())),
        }
    }

    pub fn set_metric(&self, symbol: String, key: String, value: f64) -> PyResult<()> {
        let mut w_store = self.store.write();
        let sym_map = w_store.entry(symbol).or_insert_with(HashMap::new);
        sym_map.insert(key, value);
        Ok(())
    }

    pub fn get_metric(&self, symbol: String, key: String) -> PyResult<Option<f64>> {
        let r_store = self.store.read();
        if let Some(sym_map) = r_store.get(&symbol) {
            Ok(sym_map.get(&key).copied())
        } else {
            Ok(None)
        }
    }
}
