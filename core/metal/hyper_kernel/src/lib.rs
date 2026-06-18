use pyo3::prelude::*;

#[repr(C, align(16))]
pub struct Tensor10D {
    pub data: [f32; 10],
}

/// Mutates the given pointer in place by multiplying all elements by 2.0
/// This is used to test that Python and Rust share the same memory safely without corruption.
#[pyfunction]
fn mutate_canary(ptr: usize) {
    let tensor = unsafe { &mut *(ptr as *mut Tensor10D) };
    for i in 0..10 {
        tensor.data[i] *= 2.0;
    }
}

/// Calculates the new physics features in-place inside the raw memory buffer.
#[pyfunction]
fn calculate_physics(ptr: usize) {
    let tensor = unsafe { &mut *(ptr as *mut Tensor10D) };
    
    // Dim 3: OBI d2 (Standard Scaling Mock)
    // Here we apply an O(1) transformation logic
    tensor.data[3] = tensor.data[3] * 0.5;
    
    // Dim 4: Kyle's Lambda (Log Transform para asimetría en cola gruesa)
    tensor.data[4] = (1.0 + tensor.data[4].max(0.0)).ln();
    
    // Dim 5: Shannon Entropy de flujos (Linear Scaling / Clamping)
    tensor.data[5] = tensor.data[5].clamp(0.0, 1.0);
}

#[pymodule]
fn hyper_kernel(_py: Python, m: &PyModule) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(mutate_canary, m)?)?;
    m.add_function(wrap_pyfunction!(calculate_physics, m)?)?;
    Ok(())
}
