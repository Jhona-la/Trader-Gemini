use std::sync::atomic::{AtomicU64, Ordering};

/// Un envoltorio transparente y sin locks para flotantes de 64 bits.
/// Usa la transmutación `to_bits` y `from_bits` de la especificación IEEE-754
/// para almacenar floats en `AtomicU64` de forma segura.
/// Alineado a 8 bytes nativamente.
#[repr(transparent)]
pub struct AtomicF64(AtomicU64);

impl AtomicF64 {
    #[inline(always)]
    pub const fn new(val: f64) -> Self {
        Self(AtomicU64::new(val.to_bits()))
    }

    #[inline(always)]
    pub fn load(&self, order: Ordering) -> f64 {
        f64::from_bits(self.0.load(order))
    }

    #[inline(always)]
    pub fn store(&self, val: f64, order: Ordering) {
        self.0.store(val.to_bits(), order);
    }

    #[inline(always)]
    pub fn swap(&self, val: f64, order: Ordering) -> f64 {
        f64::from_bits(self.0.swap(val.to_bits(), order))
    }

    /// Compare-And-Swap para flotantes. Utiliza repetición (spin)
    /// si otro hilo modificó el valor entre la lectura y la escritura.
    #[inline(always)]
    pub fn fetch_add(&self, val: f64, order: Ordering) -> f64 {
        let mut current = self.0.load(Ordering::Relaxed);
        loop {
            let current_f = f64::from_bits(current);
            let new_val = current_f + val;
            match self.0.compare_exchange_weak(
                current,
                new_val.to_bits(),
                order,
                Ordering::Relaxed,
            ) {
                Ok(v) => return f64::from_bits(v),
                Err(v) => current = v,
            }
        }
    }
}

impl Default for AtomicF64 {
    fn default() -> Self {
        Self::new(0.0)
    }
}
