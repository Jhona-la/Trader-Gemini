use std::sync::atomic::{AtomicU64, Ordering};

/// Un envoltorio transparente y sin locks para flotantes de 64 bits.
/// Usa la transmutación `to_bits` y `from_bits` de la especificación IEEE-754
/// para almacenar floats en `AtomicU64` de forma segura.
/// Alineado a 8 bytes nativamente.
#[repr(transparent)]
pub struct AtomicF64(AtomicU64);

#[inline(always)]
fn canonical_bits(val: f64) -> u64 {
    if val == 0.0 {
        0.0f64.to_bits()
    } else if val.is_nan() {
        f64::NAN.to_bits()
    } else {
        val.to_bits()
    }
}

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
        self.0.store(canonical_bits(val), order);
    }

    #[inline(always)]
    pub fn swap(&self, val: f64, order: Ordering) -> f64 {
        f64::from_bits(self.0.swap(canonical_bits(val), order))
    }

    /// Compare-And-Swap para flotantes. Utiliza repetición (spin)
    /// si otro hilo modificó el valor entre la lectura y la escritura.
    #[inline(always)]
    pub fn fetch_add(&self, val: f64, order: Ordering) -> f64 {
        let mut current = self.0.load(Ordering::Relaxed);
        loop {
            let current_f = f64::from_bits(current);
            let new_val = current_f + val;
            let new_bits = canonical_bits(new_val);
            match self
                .0
                .compare_exchange_weak(current, new_bits, order, Ordering::Relaxed)
            {
                Ok(v) => return f64::from_bits(v),
                Err(v) => current = v,
            }
        }
    }

    #[inline(always)]
    pub fn compare_exchange(
        &self,
        current: f64,
        new: f64,
        success: Ordering,
        failure: Ordering,
    ) -> Result<f64, f64> {
        let current_bits = canonical_bits(current);
        let new_bits = canonical_bits(new);
        match self
            .0
            .compare_exchange(current_bits, new_bits, success, failure)
        {
            Ok(v) => Ok(f64::from_bits(v)),
            Err(v) => Err(f64::from_bits(v)),
        }
    }

    #[inline(always)]
    pub fn compare_exchange_weak(
        &self,
        current: f64,
        new: f64,
        success: Ordering,
        failure: Ordering,
    ) -> Result<f64, f64> {
        let current_bits = canonical_bits(current);
        let new_bits = canonical_bits(new);
        match self
            .0
            .compare_exchange_weak(current_bits, new_bits, success, failure)
        {
            Ok(v) => Ok(f64::from_bits(v)),
            Err(v) => Err(f64::from_bits(v)),
        }
    }

    #[inline(always)]
    pub fn fetch_update<F>(
        &self,
        set_order: Ordering,
        fetch_order: Ordering,
        mut f: F,
    ) -> Result<f64, f64>
    where
        F: FnMut(f64) -> Option<f64>,
    {
        // FIX #1407: Sanitizar failure ordering para prevenir pánicos de CPU por ordering inválido
        let fail_order = match fetch_order {
            Ordering::Release => Ordering::Relaxed,
            Ordering::AcqRel => Ordering::Acquire,
            other => other,
        };
        let mut prev = self.load(fetch_order);
        while let Some(next) = f(prev) {
            match self.compare_exchange_weak(prev, next, set_order, fail_order) {
                Ok(x) => return Ok(x),
                Err(next_prev) => prev = next_prev,
            }
        }
        Err(prev)
    }
}

impl Default for AtomicF64 {
    fn default() -> Self {
        Self::new(0.0)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_atomic_f64_load_store_swap() {
        let atomic = AtomicF64::new(13.0);
        assert_eq!(atomic.load(Ordering::SeqCst), 13.0);

        atomic.store(52.0, Ordering::SeqCst);
        assert_eq!(atomic.load(Ordering::SeqCst), 52.0);

        let old = atomic.swap(100.0, Ordering::SeqCst);
        assert_eq!(old, 52.0);
        assert_eq!(atomic.load(Ordering::SeqCst), 100.0);
    }

    #[test]
    fn test_atomic_f64_fetch_add_and_nan_canonical() {
        let atomic = AtomicF64::new(10.0);
        let prev = atomic.fetch_add(5.5, Ordering::SeqCst);
        assert_eq!(prev, 10.0);
        assert!((atomic.load(Ordering::SeqCst) - 15.5).abs() < 1e-9);

        // NaN handling
        atomic.store(f64::NAN, Ordering::SeqCst);
        assert!(atomic.load(Ordering::SeqCst).is_nan());
    }

    #[test]
    fn test_atomic_f64_compare_exchange_and_fetch_update() {
        let atomic = AtomicF64::new(20.0);

        // Successful CAS
        let res_ok = atomic.compare_exchange(20.0, 30.0, Ordering::SeqCst, Ordering::SeqCst);
        assert_eq!(res_ok, Ok(20.0));
        assert_eq!(atomic.load(Ordering::SeqCst), 30.0);

        // Failed CAS
        let res_err = atomic.compare_exchange(20.0, 40.0, Ordering::SeqCst, Ordering::SeqCst);
        assert_eq!(res_err, Err(30.0));
        assert_eq!(atomic.load(Ordering::SeqCst), 30.0);

        // Fetch Update
        let update_res = atomic.fetch_update(Ordering::SeqCst, Ordering::SeqCst, |v| Some(v * 2.0));
        assert_eq!(update_res, Ok(30.0));
        assert_eq!(atomic.load(Ordering::SeqCst), 60.0);
    }
}
