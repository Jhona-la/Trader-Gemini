use std::sync::atomic::{AtomicU64, Ordering};

/// Un envoltorio transparente y sin locks para flotantes de 64 bits.
/// Usa la transmutación `to_bits` y `from_bits` de la especificación IEEE-754
/// para almacenar floats en `AtomicU64` de forma segura.
/// Alineado a 8 bytes nativamente.
#[repr(transparent)]
pub struct AtomicF64(AtomicU64);

#[inline(always)]
const fn canonical_bits(val: f64) -> u64 {
    if val == 0.0 {
        0.0f64.to_bits()
    } else if val.is_nan() {
        f64::NAN.to_bits()
    } else {
        val.to_bits()
    }
}

impl AtomicF64 {
    /// CL-154 (hallazgo de Línea G, `docs/audit/linea_g/hallazgos.tsv:10`):
    /// `new` canoniza como `store`, `swap` y los CAS. Antes guardaba los bits
    /// tal cual: un átomo nacido con −0,0 o con un NaN no canónico (0/0 en
    /// x86 da 0xFFF8…) dejaba `fetch_update` en un bucle sin fin, porque
    /// `compare_exchange_weak` canoniza el valor esperado y no coincidía
    /// nunca con los bits guardados. Para cualquier otro valor no cambia nada.
    #[inline(always)]
    pub const fn new(val: f64) -> Self {
        Self(AtomicU64::new(canonical_bits(val)))
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

    /// MOD6/8-010 (INFORME DECIMOCUARTO): resta atómica RMW. Reemplaza el
    /// patrón enfermo `load → (cur - val) → store`, que NO es atómico y
    /// perdía actualizaciones cuando ≥3 hilos mutaban `used_margin`
    /// simultáneamente (cierre del core, rollback async, reconciliación,
    /// replay). Mismo CAS-loop de `fetch_add`.
    #[inline(always)]
    pub fn fetch_sub(&self, val: f64, order: Ordering) -> f64 {
        let mut current = self.0.load(Ordering::Relaxed);
        loop {
            let current_f = f64::from_bits(current);
            let new_val = current_f - val;
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

    /// CL-154: un átomo nacido con −0,0 o con un NaN no canónico queda en la
    /// forma canónica, y `fetch_update` termina. Los bits se comprueban
    /// ANTES de `fetch_update` para que, sin el arreglo, la prueba falle en
    /// vez de colgarse.
    #[test]
    fn cl154_new_canoniza_y_fetch_update_termina() {
        let nan_no_canonico = f64::from_bits(0xFFF8_0000_0000_0000);
        assert!(nan_no_canonico.is_nan());
        assert_ne!(nan_no_canonico.to_bits(), f64::NAN.to_bits());

        let cero_negativo = AtomicF64::new(-0.0);
        assert_eq!(
            cero_negativo.load(Ordering::SeqCst).to_bits(),
            0.0f64.to_bits()
        );
        let nan = AtomicF64::new(nan_no_canonico);
        assert_eq!(nan.load(Ordering::SeqCst).to_bits(), f64::NAN.to_bits());
        static EN_ESTATICO: AtomicF64 = AtomicF64::new(-0.0);
        assert_eq!(
            EN_ESTATICO.load(Ordering::SeqCst).to_bits(),
            0.0f64.to_bits()
        );

        assert_eq!(
            cero_negativo.fetch_update(Ordering::SeqCst, Ordering::SeqCst, |v| Some(v + 1.0)),
            Ok(0.0)
        );
        assert_eq!(cero_negativo.load(Ordering::SeqCst), 1.0);
        let previo = nan.fetch_update(Ordering::SeqCst, Ordering::SeqCst, |_| Some(2.0));
        assert!(matches!(previo, Ok(v) if v.is_nan()));
        assert_eq!(nan.load(Ordering::SeqCst), 2.0);

        // Los valores normales conservan sus bits.
        for v in [
            -0.01,
            13.0,
            f64::MIN_POSITIVE,
            f64::INFINITY,
            f64::NEG_INFINITY,
        ] {
            assert_eq!(
                AtomicF64::new(v).load(Ordering::SeqCst).to_bits(),
                v.to_bits()
            );
        }
    }
}
