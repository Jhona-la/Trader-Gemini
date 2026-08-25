use std::sync::atomic::{AtomicU64, Ordering};
use crate::tensor_ring::TensorRing;
pub const NUM_TIMEFRAMES: usize = 4;
pub const NUM_FEATURES: usize = 64;

/// Quantum Tensor Store
/// A 3D tensor `[Coins, Timeframes, Features]` that allows O(1) lock-free writes and zero-copy reads
/// using AtomicU64 to store f64 bits. Optimised for L1 cache and nanosecond latency.
#[derive(Debug)]
pub struct QuantumTensorStore {
    data: Box<[AtomicU64]>,
    num_coins: usize,
}

impl QuantumTensorStore {
    pub fn new(num_coins: usize) -> Self {
        let total_size = num_coins * NUM_TIMEFRAMES * NUM_FEATURES;
        let mut vec = Vec::with_capacity(total_size);
        for _ in 0..total_size {
            vec.push(AtomicU64::new(0f64.to_bits()));
        }
        Self { data: vec.into_boxed_slice(), num_coins }
    }

    #[inline(always)]
    fn get_index(&self, coin: usize, tf: usize, feature: usize) -> usize {
        debug_assert!(coin < self.num_coins && tf < NUM_TIMEFRAMES && feature < NUM_FEATURES);
        (coin * NUM_TIMEFRAMES * NUM_FEATURES) + (tf * NUM_FEATURES) + feature
    }

    /// O(1) lock-free write
    #[inline(always)]
    pub fn write_feature(&self, coin: usize, tf: usize, feature: usize, value: f64) {
        let idx = self.get_index(coin, tf, feature);
        self.data[idx].store(value.to_bits(), Ordering::Relaxed);
    }

    /// O(1) lock-free read
    #[inline(always)]
    pub fn read_feature(&self, coin: usize, tf: usize, feature: usize) -> f64 {
        let idx = self.get_index(coin, tf, feature);
        f64::from_bits(self.data[idx].load(Ordering::Relaxed))
    }

    /// Extrae un slice tensorial (un timeframe completo de 64 features para una moneda)
    /// Este array de 64 elementos puede ser devuelto directamente a la red neuronal.
    #[inline(always)]
    pub fn extract_feature_vector(&self, coin: usize, tf: usize) -> [f64; NUM_FEATURES] {
        let mut vec = [0.0; NUM_FEATURES];
        let base_idx = self.get_index(coin, tf, 0);
        for i in 0..NUM_FEATURES {
            vec[i] = f64::from_bits(self.data[base_idx + i].load(Ordering::Relaxed));
        }
        vec
    }

    /// Calcula la matriz de covarianza o el exponente de Lyapunov de manera aproximada
    /// basándose en los features. Retorna un valor de "caos".
    #[inline(always)]
    pub fn calculate_lyapunov_chaos(&self, coin: usize) -> f64 {
        // Simplificación: desviación estándar de los features a lo largo del tiempo
        let tf0 = self.extract_feature_vector(coin, 0);
        let tf1 = self.extract_feature_vector(coin, 1);
        
        let mut diff_sum = 0.0;
        for i in 0..NUM_FEATURES {
            let diff = tf0[i] - tf1[i];
            diff_sum += diff * diff;
        }
        
        // Retorna una métrica pseudo-Lyapunov (divergencia exponencial)
        (diff_sum / NUM_FEATURES as f64).sqrt()
    }

    /// 🛡️ FASE 5: Ingestión Cuántica de Cinemática
    /// Toma los anillos tensoriales crudos (del data-ingest) y extrae la física subyacente.
    /// Escribe Velocidad (1ra derivada), Aceleración (2da derivada) y Jerk (3ra derivada)
    /// directamente en las características 0, 1 y 2 del tensor, que luego consumirá la Neural Net.
    #[inline(always)]
    pub fn update_kinematics_from_ring<const N: usize>(&self, coin: usize, tf: usize, price_ring: &TensorRing<N>) {
        let v = price_ring.velocity();
        let a = price_ring.acceleration();
        let j = price_ring.jerk();

        // Guardamos estas características críticas (Features) en los primeros índices.
        self.write_feature(coin, tf, 0, v);
        self.write_feature(coin, tf, 1, a);
        self.write_feature(coin, tf, 2, j);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_quantum_tensor_store_read_write() {
        let store = QuantumTensorStore::new(10);
        store.write_feature(0, 1, 5, 42.5);
        assert_eq!(store.read_feature(0, 1, 5), 42.5);
        assert_eq!(store.read_feature(0, 1, 6), 0.0);
    }

    #[test]
    fn test_quantum_tensor_store_extract_feature_vector() {
        let store = QuantumTensorStore::new(5);
        for feat in 0..NUM_FEATURES {
            store.write_feature(1, 0, feat, feat as f64);
        }

        let vector = store.extract_feature_vector(1, 0);
        assert_eq!(vector.len(), NUM_FEATURES);
        for feat in 0..NUM_FEATURES {
            assert_eq!(vector[feat], feat as f64);
        }
    }

    #[test]
    fn test_quantum_tensor_store_lyapunov_chaos() {
        let store = QuantumTensorStore::new(5);
        store.write_feature(0, 0, 0, 10.0);
        store.write_feature(0, 1, 0, 6.0);

        let chaos = store.calculate_lyapunov_chaos(0);
        assert!(chaos > 0.0 && chaos.is_finite());
    }
}

