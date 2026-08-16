/// 🛡️ ALGORITMO #155: ENTROPÍA DE SHANNON CUÁNTICA (MARKET NOISE)
/// Mide la incertidumbre estocástica del libro de órdenes (Order Book).
/// Si la entropía tiende a 1.0, el mercado es ruido puro (no predecible).
/// Si la entropía tiende a 0.0, hay un colapso determinista (alta predictibilidad).

#[derive(Debug, Clone, Copy, Default)]
#[repr(C, align(64))]
pub struct QuantumEntropyCalculator;

impl QuantumEntropyCalculator {
    /// Calcula la entropía del flujo de órdenes en O(N).
    /// `probabilities` es un slice de f64 representando la distribución de volumen o liquidez.
    #[inline(always)]
    pub fn calculate_entropy(probabilities: &[f64]) -> f64 {
        let mut entropy = 0.0;
        let mut valid_states = 0;
        
        for &p in probabilities {
            if p > 0.0 {
                // E = - Σ P(x) * log2(P(x))
                entropy -= p * p.log2();
                valid_states += 1;
            }
        }
        
        // Normalizar entropía entre 0.0 y 1.0 basándose en el número de estados
        if valid_states > 1 {
            let max_entropy = (valid_states as f64).log2();
            if max_entropy > 0.0 {
                entropy / max_entropy
            } else {
                0.0
            }
        } else {
            0.0 // Entropía Cero (Colapso absoluto, 1 solo estado acapara el 100%)
        }
    }

    /// Método de conveniencia para calcular entropía del desequilibrio de liquidez bid/ask
    #[inline(always)]
    pub fn calculate_l2_entropy(bid_vol: f64, ask_vol: f64) -> f64 {
        let total = bid_vol + ask_vol;
        if total <= 0.0 {
            return 1.0; // Caos total sin liquidez
        }
        let p_bid = bid_vol / total;
        let p_ask = ask_vol / total;
        
        Self::calculate_entropy(&[p_bid, p_ask])
    }
}
