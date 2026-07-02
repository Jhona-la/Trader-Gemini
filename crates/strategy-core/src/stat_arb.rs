use crate::{SignalType, SignalIntent};

pub struct StatArbEngine {
    window_size: usize,
    history: Vec<f64>,
    index: usize,
    count: usize,
    sum: f64,
    sum_sq: f64,
    z_score_threshold: f64,
    is_in_position: bool,
    current_direction: SignalType, // Long means Long A / Short B
}

impl StatArbEngine {
    pub fn new(window_size: usize, z_score_threshold: f64) -> Self {
        Self {
            window_size,
            history: vec![0.0; window_size],
            index: 0,
            count: 0,
            sum: 0.0,
            sum_sq: 0.0,
            z_score_threshold,
            is_in_position: false,
            current_direction: SignalType::Flat,
        }
    }

    /// Toma los precios de dos activos correlacionados y devuelve la intención de arbitraje sobre el Activo A.
    /// (El Activo B debe operar en la dirección contraria).
    #[inline(always)]
    pub fn update(&mut self, price_a: f64, price_b: f64) -> SignalIntent {
        let spread = price_a.ln() - price_b.ln();
        
        let old_val = self.history[self.index];
        self.history[self.index] = spread;
        
        if self.count < self.window_size {
            self.count += 1;
            self.sum += spread;
            self.sum_sq += spread * spread;
        } else {
            self.sum += spread - old_val;
            self.sum_sq += (spread * spread) - (old_val * old_val);
        }
        
        self.index = (self.index + 1) % self.window_size;
        
        if self.count < self.window_size {
            return SignalIntent::flat();
        }
        
        let mean = self.sum / self.window_size as f64;
        let variance = (self.sum_sq / self.window_size as f64) - (mean * mean);
        let stdev = if variance > 0.0 { variance.sqrt() } else { 0.000001 };
        
        let z_score = (spread - mean) / stdev;
        
        if !self.is_in_position {
            if z_score > self.z_score_threshold {
                self.is_in_position = true;
                self.current_direction = SignalType::Short;
                // A está sobrevalorado respecto a B -> Short A, Long B
                return SignalIntent { signal: SignalType::Short, confidence: z_score.abs() };
            } else if z_score < -self.z_score_threshold {
                self.is_in_position = true;
                self.current_direction = SignalType::Long;
                // A está infravalorado respecto a B -> Long A, Short B
                return SignalIntent { signal: SignalType::Long, confidence: z_score.abs() };
            }
        } else {
            // Regresión a la media (Exit)
            if z_score.abs() < 0.1 {
                self.is_in_position = false;
                self.current_direction = SignalType::Flat;
                return SignalIntent { signal: SignalType::Flat, confidence: 1.0 }; // Flag de Cierre
            } else {
                // Mantener
                return SignalIntent { signal: self.current_direction, confidence: z_score.abs() };
            }
        }
        
        SignalIntent::flat()
    }
}
