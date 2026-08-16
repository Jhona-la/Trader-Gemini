/// Fase A: Filtro de Kalman Cuántico
/// Un filtro de estado ultra-rápido para la predictibilidad de micro-tendencias (Scalping).
/// Modela la incertidumbre y suaviza señales con varianza adaptativa.

#[derive(Debug, Clone)]
pub struct KalmanFilter1D {
    // Estado estimado (ej. precio o velocidad)
    pub x: f64,
    // Covarianza del error de estimación
    pub p: f64,
    // Ruido del proceso (varianza del modelo, qué tan rápido cambia la realidad)
    pub q: f64,
    // Ruido de la medición (varianza del sensor, qué tan ruidoso es el tick)
    pub r: f64,
}

impl KalmanFilter1D {
    pub fn new(initial_x: f64, initial_p: f64, q: f64, r: f64) -> Self {
        Self {
            x: initial_x,
            p: initial_p,
            q,
            r,
        }
    }

    /// Actualiza el filtro con una nueva medición O(1)
    #[inline(always)]
    pub fn update(&mut self, measurement: f64) -> f64 {
        // Predicción
        self.p += self.q;

        // Actualización
        let k = self.p / (self.p + self.r);
        self.x += k * (measurement - self.x);
        self.p = (1.0 - k) * self.p;

        self.x
    }

    /// Actualiza dinámicamente el ruido de medición basado en la volatilidad reciente
    #[inline(always)]
    pub fn update_with_dynamic_r(&mut self, measurement: f64, dynamic_r: f64) -> f64 {
        self.r = dynamic_r;
        self.update(measurement)
    }
}
