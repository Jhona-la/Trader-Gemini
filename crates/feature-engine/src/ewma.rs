/// Media exponencial con memoria O(1). `update` usa una ganancia por evento;
/// `update_elapsed` usa una constante de tiempo física explícita, sin etiquetas
/// de estrategia. Ninguna variante calibra automáticamente la escala elegida.

#[derive(Debug, Clone, Copy)]
pub struct Ewma {
    pub value: f64,
    pub alpha: f64,
    pub is_initialized: bool,
}

impl Ewma {
    /// Crea un nuevo EWMA.
    /// `alpha` determina la ganancia por evento (0 < alpha <= 1).
    /// # Panics
    /// Si alpha no pertenece al dominio; para configuración externa usar try_new.
    #[inline(always)]
    pub fn new(alpha: f64) -> Self {
        Self::try_new(alpha).expect("EWMA alpha must be finite and in (0, 1]")
    }

    pub fn try_new(alpha: f64) -> Result<Self, &'static str> {
        if !alpha.is_finite() || alpha <= 0.0 || alpha > 1.0 {
            return Err("EWMA alpha must be finite and in (0, 1]");
        }
        Ok(Self {
            value: 0.0,
            alpha,
            is_initialized: false,
        })
    }

    /// Crea un EWMA basado en el periodo `N`.
    /// La fórmula estándar es: alpha = 2 / (N + 1).
    /// N es un periodo nominal por eventos, no duración ni garantía de evidencia.
    /// # Panics
    /// Si N no es finito o es menor que uno. Usar try_from_period para validar.
    #[inline(always)]
    pub fn from_period(period: f64) -> Self {
        Self::try_from_period(period).expect("EWMA period must be finite and at least one")
    }

    pub fn try_from_period(period: f64) -> Result<Self, &'static str> {
        if !period.is_finite() || period < 1.0 {
            return Err("EWMA period must be finite and at least one");
        }
        let alpha = 2.0 / (period + 1.0);
        Self::try_new(alpha)
    }

    /// Solución exacta de dm/dt=(x-m)/tau si x se mantiene constante durante dt.
    /// Ganancia = -expm1(-dt/tau), estable incluso para dt/tau muy pequeño.
    /// El llamador define la señal retenida y proporciona dt>=0, tau>0 en ms.
    /// No infiere timestamps, no deduplica eventos y no estima tau.
    /// La primera observación es condición inicial (incluso con dt=0); después
    /// dt=0 no cambia el estado. No altera alpha de la API histórica por eventos.
    /// Entradas inválidas o resultado no representable no modifican el objeto.
    pub fn update_elapsed(
        &mut self,
        new_val: f64,
        elapsed_ms: f64,
        tau_ms: f64,
    ) -> Result<f64, &'static str> {
        if !new_val.is_finite()
            || !elapsed_ms.is_finite()
            || elapsed_ms < 0.0
            || !tau_ms.is_finite()
            || tau_ms <= 0.0
        {
            return Err("EWMA requires finite value, elapsed_ms >= 0 and tau_ms > 0");
        }
        if !self.is_initialized {
            self.value = new_val;
            self.is_initialized = true;
            return Ok(new_val);
        }
        if elapsed_ms == 0.0 {
            return Ok(self.value);
        }
        let gain = -(-elapsed_ms / tau_ms).exp_m1();
        let next = gain * new_val + (1.0 - gain) * self.value;
        if !next.is_finite() {
            return Err("EWMA next value is not finite");
        }
        self.value = next;
        Ok(next)
    }

    /// Actualiza el valor con la nueva observación en O(1)
    #[inline(always)]
    pub fn update(&mut self, new_val: f64) -> f64 {
        if !new_val.is_finite() {
            return self.value;
        }
        if !self.is_initialized {
            self.value = new_val;
            self.is_initialized = true;
        } else {
            // S_t = (alpha * X_t) + ((1 - alpha) * S_{t-1})
            self.value = (self.alpha * new_val) + ((1.0 - self.alpha) * self.value);
        }
        self.value
    }

    #[inline(always)]
    pub fn get(&self) -> f64 {
        self.value
    }
}
