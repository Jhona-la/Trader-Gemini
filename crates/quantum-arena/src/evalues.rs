//! #661 — E-VALUES ANYTIME-VALID (martingalas de Ville) para la
//! significancia de la habilidad espectral.
//!
//! El problema: los umbrales fijos de significancia (Fisher 2/√(n−3),
//! anclado a n=128 por H5/#648) son válidos SOLO si el número de
//! observaciones se fija ANTES del experimento. El sistema consulta la
//! habilidad de cada motor×escala en CADA evento de depth (composición
//! del consenso) y selecciona el MÁXIMO IC entre 32 escalas (banco de
//! τ* #594) — optional stopping y multiplicidad que rompen la garantía:
//! «en ruido el máximo de varias IC suele ser positivo» (diagnóstico del
//! consejo, abierto desde #594).
//!
//! La solución: un E-PROCESO — producto de factores de apuesta no
//! negativos. Bajo H0 (la señal NO tiene habilidad: mediana del producto
//! señal·retorno ≤ 0), cada factor tiene esperanza ≤ 1 y el producto es
//! una SUPERMARTINGALA. La desigualdad de Ville da la garantía
//! ANYTIME-VALID: P(∃t: e_t ≥ 1/α) ≤ α — para CUALQUIER tiempo de
//! parada, incluyendo consultas continuas y selección del máximo.
//!
//! Robustez: el factor apuesta sobre el SIGNO de señal·retorno
//! (winsorizado por construcción a ±1) — inmune a las colas pesadas de
//! los retornos financieros (un solo |r| de 10σ no puede romper el
//! proceso). La potencia se recupera acumulando evidencia: el capital
//! crece exponencialmente cuando la señal acierta direcciones.

/// α por defecto del gate de significancia (5%): el umbral de capital
/// es 1/α = 20 — el proceso necesita multiplicar su capital ×20 por
/// pura dirección acertada antes de declarar habilidad.
pub const ALFA_SIGNIFICANCIA: f64 = 0.05;

/// λ por defecto (fracción de capital apostada por observación): acota
/// el factor a [1−λ, 1+λ]. Pequeño ⇒ drawdowns lentos del capital y
/// potencia moderada por observación, pero la ley de grandes números
/// acumula: n·λ·(2p−1) en log-capital con p = tasa de acierto.
pub const LAMBDA_APOSTADA: f64 = 0.10;

/// Muestras mínimas antes de poder declarar (evita capital trivial de
/// 2-3 aciertos suertudos con λ pequeño... el umbral 1/α=20 ya exige
/// ~35 aciertos netos con λ=0.10; esta guardia es documental).
pub const N_MIN_EVALUE: u64 = 20;

/// #661 — E-PROCESO de Ville sobre el signo de señal·retorno.
///
/// factor_i = 1 + λ·sign(s_i · r_i) — bajo H0 (mediana(s·r) ≤ 0):
/// E[factor] = 1 + λ·E[sign] ≤ 1 ⇒ e_t supermartingala no-negativa.
/// Ville: P(sup_t e_t ≥ 1/α) ≤ α para cualquier tiempo de parada.
#[derive(Debug, Clone, Copy)]
pub struct EProceso {
    /// Capital del apostador (producto de factores). Arranca en 1.0.
    capital: f64,
    /// λ acotada a (0, 0.5): factor siempre positivo.
    lambda: f64,
    /// α del gate; umbral = 1/α.
    alfa: f64,
    /// Observaciones consumidas.
    n: u64,
}

impl EProceso {
    pub fn new() -> Self {
        Self {
            capital: 1.0,
            lambda: LAMBDA_APOSTADA,
            alfa: ALFA_SIGNIFICANCIA,
            n: 0,
        }
    }

    /// Constructor con parámetros (tests/calibración).
    pub fn con_parametros(lambda: f64, alfa: f64) -> Self {
        Self {
            capital: 1.0,
            lambda: lambda.clamp(1e-4, 0.5),
            alfa: alfa.clamp(1e-4, 0.5),
            n: 0,
        }
    }

    /// Observa un par (señal, retorno) del bloque madurado y actualiza
    /// el capital. Sanitización: no-finitos ⇒ abstención (factor 1).
    #[inline(always)]
    pub fn observar(&mut self, senal: f64, retorno: f64) {
        self.n += 1;
        let producto = senal * retorno;
        if !producto.is_finite() || producto == 0.0 {
            return; // sin información: el capital no cambia
        }
        let apuesta = if producto > 0.0 { 1.0 } else { -1.0 };
        let factor = 1.0 + self.lambda * apuesta;
        self.capital *= factor;
        // El capital nunca puede tocar 0 (λ ≤ 0.5 ⇒ factor ≥ 0.5).
    }

    /// ¿El proceso ha cruzado el umbral de Ville (capital ≥ 1/α)?
    /// Esta es la significancia ANYTIME-VALID: puede consultarse en
    /// cualquier momento, cualquier número de veces, sin inflar el error
    /// de Tipo I más allá de α.
    #[inline]
    pub fn significativo(&self) -> bool {
        self.n >= N_MIN_EVALUE && self.capital >= 1.0 / self.alfa
    }

    /// Capital actual (telemetría: qo_661_evalue).
    #[inline]
    pub fn capital(&self) -> f64 {
        self.capital
    }

    #[inline]
    pub fn n(&self) -> u64 {
        self.n
    }
}

impl Default for EProceso {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Villa en acción: una señal SIN habilidad (acierto 50%) casi
    /// nunca cruza 1/α, no importa CUÁNDO se mire.
    #[test]
    fn qo_661_proceso_no_declara_en_ruido() {
        // Determinista con LCG simple para reproducibilidad.
        let mut estado = 0x2545F4914F6CDD1Du64;
        let mut rng = || {
            estado ^= estado << 13;
            estado ^= estado >> 7;
            estado ^= estado << 17;
            estado
        };
        let falsos_positivos = (0..200)
            .filter(|_| {
                let mut e = EProceso::new();
                for _ in 0..5_000 {
                    let ruido_signo = if rng() % 2 == 0 { 1.0 } else { -1.0 };
                    // señal arbitraria, retorno aleatorio: sin habilidad
                    e.observar(1.0, ruido_signo * 0.01);
                    if e.significativo() {
                        return true; // cruce en ALGÚN momento (optional stopping)
                    }
                }
                false
            })
            .count();
        // Ville garantiza P ≤ α = 5% por proceso. En MC con 200 procesos:
        // esperado 10, σ = √(200·0.05·0.95) ≈ 3.1 ⇒ umbral 10 + 3σ ≈ 20.
        assert!(
            falsos_positivos <= 20,
            "Ville violada: {falsos_positivos}/200 procesos de ruido cruzaron 1/α"
        );
    }

    /// Una señal CON habilidad (p≈0.58, como el mejor consenso medido
    /// por GLM LXXXIII) cruza el umbral con evidencia acumulada.
    #[test]
    fn qo_661_proceso_declara_con_habilidad() {
        let mut estado = 0x9E3779B97F4A7C15u64;
        let mut rng = || {
            estado ^= estado << 13;
            estado ^= estado >> 7;
            estado ^= estado << 17;
            estado
        };
        let mut e = EProceso::new();
        let mut cruzo = false;
        // p=0.58: E[sign]=0.16 ⇒ log-capital ≈ n·λ·0.16·(ajuste) —
        // cruza 20 en ~n=800.
        for _ in 0..4_000 {
            let acierta = (rng() % 100) < 58;
            let r = if acierta { 0.02 } else { -0.019 };
            e.observar(1.0, r);
            if e.significativo() {
                cruzo = true;
                break;
            }
        }
        assert!(cruzo, "señal con p=0.58 debe acumular capital ≥ 1/α");
    }

    /// Inmunidad a colas: un único retorno de 1e12 (10σ+) no rompe el
    /// proceso ni declara por accidente (signo winsorizado).
    #[test]
    fn qo_661_inmunidad_colas_pesadas() {
        let mut e = EProceso::new();
        for _ in 0..100 {
            e.observar(1.0, 1e12); // un solo acierto GIGANTE
        }
        // 100 aciertos seguidos: capital = 1.1^100 ≈ 13_780 > 20 — SÍ
        // declara (100 direcciones acertadas ES evidencia); pero un
        // SOLO outlier no puede:
        let mut e2 = EProceso::new();
        e2.observar(1.0, 1e12);
        e2.observar(1.0, -1e-6);
        assert!(!e2.significativo(), "un outlier aislado no declara");
        // y el proceso nunca es NaN/0:
        assert!(e2.capital().is_finite() && e2.capital() > 0.0);
    }

    /// Sanitización: no-finitos y producto cero son abstención.
    #[test]
    fn qo_661_sanitizacion() {
        let mut e = EProceso::new();
        e.observar(f64::NAN, 1.0);
        e.observar(1.0, f64::INFINITY);
        e.observar(0.5, 0.0);
        assert_eq!(e.capital(), 1.0);
        assert_eq!(e.n(), 3);
        assert!(!e.significativo());
    }
}
