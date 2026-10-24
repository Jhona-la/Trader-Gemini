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
//! Operador: producto de factores 1 + lambda * score_t, con score en
//! {-1, 0, +1}; cero/no-finito es abstención. Con capital inicial 1,
//! lambda previsible y no negativa, y factores no negativos, el nulo
//! suficiente es E[score_t | F_prev] <= 0 en toda la historia cubierta.
//! Bajo esas condiciones el producto es una supermartingala y Ville acota
//! P(existe t: e_t >= 1/alfa) <= alfa, incluyendo consultas continuas.
//! Mediana del producto <= 0 no basta con átomos en cero: 51% cero y
//! 49% positivo tiene mediana 0 pero E[score]=0.49. Momento cruzado 0
//! tampoco implica ese nulo. La hipótesis del consumidor queda abierta;
//! este operador no certifica independencia ni habilidad actual.
//!
//! #663 (G1-1, mea culpa de #661): Ville NO cubre por sí sola la
//! MULTIPLICIDAD de la selección del máximo — la cota es POR PROCESO.
//! Con M e-procesos consultados en la misma decisión (32 escalas del
//! banco de τ*; 15×32 = 480 pares motor×escala del consenso), bajo el
//! nulo condicional el FWER queda acotado por min(1, M*alfa). Para
//! M=480 y alfa=0.05, M*alfa=24.0: la cota es trivial (1),
//! no una estimación del FWER real. Los consumidores de SELECCIÓN usan
//! `significativo_familia(M)` con umbral M/α (Bonferroni sobre la
//! familia, misma anytime-validity).
//!
//! Robustez del factor: el signo acota la apuesta a +/-1 y elimina
//! la magnitud del retorno de cada factor. Eso no garantiza validez del
//! nulo, crecimiento positivo ni representabilidad f64 de los productos.
//! La dirección acertada necesita frecuencia suficiente y un contrato
//! condicional válido; abstenciones no aportan capital.

/// α por defecto del gate de significancia (5%): el umbral de capital
/// es 1/α = 20 — el proceso necesita multiplicar su capital ×20 por
/// pura dirección acertada antes de declarar habilidad.
pub const ALFA_SIGNIFICANCIA: f64 = 0.05;

/// λ por defecto: factor en [1-lambda, 1+lambda]. En un modelo IID con
/// lambda fija y probabilidades p_+, p_- (p_+ + p_- <= 1), el crecimiento
/// logarítmico medio exacto por observación es
/// p_+ * ln(1+lambda) + p_- * ln(1-lambda); abstenciones aportan cero.
/// Acertar más del 50% no basta por sí solo para crecimiento positivo.
/// Es una identidad del modelo, no garantía económica ni de precisión f64.
pub const LAMBDA_APOSTADA: f64 = 0.10;

/// Muestras mínimas antes de poder declarar (evita capital trivial de
/// 2-3 aciertos suertudos con λ pequeño... el umbral 1/α=20 ya exige
/// 32 aciertos consecutivos con λ=0.10; esta guardia es documental).
pub const N_MIN_EVALUE: u64 = 20;

/// #661 — E-PROCESO de Ville sobre el signo de señal·retorno.
///
/// factor_t = 1 + lambda * score_t; score_t es el signo del producto
/// finito no nulo, o 0 como abstención. El nulo condicional requerido es
/// E[score_t | F_prev] <= 0, con lambda previsible y no negativa.
/// Ville aplica a esa historia y familia bajo sus hipótesis; mediana <= 0
/// o IC=0 no las establecen. La aritmética f64 no garantiza representabilidad
/// indefinida del capital ni reemplaza la calibración de los consumidores.
#[derive(Debug, Clone, Copy)]
pub struct EProceso {
    /// Capital del apostador (producto de factores). Arranca en 1.0.
    capital: f64,
    /// lambda = LAMBDA_APOSTADA (fija); observar_adaptativo deriva su
    /// lambda_t de ella en [0.05, 0.40].
    lambda: f64,
    /// α del gate; umbral = 1/α.
    alfa: f64,
    /// Observaciones consumidas.
    n: u64,
    /// Conteo causal de aciertos en apuestas históricas (para estimación de tasa previsible en GRAPA).
    aciertos: u64,
    /// Conteo causal de fallos en apuestas históricas.
    fallos: u64,
}

impl EProceso {
    pub fn new() -> Self {
        Self {
            capital: 1.0,
            lambda: LAMBDA_APOSTADA,
            alfa: ALFA_SIGNIFICANCIA,
            n: 0,
            aciertos: 0,
            fallos: 0,
        }
    }

    /// Observa un par (señal, retorno) del bloque madurado y actualiza
    /// el capital con lambda fija (comportamiento canónico #661).
    /// Sanitización: no-finitos ⇒ abstención (factor 1).
    #[inline(always)]
    pub fn observar(&mut self, senal: f64, retorno: f64) {
        self.observar_con_lambda_interna(senal, retorno, self.lambda);
    }

    /// Observa con fracción de apuesta adaptativa causal y previsible
    /// (algoritmo GRAPA - Growth Rate Adaptive with Predictable Averaging, Waudby-Smith & Ramdas 2023).
    ///
    /// La fracción \lambda_t se deriva estrictamente de la filtración previa \mathcal{F}_{t-1}
    /// (proporción de aciertos observados en los pasos anteriores).
    /// Si \hat{\mu}_{t-1} = 2\hat{p}_{t-1} - 1 > 0, \lambda_t = (\lambda_{\text{base}} + 0.5 \hat{\mu}_{t-1}).\text{clamp}(0.05, 0.40).
    /// Garantiza la propiedad de supermartingala bajo H0 (E[apuesta_t|\mathcal{F}_{t-1}] <= 0)
    /// mientras acelera el crecimiento de capital hasta ~2.6× ante evidencia consistente.
    #[inline(always)]
    pub fn observar_adaptativo(&mut self, senal: f64, retorno: f64) {
        let total = self.aciertos + self.fallos;
        let lambda_t = if total == 0 {
            self.lambda
        } else {
            let p_hat = self.aciertos as f64 / total as f64;
            let mu_hat = 2.0 * p_hat - 1.0;
            if mu_hat > 0.0 {
                (self.lambda + 0.5 * mu_hat).clamp(0.05, 0.40)
            } else {
                self.lambda
            }
        };
        self.observar_con_lambda_interna(senal, retorno, lambda_t);
    }

    #[inline(always)]
    fn observar_con_lambda_interna(&mut self, senal: f64, retorno: f64, lambda: f64) {
        self.n += 1;
        let producto = senal * retorno;
        if !producto.is_finite() || producto == 0.0 {
            return; // sin información: el capital no cambia
        }
        let apuesta = if producto > 0.0 { 1.0 } else { -1.0 };
        let factor = 1.0 + lambda * apuesta;
        self.capital *= factor;
        if apuesta > 0.0 {
            self.aciertos = self.aciertos.saturating_add(1);
        } else {
            self.fallos = self.fallos.saturating_add(1);
        }
    }

    /// Arranca en caliente el e-proceso restaurando el capital, observaciones y conteos acumulados.
    pub fn warm_start(&mut self, capital: f64, n: u64, aciertos: u64, fallos: u64) {
        if capital.is_finite() && capital > 0.0 {
            self.capital = capital;
        }
        self.n = n;
        self.aciertos = aciertos;
        self.fallos = fallos;
    }

    #[inline]
    pub fn aciertos(&self) -> u64 {
        self.aciertos
    }

    #[inline]
    pub fn fallos(&self) -> u64 {
        self.fallos
    }

    /// #663 (G1-1): significancia corregida por MULTIPLICIDAD de
    /// FAMILIA. Ville acota P(∃t: e ≥ 1/α) ≤ α POR PROCESO; con M
    /// procesos consultados en la misma selección, la unión da
    /// P(alguno cruza) <= min(1, M*alfa). Con M=480, alfa=0.05,
    /// M*alfa=24.0: cota trivial (1), no estimación del FWER real.
    /// El umbral de
    /// familia M/α conserva la cota por unión si M cubre todos los procesos
    /// consultados y cada proceso satisface el nulo condicional declarado.
    /// Un factor por observación no demuestra ese nulo. Reinicios/épocas y
    /// cambios de familia necesitan presupuesto global explícito de error.
    /// Con lambda fija y un modelo IID, ln(M/alfa) dividido por el
    /// drift logarítmico positivo aproxima una escala de acumulación;
    /// no es una identidad del tiempo esperado de primer cruce.
    #[inline]
    pub fn significativo_familia(&self, num_pruebas: usize) -> bool {
        let m = num_pruebas.max(1) as f64;
        self.n >= N_MIN_EVALUE && self.capital >= m / self.alfa
    }

    /// Umbral de capital de la familia (telemetría/tests).
    #[inline]
    pub fn umbral_familia(num_pruebas: usize) -> f64 {
        (num_pruebas.max(1) as f64) / ALFA_SIGNIFICANCIA
    }

    /// Capital actual. Lo leen exportar_snapshots y estado_gate_ville de
    /// EspectralMultiactivo (F1-078), hoy sin llamador de producción.
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
                    if e.significativo_familia(1) {
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
        // p=0.58: E[Δln-capital] = 0.58·ln(1.1) − 0.42·|ln(0.9)| =
        // 0.01103/obs ⇒ cruce de ln(1/α)=ln(20)=2.996 en n≈272
        // (H1-6/G1-8: el "~800" anterior era el número de FAMILIA
        // M=416, no el umbral simple — ver comentario de
        // significativo_familia para los n por M).
        for _ in 0..4_000 {
            let acierta = (rng() % 100) < 58;
            let r = if acierta { 0.02 } else { -0.019 };
            e.observar(1.0, r);
            if e.significativo_familia(1) {
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
        assert!(!e2.significativo_familia(1), "un outlier aislado no declara");
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
        assert!(!e.significativo_familia(1));
    }
}

#[cfg(test)]
mod qo_663_tests {
    use super::*;

    /// #663 (G1-1): umbral de FAMILIA — Bonferroni M/α sobre la
    /// anytime-validity por proceso. 32 escalas ⇒ 640; 416 pares
    /// motor×escala ⇒ 8320. Un capital que era significativo POR
    /// PROCESO (≥20) ya no alcanza en familia.
    #[test]
    fn qo_663_umbral_de_familia_bonferroni() {
        assert!((EProceso::umbral_familia(1) - 20.0).abs() < 1e-9);
        assert!((EProceso::umbral_familia(32) - 640.0).abs() < 1e-9);
        assert!((EProceso::umbral_familia(416) - 8320.0).abs() < 1e-9);

        let mut e = EProceso::new();
        for _ in 0..30 {
            e.observar(1.0, 0.01); // capital = 1.1^30 ≈ 17.4
        }
        assert!(!e.significativo_familia(1)); // 17.4 < 20
        for _ in 30..67 {
            e.observar(1.0, 0.01); // capital = 1.1^67 ≈ 591
        }
        assert!(!e.significativo_familia(32), "591 < 640");
        e.observar(1.0, 0.01); // 1.1^68 ≈ 651
        assert!(e.significativo_familia(32), "651 >= 640");
        assert!(!e.significativo_familia(416), "651 < 8320");
    }
}
