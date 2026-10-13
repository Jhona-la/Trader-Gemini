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
//! banco de τ*; 13×32 = 416 pares motor×escala del consenso), bajo el
//! nulo condicional el FWER queda acotado por min(1, M*alfa). Para
//! M=416 y alfa=0.05, M*alfa=20.8: la cota es trivial (1),
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
    /// Con parámetro finito, lambda en [1e-4, 0.5]: factor en [0.5, 1.5].
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

    /// Constructor con parámetros finitos (tests/calibración).
    /// Acota lambda y alfa a [1e-4, 0.5]; no valida parámetros NaN.
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
        // Factores positivos en aritmética exacta. f64 puede sub/sobrefluir;
        // la representabilidad indefinida requiere un contrato independiente.
    }

    /// ¿El proceso ha cruzado el umbral de Ville (capital ≥ 1/α)?
    /// Puede consultarse continuamente bajo el nulo condicional declarado:
    /// la cota de Ville es POR PROCESO y para la historia cubierta.
    /// El booleano no demuestra esas hipótesis ni vigencia tras drift.
    #[inline]
    pub fn significativo(&self) -> bool {
        self.n >= N_MIN_EVALUE && self.capital >= 1.0 / self.alfa
    }

    /// #663 (G1-1): significancia corregida por MULTIPLICIDAD de
    /// FAMILIA. Ville acota P(∃t: e ≥ 1/α) ≤ α POR PROCESO; con M
    /// procesos consultados en la misma selección, la unión da
    /// P(alguno cruza) <= min(1, M*alfa). Con M=448, alfa=0.05,
    /// M*alfa=22.4: cota trivial (1), no estimación del FWER real.
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
        // p=0.58: E[Δln-capital] = 0.58·ln(1.1) − 0.42·|ln(0.9)| =
        // 0.01103/obs ⇒ cruce de ln(1/α)=ln(20)=2.996 en n≈272
        // (H1-6/G1-8: el "~800" anterior era el número de FAMILIA
        // M=416, no el umbral simple — ver comentario de
        // significativo_familia para los n por M).
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
        assert!(e.significativo() == false); // 17.4 < 20
        for _ in 30..67 {
            e.observar(1.0, 0.01); // capital = 1.1^67 ≈ 591
        }
        assert!(!e.significativo_familia(32), "591 < 640");
        e.observar(1.0, 0.01); // 1.1^68 ≈ 651
        assert!(e.significativo_familia(32), "651 >= 640");
        assert!(!e.significativo_familia(416), "651 < 8320");
    }
}
