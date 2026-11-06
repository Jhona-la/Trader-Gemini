//! # Módulo: Microprecio Continuo Markoviano Sub-Tick de Stoikov & Gobernador Dinámico `recvWindow`
//!
//! ## Método Pedagógico Riguroso (Modo Profesor)
//! - **QUÉ:** Motor de cálculo del microprecio de Stoikov basado en cadenas de Markov con estados
//!   absorbentes acoplado a un gobernador táctico dinámico de `recvWindow` para APIs de intercambio (Binance).
//! - **POR QUÉ:**
//!   1. El mid-price aritmético $(P_{\text{bid}} + P_{\text{ask}})/2$ es un estimador estático ruidoso que no
//!      incorpora la asimetría de liquidez ni las probabilidades de consumo de cola en el spread. Stoikov (2018)
//!      demostró que el valor fundamental esperado del próximo movimiento de tick depende de la matriz
//!      fundamental $(I - Q)^{-1} R$.
//!   2. Los saltos de latencia de red (jitter) provocan rechazos `TIMESTAMP_AHEAD_OF_SERVER` o ejecuciones rancias
//!      si el `recvWindow` no se adapta de forma continua y robusta a la distribución empírica de RTT.
//! - **PARA QUÉ:**
//!   1. Generar una señal predictiva de microestructura sub-tick de alta frecuencia para colocación Maker
//!      y captura de rebates con cota de selección adversa.
//!   2. Garantizar tasa de rechazo cero en envíos de órdenes de Binance sin relajar la seguridad temporal.
//! - **CÓMO:**
//!   - Discretización del desequilibrio de volumen $I = (q^b - q^a)/(q^b + q^a)$ en $K=5$ estados en stack.
//!   - Modelado de matriz estocástica $Q \in \mathbb{R}^{5 \times 5}$ y matriz de absorción $R \in \mathbb{R}^{5 \times 2}$.
//!   - Inversión de matriz $N = (I - Q)^{-1}$ mediante eliminación de Gauss-Jordan sin asignaciones de heap.
//!   - Probabilidad de absorción alcista $B_{i, 1} = (N R)_{i, 1}$ y corrección de tick $P^* = S + \delta(B_{i,1} - 0.5)$.
//!   - Gobernador `recvWindow`: estimador mediano y MAD (Median Absolute Deviation) sobre buffer circular.
//! - **CUÁNDO:** En cada actualización del libro L1/L2 antes de cotizar o ejecutar.
//! - **DÓNDE:** `crates/feature-engine/src/stoikov_microprice.rs`.
//! - **QUIÉN:** Consejo Colegiado (Rol 2: Quant de Microestructura y Rol 1: Arquitecto Senior).

#![allow(clippy::all)]

/// Número de estados discretos de desequilibrio (Imbalance) en la cadena markoviana.
pub const IMBALANCE_STATES: usize = 5;

/// Capacidad del buffer circular de latencias RTT en microsegundos/milisegundos.
pub const RTT_WINDOW_SIZE: usize = 16;

/// Reporte cuantitativo emitido por el motor de Stoikov.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct StoikovMicropriceReport {
    /// Mid-price aritmético de referencia.
    pub mid_price: f64,
    /// Desequilibrio continuo de volumen [-1.0, 1.0].
    pub imbalance: f64,
    /// Estado discreto asignado [0, 4].
    pub state_idx: usize,
    /// Probabilidad de que el próximo movimiento discreto de tick sea alcista (+delta).
    pub prob_up: f64,
    /// Probabilidad de que el próximo movimiento discreto de tick sea bajista (-delta).
    pub prob_down: f64,
    /// Microprecio de Stoikov ajustado por absorción markoviana.
    pub microprice: f64,
    /// Desviación del microprecio respecto al mid-price (spread bias).
    pub price_delta: f64,
    /// Ventana recvWindow adaptativa sugerida (en milisegundos).
    pub dynamic_recv_window_ms: u64,
}

/// Parámetros de calibración del motor de Stoikov.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct StoikovMicropriceParams {
    /// Tamaño mínimo de tick del activo (delta).
    pub tick_size: f64,
    /// recvWindow mínimo absoluto (ms).
    pub min_recv_window_ms: u64,
    /// recvWindow máximo absoluto (ms).
    pub max_recv_window_ms: u64,
    /// Multiplicador de escala MAD para jitter de red (típico: 3.0).
    pub mad_multiplier: f64,
}

impl Default for StoikovMicropriceParams {
    fn default() -> Self {
        Self {
            tick_size: 0.01,
            min_recv_window_ms: 1_000,
            max_recv_window_ms: 5_000,
            mad_multiplier: 3.0,
        }
    }
}

/// Motor de Microprecio Continuo de Stoikov con Gobernador Dinámico de recvWindow.
pub struct StoikovMicropriceEngine {
    params: StoikovMicropriceParams,
    /// Matriz de transición entre estados transitorios de desequilibrio Q (5x5).
    q_matrix: [[f64; IMBALANCE_STATES]; IMBALANCE_STATES],
    /// Matriz de absorción hacia saltos de tick R (5x2): col 0 = Down, col 1 = Up.
    r_matrix: [[f64; 2]; IMBALANCE_STATES],
    /// Matriz fundamental N = (I - Q)^(-1) precomputada / actualizada.
    fundamental_matrix: [[f64; IMBALANCE_STATES]; IMBALANCE_STATES],
    /// Probabilidades de absorción B = N * R (5x2).
    absorption_matrix: [[f64; 2]; IMBALANCE_STATES],
    /// Buffer circular en stack de muestras de latencia RTT (en microsegundos).
    rtt_samples: [u64; RTT_WINDOW_SIZE],
    rtt_count: usize,
    rtt_head: usize,
}

impl Default for StoikovMicropriceEngine {
    fn default() -> Self {
        Self::new(StoikovMicropriceParams::default())
    }
}

impl StoikovMicropriceEngine {
    /// Crea una nueva instancia del motor con matrices de transición markoviana calibradas.
    pub fn new(params: StoikovMicropriceParams) -> Self {
        let mut engine = Self {
            params,
            q_matrix: [[0.0; IMBALANCE_STATES]; IMBALANCE_STATES],
            r_matrix: [[0.0; 2]; IMBALANCE_STATES],
            fundamental_matrix: [[0.0; IMBALANCE_STATES]; IMBALANCE_STATES],
            absorption_matrix: [[0.0; 2]; IMBALANCE_STATES],
            rtt_samples: [25_000; RTT_WINDOW_SIZE], // Inicializado con 25ms típicos
            rtt_count: 0,
            rtt_head: 0,
        };

        engine.init_default_transition_matrices();
        engine.recompute_fundamental_matrix();
        engine
    }

    /// Inicializa una matriz markoviana empírica típica de orden de microestructura.
    /// Estados:
    /// 0: Extremo Ask (fuerte presión bajista)
    /// 1: Moderado Ask
    /// 2: Balanceado
    /// 3: Moderado Bid
    /// 4: Extremo Bid (fuerte presión alcista)
    fn init_default_transition_matrices(&mut self) {
        // Matriz Q de transiciones transitorias (filas suman < 1.0, el remanente va a R)
        self.q_matrix = [
            [0.55, 0.25, 0.05, 0.01, 0.00], // Estado 0
            [0.15, 0.50, 0.20, 0.04, 0.01], // Estado 1
            [0.05, 0.25, 0.40, 0.25, 0.05], // Estado 2
            [0.01, 0.04, 0.20, 0.50, 0.15], // Estado 3
            [0.00, 0.01, 0.05, 0.25, 0.55], // Estado 4
        ];

        // Matriz R de absorción [Down, Up]
        self.r_matrix = [
            [0.13, 0.01],   // Estado 0: 13% cae tick, 1% sube tick
            [0.08, 0.02],   // Estado 1: 8% cae, 2% sube
            [0.025, 0.025], // Estado 2: simétrico
            [0.02, 0.08],   // Estado 3: 2% cae, 8% sube
            [0.01, 0.13],   // Estado 4: 1% cae, 13% sube
        ];
    }

    /// Resuelve N = (I - Q)^(-1) y B = N * R usando inversión de Gauss-Jordan en stack.
    pub fn recompute_fundamental_matrix(&mut self) {
        // Construir matriz A = I - Q
        let mut a = [[0.0; IMBALANCE_STATES]; IMBALANCE_STATES];
        for i in 0..IMBALANCE_STATES {
            for j in 0..IMBALANCE_STATES {
                let delta = if i == j { 1.0 } else { 0.0 };
                a[i][j] = delta - self.q_matrix[i][j];
            }
        }

        // Matriz aumentada con la identidad para Gauss-Jordan: [A | I]
        let mut aug = [[0.0; IMBALANCE_STATES * 2]; IMBALANCE_STATES];
        for i in 0..IMBALANCE_STATES {
            for j in 0..IMBALANCE_STATES {
                aug[i][j] = a[i][j];
                aug[i][j + IMBALANCE_STATES] = if i == j { 1.0 } else { 0.0 };
            }
        }

        // Eliminación Gauss-Jordan con pivoteo parcial
        for col in 0..IMBALANCE_STATES {
            // Buscar pivote máximo
            let mut max_row = col;
            let mut max_val = aug[col][col].abs();
            for row in (col + 1)..IMBALANCE_STATES {
                let v = aug[row][col].abs();
                if v > max_val {
                    max_val = v;
                    max_row = row;
                }
            }

            if max_val < 1e-12 {
                // Matriz casi singular: fallback a matriz identidad para N
                for i in 0..IMBALANCE_STATES {
                    for j in 0..IMBALANCE_STATES {
                        self.fundamental_matrix[i][j] = if i == j { 1.0 } else { 0.0 };
                    }
                }
                return;
            }

            // Intercambiar filas si es necesario
            if max_row != col {
                for k in 0..(IMBALANCE_STATES * 2) {
                    let tmp = aug[col][k];
                    aug[col][k] = aug[max_row][k];
                    aug[max_row][k] = tmp;
                }
            }

            // Normalizar fila del pivote
            let pivot = aug[col][col];
            for k in 0..(IMBALANCE_STATES * 2) {
                aug[col][k] /= pivot;
            }

            // Anular otras filas
            for row in 0..IMBALANCE_STATES {
                if row != col {
                    let factor = aug[row][col];
                    if factor.abs() > 1e-15 {
                        for k in 0..(IMBALANCE_STATES * 2) {
                            aug[row][k] -= factor * aug[col][k];
                        }
                    }
                }
            }
        }

        // Extraer N de la mitad derecha de aug
        for i in 0..IMBALANCE_STATES {
            for j in 0..IMBALANCE_STATES {
                self.fundamental_matrix[i][j] = aug[i][j + IMBALANCE_STATES];
            }
        }

        // Multiplicar B = N * R (5x5 * 5x2 = 5x2)
        for i in 0..IMBALANCE_STATES {
            for k in 0..2 {
                let mut sum = 0.0;
                for j in 0..IMBALANCE_STATES {
                    sum += self.fundamental_matrix[i][j] * self.r_matrix[j][k];
                }
                self.absorption_matrix[i][k] = sum;
            }
        }
    }

    /// Clasifica el desequilibrio de volumen continuo [-1.0, 1.0] en su bucket discreto [0, 4].
    #[inline(always)]
    pub fn discretize_imbalance(imbalance: f64) -> usize {
        let clamped = imbalance.clamp(-1.0, 1.0);
        if clamped < -0.6 {
            0
        } else if clamped < -0.2 {
            1
        } else if clamped <= 0.2 {
            2
        } else if clamped <= 0.6 {
            3
        } else {
            4
        }
    }

    /// Registra una nueva muestra de latencia de red RTT (en microsegundos).
    #[inline]
    pub fn record_rtt_sample(&mut self, rtt_micros: u64) {
        self.rtt_samples[self.rtt_head] = rtt_micros;
        self.rtt_head = (self.rtt_head + 1) % RTT_WINDOW_SIZE;
        if self.rtt_count < RTT_WINDOW_SIZE {
            self.rtt_count += 1;
        }
    }

    /// Calcula la ventana táctica dinámica `recvWindow` (en milisegundos).
    /// recvWindow = clamp(median + k * MAD, min, max)
    pub fn compute_dynamic_recv_window(&self) -> u64 {
        let n = if self.rtt_count == 0 {
            return self.params.min_recv_window_ms;
        } else {
            self.rtt_count
        };

        // Copiar a buffer local en stack y ordenar por inserción (sin heap)
        let mut sorted = [0u64; RTT_WINDOW_SIZE];
        for i in 0..n {
            sorted[i] = self.rtt_samples[i];
        }

        // Insertion sort O(n^2) en stack para n=16 (tarda ~10 ns)
        for i in 1..n {
            let key = sorted[i];
            let mut j = i;
            while j > 0 && sorted[j - 1] > key {
                sorted[j] = sorted[j - 1];
                j -= 1;
            }
            sorted[j] = key;
        }

        let median_micros = if n % 2 == 1 {
            sorted[n / 2] as f64
        } else {
            (sorted[(n / 2) - 1] as f64 + sorted[n / 2] as f64) * 0.5
        };

        // Calcular desviaciones absolutas para MAD
        let mut dev = [0.0; RTT_WINDOW_SIZE];
        for i in 0..n {
            dev[i] = (sorted[i] as f64 - median_micros).abs();
        }

        for i in 1..n {
            let key = dev[i];
            let mut j = i;
            while j > 0 && dev[j - 1] > key {
                dev[j] = dev[j - 1];
                j -= 1;
            }
            dev[j] = key;
        }

        let mad_micros = if n % 2 == 1 {
            dev[n / 2]
        } else {
            (dev[(n / 2) - 1] + dev[n / 2]) * 0.5
        };

        // Estimador de desviación estándar consistente con normal: sigma = 1.4826 * MAD
        let jitter_micros = 1.4826 * mad_micros;
        let tactical_micros = median_micros + self.params.mad_multiplier * jitter_micros;
        let tactical_ms = (tactical_micros / 1000.0).round() as u64;

        tactical_ms.clamp(
            self.params.min_recv_window_ms,
            self.params.max_recv_window_ms,
        )
    }

    /// Evalúa el microprecio de Stoikov para un nivel L1 dado.
    pub fn evaluate(
        &self,
        best_bid_price: f64,
        best_bid_qty: f64,
        best_ask_price: f64,
        best_ask_qty: f64,
    ) -> StoikovMicropriceReport {
        let mid_price = (best_bid_price + best_ask_price) * 0.5;
        let total_qty = best_bid_qty + best_ask_qty;

        let imbalance = if total_qty > 1e-12 {
            (best_bid_qty - best_ask_qty) / total_qty
        } else {
            0.0
        };

        let state_idx = Self::discretize_imbalance(imbalance);

        // Probabilidades de absorción normalizadas
        let raw_down = self.absorption_matrix[state_idx][0];
        let raw_up = self.absorption_matrix[state_idx][1];
        let sum_p = raw_down + raw_up;

        let (prob_down, prob_up) = if sum_p > 1e-12 {
            (raw_down / sum_p, raw_up / sum_p)
        } else {
            (0.5, 0.5)
        };

        // Corrección de tick Stoikov: P* = S + delta * (prob_up - 0.5)
        let price_delta = self.params.tick_size * (prob_up - 0.5);
        let microprice = mid_price + price_delta;

        let dynamic_recv_window_ms = self.compute_dynamic_recv_window();

        StoikovMicropriceReport {
            mid_price,
            imbalance,
            state_idx,
            prob_up,
            prob_down,
            microprice,
            price_delta,
            dynamic_recv_window_ms,
        }
    }
}
