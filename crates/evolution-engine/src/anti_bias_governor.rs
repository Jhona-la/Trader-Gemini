/// 🛡️ ALGORITMO #65: GOBERNADOR ANTI-SESGO Y PREVENCIÓN DE CURVE-FITTING
/// Evalúa a los genomas basándose en propiedades estadísticas profundas 
/// (P-Value, Kurtosis, y Sharpe Deflation) para evitar el sobreajuste a la curva (Overfitting).

pub struct AntiBiasGovernor;

impl AntiBiasGovernor {
    /// Calcula el Sharpe Ratio Deflactado (DSR - Deflated Sharpe Ratio).
    /// El DSR penaliza matemáticamente el Sharpe Ratio cuando se han hecho 
    /// muchas pruebas repetidas (Multiple Testing Bias), lo que es inherente a los Algoritmos Evolutivos (NEAT).
    pub fn calculate_deflated_sharpe(
        base_sharpe: f64, 
        num_trials: usize, // Número de mutaciones evaluadas (generaciones * población)
        kurtosis: f64,
        skewness: f64,
        num_trades: usize,
    ) -> f64 {
        if num_trades < 30 {
            return 0.0; // P-Value estadísticamente nulo si hay muy pocos trades
        }

        // 1. Expected Maximum Sharpe Ratio (E[M_T]) bajo la asunción de pruebas múltiples nulas
        // Approximation using Euler-Mascheroni constant
        let euler_mascheroni = 0.5772156649;
        let t = num_trials as f64;
        let max_sharpe_expected = ((1.0 - euler_mascheroni) * (1.0 / (2.0 * t).ln()).sqrt()) 
                                  + (2.0 * t.ln()).sqrt();

        // 2. Varianza de la estimación del Sharpe (considerando no-normalidad: Skewness y Kurtosis)
        let n = num_trades as f64;
        let sharpe_var = (1.0 - (skewness * base_sharpe) + ((kurtosis - 1.0) / 4.0) * base_sharpe.powi(2)) / (n - 1.0);
        let sharpe_std = sharpe_var.sqrt().max(1e-9);

        // 3. Probabilidad Deflactada (P-Value aproximado de DSR) usando CDF de la Normal
        // Z = (Observed Sharpe - Expected Maximum Sharpe) / Std(Observed Sharpe)
        let z_score = (base_sharpe - max_sharpe_expected) / sharpe_std;
        
        // Approximate CDF of Standard Normal via Error Function
        // A simple polynomial approximation for Normal CDF:
        let cdf = 0.5 * (1.0 + f64::tanh((2.0 / std::f64::consts::PI).sqrt() * (z_score + 0.044715 * z_score.powi(3))));
        let dsr_prob = cdf;
        
        // Si la probabilidad de que el Sharpe sea genuino es menor al 90%, lo consideramos ruido.
        if dsr_prob < 0.90 {
            0.0 // Purga inmediata
        } else {
            base_sharpe * dsr_prob // Sharpe castigado
        }
    }

    /// Valida un modelo usando el régimen OOS (Out-of-Sample).
    /// Retorna `true` si el modelo pasa la prueba estadística.
    pub fn validate_out_of_sample(
        is_pnl: f64,     // PnL In-Sample (Entrenamiento)
        oos_pnl: f64,    // PnL Out-of-Sample (Validación ciega)
        is_trades: usize,
        oos_trades: usize
    ) -> bool {
        if is_trades == 0 || oos_trades == 0 {
            return false;
        }

        // El rendimiento promedio por trade no debe degradarse drásticamente.
        let avg_is = is_pnl / (is_trades as f64);
        let avg_oos = oos_pnl / (oos_trades as f64);
        
        if avg_oos < 0.0 {
            return false; // Fracaso absoluto en OOS
        }

        // Tolerancia de degradación (Haircut del 50%).
        // Si el OOS es menos de la mitad de bueno que el IS, es sospechoso de Curve Fitting.
        if avg_oos < (avg_is * 0.5) {
            return false; 
        }

        true
    }

    /// V10 REALITY CHECK: Valida mutantes evolutivos contra ejecución en entorno DEMO/LIVE.
    /// Esto evita la propagación de "Backtest-only artifacts".
    pub fn validate_with_live_reality(
        mutant_simulated_winrate: f64,
        live_demo_winrate: f64,
        live_trades_count: usize,
    ) -> bool {
        // Necesitamos suficientes trades reales para validar la hipótesis.
        if live_trades_count < 10 {
            return false; // Demasiado joven para confirmar
        }

        // Si el bot en vivo pierde más del 15% de su WR teórico, la simulación era mentira.
        let wr_degradation = mutant_simulated_winrate - live_demo_winrate;
        if wr_degradation > 0.15 {
            return false; // Fake strategy detected, purge.
        }

        // El winrate en vivo debe ser mínimo del 50%
        if live_demo_winrate < 0.50 {
            return false; // Perdedor en la vida real.
        }

        true
    }
}
