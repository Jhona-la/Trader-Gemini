use std::sync::atomic::Ordering;
use std::sync::Arc;
use quantum_arena::GlobalArena;

/// FASE 9: Meta-Evolución y Auto-Arquitectura
/// Módulo que monitorea el rendimiento sistémico e infiere si los fallos
/// son de parámetros (solucionable vía CMA-ES) o estructurales/arquitectónicos.
pub struct MetaEvolver {
    arena: Arc<GlobalArena>,
}

impl MetaEvolver {
    pub fn new(arena: Arc<GlobalArena>) -> Self {
        Self { arena }
    }

    /// Evalúa la salud estructural del bot y emite sugerencias de refactorización
    pub fn audit_system_architecture(&self, current_best_sharpe: f64) {
        let _capital = self.arena.unified_capital.load(Ordering::Relaxed);
        let mut total_trades = 0;
        let mut stagnant_coins = 0;

        for coin in self.arena.coins.iter() {
            let pnl = coin.scalp.pnl_realized.load(Ordering::Relaxed);
            let wr = coin.scalp.win_rate.load(Ordering::Relaxed);
            total_trades += coin.scalp.trade_count.load(Ordering::Relaxed);

            if pnl < 0.0 && wr < 0.35 {
                stagnant_coins += 1;
            }
        }

        println!("============================================================");
        println!("🧠 [META-EVOLVER] Iniciando Reflexión Estructural...");

        if current_best_sharpe < 0.5 {
            println!("🚨 ANOMALÍA ESTRUCTURAL: El CMA-ES no logra converger en rentabilidad (Mejor Sharpe: {:.2}).", current_best_sharpe);
            println!("⚠️ SUGERENCIA AUTO-GENERADA (Refactorización):");
            println!("   -> La dimensionalidad del Feature Engine es insuficiente o tiene ruido excesivo.");
            println!("   -> ACCIÓN: Dividir 'OmniStrategyEngine' en dos features ortogonales (ej. Micro-Imbalance y Macro-Tendencia).");
            println!("   -> ACCIÓN: Eliminar las señales basadas estrictamente en RSI (Capa de Ruido).");
        } else if stagnant_coins > 15 {
            println!("🚨 ANOMALÍA SISTÉMICA: Más del 50% de las monedas están estancadas en pérdidas.");
            println!("⚠️ SUGERENCIA AUTO-GENERADA (Refactorización de Risk):");
            println!("   -> El PortfolioOrchestrator está fallando en cortar la correlación cruzada.");
            println!("   -> ACCIÓN: Implementar una matriz lock-free de covarianza en tiempo real para bloquear entradas direccionales síncronas.");
        } else if total_trades == 0 {
            println!("🚨 ANOMALÍA DE EJECUCIÓN: El bot está completamente en 'Standby'.");
            println!("⚠️ SUGERENCIA AUTO-GENERADA (Sensibilidad):");
            println!("   -> Los umbrales de disparo (Z-Score) son matemáticamente inalcanzables en el régimen de volatilidad actual.");
            println!("   -> ACCIÓN: Incorporar normalización dinámica del umbral basado en ATR (Average True Range).");
        } else {
            println!("✅ SALUD ESTRUCTURAL ÓPTIMA: La arquitectura soporta la presión del mercado. El motor CMA-ES es suficiente.");
        }
        println!("============================================================");
    }
}
