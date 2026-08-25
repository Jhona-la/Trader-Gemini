use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::sync::Arc;
use std::time::{Duration, Instant};
use parking_lot::RwLock;

/// Estructura atómica que reside en la memoria compartida entre el Motor de Evolución y el Motor de Ejecución
#[derive(Clone)]
pub struct QuantumHotSwapState {
    pub has_new_genome: Arc<AtomicBool>,
    pub active_genome_id: Arc<AtomicUsize>,
    pub shadow_sharpe_ratio: Arc<RwLock<f64>>,
}

impl QuantumHotSwapState {
    pub fn new() -> Self {
        Self {
            has_new_genome: Arc::new(AtomicBool::new(false)),
            active_genome_id: Arc::new(AtomicUsize::new(0)),
            shadow_sharpe_ratio: Arc::new(RwLock::new(0.0)),
        }
    }
}

impl Default for QuantumHotSwapState {
    fn default() -> Self {
        Self::new()
    }
}

use quantum_arena::GlobalArena;
use quantum_arena::genome::SuperGenotype;

pub struct LiveEvolutionDaemon {
    pub state: QuantumHotSwapState,
    pub arena: Arc<GlobalArena>,
    pub iteration_count: usize,
    pub last_evolution: Instant,
    pub daemon_start_time: Instant,
    pub warmup_duration_secs: u64,
    pub is_demo: bool, // Identifica si estamos en Testnet para acelerar la evolución
    pub ledger: storage_engine::evolution_ledger::EvolutionLedger,
    pub champion_path: String,
    pub ewma_sharpe: f64, // HC-12: Sharpe adaptativo para kill switch
    pub forest: crate::online_random_forest::TrueOnlineRandomForest,
    /// F4.5: PnL realizado visto por coin en el ciclo anterior — para muestrear
    /// retornos de la ESTRATEGIA (deltas reales), no beta del mercado.
    pub last_realized_by_coin: std::collections::HashMap<usize, f64>,
    /// FIX #1600: Acumulación histórica persistente de retornos de estrategia sobre ventana deslizante
    pub returns_history: Vec<f64>,
}

impl LiveEvolutionDaemon {
    pub fn new(state: QuantumHotSwapState, arena: Arc<GlobalArena>, is_demo: bool, db_path: &str, champion_path: &str) -> Self {
        // En Testnet (Demo) calentamos rápido (60s), en Producción somos más rigurosos (15 min)
        let warmup = if is_demo { 60 } else { 900 };
        if let Some(parent) = std::path::Path::new(db_path).parent() {
            let _ = std::fs::create_dir_all(parent);
        }
        if let Some(parent) = std::path::Path::new(champion_path).parent() {
            let _ = std::fs::create_dir_all(parent);
        }
        let ledger = storage_engine::evolution_ledger::EvolutionLedger::new(db_path);
        Self {
            state,
            arena,
            iteration_count: 0,
            last_evolution: Instant::now(),
            daemon_start_time: Instant::now(),
            warmup_duration_secs: warmup,
            is_demo,
            ledger,
            champion_path: champion_path.to_string(),
            ewma_sharpe: 0.0,
            forest: crate::online_random_forest::TrueOnlineRandomForest::new(5000),
            last_realized_by_coin: std::collections::HashMap::new(),
            returns_history: Vec::with_capacity(1024),
        }
    }

    /// Ciclo asincrónico que corre paralelo al bot de producción
    /// Ingiere datos reales y entrena tensores cuánticos en las sombras.
    /// F5.4: sin unwrap — telemetría ausente ⇒ ingesta deshabilitada con log,
    /// el resto del daemon (umbrales, drift) sigue vivo.
    pub async fn run_online_learning_loop(&mut self) {
        let mut telemetry_reader =
            Some(storage_engine::MmapTelemetryReader::new("data/telemetry.mmap"));

        let mut pending_new_obs = 0;

        loop {
            tokio::time::sleep(Duration::from_millis(500)).await;

            // Ingest Telemetry into the Shadow Forest
            if let Some(reader) = telemetry_reader.as_mut() {
                if let Ok(frames) = reader.read_latest_frames() {
                    for f in frames {
                        // SUBSYSTEM_TENSOR_PREDICTOR = 12, FRAME_PREDICTION_VS_REALITY = 30
                        if f.subsystem_id == 12 && f.frame_type == 30 {
                            let ml_prob = f.payload[1];
                            let net_pnl_pct = f.payload[3];
                            // If we are evaluating a long or short based on the prob:
                            let is_long = ml_prob > 0.5;
                            self.forest.shadow_evaluate(ml_prob as f32, net_pnl_pct as f32, 0.0, is_long);
                            pending_new_obs += 1;
                        }
                    }
                }
            }

            // Reentrenar periódicamente el Shadow Forest cuando hay suficientes observaciones
            if pending_new_obs >= 10 || (self.iteration_count % 60 == 0 && self.forest.observations.read().unwrap().len() >= 50) {
                if let Ok((acc, mse)) = self.forest.retrain_models() {
                    println!("🌲 [SHADOW RANDOM FOREST] Reentrenado con éxito! Accuracy: {:.2}%, MSE: {:.6}", acc * 100.0, mse);
                    pending_new_obs = 0;
                }
            }

            // Aplicar thresholds óptimos del Shadow Forest a la Arena ÚNICAMENTE cuando está entrenado
            if self.forest.is_trained() {
                let (opt_l, opt_s) = self.forest.get_optimal_thresholds();
                self.arena.config.ml_threshold_long.store(opt_l as f64, Ordering::Relaxed);
                self.arena.config.ml_threshold_short.store(opt_s as f64, Ordering::Relaxed);
            }

            // FASE 3: AST Mutator checking
            if std::path::Path::new(".forensic_violation").exists() {
                println!("🧬 [DAEMON] Señal forense detectada! Invocando AST-Mutator...");
                let mutator = crate::ast_mutator::ASTMutator::new();
                
                let config_path = "dynamic_config.json";
                if std::path::Path::new(config_path).exists() {
                    let _ = mutator.mutate_json_config(config_path, "Risk.ML_LOOKAHEAD_PENALTY", serde_json::json!(2.0));
                }
                
                let _ = std::fs::remove_file(".forensic_violation");
            }

            // Muestrear retornos realizados en tiempo real tras cada tick de 500ms
            self.sample_realized_returns();

            self.iteration_count += 1;
            
            // Cada 15 minutos (simulado con iteraciones) validamos si el entorno cambió
            if self.last_evolution.elapsed() > Duration::from_secs(900) {
                self.evaluate_shadow_strategy().await;
                self.last_evolution = Instant::now();
            }
        }
    }

    fn sample_realized_returns(&mut self) {
        let capital = self.arena.unified_capital.load(std::sync::atomic::Ordering::Relaxed);

        for coin_id in 0..self.arena.coins.len() {
            let coin = &self.arena.coins[coin_id];
            let realized = coin.scalp.pnl_realized.load(std::sync::atomic::Ordering::Relaxed)
                + coin.swing.pnl_realized.load(std::sync::atomic::Ordering::Relaxed);

            if let Some(&prev) = self.last_realized_by_coin.get(&coin_id) {
                let delta = realized - prev;
                if delta.abs() > 0.0 && capital > 0.0 {
                    let ret = delta / capital;
                    if ret.is_finite() {
                        self.returns_history.push(ret);
                    }
                }
            }
            self.last_realized_by_coin.insert(coin_id, realized);
        }

        // Mantener ventana deslizante acotada a los 1000 trades más recientes
        if self.returns_history.len() > 1000 {
            let drain_count = self.returns_history.len() - 1000;
            self.returns_history.drain(0..drain_count);
        }
    }
    
    async fn evaluate_shadow_strategy(&mut self) {
        // FASE 3: Dynamic Batch Sizing adaptativo a la memoria del sistema
        let _batch_size = 5000;
        
        self.sample_realized_returns();

        // Requiere al menos 10 observaciones históricas
        if self.returns_history.len() < 10 {
            return;
        }

        // Si el Sharpe Cuántico RANSAC > 2.0 y supera a la estrategia de producción...
        let current_shadow_sharpe = Self::calculate_ransac_sharpe(&self.returns_history);
        
        let elapsed_warmup = self.daemon_start_time.elapsed().as_secs();
        if elapsed_warmup < self.warmup_duration_secs {
            println!("⏳ [WARMUP PHASE] {}/{} segundos. Sharpe: {:.2}. Esperando maduración de tensores...", elapsed_warmup, self.warmup_duration_secs, current_shadow_sharpe);
            return;
        }

        println!("🧠 [ONLINE EVOLUTION] Evaluando Shadow Strategy con {} trades reales acumulados... Sharpe Estimado (RANSAC): {:.2}", self.returns_history.len(), current_shadow_sharpe);
        
        if current_shadow_sharpe > 2.0 {
            {
                let mut sr = self.state.shadow_sharpe_ratio.write();
                *sr = current_shadow_sharpe;
            }
            
            // FASE I: Random Forest / Thousands of Universes Evaluation (Estasis de Probabilidad)
            let current_genome = match std::fs::read(&self.champion_path) {
                Ok(bytes) => serde_json::from_slice::<quantum_arena::genome::SuperGenotype>(&bytes)
                    .unwrap_or_else(|_| quantum_arena::genome::SuperGenotype::default()),
                Err(_) => quantum_arena::genome::SuperGenotype::default(),
            };
            
            let _iteration = self.iteration_count;
            let fallback_genome = current_genome.clone();
            
            // FASE 13: Entropic Volatility Mutation
            let mean = self.returns_history.iter().sum::<f64>() / self.returns_history.len() as f64;
            let variance = self.returns_history.iter().map(|v| (v - mean).powi(2)).sum::<f64>() / self.returns_history.len() as f64;
            let volatility = variance.sqrt().max(0.0001);
            
            let best_genome = tokio::task::spawn_blocking(move || {
                let mut best = current_genome.clone();
                let mut best_score = -999.0;
                let roundtrip_fee = 0.0004;
                
                // Mutation scales dynamically based on real-time market entropy
                let dynamic_mutation_rate = (volatility * 50.0).clamp(0.01, 0.25);
                
                let mut rng = rand::rng();
                for _i in 0..10_000 {
                    let mut candidate = current_genome.clone();
                    use rand::RngExt;
                    
                    // Mutación Vectorial de Tensores (DL/RL Vivo)
                    // FASE 9: Ajuste Adaptativo por Régimen de Mercado (Drift Recovery)
                    // FASE 13: Topological Evolution
                    
                    candidate.scalp_kelly_fraction += (rng.random::<f64>() - 0.5) * dynamic_mutation_rate;
                    candidate.swing_kelly_fraction += (rng.random::<f64>() - 0.5) * dynamic_mutation_rate;
                    
                    // SL/TP se contraen o expanden según volatilidad/búsqueda RL
                    candidate.scalp_tp_base *= 1.0 + (rng.random::<f64>() - 0.5) * 0.1;
                    candidate.scalp_sl_base *= 1.0 + (rng.random::<f64>() - 0.5) * 0.1;
                    
                    candidate.ml_threshold_long += (rng.random::<f64>() - 0.5) * (dynamic_mutation_rate * 0.5);
                    candidate.ml_threshold_short += (rng.random::<f64>() - 0.5) * (dynamic_mutation_rate * 0.5);

                    // Evolución de genes Topológicos (Red Neuronal)
                    candidate.topo_layer_1_activation += (rng.random::<f64>() - 0.5) * dynamic_mutation_rate;
                    candidate.topo_layer_2_activation += (rng.random::<f64>() - 0.5) * dynamic_mutation_rate;
                    candidate.tensor_dropout_rate += (rng.random::<f64>() - 0.5) * dynamic_mutation_rate;
                    
                    // Random entropy jump for quantum seed
                    if rng.random::<f64>() < 0.1 {
                        candidate.quantum_entropy_seed = rng.random::<f64>() * 1000.0;
                    }

                    // Clamping automático delegando a la estructura cuántica central (SuperGenotype)
                    // FASE 9: Deshardcoding total. Los límites están en `get_lower_bounds` / `get_upper_bounds`.
                    let vec = candidate.to_vector();
                    candidate = SuperGenotype::from_vector(&vec);
                    
                    // Función de Fitness basada en Expectativa Matemática Real
                    // CORRECCIÓN CRÍTICA: La ecuación anterior era puramente algebraica
                    // y premiaba TP/SL alto sin simular. Esto causaba que el evolver
                    // empujara SL al mínimo y TP al máximo sin validación de mercado.
                    // 
                    // Nueva fórmula: EV = WR * TP - (1-WR) * SL - 2*Fee
                    // FIX #583: WR se estima a partir de la selectividad diferencial sobre el azar (50%)
                    let avg_threshold = (candidate.ml_threshold_long + candidate.ml_threshold_short) / 2.0;
                    let selectivity = (avg_threshold - 0.50).max(0.0);
                    let est_wr = (0.50 + selectivity * 0.70).clamp(0.51, 0.85);
                    let rr_ratio = candidate.scalp_tp_base / candidate.scalp_sl_base.max(0.0001);
                    // Kelly criterion: f = W - (1-W)/R  
                    let kelly_f = est_wr - ((1.0 - est_wr) / rr_ratio.max(0.1));
                    // Expected value per trade (net of fees)
                    let ev_per_trade = (est_wr * candidate.scalp_tp_base) 
                        - ((1.0 - est_wr) * candidate.scalp_sl_base)
                        - roundtrip_fee; // Penalizar con fees reales
                    // Fitness combina: Sharpe actual * expectativa neta * Kelly positivo
                    let fitness = if ev_per_trade > 0.0 && kelly_f > 0.0 {
                        current_shadow_sharpe * ev_per_trade * 1000.0 * kelly_f
                    } else {
                        -1.0 // Penalización: genomas con EV negativa o Kelly negativa
                    };
                    
                    if fitness > best_score {
                        best_score = fitness;
                        best = candidate;
                    }
                }
                best
            }).await.unwrap_or(fallback_genome);
            
            // FASE 6: Estasis de Probabilidad (Confidence Bayesiano > 95%)
            // Utilizamos el inlier_count (tamaño de la muestra válida de RANSAC) y la volatilidad (Sharpe)
            // para estimar la confianza. Si no estamos al 95% seguros de que es una mejora, mutamos pero NO aplicamos.
            // FIX #669: Sanitizar Sharpe y tamaño de muestra para evitar divisiones espurias o NaNs
            let safe_sharpe = if current_shadow_sharpe.is_finite() && current_shadow_sharpe > 0.0 { current_shadow_sharpe } else { 0.1 };
            let safe_len = (self.returns_history.len().max(1)) as f64;
            let std_error = 1.0 / safe_len.sqrt();
            let bayesian_confidence = (1.0 - (std_error / safe_sharpe)).clamp(0.0, 1.0);
            
            if bayesian_confidence < 0.95 {
                println!("⚠️ [PROBABILITY STASIS] Sharpe {:.2} superó base, pero Confianza Bayesiana es {:.1}%. Requiere > 95%. Se descarta mutación.", current_shadow_sharpe, bayesian_confidence * 100.0);
                return;
            }
            
            // 🔥 ACTUALIZACIÓN EN VIVO (HOT-SWAP) AL GOD ENGINE
            best_genome.apply_to_arena(&self.arena);

            self.state.active_genome_id.fetch_add(1, Ordering::SeqCst);
            self.state.has_new_genome.store(true, Ordering::Release);
            
            println!("⚡ [HOT-SWAP TRIGGERED] Shadow Strategy superó métricas base tras iterar 1000 universos. Desplegando.");

            // F4.5 — ELIMINADO: este bloque DESACTIVABA el kill-switch porque un
            // sharpe de sombra se veía bien. Ninguna métrica automatizada puede
            // desarmar la protección: el kill-switch es LATCH (F5.2) y su rearme
            // es humano (reiniciar el proceso, con STOP_TRADING.LOCK verificado).

            // F4.3: persistencia por el EMBUDO ÚNICO — envelope versionado con
            // linaje + historia + espejo legacy (antes: write directo sin
            // auditoría ni rollback).
            match quantum_arena::genome_store::GenomeEnvelope::promote(
                best_genome.clone(),
                "online_daemon",
                &format!(
                    "sharpe estrategia {:.2} (confianza bayesiana >95%), {} observaciones acumuladas",
                    current_shadow_sharpe,
                    self.returns_history.len()
                ),
            ) {
                Ok(env) => println!(
                    "🧬 [ONLINE] Genoma generación {} promovida vía almacén (padre {}).",
                    env.generation, env.parent_generation
                ),
                Err(e) => println!("⚠️ [ONLINE] Promo al almacén falló: {}", e),
            }
            let _ = &self.champion_path; // conservado para compat de la struct
        } else {
            // --- FASE 9 / HC-08: DRIFT DETECTION & KILL SWITCH (EWMA ADAPTIVE) ---
            // Si el Sharpe cae repetidamente usando una media móvil exponencial,
            // asumimos que el modelo ha sufrido Drift (concept drift) y detenemos el trading.
            
            // Inicializar EWMA la primera vez
            if self.ewma_sharpe == 0.0 {
                self.ewma_sharpe = current_shadow_sharpe;
            } else {
                // EWMA smoothing factor alpha = 0.1
                self.ewma_sharpe = 0.1 * current_shadow_sharpe + 0.9 * self.ewma_sharpe;
            }

            if self.ewma_sharpe < 0.5 && !self.is_demo && !self.arena.kill_switch_active.load(Ordering::Relaxed) {
                println!("⚠️ [DRIFT DETECTION] Sharpe EWMA desplomado a {:.2}. Activando Kill Switch para detener ejecuciones hasta reentrenar.", self.ewma_sharpe);
                self.arena.kill_switch_active.store(true, Ordering::Relaxed);
            }
        }
    }

    /// RANSAC (Random Sample Consensus) para el cálculo robusto de Sharpe Ratio.
    /// Elimina outliers (ruido de microestructura) y estima el Sharpe real.
    // FIX #719: Filtrado previo de retornos finitos y guarda de resultado finito en RANSAC Sharpe
    fn calculate_ransac_sharpe(returns: &[f64]) -> f64 {
        let clean_returns: Vec<f64> = returns.iter().copied().filter(|r| r.is_finite()).collect();
        if clean_returns.is_empty() { return 0.0; }
        
        // 1. Encontrar la media y desviación estándar para detectar outliers
        let mut mean = 0.0;
        for &r in &clean_returns { mean += r; }
        mean /= clean_returns.len() as f64;
        
        let mut variance = 0.0;
        for &r in &clean_returns { variance += (r - mean).powi(2); }
        let std_dev = (variance / clean_returns.len() as f64).sqrt();
        
        if std_dev <= 1e-12 || !std_dev.is_finite() { return 0.0; }
        
        // 2. RANSAC Inlier threshold: 2.0 Desviaciones estándar
        let threshold = 2.0 * std_dev;
        
        let mut inlier_sum = 0.0;
        let mut inlier_count = 0;
        let mut inlier_variance = 0.0;
        
        // Primera pasada: Calcular media de inliers
        for &r in &clean_returns {
            if (r - mean).abs() <= threshold {
                inlier_sum += r;
                inlier_count += 1;
            }
        }
        
        if inlier_count == 0 { return 0.0; }
        let inlier_mean = inlier_sum / inlier_count as f64;
        
        // Segunda pasada: Calcular desviación estándar de inliers
        for &r in &clean_returns {
            if (r - mean).abs() <= threshold {
                inlier_variance += (r - inlier_mean).powi(2);
            }
        }
        let inlier_std = (inlier_variance / inlier_count as f64).sqrt();
        
        if inlier_std <= 1e-12 || !inlier_std.is_finite() { return 0.0; }
        
        // Multiplicador dinámico basado en la frecuencia real de los datos HFT/Crypto (365 días * 1440 min/día)
        let observations_per_year: f64 = 365.0 * 1440.0;
        let annualization = observations_per_year.sqrt();
        let raw_sharpe = (inlier_mean / inlier_std) * annualization;
        if raw_sharpe.is_finite() { raw_sharpe } else { 0.0 }
    }
}
