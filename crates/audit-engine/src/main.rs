use std::sync::Arc;
use phase_runner::{Phase, AdaptiveTimer, PhaseExecutor, PhaseResult};
use omniscient_registry::OmniscientRegistry;
use graph_4d::SystemGraph;
use std::path::PathBuf;
use crossbeam::channel;
use std::thread;
use std::time::Duration;

#[tokio::main]
async fn main() {
    println!("============================================================");
    println!("🔍 SUPREME QUANTUM AUDIT ENGINE ACTIVATED");
    println!("[SAGES COUNCIL] Inicializando Audit Engine Cuantico...");

    let registry = Arc::new(OmniscientRegistry::new());
    let mut current_phase = Phase::Alpha;
    
    // Adaptive Timer: 5 seconds base interval, assuming 8000 MB max memory for tracking
    let timer = AdaptiveTimer::new(5000, 8000);

    // Bounded channel for lock-free PhaseResult reporting
    let (result_tx, result_rx) = channel::bounded::<PhaseResult>(100);

    // Spawn an isolated background thread to handle results without blocking the hot path
    thread::spawn(move || {
        while let Ok(result) = result_rx.recv() {
            println!("✅ [RESULT LOGGER] Phase {} completed in {} ms.", result.phase.to_str(), result.duration_ms);
            for finding in result.findings {
                println!("   -> 🔬 Finding: {}", finding);
            }
        }
    });

    // Initial Graph Build (Phase Alpha)
    let mut system_graph = SystemGraph::new();
    println!("Phase [Alpha]: Building System Graph from Cargo Workspace...");
    let workspace_root = PathBuf::from(".");
    let crates_dir = workspace_root.join("crates");

    if let Ok(entries) = std::fs::read_dir(crates_dir) {
        for entry in entries.flatten() {
            if entry.file_type().map(|ft| ft.is_dir()).unwrap_or(false) {
                let src_dir = entry.path().join("src");
                if src_dir.exists() {
                    let lib_file = src_dir.join("lib.rs");
                    if lib_file.exists() {
                        if let Err(e) = system_graph.parse_file(&lib_file) {
                            eprintln!("Warning: Failed to parse {}: {}", lib_file.display(), e);
                        }
                    }
                }
            }
        }
    }
    
    println!("Phase [Alpha]: System Graph built. Nodes: {}", system_graph.graph.node_count());

    // Enter Infinite Audit Loop
    println!("🔄 Entering Perpetual Audit Loop...");
    
    loop {
        current_phase = current_phase.next();
        println!("🚀 Executing Phase: [{}]", current_phase.to_str());
        
        let result = match current_phase {
            Phase::Beta => {
                let collisions = registry.detect_collisions();
                if !collisions.is_empty() {
                    println!("⚠️ WARNING: Detected parameter collisions: {:?}", collisions);
                }
                PhaseExecutor::run(current_phase, Duration::from_millis(5000))
            },
            Phase::Gamma => {
                let params = registry.scan_all();
                println!("📊 Monitored parameters in registry: {}", params.len());
                PhaseExecutor::run(current_phase, Duration::from_millis(5000))
            },
            _ => PhaseExecutor::run(current_phase, Duration::from_millis(5000))
        };

        // Enqueue result lock-free
        if let Err(e) = result_tx.try_send(result) {
            eprintln!("⚠️ WARNING: Result channel full, dropping result: {}", e);
        }
        
        // Wait adaptively to avoid stealing resources from live_trader / god_engine
        timer.wait_next_cycle().await;
    }
}
