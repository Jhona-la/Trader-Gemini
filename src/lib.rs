pub mod parsers;
pub mod dark_alpha_router;
pub mod dark_alpha_sniffer;
pub mod trailing;
pub mod quantum_arena;
pub mod orderbook;
pub mod config;
pub mod dashboard;
pub mod multi_asset_orchestrator;
pub mod symbol_manager;
pub use quantum_arena::{QuantumRingBuffer, FEATURE_SIZE, QuantumStateArena};
pub use trailing::{evaluate_quantum_trailing, TrailingResult};

// NOTA: Las funciones FFI extern "C" (quantum_ring_new, quantum_ring_free, quantum_ring_read_tick)
// han sido eliminadas en la Fase 4 de Auditoría Suprema.
// Razón: Eran residuos de la era Python. Ningún módulo Rust las invocaba.
// El sistema es 100% Rust nativo — no necesita exports C.
