pub mod config;
pub mod dark_alpha_router;
pub mod dark_alpha_sniffer;
pub mod dashboard;
pub mod multi_asset_orchestrator;
pub mod orderbook;
pub mod parsers;
pub mod quantum_arena;
pub mod symbol_manager;
pub mod trailing;
pub use quantum_arena::{QuantumRingBuffer, QuantumStateArena, FEATURE_SIZE};
pub use trailing::{evaluate_quantum_trailing, TrailingResult};

// NOTA: Las funciones FFI extern "C" (quantum_ring_new, quantum_ring_free, quantum_ring_read_tick)
// han sido eliminadas en la Fase 4 de Auditoría Suprema.
// Razón: Eran residuos de la era Python. Ningún módulo Rust las invocaba.
// El sistema es 100% Rust nativo — no necesita exports C.
