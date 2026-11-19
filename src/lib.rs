// DEMO-A: contrato de arranque del perfil explicito (contexto sellado del proceso).
pub mod arranque_entorno;
// CICLOS-2: ciclos completos y certificado v2 de la puerta de dinero real.
pub mod ciclos_completos;
pub mod config;
pub mod dark_alpha_router;
pub mod dark_alpha_sniffer;
pub mod dashboard;
pub mod demo_tel;
pub mod env_manager;
pub mod host_oms;
pub mod model_manifest;
pub mod orderbook;
pub mod parsers;
pub mod symbol_manager;
pub mod sa_params;
// TGBG-3: puente comun de admision de generacion de cintas REAL (train_forest y dataset_admission).
pub mod tgb_generation;

// NOTA: Las funciones FFI extern "C" (quantum_ring_new, quantum_ring_free, quantum_ring_read_tick)
// han sido eliminadas en la Fase 4 de Auditoría Suprema.
// Razón: Eran residuos de la era Python. Ningún módulo Rust las invocaba.
// El sistema es 100% Rust nativo — no necesita exports C.
