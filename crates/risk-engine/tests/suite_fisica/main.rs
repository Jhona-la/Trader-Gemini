//! Suite de pruebas de risk-engine (tarea 3 del integrador): contratos fNN_001 de analogias fisicas (geometria, cuantica, gravedad).
//! Un solo binario de prueba; cada modulo es el antiguo `tests/<modulo>.rs` sin cambios
//! salvo las rutas de `include_str!`. Solo entran archivos que no escriben estado global
//! del proceso (registro de simbolos, contadores de rechazo, variables de entorno).

mod f102_001_drgt_massive_gravity_contract;
mod f105_001_kerr_newman_contract;
mod f30_001_ville_grossman_ignition_contract;
mod f36_001_adaptive_quantum_veto_contract;
mod f42_001_tensor_network_mps_dmrg_contract;
mod f51_001_quantum_no_arbitrage_conformal_contract;
mod f54_001_chern_simons_berry_phase_contract;
mod f57_001_gromov_symplectic_barrier_contract;
mod f60_001_reeb_contact_vortex_contract;
mod f63_001_umegaki_relative_entropy_contract;
mod f66_001_dirac_monopole_chern_bundle_contract;
mod f69_001_polyakov_schwarzschild_horizon_contract;
mod f72_001_lindblad_master_decoherence_contract;
mod f75_001_ryu_takayanagi_entanglement_contract;
mod f78_001_toric_code_stabilizer_contract;
mod f84_001_bekenstein_mukhanov_quantization_contract;
mod f87_001_holographic_lloyd_complexity_contract;
mod f90_001_quantum_extremal_surface_contract;
mod f93_001_fisher_rao_amari_contract;
mod f96_001_quaternionic_hyperkahler_contract;
mod f99_001_asymptotic_safety_contract;
