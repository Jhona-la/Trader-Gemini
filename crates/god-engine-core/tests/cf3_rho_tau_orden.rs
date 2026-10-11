//! C-F3 · F3.2-02 (Línea C): el veto de grupo debe leer el ρ espectral a la
//! τ de la orden, no a la τ en que el núcleo lo publicó.
//!
//! Teoría que se prueba: la dependencia entre dos activos es una función de
//! la escala, ρ(τ). El multiactivo (`EspectralMultiactivo`) mide un IC por
//! par y escala; el núcleo publica `qo_613_rho_tau` a la τ espectral del
//! tick y el risk-engine lo usa para apretar el ρ̄ del grupo
//! (`qo_613_aprietes`). Si la orden vive a otra escala, el ρ que la juzga es
//! el de una escala donde el grupo puede estar acoplado de otra manera.
//! Verdad del fixture: IC = 1 en la escala A = 19 (274,9 s) y sin evidencia
//! (None) en B = 20 (1 099,5 s).
//!
//! Fixture: copia literal de evidence_expiry_contract.rs (multiactivo
//! maduro con 120 pares alternos ±0,02, arena de riesgo y un par
//! correlacionado abierto con ρ escalar ∈ (0,5; 0,8)).
//!
//! Convención (fase C-F3): un test que hoy falla por la causa de un hallazgo
//! lleva `#[ignore = "OPEN F3.x-nn (dueño): causa"]` con el assert intacto.

use god_engine_core::evidence_publication::publicar_coherencia;
use quantum_arena::{
    espectral_multiactivo::EspectralMultiactivo, state::CompactTick,
    temporal_spectrum::SPECTRUM_SCALES_MS, GlobalArena,
};
use risk_engine::RiskEngine;
use signal_engine::{SignalIntent, SignalType};
use std::sync::{atomic::Ordering::Relaxed, Arc, Once};

const A: usize = 19; // 274,9 s
const B: usize = 20; // 1 099,5 s

// Copia literal de evidence_expiry_contract.rs (multiactivo_maduro).
fn multiactivo_maduro() -> EspectralMultiactivo {
    let mut ma = EspectralMultiactivo::new(3);
    let tau = SPECTRUM_SCALES_MS[A];
    for i in 0..120u64 {
        let ts = ((i + 1) as f64 * tau) as u64;
        let r = if i % 2 == 0 { 0.02 } else { -0.02 };
        ma.observar_maduracion(0, tau, A, ts, r);
        ma.observar_maduracion(1, tau, A, ts, r);
    }
    assert_eq!(ma.coherencia_media_con_todas(0, A), Some(1.0));
    assert_eq!(ma.coherencia_media_con_todas(0, B), None);
    ma
}

// Copia literal de evidence_expiry_contract.rs (risk_fixture).
fn risk_fixture() -> (Arc<GlobalArena>, SignalIntent) {
    static SPECS: Once = Once::new();
    SPECS.call_once(|| {
        quantum_arena::symbol_registry::update_registry(
            ["BTCUSDT", "ETHUSDT"]
                .iter()
                .map(|symbol| quantum_arena::symbol_registry::SymbolSpec {
                    symbol: (*symbol).into(),
                    step_size: 0.001,
                    tick_size: 0.01,
                    min_qty: 0.001,
                    min_notional: 5.0,
                    max_leverage: 1,
                    maker_fee: 0.0002,
                    taker_fee: 0.0004,
                    is_shadow: false,
                })
                .collect(),
        );
    });
    let arena = GlobalArena::build_in_own_stack(100.0);
    let coin = &arena.coins[0];
    coin.current_price.store(100.0, Relaxed);
    coin.current_atr.store(1.0, Relaxed);
    coin.hurst_exponent.store(0.5, Relaxed);
    coin.metrics.trade_count.store(100, Relaxed);
    coin.metrics.profit_factor.store(2.0, Relaxed);
    coin.metrics.win_rate.store(0.75, Relaxed);
    coin.metrics.kelly_fraction.store(0.15, Relaxed);
    arena.config.latency_penalty_ms.store(0.0, Relaxed);
    arena.config.live_taker_fee.store(0.0004, Relaxed);
    arena.config.base_slippage_floor.store(0.00001, Relaxed);
    arena.config.global_max_drawdown.store(0.2, Relaxed);
    arena.config.kelly_clamp_min.store(0.01, Relaxed);
    arena.config.kelly_clamp_max.store(0.25, Relaxed);
    arena
        .config
        .global_correlation_threshold
        .store(0.5, Relaxed);
    arena
        .riesgo_por_operacion
        .store(0.75 * risk_engine::ruin::clamp_ruin(1.0, 0.25), Relaxed);
    (
        arena,
        SignalIntent {
            signal: SignalType::Long,
            confidence: 0.9,
            expected_duration_ms: 60_000,
            ..SignalIntent::default()
        },
    )
}

// Copia literal de evidence_expiry_contract.rs (open_correlated_peer).
fn open_correlated_peer(arena: &GlobalArena) {
    for i in 0..200u64 {
        let t = i as f64 * 0.7;
        for coin in 0..2 {
            let change = if coin == 0 {
                t.sin()
            } else {
                0.6 * t.sin() + 0.8 * t.cos()
            };
            let price = 100.0 * (0.01 * change).exp();
            arena.coins[coin].tick_ring.push(CompactTick {
                timestamp: 1000 + i * 10,
                bid_price: price,
                ask_price: price,
                bid_qty: 1.0,
                ask_qty: 1.0,
            });
        }
    }
    arena.coins[1].positions.slots()[0].open(true, 100.0, 8.0, 0.1, 1000, 102.0, 99.0);
    let d = risk_engine::correlation_guard::dependency_exposure(arena, 0, true, 0.5).unwrap();
    assert_eq!(d.same_bet_positions, 1);
    let rho = d.same_bet_rho_efectivo().unwrap();
    assert!(rho > 0.5 && rho < 0.8, "rho={rho}");
}

/// El núcleo publica ρ a la τ espectral (A). La orden va a `tau_orden_ms`.
/// Devuelve cuántas veces apretó el veto de grupo (0 o 1).
fn aprietes_con_orden_a(tau_orden_ms: f64) -> f64 {
    let (arena, mut intent) = risk_fixture();
    open_correlated_peer(&arena);
    let ma = multiactivo_maduro();
    publicar_coherencia(&arena.registry, &ma, 0, SPECTRUM_SCALES_MS[A]);
    intent.expected_duration_ms = tau_orden_ms as u64;
    RiskEngine::new(100.0).evaluate_quantum_order(0, &intent, &arena);
    arena.registry.get_for_coin_or(0, "qo_613_aprietes", 0.0)
}

/// Control: orden y ρ publicados en la misma escala A ⇒ el veto aprieta con
/// el IC maduro (1,0 > ρ escalar). Exacto: el contador sube de 0 a 1.
#[test]
fn cf3_control_rho_y_orden_en_la_misma_escala_aprieta() {
    let n = aprietes_con_orden_a(SPECTRUM_SCALES_MS[A]);
    println!("observado: aprietes con la orden a τ_A = {n}");
    assert_eq!(n, 1.0);
}

/// Verdad: en la escala B el multiactivo no tiene evidencia (None), así que
/// el veto de grupo de una orden a τ_B no debe apretar con el ρ maduro de la
/// escala A. Exacto: 0 aprietes.
#[test]
#[ignore = "OPEN F3.2-02 (Línea C, publicación del núcleo y lector del risk-engine): qo_613_rho_tau no lleva τ; el lector lo aplica a cualquier orden sin compararlo con su τ de dimensionamiento"]
fn cf3_rho_del_veto_de_grupo_es_el_de_la_tau_de_la_orden() {
    let n = aprietes_con_orden_a(SPECTRUM_SCALES_MS[B]);
    println!("observado: aprietes con la orden a τ_B = {n}");
    assert_eq!(
        n, 0.0,
        "el veto de grupo usó el ρ de τ_A para una orden a τ_B"
    );
}
