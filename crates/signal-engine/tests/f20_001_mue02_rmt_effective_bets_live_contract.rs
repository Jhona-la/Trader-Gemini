//! F20-001 / MUE-02 — contrato del PRODUCTOR VIVO de `rmt_effective_bets_ratio`.
//!
//! Hallazgo de auditoría: `TensorVoteOrchestrator` leía `rmt_effective_bets_ratio`
//! (default 1.0) para descontar el boost de ensamble cuando las voces son colineales,
//! pero en producción NADIE escribía esa clave: sólo los tests la fijaban a mano. El
//! descuento de Marchenko-Pastur era una arista muerta del grafo.
//!
//! Este contrato fija:
//! - C1: forma cerrada exacta en la matriz equicorrelacionada (λ₁ = 1+(m−1)ρ, bulk 1−ρ).
//! - C2: identidad ⇒ ratio 1.0 y n_eff = m (sin modos sobre el borde MP).
//! - C3: dominio inválido ⇒ None (nunca un ratio inventado).
//! - C4: monotonía: más colinealidad ⇒ menos apuestas efectivas.
//! - C5: cableado vivo — voces colineales variables publican el ratio sin intervención
//!   y atenúan la confianza frente al mismo instante sin calibrar.
//! - C6: voces independientes y clones de UNA fuente no se descuentan (columna por fuente).
//! - C7: arranque frío / voces constantes ⇒ no se publica nada (paridad bit a bit).
//! - C8: una clave por moneda fijada externamente conserva precedencia.

use omniscient_registry::OmniscientRegistry;
use quantum_arena::GlobalArena;
use signal_engine::orchestrator::{
    compute_rmt_voter_effective_bets_ratio, TensorVoteOrchestrator, RMT_SAMPLE_STRIDE,
    RMT_WINDOW_SAMPLES,
};
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;
use strategy_core::QuantumStrategy;

/// splitmix64 determinista ⇒ uniforme en [-1, 1].
fn u(tick: u64, salt: u64) -> f64 {
    let mut z = tick
        .wrapping_mul(0x9E37_79B9_7F4A_7C15)
        .wrapping_add(salt.wrapping_mul(0xBF58_476D_1CE4_E5B9))
        .wrapping_add(0x94D0_49BB_1331_11EB);
    z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    z ^= z >> 31;
    ((z >> 11) as f64 / (1u64 << 53) as f64) * 2.0 - 1.0
}

#[derive(Clone, Copy)]
enum Modo {
    /// voto = 0.3 + 0.2·(f + 0.1·eᵢ)/1.1 — factor común dominante (ρ ≈ 0.99).
    Colineal,
    /// voto = 0.3 + 0.2·eᵢ — independientes.
    Independiente,
    /// voto constante.
    Constante,
}

struct Voz {
    name: &'static str,
    salt: u64,
    modo: Modo,
    tick: Arc<AtomicU64>,
}

impl QuantumStrategy for Voz {
    fn name(&self) -> &str {
        self.name
    }
    fn init(&mut self, _registry: Arc<OmniscientRegistry>) -> Result<(), String> {
        Ok(())
    }
    fn evaluate(&self) -> f64 {
        let t = self.tick.load(Ordering::Relaxed);
        match self.modo {
            Modo::Colineal => 0.3 + 0.2 * (u(t, 0) + 0.1 * u(t, self.salt)) / 1.1,
            Modo::Independiente => 0.3 + 0.2 * u(t, self.salt),
            Modo::Constante => 0.4,
        }
    }
}

fn arena() -> Arc<GlobalArena> {
    let arena = GlobalArena::build_in_own_stack(13.0);
    arena
        .config
        .min_confidence_btc
        .store(0.50, Ordering::Relaxed);
    arena
}

fn orquestador(
    arena: &Arc<GlobalArena>,
    voces: &[(&'static str, u64, Modo)],
    tick: &Arc<AtomicU64>,
) -> TensorVoteOrchestrator {
    let mut orch = TensorVoteOrchestrator::new(Arc::clone(arena));
    for &(name, salt, modo) in voces {
        orch.add_strategy(Box::new(Voz {
            name,
            salt,
            modo,
            tick: Arc::clone(tick),
        }));
    }
    orch
}

/// Evaluaciones suficientes para llenar la ventana y cruzar la cadencia 1024 del censista.
fn calentar(orch: &TensorVoteOrchestrator, tick: &Arc<AtomicU64>) {
    let n = RMT_SAMPLE_STRIDE as usize * RMT_WINDOW_SAMPLES + 64;
    for t in 0..n as u64 {
        tick.store(t, Ordering::Relaxed);
        let _ = orch.evaluate_continuous_consensus_for_coin(0, "BTCUSDT");
    }
}

fn equicorrelada(m: usize, rho: f64) -> Vec<f64> {
    let mut c = vec![rho; m * m];
    for i in 0..m {
        c[i * m + i] = 1.0;
    }
    c
}

#[test]
fn c1_forma_cerrada_equicorrelada() {
    // m = 6, ρ = 0.9, T = 64: λ₁ = 5.5, bulk 5×0.1, borde (1+√(6/64))² ≈ 1.706.
    let (ratio, n_eff, edge) =
        compute_rmt_voter_effective_bets_ratio(&equicorrelada(6, 0.9), 6, 64).expect("dominio");
    let edge_esperado = (1.0 + (6.0f64 / 64.0).sqrt()).powi(2);
    let n_eff_esperado = 36.0 / (5.5 * 5.5 + 5.0 * 0.1 * 0.1);
    assert!((edge - edge_esperado).abs() < 1e-12, "borde {edge}");
    assert!(
        (n_eff - n_eff_esperado).abs() < 1e-6,
        "n_eff {n_eff} vs {n_eff_esperado}"
    );
    assert!((ratio - n_eff_esperado / 6.0).abs() < 1e-6, "ratio {ratio}");
}

#[test]
fn c2_identidad_sin_descuento() {
    let m = 8;
    let (ratio, n_eff, _) =
        compute_rmt_voter_effective_bets_ratio(&equicorrelada(m, 0.0), m, 64).expect("dominio");
    assert_eq!(ratio, 1.0);
    assert_eq!(n_eff, m as f64);
}

#[test]
fn c3_dominio_invalido_devuelve_none() {
    // T ≤ m
    assert!(compute_rmt_voter_effective_bets_ratio(&equicorrelada(6, 0.5), 6, 6).is_none());
    // m < 2 y m > 32
    assert!(compute_rmt_voter_effective_bets_ratio(&[1.0], 1, 64).is_none());
    assert!(compute_rmt_voter_effective_bets_ratio(&equicorrelada(33, 0.1), 33, 128).is_none());
    // diagonal no unitaria
    let mut c = equicorrelada(4, 0.2);
    c[0] = 2.0;
    assert!(compute_rmt_voter_effective_bets_ratio(&c, 4, 64).is_none());
    // asimetría, NaN y longitud incorrecta
    let mut c = equicorrelada(4, 0.2);
    c[1] = 0.9;
    assert!(compute_rmt_voter_effective_bets_ratio(&c, 4, 64).is_none());
    let mut c = equicorrelada(4, 0.2);
    c[1] = f64::NAN;
    c[4] = f64::NAN;
    assert!(compute_rmt_voter_effective_bets_ratio(&c, 4, 64).is_none());
    assert!(compute_rmt_voter_effective_bets_ratio(&equicorrelada(4, 0.2), 5, 64).is_none());
}

#[test]
fn c4_monotonia_en_la_colinealidad() {
    let r = |rho: f64| {
        compute_rmt_voter_effective_bets_ratio(&equicorrelada(6, rho), 6, 64)
            .expect("dominio")
            .0
    };
    let (r3, r6, r9) = (r(0.3), r(0.6), r(0.9));
    assert!(r3 > r6 && r6 > r9, "no monótono: {r3} {r6} {r9}");
    assert!(r3 <= 1.0 && r9 >= 0.10);
}

fn voces(modo: Modo) -> Vec<(&'static str, u64, Modo)> {
    (0..6u64)
        .map(|i| {
            let name: &'static str = Box::leak(format!("F20_{i}").into_boxed_str());
            (name, i + 1, modo)
        })
        .collect()
}

#[test]
fn c5_cableado_vivo_publica_y_atenua_voces_colineales() {
    let tick = Arc::new(AtomicU64::new(0));
    let a = arena();
    let orch = orquestador(&a, &voces(Modo::Colineal), &tick);
    calentar(&orch, &tick);

    // Publicado por la cadencia del censista, sin llamada explícita.
    let ratio = a.registry.get_value_or("rmt_effective_bets_ratio", -1.0);
    let n_eff = a.registry.get_value_or("rmt_effective_bets_measured", -1.0);
    assert!(ratio > 0.0 && ratio < 0.35, "ratio colineal {ratio}");
    assert!((1.0..2.0).contains(&n_eff), "n_eff colineal {n_eff}");
    assert_eq!(
        a.registry.get_value_or("rmt_active_sources_measured", -1.0),
        6.0
    );

    // Mismo instante en un orquestador sin calibrar (ratio por defecto 1.0).
    let t_eval = 5_000;
    tick.store(t_eval, Ordering::Relaxed);
    let calibrada = orch.evaluate_continuous_consensus_for_coin(0, "BTCUSDT");
    let control_arena = arena();
    let control = orquestador(&control_arena, &voces(Modo::Colineal), &tick);
    let fria = control.evaluate_continuous_consensus_for_coin(0, "BTCUSDT");
    assert!(
        calibrada.net_confidence < fria.net_confidence * 0.80,
        "el descuento RMT no llegó a la decisión: {} vs {}",
        calibrada.net_confidence,
        fria.net_confidence
    );
}

#[test]
fn c6_independientes_y_clones_de_una_fuente_no_se_descuentan() {
    // Seis voces independientes.
    let tick = Arc::new(AtomicU64::new(0));
    let a = arena();
    let orch = orquestador(&a, &voces(Modo::Independiente), &tick);
    calentar(&orch, &tick);
    let ratio = a.registry.get_value_or("rmt_effective_bets_ratio", -1.0);
    assert!(ratio >= 0.80, "independientes descontadas: {ratio}");

    // Tres instancias de UNA fuente + tres independientes: cuatro columnas, no seis.
    // Sin columna por fuente los tres clones parecerían colineales y se descontarían.
    let tick = Arc::new(AtomicU64::new(0));
    let a = arena();
    let mut mezcla = vec![
        ("F20_CLON", 1u64, Modo::Independiente),
        ("F20_CLON", 1, Modo::Independiente),
        ("F20_CLON", 1, Modo::Independiente),
    ];
    mezcla.extend(voces(Modo::Independiente).into_iter().skip(3));
    let orch = orquestador(&a, &mezcla, &tick);
    calentar(&orch, &tick);
    assert_eq!(
        a.registry.get_value_or("rmt_active_sources_measured", -1.0),
        4.0
    );
    let ratio = a.registry.get_value_or("rmt_effective_bets_ratio", -1.0);
    assert!(ratio >= 0.80, "clones de una fuente descontados: {ratio}");
}

#[test]
fn c7_voces_constantes_no_publican() {
    let tick = Arc::new(AtomicU64::new(0));
    let a = arena();
    let orch = orquestador(&a, &voces(Modo::Constante), &tick);
    calentar(&orch, &tick);
    assert_eq!(
        a.registry.get_value_or("rmt_effective_bets_ratio", -1.0),
        -1.0
    );
    assert!(orch.calibrar_rmt_apuestas_efectivas().is_none());
}

#[test]
fn c8_clave_por_moneda_tiene_precedencia() {
    let tick = Arc::new(AtomicU64::new(0));
    let a = arena();
    let orch = orquestador(&a, &voces(Modo::Colineal), &tick);
    calentar(&orch, &tick);
    assert!(a.registry.get_value_or("rmt_effective_bets_ratio", -1.0) < 0.35);

    a.registry.set_for_coin(0, "rmt_effective_bets_ratio", 1.0);
    let t_eval = 7_000;
    tick.store(t_eval, Ordering::Relaxed);
    let con_override = orch.evaluate_continuous_consensus_for_coin(0, "BTCUSDT");
    let control_arena = arena();
    let control = orquestador(&control_arena, &voces(Modo::Colineal), &tick);
    let fria = control.evaluate_continuous_consensus_for_coin(0, "BTCUSDT");
    assert!(
        (con_override.net_confidence - fria.net_confidence).abs() < 1e-12,
        "override por moneda ignorado: {} vs {}",
        con_override.net_confidence,
        fria.net_confidence
    );
}
