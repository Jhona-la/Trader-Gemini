//! Domain and live-admission contracts, no orders, exchange or engine process.
use quantum_arena::{state::CompactTick, GlobalArena};
use risk_engine::correlation_guard::{
    rho_efectivo_para_agregacion, rho_promedio, CorrelationGuardEngine,
};
use risk_engine::{RiskEngine, REJECT_COUNTERS};
use signal_engine::{SignalIntent, SignalType};
use std::sync::{atomic::Ordering::Relaxed, Arc, Once};

fn matrix(n: usize, rho: f64) -> Vec<Vec<f64>> {
    let mut c = vec![vec![rho; n]; n];
    for i in 0..n {
        c[i][i] = 1.0;
    }
    c
}

#[test]
fn incomplete_or_invalid_matrices_never_produce_a_group_correlation() {
    let mut partial = matrix(3, 0.2);
    partial[0][2] = f64::NAN;
    partial[2][0] = f64::NAN;
    let mut asymmetric = matrix(3, 0.2);
    asymmetric[2][0] = 0.4;
    let mut invalid_diagonal = matrix(3, 0.2);
    invalid_diagonal[1][1] = 2.0;
    for (name, c) in [
        ("partial", partial),
        ("asymmetric", asymmetric),
        ("diagonal", invalid_diagonal),
        ("range", matrix(3, 1.1)),
        ("negative variance", matrix(5, -0.4)),
    ] {
        assert!(rho_promedio(&c).is_none(), "{name}");
        assert!(rho_efectivo_para_agregacion(&c).is_none(), "{name}");
    }
}

#[test]
fn complete_valid_negative_and_positive_dependence_is_not_erased() {
    for (n, rho) in [(2, -1.0), (5, -0.25), (5, -0.2), (5, 0.0), (5, 0.8)] {
        assert!((rho_promedio(&matrix(n, rho)).unwrap() - rho).abs() < 1e-12);
    }
}

#[test]
fn invalid_correlation_does_not_look_like_a_hedge() {
    for bad in [-2.0, f64::NAN, f64::INFINITY, f64::NEG_INFINITY] {
        assert!(
            CorrelationGuardEngine::es_la_misma_apuesta(Some(bad), 0.5),
            "{bad}"
        );
    }
    assert!(!CorrelationGuardEngine::es_la_misma_apuesta(
        Some(-0.9),
        0.5
    ));
    assert!(!CorrelationGuardEngine::es_la_misma_apuesta(Some(0.0), 0.5));
}

#[test]
fn impossible_equicorrelation_never_grants_a_variance_discount() {
    let cap = risk_engine::ruin::clamp_ruin(1.0, 0.5);
    for rho in [-0.2501, -0.4, -1.0, -2.0] {
        assert!(
            CorrelationGuardEngine::veto_por_exposicion_estructural(4, cap / 2.0, 0.5, Some(rho)),
            "rho={rho}"
        );
    }
}

#[test]
fn feasible_negative_correlation_remains_a_valid_algebraic_input() {
    let cap = risk_engine::ruin::clamp_ruin(1.0, 0.5);
    assert!(!CorrelationGuardEngine::veto_por_exposicion_estructural(
        4,
        cap / 2.0,
        0.5,
        Some(-0.2)
    ));
    assert!(CorrelationGuardEngine::veto_por_exposicion_estructural(
        4,
        cap / 2.0,
        0.5,
        Some(0.0)
    ));
}

fn fixture() -> (Arc<GlobalArena>, SignalIntent) {
    static REGISTRY: Once = Once::new();
    REGISTRY.call_once(|| {
        quantum_arena::symbol_registry::update_registry(
            ["BTCUSDT", "ETHUSDT"]
                .iter()
                .map(|symbol| quantum_arena::symbol_registry::SymbolSpec {
                    symbol: (*symbol).into(),
                    step_size: 0.001,
                    tick_size: 0.01,
                    min_qty: 0.001,
                    min_notional: 5.0,
                    max_leverage: 20,
                    maker_fee: 0.0002,
                    taker_fee: 0.0004,
                    is_shadow: false,
                })
                .collect(),
        )
    });
    let arena = GlobalArena::build_in_own_stack(100.0);
    arena.coins[0].current_price.store(100.0, Relaxed);
    arena.coins[0].current_atr.store(1.0, Relaxed);
    arena.coins[0].hurst_exponent.store(0.5, Relaxed);
    arena.coins[0].metrics.trade_count.store(100, Relaxed);
    arena.coins[0].metrics.profit_factor.store(2.0, Relaxed);
    arena.coins[0].metrics.win_rate.store(0.75, Relaxed);
    arena.coins[0].metrics.kelly_fraction.store(0.15, Relaxed);
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
    let cap = risk_engine::ruin::clamp_ruin(1.0, 0.25);
    arena.riesgo_por_operacion.store(0.75 * cap, Relaxed);
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

fn open(arena: &GlobalArena, coin: usize, slot: usize, is_long: bool) {
    // XLVI·E (SPECTRAL-010): qty=8 da al miembro same-bet un riesgo REAL
    // al stop de 8·|100−99|/100 = 8% del capital — posición realista cuyo
    // riesgo medido PESA en la agregación del veto. Con el qty=0.01 legado
    // (riesgo real 0.01%) el veto por riesgo medido no ve al miembro y los
    // tests de conteo doctrinal perdían su fuerza.
    arena.coins[coin].positions.slots()[slot].open(
        is_long,
        100.0,
        8.0,
        0.1,
        1000,
        if is_long { 102.0 } else { 98.0 },
        if is_long { 99.0 } else { 101.0 },
    );
    assert!(arena.coins[coin].positions.slots()[slot].is_open());
}

fn pair_ticks(arena: &GlobalArena, anticorrelated: bool) {
    for i in 0..200 {
        let log_change = 0.01 * (i as f64 * 0.7).sin();
        for coin in 0..2 {
            let sign = if coin == 1 && anticorrelated {
                -1.0
            } else {
                1.0
            };
            let price = 100.0 * (sign * log_change).exp();
            arena.coins[coin].tick_ring.push(CompactTick {
                timestamp: 1000 + i * 10,
                bid_price: price,
                ask_price: price,
                bid_qty: 1.0,
                ask_qty: 1.0,
            });
        }
    }
}

#[test]
fn no_positions_does_not_add_a_blanket_correlation_veto() {
    let (arena, intent) = fixture();
    assert_eq!(
        RiskEngine::new(100.0)
            .evaluate_quantum_order(0, &intent, &arena)
            .signal,
        SignalType::Long
    );
}

#[test]
fn missing_evidence_is_not_imputed_to_independence_in_live_admission() {
    for slot in 0..3 {
        for side in [true, false] {
            let (arena, intent) = fixture();
            open(&arena, 1, slot, side);
            let before = REJECT_COUNTERS[2].load(Relaxed);
            let order = RiskEngine::new(100.0).evaluate_quantum_order(0, &intent, &arena);
            assert_eq!(order.signal, SignalType::Flat, "slot={slot},long={side}");
            assert!(REJECT_COUNTERS[2].load(Relaxed) > before);
        }
    }
}

#[test]
fn all_spectral_slots_of_the_same_asset_are_counted() {
    for slot in 0..3 {
        let (arena, intent) = fixture();
        open(&arena, 0, slot, true);
        assert_eq!(
            RiskEngine::new(100.0)
                .evaluate_quantum_order(0, &intent, &arena)
                .signal,
            SignalType::Flat,
            "slot={slot}"
        );
    }
}

#[test]
fn all_spectral_slots_of_other_assets_are_counted() {
    for slot in 0..3 {
        let (arena, intent) = fixture();
        pair_ticks(&arena, false);
        open(&arena, 1, slot, true);
        assert_eq!(
            RiskEngine::new(100.0)
                .evaluate_quantum_order(0, &intent, &arena)
                .signal,
            SignalType::Flat,
            "slot={slot}"
        );
    }
}

#[test]
fn opposite_side_on_anticorrelated_asset_is_concentrated_pnl() {
    let (arena, intent) = fixture();
    pair_ticks(&arena, true);
    open(&arena, 1, 2, false);
    assert_eq!(
        RiskEngine::new(100.0)
            .evaluate_quantum_order(0, &intent, &arena)
            .signal,
        SignalType::Flat
    );
}

#[test]
fn measured_hedge_is_not_mistaken_for_same_bet() {
    let (arena, intent) = fixture();
    pair_ticks(&arena, true);
    open(&arena, 1, 2, true);
    assert_eq!(
        RiskEngine::new(100.0)
            .evaluate_quantum_order(0, &intent, &arena)
            .signal,
        SignalType::Long
    );
}

#[test]
fn dependency_evidence_retains_unknown_counts_separately() {
    use risk_engine::correlation_guard::dependency_exposure;
    let (arena, _) = fixture();
    for slot in 0..3 {
        open(&arena, 1, slot, slot != 1);
    }
    open(&arena, 0, 0, true);
    let e = dependency_exposure(&arena, 0, true, 0.5).unwrap();
    assert_eq!(e.open_positions, 4);
    assert_eq!(e.same_bet_positions, 4);
    assert_eq!(e.unknown_positions, 3);
}

#[test]
fn dependency_transform_covers_all_signs_and_slots() {
    use risk_engine::correlation_guard::dependency_exposure;
    for anticorrelated in [false, true] {
        for candidate_long in [false, true] {
            for position_long in [false, true] {
                for slot in 0..3 {
                    let (arena, _) = fixture();
                    pair_ticks(&arena, anticorrelated);
                    open(&arena, 1, slot, position_long);
                    let e = dependency_exposure(&arena, 0, candidate_long, 0.5).unwrap();
                    let concentrated = (candidate_long == position_long) != anticorrelated;
                    assert_eq!(e.open_positions, 1);
                    assert_eq!(e.unknown_positions, 0);
                    assert_eq!(e.same_bet_positions, usize::from(concentrated));
                }
            }
        }
    }
}

#[test]
fn same_asset_opposite_side_is_signed_without_invented_market_samples() {
    use risk_engine::correlation_guard::dependency_exposure;
    let (arena, _) = fixture();
    open(&arena, 0, 0, false);
    let e = dependency_exposure(&arena, 0, true, 0.5).unwrap();
    assert_eq!(e.open_positions, 1);
    assert_eq!(e.same_bet_positions, 0);
    assert_eq!(e.unknown_positions, 0);
}

#[test]
fn invalid_candidate_has_no_dependency_snapshot() {
    use risk_engine::correlation_guard::dependency_exposure;
    let (arena, _) = fixture();
    assert!(dependency_exposure(&arena, arena.coins.len(), true, 0.5).is_none());
}

// ── XLVI·D: ρ efectiva medida del grupo same-bet ──────────────────────

/// Continuidad con el legado: grupo same-bet enteramente NO medido ⇒
/// ρ_efectiva = 1.0 ⇒ el veto decide EXACTAMENTE como con None (presupuesto
/// lineal). Ningún comportamiento previo cambia por la medición.
#[test]
fn xlvid_grupo_no_medido_reproduce_el_presupuesto_lineal() {
    use risk_engine::correlation_guard::CorrelationGuardEngine as C;
    let tope = risk_engine::ruin::clamp_ruin(1.0, 0.5);
    let riesgo = tope / 8.0; // proxy de arranque (XLI·D2)
    for k in 1..12usize {
        let con_none = C::veto_por_exposicion_estructural(k, riesgo, 0.5, None);
        let con_rho_uno = C::veto_por_exposicion_estructural(k, riesgo, 0.5, Some(1.0));
        assert_eq!(
            con_none, con_rho_uno,
            "k={k}: ρ̄=1 debe reproducir el lineal ({con_none} vs {con_rho_uno})"
        );
    }
}

/// El desbloque medido: k=9 mismas-apuestas con ρ̄=0.5 y riesgo de arranque
/// — el lineal veta (9/8·tope > tope), la agregación de varianza NO
/// (√(9+9·8·0.5)·tope/8 = 0.84·tope < tope). La concurrencia queda
/// gobernada por la dependencia MEDIDA, no por el peor caso permanente.
#[test]
fn xlvid_rho_medida_desbloquea_concurrencia_consistente() {
    use risk_engine::correlation_guard::CorrelationGuardEngine as C;
    let tope = risk_engine::ruin::clamp_ruin(1.0, 0.5);
    let riesgo = tope / 8.0;
    let k = 9usize;
    assert!(
        C::veto_por_exposicion_estructural(k, riesgo, 0.5, None),
        "legado lineal debe vetar k=9 al riesgo de arranque"
    );
    assert!(
        !C::veto_por_exposicion_estructural(k, riesgo, 0.5, Some(0.5)),
        "ρ̄=0.5 medida: √(9+36)·riesgo = 0.84·tope — no veto"
    );
    // ρ̄→1 recupera el veto (continuidad desde arriba).
    assert!(
        C::veto_por_exposicion_estructural(k, riesgo, 0.5, Some(0.999)),
        "ρ̄→1 debe vetar como el lineal"
    );
}

/// El acceso a la ρ efectiva: grupo vacío ⇒ None (el veto no la usa);
/// grupo con miembros ⇒ valor acotado a [−1,1] pase lo que pase en bits.
#[test]
fn xlvid_acceso_rho_efectiva_tiene_contornos() {
    use risk_engine::correlation_guard::DependencyExposure;
    let mut e = DependencyExposure::default();
    assert_eq!(e.same_bet_rho_efectivo(), None, "grupo vacío: None");
    e.same_bet_positions = 4;
    e.same_bet_rho_efectivo_bits = f64::to_bits(0.55);
    assert!((e.same_bet_rho_efectivo().unwrap() - 0.55).abs() < 1e-12);
    // Bits fuera de rango quedan acotados por el acceso, no por fe.
    e.same_bet_rho_efectivo_bits = f64::to_bits(7.5);
    assert_eq!(e.same_bet_rho_efectivo(), Some(1.0), "clamp de saneamiento");
}

// ── XLVI·E (SPECTRAL-010): veto por riesgo real medido ────────────────

/// Reducción EXACTA a D-748: riesgos uniformes [r; n] con ρ ≥ 0 deben dar
/// el MISMO veredicto que la fórmula del veto por conteo
/// `r·sqrt(n+n(n−1)ρ̄) > tope` — la nueva agregación ponderada es la misma
/// doctrina, no otra.
#[test]
fn xlvie_riesgos_uniformes_reducen_a_la_formula_d748() {
    use risk_engine::correlation_guard::veto_por_riesgo_real_medido as veto;
    let r = 0.00125_f64; // tope/8 con tope = 0.01
    let tope = 0.01_f64;
    for n in [1usize, 2, 4, 8, 12] {
        for rho in [0.0_f64, 0.3, 0.5, 0.9, 1.0] {
            let riesgos = vec![r; n];
            let nuevo = veto(&riesgos, Some(rho), tope);
            let kf = n as f64;
            let viejo = r * (kf + kf * (kf - 1.0) * rho).sqrt() > tope;
            assert_eq!(
                nuevo, viejo,
                "n={n} rho={rho}: nuevo={nuevo} viejo={viejo} — la reducción se rompió"
            );
        }
    }
}

/// Híbrido frío == legado: vector enteramente de proxies (miembros no
/// medidos + candidata sin EWMA) decide igual que el veto por conteo con
/// el proxy tope/8 — el arranque frío no cambia de comportamiento.
#[test]
fn xlvie_hibrido_frio_coincide_con_el_veto_legado() {
    use risk_engine::correlation_guard::veto_por_riesgo_real_medido as veto;
    let tope = risk_engine::ruin::clamp_ruin(1.0, 0.5);
    let proxy = tope / 8.0;
    for k in 1..12usize {
        let riesgos = vec![proxy; k + 1]; // k miembros + candidata
        for rho in [None, Some(1.0), Some(0.5)] {
            let nuevo = veto(&riesgos, rho, tope);
            let viejo = risk_engine::correlation_guard::CorrelationGuardEngine::veto_por_exposicion_estructural(
                k, proxy, 0.5, rho,
            );
            assert_eq!(
                nuevo, viejo,
                "k={k} rho={rho:?}: frío nuevo={nuevo} legado={viejo}"
            );
        }
    }
}

/// El unlock de SPECTRAL-010: miembros con stops REALES pequeños dejan de
/// pagar el proxy del peor caso. Tres miembros al 0.2% + candidata al 0.5%
/// con tope 1%: la suma lineal veta (1.1%), la agregación con ρ̄=0.5 no
/// (√(Σr²+ρ̄((Σr)²−Σr²)) ≈ 0.75%).
#[test]
fn xlvie_riesgo_real_medido_desbloquea_stops_pequenos() {
    use risk_engine::correlation_guard::veto_por_riesgo_real_medido as veto;
    let tope = 0.01_f64;
    let riesgos = vec![0.002, 0.002, 0.002, 0.005]; // 3 miembros + candidata
    assert!(veto(&riesgos, None, tope), "lineal: 1.1% > 1% veta");
    assert!(
        !veto(&riesgos, Some(0.5), tope),
        "ρ̄=0.5: agregación ≈0.75% no veta — stops reales medidos"
    );
}

/// Piso generalizado: correlación negativa extrema (o imposible) nunca
/// baja el riesgo de grupo del MAYOR riesgo individual — la cobertura de
/// fantasía no borra la peor exposición aislada.
#[test]
fn xlvie_piso_del_mayor_riesgo_individual() {
    use risk_engine::correlation_guard::veto_por_riesgo_real_medido as veto;
    // Un miembro al 0.9% solo, con tope 0.5%: aunque la agregación con ρ=−0.9
    // prometa menos, la exposición individual YA excede el tope.
    assert!(veto(&[0.009, 0.001], Some(-0.9), 0.005));
    // ρ imposible (ρ < −1/(n−1)) ⇒ caso adverso lineal.
    let riesgos = vec![0.004; 4];
    assert!(veto(&riesgos, Some(-0.9), 0.01), "lineal 1.6% veta");
}

/// Sanidad terminal del veto: NaN/0/negativos en el vector no fabrican ni
/// vetos fantasma ni descuentos; y el híbrido mapea bits-0 (no medido) al
/// fallback del llamador.
#[test]
fn xlvie_sanidad_e_hibrido_de_acceso() {
    use risk_engine::correlation_guard::{DependencyExposure, veto_por_riesgo_real_medido as veto};
    assert!(!veto(&[f64::NAN, 0.0, -1.0, 0.001], None, 0.01));
    assert!(veto(&[f64::NAN, 0.002, 0.002, 0.008], None, 0.01), "sólo los válidos suman");
    let mut e = DependencyExposure::default();
    e.same_bet_riesgos_bits = vec![0, f64::to_bits(0.003), 0];
    let h = e.same_bet_riesgos_hibridos(0.00125);
    assert_eq!(h, vec![0.00125, 0.003, 0.00125], "no medidos ⇒ fallback");
    // Fallback inválido ⇒ prohibitivo (1.0), nunca gratis.
    assert_eq!(e.same_bet_riesgos_hibridos(f64::NAN), vec![1.0, 0.003, 1.0]);
}

/// End-to-end del riesgo medido: posición misma-apuesta en el MISMO activo
/// (ρ=1.0, siempre same-bet) con entry 100 / SL 99 / qty 8 y capital 100 ⇒
/// riesgo real = 8·1/100 = 0.08 medido en el vector de bits.
#[test]
fn xlvie_dependency_exposure_mide_riesgo_real_del_snapshot() {
    use risk_engine::correlation_guard::dependency_exposure;
    let (arena, _) = fixture();
    open(&arena, 0, 0, true); // long: entry 100, sl 99, qty 8
    let e = dependency_exposure(&arena, 0, true, 0.5).unwrap();
    assert_eq!(e.same_bet_positions, 1, "mismo activo misma dirección");
    assert_eq!(e.same_bet_riesgos_bits.len(), 1);
    let r = f64::from_bits(e.same_bet_riesgos_bits[0]);
    assert!(
        (r - 0.08).abs() < 1e-12,
        "riesgo real qty·|entry−sl|/capital = 8·1/100, medido {r}"
    );
}

/// AGY-AUD-P06: Cuando el universo presenta alta vorticidad de Helmholtz-Hodge
/// (hawkes_contagion_curl_share elevado), rho_efectivo escala continuamente
/// hacia 1.0 (dependencia sistémica), reduciendo la ilusión de diversificación.
#[test]
fn agy_aud_p06_hodge_curl_share_systemic_rho_escalation() {
    use risk_engine::correlation_guard::dependency_exposure;
    let (arena, _) = fixture();
    // Inyectar ticks con desfase angular para obtener correlación positiva moderada (0 < r < 1)
    for i in 0..200 {
        let p0 = 100.0 * (0.01 * (i as f64 * 0.7).sin()).exp();
        let p1 = 100.0 * (0.01 * (i as f64 * 0.7 + 0.8).sin()).exp();
        arena.coins[0].tick_ring.push(CompactTick {
            timestamp: 1000 + i * 10,
            bid_price: p0,
            ask_price: p0,
            bid_qty: 1.0,
            ask_qty: 1.0,
        });
        arena.coins[1].tick_ring.push(CompactTick {
            timestamp: 1000 + i * 10,
            bid_price: p1,
            ask_price: p1,
            bid_qty: 1.0,
            ask_qty: 1.0,
        });
    }
    open(&arena, 1, 0, true);

    // Sin curl_share publicado: rho base normal
    let e_base = dependency_exposure(&arena, 0, true, 0.5).unwrap();
    let rho_base = e_base.same_bet_rho_efectivo().unwrap();
    assert!(rho_base > 0.1 && rho_base < 0.95, "rho base moderado: {rho_base}");

    // Con curl_share = 0.80 (fuerte feedback cíclico en cascada)
    arena.registry.set("hawkes_contagion_curl_share", 0.80);
    let e_systemic = dependency_exposure(&arena, 0, true, 0.5).unwrap();
    let rho_systemic = e_systemic.same_bet_rho_efectivo().unwrap();

    assert!(
        rho_systemic > rho_base,
        "rho sistémico {rho_systemic} debe superar rho base {rho_base} ante curl alto"
    );
    // Verificación de la interpolación cuadrática: rho + (1 - rho) * 0.8^2 = rho + (1 - rho) * 0.64
    let expected = (rho_base + (1.0 - rho_base) * 0.64).clamp(-1.0, 1.0);
    assert!((rho_systemic - expected).abs() < 1e-6);
}

/// AGY-AUD-P06: Un activo seguidor neto bajo fuerte excitación de contagio (net_role < -3.0)
/// amplifica su correlación observada mediante amplificar_por_contagio.
#[test]
fn agy_aud_p06_contagion_amplification_in_dependency_exposure() {
    use risk_engine::correlation_guard::dependency_exposure;
    let (arena, _) = fixture();
    pair_ticks(&arena, false);
    open(&arena, 1, 0, true);

    // Activo 1 como seguidor extremo recibiendo contagio Hawkes: net_role = -8.0 (z = 8.0)
    arena.registry.set_for_coin(1, "hawkes_contagion_net_role", -8.0);
    let e = dependency_exposure(&arena, 0, true, 0.5).unwrap();
    assert_eq!(e.same_bet_positions, 1);
}

/// #602 (Ola 24) — el call-site del veto de grupo consume el slot POR
/// MONEDA del estimador Cramér-Lundberg (`c{id}:lundberg_r_nocional`,
/// escrito por el core en cada cierre) con ε = 0.05 de política, y deja
/// el tope_streak INTACTO cuando la clave está ausente (arranque frío:
/// la cota no significa nada sin edge medido, D-754). Contable vía
/// `qo_602_veto_lundberg`.
#[test]
fn qo_602_el_veto_de_grupo_consume_la_cota_lundberg_del_registro() {
    let src: String = include_str!("../src/lib.rs").split_whitespace().collect();
    // Fuente única: el R viene del slot de moneda del publicador (#600)…
    assert!(
        src.contains("get_for_coin_or(coin_id,\"lundberg_r_nocional\",0.0)"),
        "el R debe leerse del slot de moneda que publica el estimador"
    );
    // …con arranque frío honesto (ausente/≤0 ⇒ None ⇒ tope intacto)…
    assert!(
        src.contains("(r_raw.is_finite()&&r_raw>0.0).then_some(r_raw)"),
        "sin R medido no hay apriete — bit a bit el veto anterior"
    );
    // …el gate es el de la cota (no el plano)…
    assert!(
        src.contains("veto_por_riesgo_cramer_lundberg("),
        "el call-site debe usar la variante con cota actuarial"
    );
    // …ε de política 0.05 (ψ ≤ 5%, convención del margen_5pct)…
    assert!(
        src.contains("r_lundberg,\n0.05,") || src.contains("r_lundberg,0.05,"),
        "ε = 0.05 de política en el call-site"
    );
    // …y contable sólo con cota disponible.
    assert!(
        src.contains("qo_602_veto_lundberg"),
        "los rechazos con cota activa deben ser contables para el consejo"
    );
    // El propio gate (Antigravity Ola 9) aprieta con margen_de_cota y cae
    // al tope_streak sin R — verificado por su suite:
    //   ola9_veto_por_riesgo_cramer_lundberg_bounds (risk-engine --lib).
    let gate: String =
        include_str!("../src/correlation_guard.rs").split_whitespace().collect();
    assert!(gate.contains("margen_de_cota(r,epsilon)"));
    assert!(gate.contains("_=>tope_streak,"));
}
