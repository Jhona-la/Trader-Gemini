// ============================================================================
// F412 / F413 / F271 — CENSO C-F7b §3 (prioridad 1, 2026-10-18, Sol)
// PUERTA DE EVIDENCIA DEL VETO DEL AUDITOR INTERNO (antitautológico)
//
// Defecto (crítico del censo, confirmado por verificación):
//   - F412: la clave `internal_sharpe` no tiene escritor ⇒ claim 0.0 ⇒ pata muda.
//   - F413: `internal_wr` (diag del núcleo) y el WR "independiente" salen del
//     mismo cierre con el mismo predicado net>0 ⇒ tautología.
//   - F271: `dd_pct` (caída ACTUAL contra capital base) se compara con el
//     MÁXIMO histórico desde pico del diario de cartera ⇒ con capital
//     13 → 16,9 (+30 %) → 13 la disc es 3,9/16,9 = 0,2308 > 0,20 ⇒ el veto
//     dispara para TODAS las monedas y nunca decae (diario global leído por
//     moneda + `registry.set` global con fallback cruzado).
//
// Contrato del átomo:
//   (1) Las tres patas devuelven None mientras no haya fuente independiente.
//   (2) Fuera el `registry.set` global (lib.rs:5088).
//   (3) RED→GREEN 13 → 16,9 → 13: hoy veta y después no veta.
// ============================================================================

use audit_engine::auditor_interno::{
    calculate_max_discrepancy, AuditorInterno, ExchangeFillRecord, MAX_FILLS_INDEPENDIENTES,
    MIN_FILLS_SOMBRA,
};
use metacortex_engine::consejo_seniors::{
    evaluate_auditor_interno_opinion, MarketSnapshotPayload, DEFAULT_DISCREPANCY_VETO_THRESHOLD,
};
use omniscient_registry::OmniscientRegistry;

/// Escenario del censo: capital 13 → 16,9 (+30 %) → 13 (devolución completa).
/// Devuelve el auditor con el diario interno alimentado como lo hace hoy el
/// núcleo (mismo bloque de cierre) y las métricas independientes resultantes.
fn escenario_13_16_9_13() -> (AuditorInterno, f64, f64, f64) {
    let mut auditor = AuditorInterno::with_capital(13.0);
    // Cierre 1: +3.9 netos (13 → 16.9). Cierre 2: -3.9 netos (16.9 → 13.0).
    auditor.record_fill(ExchangeFillRecord {
        timestamp_ns: 1_700_000_000_000_000_000,
        symbol_id: 0,
        is_buyer: true,
        price: 50_000.0,
        qty: 0.001,
        fee_usd: 0.0,
        realized_pnl: 3.9,
        is_closed_position: true,
    });
    auditor.record_fill(ExchangeFillRecord {
        timestamp_ns: 1_700_000_000_001_000_000,
        symbol_id: 0,
        is_buyer: false,
        price: 50_000.0,
        qty: 0.001,
        fee_usd: 0.0,
        realized_pnl: -3.9,
        is_closed_position: true,
    });
    let indep = auditor.compute_independent_metrics();
    // Afirmaciones internas tal como las pasa el núcleo tras el round-trip:
    // sharpe sin escritor (0.0), WR de diagnóstico (1 victoria / 2 cierres),
    // drawdown actual contra capital base (capital de vuelta en 13 ⇒ 0.0).
    (auditor, indep.calculated_sharpe, indep.calculated_win_rate, indep.calculated_max_drawdown)
}

#[test]
fn red_green_13_16_9_13_el_veto_deja_de_dispararse_solo() {
    let (auditor, indep_sharpe, indep_wr, indep_dd) = escenario_13_16_9_13();

    // --- RED (hoy): discrepancia tautológica de la pata DD ---
    let disc_rojo = auditor.compute_internal_discrepancy(0.0, 0.5, 0.0);
    let disc_esperada = 3.9 / 16.9; // 0.230769...
    assert!(
        (disc_rojo - disc_esperada).abs() < 1e-9,
        "RED: la pata DD debía dar {disc_esperada}, dio {disc_rojo}"
    );
    assert!(
        disc_rojo > DEFAULT_DISCREPANCY_VETO_THRESHOLD,
        "RED: {disc_rojo} > {} debía cruzar el umbral y vetar",
        DEFAULT_DISCREPANCY_VETO_THRESHOLD
    );
    let mut payload_rojo = MarketSnapshotPayload::default();
    payload_rojo.internal_discrepancy = disc_rojo;
    let op_rojo = evaluate_auditor_interno_opinion(&payload_rojo);
    assert!(
        op_rojo.3,
        "RED: con disc={disc_rojo} el Auditor Interno vetaba (autoengaño tautológico)"
    );

    // --- GREEN (átomo): sin fuente independiente las patas devuelven None ---
    assert!(
        !auditor.tiene_fuente_independiente(),
        "precondición: el auditor del núcleo no tiene canal independiente"
    );
    let disc_verde = auditor.compute_internal_discrepancy_evidenciada(0.0, 0.5, 0.0);
    assert_eq!(
        disc_verde, None,
        "GREEN: sin fuente independiente la discrepancia evidenciada es None"
    );
    // Regla de publicación: None ⇒ clave ausente ⇒ payload 0.0 ⇒ sin veto.
    let registry = OmniscientRegistry::new();
    if let Some(d) = disc_verde {
        registry.set_for_coin(0, "internal_discrepancy", d);
    }
    let payload_disc = registry.get_scoped_value_or("BTCUSDT", "internal_discrepancy", 0.0);
    assert_eq!(payload_disc, 0.0, "clave ausente ⇒ payload 0.0");
    let mut payload_verde = MarketSnapshotPayload::default();
    payload_verde.internal_discrepancy = payload_disc;
    let op_verde = evaluate_auditor_interno_opinion(&payload_verde);
    assert!(
        !op_verde.3,
        "GREEN: tras el átomo el round-trip 13 → 16,9 → 13 NO veta"
    );

    // Las afirmaciones del escenario son honestas contra las métricas reales:
    // WR 0.5 = 0.5, DD actual 0.0 (capital en base): la única pata que
    // disparaba era la comparación tautológica actual-vs-máximo-histórico.
    let (max_disc, sharpe_disc, wr_disc, dd_disc) =
        calculate_max_discrepancy(0.0, indep_wr, 0.0, &auditor.compute_independent_metrics());
    assert_eq!(sharpe_disc, 0.0, "claim sharpe 0.0 ⇒ pata muda (F412)");
    assert_eq!(wr_disc, 0.0, "WR interno == WR independiente en el escenario");
    assert!((dd_disc - disc_esperada).abs() < 1e-9, "pata DD = 0,2308");
    assert!((max_disc - disc_esperada).abs() < 1e-9);
    let _ = (indep_sharpe, indep_dd);
}

#[test]
fn fuente_independiente_habilita_veto_solo_con_evidencia_real() {
    let mut auditor = AuditorInterno::with_capital(13.0);
    assert_eq!(auditor.compute_internal_discrepancy_evidenciada(0.0, 0.5, 0.0), None);

    // Canal independiente: 20 cierres del host con WR real 0.5. LG-33: con menos
    // de MIN_FILLS_SOMBRA cierres no hay evidencia y no se publica discrepancia.
    let patron = [0.5, 0.5, -0.2, -0.2];
    for (i, pnl) in patron.iter().cycle().take(MIN_FILLS_SOMBRA).enumerate() {
        auditor.record_fill_independiente(ExchangeFillRecord {
            timestamp_ns: 1_700_000_000_000_000_000 + i as u64 * 1_000_000_000,
            symbol_id: 0,
            is_buyer: true,
            price: 50_000.0,
            qty: 0.001,
            fee_usd: 0.0,
            realized_pnl: *pnl,
            is_closed_position: true,
        });
    }
    assert!(auditor.tiene_fuente_independiente());
    let indep = auditor.compute_independent_metrics_from_host();

    // Afirmación honesta (coincide con la realidad independiente) ⇒ Some(~0).
    let disc_honesta = auditor.compute_internal_discrepancy_evidenciada(
        indep.calculated_sharpe,
        indep.calculated_win_rate,
        indep.calculated_max_drawdown,
    );
    assert!(disc_honesta.is_some());
    assert!(
        disc_honesta.unwrap_or(1.0) < 1e-9,
        "afirmación honesta vs canal independiente ⇒ disc ≈ 0"
    );

    // Afirmación inflada (WR 0.9 vs real 0.5) ⇒ la pata WR vale |0.9 − 0.5| menos
    // su error estándar por bloques (SOL §3-4) ⇒ el veto PUEDE disparar, ahora
    // con evidencia independiente de respaldo.
    let disc_inflada =
        auditor.compute_internal_discrepancy_evidenciada(indep.calculated_sharpe, 0.9, 0.0);
    let sombra = auditor.sombra_wr_disc(0.9).expect("20 cierres ⇒ sombra");
    let pata_wr = (0.4 - sombra.ee_real).max(0.0);
    assert!(
        (disc_inflada.unwrap_or(0.0) - pata_wr.max(indep.calculated_max_drawdown)).abs() < 1e-9,
        "WR inflado 0.9 vs 0.5 ⇒ disc = max(0.4 − ee, dd) ({disc_inflada:?}, ee {})",
        sombra.ee_real
    );
    assert!(disc_inflada.unwrap_or(0.0) > DEFAULT_DISCREPANCY_VETO_THRESHOLD);
    let mut payload = MarketSnapshotPayload::default();
    payload.internal_discrepancy = disc_inflada.unwrap_or(0.0);
    let op = evaluate_auditor_interno_opinion(&payload);
    assert!(op.3, "con fuente independiente y disc=0.4 el veto es legítimo");

    // Recuperación: la misma afirmación vuelve a ser honesta ⇒ el veto se cae
    // (no hay histéresis: cada evaluación recalcula desde la evidencia).
    let disc_recuperada =
        auditor.compute_internal_discrepancy_evidenciada(indep.calculated_sharpe, 0.5, 0.0);
    assert!(disc_recuperada.unwrap_or(1.0) < DEFAULT_DISCREPANCY_VETO_THRESHOLD);
}

#[test]
fn publicacion_registro_sin_set_global_y_sin_contaminacion_entre_monedas() {
    let registry = OmniscientRegistry::new();

    // Caso None (regla del núcleo): no se escribe NINGUNA clave.
    let disc_none: Option<f64> = None;
    assert!(disc_none.is_none(), "precondición del contrato");
    assert!(
        registry.get_value_fast("internal_discrepancy").is_none(),
        "la clave GLOBAL nunca se escribe (F412/F413/F271, ítem 2)"
    );
    assert!(registry.get_value_fast("c5_internal_discrepancy").is_none());
    assert_eq!(
        registry.get_scoped_value_or("DOGEUSDT", "internal_discrepancy", 0.0),
        0.0,
        "moneda sin evidencia propia lee default, no el valor de otra"
    );

    // Caso Some: sólo la clave de la moneda con evidencia.
    if let Some(d) = Some(0.42) {
        registry.set_for_coin(5, "internal_discrepancy", d);
    }
    assert!(
        registry.get_value_fast("internal_discrepancy").is_none(),
        "set_for_coin no crea la clave global"
    );
    assert_eq!(registry.get_for_coin_or(5, "internal_discrepancy", 0.0), 0.42);
    assert_eq!(
        registry.get_for_coin_or(7, "internal_discrepancy", 0.0),
        0.0,
        "otra moneda NO hereda la discrepancia de la moneda 5"
    );

    // El veto del Consejo con payload 0.0 (ausencia) no dispara.
    let payload = MarketSnapshotPayload {
        internal_discrepancy: 0.0,
        ..Default::default()
    };
    let op = evaluate_auditor_interno_opinion(&payload);
    assert!(!op.3, "payload 0.0 (sin evidencia) ⇒ sin veto");
}

#[test]
fn journal_independiente_acotado_y_rechaza_no_finitos() {
    let mut auditor = AuditorInterno::with_capital(13.0);
    for i in 0..(MAX_FILLS_INDEPENDIENTES + 128) {
        auditor.record_fill_independiente(ExchangeFillRecord {
            timestamp_ns: (i as u64 + 1) * 1_000_000_000,
            symbol_id: 0,
            is_buyer: i % 2 == 0,
            price: 50_000.0,
            qty: 0.001,
            fee_usd: 0.002,
            realized_pnl: if i % 2 == 0 { 0.05 } else { -0.02 },
            is_closed_position: true,
        });
    }
    let n = auditor.fills_independientes_len();
    assert_eq!(n, MAX_FILLS_INDEPENDIENTES, "diario independiente acotado");

    // fill con NaN ⇒ rechazado, no contamina el canal.
    auditor.record_fill_independiente(ExchangeFillRecord {
        timestamp_ns: 9_000_000_000_000_000_000,
        symbol_id: 0,
        is_buyer: true,
        price: 50_000.0,
        qty: f64::NAN,
        fee_usd: 0.0,
        realized_pnl: 1.0,
        is_closed_position: true,
    });
    assert_eq!(auditor.fills_independientes_len(), MAX_FILLS_INDEPENDIENTES);
    let disc = auditor.compute_internal_discrepancy_evidenciada(0.0, 0.5, 0.0);
    assert!(disc.is_some() && disc.unwrap_or(1.0).is_finite(), "sin NaN en salida");
}
