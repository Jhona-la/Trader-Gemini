//! F412/F413/F271 (Sol, censo C-F7b §3): CANAL INDEPENDIENTE DEL HOST —
//! cierres confirmados por el exchange (user-data stream → AccountSink →
//! `registrar_fill_independiente_host`) alimentan la puerta de evidencia del
//! veto del Auditor Interno.
//!
//! Doctrina (D-690 / F412/F413/F271):
//! - Sin fuente independiente, las tres patas de la discrepancia son
//!   tautológicas (sharpe sin escritor, WR del mismo cierre, dd actual vs
//!   máximo histórico) ⇒ la puerta devuelve `None` y el veto NO dispara.
//! - Con el canal del host (fills REALES del exchange, con PnL y comisiones
//!   del fill), la discrepancia se mide contra evidencia ajena al núcleo:
//!   claims honestos ⇒ disc ≈ 0; claims inflados ⇒ disc > umbral y el veto
//!   es legítimo.
//! - Higiene: fills no finitos rechazados; journal acotado.

use audit_engine::auditor_interno::{ExchangeFillRecord, MAX_FILLS_INDEPENDIENTES, MIN_FILLS_SOMBRA};
use god_engine_core::GodEngineCore;
use quantum_arena::GlobalArena;

fn fill(ts_ns: u64, pnl: f64, fee: f64) -> ExchangeFillRecord {
    ExchangeFillRecord {
        timestamp_ns: ts_ns,
        symbol_id: 0,
        is_buyer: false,
        price: 50_000.0,
        qty: 0.001,
        fee_usd: fee,
        realized_pnl: pnl,
        is_closed_position: true,
    }
}

#[test]
fn sin_canal_host_la_puerta_no_dispara() {
    let arena = GlobalArena::build_in_own_stack(13.0);
    let core = GodEngineCore::new(arena);
    // Sin fuente independiente: las tres patas devuelven None (F412/F413/F271).
    assert!(
        core.auditor_interno
            .compute_internal_discrepancy_evidenciada(0.0, 0.5, 0.0)
            .is_none(),
        "sin canal del host la discrepancia evidenciada debe ser None"
    );
}

#[test]
fn canal_hook_habilita_veto_solo_con_evidencia_real() {
    let arena = GlobalArena::build_in_own_stack(13.0);
    let mut core = GodEngineCore::new(arena);

    // 20 cierres del host: WR real 0.5 (+0.5, +0.5, −0.2, −0.2 en ciclo, sin
    // comisión). LG-33: con menos de MIN_FILLS_SOMBRA cierres no hay evidencia.
    let patron = [0.5, 0.5, -0.2, -0.2];
    for (i, pnl) in patron.iter().cycle().take(MIN_FILLS_SOMBRA).enumerate() {
        core.registrar_fill_independiente_host(fill(1_700_000_000_000_000_000 + i as u64 * 1_000_000_000, *pnl, 0.0));
    }
    assert!(
        core.auditor_interno.tiene_fuente_independiente(),
        "el canal del host debe activar la fuente independiente"
    );
    let indep = core.auditor_interno.compute_independent_metrics_from_host();
    assert!(
        (indep.calculated_win_rate - 0.5).abs() < 1e-9,
        "WR del canal host = 0.5 (evidencia real), obtuvo {}",
        indep.calculated_win_rate
    );

    // Claims honestos contra la evidencia independiente ⇒ disc ≈ 0, sin veto.
    let disc_honesta = core.auditor_interno.compute_internal_discrepancy_evidenciada(
        indep.calculated_sharpe,
        indep.calculated_win_rate,
        indep.calculated_max_drawdown,
    );
    assert!(disc_honesta.is_some());
    assert!(
        disc_honesta.unwrap_or(1.0) < 1e-9,
        "afirmación honesta vs canal host ⇒ disc ≈ 0"
    );

    // Claim inflado (WR 0.9 vs real 0.5) ⇒ la pata WR vale 0.4 menos su error
    // estándar por bloques, por encima del umbral 0.20: el veto PUEDE disparar,
    // ahora respaldado por evidencia independiente.
    let disc_inflada = core
        .auditor_interno
        .compute_internal_discrepancy_evidenciada(indep.calculated_sharpe, 0.9, 0.0);
    let sombra = core.auditor_interno.sombra_wr_disc(0.9).expect("20 cierres ⇒ sombra");
    let esperado = (0.4 - sombra.ee_real).max(0.0).max(indep.calculated_max_drawdown);
    assert!(
        (disc_inflada.unwrap_or(0.0) - esperado).abs() < 1e-9,
        "WR inflado 0.9 vs real 0.5 ⇒ disc {esperado} (obtuvo {:?})",
        disc_inflada
    );
    assert!(disc_inflada.unwrap_or(0.0) > 0.20);
}

#[test]
fn canal_host_rechaza_no_finitos_y_queda_acotado() {
    let arena = GlobalArena::build_in_own_stack(13.0);
    let mut core = GodEngineCore::new(arena);

    // Fill con NaN en el PnL: rechazado, no contamina el canal.
    core.registrar_fill_independiente_host(fill(1, f64::NAN, 0.0));
    assert!(
        !core.auditor_interno.tiene_fuente_independiente(),
        "un fill no finito no crea fuente independiente"
    );

    // Journal acotado: MAX + desborde no crece.
    for i in 0..(MAX_FILLS_INDEPENDIENTES + 64) {
        core.registrar_fill_independiente_host(fill(
            1_700_000_000_000_000_000 + i as u64 * 1_000_000_000,
            if i % 2 == 0 { 0.05 } else { -0.02 },
            0.002,
        ));
    }
    assert_eq!(
        core.auditor_interno.fills_independientes_len(),
        MAX_FILLS_INDEPENDIENTES,
        "el diario del canal independiente debe quedar acotado"
    );
    let disc = core
        .auditor_interno
        .compute_internal_discrepancy_evidenciada(0.0, 0.5, 0.0);
    assert!(
        disc.is_some() && disc.unwrap_or(1.0).is_finite(),
        "salida finita con journal lleno"
    );
}
