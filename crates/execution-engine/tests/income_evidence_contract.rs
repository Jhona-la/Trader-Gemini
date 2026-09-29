//! Pure/mocked contracts: no account, environment, journal or network access.
use execution_engine::income_evidence::*;
use execution_engine::order_types::IncomeEntry;

fn row(id: u64, time: u64, income: f64) -> IncomeEntry {
    IncomeEntry {
        symbol: "TESTUSDT".into(),
        income_type: "REALIZED_PNL".into(),
        income,
        asset: "USDT".into(),
        time,
        tran_id: id,
        trade_id: id.to_string(),
    }
}
fn kind(id: u64, income: f64, name: &str, asset: &str) -> IncomeEntry {
    let mut e = row(id, 10, income);
    e.income_type = name.into();
    e.asset = asset.into();
    e
}
async fn collect(
    pages: Vec<Vec<IncomeEntry>>,
    size: u32,
    budget: u32,
) -> Result<IncomeWindow, IncomeEvidenceError> {
    collect_income_window(1, 100, budget, size, |page| {
        std::future::ready(
            pages
                .get((page - 1) as usize)
                .cloned()
                .ok_or_else(|| "unexpected fetch".into()),
        )
    })
    .await
}

#[test]
fn query_freezes_interval_and_uses_numbered_pages() {
    assert_eq!(
        income_page_query(10, 99, 2, 1000, &[], 100).unwrap(),
        "startTime=10&endTime=99&page=2&limit=1000&timestamp=100"
    );
    assert!(income_page_query(10, 99, 3, 10, &["COMMISSION"], 101)
        .unwrap()
        .ends_with("&incomeType=COMMISSION"));
}
#[test]
fn query_rejects_invalid_ranges_sizes_timestamps_and_filters() {
    assert_eq!(
        income_page_query(2, 1, 1, 1, &[], 3),
        Err(IncomeEvidenceError::InvalidRange)
    );
    assert_eq!(
        income_page_query(0, u64::MAX, 1, 1, &[], 3),
        Err(IncomeEvidenceError::InvalidRange)
    );
    assert_eq!(
        income_page_query(0, 1, 0, 1, &[], 3),
        Err(IncomeEvidenceError::InvalidPage)
    );
    for size in [0, 1001] {
        assert_eq!(
            income_page_query(0, 1, 1, size, &[], 3),
            Err(IncomeEvidenceError::InvalidPageSize)
        );
    }
    for ts in [0, u64::MAX] {
        assert_eq!(
            income_page_query(0, 1, 1, 1, &[], ts),
            Err(IncomeEvidenceError::InvalidTimestamp)
        );
    }
    for filter in [
        "",
        "COMMISSION&limit=1",
        "COMMISSION,REALIZED_PNL",
        "commission",
        "FUNDING FEE",
    ] {
        assert_eq!(
            income_page_query(0, 1, 1, 1, &[filter], 3),
            Err(IncomeEvidenceError::UnsupportedFilter)
        );
    }
    assert_eq!(
        income_page_query(0, 1, 1, 1, &["COMMISSION", "FUNDING_FEE"], 3),
        Err(IncomeEvidenceError::UnsupportedFilter)
    );
}
#[tokio::test]
async fn more_than_one_thousand_events_at_one_millisecond_are_preserved() {
    let first: Vec<_> = (1..=1000).map(|id| row(id, 50, 1.0)).collect();
    let w = collect(vec![first, vec![row(1001, 50, 1.0)]], 1000, 4)
        .await
        .unwrap();
    assert_eq!(w.pages_read, 2);
    assert_eq!(w.coverage, IncomeCoverage::PageExhausted);
    assert_eq!(w.into_exhausted_entries().unwrap().len(), 1001);
}
#[tokio::test]
async fn full_last_allowed_page_is_not_implicitly_complete() {
    let w = collect(vec![vec![row(1, 10, 1.0)]], 1, 1).await.unwrap();
    assert_eq!(
        w.into_exhausted_entries().unwrap_err(),
        IncomeEvidenceError::Incomplete(IncomeCoverage::PageBudgetExceeded)
    );
}
#[tokio::test]
async fn exact_multiple_requires_an_empty_terminal_page() {
    let w = collect(vec![vec![row(1, 10, 1.0)], vec![]], 1, 2)
        .await
        .unwrap();
    assert_eq!(w.pages_read, 2);
    assert_eq!(w.into_exhausted_entries().unwrap().len(), 1);
}
#[tokio::test]
async fn repeated_page_is_no_progress_even_when_short() {
    let w = collect(
        vec![
            vec![row(1, 10, 1.0), row(2, 10, 2.0)],
            vec![row(1, 10, 1.0)],
        ],
        2,
        3,
    )
    .await
    .unwrap();
    assert_eq!(w.coverage, IncomeCoverage::NoProgress);
    assert_eq!(w.pages_read, 2);
    assert!(w.into_exhausted_entries().is_err());
}
#[tokio::test]
async fn same_visible_identity_with_a_different_amount_is_a_conflict() {
    // FMT-285b: el conflicto se CUARENTENA (no recuperable — exige
    // conciliación) en vez de matar la ventana; la evidencia válida viaja.
    let w = collect(vec![vec![row(1, 10, 1.0)], vec![row(1, 10, -1.0)]], 1, 2)
        .await
        .unwrap();
    assert_eq!(w.entries.len(), 1, "la primera lectura sobrevive");
    assert_eq!(w.quarantined.len(), 1);
    assert_eq!(w.quarantined[0].reason, QuarantineReason::ConflictingIdentity);
    assert!(!w.quarantined[0].recoverable, "exige conciliación");
}
#[tokio::test]
async fn asset_and_trade_id_prevent_false_duplicate_collapse() {
    let a = row(1, 10, 1.0);
    let mut b = a.clone();
    b.asset = "BNB".into();
    let mut c = a.clone();
    c.trade_id = "different".into();
    let w = collect(vec![vec![a, b, c]], 10, 1).await.unwrap();
    assert_eq!(w.into_exhausted_entries().unwrap().len(), 3);
}
#[tokio::test]
async fn identical_overlap_is_consumed_once_and_zero_sign_is_not_a_conflict() {
    let mut minus = row(1, 10, 0.0);
    minus.income = -0.0;
    let w = collect(vec![vec![row(1, 10, 0.0), minus, row(2, 10, -1.0)]], 10, 1)
        .await
        .unwrap();
    assert_eq!(w.into_exhausted_entries().unwrap().len(), 2);
}
#[tokio::test]
async fn invalid_missing_and_outside_window_rows_are_not_admitted() {
    // FMT-285b: un registro inválido va a CUARENTENA — ya no aborta la
    // ventana; sigue sin ser admitido como evidencia.
    let mut invalids = vec![];
    let mut x = row(1, 10, 1.0);
    x.asset.clear();
    invalids.push(x);
    let mut x = row(1, 10, 1.0);
    x.income_type.clear();
    invalids.push(x);
    invalids.push(row(0, 10, 1.0));
    invalids.push(row(1, 0, 1.0));
    invalids.push(row(1, 10, f64::NAN));
    invalids.push(row(1, 10, f64::INFINITY));
    for x in invalids {
        let w = collect(vec![vec![x]], 10, 1).await.unwrap();
        assert_eq!(w.entries.len(), 0, "el inválido no se admite");
        assert_eq!(w.quarantined.len(), 1);
        assert_eq!(w.quarantined[0].reason, QuarantineReason::InvalidRecord);
        assert!(w.quarantined[0].recoverable);
    }
    // Fuera de ventana: cuarentena con su propio motivo (recuperable
    // ampliando la ventana).
    let w = collect(vec![vec![row(1, 101, 1.0)]], 10, 1).await.unwrap();
    assert_eq!(w.quarantined[0].reason, QuarantineReason::OutOfRange);
    assert_eq!(w.entries.len(), 0);
    // OversizedPage sigue LETAL: es una violación de protocolo de página,
    // no un registro malo.
    assert_eq!(
        collect(vec![vec![row(1, 10, 1.0), row(2, 10, 1.0)]], 1, 1)
            .await
            .unwrap_err(),
        IncomeEvidenceError::OversizedPage
    );
}
#[tokio::test]
async fn invalid_request_does_not_invoke_fetcher() {
    for (start, end, budget, size) in [(2, 1, 1, 1), (0, 1, 0, 1), (0, 1, 1, 0), (0, 1, 1, 1001)] {
        let mut calls = 0;
        let result = collect_income_window(start, end, budget, size, |_| {
            calls += 1;
            std::future::ready(Ok(vec![]))
        })
        .await;
        assert!(result.is_err());
        assert_eq!(calls, 0);
    }
}
#[tokio::test]
async fn transport_error_after_partial_success_is_not_empty_or_complete_evidence() {
    // FMT-285b: el transporte que falla TRAS la primera página devuelve la
    // evidencia parcial con cobertura TransportTruncated y el motivo —
    // jamás se presenta como agotada (into_exhausted_entries la rechaza).
    let w = collect_income_window(1, 100, 3, 1, |p| {
        std::future::ready(if p == 1 {
            Ok(vec![row(1, 10, 1.0)])
        } else {
            Err("offline".into())
        })
    })
    .await
    .unwrap();
    assert_eq!(w.coverage, IncomeCoverage::TransportTruncated);
    assert_eq!(w.pages_read, 1);
    assert_eq!(w.entries.len(), 1, "la página leída sobrevive");
    assert_eq!(w.transport_error.as_deref(), Some("offline"));
    assert!(w.into_exhausted_entries().is_err());
    // Falla en la página 1: sin evidencia no hay ventana — Err como antes.
    let err = collect_income_window(1, 100, 3, 1, |_| {
        std::future::ready(Err("offline".into()))
    })
    .await
    .unwrap_err();
    assert_eq!(err, IncomeEvidenceError::Transport("offline".into()));
}
#[test]
fn single_asset_guard_never_converts_or_adds_currencies() {
    assert_eq!(single_income_asset(&[]), Ok(None));
    assert_eq!(single_income_asset(&[row(1, 10, 1.0)]), Ok(Some("USDT")));
    assert_eq!(
        single_income_asset(&[row(1, 10, 1.0), kind(2, -1.0, "COMMISSION", "BNB")]),
        Err(IncomeEvidenceError::MixedAssets)
    );
}
#[test]
fn aggregation_separates_assets_preserves_signed_costs_and_exposes_other_flows() {
    let s = aggregate_income_by_asset(&[
        kind(1, 10.0, "REALIZED_PNL", "USDT"),
        kind(2, -2.0, "COMMISSION", "USDT"),
        kind(3, 0.5, "COMMISSION", "USDT"),
        kind(4, -1.0, "FUNDING_FEE", "USDT"),
        kind(5, 100.0, "TRANSFER", "USDT"),
        kind(6, 3.0, "REALIZED_PNL", "BNB"),
    ])
    .unwrap();
    assert_eq!(s.by_asset.len(), 2);
    assert_eq!(s.by_symbol.len(), 2);
    let t = &s.by_asset["USDT"];
    assert_eq!(t.selected_net().unwrap(), 7.5);
    assert_eq!(t.commission, -1.5);
    assert_eq!(t.other_income, 100.0);
    assert_eq!(t.other_rows, 1);
    assert_eq!(s.by_asset["BNB"].selected_net().unwrap(), 3.0);
}

#[test]
fn mixed_fee_currency_is_quarantined_only_for_the_affected_symbol() {
    let a = row(1, 10, 1.0);
    let b = kind(2, -0.1, "COMMISSION", "BNB");
    let mut c = row(3, 10, 2.0);
    c.symbol = "OTHERUSDT".into();
    let partition = partition_legacy_fee_evidence(&[a, b, c]);
    assert_eq!(
        partition.rejected_by_symbol["TESTUSDT"],
        IncomeEvidenceError::MixedAssets
    );
    assert!(!partition.same_asset_by_symbol.contains_key("TESTUSDT"));
    assert_eq!(partition.same_asset_by_symbol["OTHERUSDT"].len(), 1);
}

#[test]
fn global_transfer_does_not_suppress_single_currency_fee_evidence() {
    let mut transfer = kind(2, 1.0, "TRANSFER", "BTC");
    transfer.symbol.clear();
    let partition = partition_legacy_fee_evidence(&[row(1, 10, 1.0), transfer]);
    assert!(partition.rejected_by_symbol.is_empty());
    assert_eq!(partition.same_asset_by_symbol["TESTUSDT"].len(), 1);
}

#[test]
fn symbols_in_distinct_settlement_assets_can_each_be_reviewed_separately() {
    let mut other = kind(2, 1.0, "REALIZED_PNL", "USDC");
    other.symbol = "OTHERUSDC".into();
    let partition = partition_legacy_fee_evidence(&[row(1, 10, 1.0), other]);
    assert!(partition.rejected_by_symbol.is_empty());
    assert_eq!(partition.same_asset_by_symbol.len(), 2);
}
#[test]
fn positive_row_fraction_includes_flat_rows_but_does_not_claim_trade_win_rate() {
    let s =
        aggregate_income_by_asset(&[row(1, 10, 10.0), row(2, 10, 0.0), row(3, 10, -1.0)]).unwrap();
    assert_eq!(s.by_asset["USDT"].pnl_rows, 3);
    assert_eq!(s.by_asset["USDT"].positive_row_fraction(), Some(1.0 / 3.0));
    assert_eq!(IncomeTotals::default().positive_row_fraction(), None);
}
#[test]
fn ratios_are_unit_invariant_and_undefined_is_not_zero() {
    for scale in [1e-12, 1.0, 100.0] {
        let s = aggregate_income_by_asset(&[
            kind(1, 10.0 * scale, "REALIZED_PNL", "USDT"),
            kind(2, -2.0 * scale, "COMMISSION", "USDT"),
        ])
        .unwrap();
        assert!((s.by_asset["USDT"].cost_to_abs_realized_ratio().unwrap() + 0.2).abs() < 1e-14);
    }
    let t = IncomeTotals {
        commission: -1.0,
        ..Default::default()
    };
    assert_eq!(t.cost_to_abs_realized_ratio(), None);
}
#[test]
fn finite_rows_cannot_overflow_components_or_selected_net_silently() {
    for rows in [
        vec![row(1, 10, f64::MAX), row(2, 10, f64::MAX)],
        vec![
            row(1, 10, f64::MAX),
            kind(2, f64::MAX, "COMMISSION", "USDT"),
        ],
        vec![
            kind(1, f64::MAX, "TRANSFER", "USDT"),
            kind(2, f64::MAX, "TRANSFER", "USDT"),
        ],
    ] {
        assert_eq!(
            aggregate_income_by_asset(&rows).unwrap_err(),
            IncomeEvidenceError::NonFiniteAggregate
        );
    }
}
#[test]
fn aggregation_is_permutation_invariant_for_exactly_representable_fixture() {
    let rows = vec![
        row(1, 10, 2.0),
        row(2, 10, -1.0),
        kind(3, -0.5, "COMMISSION", "USDT"),
    ];
    let a = aggregate_income_by_asset(&rows).unwrap();
    let mut reversed = rows;
    reversed.reverse();
    let b = aggregate_income_by_asset(&reversed).unwrap();
    assert_eq!(
        a.by_asset["USDT"].selected_net(),
        b.by_asset["USDT"].selected_net()
    );
}
#[test]
fn lookback_rejects_overflow_zero_and_pre_epoch_instead_of_wrapping() {
    assert_eq!(income_lookback_start(2 * 86_400_000, 1), Ok(86_400_000));
    for (now, days) in [(100, 0), (100, 1), (u64::MAX, u64::MAX)] {
        assert!(income_lookback_start(now, days).is_err());
    }
}
#[tokio::test]
async fn real_adapter_paper_mode_is_explicitly_not_exchange_evidence() {
    let mut exec =
        execution_engine::executor::OrderExecutor::new("fixture".into(), "fixture".into(), true);
    exec.set_paper_trading(true);
    let w = exec.fetch_income_window(&[], 1, 100, 2).await.unwrap();
    assert_eq!(w.coverage, IncomeCoverage::Simulated);
    assert_eq!(w.pages_read, 0);
    assert!(w.into_exhausted_entries().is_err());
    assert!(exec.fetch_income_paged(&[], 1, 2).await.is_err());
    assert!(exec.fetch_income_window(&[], 2, 1, 2).await.is_err());
    assert!(exec.fetch_income_window(&[], 1, 100, 0).await.is_err());
}

// ═════════════════ FMT-285 (OLA XL): cobertura por símbolo + cuarentena ═════════════════

#[test]
fn fmt285_invalid_row_is_quarantined_not_window_fatal() {
    // Antes (FMT-282): identity() erróneo abortaba la tanda entera. Ahora la
    // fila inválida se aparta RECUPERABLE y las demás conservan su evidencia.
    let mut bad = row(7, 10, 1.0);
    bad.tran_id = 0; // contrato de identidad violado
    let batch = vec![row(1, 10, 5.0), bad, row(2, 20, -3.0)];
    let p = partition_income(batch.clone());
    assert_eq!(p.accepted.len(), 2);
    assert_eq!(p.quarantined.len(), 1);
    assert_eq!(p.quarantined[0].reason, QuarantineReason::InvalidRecord);
    assert!(p.quarantined[0].recoverable);
    // Conservación: nada desaparece ni se anota a cero.
    assert_eq!(p.accounted_rows(), batch.len() as u64);
}

#[test]
fn fmt285_conflicting_identity_quarantines_as_non_recoverable() {
    // Identidad visible repetida con importe distinto: antes ConflictingRecord
    // mataba la recolección; ahora se aparta exigindo conciliación, y la fila
    // ORIGINAL permanece aceptada con su importe.
    let original = row(42, 10, 5.0);
    // MISMA identidad visible (mismo tipo/símbolo/asset/trade) y OTRO importe.
    let conflicting = row(42, 10, 99.0);
    let p = partition_income(vec![original.clone(), conflicting]);
    assert_eq!(p.accepted.len(), 1);
    assert_eq!(p.accepted[0].income, 5.0);
    assert_eq!(p.quarantined.len(), 1);
    assert_eq!(p.quarantined[0].reason, QuarantineReason::ConflictingIdentity);
    assert!(!p.quarantined[0].recoverable);
}

#[test]
fn fmt285_identical_duplicate_is_dropped_without_quarantine() {
    // La repetición EXACTA de una identidad ya aceptada ya está contada: no es
    // conflicto ni cuarentena, sólo descarte de doble lectura.
    let p = partition_income(vec![row(9, 10, 2.0), row(9, 10, 2.0), row(10, 30, 1.0)]);
    assert_eq!(p.accepted.len(), 2);
    assert!(p.quarantined.is_empty());
    assert_eq!(p.accounted_rows(), 3);
}

#[test]
fn fmt285_interval_by_symbol_observes_only_this_traversal() {
    let mut a1 = row(1, 100, 1.0);
    a1.symbol = "AAAUSDT".into();
    let mut a2 = row(2, 300, 1.0);
    a2.symbol = "AAAUSDT".into();
    let mut b1 = row(3, 200, 1.0);
    b1.symbol = "BBBUSDT".into();
    let p = partition_income(vec![a1, b1, a2]);
    let aaa = p.interval_by_symbol.get("AAAUSDT").unwrap();
    assert_eq!((aaa.first_time_ms, aaa.last_time_ms, aaa.rows), (100, 300, 2));
    let bbb = p.interval_by_symbol.get("BBBUSDT").unwrap();
    assert_eq!((bbb.first_time_ms, bbb.last_time_ms, bbb.rows), (200, 200, 1));
    // Símbolo ausente: sin entrada — ausencia de filas NO es cobertura vacía
    // ni prueba de retención; el mapa sólo atestigua lo observado.
    assert!(!p.interval_by_symbol.contains_key("CCCUSDT"));
}

#[test]
fn fmt285_quarantine_debits_only_its_own_symbol_coverage() {
    // La cuarentena de UN símbolo no puede suprimir la cobertura observada de
    // los demás: se reporta por símbolo, no como veto global.
    let mut bad = row(5, 50, 1.0);
    bad.symbol = "AAAUSDT".into();
    bad.tran_id = 0;
    let mut ok = row(6, 60, 1.0);
    ok.symbol = "BBBUSDT".into();
    let p = partition_income(vec![bad, ok]);
    assert!(p.interval_by_symbol.contains_key("BBBUSDT"));
    assert!(!p.interval_by_symbol.contains_key("AAAUSDT"));
    let debilitated = p.symbols_with_quarantine();
    assert_eq!(debilitated.get("AAAUSDT"), Some(&1));
    assert!(!debilitated.contains_key("BBBUSDT"));
}

// ── FMT-285b: el recorrido con transporte alimenta la cuarentena ──────

/// Una página con MEZCLA de válidos e inválidos: los válidos entran, los
/// inválidos a cuarentena, la ventana NO aborta y termina agotada — un
/// registro malo del exchange ya no cuesta la evidencia entera del día.
#[tokio::test]
async fn fmt285b_registro_malo_no_mata_la_ventana_mixta() {
    let mut invalid = row(2, 10, 1.0);
    invalid.asset.clear();
    let w = collect(
        vec![vec![row(1, 10, 5.0), invalid, row(3, 20, -1.0)]],
        10,
        1,
    )
    .await
    .unwrap();
    assert_eq!(w.entries.len(), 2, "los válidos entran");
    assert_eq!(w.quarantined.len(), 1, "el inválido se cuarentena");
    assert_eq!(w.coverage, IncomeCoverage::PageExhausted);
    let ids: Vec<u64> = w.entries.iter().map(|e| e.tran_id).collect();
    assert_eq!(ids, vec![1, 3]);
    // La ventana agotada sigue siendo consumible como evidencia completa.
    assert_eq!(w.into_exhausted_entries().unwrap().len(), 2);
}

/// Cuarentenas NUEVAS son progreso del recorrido: una página llena que sólo
/// trae registros fuera de rango NO se reporta como NoProgress (el recorrido
/// siguió hasta agotar su presupuesto) — pero una página de puros
/// duplicados exactos SÍ es estancamiento.
#[tokio::test]
async fn fmt285b_cuarentena_nueva_es_progreso_no_estancamiento() {
    // Presupuesto exacto de 2 páginas: la página 2 (llena, toda en
    // cuarentena) NO estanca — el recorrido la consume y agota presupuesto.
    let w = collect(
        vec![
            vec![row(1, 10, 1.0), row(2, 20, 1.0)],
            vec![row(3, 999, 1.0), row(4, 999, 1.0)],
        ],
        2,
        2,
    )
    .await
    .unwrap();
    assert_eq!(w.pages_read, 2, "la página de cuarentenas se leyó");
    assert_eq!(w.quarantined.len(), 2);
    assert_eq!(
        w.coverage,
        IncomeCoverage::PageBudgetExceeded,
        "no se estancó en NoProgress tras la página 1"
    );
    // Pura repetición exacta: eso SÍ es NoProgress (sin nueva información).
    let w = collect(
        vec![
            vec![row(1, 10, 1.0), row(2, 20, 1.0)],
            vec![row(1, 10, 1.0), row(2, 20, 1.0)],
        ],
        2,
        3,
    )
    .await
    .unwrap();
    assert_eq!(w.coverage, IncomeCoverage::NoProgress);
    assert_eq!(w.exact_duplicates_dropped, 2);
}

/// PARIDAD con partition_income: la misma tanda procesada por el recorrido
/// (una página) y por la partición pura produce las MISMAS admisiones y los
/// MISMOS motivos de cuarentena — un solo contrato de identidad, dos puertas.
#[tokio::test]
async fn fmt285b_paridad_con_partition_income() {
    let mut invalid = row(2, 10, 1.0);
    invalid.tran_id = 0;
    let batch = vec![row(1, 10, 1.0), invalid, row(1, 10, -9.0), row(3, 30, 2.0)];
    let w = collect(vec![batch.clone()], 10, 1).await.unwrap();
    let p = partition_income(batch);
    assert_eq!(w.entries.len(), p.accepted.len());
    assert_eq!(w.quarantined.len(), p.quarantined.len());
    for (a, b) in w.quarantined.iter().zip(p.quarantined.iter()) {
        assert_eq!(a.reason, b.reason);
        assert_eq!(a.recoverable, b.recoverable);
    }
    assert_eq!(
        w.exact_duplicates_dropped,
        p.exact_duplicates_dropped as u64
    );
}

// ── XLVI·G / §13.1: payload de contradicción con bits exactos ────────

/// El conflicto cuarentenado lleva AMBOS importes con bits EXACTOS y el
/// instante compartido: la conciliación resuelve con el payload (¿bits
/// distintos = revisión real del proveedor?), no con la sospecha.
#[tokio::test]
async fn xlvig_conflicto_lleva_ambos_importes_en_bits() {
    let w = collect(vec![vec![row(1, 10, 1.0), row(1, 10, 1.0000000000000002)]], 10, 1)
        .await
        .unwrap();
    assert_eq!(w.quarantined.len(), 1);
    let c = w.quarantined[0].conflict.as_ref().expect("payload §13.1");
    assert_eq!(c.accepted_bits, 1.0_f64.to_bits());
    assert_eq!(c.new_bits, 1.0000000000000002_f64.to_bits());
    assert_ne!(c.accepted_bits, c.new_bits, "revisión real, no reparseo");
    assert_eq!(c.accepted_income, 1.0);
    assert_eq!(c.accepted_time_ms, 10, "instante de la identidad compartida");
    // Las cuarentenas NO-conflicto no fabrican payload.
    let mut invalid = row(9, 10, 1.0);
    invalid.tran_id = 0;
    let w2 = collect(vec![vec![invalid]], 10, 1).await.unwrap();
    assert!(w2.quarantined[0].conflict.is_none());
}

/// El payload de contradicción es idéntico por las DOS puertas (recorrido
/// y partición) — un contrato de identidad, una evidencia de conflicto.
#[tokio::test]
async fn xlvig_payload_de_conflicto_paridad_entre_puertas() {
    let batch = vec![row(1, 10, 5.0), row(1, 10, -5.0)];
    let w = collect(vec![batch.clone()], 10, 1).await.unwrap();
    let p = partition_income(batch);
    let cw = w.quarantined[0].conflict.as_ref().unwrap();
    let cp = p.quarantined[0].conflict.as_ref().unwrap();
    assert_eq!(cw, cp);
}

// ── XLVI·G / §13.3: FX as-of y balance de flujos ─────────────────────

/// Mapa de tasas (par, instante) → tasa para los tests.
struct FxTabla<'a>(&'a [((&'a str, u64), f64)]);
impl FxAsOf for FxTabla<'_> {
    fn rate(&self, from: &str, to: &str, time_ms: u64) -> Option<f64> {
        // El proveedor responde por PAR DIRIGIDO; sin tasa para otro instante.
        self.0
            .iter()
            .find(|((f, t), _)| *f == from && *t == time_ms)
            .map(|(_, r)| *r)
            .filter(|_| to == "USDT")
    }
}

/// La conversión es AS-OF: cada flujo usa la tasa de SU instante, no la
/// actual. Flujos sin tasa quedan como subtotales independientes por activo
/// — nunca un total mixto silencioso, nunca descarte.
#[test]
fn xlvig_fx_convierte_as_of_y_deja_independiente_lo_sin_tasa() {
    let mut bnb_t1 = row(1, 100, 2.0);
    bnb_t1.asset = "BNB".into();
    let mut bnb_t2 = row(2, 200, 1.0);
    bnb_t2.asset = "BNB".into();
    let mut btc = row(3, 100, 0.5);
    btc.asset = "BTC".into(); // sin tasa en el mapa
    let usdt = row(4, 100, 10.0);
    let entries = vec![bnb_t1, bnb_t2, btc, usdt];

    let fx = FxTabla(&[(("BNB", 100), 600.0), (("BNB", 200), 500.0)]);
    let b = fx_balance_as_of(&entries, "USDT", &fx).unwrap();
    // BNB@100: 2×600; BNB@200: 1×500 (as-of distinta por instante); USDT 10.
    assert!((b.converted_net - (1200.0 + 500.0 + 10.0)).abs() < 1e-9);
    // BTC sin tasa as-of: subtotal independiente, NI convertido NI descartado.
    assert!((b.unconverted_by_asset["BTC"] - 0.5).abs() < 1e-12);
    assert_eq!(b.conversions.len(), 3, "2 conversiones + 1 passthrough");
    assert!(b.conversions.iter().any(|c| c.passthrough));
    // La mirada de tasa NO es lookahead: la tasa del instante 200 (500)
    // no se aplicó al flujo del instante 100.
    assert_eq!(b.conversions[0].rate, 600.0);
    assert_eq!(b.conversions[1].rate, 500.0);
}

/// Tasa rota del proveedor (NaN/≤0) se trata como SIN tasa — el balance
/// nunca se envenena; y el balance es invariante a permutación de filas.
#[test]
fn xlvig_fx_tasa_rota_es_sin_tasa_y_el_balance_es_permutable() {
    let mut b = row(1, 100, 3.0);
    b.asset = "BNB".into();
    let fx_rota = FxTabla(&[(("BNB", 100), f64::NAN)]);
    let bal = fx_balance_as_of(&[b.clone()], "USDT", &fx_rota).unwrap();
    assert!((bal.unconverted_by_asset["BNB"] - 3.0).abs() < 1e-12);
    assert_eq!(bal.conversions.len(), 0);

    let fx_neg = FxTabla(&[(("BNB", 100), -1.0)]);
    let bal = fx_balance_as_of(&[b], "USDT", &fx_neg).unwrap();
    assert!(bal.unconverted_by_asset.contains_key("BNB"));

    // Permutación: mismo neto, mismos subtotales.
    let mut x = row(1, 100, 2.0);
    x.asset = "BNB".into();
    let y = row(2, 100, 7.0);
    let fx = FxTabla(&[(("BNB", 100), 600.0)]);
    let a = fx_balance_as_of(&[x.clone(), y.clone()], "USDT", &fx).unwrap();
    let c = fx_balance_as_of(&[y, x], "USDT", &fx).unwrap();
    assert_eq!(a.converted_net, c.converted_net);
}

/// El guard legado sigue intacto: quien NO convierte sigue recibiendo
/// MixedAssets — esta vía no lo reemplaza, lo complementa.
#[test]
fn xlvig_guard_legado_mixed_assets_sigue_disponible() {
    let mut b = row(1, 10, 1.0);
    b.asset = "BNB".into();
    assert_eq!(
        single_income_asset(&[b, row(2, 10, 1.0)]),
        Err(IncomeEvidenceError::MixedAssets)
    );
}
