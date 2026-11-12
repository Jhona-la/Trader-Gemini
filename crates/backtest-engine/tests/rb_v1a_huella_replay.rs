//! CERTIFICACIÓN RB-V1a (local, `#[ignore]`): huella bit a bit de un replay.
//!
//! QUÉ: reproduce una cinta real con el núcleo de producción y escribe una
//! línea `HUELLA_REPLAY` con las operaciones y los bits del PnL, las comisiones
//! y el capital final.
//! POR QUÉ: el libro en sombra de RB-V1a sólo observa. La prueba de que no
//! cambia ninguna decisión es correr ESTE MISMO archivo (con `cinta_real/`)
//! sobre `main` sin el libro y sobre la rama con él, y comparar las dos
//! huellas: tienen que ser idénticas bit a bit. Por eso sólo usa API de
//! `backtest-engine` anterior al átomo.
//! CÓMO: `RB_V1A_CINTA=<cinta> [RB_V1A_TICKS=n] cargo test -p backtest-engine
//! --test rb_v1a_huella_replay -- --ignored --nocapture`.

mod cinta_real;

use backtest_engine::booktick_replay::run_booktick_replay;

#[test]
#[ignore = "certificación local: necesita RB_V1A_CINTA con una cinta real de data/ (no versionada)"]
fn rb_v1a_huella_del_replay_sobre_cinta_real() {
    let simbolo = cinta_real::registrar_simbolo_de_la_cinta();
    let ticks = cinta_real::cargar_desde_entorno();
    let cfg = cinta_real::config_de_referencia();
    let genoma = cinta_real::genoma_de_referencia();
    let t0 = std::time::Instant::now();
    let s = run_booktick_replay(&ticks, &genoma, None, &cfg);
    println!(
        "HUELLA_REPLAY simbolo={simbolo} ticks={} desde_ms={} hasta_ms={} trades={} wins_net={} wins_gross={} net_pnl_bits={:016x} gross_pnl_bits={:016x} fees_bits={:016x} final_capital_bits={:016x} ejecutados={} segundos={:.1}",
        ticks.len(),
        ticks.first().map_or(0, |t| t.ts_ms),
        ticks.last().map_or(0, |t| t.ts_ms),
        s.trades,
        s.wins_net,
        s.wins_gross,
        s.net_pnl.to_bits(),
        s.gross_pnl.to_bits(),
        s.fees_est.to_bits(),
        s.final_capital.to_bits(),
        s.executed_ticks,
        t0.elapsed().as_secs_f64()
    );
    assert!(s.executed_ticks > 0, "el replay no ejecutó ningún tick");
}
