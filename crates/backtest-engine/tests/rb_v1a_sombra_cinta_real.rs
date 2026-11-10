//! CERTIFICACIÓN RB-V1a (local, `#[ignore]`): los vetos posteriores al riesgo,
//! juzgados en sombra sobre una cinta real con el núcleo de producción.
//!
//! QUÉ: reproduce la cinta con `run_booktick_replay_with_observer` (el mismo
//! camino de entrada que el vivo) y lee, antes de cada evento, el libro
//! `sombra_vetos` del núcleo. Al final escribe, por fuente (consejo, ML-GATE sin
//! modelo, ML-GATE sin ventaja), las intenciones resueltas, el R neto de
//! fricción, el radio de Hoeffding y el veredicto; y el resultado financiero
//! del replay con ROI, PnL y WR antes y después de comisiones y el periodo.
//! POR QUÉ: el mandato prohíbe dar por terminado lo que sólo compila. Esta es
//! la prueba de conducta del libro sobre microestructura real: registra, avanza
//! y resuelve intenciones de verdad.
//! CÓMO: `RB_V1A_CINTA=<cinta> [RB_V1A_TICKS=n] cargo test -p backtest-engine
//! --test rb_v1a_sombra_cinta_real -- --ignored --nocapture`.
//! Límites: genoma de referencia, omni neutro y sin modelos del roster (el
//! núcleo de un replay no los carga: LG-23), así que el ML-GATE veta como
//! «sin modelo». Lo que mide es si esas intenciones habrían pagado.

mod cinta_real;

use backtest_engine::booktick_replay::run_booktick_replay_with_observer;
use god_engine_core::sombra_vetos::{DELTA_COLA, FUENTES};

#[test]
#[ignore = "certificación local: necesita RB_V1A_CINTA con una cinta real de data/ (no versionada)"]
fn rb_v1a_sombra_de_los_vetos_sobre_cinta_real() {
    let simbolo = cinta_real::registrar_simbolo_de_la_cinta();
    let ticks = cinta_real::cargar_desde_entorno();
    let cfg = cinta_real::config_de_referencia();
    let genoma = cinta_real::genoma_de_referencia();

    let mut fuentes = [None; FUENTES.len()];
    let mut veredictos = [None; FUENTES.len()];
    let mut libro = (0usize, 0u64, 0u64, 0u64, 0u64, 0u64);
    let mut embudo = (0u64, 0u64, 0u64, 0u64, 0u64);
    let t0 = std::time::Instant::now();
    let s = run_booktick_replay_with_observer(&ticks, &genoma, None, &cfg, |_, core| {
        embudo = (
            core.diag_sin_conviccion,
            core.diag_council_vetoes,
            core.diag_ml_vetoes,
            core.diag_margin_drops,
            core.diag_opened,
        );
        let sombra = &core.sombra_vetos;
        for ((&(fuente, _), stats), veredicto) in FUENTES
            .iter()
            .zip(fuentes.iter_mut())
            .zip(veredictos.iter_mut())
        {
            *stats = sombra.estadistica(fuente);
            *veredicto = Some(sombra.veredicto(fuente));
        }
        let l = sombra.libro();
        libro = (
            l.abiertas(),
            l.descartadas_solape,
            l.descartadas_llenas,
            l.descartadas_invalidas,
            sombra.sin_geometria,
            sombra.avisos,
        );
    });
    let segundos = t0.elapsed().as_secs_f64();

    let desde = ticks.first().map_or(0, |t| t.ts_ms);
    let hasta = ticks.last().map_or(0, |t| t.ts_ms);
    let capital = cfg.initial_capital;
    let por_trade = |x: f64| {
        if s.trades > 0 {
            x / s.trades as f64
        } else {
            f64::NAN
        }
    };
    println!(
        "RB-V1a · {simbolo} · periodo {desde} → {hasta} ms epoch ({:.1} h) · {} ticks · {segundos:.0} s de replay",
        hasta.saturating_sub(desde) as f64 / 3.6e6,
        ticks.len()
    );
    println!(
        "  replay: {} operaciones · PnL bruto {:+.4} / neto {:+.4} USDT · comisiones {:.4} · ROI bruto {:+.4} % / neto {:+.4} % · WR bruto {:.1} % / neto {:.1} % · EV/op neto {:+.5} · Sharpe {:.3} · DD máx {:.4}",
        s.trades,
        s.gross_pnl,
        s.net_pnl,
        s.fees_est,
        100.0 * s.gross_pnl / capital,
        100.0 * s.net_pnl / capital,
        100.0 * por_trade(s.wins_gross as f64),
        100.0 * por_trade(s.wins_net as f64),
        por_trade(s.net_pnl),
        s.sharpe,
        s.max_dd
    );
    for ((&(_, nombre), stats), veredicto) in FUENTES.iter().zip(fuentes).zip(veredictos) {
        let Some(e) = stats else {
            panic!("la fuente {nombre} existe en el libro");
        };
        assert_eq!(
            e.tp + e.sl + e.vencidas,
            e.n,
            "{nombre}: toda resolución tiene un motivo"
        );
        println!(
            "  sombra {nombre}: {} resueltas (TP {} · SL {} · τ {}) · R neto medio {} ± {} (Hoeffding, δ {DELTA_COLA:.4}/cola) → {veredicto:?}",
            e.n,
            e.tp,
            e.sl,
            e.vencidas,
            e.media_r().map_or("—".to_string(), |m| format!("{m:+.3}")),
            e.radio_hoeffding(DELTA_COLA).map_or("—".to_string(), |r| format!("{r:.3}")),
        );
        if e.n > 0 {
            assert!(
                e.media_r().is_some_and(f64::is_finite),
                "{nombre}: media finita"
            );
        }
    }
    let (abiertas, solapes, llenas, invalidas, sin_geometria, avisos) = libro;
    println!(
        "  libro: abiertas {abiertas} · solapes {solapes} · llenas {llenas} · inválidas {invalidas} · sin geometría {sin_geometria} · avisos {avisos}"
    );
    let (sin_conviccion, consejo, ml, margen, aperturas) = embudo;
    println!(
        "  embudo del núcleo: sin convicción {sin_conviccion} · vetos del consejo {consejo} · vetos del ML-GATE {ml} · sin margen {margen} · aperturas {aperturas}"
    );
    println!("  rechazos del riesgo: {}", risk_engine::reject_report());
    assert!(s.executed_ticks > 0, "el replay no ejecutó ningún tick");
}
