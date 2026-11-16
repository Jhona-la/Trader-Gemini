//! WB-4/WB-5 EN SOMBRA (integrador, 2026-11-16) — CERTIFICACIÓN LOCAL
//! (`#[ignore]`): con la calibración contrafactual (WB-4) y el f encogido
//! (WB-5) calculados y registrados en sombra, el tamaño y la decisión de cada
//! orden del replay de referencia son IDÉNTICOS byte a byte a los de C3
//! (`5ef982bb7`).
//!
//! QUÉ: reproduce una cinta real con el arnés de referencia de RB-V1a
//! (`cinta_real`: genoma de referencia, capital 1000, calentamiento 500) por
//! `run_booktick_replay_with_observer`, el mismo camino de entrada que el
//! vivo. Antes de cada evento lee los slots de la moneda 0 y escribe una línea
//! cada vez que cambia alguno: lado, precio de entrada, cantidad, margen, TP,
//! SL e instante de entrada (los f64 como bits exactos), más el capital
//! unificado, o cada vez que cambia un contador de decisión del embudo
//! (aperturas, vetos del consejo y del ML-GATE, sin convicción, caídas por
//! margen). Al final añade el resumen del replay (`{:?}` de f64 es exacto).
//! Esa traza ES el tamaño y la decisión de cada orden y de cada intención.
//! Referencia de C3: el ejecutable de esta prueba compilado sobre C3 sin
//! cambios (`integrador\wb45_logs\traza_c3.exe`), con `RB_V1A_TICKS` mayor que
//! la cinta (cinta completa).
//! CÓMO: `RB_V1A_CINTA=<cinta> [RB_V1A_TICKS=n] WB45_TRAZA_SALIDA=<archivo>
//! [WB45_TRAZA_REFERENCIA=<traza de C3>] cargo test -p backtest-engine --test
//! wb45_sombra_identidad_ordenes -- --ignored --nocapture`. Sin referencia sólo
//! escribe (así se sacó la de C3); con ella exige igualdad byte a byte.
//! Control: la traza debe contener al menos una apertura (si no, la igualdad
//! no diría nada), y los contadores de la sombra se imprimen aparte.

mod cinta_real;

use backtest_engine::booktick_replay::run_booktick_replay_with_observer;
use std::fmt::Write as _;
use std::sync::atomic::Ordering;

/// La firma de los slots de la moneda 0: todo lo que decide y dimensiona una
/// orden abierta, con los f64 como bits.
fn firma_de_slots(core: &god_engine_core::GodEngineCore) -> String {
    let mut s = String::new();
    for (k, p) in core.arena.coins[0].positions.slots().iter().enumerate() {
        if !p.is_open.load(Ordering::Relaxed) {
            let _ = write!(s, "|{k}:-");
            continue;
        }
        let _ = write!(
            s,
            "|{k}:{} e={:016x} q={:016x} m={:016x} tp={:016x} sl={:016x} t={}",
            if p.is_long.load(Ordering::Relaxed) { "L" } else { "S" },
            p.entry_price.load(Ordering::Relaxed).to_bits(),
            p.quantity.load(Ordering::Relaxed).to_bits(),
            p.margin_used.load(Ordering::Relaxed).to_bits(),
            p.tp_price.load(Ordering::Relaxed).to_bits(),
            p.sl_price.load(Ordering::Relaxed).to_bits(),
            p.entry_time_ms.load(Ordering::Relaxed),
        );
    }
    s
}

#[test]
#[ignore = "certificación local: necesita RB_V1A_CINTA con una cinta real de data/ (no versionada)"]
fn wb45_sombra_no_cambia_ninguna_orden_del_replay_de_referencia() {
    let _simbolo = cinta_real::registrar_simbolo_de_la_cinta();
    let ticks = cinta_real::cargar_desde_entorno();
    let cfg = cinta_real::config_de_referencia();
    let genoma = cinta_real::genoma_de_referencia();

    let mut traza = String::new();
    let mut ultima = String::new();
    let mut aperturas = 0u64;
    let s = run_booktick_replay_with_observer(&ticks, &genoma, None, &cfg, |i, core| {
        // Las decisiones del embudo sobre cada intención (abrir, vetos del
        // consejo y del ML, sin convicción, caídas por margen) y la cuenta
        // de rechazos del riesgo: también son decisiones de orden.
        let firma = format!(
            "{} d={},{},{},{},{}",
            firma_de_slots(core),
            core.diag_opened,
            core.diag_council_vetoes,
            core.diag_ml_vetoes,
            core.diag_sin_conviccion,
            core.diag_margin_drops,
        );
        if firma != ultima {
            aperturas += firma.matches(" e=").count().saturating_sub(ultima.matches(" e=").count()) as u64;
            let _ = writeln!(
                traza,
                "{i} cap={:016x} {firma}",
                core.arena.unified_capital.load(Ordering::Relaxed).to_bits()
            );
            ultima = firma;
        }
    });
    let _ = writeln!(traza, "FIN {s:?}");

    let Ok(salida) = std::env::var("WB45_TRAZA_SALIDA") else {
        panic!("define WB45_TRAZA_SALIDA con la ruta de la traza a escribir");
    };
    if let Err(e) = std::fs::write(&salida, traza.as_bytes()) {
        panic!("no se pudo escribir {salida}: {e}");
    }
    println!(
        "WB45 traza: {} ticks, {} líneas, {} aperturas vistas, trades {} -> {salida}",
        ticks.len(),
        traza.lines().count(),
        aperturas,
        s.trades
    );
    assert!(
        aperturas > 0,
        "control: el replay no abrió ninguna orden; la igualdad no probaría nada"
    );
    if let Ok(referencia) = std::env::var("WB45_TRAZA_REFERENCIA") {
        let esperada = match std::fs::read(&referencia) {
            Ok(b) => b,
            Err(e) => panic!("no se pudo leer la referencia {referencia}: {e}"),
        };
        if esperada != traza.as_bytes() {
            let primera = String::from_utf8_lossy(&esperada)
                .lines()
                .zip(traza.lines())
                .position(|(a, b)| a != b);
            panic!(
                "WB-4/WB-5 en sombra cambió una orden: la traza difiere de la referencia \
                 (primera línea distinta: {primera:?}; {} vs {} bytes)",
                esperada.len(),
                traza.len()
            );
        }
        println!("WB45 traza IDÉNTICA byte a byte a {referencia}");
    }
}
