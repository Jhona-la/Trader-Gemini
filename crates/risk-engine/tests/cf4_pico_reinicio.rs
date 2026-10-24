//! C-F4 · LC-F4-10 / LC-F0-10 (Línea C, CL-99): un solo pico del capital,
//! compartido por los dos cortacircuitos de drawdown y guardado entre
//! reinicios.
//!
//! Teoría. El veto de entradas del riesgo y el sistema inmune del host miden
//! la caída desde el pico del capital. Si el pico se pierde al reiniciar, la
//! caída se borra: con el pico en 50 USD y la cuenta en 13, un proceso nuevo
//! nace con el pico en 13 y ve una caída de 0. El piso de falsación de
//! ADR-0016 (D3, d* = 0,632) deja de proteger justo cuando más falta hace.
//!
//! Antes había dos picos: `RiskEngine::peak_capital` y una variable local de
//! la tarea inmune del host, que nacía con el capital del arranque. Ninguno
//! se guardaba y ninguno veía al otro. Ahora el pico vive en el arena, los dos
//! lo actualizan con `drawdown::registrar_pico` y el host lo guarda con
//! `guardar_pico` y lo recupera con `leer_pico`. Los umbrales no cambian.
//!
//! Fixture. Cuenta de 13 USD con el gen de drawdown por defecto. Una caída
//! de 50 a 13 USD es del 74 %, por encima de d* = 0,632, la cota superior de
//! cualquier umbral del veto: se veta con cualquier gen.

use quantum_arena::GlobalArena;
use risk_engine::drawdown::{guardar_pico, leer_pico, registrar_pico, CargaPico};
use risk_engine::{RiskEngine, REJECT_NAMES};
use signal_engine::{SignalIntent, SignalType};
use std::sync::{atomic::Ordering::Relaxed, Arc};

fn larga() -> SignalIntent {
    SignalIntent {
        signal: SignalType::Long,
        confidence: 0.9,
        win_probability: 0.9,
        expected_duration_ms: 60_000,
        ..SignalIntent::default()
    }
}

/// ¿Veta por drawdown un motor de riesgo recién creado (como tras un
/// reinicio) sobre `arena`?
fn veto_de_drawdown(arena: &GlobalArena, capital_inicial: f64) -> bool {
    let orden = RiskEngine::new(capital_inicial).evaluate_quantum_order(0, &larga(), arena);
    orden.signal == SignalType::Flat
        && risk_engine::ultimo_rechazo().map(|i| REJECT_NAMES[i]) == Some("drawdown")
}

fn cuenta(capital: f64) -> Arc<GlobalArena> {
    let a = GlobalArena::build_in_own_stack(capital);
    a.coins[0].current_price.store(100.0, Relaxed);
    a.coins[0].current_atr.store(1.0, Relaxed);
    a
}

fn ruta_temporal(nombre: &str) -> std::path::PathBuf {
    std::env::temp_dir().join(format!("cl99_{}_{}", std::process::id(), nombre))
}

/// Esperado: el pico que registra el host (por ejemplo, 50 USD vistos por el
/// sistema inmune) lo ve el veto de entradas del riesgo, aunque este motor
/// nunca haya visto ese capital.
#[test]
fn cl99_el_pico_es_uno_para_host_y_riesgo() {
    let a = cuenta(13.0);
    registrar_pico(&a.pico_capital, 50.0);
    assert!(
        veto_de_drawdown(&a, 13.0),
        "el riesgo no vio el pico del host: {}",
        risk_engine::reject_report()
    );
}

/// Esperado: el pico guardado antes de reiniciar se recupera en el proceso
/// nuevo, y el veto de drawdown sigue disparando. Sin restaurarlo, el mismo
/// reinicio borra la caída (lo que pasaba antes de CL-99).
#[test]
fn cl99_el_pico_guardado_sobrevive_al_reinicio() {
    let ruta = ruta_temporal("pico_capital.txt");
    let _ = std::fs::remove_file(&ruta);

    // Proceso anterior: la cuenta llegó a 50 USD y el host guardó el pico.
    let antes = cuenta(50.0);
    let pico = registrar_pico(&antes.pico_capital, 50.0);
    guardar_pico(&ruta, pico).expect("guardar el pico");

    // Reinicio sin restaurar: el pico nace en el capital ya caído.
    let sin_restaurar = cuenta(13.0);
    let veta_sin = veto_de_drawdown(&sin_restaurar, 13.0);

    // Reinicio restaurando el pico guardado.
    let restaurado = cuenta(13.0);
    let guardado = match leer_pico(&ruta) {
        CargaPico::Leido(p) => p,
        otro => panic!("leer el pico guardado: {otro:?}"),
    };
    registrar_pico(&restaurado.pico_capital, guardado);
    let veta_con = veto_de_drawdown(&restaurado, 13.0);
    let _ = std::fs::remove_file(&ruta);

    println!("observado: sin restaurar veta {veta_sin}; restaurando veta {veta_con}");
    assert!(!veta_sin, "precondición: sin restaurar, el reinicio borra la caída");
    assert!(veta_con, "con el pico guardado, la caída del 74 % debe vetarse");
}

/// Esperado: el pico sólo sube y un capital inválido no lo mueve.
#[test]
fn cl99_el_pico_es_monotono_y_no_admite_capitales_invalidos() {
    let a = cuenta(13.0);
    assert_eq!(registrar_pico(&a.pico_capital, 20.0), 20.0);
    assert_eq!(registrar_pico(&a.pico_capital, 15.0), 20.0, "no baja");
    for invalido in [f64::NAN, f64::INFINITY, -1.0, 0.0] {
        assert_eq!(registrar_pico(&a.pico_capital, invalido), 20.0, "{invalido}");
    }
}

/// Esperado: guardar y leer devuelve el mismo valor bit a bit, y un archivo
/// ausente o corrupto no inventa un pico.
#[test]
fn cl99_guardar_y_leer_el_pico_es_exacto() {
    let ruta = ruta_temporal("pico_exacto.txt");
    let pico = 13.000_000_000_000_002_f64;
    guardar_pico(&ruta, pico).expect("guardar");
    assert!(matches!(leer_pico(&ruta), CargaPico::Leido(p) if p.to_bits() == pico.to_bits()));
    for corrupto in ["", "abc", "NaN", "inf", "-5", "0"] {
        std::fs::write(&ruta, corrupto).unwrap();
        assert!(
            matches!(leer_pico(&ruta), CargaPico::Ilegible(_)),
            "contenido {corrupto:?}"
        );
    }
    let _ = std::fs::remove_file(&ruta);
    assert!(matches!(leer_pico(&ruta), CargaPico::Ausente), "sin archivo");
    assert!(guardar_pico(&ruta, f64::NAN).is_err());
}
