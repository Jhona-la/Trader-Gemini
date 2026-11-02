//! Contrato de compuerta de capital para los bins `evolution` y
//! `multi_coin_simulator` (GLM-179, §25/CL-15).
//!
//! Continuación del `continuous_capital_contract` (GLM-178, contrato de
//! Kimi V33): ambos bins caían al 13.0 prohibido («fallback inteligente»
//! del FIX #1528 y «default sin pánico» respectivamente). Ahora comparten
//! `quantum_engine::capital_entrada::exigir_inicial()` — UNA puerta.
//!
//! Invariante: capital ausente o inválido ⇒ salida no exitosa con
//! diagnóstico explícito de INITIAL_CAPITAL y CERO artefactos en el
//! directorio de trabajo (la puerta va antes de cualquier efecto).
//! El caso «capital válido fluye» no se ejercita aquí: estos bins
//! arrancan arenas y hilos de fondo no acotados; esa dirección ya está
//! certificada por el contrato del backtester.

use std::fs;
use std::path::PathBuf;
use std::process::Command;
use std::time::{SystemTime, UNIX_EPOCH};

struct Isolated {
    root: PathBuf,
}

impl Isolated {
    fn new() -> Self {
        let nonce = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .map(|duration| duration.as_nanos())
            .unwrap_or(0);
        let root =
            std::env::temp_dir().join(format!("capital-entry-{}-{nonce}", std::process::id()));
        fs::create_dir_all(&root).expect("create isolated directory");
        Self { root }
    }
}

impl Drop for Isolated {
    fn drop(&mut self) {
        if let Err(error) = fs::remove_dir_all(&self.root) {
            eprintln!("isolated cleanup failed for {:?}: {error}", self.root);
        }
    }
}

fn assert_rejected(bin: &str, capital: Option<&str>) {
    let isolated = Isolated::new();
    let mut command = Command::new(bin);
    command
        .current_dir(&isolated.root)
        .env_remove("INITIAL_CAPITAL");
    if let Some(value) = capital {
        command.env("INITIAL_CAPITAL", value);
    }
    let output = command
        .output()
        .unwrap_or_else(|error| panic!("run {bin}: {error}"));
    assert!(
        !output.status.success(),
        "{bin} con INITIAL_CAPITAL={capital:?} debe fallar: {output:?}"
    );
    let diagnostics = format!(
        "{}{}",
        String::from_utf8_lossy(&output.stdout),
        String::from_utf8_lossy(&output.stderr)
    );
    assert!(
        diagnostics.contains("INITIAL_CAPITAL"),
        "{bin}: falta diagnóstico explícito de INITIAL_CAPITAL: {diagnostics}"
    );
    let residuo: Vec<_> = fs::read_dir(&isolated.root)
        .expect("read isolated dir")
        .filter_map(|e| e.ok())
        .map(|e| e.file_name())
        .collect();
    assert!(
        residuo.is_empty(),
        "{bin} con {capital:?}: la puerta debe ir antes de cualquier efecto; residuo: {residuo:?}"
    );
}

#[test]
fn evolution_rechaza_capital_invalido_antes_de_efectos() {
    for capital in [
        None,
        Some(""),
        Some("abc"),
        Some("NaN"),
        Some("inf"),
        Some("-inf"),
        Some("0"),
        Some("-1"),
    ] {
        assert_rejected(env!("CARGO_BIN_EXE_evolution"), capital);
    }
}

#[test]
fn multi_coin_simulator_rechaza_capital_invalido_antes_de_efectos() {
    for capital in [
        None,
        Some(""),
        Some("abc"),
        Some("NaN"),
        Some("inf"),
        Some("-inf"),
        Some("0"),
        Some("-1"),
    ] {
        assert_rejected(env!("CARGO_BIN_EXE_multi_coin_simulator"), capital);
    }
}
