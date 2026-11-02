//! Contrato de paridad de capital en la ENTRADA del backtester continuo.
//!
//! Especificación: `docs/PLAN_MAESTRO_FASES_2026-10-22.md` (Kimi V33).
//! Borrador del test (RED-first, centinela) rescatado de su WIP sin
//! trackear — respaldo `.agent_tx/agent-fabric-huerfano-20261101/` tras la
//! tramitación de huérfano del ciclo GLM-177 — y completado por GLM-178
//! con la contención de config (`TGM_CONFIG_DIR`/`TG_CONFIG_ROOT` dentro
//! del fixture) y la variante sin CSV que el plan exigía.
//!
//! Por qué vive en el paquete RAÍZ: el binario está declarado en
//! `trader-gemini-v5` ([[bin]] del Cargo.toml raíz) aunque su fuente viva
//! bajo `crates/backtest-engine/`; `CARGO_BIN_EXE_` sólo resuelve en el
//! paquete dueño del bin.
//!
//! Invariante: capital inválido o ausente se rechaza ANTES de cualquier
//! efecto secundario — ni truncar un `equity_curve.csv` preexistente ni
//! crear `config_dir` — con diagnóstico explícito; capital válido fluye
//! hasta la validación del tape. Sin defaults silenciosos (CL-15, §25).

use std::fs;
use std::path::{Path, PathBuf};
use std::process::{Command, Output};
use std::time::{SystemTime, UNIX_EPOCH};

const SENTINEL: &[u8] = b"SENTINEL-DO-NOT-TRUNCATE";

struct Isolated {
    root: PathBuf,
}

impl Isolated {
    fn new() -> Self {
        let nonce = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .map(|duration| duration.as_nanos())
            .unwrap_or(0);
        let root = std::env::temp_dir().join(format!(
            "continuous-capital-{}-{nonce}",
            std::process::id()
        ));
        fs::create_dir_all(&root).expect("create isolated directory");
        fs::write(root.join("equity_curve.csv"), SENTINEL).expect("write sentinel");
        Self { root }
    }

    /// Variante del plan SIN csv preexistente: nada debe crearse tampoco.
    fn sin_csv() -> Self {
        let nonce = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .map(|duration| duration.as_nanos())
            .unwrap_or(0);
        let root = std::env::temp_dir().join(format!(
            "continuous-capital-{}-{nonce}",
            std::process::id()
        ));
        fs::create_dir_all(&root).expect("create isolated directory");
        Self { root }
    }

    fn root(&self) -> &Path {
        &self.root
    }
}

impl Drop for Isolated {
    fn drop(&mut self) {
        if let Err(error) = fs::remove_dir_all(&self.root) {
            eprintln!("isolated cleanup failed for {:?}: {error}", self.root);
        }
    }
}

fn run(isolated: &Isolated, capital: Option<&str>) -> Output {
    let mut command = Command::new(env!("CARGO_BIN_EXE_continuous_evolution_backtest"));
    command
        .current_dir(isolated.root())
        .env("SIM_DAYS", "1")
        .env("MULTI_COIN", "0")
        .env("TG_GENOME_ENV", "backtest")
        // Plan §F13: la configuración que el proceso pudiera tocar queda
        // contenida dentro del fixture — nada escapa al sandbox.
        .env("TGM_CONFIG_DIR", isolated.root().join("tgm_config"))
        .env("TG_CONFIG_ROOT", isolated.root().join("tgm_config"));
    match capital {
        Some(value) => {
            command.env("INITIAL_CAPITAL", value);
        }
        None => {
            command.env_remove("INITIAL_CAPITAL");
        }
    }
    command.output().expect("run continuous_evolution_backtest")
}

fn assert_invalid(capital: Option<&str>) {
    let isolated = Isolated::new();
    let output = run(&isolated, capital);
    assert!(
        !output.status.success(),
        "INITIAL_CAPITAL={capital:?} must be rejected before side effects: {output:?}"
    );
    assert_eq!(
        fs::read(isolated.root().join("equity_curve.csv")).expect("read sentinel"),
        SENTINEL,
        "INITIAL_CAPITAL={capital:?} must preserve the sentinel"
    );
    assert!(
        !isolated.root().join("config_dir").exists(),
        "INITIAL_CAPITAL={capital:?} must not create config_dir"
    );
    let diagnostics = format!(
        "{}{}",
        String::from_utf8_lossy(&output.stdout),
        String::from_utf8_lossy(&output.stderr)
    );
    assert!(
        diagnostics.contains("INITIAL_CAPITAL"),
        "missing explicit INITIAL_CAPITAL diagnostic: {diagnostics}"
    );
}

#[test]
fn invalid_initial_capital_is_rejected_before_side_effects() {
    for capital in [None, Some(""), Some("abc"), Some("NaN"), Some("inf"), Some("-inf"), Some("0"), Some("-1")] {
        assert_invalid(capital);
    }
}

#[test]
fn invalid_initial_capital_creates_nothing_without_preexisting_csv() {
    // Variante del plan: sin equity_curve.csv previo, el rechazo tampoco
    // deja artefactos (ni csv nuevo ni config_dir).
    let isolated = Isolated::sin_csv();
    let output = run(&isolated, None);
    assert!(
        !output.status.success(),
        "INITIAL_CAPITAL ausente must fail: {output:?}"
    );
    assert!(
        !isolated.root().join("equity_curve.csv").exists(),
        "rechazo no debe crear equity_curve.csv"
    );
    assert!(
        !isolated.root().join("config_dir").exists(),
        "rechazo no debe crear config_dir"
    );
}

#[test]
fn valid_initial_capital_reaches_the_missing_tape() {
    let isolated = Isolated::new();
    let output = run(&isolated, Some("37.0"));
    let diagnostics = format!(
        "{}{}",
        String::from_utf8_lossy(&output.stdout),
        String::from_utf8_lossy(&output.stderr)
    );
    assert!(
        diagnostics.contains("BTCUSDT_ticks.bin"),
        "valid INITIAL_CAPITAL must reach tape validation, not config rejection: {diagnostics}"
    );
}
