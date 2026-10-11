//! G0 / LC-F1-03 (Línea C, ciclo 11): el compilador local es el de la CI.
//!
//! `rust-toolchain.toml` decía `channel = "nightly"`, sin fecha: cada
//! máquina compilaba con el nightly del día en que lo instaló, mientras las
//! dos CI fijan `nightly-2026-06-30`. Un fallo local podía no reproducirse
//! en la CI, y al revés. Test de fuente: una sola cadena con fecha en
//! `rust-toolchain.toml`, `replay-contracts.yml` y `unit-suites.yml`.

const TOOLCHAIN: &str = include_str!("../../../rust-toolchain.toml");
const WORKFLOWS: [(&str, &str); 2] = [
    (
        "replay-contracts.yml",
        include_str!("../../../.github/workflows/replay-contracts.yml"),
    ),
    (
        "unit-suites.yml",
        include_str!("../../../.github/workflows/unit-suites.yml"),
    ),
];

/// `nightly-AAAA-MM-DD`.
fn es_nightly_con_fecha(canal: &str) -> bool {
    let Some(fecha) = canal.strip_prefix("nightly-") else {
        return false;
    };
    let b = fecha.as_bytes();
    b.len() == 10
        && b.iter()
            .enumerate()
            .all(|(i, c)| if i == 4 || i == 7 { *c == b'-' } else { c.is_ascii_digit() })
}

fn canal_local() -> String {
    let linea = TOOLCHAIN
        .lines()
        .map(str::trim)
        .find(|l| l.starts_with("channel"))
        .expect("rust-toolchain.toml sin `channel`");
    let (_, valor) = linea.split_once('=').expect("`channel = \"…\"`");
    valor.trim().trim_matches('"').to_string()
}

/// Toolchains que nombra un workflow: el `+X` de `cargo +X` y `rustc +X`, y
/// la `X` de `rustup toolchain install X`.
fn toolchains_del_workflow(yml: &str) -> Vec<String> {
    let mut out = Vec::new();
    for linea in yml.lines() {
        let t: Vec<&str> = linea.split_whitespace().collect();
        for i in 0..t.len() {
            if let Some(tc) = t[i].strip_prefix('+') {
                if i > 0 && (t[i - 1] == "cargo" || t[i - 1] == "rustc") {
                    out.push(tc.to_string());
                }
            }
            if t[i] == "install" && i >= 2 && t[i - 2] == "rustup" && t[i - 1] == "toolchain" {
                if let Some(tc) = t.get(i + 1) {
                    out.push(tc.to_string());
                }
            }
        }
    }
    out
}

#[test]
fn reconoce_solo_nightly_con_fecha() {
    assert!(es_nightly_con_fecha("nightly-2026-06-30"));
    for malo in ["nightly", "stable", "nightly-2026-6-30", "nightly-2026_06_30", "1.98.0"] {
        assert!(!es_nightly_con_fecha(malo), "{malo}");
    }
    let yml = "run: |\n  rustup toolchain install nightly-2026-06-30 --profile minimal\n  cargo +nightly-2026-06-30 test -p x\n  rustc +nightly --version\n";
    assert_eq!(
        toolchains_del_workflow(yml),
        ["nightly-2026-06-30", "nightly-2026-06-30", "nightly"]
    );
}

#[test]
fn toolchain_local_fijado_con_fecha() {
    let canal = canal_local();
    assert!(
        es_nightly_con_fecha(&canal),
        "rust-toolchain.toml: channel = {canal:?}; debe ser nightly-AAAA-MM-DD, el de la CI"
    );
}

#[test]
fn las_ci_compilan_con_el_toolchain_local() {
    let canal = canal_local();
    for (nombre, yml) in WORKFLOWS {
        let usados = toolchains_del_workflow(yml);
        assert!(!usados.is_empty(), "{nombre}: no nombra ningún toolchain");
        for tc in &usados {
            assert_eq!(tc, &canal, "{nombre} usa {tc} y rust-toolchain.toml {canal}");
        }
    }
}
