//! CONTRATO RB-C1 (Codex RB, 2026-11-10): la CI no esconde rojos.
//!
//! QUÉ: sin `--no-fail-fast`, `cargo test` se detiene en el PRIMER binario de
//! tests que falla. Desde W25 el shard `nucleo` de unit-suites cae en
//! `f69_002` y ningún binario de `god-engine-core` posterior en orden
//! alfabético llega a ejecutarse en CI: si otro test se rompe, nadie lo ve
//! hasta que se arregla el primero.
//! POR QUÉ: el mandato exige main en verde y saber exactamente qué falla; un
//! rojo no puede tapar a los demás. El coste es nulo en verde (se corren los
//! mismos binarios) y en rojo se paga una sola vez por todos los fallos.
//! CÓMO: toda invocación `cargo +<toolchain> test` de los dos workflows que
//! corren tests lleva `--no-fail-fast`; el detector se prueba con un control
//! negativo sintético.

const WORKFLOWS: [(&str, &str); 2] = [
    (
        "unit-suites.yml",
        include_str!("../../../.github/workflows/unit-suites.yml"),
    ),
    (
        "replay-contracts.yml",
        include_str!("../../../.github/workflows/replay-contracts.yml"),
    ),
];

/// Líneas de un workflow que invocan `cargo +<toolchain> test` (con o sin
/// `run:` delante) y NO llevan `--no-fail-fast`. Devuelve también cuántas
/// invocaciones vio, para que un cambio de sintaxis no deje el contrato vacío.
fn invocaciones_sin_flag(yml: &str) -> (usize, Vec<String>) {
    let mut vistas = 0;
    let mut sin_flag = Vec::new();
    for linea in yml.lines() {
        let t: Vec<&str> = linea.split_whitespace().collect();
        let invoca = t
            .windows(3)
            .any(|w| w[0] == "cargo" && w[1].starts_with('+') && w[2] == "test");
        if !invoca {
            continue;
        }
        vistas += 1;
        if !t.contains(&"--no-fail-fast") {
            sin_flag.push(linea.trim().to_string());
        }
    }
    (vistas, sin_flag)
}

#[test]
fn rb_c1_el_detector_caza_una_invocacion_sin_el_flag() {
    let yml = "steps:\n  - run: cargo +nightly-2026-06-30 test -p x --locked\n  - run: |\n      cargo +nightly-2026-06-30 test --no-fail-fast -p y\n      cargo +nightly-2026-06-30 check --workspace\n";
    let (vistas, sin_flag) = invocaciones_sin_flag(yml);
    assert_eq!(vistas, 2, "dos invocaciones de test; `check` no cuenta");
    assert_eq!(
        sin_flag,
        ["- run: cargo +nightly-2026-06-30 test -p x --locked"]
    );
}

#[test]
fn rb_c1_toda_invocacion_de_tests_en_ci_corre_sin_corte_temprano() {
    for (nombre, yml) in WORKFLOWS {
        let (vistas, sin_flag) = invocaciones_sin_flag(yml);
        assert!(
            vistas > 0,
            "{nombre}: el contrato no ve ninguna invocación de tests"
        );
        assert!(
            sin_flag.is_empty(),
            "{nombre}: {} invocaciones sin --no-fail-fast (el primer rojo ocultaría a los demás): {sin_flag:#?}",
            sin_flag.len()
        );
    }
}
