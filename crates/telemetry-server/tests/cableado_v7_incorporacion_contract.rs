//! CABLEADOV7 (pedido CABLEADO HOST v7, §3/§4 y R1): el lector real, su binario,
//! el banco de contadores, la declaracion del host y el fixture del lector viven
//! en el arbol versionado y se compilan con el crate. Antes solo existian sin
//! seguimiento en el arbol vivo.

const LIB: &str = include_str!("../src/lib.rs");

#[test]
fn lector_real_y_contadores_incorporados_al_crate() {
    let lib = LIB.replace("\r\n", "\n");
    assert!(lib.contains("pub mod demo_counters;"), "lib.rs no declara demo_counters");
    assert!(lib.contains("pub mod report_demo_real;"), "lib.rs no declara report_demo_real");
    let raiz = std::path::Path::new(env!("CARGO_MANIFEST_DIR"));
    for r in [
        "src/demo_counters.rs",
        "src/report_demo_real.rs",
        "src/bin/demo_report_real.rs",
        "HOST_WIRING_PENDING.md",
        "tests/fixtures/corrida_demo_20261112_094131_recorte.log",
    ] {
        assert!(raiz.join(r).is_file(), "falta {r}");
    }
    // Comentario corregido al incorporar: executor.rs:1008 es `Err(e) => Err(e)`, no el
    // retorno del aplanado.
    let contadores = std::fs::read_to_string(raiz.join("src/demo_counters.rs")).unwrap();
    assert!(
        !contadores.contains("executor.rs:1008, retorno de flatten_all_positions"),
        "comentario historico de IMMUNE_FLATTEN sin corregir"
    );
}
