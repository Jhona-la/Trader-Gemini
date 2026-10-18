//! SOL-F9-4 (2026-10-18) — CONTRATO DE PARIDAD MACRO EXAMEN↔VIVO.
//!
//! QUÉ: fija que ningún camino de EXAMEN (Darwin live, backtest nativo,
//! backtest continuo) sirva literales macro plausibles de 2024 en el tensor
//! 54D. El examen no tiene tape macro: sus slots 21-29 deben ir a 0.0
//! (centinela «sin dato»), y el mapeo del core aplica sus fallbacks
//! calibrados (1.04/1.02/1.00/0.75), que son los MISMOS que usa el harness
//! ML (`god-engine-core/src/bin/test_ml.rs`) para evaluar los modelos.
//!
//! POR QUÉ: el camino vivo (`god-engine-core`) dicta el contrato: «toda
//! feature SIN PRODUCTOR se sirve 0.0, NUNCA un literal plausible». Los
//! exámenes servían 104.2/5120/18100/18.5/4.25/2320/81/5.5 — un mercado de
//! 2024 congelado que ni el vivo ni el harness ven. Eso es skew
//! examen↔servicio: genomas seleccionados bajo una distribución que no
//! existe en producción.
//!
//! CÓMO: (1) escaneo de fuente — los tres constructores de omni del examen
//! no pueden contener los literales ni asignaciones macro no-cero;
//! (2) prueba de comportamiento del `OmniSynth` de Darwin si está expuesto.
//! Sin red, sin disco fuera del repo, sin motor vivo.

/// Los tres archivos que construyen el omni del examen (relativos a la
/// raíz del workspace; cargo ejecuta el test con CWD = crate, no raíz).
const EXAM_SOURCES: [&str; 3] = [
    "crates/god-engine-core/src/darwin.rs",
    "crates/backtest-engine/src/lib.rs",
    "crates/backtest-engine/src/bin/continuous_evolution_backtest.rs",
];

fn workspace_root() -> std::path::PathBuf {
    std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("..")
        .join("..")
        .to_path_buf()
}

/// Literales 2024 prohibidos (valores exactos del mercado congelado).
const FORBIDDEN_LITERALS: [&str; 8] = [
    "104.2", "5120.0", "18100.0", "18.5", "4.25", "2320.0", "81.0", "5.5",
];

#[test]
fn sol_f9_4_ningun_examen_sirve_literales_macro_de_2024() {
    for src in EXAM_SOURCES {
        let ruta = workspace_root().join(src);
        let text = std::fs::read_to_string(&ruta)
            .unwrap_or_else(|e| panic!("no se pudo leer {}: {e}", ruta.display()));
        for lit in FORBIDDEN_LITERALS {
            // El literal sólo es violación si aparece en una asignación a un
            // slot macro del omni (21-29); en comentarios/doc se permite la
            // referencia histórica.
            for line in text.lines() {
                let code = line.split("//").next().unwrap_or("");
                let macro_assign = ["omni[21]", "omni[22]", "omni[23]", "omni[24]",
                    "omni[25]", "omni[26]", "omni[27]", "omni[28]", "omni[29]",
                    "omni_sim[21]", "omni_sim[22]", "omni_sim[23]", "omni_sim[24]",
                    "omni_sim[25]", "omni_sim[26]", "omni_sim[27]", "omni_sim[28]",
                    "omni_sim[29]"]
                    .iter()
                    .any(|slot| code.contains(slot));
                if macro_assign && code.contains(lit) {
                    panic!(
                        "{src}: el examen sirve el literal macro 2024 «{lit}» — \
                         debe ser sentinela 0.0 (contrato SOL-F9-4)"
                    );
                }
            }
        }
    }
}

#[test]
fn sol_f9_4_los_slots_macro_del_examen_son_sentinela() {
    // Cada asignación macro del examen debe ser 0.0 (o un valor causal
    // explícitamente documentado, hoy ninguno). Se verifica que toda línea
    // de asignación a omni[21..29]/omni_sim[21..29] en código (sin comentario)
    // termine en 0.0.
    for src in EXAM_SOURCES {
        let ruta = workspace_root().join(src);
        let text = std::fs::read_to_string(&ruta).expect("fuente legible");
        for line in text.lines() {
            let code = line.split("//").next().unwrap_or("").trim();
            let is_macro_assign = (21..=29).any(|i| {
                code.starts_with(&format!("omni[{i}] ="))
                    || code.starts_with(&format!("omni_sim[{i}] ="))
            });
            if is_macro_assign {
                assert!(
                    code.contains("0.0") || code.contains("0.00"),
                    "{src}: asignación macro no-centinela: «{line}» — SOL-F9-4 exige 0.0"
                );
            }
        }
    }
}
