//! RB-H2 (Codex RB, 2026-10-19) — FRESCURA POR CONTENIDO EN EL TARGET COMPARTIDO.
//!
//! Todos los agentes compilan en un único `target/` (scripts/cargo_semaforo.sh)
//! desde worktrees distintos del mismo repo. Cargo nombra los artefactos de los
//! paquetes del workspace con un hash de su ruta RELATIVA al workspace, así que
//! dos worktrees escriben los mismos archivos, y por defecto decide si una
//! unidad está fresca comparando el mtime de las fuentes con el del último
//! build. Si las fuentes de un worktree son más viejas que el último build
//! ajeno de esa unidad, cargo reutiliza el artefacto del OTRO worktree: el test
//! ejecuta código que no es el tuyo (falso verde o falso rojo, sin aviso). La
//! Línea C ya lo había sufrido en el oráculo y el medidor (CL-60b) y lo
//! parcheó tocando fuentes en esos dos scripts; los builds normales seguían
//! expuestos. La sonda `scripts/tests/test_frescura_target.py` lo reproduce en
//! el nightly fijado: por mtime el árbol 2 imprime «ARBOL-1»; con
//! `checksum-freshness`, «ARBOL-2».
//!
//! Este contrato exige la clave en `.cargo/config.toml` (frescura por el hash
//! del contenido) y que el toolchain fijado sea un nightly, el único que lee la
//! tabla `[unstable]`.

const CONFIG: &str = include_str!("../../../.cargo/config.toml");
const TOOLCHAIN: &str = include_str!("../../../rust-toolchain.toml");

/// Valor de `clave` dentro de la tabla `[tabla]` de un TOML plano (claves
/// `k = v` por línea; comentarios con `#` fuera de cadenas). Devuelve el valor
/// sin comillas ni blancos.
fn valor(toml: &str, tabla: &str, clave: &str) -> Option<String> {
    let cabecera = format!("[{tabla}]");
    let mut dentro = false;
    for linea in toml.lines() {
        let sin_comentario = match linea.find('#') {
            Some(i) if !linea[..i].contains('"') => &linea[..i],
            _ => linea,
        };
        let l = sin_comentario.trim();
        if l.starts_with('[') {
            dentro = l == cabecera;
            continue;
        }
        if !dentro {
            continue;
        }
        if let Some((k, v)) = l.split_once('=') {
            if k.trim() == clave {
                return Some(v.trim().trim_matches('"').to_string());
            }
        }
    }
    None
}

fn canal() -> Option<String> {
    TOOLCHAIN.lines().find_map(|l| {
        let (k, v) = l.split_once('=')?;
        (k.trim() == "channel").then(|| v.trim().trim_matches('"').to_string())
    })
}

#[test]
fn rb_h2_el_lector_toml_resuelve_tablas_y_comentarios() {
    let toml = "[build]\nrustflags = [\"-C\", \"opt-level=3\"]\n# checksum-freshness = false\n\n[unstable] # frescura\nchecksum-freshness = true # por contenido\n[otra]\nchecksum-freshness = false\n";
    assert_eq!(valor(toml, "unstable", "checksum-freshness").as_deref(), Some("true"));
    assert_eq!(valor(toml, "otra", "checksum-freshness").as_deref(), Some("false"));
    assert_eq!(valor(toml, "build", "checksum-freshness"), None, "la clave comentada no cuenta");
    assert_eq!(valor("[unstable]\n", "unstable", "checksum-freshness"), None);
}

#[test]
fn rb_h2_el_target_compartido_decide_la_frescura_por_contenido() {
    assert_eq!(
        valor(CONFIG, "unstable", "checksum-freshness").as_deref(),
        Some("true"),
        ".cargo/config.toml debe tener `[unstable] checksum-freshness = true`: sin ella, \
         con el target compartido, un worktree puede ejecutar el código de otro (RB-H2)"
    );
    let canal = canal().unwrap_or_default();
    assert!(
        canal.starts_with("nightly-"),
        "la tabla [unstable] sólo la lee un nightly; rust-toolchain.toml fija {canal:?}"
    );
}
