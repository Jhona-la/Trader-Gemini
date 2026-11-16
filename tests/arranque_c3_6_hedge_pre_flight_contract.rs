//! ARRANQUE ORDENADO C3-6 (pedido `junie_cli_pedido_arranque_ordenado_v4.md` con la revision
//! `qoder_cli_revision_pedido_arranque_v4_junie.md`): el pre-flight F1.11 del host solo llama a
//! `ensure_hedge_mode` (POST firmado `dualSidePosition=true`, muta la cuenta) cuando el venue es
//! Testnet. La ruta OMS (`OmsSession::refresh`, `host_oms.rs`, fallo `OMS_MODE_UNCONFIRMED`) no se
//! toca y no es objeto de esta prueba.
//!
//! Ancla de fuente (texto con CRLF normalizado), no ejecucion: la guardia vive en linea en el
//! binario `god_engine.rs`. Se cuenta la FORMA DE LLAMADA, no el simbolo suelto: el simbolo
//! tambien aparece en el comentario historico del fix B1.2 (D-700), que no se toca.

const HOST: &str = include_str!("../src/bin/god_engine.rs");
const LLAMADA: &str = ".ensure_hedge_mode(";
const INICIO_PRE_FLIGHT: &str = "// ── F1.11 MODO HEDGE";
const FIN_PRE_FLIGHT: &str = "// ── F1.7 RECONCILIACIÓN AL ARRANQUE";

fn es_palabra(c: char) -> bool {
    c.is_alphanumeric() || c == '_'
}

/// `is_testnet` como identificador completo (no `is_testnet_cfg`, no `is_env_testnet`).
fn contiene_identificador(linea: &str, ident: &str) -> Option<usize> {
    let mut desde = 0;
    while let Some(i) = linea[desde..].find(ident) {
        let ini = desde + i;
        let fin = ini + ident.len();
        let antes = linea[..ini].chars().next_back();
        let despues = linea[fin..].chars().next();
        if !antes.map_or(false, es_palabra) && !despues.map_or(false, es_palabra) {
            return Some(ini);
        }
        desde = fin;
    }
    None
}

#[test]
fn guardia_hedge_pre_flight_incluye_is_testnet() {
    let host = HOST.replace("\r\n", "\n");

    // 1. Forma de llamada unica en el host: la del bloque pre-flight.
    let llamadas: Vec<usize> = host.match_indices(LLAMADA).map(|(i, _)| i).collect();
    assert_eq!(
        llamadas.len(),
        1,
        "god_engine.rs debe tener exactamente una llamada `{LLAMADA}` (la del pre-flight F1.11)"
    );
    let pos = llamadas[0];
    let ini = host
        .find(INICIO_PRE_FLIGHT)
        .expect("ancla ausente: inicio del bloque F1.11");
    let fin = host
        .find(FIN_PRE_FLIGHT)
        .expect("ancla ausente: bloque F1.7 que sigue al pre-flight");
    assert!(
        ini < pos && pos < fin,
        "la llamada debe vivir dentro del bloque pre-flight F1.11"
    );

    // 2. La guardia que abre ese bloque es la linea `if` que precede a la llamada.
    let bloque = &host[ini..pos];
    let guardia = bloque
        .lines()
        .map(str::trim)
        .filter(|l| l.starts_with("if "))
        .last()
        .expect("el bloque pre-flight debe abrir con una guardia `if`");
    let i = contiene_identificador(guardia, "is_testnet")
        .unwrap_or_else(|| panic!("la guardia del pre-flight no incluye `is_testnet`: {guardia}"));
    assert!(
        !guardia[..i].trim_end().ends_with('!'),
        "`is_testnet` no puede ir negado en la guardia: {guardia}"
    );
    assert!(
        guardia.contains("&&") && !guardia.contains("||"),
        "`is_testnet` debe ser condicion necesaria (conjuncion), no alternativa: {guardia}"
    );
    assert!(
        guardia.contains("!exec.load().is_paper_trading()"),
        "la guardia conserva la exclusion de papel: {guardia}"
    );
}
