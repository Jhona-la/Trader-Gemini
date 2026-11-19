//! DEMO-A (pedido «Demo en Testnet sin ambiguedad»; `codex_app_demo_especificacion_pruebas.md`
//! §A.3, casos A1-A10; revision N1 de NVIDIA, cambios 1-3 y 6): CONTRATO DE ARRANQUE.
//!
//! - A1-A5: el nucleo puro `arranque_entorno::validar` con entorno, argumentos y marcadores
//!   inyectados (llaves CANARIO ficticias). Cada negativo compara el campo y el motivo exactos.
//! - A1-A3 (proceso): el binario real `god_engine` lanzado con entorno limpio en una carpeta
//!   temporal vacia. Solo casos NEGATIVOS, cada uno con un ultimo campo invalido de respaldo
//!   (`BEST_WS_ENDPOINT` vacio) para que ni una mutacion del nucleo lo deje arrancar: una sola
//!   linea `ENTORNO_RECHAZADO` en stderr, codigo 2, stdout vacio, ningun archivo creado y ningun
//!   canario en la salida. Ninguna conexion: el proceso sale antes de cualquier transporte.
//! - A6: recibo coherente de Demo (REST, WS privado y feed del mismo venue) y papel sin llaves.
//! - A7: cliente del perfil sellado sin respaldo (cabecera invalida = error) y sin seguir 3xx
//!   (dos servidores locales en 127.0.0.1: el segundo, destino de la redireccion, no recibe nada).
//! - A8-A10 y HEDGE: anclas de fuente del cableado del binario (texto con CRLF normalizado):
//!   validacion antes de todo efecto, relevo, reconexion, OI y demonio de simbolos con el
//!   contexto sellado. Son anclas, no ejecucion.
//!
//! Sin red externa, sin `data\`, sin `.env`, sin llaves reales. Ningun host de Binance escrito
//! aqui (escaner P5): los hosts salen de las constantes del codigo.

use execution_engine::client::{BinanceClient, FalloConstruccionCliente, BINANCE_BASE_URL, BINANCE_TESTNET_URL};
use execution_engine::user_data_stream::{
    HOST_FEED_MAINNET_USDM, HOST_FEED_TESTNET_USDM, WS_PRIVADO_MAINNET, WS_PRIVADO_TESTNET,
};
use quantum_engine::arranque_entorno::{
    self as ae, ContextoSellado, Instantanea, Marcadores, Motivo, Perfil, Rechazo,
};
use std::collections::HashMap;

const CANARIO_LLAVE: &str = "CANARIO-LLAVE-7f3a";
const CANARIO_SECRETO: &str = "CANARIO-SECRETO-91bd";
const CANARIO_PROD: &str = "CANARIO-PROD-4c2e";

const HOST: &str = include_str!("../src/bin/god_engine.rs");
const SIMBOLOS: &str = include_str!("../src/symbol_manager.rs");

fn host() -> String {
    HOST.replace("\r\n", "\n")
}

fn produccion(texto: &str) -> String {
    texto.replace("\r\n", "\n").split("#[cfg(test)]").next().unwrap_or("").to_string()
}

fn args(v: &[&str]) -> Vec<String> {
    v.iter().map(|s| s.to_string()).collect()
}

fn entorno_demo() -> HashMap<String, String> {
    let mut e = HashMap::new();
    e.insert("USE_TESTNET".into(), "true".into());
    e.insert("TESTNET_API_KEY".into(), CANARIO_LLAVE.into());
    e.insert("TESTNET_SECRET_KEY".into(), CANARIO_SECRETO.into());
    e
}

fn entorno_mainnet() -> HashMap<String, String> {
    let mut e = HashMap::new();
    e.insert("USE_TESTNET".into(), "false".into());
    e.insert("MAINNET_API_KEY".into(), CANARIO_LLAVE.into());
    e.insert("MAINNET_SECRET_KEY".into(), CANARIO_SECRETO.into());
    e
}

const ARMADO: Marcadores = Marcadores { armado_canonico: true, armado_cwd: false, parada: false };

fn validar(a: &[&str], e: &HashMap<String, String>, m: Marcadores) -> Result<ContextoSellado, Rechazo> {
    let a = args(a);
    let lector = |k: &str| e.get(k).cloned();
    ae::validar(&Instantanea { args: &a, entorno: &lector, marcadores: m })
}

fn rechazo(r: Result<ContextoSellado, Rechazo>) -> (&'static str, Motivo) {
    match r {
        Ok(ctx) => panic!("se esperaba rechazo y el arranque siguio: {ctx:?}"),
        Err(r) => (r.campo, r.motivo),
    }
}

const DEMO: [&str; 2] = ["--execution-profile", "demo-testnet"];

// ------------------------------------------------------------------ A1 ----

#[test]
fn a1_perfil_omitido_duplicado_incompleto_o_desconocido() {
    let e = entorno_demo();
    let m = Marcadores::default();
    assert_eq!(rechazo(validar(&[], &e, m)), ("execution-profile", Motivo::Ausente));
    assert_eq!(
        rechazo(validar(&["--execution-profile", "demo-testnet", "--execution-profile", "demo-testnet"], &e, m)),
        ("execution-profile", Motivo::Duplicado),
        "dos flags iguales tambien fallan"
    );
    assert_eq!(
        rechazo(validar(&["--execution-profile=demo-testnet", "--execution-profile", "paper-testnet"], &e, m)),
        ("execution-profile", Motivo::Duplicado)
    );
    assert_eq!(rechazo(validar(&["--execution-profile"], &e, m)), ("execution-profile", Motivo::Incompleto));
    assert_eq!(
        rechazo(validar(&["--execution-profile", "--force-live"], &e, m)),
        ("execution-profile", Motivo::Incompleto)
    );
    assert_eq!(rechazo(validar(&["--execution-profile="], &e, m)), ("execution-profile", Motivo::Incompleto));
    for desconocido in ["demo", "DEMO-TESTNET", "testnet", "live", "demo-testnet "] {
        assert_eq!(
            rechazo(validar(&["--execution-profile", desconocido], &e, m)),
            ("execution-profile", Motivo::Desconocido),
            "{desconocido:?}"
        );
    }
    assert_eq!(
        rechazo(validar(&["--execution-profile", "demo-testnet", "--modo", "x"], &e, m)),
        ("argumento", Motivo::Desconocido),
        "ningun alias de modo"
    );
    // Control positivo: el mismo entorno con el perfil una vez arranca.
    let ctx = validar(&DEMO, &e, m).expect("demo-testnet positivo");
    assert_eq!(ctx.perfil(), Perfil::DemoTestnet);
    let ctx = validar(&["--execution-profile=demo-testnet"], &e, m).expect("forma con =");
    assert_eq!(ctx.perfil(), Perfil::DemoTestnet);
}

// ------------------------------------------------------------------ A2 ----

#[test]
fn a2_use_testnet_ausente_invalido_o_en_conflicto() {
    let m = Marcadores::default();
    let mut e = entorno_demo();
    e.remove("USE_TESTNET");
    assert_eq!(rechazo(validar(&DEMO, &e, m)), ("USE_TESTNET", Motivo::Ausente));
    for invalido in ["TRUE", "True", "", "1", " true", "true ", "yes", "0"] {
        let mut e = entorno_demo();
        e.insert("USE_TESTNET".into(), invalido.into());
        assert_eq!(rechazo(validar(&DEMO, &e, m)), ("USE_TESTNET", Motivo::Invalido), "{invalido:?}");
    }
    let mut e = entorno_demo();
    e.insert("USE_TESTNET".into(), "false".into());
    assert_eq!(rechazo(validar(&DEMO, &e, m)), ("USE_TESTNET", Motivo::Conflicto), "falso en Demo");
    let mut e = entorno_mainnet();
    e.insert("USE_TESTNET".into(), "true".into());
    assert_eq!(
        rechazo(validar(&["--execution-profile", "mainnet-live", "--force-live"], &e, ARMADO)),
        ("USE_TESTNET", Motivo::Conflicto),
        "verdadero en produccion"
    );
    for (perfil, literal) in [("paper-testnet", "false"), ("paper-mainnet", "true")] {
        let mut e = HashMap::new();
        e.insert("USE_TESTNET".to_string(), literal.to_string());
        assert_eq!(
            rechazo(validar(&["--execution-profile", perfil], &e, m)),
            ("USE_TESTNET", Motivo::Conflicto),
            "{perfil}"
        );
    }
    // Positivos: cada perfil con su literal.
    for (perfil, literal) in [("paper-testnet", "true"), ("paper-mainnet", "false")] {
        let mut e = HashMap::new();
        e.insert("USE_TESTNET".to_string(), literal.to_string());
        assert!(validar(&["--execution-profile", perfil], &e, m).is_ok(), "{perfil}");
    }
    assert!(validar(&DEMO, &entorno_demo(), m).is_ok());
}

// ------------------------------------------------------------------ A3 ----

#[test]
fn a3_parejas_de_llaves_incompleta_cruzada_ambas_vacia_o_de_produccion() {
    let m = Marcadores::default();
    let base = || {
        let mut e = HashMap::new();
        e.insert("USE_TESTNET".to_string(), "true".to_string());
        e
    };
    let con = |pares: &[(&str, &str)]| {
        let mut e = base();
        for (k, v) in pares {
            e.insert(k.to_string(), v.to_string());
        }
        e
    };
    assert_eq!(rechazo(validar(&DEMO, &base(), m)), ("TESTNET_API_KEY", Motivo::Ausente));
    assert_eq!(
        rechazo(validar(&DEMO, &con(&[("TESTNET_API_KEY", CANARIO_LLAVE)]), m)),
        ("TESTNET_SECRET_KEY", Motivo::ParejaIncompleta)
    );
    assert_eq!(
        rechazo(validar(&DEMO, &con(&[("BINANCE_TESTNET_SECRET_KEY", CANARIO_SECRETO)]), m)),
        ("BINANCE_TESTNET_API_KEY", Motivo::ParejaIncompleta)
    );
    assert_eq!(
        rechazo(validar(
            &DEMO,
            &con(&[("TESTNET_API_KEY", CANARIO_LLAVE), ("BINANCE_TESTNET_SECRET_KEY", CANARIO_SECRETO)]),
            m
        )),
        ("TESTNET_API_KEY", Motivo::ParejaCruzada)
    );
    assert_eq!(
        rechazo(validar(
            &DEMO,
            &con(&[
                ("TESTNET_API_KEY", CANARIO_LLAVE),
                ("TESTNET_SECRET_KEY", CANARIO_SECRETO),
                ("BINANCE_TESTNET_API_KEY", CANARIO_LLAVE),
                ("BINANCE_TESTNET_SECRET_KEY", CANARIO_SECRETO),
            ]),
            m
        )),
        ("TESTNET_API_KEY", Motivo::AmbasParejas)
    );
    assert_eq!(
        rechazo(validar(&DEMO, &con(&[("TESTNET_API_KEY", ""), ("TESTNET_SECRET_KEY", CANARIO_SECRETO)]), m)),
        ("TESTNET_API_KEY", Motivo::Vacio)
    );
    assert_eq!(
        rechazo(validar(&DEMO, &con(&[("TESTNET_API_KEY", CANARIO_LLAVE), ("TESTNET_SECRET_KEY", "  ")]), m)),
        ("TESTNET_SECRET_KEY", Motivo::Vacio)
    );
    // Produccion presente en Demo (cualquiera de los cuatro nombres, aun vacio o "shadow").
    for nombre in ["MAINNET_API_KEY", "MAINNET_SECRET_KEY", "BINANCE_API_KEY", "BINANCE_SECRET_KEY"] {
        let mut e = entorno_demo();
        e.insert(nombre.to_string(), CANARIO_PROD.to_string());
        let r = validar(&DEMO, &e, m).expect_err("produccion en Demo");
        assert_eq!((r.campo, r.motivo), (nombre, Motivo::FamiliaAjena));
        let linea = r.linea();
        for canario in [CANARIO_LLAVE, CANARIO_SECRETO, CANARIO_PROD] {
            assert!(!linea.contains(canario), "redaccion: {linea}");
        }
        assert_eq!(linea, format!("ENTORNO_RECHAZADO campo={nombre} motivo=FAMILIA_AJENA perfil=demo-testnet"));
    }
    // Testnet presente en produccion.
    let mut e = entorno_mainnet();
    e.insert("BINANCE_TESTNET_API_KEY".into(), CANARIO_PROD.into());
    assert_eq!(
        rechazo(validar(&["--execution-profile", "mainnet-live", "--force-live"], &e, ARMADO)),
        ("BINANCE_TESTNET_API_KEY", Motivo::FamiliaAjena)
    );
    // Positivos: cada pareja de cada familia por separado.
    for (k, s) in ae::PAREJAS_TESTNET {
        let ctx = validar(&DEMO, &con(&[(k, CANARIO_LLAVE), (s, CANARIO_SECRETO)]), m).expect("pareja Testnet");
        assert_eq!(ctx.llave(), CANARIO_LLAVE);
        assert_eq!(ctx.secreto(), CANARIO_SECRETO);
        assert_eq!(ctx.credenciales().map(|c| c.nombre_llave), Some(k));
        let depurado = format!("{ctx:?}");
        assert!(!depurado.contains(CANARIO_LLAVE) && !depurado.contains(CANARIO_SECRETO), "{depurado}");
        assert!(!ctx.recibo().contains(CANARIO_LLAVE) && !ctx.recibo().contains(CANARIO_SECRETO));
    }
    for (k, s) in ae::PAREJAS_PRODUCCION {
        let mut e = HashMap::new();
        e.insert("USE_TESTNET".to_string(), "false".to_string());
        e.insert(k.to_string(), CANARIO_LLAVE.to_string());
        e.insert(s.to_string(), CANARIO_SECRETO.to_string());
        let ctx = validar(&["--execution-profile", "mainnet-live", "--force-live"], &e, ARMADO)
            .expect("pareja de produccion (doble offline)");
        assert_eq!(ctx.credenciales().map(|c| c.nombre_llave), Some(k));
    }
}

// ------------------------------------------------------------------ A4 ----

#[test]
fn a4_force_live_armado_y_parada() {
    let e = entorno_demo();
    let solo_cwd = Marcadores { armado_canonico: false, armado_cwd: true, parada: false };
    let parada = Marcadores { parada: true, ..Marcadores::default() };
    assert_eq!(
        rechazo(validar(&["--execution-profile", "demo-testnet", "--force-live"], &e, Marcadores::default())),
        ("force-live", Motivo::Conflicto)
    );
    assert_eq!(rechazo(validar(&DEMO, &e, ARMADO)), ("MAINNET_ARMED", Motivo::Conflicto));
    assert_eq!(rechazo(validar(&DEMO, &e, solo_cwd)), ("MAINNET_ARMED", Motivo::Conflicto));
    assert_eq!(rechazo(validar(&DEMO, &e, parada)), ("STOP_TRADING.LOCK", Motivo::Conflicto));
    let em = entorno_mainnet();
    let live = ["--execution-profile", "mainnet-live", "--force-live"];
    assert_eq!(rechazo(validar(&live, &em, Marcadores::default())), ("MAINNET_ARMED", Motivo::Ausente));
    assert_eq!(rechazo(validar(&live, &em, solo_cwd)), ("MAINNET_ARMED", Motivo::Conflicto), "desacuerdo con el CWD");
    assert_eq!(
        rechazo(validar(&live, &em, Marcadores { parada: true, ..ARMADO })),
        ("STOP_TRADING.LOCK", Motivo::Conflicto)
    );
    assert_eq!(
        rechazo(validar(&["--execution-profile", "mainnet-live"], &em, ARMADO)),
        ("force-live", Motivo::Ausente)
    );
    assert_eq!(
        rechazo(validar(&["--execution-profile", "mainnet-live", "--force-live", "--force-live"], &em, ARMADO)),
        ("force-live", Motivo::Duplicado)
    );
    for flag in ae::FLAGS_AUXILIARES {
        let mut em2 = em.clone();
        em2.insert(flag.to_string(), "true".to_string());
        assert_eq!(rechazo(validar(&live, &em2, ARMADO)), (flag, Motivo::Conflicto), "{flag} en produccion");
        let mut ed = entorno_demo();
        ed.insert(flag.to_string(), "TRUE".to_string());
        assert_eq!(rechazo(validar(&DEMO, &ed, Marcadores::default())), (flag, Motivo::Invalido));
    }
    let mut ed = entorno_demo();
    ed.insert(ae::VARIABLE_IPS_FEED.to_string(), "127.0.0.1:443".to_string());
    assert_eq!(rechazo(validar(&DEMO, &ed, Marcadores::default())), (ae::VARIABLE_IPS_FEED, Motivo::Conflicto));
    // Positivo de produccion solo con dobles offline (nunca se construye nada aqui).
    let ctx = validar(&live, &em, ARMADO).expect("mainnet-live armado");
    assert!(!ctx.modo_demo() && !ctx.testnet() && !ctx.papel());
}

// ------------------------------------------------------------------ A5 ----

#[test]
fn a5_override_del_feed_ausente_literal_o_invalido() {
    let m = Marcadores::default();
    let ctx = validar(&DEMO, &entorno_demo(), m).expect("sin override");
    assert_eq!(ctx.host_feed(), HOST_FEED_TESTNET_USDM);
    let mut e = entorno_demo();
    e.insert("BEST_WS_ENDPOINT".into(), HOST_FEED_TESTNET_USDM.into());
    assert_eq!(validar(&DEMO, &e, m).expect("literal permitido").host_feed(), HOST_FEED_TESTNET_USDM);
    let invalidos = [
        (String::new(), Motivo::Vacio),
        (format!("wss://{HOST_FEED_TESTNET_USDM}"), Motivo::Invalido),
        (format!("{HOST_FEED_TESTNET_USDM}:443"), Motivo::Invalido),
        (format!("u@{HOST_FEED_TESTNET_USDM}"), Motivo::Invalido),
        (format!("{HOST_FEED_TESTNET_USDM}/ws"), Motivo::Invalido),
        (format!("{HOST_FEED_TESTNET_USDM}?x=1"), Motivo::Invalido),
        (format!("{HOST_FEED_TESTNET_USDM}#f"), Motivo::Invalido),
        (format!(" {HOST_FEED_TESTNET_USDM}"), Motivo::Invalido),
        // El host de Mainnet en Demo: en la base ganaba por sintaxis.
        (HOST_FEED_MAINNET_USDM.to_string(), Motivo::HostNoCanonico),
        (format!("x.{HOST_FEED_TESTNET_USDM}"), Motivo::HostNoCanonico),
    ];
    for (valor, motivo) in invalidos {
        let mut e = entorno_demo();
        e.insert("BEST_WS_ENDPOINT".into(), valor.clone());
        assert_eq!(rechazo(validar(&DEMO, &e, m)), ("BEST_WS_ENDPOINT", motivo), "{valor:?}");
    }
    let mut em = entorno_mainnet();
    em.insert("BEST_WS_ENDPOINT".into(), HOST_FEED_TESTNET_USDM.into());
    assert_eq!(
        rechazo(validar(&["--execution-profile", "mainnet-live", "--force-live"], &em, ARMADO)),
        ("BEST_WS_ENDPOINT", Motivo::HostNoCanonico)
    );
    let mut ep = HashMap::new();
    ep.insert("USE_TESTNET".to_string(), "false".to_string());
    assert_eq!(
        validar(&["--execution-profile", "paper-mainnet"], &ep, m).expect("papel mainnet").host_feed(),
        HOST_FEED_MAINNET_USDM
    );
}

// ------------------------------------------------------------------ A6 ----

#[test]
fn a6_recibo_de_demo_y_papel_sin_llaves() {
    let m = Marcadores::default();
    let ctx = validar(&DEMO, &entorno_demo(), m).expect("demo");
    assert!(ctx.es_demo_testnet() && ctx.testnet() && !ctx.papel() && ctx.modo_demo());
    assert_eq!(ctx.base_rest(), BINANCE_TESTNET_URL);
    assert_eq!(ctx.base_ws_privado(), WS_PRIVADO_TESTNET);
    // El cliente del mismo contexto entrega al transporte la misma base que el recibo.
    let cliente = BinanceClient::try_new(ctx.llave().to_string(), ctx.testnet()).expect("cliente");
    assert_eq!(cliente.get_base_url(), ctx.base_rest());
    let simbolos = vec!["btcusdt".to_string(), "ethusdt".to_string()];
    let urls = data_ingest::rutas_feed::urls_feed(ctx.host_feed(), &simbolos).expect("urls del feed");
    for u in [urls.publica.as_str(), urls.mercado.as_str()] {
        assert!(ctx.url_de_feed_admitida(u), "{u}");
    }
    let recibo = ctx.recibo();
    for parte in [
        "perfil=demo-testnet".to_string(),
        "venue=TESTNET_USDM".to_string(),
        "modo=DEMO".to_string(),
        "papel=false".to_string(),
        format!("rest_privado={BINANCE_TESTNET_URL}"),
        format!("ws_privado={WS_PRIVADO_TESTNET}"),
        format!("feed=wss://{HOST_FEED_TESTNET_USDM}"),
        "familia_llaves=TESTNET_API_KEY".to_string(),
    ] {
        assert!(recibo.contains(&parte), "{parte} en {recibo}");
    }
    // Papel: contexto privado vacio aunque haya llaves en el entorno; feed de su venue.
    for (perfil, testnet) in [("paper-testnet", true), ("paper-mainnet", false)] {
        let mut e = if testnet { entorno_demo() } else { entorno_mainnet() };
        e.insert("USE_TESTNET".into(), testnet.to_string());
        let p = validar(&["--execution-profile", perfil], &e, m).expect(perfil);
        assert!(p.papel() && p.modo_demo());
        assert!(p.credenciales().is_none() && p.llave().is_empty() && p.secreto().is_empty(), "{perfil}");
        assert_eq!(p.host_feed(), if testnet { HOST_FEED_TESTNET_USDM } else { HOST_FEED_MAINNET_USDM });
        assert!(p.recibo().contains("rest_privado=ninguno") && p.recibo().contains("familia_llaves=ninguna"));
    }
    let pm = validar(&["--execution-profile", "mainnet-live", "--force-live"], &entorno_mainnet(), ARMADO).expect("live");
    assert_eq!((pm.base_rest(), pm.base_ws_privado(), pm.host_feed()), (BINANCE_BASE_URL, WS_PRIVADO_MAINNET, HOST_FEED_MAINNET_USDM));
}

// ------------------------------------------------------------------ A7 ----

#[test]
fn a7_cliente_sin_respaldo_y_cabecera_invalida() {
    assert_eq!(
        BinanceClient::try_new("llave\ncon salto".into(), true).err(),
        Some(FalloConstruccionCliente::CabeceraLlave)
    );
    assert_eq!(
        execution_engine::executor::OrderExecutor::try_new("llave\u{7f}".into(), "s".into(), true).err(),
        Some(FalloConstruccionCliente::CabeceraLlave)
    );
    // Control: el constructor heredado sigue con su respaldo (cabecera vacia), fuera del perfil.
    let heredado = BinanceClient::new("llave\ncon salto".into(), true);
    assert_eq!(heredado.api_key.load().as_bytes(), b"");
    let bueno = BinanceClient::try_new(CANARIO_LLAVE.into(), true).expect("llave valida");
    assert_eq!(bueno.api_key.load().as_bytes(), CANARIO_LLAVE.as_bytes());
}

/// Un 3xx privado no produce una segunda solicitud: el destino de la redireccion (otro origen,
/// otro puerto local) no recibe ninguna conexion, y la primera ve la ruta del listenKey.
#[test]
fn a7_un_3xx_privado_no_se_sigue() {
    use std::io::{Read, Write};
    let destino = std::net::TcpListener::bind("127.0.0.1:0").expect("destino");
    destino.set_nonblocking(true).expect("no bloqueante");
    let puerto_destino = destino.local_addr().expect("addr").port();
    let origen = std::net::TcpListener::bind("127.0.0.1:0").expect("origen");
    let base = format!("http://{}", origen.local_addr().expect("addr"));
    let servidor = std::thread::spawn(move || {
        let (mut s, _) = origen.accept().expect("una solicitud");
        let mut buf = [0u8; 4096];
        let n = s.read(&mut buf).unwrap_or(0);
        let peticion = String::from_utf8_lossy(&buf[..n]).to_string();
        let resp = format!(
            "HTTP/1.1 302 Found\r\nLocation: http://127.0.0.1:{puerto_destino}/robado\r\nContent-Length: 0\r\nConnection: close\r\n\r\n"
        );
        let _ = s.write_all(resp.as_bytes());
        peticion
    });
    let cliente = BinanceClient::try_with_timeout(CANARIO_LLAVE.into(), true, std::time::Duration::from_secs(5))
        .expect("cliente")
        .con_base_de_prueba(&base);
    let rt = tokio::runtime::Builder::new_current_thread().enable_all().build().expect("runtime");
    let resultado = rt.block_on(cliente.create_listen_key());
    let peticion = servidor.join().expect("servidor");
    assert!(resultado.is_err(), "un 3xx no es un listenKey: {resultado:?}");
    assert!(peticion.starts_with("POST /fapi/v1/listenKey "), "ruta entregada al transporte: {peticion}");
    std::thread::sleep(std::time::Duration::from_millis(200));
    assert!(
        matches!(destino.accept(), Err(ref e) if e.kind() == std::io::ErrorKind::WouldBlock),
        "el destino de la redireccion no recibe ninguna conexion"
    );
}

// ---------------------------------------------- A8, A9, A10, HEDGE (anclas) ----

#[test]
fn a8_a10_hedge_la_validacion_precede_a_todo_efecto_del_arranque() {
    let h = host();
    let main = h.find("async fn main(").expect("main");
    let cuerpo = &h[main..];
    let validacion = cuerpo.find("arranque_entorno::validar_proceso()").expect("validacion en main");
    let salida = cuerpo.find("std::process::exit(quantum_engine::arranque_entorno::CODIGO_RECHAZO)").expect("codigo 2");
    assert!(validacion < salida);
    for efecto in [
        "init_telemetry_logger",
        "execute_phase_1_and_2(",
        "urls_feed(",
        "OrderExecutor::try_new(",
        "OmsSession::new(",
        "PuenteCuenta::con_sesion(",
        ".ensure_hedge_mode(",
        "execute_phase_3_warmup(",
    ] {
        let i = cuerpo.find(efecto).unwrap_or_else(|| panic!("ancla ausente: {efecto}"));
        assert!(salida < i, "{efecto} antes del rechazo del entorno");
    }
    let p = produccion(HOST);
    assert!(!p.contains("OrderExecutor::new(active_key"), "sin el constructor con respaldo en el arranque");
    for lectura in ["var(\"USE_TESTNET\")", "var(\"BEST_WS_ENDPOINT\")", "var(\"TESTNET_API_KEY\")", "var(\"BINANCE_API_KEY\")", "args.contains(", "EnvManager::is_demo_env()"] {
        assert!(!p.contains(lectura), "el binario no relee {lectura}: solo el contexto sellado");
    }
}

#[test]
fn a8_relevo_consume_el_contexto_sellado() {
    let h = host();
    let bloque = h.split("if newly_transitioned {").nth(2).expect("bloque de la transicion");
    let paso = bloque.find("oms_session.paso_transicion(").expect("paso");
    let publicada = bloque.find("ResultadoTransicion::Publicada").expect("publicada");
    let cierre = &bloque[paso..publicada];
    assert!(cierre.contains("let is_env_testnet = contexto_arranque.testnet();"), "venue sellado");
    assert!(cierre.contains("OrderExecutor::try_sucesor_de("), "sin respaldo en el relevo");
    assert!(cierre.contains("return previo;"), "fallo del cliente: se conserva el transporte acreditado");
    assert!(cierre.contains("if contexto_arranque.papel() {"), "el papel del perfil cruza el relevo");
    assert!(bloque.contains("let testnet_del_relevo = contexto_arranque.testnet();"), "rotulo de la cuenta sellado");
    // Sellado inmutable: una segunda instantanea no cambia la del proceso.
    let primero = validar(&DEMO, &entorno_demo(), Marcadores::default()).expect("demo");
    let mut otro = HashMap::new();
    otro.insert("USE_TESTNET".to_string(), "false".to_string());
    let segundo = validar(&["--execution-profile", "paper-mainnet"], &otro, Marcadores::default()).expect("papel");
    let antes = ae::sellado().cloned();
    let sello1 = ae::sellar(primero.clone());
    let sello2 = ae::sellar(segundo);
    assert!(!(sello1 && sello2), "solo una instantanea se sella");
    let vigente = ae::sellado().expect("sellado").clone();
    assert_eq!(Some(vigente.clone()), antes.or(Some(primero)), "la primera manda");
    // A10: las rutas auxiliares consumen el sellado, no el entorno.
    assert_eq!(ae::venue_testnet_auxiliar(), vigente.testnet());
}

#[test]
fn a8_sucesor_sellado_hereda_como_el_heredado_y_rechaza_cabecera_invalida() {
    use execution_engine::executor::{HerenciaDeCuota, OrderExecutor};
    let arena = quantum_arena::GlobalArena::build_in_own_stack(100.0);
    let previo = OrderExecutor::new(String::new(), "offline".into(), true);
    previo.arena.store(Some(std::sync::Arc::clone(&arena)));
    let (_, h_heredado) = OrderExecutor::sucesor_de(&previo, "k".into(), "s".into(), true, std::sync::Arc::clone(&arena));
    let (sucesor, h_sellado) =
        OrderExecutor::try_sucesor_de(&previo, "k".into(), "s".into(), true, std::sync::Arc::clone(&arena)).expect("sellado");
    assert!(matches!(h_heredado, HerenciaDeCuota::Compartida));
    assert!(matches!(h_sellado, HerenciaDeCuota::Compartida), "misma herencia que sucesor_de");
    assert_eq!(sucesor.client().get_base_url(), previo.client().get_base_url());
    assert_eq!(
        OrderExecutor::try_sucesor_de(&previo, "k\n".into(), "s".into(), true, arena).err(),
        Some(FalloConstruccionCliente::CabeceraLlave)
    );
}

#[test]
fn a9_reconexion_del_feed_solo_al_host_sellado() {
    let ctx = validar(&DEMO, &entorno_demo(), Marcadores::default()).expect("demo");
    let simbolos = vec!["btcusdt".to_string()];
    let canon = data_ingest::rutas_feed::urls_feed(HOST_FEED_TESTNET_USDM, &simbolos).expect("canonica");
    assert!(ctx.url_de_feed_admitida(&canon.publica));
    let ajena = data_ingest::rutas_feed::urls_feed(HOST_FEED_MAINNET_USDM, &simbolos).expect("ajena");
    for u in [
        ajena.publica.clone(),
        ajena.mercado.clone(),
        canon.publica.replacen("wss://", "ws://", 1),
        canon.publica.replacen(HOST_FEED_TESTNET_USDM, &format!("{HOST_FEED_TESTNET_USDM}:9443"), 1),
        canon.publica.replacen(HOST_FEED_TESTNET_USDM, &format!("u@{HOST_FEED_TESTNET_USDM}"), 1),
        format!("{}#x", canon.publica),
        "no es una url".to_string(),
    ] {
        assert!(!ctx.url_de_feed_admitida(&u), "{u}");
    }
    // Ancla: el lector valida la URL efectiva antes de resolver o conectar, en cada vuelta.
    let h = host();
    let lazo = h.find("let current_url = cfg.url.load().to_string();").expect("lazo del lector");
    let resto = &h[lazo..];
    let guarda = resto.find(".is_none_or(|ctx| ctx.url_de_feed_admitida(&current_url))").expect("guarda A9");
    let dns = resto.find("tokio::net::lookup_host(&host_port)").expect("dns");
    assert!(guarda < dns, "la guarda va antes de cualquier conexion");
    assert!(resto[..dns].contains("continue;"), "la URL indebida no se conecta");
    assert!(
        resto.contains("if quantum_engine::arranque_entorno::sellado().is_some() { Err(()) } else { Ok(v) }"),
        "sin IPs inyectadas con perfil sellado"
    );
}

#[test]
fn a10_oi_y_demonio_de_simbolos_con_el_venue_sellado() {
    let h = produccion(HOST);
    let oi = h.find("[OI-POLLER] open interest per-símbolo activo").expect("poller de OI");
    let inicio = h[..oi].rfind("rt_handle.spawn(async move {").expect("tarea del OI");
    let tramo = &h[inicio..oi];
    assert!(tramo.contains("let is_testnet = quantum_engine::arranque_entorno::venue_testnet_auxiliar();"));
    assert!(tramo.contains("execution_engine::client::BINANCE_TESTNET_URL"));
    assert!(tramo.contains(".redirect(reqwest::redirect::Policy::none())"));
    assert!(!tramo.contains("reqwest::Client::new()"), "sin cliente de respaldo");
    let s = produccion(SIMBOLOS);
    assert_eq!(s.matches("crate::arranque_entorno::venue_testnet_auxiliar()").count(), 2);
    assert!(!s.contains("EnvManager::is_demo_env()"));
    let lector = h.find("let is_testnet_cfg = quantum_engine::arranque_entorno::venue_testnet_auxiliar();");
    assert!(lector.is_some(), "el watchdog del lector usa el venue sellado");
}

// ------------------------------------------------------ A1-A3 (proceso) ----

/// El binario real con entorno limpio en una carpeta temporal vacia. Precondicion: ninguna
/// `.env` en la carpeta ni en sus ancestros (dotenvy las buscaria); si existe, la prueba falla
/// sin lanzar nada.
fn lanzar(argumentos: &[&str], entorno: &[(&str, &str)]) -> (Option<i32>, String, String, usize) {
    let dir = std::env::temp_dir().join(format!("tg_arranque_a_{}_{}", std::process::id(), nanos()));
    std::fs::create_dir_all(&dir).expect("carpeta temporal");
    let mut anc = Some(dir.as_path());
    while let Some(d) = anc {
        assert!(!d.join(".env").exists(), "precondicion: hay una .env en {} (no se lanza)", d.display());
        anc = d.parent();
    }
    let mut cmd = std::process::Command::new(env!("CARGO_BIN_EXE_god_engine"));
    cmd.env_clear().current_dir(&dir).args(argumentos);
    for k in ["SystemRoot", "SYSTEMROOT", "windir"] {
        if let Some(v) = std::env::var_os(k) {
            cmd.env(k, v);
        }
    }
    for (k, v) in entorno {
        cmd.env(k, v);
    }
    // Respaldo: un ultimo campo invalido para que ninguna mutacion deje arrancar el motor.
    cmd.env("BEST_WS_ENDPOINT", "");
    cmd.stdout(std::process::Stdio::piped()).stderr(std::process::Stdio::piped());
    let mut hijo = cmd.spawn().expect("lanzar god_engine");
    let limite = std::time::Instant::now() + std::time::Duration::from_secs(60);
    loop {
        if hijo.try_wait().expect("estado").is_some() {
            break;
        }
        if std::time::Instant::now() > limite {
            let _ = hijo.kill();
            panic!("god_engine siguio vivo: el rechazo no fue inmediato");
        }
        std::thread::sleep(std::time::Duration::from_millis(20));
    }
    let salida = hijo.wait_with_output().expect("salida");
    let creados = std::fs::read_dir(&dir).map(|r| r.count()).unwrap_or(0);
    let _ = std::fs::remove_dir(&dir);
    (
        salida.status.code(),
        String::from_utf8_lossy(&salida.stdout).to_string(),
        String::from_utf8_lossy(&salida.stderr).to_string(),
        creados,
    )
}

fn nanos() -> u128 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_nanos())
        .unwrap_or(0)
}

fn assert_rechazo_de_proceso(argumentos: &[&str], entorno: &[(&str, &str)], linea: &str) {
    let (codigo, stdout, stderr, creados) = lanzar(argumentos, entorno);
    assert_eq!(codigo, Some(2), "codigo de salida; stderr={stderr:?}");
    let lineas: Vec<&str> = stderr.lines().filter(|l| !l.trim().is_empty()).collect();
    assert_eq!(lineas, vec![linea], "una sola linea en stderr");
    assert!(stdout.trim().is_empty(), "nada en stdout (ni logger ni arranque): {stdout:?}");
    assert_eq!(creados, 0, "ningun archivo creado en la carpeta de trabajo");
    for canario in [CANARIO_LLAVE, CANARIO_SECRETO, CANARIO_PROD] {
        assert!(!stderr.contains(canario) && !stdout.contains(canario), "redaccion");
    }
}

#[test]
fn a1_proceso_sin_perfil_sale_con_codigo_2_sin_efectos() {
    assert_rechazo_de_proceso(
        &[],
        &[("USE_TESTNET", "true"), ("TESTNET_API_KEY", CANARIO_LLAVE), ("TESTNET_SECRET_KEY", CANARIO_SECRETO)],
        "ENTORNO_RECHAZADO campo=execution-profile motivo=AUSENTE",
    );
}

#[test]
fn a2_proceso_use_testnet_en_mayusculas_sale_con_codigo_2() {
    assert_rechazo_de_proceso(
        &DEMO,
        &[("USE_TESTNET", "TRUE"), ("TESTNET_API_KEY", CANARIO_LLAVE), ("TESTNET_SECRET_KEY", CANARIO_SECRETO)],
        "ENTORNO_RECHAZADO campo=USE_TESTNET motivo=INVALIDO perfil=demo-testnet",
    );
}

#[test]
fn a3_proceso_llave_de_produccion_en_demo_sale_con_codigo_2_y_redactada() {
    assert_rechazo_de_proceso(
        &DEMO,
        &[
            ("USE_TESTNET", "true"),
            ("TESTNET_API_KEY", CANARIO_LLAVE),
            ("TESTNET_SECRET_KEY", CANARIO_SECRETO),
            ("BINANCE_API_KEY", CANARIO_PROD),
        ],
        "ENTORNO_RECHAZADO campo=BINANCE_API_KEY motivo=FAMILIA_AJENA perfil=demo-testnet",
    );
}
