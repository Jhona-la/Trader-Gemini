//! DEMO-A (pedido del coordinador «Demo en Testnet sin ambiguedad»; especificacion
//! `codex_app_demo_especificacion_pruebas.md` §A.2 y revision N1 de NVIDIA, cambios 1-3 y 6):
//! CONTRATO DE ARRANQUE.
//!
//! El binario `god_engine` valida UNA vez, antes de construir cualquier cliente, socket, OMS,
//! puente de cuenta, feed o pre-flight hedge, una instantanea de sus argumentos, su entorno y los
//! marcadores del perfil. El nucleo de este modulo es puro: recibe la instantanea y devuelve un
//! [`ContextoSellado`] o un [`Rechazo`]. El adaptador del proceso ([`validar_proceso`]) solo lee
//! `std::env` y el disco (dos marcadores) y no abre ningun transporte.
//!
//! Rechazo = una linea `ENTORNO_RECHAZADO campo=<nombre> motivo=<enum>` en stderr y codigo 2.
//! La linea lleva solo el NOMBRE del campo y el motivo: nunca un valor (ni de llave ni de flag).
//!
//! Perfiles (`--execution-profile <valor>`, obligatorio y una sola vez):
//!
//! | perfil          | `USE_TESTNET` | modo      | ejecutor          | llaves                   |
//! | --------------- | ------------- | --------- | ----------------- | ------------------------ |
//! | `demo-testnet`  | `true`        | Demo      | Testnet, no papel | una pareja Testnet       |
//! | `mainnet-live`  | `false`       | Produccion| Mainnet, no papel | una pareja de produccion |
//! | `paper-testnet` | `true`        | Demo      | papel local       | ninguna (contexto vacio) |
//! | `paper-mainnet` | `false`       | Demo      | papel local       | ninguna (contexto vacio) |
//!
//! El contexto queda SELLADO en un `OnceLock` del proceso ([`sellar`] / [`sellado`]): relevo,
//! reconexion del feed, poller de OI y demonio de simbolos lo consumen en lugar de releer el
//! entorno mutable.

use std::fmt;
use std::sync::OnceLock;

/// Codigo de salida del rechazo del entorno.
pub const CODIGO_RECHAZO: i32 = 2;

/// Prefijo de la unica linea que el rechazo escribe en stderr.
pub const PREFIJO_RECHAZO: &str = "ENTORNO_RECHAZADO";

/// Argumento del perfil.
pub const ARG_PERFIL: &str = "--execution-profile";
/// Alias heredado de modo: solo vale (y es obligatorio, una vez) en `mainnet-live`.
pub const ARG_FORZAR_VIVO: &str = "--force-live";

/// Host publico canonico del feed de Testnet USD-M (el mismo que usaba la base).
pub const HOST_FEED_TESTNET: &str = execution_engine::user_data_stream::HOST_FEED_TESTNET_USDM;
/// Host publico canonico del feed de Mainnet USD-M en este contrato.
pub const HOST_FEED_MAINNET: &str = execution_engine::user_data_stream::HOST_FEED_MAINNET_USDM;
/// WS privado (user data stream) de Testnet y de Mainnet: las bases que usa
/// `execution_engine::user_data_stream` (recibo; el stream las deriva del mismo venue).
pub const WS_PRIVADO_TESTNET: &str = execution_engine::user_data_stream::WS_PRIVADO_TESTNET;
pub const WS_PRIVADO_MAINNET: &str = execution_engine::user_data_stream::WS_PRIVADO_MAINNET;

/// Parejas de nombres de llaves por familia (llave, secreto).
pub const PAREJAS_TESTNET: [(&str, &str); 2] = [
    ("TESTNET_API_KEY", "TESTNET_SECRET_KEY"),
    ("BINANCE_TESTNET_API_KEY", "BINANCE_TESTNET_SECRET_KEY"),
];
pub const PAREJAS_PRODUCCION: [(&str, &str); 2] = [
    ("MAINNET_API_KEY", "MAINNET_SECRET_KEY"),
    ("BINANCE_API_KEY", "BINANCE_SECRET_KEY"),
];

/// Flags heredados que cambian la raiz de datos o el venue de las rutas auxiliares
/// (`quantum_arena::paths::is_demo_env`).
pub const FLAGS_AUXILIARES: [&str; 2] = ["SHADOW_MODE", "BINANCE_USE_DEMO"];

/// IPs inyectadas a mano al lector del feed (`god_engine`, Happy Eyeballs): con perfil sellado
/// no se admiten.
pub const VARIABLE_IPS_FEED: &str = "BINANCE_WS_IPS";

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Perfil {
    DemoTestnet,
    MainnetLive,
    PaperTestnet,
    PaperMainnet,
}

impl Perfil {
    pub const TODOS: [Perfil; 4] = [
        Perfil::DemoTestnet,
        Perfil::MainnetLive,
        Perfil::PaperTestnet,
        Perfil::PaperMainnet,
    ];

    pub fn nombre(self) -> &'static str {
        match self {
            Perfil::DemoTestnet => "demo-testnet",
            Perfil::MainnetLive => "mainnet-live",
            Perfil::PaperTestnet => "paper-testnet",
            Perfil::PaperMainnet => "paper-mainnet",
        }
    }

    pub fn desde_nombre(valor: &str) -> Option<Perfil> {
        Perfil::TODOS.into_iter().find(|p| p.nombre() == valor)
    }

    /// Venue del perfil: Testnet USD-M o Mainnet USD-M.
    pub fn testnet(self) -> bool {
        matches!(self, Perfil::DemoTestnet | Perfil::PaperTestnet)
    }

    /// Papel local: sin transporte privado de trading.
    pub fn papel(self) -> bool {
        matches!(self, Perfil::PaperTestnet | Perfil::PaperMainnet)
    }

    /// Modo Demo del orquestador (todo salvo `mainnet-live`).
    pub fn modo_demo(self) -> bool {
        !matches!(self, Perfil::MainnetLive)
    }

    /// Valor literal que debe tener `USE_TESTNET`.
    pub fn use_testnet_literal(self) -> &'static str {
        if self.testnet() {
            "true"
        } else {
            "false"
        }
    }

    pub fn host_feed(self) -> &'static str {
        if self.testnet() {
            HOST_FEED_TESTNET
        } else {
            HOST_FEED_MAINNET
        }
    }
}

/// Motivo del rechazo (enum cerrado de la linea `ENTORNO_RECHAZADO`).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Motivo {
    /// El campo obligatorio no esta.
    Ausente,
    /// El argumento aparece mas de una vez.
    Duplicado,
    /// El argumento esta sin valor.
    Incompleto,
    /// Valor no reconocido (perfil o argumento).
    Desconocido,
    /// Valor fuera del literal exigido (`TRUE`, `1`, con espacio...).
    Invalido,
    /// Valor valido que contradice el perfil.
    Conflicto,
    /// Valor presente pero vacio.
    Vacio,
    /// Pareja de llaves con una sola mitad.
    ParejaIncompleta,
    /// Llave de una pareja y secreto de la otra.
    ParejaCruzada,
    /// Dos parejas completas de la misma familia.
    AmbasParejas,
    /// Llave de la otra familia en el proceso (produccion en Demo o Testnet en produccion).
    FamiliaAjena,
    /// Host del feed distinto del canonico del perfil.
    HostNoCanonico,
}

impl Motivo {
    pub fn nombre(self) -> &'static str {
        match self {
            Motivo::Ausente => "AUSENTE",
            Motivo::Duplicado => "DUPLICADO",
            Motivo::Incompleto => "INCOMPLETO",
            Motivo::Desconocido => "DESCONOCIDO",
            Motivo::Invalido => "INVALIDO",
            Motivo::Conflicto => "CONFLICTO",
            Motivo::Vacio => "VACIO",
            Motivo::ParejaIncompleta => "PAREJA_INCOMPLETA",
            Motivo::ParejaCruzada => "PAREJA_CRUZADA",
            Motivo::AmbasParejas => "AMBAS_PAREJAS",
            Motivo::FamiliaAjena => "FAMILIA_AJENA",
            Motivo::HostNoCanonico => "HOST_NO_CANONICO",
        }
    }
}

/// Rechazo del entorno: nombre del campo (nunca su valor) y motivo.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Rechazo {
    pub campo: &'static str,
    pub motivo: Motivo,
    /// Perfil reconocido (no sensible), si se llego a reconocer.
    pub perfil: Option<Perfil>,
}

impl Rechazo {
    fn de(campo: &'static str, motivo: Motivo, perfil: Option<Perfil>) -> Self {
        Self { campo, motivo, perfil }
    }

    /// La unica linea del rechazo en stderr.
    pub fn linea(&self) -> String {
        match self.perfil {
            Some(p) => format!(
                "{PREFIJO_RECHAZO} campo={} motivo={} perfil={}",
                self.campo,
                self.motivo.nombre(),
                p.nombre()
            ),
            None => format!("{PREFIJO_RECHAZO} campo={} motivo={}", self.campo, self.motivo.nombre()),
        }
    }
}

/// Marcadores del disco que la base consultaba (ruta canonica y CWD).
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct Marcadores {
    /// `config_dir/MAINNET_ARMED` en la ruta canonica (`paths::data_join`).
    pub armado_canonico: bool,
    /// `config_dir/MAINNET_ARMED` relativo al CWD.
    pub armado_cwd: bool,
    /// `STOP_TRADING.LOCK` (CWD o carpeta del ejecutable).
    pub parada: bool,
}

/// Familia de llaves elegida (nombre de la pareja, nunca el valor).
#[derive(Clone, PartialEq, Eq)]
pub struct Credenciales {
    pub nombre_llave: &'static str,
    pub nombre_secreto: &'static str,
    llave: String,
    secreto: String,
}

impl Credenciales {
    pub fn llave(&self) -> &str {
        &self.llave
    }
    pub fn secreto(&self) -> &str {
        &self.secreto
    }
}

impl fmt::Debug for Credenciales {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("Credenciales")
            .field("nombre_llave", &self.nombre_llave)
            .field("nombre_secreto", &self.nombre_secreto)
            .field("llave", &"<redactada>")
            .field("secreto", &"<redactado>")
            .finish()
    }
}

/// Contexto sellado del arranque: lo unico que el resto del proceso consulta sobre perfil,
/// venue, papel, llaves y feed.
#[derive(Clone, PartialEq, Eq)]
pub struct ContextoSellado {
    perfil: Perfil,
    credenciales: Option<Credenciales>,
    host_feed: &'static str,
}

impl fmt::Debug for ContextoSellado {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("ContextoSellado")
            .field("perfil", &self.perfil)
            .field("credenciales", &self.credenciales)
            .field("host_feed", &self.host_feed)
            .finish()
    }
}

impl ContextoSellado {
    pub fn perfil(&self) -> Perfil {
        self.perfil
    }
    pub fn testnet(&self) -> bool {
        self.perfil.testnet()
    }
    pub fn papel(&self) -> bool {
        self.perfil.papel()
    }
    pub fn modo_demo(&self) -> bool {
        self.perfil.modo_demo()
    }
    pub fn es_demo_testnet(&self) -> bool {
        self.perfil == Perfil::DemoTestnet
    }
    /// Credenciales de la familia del perfil (`None` en papel).
    pub fn credenciales(&self) -> Option<&Credenciales> {
        self.credenciales.as_ref()
    }
    /// Llave publica (vacia en papel).
    pub fn llave(&self) -> &str {
        self.credenciales.as_ref().map_or("", |c| c.llave())
    }
    /// Secreto (vacio en papel).
    pub fn secreto(&self) -> &str {
        self.credenciales.as_ref().map_or("", |c| c.secreto())
    }
    /// Host canonico del feed publico del perfil.
    pub fn host_feed(&self) -> &'static str {
        self.host_feed
    }
    /// Base del REST privado (la misma que deriva `BinanceClient::get_base_url`).
    pub fn base_rest(&self) -> &'static str {
        if self.testnet() {
            execution_engine::client::BINANCE_TESTNET_URL
        } else {
            execution_engine::client::BINANCE_BASE_URL
        }
    }
    /// Base del WS privado (user data stream).
    pub fn base_ws_privado(&self) -> &'static str {
        if self.testnet() {
            WS_PRIVADO_TESTNET
        } else {
            WS_PRIVADO_MAINNET
        }
    }

    /// A9: una URL del feed (inicial o de reconexion) solo se acepta si es `wss://` al host
    /// canonico del perfil, sin usuario, sin puerto explicito y sin fragmento.
    pub fn url_de_feed_admitida(&self, url: &str) -> bool {
        match url::Url::parse(url) {
            Ok(u) => {
                u.scheme() == "wss"
                    && u.host_str() == Some(self.host_feed)
                    && u.port().is_none()
                    && u.username().is_empty()
                    && u.password().is_none()
                    && u.fragment().is_none()
            }
            Err(_) => false,
        }
    }

    /// Recibo del arranque: perfil + venue privado + modo/papel + identidad (sin valores) + feed.
    /// La generacion OMS y la cuenta los agrega el host al publicarlo.
    pub fn recibo(&self) -> String {
        let familia = self
            .credenciales
            .as_ref()
            .map_or("ninguna", |c| c.nombre_llave);
        format!(
            "ARRANQUE_SELLADO perfil={} venue={} modo={} papel={} rest_privado={} ws_privado={} feed=wss://{} familia_llaves={}",
            self.perfil.nombre(),
            if self.testnet() { "TESTNET_USDM" } else { "MAINNET_USDM" },
            if self.modo_demo() { "DEMO" } else { "PRODUCCION" },
            self.papel(),
            if self.papel() { "ninguno" } else { self.base_rest() },
            if self.papel() { "ninguno" } else { self.base_ws_privado() },
            self.host_feed,
            familia
        )
    }
}

/// Instantanea de entrada del nucleo: argumentos (sin el ejecutable), lector del entorno y
/// marcadores.
pub struct Instantanea<'a> {
    pub args: &'a [String],
    pub entorno: &'a dyn Fn(&str) -> Option<String>,
    pub marcadores: Marcadores,
}

fn leer_perfil(args: &[String]) -> Result<Perfil, Rechazo> {
    let mut valores: Vec<Option<&str>> = Vec::new();
    let mut i = 0;
    while i < args.len() {
        let a = args[i].as_str();
        if a == ARG_PERFIL {
            let v = args.get(i + 1).map(String::as_str).filter(|v| !v.starts_with("--"));
            valores.push(v);
            i += if v.is_some() { 2 } else { 1 };
            continue;
        }
        if let Some(v) = a.strip_prefix(ARG_PERFIL).and_then(|r| r.strip_prefix('=')) {
            valores.push(Some(v));
        }
        i += 1;
    }
    match valores.as_slice() {
        [] => Err(Rechazo::de("execution-profile", Motivo::Ausente, None)),
        [_, _, ..] => Err(Rechazo::de("execution-profile", Motivo::Duplicado, None)),
        [None] | [Some("")] => Err(Rechazo::de("execution-profile", Motivo::Incompleto, None)),
        [Some(v)] => Perfil::desde_nombre(v)
            .ok_or_else(|| Rechazo::de("execution-profile", Motivo::Desconocido, None)),
    }
}

/// Todo argumento que no sea el perfil (con su valor) o `--force-live` es desconocido: ningun
/// alias puede contradecir el perfil.
fn argumentos_conocidos(args: &[String], perfil: Perfil) -> Result<(), Rechazo> {
    let mut i = 0;
    let mut forzar = 0usize;
    while i < args.len() {
        let a = args[i].as_str();
        if a == ARG_PERFIL {
            i += 2;
            continue;
        }
        if a.starts_with(&format!("{ARG_PERFIL}=")) {
            i += 1;
            continue;
        }
        if a == ARG_FORZAR_VIVO {
            forzar += 1;
            i += 1;
            continue;
        }
        return Err(Rechazo::de("argumento", Motivo::Desconocido, Some(perfil)));
    }
    match (perfil, forzar) {
        (Perfil::MainnetLive, 0) => Err(Rechazo::de("force-live", Motivo::Ausente, Some(perfil))),
        (Perfil::MainnetLive, 1) => Ok(()),
        (Perfil::MainnetLive, _) => Err(Rechazo::de("force-live", Motivo::Duplicado, Some(perfil))),
        (_, 0) => Ok(()),
        (_, _) => Err(Rechazo::de("force-live", Motivo::Conflicto, Some(perfil))),
    }
}

fn leer_use_testnet(perfil: Perfil, entorno: &dyn Fn(&str) -> Option<String>) -> Result<(), Rechazo> {
    match entorno("USE_TESTNET") {
        None => Err(Rechazo::de("USE_TESTNET", Motivo::Ausente, Some(perfil))),
        Some(v) if v != "true" && v != "false" => {
            Err(Rechazo::de("USE_TESTNET", Motivo::Invalido, Some(perfil)))
        }
        Some(v) if v != perfil.use_testnet_literal() => {
            Err(Rechazo::de("USE_TESTNET", Motivo::Conflicto, Some(perfil)))
        }
        Some(_) => Ok(()),
    }
}

fn leer_marcadores(perfil: Perfil, m: Marcadores) -> Result<(), Rechazo> {
    match perfil {
        Perfil::MainnetLive => {
            if m.parada {
                return Err(Rechazo::de("STOP_TRADING.LOCK", Motivo::Conflicto, Some(perfil)));
            }
            if !m.armado_canonico {
                // Armado solo en el CWD: desacuerdo con la ruta canonica.
                let motivo = if m.armado_cwd { Motivo::Conflicto } else { Motivo::Ausente };
                return Err(Rechazo::de("MAINNET_ARMED", motivo, Some(perfil)));
            }
            Ok(())
        }
        _ => {
            if m.armado_canonico || m.armado_cwd {
                return Err(Rechazo::de("MAINNET_ARMED", Motivo::Conflicto, Some(perfil)));
            }
            if perfil == Perfil::DemoTestnet && m.parada {
                return Err(Rechazo::de("STOP_TRADING.LOCK", Motivo::Conflicto, Some(perfil)));
            }
            Ok(())
        }
    }
}

fn leer_flags_auxiliares(perfil: Perfil, entorno: &dyn Fn(&str) -> Option<String>) -> Result<(), Rechazo> {
    // IPs inyectadas al lector del feed: saltarian el host del perfil.
    if entorno(VARIABLE_IPS_FEED).is_some() {
        return Err(Rechazo::de(VARIABLE_IPS_FEED, Motivo::Conflicto, Some(perfil)));
    }
    for nombre in FLAGS_AUXILIARES {
        if let Some(v) = entorno(nombre) {
            if v != "true" && v != "false" {
                return Err(Rechazo::de(nombre, Motivo::Invalido, Some(perfil)));
            }
            // En un perfil Mainnet, `true` haria que las rutas auxiliares (`is_demo_env`)
            // eligieran Testnet o la raiz demo.
            if v == "true" && !perfil.testnet() {
                return Err(Rechazo::de(nombre, Motivo::Conflicto, Some(perfil)));
            }
        }
    }
    Ok(())
}

/// Una sola pareja completa de `parejas`; ninguna variable de `ajenas` presente.
fn leer_llaves(
    perfil: Perfil,
    entorno: &dyn Fn(&str) -> Option<String>,
    parejas: &[(&'static str, &'static str); 2],
    ajenas: &[(&'static str, &'static str); 2],
) -> Result<Credenciales, Rechazo> {
    for (k, s) in ajenas {
        for nombre in [*k, *s] {
            if entorno(nombre).is_some() {
                return Err(Rechazo::de(nombre, Motivo::FamiliaAjena, Some(perfil)));
            }
        }
    }
    let leidas: Vec<(Option<String>, Option<String>)> =
        parejas.iter().map(|(k, s)| (entorno(k), entorno(s))).collect();
    // Cruzada: llave de una pareja y secreto de la otra, sin ninguna pareja entera.
    let mitad = |i: usize| leidas[i].0.is_some() != leidas[i].1.is_some();
    if mitad(0) && mitad(1) && leidas[0].0.is_some() != leidas[1].0.is_some() {
        return Err(Rechazo::de(parejas[0].0, Motivo::ParejaCruzada, Some(perfil)));
    }
    let mut completas = Vec::new();
    for (i, (k, s)) in leidas.iter().enumerate() {
        match (k, s) {
            (None, None) => {}
            (Some(_), None) => {
                return Err(Rechazo::de(parejas[i].1, Motivo::ParejaIncompleta, Some(perfil)))
            }
            (None, Some(_)) => {
                return Err(Rechazo::de(parejas[i].0, Motivo::ParejaIncompleta, Some(perfil)))
            }
            (Some(k), Some(s)) => {
                if k.trim().is_empty() {
                    return Err(Rechazo::de(parejas[i].0, Motivo::Vacio, Some(perfil)));
                }
                if s.trim().is_empty() {
                    return Err(Rechazo::de(parejas[i].1, Motivo::Vacio, Some(perfil)));
                }
                completas.push(i);
            }
        }
    }
    match completas.as_slice() {
        [] => Err(Rechazo::de(parejas[0].0, Motivo::Ausente, Some(perfil))),
        [i] => {
            let (k, s) = &leidas[*i];
            Ok(Credenciales {
                nombre_llave: parejas[*i].0,
                nombre_secreto: parejas[*i].1,
                llave: k.clone().unwrap_or_default(),
                secreto: s.clone().unwrap_or_default(),
            })
        }
        _ => Err(Rechazo::de(parejas[0].0, Motivo::AmbasParejas, Some(perfil))),
    }
}

fn leer_feed(perfil: Perfil, entorno: &dyn Fn(&str) -> Option<String>) -> Result<&'static str, Rechazo> {
    let canonico = perfil.host_feed();
    match entorno("BEST_WS_ENDPOINT") {
        None => Ok(canonico),
        Some(v) if v.is_empty() => Err(Rechazo::de("BEST_WS_ENDPOINT", Motivo::Vacio, Some(perfil))),
        Some(v) if v == canonico => Ok(canonico),
        Some(v) => {
            // Esquema, puerto, usuario, ruta, consulta, fragmento o espacios: no es un host.
            let sintaxis_de_host = !v.contains([':', '/', '?', '#', '@', ' ', '\\'])
                && url::Url::parse(&format!("wss://{v}/"))
                    .map(|u| u.host_str() == Some(v.as_str()))
                    .unwrap_or(false);
            let motivo = if sintaxis_de_host { Motivo::HostNoCanonico } else { Motivo::Invalido };
            Err(Rechazo::de("BEST_WS_ENDPOINT", motivo, Some(perfil)))
        }
    }
}

/// NUCLEO PURO: valida la instantanea completa. Orden: perfil, argumentos, `USE_TESTNET`,
/// marcadores, flags auxiliares, llaves, feed. El primer fallo decide.
pub fn validar(inst: &Instantanea<'_>) -> Result<ContextoSellado, Rechazo> {
    let perfil = leer_perfil(inst.args)?;
    argumentos_conocidos(inst.args, perfil)?;
    leer_use_testnet(perfil, inst.entorno)?;
    leer_marcadores(perfil, inst.marcadores)?;
    leer_flags_auxiliares(perfil, inst.entorno)?;
    let credenciales = match perfil {
        Perfil::DemoTestnet => Some(leer_llaves(perfil, inst.entorno, &PAREJAS_TESTNET, &PAREJAS_PRODUCCION)?),
        Perfil::MainnetLive => Some(leer_llaves(perfil, inst.entorno, &PAREJAS_PRODUCCION, &PAREJAS_TESTNET)?),
        // Papel: contexto privado vacio (cero autenticacion). Las llaves presentes no se leen.
        Perfil::PaperTestnet | Perfil::PaperMainnet => None,
    };
    let host_feed = leer_feed(perfil, inst.entorno)?;
    Ok(ContextoSellado { perfil, credenciales, host_feed })
}

static SELLADO: OnceLock<ContextoSellado> = OnceLock::new();

/// Sella el contexto del proceso. Una segunda llamada no lo cambia y devuelve `false`.
pub fn sellar(ctx: ContextoSellado) -> bool {
    SELLADO.set(ctx).is_ok()
}

/// El contexto sellado del proceso (`None` fuera de `god_engine`).
pub fn sellado() -> Option<&'static ContextoSellado> {
    SELLADO.get()
}

/// Venue de las rutas auxiliares: el sellado si existe; si no (otros binarios), la regla
/// heredada `quantum_arena::paths::is_demo_env`.
pub fn venue_testnet_auxiliar() -> bool {
    match sellado() {
        Some(ctx) => ctx.testnet(),
        None => quantum_arena::paths::is_demo_env(),
    }
}

/// Marcadores leidos del disco por el adaptador del proceso (las dos rutas de la base).
pub fn marcadores_del_proceso() -> Marcadores {
    let canonica = quantum_arena::paths::data_join("../../config_dir/MAINNET_ARMED").replace("data/../../", "");
    let parada_exe = std::env::current_exe()
        .ok()
        .and_then(|e| e.parent().map(|d| d.join("STOP_TRADING.LOCK").exists()))
        .unwrap_or(false);
    Marcadores {
        armado_canonico: std::path::Path::new(&canonica).exists(),
        armado_cwd: std::path::Path::new("config_dir/MAINNET_ARMED").exists(),
        parada: std::path::Path::new("STOP_TRADING.LOCK").exists() || parada_exe,
    }
}

/// ADAPTADOR DEL PROCESO: argumentos (sin el ejecutable), `std::env` y marcadores del disco.
/// No abre ningun transporte.
pub fn validar_proceso() -> Result<ContextoSellado, Rechazo> {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let entorno = |k: &str| std::env::var_os(k).map(|v| v.to_string_lossy().into_owned());
    validar(&Instantanea {
        args: &args,
        entorno: &entorno,
        marcadores: marcadores_del_proceso(),
    })
}
