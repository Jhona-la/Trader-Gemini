//! PUERTA-REST (P1): la puerta REST única hacia Binance USDⓈ-M Futures.
//!
//! Pedido del coordinador 20261116-0347 (especificación
//! `vibe_cli_pedido_puerta_rest_unica_v3.md`, revisada APTA por KILO-CLI).
//! Objetivo del operador: usar la API con cuidado para que Binance nunca
//! bloquee la IP. Este módulo centraliza el presupuesto por IP de las tres
//! ventanas de `exchangeInfo.rateLimits` (peso 1 min, órdenes 10 s y 1 min).
//!
//! P1 son sólo tipos y la contabilidad, sin tráfico:
//!
//! - `CallPriority`: tramo de cuota de una petición. Regla de ruina del
//!   pedido: las órdenes de cierre y de protección nunca quedan detrás de
//!   lecturas ni de lotes; si el presupuesto no alcanza, se frenan primero
//!   `Batch` y `Lectura`.
//! - `VentanaUso`: uso de un contador. El uso efectivo es
//!   `max(estimación local, cabecera + en vuelo + perdido)`: la parte de
//!   cabeceras ES el `ContadorExchange` del ejecutor (heredado, no duplicado,
//!   el de `OrderExecutor::observar_limites`), la parte local suma el coste de
//!   lo que la puerta admitió dentro de la ventana, y el libro en vuelo y lo
//!   perdido sin respuesta son los del ejecutor (CL-179, CL-201).
//! - `LimitesExchange`: los tres límites con su ventana; el respaldo son las
//!   constantes LG-12 de `binance_api.rs` (sin copiarlas).
//! - `GatewayCallRecord`: una fila de la cinta de auditoría.
//!
//! Sin números puestos a mano: cada constante cita su fuente.

use crate::binance_api::{
    CosteBinance, LIMITE_ORDENES_10S, LIMITE_ORDENES_1M, LIMITE_PESO_1M, VENTANA_ORDENES_10S_MS,
    VENTANA_ORDENES_1M_MS, VENTANA_PESO_MS,
};
use crate::client::BinanceRateLimits;
use crate::executor::ContadorExchange;
use std::collections::VecDeque;

/// Fracción del límite por encima de la cual frenan las peticiones que no
/// son críticas. Fuente: `executor.rs`, `const HEADROOM: f64 = 0.8;` del tramo
/// `Lectura` (F1.8, N5 CL-179) y `docs/BINANCE_API_MAP.md` §1 nota N5 (CL-187:
/// las lecturas «frenan con el uso por encima del 80 % del límite»). La prueba
/// `gateway_fraccion_lectura_es_la_del_ejecutor` exige que sea el mismo literal
/// que el del ejecutor: si uno cambia, la prueba falla.
pub const FRACCION_FRENO_LECTURA: f64 = 0.8;

/// Índices de los tres contadores, en el orden de `exchangeInfo.rateLimits`
/// que usa el ejecutor (`UsoExchange`): peso 1 min, órdenes 10 s, órdenes 1 min.
pub const CONTADOR_PESO_1M: usize = 0;
pub const CONTADOR_ORDENES_10S: usize = 1;
pub const CONTADOR_ORDENES_1M: usize = 2;
/// Nombres de los contadores para los mensajes y la cinta.
pub const NOMBRES_CONTADORES: [&str; 3] = ["PESO_1M", "ORDENES_10S", "ORDENES_1M"];

/// Tramo de cuota de una petición que pasa por la puerta.
///
/// Orden de admisión (la regla de ruina): para cualquier estado del uso, lo
/// que admite `Batch` lo admite `Lectura`, lo que admite `Lectura` lo admite
/// `Alta` y lo que admite `Alta` lo admite `Critica`. Ninguna prioridad pasa
/// nunca del límite del exchange en un contador en el que gasta.
///
/// - `Critica`: cierres, protección y lo que los sostiene (piernas TP/SL,
///   cancelaciones, positionRisk de reconciliación, consulta de una orden
///   ambigua). Pasa mientras `uso + coste ≤ L` en cada contador en el que gasta
///   y `uso ≤ L` en los demás: la regla del exchange, sin literal (la del tramo
///   `Critica` del ejecutor, CL-179/CL-181). Su coste queda reservado en la
///   estimación local al admitirla.
/// - `Alta`: entradas (órdenes que abren). Frena por encima de
///   `FRACCION_FRENO_LECTURA·L`, como las entradas del ejecutor (CL-179:
///   «entradas y lecturas frenan con el uso por encima del 80 %»).
/// - `Lectura`: lecturas de cuenta y de mercado. Frena por encima de
///   `FRACCION_FRENO_LECTURA·L` (CL-187).
/// - `Batch`: lotes y sondeos masivos. Frena con el mismo umbral que
///   `Lectura`: el repo no tiene una fuente para un umbral menor y el pedido
///   prohíbe inventarlo. Queda anotado en el recibo como decisión pendiente.
///
/// La franja `(0,8·L, L]` es la reserva del tramo crítico: allí sólo pasan
/// `Critica`.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, serde::Serialize)]
pub enum CallPriority {
    Critica,
    Alta,
    Lectura,
    Batch,
}

impl CallPriority {
    /// Las cuatro, de la que más pasa a la que menos.
    pub const TODAS: [CallPriority; 4] = [
        CallPriority::Critica,
        CallPriority::Alta,
        CallPriority::Lectura,
        CallPriority::Batch,
    ];
}

/// De dónde salió un límite.
#[derive(Clone, Copy, Debug, PartialEq, Eq, serde::Serialize)]
pub enum OrigenLimite {
    /// `GET /fapi/v1/exchangeInfo` → `rateLimits` (P3).
    ExchangeInfo,
    /// Constantes LG-12 / N5 de `binance_api.rs` (respaldo).
    RespaldoLg12,
}

/// Un límite y la ventana en la que vale su contador.
#[derive(Clone, Copy, Debug, PartialEq, Eq, serde::Serialize)]
pub struct LimiteVentana {
    pub limite: usize,
    pub ventana_ms: u64,
    pub origen: OrigenLimite,
}

/// Los tres límites de `exchangeInfo.rateLimits`.
#[derive(Clone, Copy, Debug, PartialEq, Eq, serde::Serialize)]
pub struct LimitesExchange {
    pub peso_1m: LimiteVentana,
    pub ordenes_10s: LimiteVentana,
    pub ordenes_1m: LimiteVentana,
}

impl LimitesExchange {
    /// Respaldo: las constantes LG-12 (`binance_api.rs`, `LIMITE_PESO_1M`,
    /// `LIMITE_ORDENES_10S`, `LIMITE_ORDENES_1M`) con sus ventanas N5
    /// (`VENTANA_PESO_MS`, `VENTANA_ORDENES_10S_MS`, `VENTANA_ORDENES_1M_MS`).
    pub const fn respaldo_lg12() -> Self {
        Self {
            peso_1m: LimiteVentana {
                limite: LIMITE_PESO_1M,
                ventana_ms: VENTANA_PESO_MS,
                origen: OrigenLimite::RespaldoLg12,
            },
            ordenes_10s: LimiteVentana {
                limite: LIMITE_ORDENES_10S,
                ventana_ms: VENTANA_ORDENES_10S_MS,
                origen: OrigenLimite::RespaldoLg12,
            },
            ordenes_1m: LimiteVentana {
                limite: LIMITE_ORDENES_1M,
                ventana_ms: VENTANA_ORDENES_1M_MS,
                origen: OrigenLimite::RespaldoLg12,
            },
        }
    }

    /// Los tres, en el orden de los contadores.
    pub fn como_arreglo(&self) -> [LimiteVentana; 3] {
        [self.peso_1m, self.ordenes_10s, self.ordenes_1m]
    }
}

// ------------------------------------------------------------------ P3 ---

/// El resultado de leer `exchangeInfo.rateLimits`: los tres límites (cada uno
/// con su origen) y las entradas que no se usaron, con el motivo.
#[derive(Clone, Debug, PartialEq, Eq, serde::Serialize)]
pub struct LecturaRateLimits {
    pub limites: LimitesExchange,
    pub ignorados: Vec<String>,
}

/// La duración en ms de un `interval` × `intervalNum` de `rateLimits`. Sólo
/// SECOND y MINUTE: son los de los tres contadores (binance_api.rs, N5);
/// cualquier otro se ignora. Los segundos de cada unidad son su definición,
/// no un parámetro.
fn ventana_de_intervalo(intervalo: &str, numero: u64) -> Option<u64> {
    let unidad = match intervalo {
        "SECOND" => std::time::Duration::from_secs(1),
        "MINUTE" => std::time::Duration::from_secs(60),
        _ => return None,
    };
    u64::try_from(unidad.as_millis())
        .ok()
        .and_then(|ms| ms.checked_mul(numero))
}

impl LimitesExchange {
    /// Lee `rateLimits` del cuerpo de `GET /fapi/v1/exchangeInfo`.
    ///
    /// - Cada contador toma la entrada cuyo tipo y ventana son los suyos
    ///   (`REQUEST_WEIGHT` con `VENTANA_PESO_MS`, `ORDERS` con
    ///   `VENTANA_ORDENES_10S_MS` y con `VENTANA_ORDENES_1M_MS`): las
    ///   cabeceras `X-MBX-USED-WEIGHT-1M` / `X-MBX-ORDER-COUNT-10S/1M` valen
    ///   en esas ventanas.
    /// - El límite es el que declara el exchange (cambia por cuenta/VIP,
    ///   `docs/BINANCE_API_MAP.md` cabecera), mayor o menor que LG-12.
    /// - Un contador sin entrada válida cae a su constante LG-12 (respaldo por
    ///   contador). Dos entradas para el mismo contador: la menor (lo
    ///   conservador).
    /// - Entradas de otro tipo (RAW_REQUESTS), otra ventana, límite 0 o mal
    ///   formadas no se usan y quedan en `ignorados`.
    ///
    /// Error si el cuerpo no es JSON o no trae `rateLimits` como lista.
    pub fn desde_exchange_info(cuerpo: &str) -> Result<LecturaRateLimits, String> {
        let valor: serde_json::Value = serde_json::from_str(cuerpo)
            .map_err(|e| format!("exchangeInfo no es JSON: {e}"))?;
        let entradas = valor
            .get("rateLimits")
            .and_then(|v| v.as_array())
            .ok_or_else(|| "exchangeInfo sin rateLimits".to_string())?;
        let mut limites = LimitesExchange::respaldo_lg12();
        let mut ignorados = Vec::new();
        for (i, e) in entradas.iter().enumerate() {
            let tipo = e.get("rateLimitType").and_then(|v| v.as_str());
            let intervalo = e.get("interval").and_then(|v| v.as_str());
            let numero = e.get("intervalNum").and_then(|v| v.as_u64());
            let limite = e
                .get("limit")
                .and_then(|v| v.as_u64())
                .and_then(|l| usize::try_from(l).ok());
            let (Some(tipo), Some(intervalo), Some(numero), Some(limite)) =
                (tipo, intervalo, numero, limite)
            else {
                ignorados.push(format!("rateLimits[{i}]: mal formada"));
                continue;
            };
            let Some(ventana_ms) = ventana_de_intervalo(intervalo, numero) else {
                ignorados.push(format!("rateLimits[{i}]: {tipo} {numero} {intervalo}: ventana sin contador"));
                continue;
            };
            if limite == 0 {
                ignorados.push(format!("rateLimits[{i}]: {tipo} limite 0"));
                continue;
            }
            let hueco = match (tipo, ventana_ms) {
                ("REQUEST_WEIGHT", v) if v == VENTANA_PESO_MS => &mut limites.peso_1m,
                ("ORDERS", v) if v == VENTANA_ORDENES_10S_MS => &mut limites.ordenes_10s,
                ("ORDERS", v) if v == VENTANA_ORDENES_1M_MS => &mut limites.ordenes_1m,
                _ => {
                    ignorados.push(format!(
                        "rateLimits[{i}]: {tipo} {numero} {intervalo}: sin contador"
                    ));
                    continue;
                }
            };
            if hueco.origen == OrigenLimite::ExchangeInfo && hueco.limite <= limite {
                ignorados.push(format!("rateLimits[{i}]: {tipo} repetida, queda la menor"));
                continue;
            }
            *hueco = LimiteVentana {
                limite,
                ventana_ms,
                origen: OrigenLimite::ExchangeInfo,
            };
        }
        Ok(LecturaRateLimits { limites, ignorados })
    }

    /// Como `desde_exchange_info`, pero sin cuerpo o con un cuerpo ilegible
    /// devuelve el respaldo LG-12 entero (con el motivo en `ignorados`).
    pub fn desde_exchange_info_o_respaldo(cuerpo: Option<&str>) -> LecturaRateLimits {
        match cuerpo.map(LimitesExchange::desde_exchange_info) {
            Some(Ok(l)) => l,
            Some(Err(e)) => LecturaRateLimits {
                limites: LimitesExchange::respaldo_lg12(),
                ignorados: vec![e],
            },
            None => LecturaRateLimits {
                limites: LimitesExchange::respaldo_lg12(),
                ignorados: vec!["sin exchangeInfo".to_string()],
            },
        }
    }
}

/// Lo que una petición suma a cada contador (peso, órdenes 10 s, órdenes
/// 1 min): `CosteBinance` reparte `ordenes` en los dos contadores ORDERS.
#[inline]
pub fn contadores_de(coste: CosteBinance) -> [usize; 3] {
    [coste.peso, coste.ordenes, coste.ordenes]
}

/// Uso de UN contador vigente en la puerta.
///
/// - `cabecera`: las muestras `X-MBX-USED-WEIGHT-1M` / `X-MBX-ORDER-COUNT-*`
///   con el `ContadorExchange` del ejecutor (máximo de las vigentes, CL-177 y
///   CL-180).
/// - `local`: el coste de cada petición que la puerta admitió, con su sello;
///   cuenta mientras su edad es menor que la ventana. Se anota al admitir
///   (antes de enviar), así que cubre también las peticiones en vuelo y las
///   que salieron sin respuesta (lo conservador, como `SinRespuesta`, CL-201).
///
/// El uso efectivo es `max(local, cabecera + en vuelo + perdido)`:
///
/// - la cabecera corrige la deriva de la estimación local y trae el consumo
///   de otros procesos con la misma IP (`max`, como `observar_limites`);
/// - `en_vuelo`: el coste de las peticiones admitidas cuya respuesta no llegó
///   todavía; ninguna cabecera lo dice aún, así que se suma a la cabecera
///   (el libro en vuelo del ejecutor, CL-179/CL-181: N peticiones
///   concurrentes no caben en el hueco de una);
/// - `perdido`: el coste de las que salieron sin respuesta, hasta que caduca
///   la ventana (`SinRespuesta` del ejecutor, CL-201);
/// - la estimación local cubre lo que la puerta envió en la ventana aunque
///   ninguna respuesta haya traído cabecera.
///
/// Las colas están acotadas por la admisión: ninguna petición entra si
/// dejaría el contador por encima de su límite, y sólo se guardan costes
/// mayores que cero.
#[derive(Clone, Debug)]
pub struct VentanaUso {
    limite: LimiteVentana,
    cabecera: ContadorExchange,
    local: VecDeque<(u64, usize)>,
    en_vuelo: usize,
    perdido: VecDeque<(u64, usize)>,
}

/// Un sello futuro (reloj que retrocede) alargaría la vida de su coste más
/// allá de la ventana: se acota a ahora (conserva el coste). Las caducadas
/// son un prefijo (sellos no decrecientes) y salen.
fn podar_cola(cola: &mut VecDeque<(u64, usize)>, ahora_ms: u64, ventana_ms: u64) {
    for (sello, _) in cola.iter_mut().rev() {
        if *sello <= ahora_ms {
            break;
        }
        *sello = ahora_ms;
    }
    while cola
        .front()
        .is_some_and(|(sello, _)| ahora_ms.saturating_sub(*sello) >= ventana_ms)
    {
        cola.pop_front();
    }
}

fn suma_cola(cola: &VecDeque<(u64, usize)>) -> usize {
    cola.iter()
        .fold(0usize, |suma, (_, coste)| suma.saturating_add(*coste))
}

impl VentanaUso {
    pub fn nueva(limite: LimiteVentana) -> Self {
        Self {
            limite,
            cabecera: ContadorExchange::default(),
            local: VecDeque::new(),
            en_vuelo: 0,
            perdido: VecDeque::new(),
        }
    }

    pub fn limite(&self) -> LimiteVentana {
        self.limite
    }

    /// Una petición admitida: su coste entra en la estimación local y en el
    /// libro en vuelo.
    pub fn reservar(&mut self, coste: usize, ahora_ms: u64) {
        if coste == 0 {
            return;
        }
        podar_cola(&mut self.local, ahora_ms, self.limite.ventana_ms);
        self.local.push_back((ahora_ms, coste));
        self.en_vuelo = self.en_vuelo.saturating_add(coste);
    }

    /// Su respuesta llegó (o fue un 429/418): sale del libro en vuelo. Si
    /// salió sin respuesta (`sin_respuesta`), su coste queda como perdido
    /// hasta que caduca la ventana.
    pub fn liberar(&mut self, coste: usize, sin_respuesta: bool, ahora_ms: u64) {
        if coste == 0 {
            return;
        }
        self.en_vuelo = self.en_vuelo.saturating_sub(coste);
        if sin_respuesta {
            podar_cola(&mut self.perdido, ahora_ms, self.limite.ventana_ms);
            self.perdido.push_back((ahora_ms, coste));
        }
    }

    /// Registra una cabecera de uso de una respuesta.
    pub fn observar_cabecera(&mut self, uso: usize, ahora_ms: u64) {
        self.cabecera.observar(uso, ahora_ms, self.limite.ventana_ms);
    }

    /// La estimación local vigente.
    pub fn uso_local(&mut self, ahora_ms: u64) -> usize {
        podar_cola(&mut self.local, ahora_ms, self.limite.ventana_ms);
        suma_cola(&self.local)
    }

    /// La mayor cabecera vigente.
    pub fn uso_cabecera(&self, ahora_ms: u64) -> usize {
        self.cabecera.vigente(ahora_ms, self.limite.ventana_ms)
    }

    /// El coste admitido que aún no tiene respuesta.
    pub fn en_vuelo(&self) -> usize {
        self.en_vuelo
    }

    /// `max(local, cabecera + en vuelo + perdido)`.
    pub fn uso_efectivo(&mut self, ahora_ms: u64) -> usize {
        let local = self.uso_local(ahora_ms);
        podar_cola(&mut self.perdido, ahora_ms, self.limite.ventana_ms);
        let medido = self
            .uso_cabecera(ahora_ms)
            .saturating_add(self.en_vuelo)
            .saturating_add(suma_cola(&self.perdido));
        local.max(medido)
    }
}

/// Una petición admitida: su coste está en el libro en vuelo hasta que se
/// cierra con `PresupuestoPuerta::cerrar`.
#[must_use = "una reserva sin cerrar deja su coste en vuelo para siempre"]
#[derive(Debug, PartialEq, Eq)]
pub struct Reserva {
    pub coste: CosteBinance,
    pub uso_antes: [usize; 3],
}

/// Cómo terminó una petición admitida.
#[derive(Clone, Copy, Debug)]
pub enum Desenlace {
    /// Hubo respuesta (2xx o error que no es 429/418): su cabecera manda.
    Respuesta(BinanceRateLimits),
    /// 429/418: su cabecera nunca se registra, manda su Retry-After (CL-196).
    LimiteExchange,
    /// Salió y no hubo respuesta: su coste queda perdido en la ventana.
    SinRespuesta,
}

// ------------------------------------------------------------------ P2 ---

/// Método HTTP de una firma.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, serde::Serialize)]
pub enum Metodo {
    Get,
    Post,
    Put,
    Delete,
}

impl Metodo {
    pub const fn como_str(self) -> &'static str {
        match self {
            Metodo::Get => "GET",
            Metodo::Post => "POST",
            Metodo::Put => "PUT",
            Metodo::Delete => "DELETE",
        }
    }

    pub fn de_str(s: &str) -> Option<Metodo> {
        match s.trim() {
            "GET" => Some(Metodo::Get),
            "POST" => Some(Metodo::Post),
            "PUT" => Some(Metodo::Put),
            "DELETE" => Some(Metodo::Delete),
            _ => None,
        }
    }
}

/// Si el peso de la firma depende de que la petición lleve `symbol`.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, serde::Serialize)]
pub enum Alcance {
    /// Un solo peso, con o sin símbolo.
    Unico,
    /// Peso de la petición con `symbol`.
    ConSimbolo,
    /// Peso de la petición sin `symbol` (todos los símbolos).
    SinSimbolo,
}

/// El peso de una firma según el mapa.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum PesoFirma {
    /// Marcado ✔ en `docs/BINANCE_API_MAP.md` §1 (comprobado el 2026-11-01
    /// contra la documentación oficial, N22/CL-218).
    Verificado(CosteBinance),
    /// En el mapa pero sin ✔: N/D. La puerta lo rechaza (falla cerrado).
    NoVerificado,
}

/// Una fila de la tabla de pesos: una firma (método + ruta + alcance).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct FilaPeso {
    pub metodo: Metodo,
    pub ruta: &'static str,
    pub alcance: Alcance,
    pub peso: PesoFirma,
}

const fn fila(metodo: Metodo, ruta: &'static str, alcance: Alcance, peso: PesoFirma) -> FilaPeso {
    FilaPeso {
        metodo,
        ruta,
        alcance,
        peso,
    }
}

const fn verificado(peso: usize, ordenes: usize) -> PesoFirma {
    PesoFirma::Verificado(CosteBinance { peso, ordenes })
}

/// La tabla de pesos por firma: TODAS las filas de `docs/BINANCE_API_MAP.md`
/// §1 «Endpoints EN USO», ni una más. La prueba
/// `gateway_tabla_pesos_paridad_con_mapa` la compara fila a fila con el mapa:
/// si el mapa cambia, la prueba falla hasta que la tabla lo siga. Los costes
/// del ejecutor (`COSTE_*` de binance_api.rs) se usan donde existen, así que
/// la misma prueba ata también esos costes al mapa. Las líneas citadas son las
/// del mapa en el commit de P2.
///
/// Lo que no está en §1 (income, openInterest, premiumIndex, fundingRate,
/// ticker, `/futures/data/*`, ...) no tiene fila y falla cerrado como firma
/// desconocida; nunca se supone peso 1.
pub const TABLA_PESOS: &[FilaPeso] = {
    use crate::binance_api::{
        COSTE_ALGO_ABIERTAS_SIMBOLO, COSTE_CANCELACION, COSTE_CONSULTA_ORDEN, COSTE_ORDEN,
        COSTE_PIERNA_ALGO, COSTE_POSITION_RISK,
    };
    use Alcance::{ConSimbolo, SinSimbolo, Unico};
    use Metodo::{Delete, Get, Post, Put};
    use PesoFirma::{NoVerificado, Verificado};
    &[
        // :13 «0 ✔ (y 1 en cada contador de órdenes, 10 s y 1 min)»
        fila(Post, "/fapi/v1/order", Unico, Verificado(COSTE_ORDEN)),
        // :14 «1 ✔»
        fila(Get, "/fapi/v1/order", Unico, Verificado(COSTE_CONSULTA_ORDEN)),
        // :15 «1 ✔»
        fila(Delete, "/fapi/v1/order", Unico, Verificado(COSTE_CANCELACION)),
        // :16 «5» sin ✔
        fila(Post, "/fapi/v1/batchOrders", Unico, NoVerificado),
        // :17 POST/PUT «1» sin ✔
        fila(Post, "/fapi/v1/listenKey", Unico, NoVerificado),
        fila(Put, "/fapi/v1/listenKey", Unico, NoVerificado),
        // :18 «5 ✔»
        fila(Get, "/fapi/v2/balance", Unico, verificado(5, 0)),
        // :19 «5 ✔»
        fila(Get, "/fapi/v2/positionRisk", Unico, Verificado(COSTE_POSITION_RISK)),
        // :20 «5 ✔»
        fila(Get, "/fapi/v2/account", Unico, verificado(5, 0)),
        // :21 «20 ✔ (antes figuraba 1)»
        fila(Get, "/fapi/v1/commissionRate", Unico, verificado(20, 0)),
        // :22 «1» sin ✔
        fila(Get, "/fapi/v1/leverageBracket", Unico, NoVerificado),
        // :23 «1» sin ✔
        fila(Get, "/fapi/v1/exchangeInfo", Unico, NoVerificado),
        // :24 «1 ✔»
        fila(Post, "/fapi/v1/leverage", Unico, verificado(1, 0)),
        // :25 «1» sin ✔
        fila(Post, "/fapi/v1/marginType", Unico, NoVerificado),
        // :26 «1» sin ✔
        fila(Get, "/fapi/v1/time", Unico, NoVerificado),
        // :27 «1-10» sin ✔
        fila(Get, "/fapi/v1/klines", Unico, NoVerificado),
        // :28 «20» sin ✔
        fila(Get, "/fapi/v1/aggTrades", Unico, NoVerificado),
        // :29 «0 ✔ (y 1 en cada contador de órdenes; ...)»
        fila(Post, "/fapi/v1/algoOrder", Unico, Verificado(COSTE_PIERNA_ALGO)),
        // :30 «1 ✔»
        fila(Delete, "/fapi/v1/algoOrder", Unico, Verificado(COSTE_CANCELACION)),
        // :31 «1 ✔»
        fila(Delete, "/fapi/v1/algoOpenOrders", Unico, Verificado(COSTE_CANCELACION)),
        // :32 «1 con símbolo / 40 sin él ✔»
        fila(
            Get,
            "/fapi/v1/openAlgoOrders",
            ConSimbolo,
            Verificado(COSTE_ALGO_ABIERTAS_SIMBOLO),
        ),
        fila(Get, "/fapi/v1/openAlgoOrders", SinSimbolo, verificado(40, 0)),
        // :33 «1 ✔»
        fila(Delete, "/fapi/v1/allOpenOrders", Unico, Verificado(COSTE_CANCELACION)),
        // :34 «1 con símbolo / 40 sin él ✔»
        fila(Get, "/fapi/v1/openOrders", ConSimbolo, verificado(1, 0)),
        fila(Get, "/fapi/v1/openOrders", SinSimbolo, verificado(40, 0)),
        // :35 «GET 30 ✔ / POST 1 ✔»
        fila(Get, "/fapi/v1/positionSide/dual", Unico, verificado(30, 0)),
        fila(Post, "/fapi/v1/positionSide/dual", Unico, verificado(1, 0)),
    ]
};

/// La fila de la tabla para una petición, si existe.
pub fn fila_de(metodo: Metodo, ruta: &str, con_simbolo: bool) -> Option<&'static FilaPeso> {
    TABLA_PESOS.iter().find(|f| {
        f.metodo == metodo
            && f.ruta == ruta
            && match f.alcance {
                Alcance::Unico => true,
                Alcance::ConSimbolo => con_simbolo,
                Alcance::SinSimbolo => !con_simbolo,
            }
    })
}

/// El coste de una petición según la tabla, o el rechazo (falla cerrado):
/// firma fuera del mapa o con peso N/D.
pub fn coste_de_firma(
    metodo: Metodo,
    ruta: &str,
    con_simbolo: bool,
) -> Result<CosteBinance, RechazoPuerta> {
    let firma = format!(
        "{} {}{}",
        metodo.como_str(),
        ruta,
        if con_simbolo { " (con symbol)" } else { "" }
    );
    match fila_de(metodo, ruta, con_simbolo) {
        None => Err(RechazoPuerta::FirmaDesconocida(firma)),
        Some(FilaPeso {
            peso: PesoFirma::NoVerificado,
            ..
        }) => Err(RechazoPuerta::PesoNoVerificado(firma)),
        Some(FilaPeso {
            peso: PesoFirma::Verificado(coste),
            ..
        }) => Ok(*coste),
    }
}

/// Por qué la puerta no dejó salir una petición, o por qué falló.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum RechazoPuerta {
    /// La firma (método + ruta + alcance) no está en la tabla de pesos (P2):
    /// falla cerrado, nunca se supone peso 1.
    FirmaDesconocida(String),
    /// La firma está en el mapa pero su peso no está verificado (N/D): falla
    /// cerrado.
    PesoNoVerificado(String),
    /// El freno 429/418 heredado (`FrenoExchange`) está activo.
    Frenado(String),
    /// La prioridad no cabe en el presupuesto de algún contador.
    Presupuesto(String),
    /// La respuesta fue un 429 o un 418: el freno quedó armado con su
    /// Retry-After.
    LimiteExchange(String),
    /// Error de transporte (sin respuesta).
    Red(String),
}

impl std::fmt::Display for RechazoPuerta {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            RechazoPuerta::FirmaDesconocida(t) => write!(f, "PUERTA_FIRMA_DESCONOCIDA: {t}"),
            RechazoPuerta::PesoNoVerificado(t) => write!(f, "PUERTA_PESO_NO_VERIFICADO: {t}"),
            RechazoPuerta::Frenado(t) => write!(f, "PUERTA_FRENADA: {t}"),
            RechazoPuerta::Presupuesto(t) => write!(f, "PUERTA_PRESUPUESTO: {t}"),
            RechazoPuerta::LimiteExchange(t) => write!(f, "PUERTA_LIMITE_EXCHANGE: {t}"),
            RechazoPuerta::Red(t) => write!(f, "PUERTA_RED: {t}"),
        }
    }
}

/// El presupuesto por IP de la puerta: una `VentanaUso` por contador.
#[derive(Clone, Debug)]
pub struct PresupuestoPuerta {
    limites: LimitesExchange,
    ventanas: [VentanaUso; 3],
}

impl PresupuestoPuerta {
    pub fn nuevo(limites: LimitesExchange) -> Self {
        let [peso, o10, o1m] = limites.como_arreglo();
        Self {
            limites,
            ventanas: [
                VentanaUso::nueva(peso),
                VentanaUso::nueva(o10),
                VentanaUso::nueva(o1m),
            ],
        }
    }

    pub fn limites(&self) -> LimitesExchange {
        self.limites
    }

    /// El uso efectivo `max(local, cabecera)` de los tres contadores.
    pub fn uso_efectivo(&mut self, ahora_ms: u64) -> [usize; 3] {
        let mut uso = [0usize; 3];
        for (k, v) in self.ventanas.iter_mut().enumerate() {
            uso[k] = v.uso_efectivo(ahora_ms);
        }
        uso
    }

    /// La estimación local de los tres contadores.
    pub fn uso_local(&mut self, ahora_ms: u64) -> [usize; 3] {
        let mut uso = [0usize; 3];
        for (k, v) in self.ventanas.iter_mut().enumerate() {
            uso[k] = v.uso_local(ahora_ms);
        }
        uso
    }

    /// Registra las cabeceras de una respuesta: sólo los contadores cuya
    /// cabecera llegó, cada uno con su sello (como `observar_limites`).
    pub fn observar(&mut self, limites: &BinanceRateLimits, ahora_ms: u64) {
        let muestras = [limites.weight_1m, limites.orders_10s, limites.orders_1m];
        for (k, m) in muestras.into_iter().enumerate() {
            if let Some(uso) = m {
                self.ventanas[k].observar_cabecera(uso, ahora_ms);
            }
        }
    }

    /// Decide sin anotar nada.
    pub fn cabe(
        &mut self,
        prioridad: CallPriority,
        coste: CosteBinance,
        ahora_ms: u64,
    ) -> Result<[usize; 3], RechazoPuerta> {
        let uso = self.uso_efectivo(ahora_ms);
        let gasto = contadores_de(coste);
        for k in 0..3 {
            let limite = self.ventanas[k].limite().limite;
            // Regla del exchange para todas las prioridades: nada sale si
            // dejaría un contador en el que gasta por encima de su límite, ni
            // con un contador ya por encima de su límite.
            let tope = if gasto[k] > 0 {
                uso[k].saturating_add(gasto[k]) > limite
            } else {
                uso[k] > limite
            };
            if tope {
                return Err(RechazoPuerta::Presupuesto(format!(
                    "{:?} {}: uso {} + coste {} > limite {}",
                    prioridad, NOMBRES_CONTADORES[k], uso[k], gasto[k], limite
                )));
            }
            if prioridad != CallPriority::Critica
                && uso[k] as f64 > limite as f64 * FRACCION_FRENO_LECTURA
            {
                return Err(RechazoPuerta::Presupuesto(format!(
                    "{:?} {}: uso {} > {} del limite {}",
                    prioridad, NOMBRES_CONTADORES[k], uso[k], FRACCION_FRENO_LECTURA, limite
                )));
            }
        }
        Ok(uso)
    }

    /// Admite la petición y reserva su coste (estimación local y libro en
    /// vuelo), o la rechaza sin tocar nada.
    pub fn admitir(
        &mut self,
        prioridad: CallPriority,
        coste: CosteBinance,
        ahora_ms: u64,
    ) -> Result<Reserva, RechazoPuerta> {
        let uso_antes = self.cabe(prioridad, coste, ahora_ms)?;
        let gasto = contadores_de(coste);
        for (k, v) in self.ventanas.iter_mut().enumerate() {
            v.reservar(gasto[k], ahora_ms);
        }
        Ok(Reserva { coste, uso_antes })
    }

    /// Cierra una reserva: con respuesta registra antes su cabecera (CL-182:
    /// la cabecera entra antes de que el coste salga del libro en vuelo).
    pub fn cerrar(&mut self, reserva: Reserva, desenlace: Desenlace, ahora_ms: u64) {
        if let Desenlace::Respuesta(limites) = desenlace {
            self.observar(&limites, ahora_ms);
        }
        let sin_respuesta = matches!(desenlace, Desenlace::SinRespuesta);
        let gasto = contadores_de(reserva.coste);
        for (k, v) in self.ventanas.iter_mut().enumerate() {
            v.liberar(gasto[k], sin_respuesta, ahora_ms);
        }
    }

    /// El coste en vuelo de los tres contadores.
    pub fn en_vuelo(&self) -> [usize; 3] {
        [
            self.ventanas[0].en_vuelo(),
            self.ventanas[1].en_vuelo(),
            self.ventanas[2].en_vuelo(),
        ]
    }
}

/// Cómo terminó una petición en la puerta.
#[derive(Clone, Debug, PartialEq, Eq, serde::Serialize)]
pub enum ResultadoLlamada {
    /// No salió: firma desconocida o peso no verificado.
    RechazadaFirma,
    /// No salió: freno 429/418 activo.
    RechazadaFreno,
    /// No salió: no cabía en el presupuesto de su prioridad.
    RechazadaPresupuesto,
    /// Salió y respondió con este estado HTTP (ni 429 ni 418).
    Respondida { estado: u16 },
    /// Salió y respondió 429 o 418: el freno quedó armado.
    LimiteExchange { estado: u16 },
    /// Salió (o pudo salir) y no hubo respuesta.
    ErrorRed,
}

/// Una fila de la cinta de auditoría de la puerta: una por pedido, haya
/// salido o no. Nunca lleva la consulta ni la firma HMAC (CL-202: los errores
/// no llevan la URL firmada), sólo la firma de peso (método y ruta).
#[derive(Clone, Debug, PartialEq, Eq, serde::Serialize)]
pub struct GatewayCallRecord {
    /// Reloj de la puerta (ms) al decidir.
    pub sello_ms: u64,
    pub metodo: String,
    pub ruta: String,
    pub con_simbolo: bool,
    pub prioridad: CallPriority,
    /// Coste de la tabla de pesos (`None` si la firma no tiene peso
    /// verificado).
    pub coste_peso: Option<usize>,
    pub coste_ordenes: Option<usize>,
    pub resultado: ResultadoLlamada,
    /// Detalle del rechazo o del error; vacío si respondió.
    pub detalle: String,
    /// Cabeceras de uso de la respuesta (si llegaron).
    pub cabecera_peso_1m: Option<usize>,
    pub cabecera_ordenes_10s: Option<usize>,
    pub cabecera_ordenes_1m: Option<usize>,
    /// Uso efectivo `max(local, cabecera)` tras el pedido.
    pub uso_efectivo: [usize; 3],
}
