//! CAPITAL-UNICO — puente de la cuenta: DOS magnitudes con nombre, cada una
//! definida igual en REST (`/fapi/v2/account`) y en el user-data stream
//! (`ACCOUNT_UPDATE`).
//!
//! - **SALDO** = `walletBalance` del activo de margen (USDT). Alimenta el
//!   arena y el tamaño. Nunca lleva PnL no realizado.
//! - **EQUITY** = saldo + PnL no realizado de las posiciones con margen en
//!   USDT. Observación separada para el freno y las guardas con posición
//!   abierta. `marginBalance` de REST NO siembra el arena.
//!
//! V3 (dictamen de Codex QRA sobre 1757709c4; decisiones del coordinador del
//! 2026-11-12):
//!
//! - **Frescura por componente (P1.1)**: el saldo y cada posición llevan su
//!   propia marca del exchange (`updateTime` de su fila en REST; `T`/`E` del
//!   evento en el stream). Se compara SIEMPRE (no sólo si cambió algo desde el
//!   ticket). Un dato con marca más nueva se aplica; uno con marca igual o más
//!   vieja, o sin marca, sólo se acepta si NO modifica lo ya observado (mismo
//!   saldo; misma cantidad de la posición). Un empate con valor distinto es una
//!   contradicción y no sobrescribe. Una lectura REST se acepta entera o se
//!   rechaza entera. La forma V2 ([`LecturaRest`], una sola marca compuesta)
//!   se sigue aceptando, pero su marca no prueba la frescura de ningún
//!   componente: cuenta como «sin orden».
//! - **Era (P1.2)**: una por sesión de cuenta. [`PuenteCuenta::nueva_era`]
//!   deja el saldo sin observar y la equity incompleta: hasta una lectura
//!   aceptada de la era nueva, [`PuenteCuenta::veta_entradas`] es verdadero y
//!   el resincronizador reintenta. La reconexión, los eventos y las lecturas
//!   llevan su era; los de otra era se rechazan sin tocar nada. La publicación
//!   se hace dentro de la misma sección crítica ([`PuenteCuenta::con_era_vigente`]).
//! - **Evidencia incompleta (P1.3)**: una posición con cantidad o PnL ausente
//!   o no finito no se convierte en cero ni en cierre: se conserva la
//!   evidencia anterior, la equity queda incompleta (no sube sobre la última
//!   completa: las pérdidas pasan, las ganancias esperan) y se vetan las
//!   entradas hasta una lectura completa. El cero explícito sí se aplica.
//!
//! V4 (hallazgo N1 de la revisión independiente de 2c3b06225): dos
//! `ACCOUNT_UPDATE` con la misma `T` (dos fills de una orden en la misma
//! transacción) aplican el último si la marca vigente de ese componente la
//! puso el stream en la misma conexión. En REST el empate sigue siendo
//! contradicción; tras una reconexión o una REST con esa marca, también.
//! Una fila con la misma cantidad y marca más vieja no pisa el PnL no
//! realizado más nuevo.
//!
//! V4B (dictamen de Codex QRA sobre la propuesta de N1 y contrato
//! `qra_capital_v4b.md`): la generación del SOCKET es distinta de la era de
//! la cuenta.
//!
//! - Cada socket del stream tiene su número ([`PuenteCuenta::socket`]). La
//!   pérdida ([`PuenteCuenta::perdida_socket`], al salir del lazo de lectura)
//!   y la apertura de uno nuevo ([`PuenteCuenta::reconexion`]) suben el
//!   número, dejan la equity incompleta y vetan las entradas desde ese mismo
//!   instante. La procedencia del stream de una marca vale sólo para el
//!   socket que la puso; un evento de otro socket se rechaza
//!   ([`Rechazo::OtroSocket`]).
//! - Dentro de un socket se conservan las dos marcas del evento (`T` y `E`).
//!   Con la misma `T`: `E` mayor aplica, `E` igual con el mismo contenido se
//!   deduplica y con otro contenido es contradicción, `E` menor es una
//!   regresión y se descarta. Sin `E` vale el orden de entrega del socket.
//! - Un empate sin orden demostrable (REST contra stream, o tras reconectar)
//!   no se resuelve por llegada: rechaza, deja el componente en empate, veta
//!   las entradas y sólo lo resuelve una lectura REST con ticket posterior.
//! - El ticket de REST lleva era, socket y generación. Una lectura de otro
//!   socket (tomada antes del corte, o con un corte en medio) se rechaza
//!   entera. Sólo suelta el veto una lectura completa con ticket tomado
//!   después del último anclaje (corte, reconexión o rechazo que dejó la
//!   equity incompleta) y con el socket abierto. La cuenta quieta (mismo
//!   saldo y mismo `updateTime`) se confirma igual: no se compara con la hora
//!   local. Una foto retrasada no pisa el PnL no realizado de un componente
//!   observado después de tomar su ticket.
//!
//! El módulo sólo usa `std` (se compila solo en las pruebas de auditoría).
//! No toca d*, `capital_del_freno` ni ningún umbral.

use std::collections::HashMap;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Mutex, OnceLock};

/// (símbolo, `positionSide`).
pub type ClavePosicion = (String, String);

/// Lo que el puente publica: las dos magnitudes, coherentes entre sí.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Observacion {
    pub saldo: f64,
    pub equity: f64,
}

/// Lo que dice una fila de posición (REST o stream).
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum DatoPosicion {
    /// Cantidad distinta de cero y PnL no realizado, los dos finitos.
    Abierta { cantidad: f64, upnl: f64 },
    /// Cantidad cero.
    Cerrada,
    /// Cantidad o PnL ausente o no finito: no se sabe. No modifica nada y
    /// deja la equity incompleta.
    Invalida,
}

/// Una posición de una lectura REST, con su marca (`updateTime` de su fila).
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct PosicionRest {
    pub dato: DatoPosicion,
    pub tiempo_ms: Option<u64>,
}

/// Lectura REST de `/fapi/v2/account` (V3), cada componente con su marca. La
/// construye [`parse_account_lectura`](crate::execution_evidence::parse_account_lectura).
#[derive(Debug, Clone, PartialEq)]
pub struct LecturaCuenta {
    /// `walletBalance` de USDT.
    pub saldo: f64,
    /// `updateTime` de la fila de USDT (`None` si falta o es 0).
    pub tiempo_saldo_ms: Option<u64>,
    /// Posiciones con margen en USDT (abiertas y cerradas). `None` si la
    /// respuesta no trae `positions[]`: la equity queda incompleta y las
    /// posiciones ya observadas se conservan.
    pub posiciones: Option<HashMap<ClavePosicion, PosicionRest>>,
    /// Revisión independiente 3a: saldo de USDT 0 con fondos en otros activos
    /// o modo multi-activo. El cero se aplica al plano de seguridad, pero el
    /// arranque no lo acepta como capital con que operar.
    pub rechazo_arranque: Option<String>,
    /// `unrealizedProfit` de la fila de USDT (suma del PnL abierto de sus
    /// posiciones), si es finito. Sólo acota la equity cuando las posiciones
    /// vienen incompletas: así una posición nunca observada tampoco cuenta 0.
    pub upnl_del_activo: Option<f64>,
}

/// Forma V2 de la lectura: una sola marca compuesta (el `updateTime` más nuevo
/// de la fila de USDT y de las posiciones). Se acepta por compatibilidad,
/// pero esa marca no prueba la frescura del saldo ni de ninguna posición
/// (dictamen de Codex QRA, P1.1): cuenta como «sin orden».
#[derive(Debug, Clone, PartialEq)]
pub struct LecturaRest {
    pub saldo: f64,
    pub upnl: Option<HashMap<ClavePosicion, f64>>,
    pub upnl_del_activo: f64,
    pub tiempo_ms: Option<u64>,
}

impl LecturaRest {
    /// La equity de esta lectura (saldo + PnL no realizado).
    pub fn equity(&self) -> f64 {
        let u = match &self.upnl {
            Some(m) => m.values().sum(),
            None => self.upnl_del_activo,
        };
        let e = self.saldo + u;
        if e.is_finite() {
            e.max(0.0)
        } else {
            self.saldo
        }
    }
}

/// Lo que [`PuenteCuenta::observar_rest`] acepta.
pub trait FuenteRest {
    fn lectura_cuenta(&self) -> LecturaCuenta;
}

impl FuenteRest for LecturaCuenta {
    fn lectura_cuenta(&self) -> LecturaCuenta {
        self.clone()
    }
}

impl FuenteRest for LecturaRest {
    fn lectura_cuenta(&self) -> LecturaCuenta {
        LecturaCuenta {
            saldo: self.saldo,
            tiempo_saldo_ms: None,
            posiciones: self.upnl.as_ref().map(|m| {
                m.iter()
                    .map(|(k, u)| {
                        let dato = if u.is_finite() {
                            // Sin la cantidad, la posición no es comparable
                            // con una ya observada.
                            DatoPosicion::Abierta { cantidad: f64::NAN, upnl: *u }
                        } else {
                            DatoPosicion::Invalida
                        };
                        (k.clone(), PosicionRest { dato, tiempo_ms: None })
                    })
                    .collect()
            }),
            rechazo_arranque: None,
            upnl_del_activo: self.upnl_del_activo.is_finite().then_some(self.upnl_del_activo),
        }
    }
}

/// Un `ACCOUNT_UPDATE` ya filtrado al activo de margen.
#[derive(Debug, Clone, PartialEq, Default)]
pub struct EventoStream {
    /// `T` (o `E`) del exchange.
    pub tiempo_ms: Option<u64>,
    /// `wb` de USDT. `None` = la fila falta o no trae `wb` (no es un cero).
    pub saldo: Option<f64>,
    /// Posiciones con margen en USDT.
    pub posiciones: Vec<(ClavePosicion, DatoPosicion)>,
}

/// Se toma ANTES de pedir la lectura REST: ata la lectura a la era, al socket
/// del stream y al orden de adquisición (generación).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Ticket {
    era: u64,
    /// V4E (P1-c): sesión OMS a la que pertenece la era.
    sesion: u64,
    socket: u64,
    generacion: u64,
}

impl Ticket {
    pub fn era(&self) -> u64 {
        self.era
    }

    /// V4E (P1-c): sesión OMS (generación de `SessionToken`) de la era del ticket.
    pub fn sesion(&self) -> u64 {
        self.sesion
    }

    /// V4B: el socket del stream vigente al tomarlo.
    pub fn socket(&self) -> u64 {
        self.socket
    }

    /// Orden de adquisición: cuántos cambios había registrado el puente al
    /// tomarlo. Una lectura con generación mayor que la de una evidencia se
    /// pidió después de observarla.
    pub fn generacion(&self) -> u64 {
        self.generacion
    }
}

/// Por qué no se aceptó una observación.
#[derive(Debug, Clone, PartialEq)]
pub enum Rechazo {
    /// De otra sesión de cuenta (era anterior o posterior).
    OtraEra,
    /// Marca más vieja que la ya aceptada para ese componente, con otro valor.
    Atrasada {
        componente: String,
        tiempo_ms: Option<u64>,
        ultimo_ms: u64,
    },
    /// Misma marca que la ya aceptada, con otro valor: no se sobrescribe.
    Contradiccion { componente: String, tiempo_ms: u64 },
    /// Sin marca, con otro valor que el ya observado: no lo modifica.
    SinOrden { componente: String },
    /// V4B: de otro socket del stream (un evento del socket anterior, o una
    /// lectura REST con ticket tomado antes de un corte). No cambia nada.
    OtroSocket { socket: u64, vigente: u64 },
    /// V4E (P1-c): lectura de un ejecutor de otra sesión OMS que la de la era
    /// vigente. No cambia nada, no publica ni suelta el veto.
    OtraSesion { sesion: u64, vigente: u64 },
}

impl Rechazo {
    /// Nombre corto del motivo, para el log.
    pub fn motivo(&self) -> &'static str {
        match self {
            Rechazo::OtraEra => "OtraEra",
            Rechazo::OtraSesion { .. } => "OtraSesion",
            Rechazo::Atrasada { .. } => "Atrasada",
            Rechazo::Contradiccion { .. } => "Contradiccion",
            Rechazo::SinOrden { .. } => "SinOrden",
            Rechazo::OtroSocket { .. } => "OtroSocket",
        }
    }
}

impl std::fmt::Display for Rechazo {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Rechazo::OtraEra => write!(f, "observación de otra sesión de cuenta (era)"),
            Rechazo::OtraSesion { sesion, vigente } => write!(
                f,
                "lectura de un ejecutor de la sesión OMS {sesion}, vigente {vigente}: no cambia nada"
            ),
            Rechazo::OtroSocket { socket, vigente } => write!(
                f,
                "observación del socket {socket}, vigente {vigente}: anterior al corte; no cambia nada"
            ),
            Rechazo::Atrasada { componente, tiempo_ms, ultimo_ms } => write!(
                f,
                "{componente}: observación atrasada (marca {}, última aceptada {ultimo_ms})",
                tiempo_ms.map_or_else(|| "N/D".to_string(), |t| t.to_string())
            ),
            Rechazo::Contradiccion { componente, tiempo_ms } => write!(
                f,
                "{componente}: contradicción con la misma marca {tiempo_ms}; no se sobrescribe"
            ),
            Rechazo::SinOrden { componente } => write!(
                f,
                "{componente}: dato sin marca de tiempo distinto del ya observado; no lo modifica"
            ),
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Veredicto {
    Aplicar,
    Igual,
}

/// V4B: cómo se ordena un dato con la MISMA marca `T` que la vigente.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Empate {
    /// Orden verificable: el dato es posterior (misma conexión con `E` mayor
    /// o, sin `E`, entregado después; o una REST con ticket posterior a un
    /// empate ya señalado). Se aplica.
    Posterior,
    /// Mismas marcas o sin orden demostrable: igual = duplicado; distinto =
    /// contradicción (pide reconciliación).
    SinOrden,
    /// Misma conexión con `E` menor: regresión, se descarta.
    Anterior,
}

/// Regla única de frescura de un componente.
fn decidir(
    componente: impl Fn() -> String,
    observado: bool,
    igual: bool,
    t: Option<u64>,
    marca: Option<u64>,
    empate: Empate,
) -> Result<Veredicto, Rechazo> {
    if !observado {
        return Ok(Veredicto::Aplicar);
    }
    match (t, marca) {
        (Some(t), Some(m)) if t > m || (t == m && empate == Empate::Posterior) => Ok(Veredicto::Aplicar),
        (Some(_), None) => Ok(Veredicto::Aplicar),
        _ if igual => Ok(Veredicto::Igual),
        (Some(t), Some(m)) if t == m && empate == Empate::SinOrden => Err(Rechazo::Contradiccion {
            componente: componente(),
            tiempo_ms: t,
        }),
        (Some(t), Some(m)) => Err(Rechazo::Atrasada {
            componente: componente(),
            tiempo_ms: Some(t),
            ultimo_ms: m,
        }),
        (None, _) => Err(Rechazo::SinOrden {
            componente: componente(),
        }),
    }
}

fn mismo_saldo(a: f64, b: f64) -> bool {
    (a - b).abs() <= 1e-9 * a.abs().max(b.abs()).max(1.0)
}

fn maxima(a: Option<u64>, b: Option<u64>) -> Option<u64> {
    match (a, b) {
        (Some(x), Some(y)) => Some(x.max(y)),
        (x, y) => x.or(y),
    }
}

/// V4C (F2): procedencia del stream del socket `socket` con hora `hora`, sin
/// olvidar la `E` mayor ya vista en ese socket (un evento sin `E` no la borra).
fn con_e_mayor(previa: Option<Origen>, socket: u64, hora: Option<u64>) -> Option<Origen> {
    let vista = match previa {
        Some(Origen::Stream { socket: s, e }) if s == socket => e,
        _ => None,
    };
    Some(Origen::Stream {
        socket,
        e: maxima(hora, vista),
    })
}

/// V4B: quién puso la marca vigente de un componente.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Origen {
    Rest,
    /// El stream, en el socket `socket`, con la hora de evento `e` (la mayor
    /// vista con esa `T`).
    Stream { socket: u64, e: Option<u64> },
}

/// V4B: evidencia de un componente (el saldo o una posición) además de su
/// valor y su marca.
#[derive(Debug, Clone, Copy, Default)]
struct Evidencia {
    origen: Option<Origen>,
    /// Generación en que se escribió por última vez (orden de adquisición).
    generacion: u64,
    /// Empate sin orden pendiente: desde esta generación (sólo lo resuelve una
    /// REST con ticket de generación igual o mayor y del socket abierto).
    empate_desde: Option<u64>,
}

impl Evidencia {
    /// Cómo ordena el stream (socket `socket`, hora `e`) un dato con la misma `T`.
    fn empate_stream(&self, socket: u64, e: Option<u64>) -> Empate {
        if self.empate_desde.is_some() {
            return Empate::SinOrden;
        }
        match self.origen {
            Some(Origen::Stream { socket: s, e: previa }) if s == socket => match (e, previa) {
                (Some(e), Some(p)) if e > p => Empate::Posterior,
                (Some(e), Some(p)) if e == p => Empate::SinOrden,
                (Some(_), Some(_)) => Empate::Anterior,
                // Sin `E`: la conexión entrega en orden.
                _ => Empate::Posterior,
            },
            _ => Empate::SinOrden,
        }
    }

    /// Cómo ordena una REST con `ticket` un dato con la misma `T`.
    fn empate_rest(&self, ticket: &Ticket, anclada: bool) -> Empate {
        match self.empate_desde {
            Some(g) if anclada && ticket.generacion >= g => Empate::Posterior,
            _ => Empate::SinOrden,
        }
    }
}

#[derive(Debug, Default)]
struct Diagnostico {
    /// (fuente, motivo) → rechazos desde la última toma.
    rechazos: HashMap<(&'static str, &'static str), u64>,
    ultimo: Option<String>,
}

#[derive(Debug, Default)]
struct Estado {
    era: u64,
    /// V4E (P1-c): sesión OMS de la era (generación de `SessionToken`). Una
    /// lectura de otra sesión se rechaza ([`Rechazo::OtraSesion`]).
    sesion: u64,
    /// Cambios registrados (observaciones aceptadas, cortes, anclajes).
    generacion: u64,
    /// V4B: número del socket del stream vigente.
    socket: u64,
    /// V4B: `false` entre la pérdida del socket y la apertura del siguiente.
    socket_cerrado: bool,
    /// V4B: generación del último anclaje. Una REST sólo suelta el veto con
    /// ticket de generación igual o mayor.
    ancla: u64,
    saldo: Option<f64>,
    marca_saldo: Option<u64>,
    ev_saldo: Evidencia,
    /// V4C: último saldo que el stream de esta era entregó, con su `T` y su
    /// `E`, sólo mientras su orden sea demostrable (ver
    /// [`Estado::registrar_saldo_stream`]). Una REST con esa misma `T` y otro
    /// valor puede ser una foto atrasada: no cierra un empate ni suelta el veto.
    ultimo_saldo_stream: Option<(u64, Option<u64>, f64)>,
    /// V4E (INV-7): saldos que el stream entregó con la misma `T` que
    /// `ultimo_saldo_stream`, con su `E`, y que se rechazaron por contradicción
    /// (evidencia pendiente). No son testigo: sólo dejan que la REST posterior
    /// arbitre a favor de uno de ellos (contrato V4B y F2 de V4C). Se vacían
    /// cuando cambia el último saldo aceptado.
    candidatos_stream: Vec<(Option<u64>, f64)>,
    /// V4C (F2): generación en que se aceptó el último evento del stream. Una
    /// REST sólo puede soltar el veto o cerrar un empate con un ticket
    /// posterior (pedida después de ese evento).
    ultimo_evento: Option<u64>,
    /// V4D (P1-b): la última REST aceptada trajo el rechazo 3a
    /// ([`LecturaCuenta::rechazo_arranque`]). Veta las entradas hasta una REST
    /// sin rechazo o un saldo positivo aplicado (el 3a exige saldo cero).
    rechazo_3a: bool,
    /// Posiciones abiertas observadas: (cantidad, PnL no realizado).
    abiertas: HashMap<ClavePosicion, (f64, f64)>,
    /// Marca de cada posición observada (abierta o cerrada).
    marcas_pos: HashMap<ClavePosicion, Option<u64>>,
    ev_pos: HashMap<ClavePosicion, Evidencia>,
    completa: bool,
    /// Mientras la equity está incompleta: la última completa (techo).
    techo: Option<f64>,
    /// Equity de la era anterior (sólo para el freno, nunca para entradas).
    anterior: Option<f64>,
    /// Por qué está incompleta (diagnóstico).
    motivo: Option<&'static str>,
    diag: Diagnostico,
}

impl Estado {
    fn equity_cruda(&self) -> Option<f64> {
        let s = self.saldo?;
        let e = s + self.abiertas.values().map(|(_, u)| *u).sum::<f64>();
        Some(if e.is_finite() { e.max(0.0) } else { s })
    }

    fn observacion(&self) -> Option<Observacion> {
        let saldo = self.saldo?;
        let mut equity = self.equity_cruda()?;
        if !self.completa {
            if let Some(t) = self.techo {
                equity = equity.min(t);
            }
        }
        Some(Observacion { saldo, equity })
    }

    /// V4D (P1-b): el cero del caso 3a (USDT en cero con fondos en otro
    /// activo o modo multi-activo) también veta. El cero llega igual al freno
    /// (la observación no cambia).
    fn veta_entradas(&self) -> bool {
        self.necesita_lectura() || self.rechazo_3a
    }

    /// Sin saldo observado en la era o con la equity incompleta: hace falta
    /// una REST (el caso 3a no la pide: es un dato completo).
    fn necesita_lectura(&self) -> bool {
        self.saldo.is_none() || !self.completa
    }

    fn motivo_veto(&self) -> Option<&'static str> {
        if self.saldo.is_none() {
            Some("SinSaldo")
        } else if self.completa && self.rechazo_3a {
            Some("RechazoArranque")
        } else if !self.completa {
            Some(self.motivo.unwrap_or("Incompleta"))
        } else {
            None
        }
    }

    /// Deja la equity incompleta con techo en la equity actual.
    fn incompleta(&mut self, motivo: &'static str) {
        let actual = self.observacion().map(|o| o.equity);
        self.techo = match (self.techo, actual) {
            (Some(t), Some(a)) => Some(t.min(a)),
            (t, a) => t.or(a),
        };
        self.completa = false;
        self.motivo = Some(motivo);
    }

    /// V4B: incompleta y anclaje nuevo; devuelve la generación del anclaje.
    fn anclar(&mut self, motivo: &'static str) -> u64 {
        self.incompleta(motivo);
        self.generacion += 1;
        self.ancla = self.generacion;
        self.ancla
    }

    fn registrar(&mut self, fuente: &'static str, t: Option<u64>, e: Option<u64>, r: &Rechazo) {
        *self.diag.rechazos.entry((fuente, r.motivo())).or_insert(0) += 1;
        let marca = |x: Option<u64>| x.map_or_else(|| "N/D".to_string(), |v| v.to_string());
        self.diag.ultimo = Some(format!(
            "fuente={fuente} T={} E={} motivo={}: {r}",
            marca(t),
            marca(e),
            r.motivo()
        ));
    }

    /// V4C: registra el saldo `s` que entregó el stream con `T = t` y `E =
    /// hora`. `aceptado`: el puente lo aplicó o lo confirmó como igual;
    /// si no, fue una contradicción. Con la misma `T` sólo avanza con `E`
    /// mayor (la hora de evento de Binance, que una reemisión conserva); con
    /// `E` igual u omitida y otro valor no hay orden y se olvida (arbitra la
    /// REST con ticket posterior, como en V4B); con `E` menor (una reemisión
    /// vieja) no cambia nada.
    ///
    /// V4E (INV-7): un saldo RECHAZADO nunca reemplaza ni borra el último
    /// aceptado (antes, con la misma `T` y `E` igual u omitida, lo borraba y una
    /// foto atrasada con esa `T` volvía a cerrar el empate). Si trae la misma
    /// `T` que el último aceptado y otro valor, queda como candidato aparte.
    fn registrar_saldo_stream(&mut self, t: Option<u64>, hora: Option<u64>, s: f64, aceptado: bool) {
        let Some(t) = t else { return };
        if !aceptado {
            if let Some((ts, es, vs)) = self.ultimo_saldo_stream {
                // Una reemisión vieja (`E` menor que la del aceptado) no cuenta.
                let vieja = matches!((hora, es), (Some(h), Some(p)) if h < p);
                let repetido = self.candidatos_stream.iter().any(|&(e, c)| e == hora && mismo_saldo(c, s));
                if ts == t && !vieja && !mismo_saldo(vs, s) && !repetido {
                    self.candidatos_stream.push((hora, s));
                }
            }
            return;
        }
        let antes = self.ultimo_saldo_stream;
        match self.ultimo_saldo_stream {
            Some((ts, _, _)) if ts > t => {}
            Some((ts, es, vs)) if ts == t => match (hora, es) {
                (Some(h), Some(p)) if h > p => self.ultimo_saldo_stream = Some((t, hora, s)),
                (Some(h), Some(p)) if h < p => {}
                _ if mismo_saldo(vs, s) => {}
                _ => self.ultimo_saldo_stream = Some((t, hora.or(es), s)),
            },
            _ => self.ultimo_saldo_stream = Some((t, hora, s)),
        }
        let cambio = match (antes, self.ultimo_saldo_stream) {
            (Some((ta, _, va)), Some((tn, _, vn))) => ta != tn || !mismo_saldo(va, vn),
            _ => true,
        };
        if cambio {
            self.candidatos_stream.clear();
        }
    }

    /// V4E (INV-7): `s` es uno de los saldos que el stream entregó con la `T`
    /// del último aceptado y que pueden ser el último: los de la `E` mayor vista
    /// para esa `T` (el aceptado o un candidato rechazado) y los que no traen
    /// `E` (sin orden frente a los demás). Un saldo de una `E` anterior es una
    /// foto atrasada.
    fn saldo_del_stream_en_esa_marca(&self, s: f64) -> bool {
        let Some((_, es, vs)) = self.ultimo_saldo_stream else {
            return false;
        };
        let mayor = self.candidatos_stream.iter().filter_map(|&(e, _)| e).chain(es).max();
        let vale = |e: Option<u64>, v: f64| mismo_saldo(v, s) && (e.is_none() || e == mayor);
        vale(es, vs) || self.candidatos_stream.iter().any(|&(e, v)| vale(e, v))
    }

    /// V4C (F3): ante una contradicción, el techo del freno baja a la equity
    /// con el saldo menor y, por posición, el PnL no realizado menor entre lo
    /// vigente y lo que trae la lectura rechazada (`filas`; `completa`: una
    /// posición abierta que no viene se toma por cerrada). Una fila inválida
    /// no aporta. Sólo baja el techo: no escribe saldo, cantidad ni marcas.
    fn bajar_techo_en_conflicto(&mut self, saldo: Option<f64>, filas: &[(ClavePosicion, DatoPosicion)], completa: bool) {
        let Some(s) = (match (self.saldo, saldo) {
            (Some(a), Some(b)) => Some(a.min(b)),
            (a, b) => a.or(b),
        }) else {
            return;
        };
        let vigente = |c: &ClavePosicion| self.abiertas.get(c).map(|(_, u)| *u).filter(|u| u.is_finite());
        let mut upnl = 0.0;
        for (clave, u) in &self.abiertas {
            if !u.1.is_finite() {
                continue;
            }
            let otra = match filas.iter().find(|(c, _)| c == clave).map(|(_, d)| d) {
                Some(DatoPosicion::Abierta { upnl, .. }) if upnl.is_finite() => *upnl,
                Some(DatoPosicion::Cerrada) => 0.0,
                None if completa => 0.0,
                _ => u.1,
            };
            upnl += u.1.min(otra);
        }
        for (clave, dato) in filas {
            if let (DatoPosicion::Abierta { upnl: u, .. }, None) = (dato, vigente(clave)) {
                if u.is_finite() && !self.abiertas.contains_key(clave) {
                    upnl += u.min(0.0);
                }
            }
        }
        let cota = (s + upnl).max(0.0);
        if cota.is_finite() {
            self.techo = Some(self.techo.map_or(cota, |t| t.min(cota)));
        }
    }

    fn evidencia_pos(&self, clave: &ClavePosicion) -> Evidencia {
        self.ev_pos.get(clave).copied().unwrap_or_default()
    }

    /// Veredicto de una posición contra lo observado.
    fn decidir_posicion(
        &self,
        clave: &ClavePosicion,
        dato: &DatoPosicion,
        t: Option<u64>,
        empate: Empate,
    ) -> Result<Veredicto, Rechazo> {
        let observado = self.marcas_pos.contains_key(clave) || self.abiertas.contains_key(clave);
        let igual = match (dato, self.abiertas.get(clave)) {
            (DatoPosicion::Abierta { cantidad, .. }, Some((c, _))) => mismo_saldo(*cantidad, *c),
            (DatoPosicion::Cerrada, None) => observado,
            _ => false,
        };
        let marca = self.marcas_pos.get(clave).copied().flatten();
        decidir(|| format!("posición {} {}", clave.0, clave.1), observado, igual, t, marca, empate)
    }

    /// Aplica según el veredicto. Un `Igual` sólo confirma la cantidad salvo
    /// que el dato sea posterior a la evidencia vigente: marca vigente
    /// ausente, o la misma marca con orden verificable (`posterior`). Así una
    /// fila con marca más vieja, un repetido tras reconectar o una foto
    /// pedida antes de la evidencia no pisan el PnL no realizado más nuevo.
    ///
    /// V4C (F3): en ese caso, si NO hay orden demostrable entre las dos
    /// lecturas del PnL no realizado (`anterior_demostrable == false`; p. ej.
    /// la fila REST trae la marca del último cambio de cantidad, o la marca
    /// la puso la otra fuente), el freno ve la MENOR de las dos: las pérdidas
    /// pasan, las ganancias esperan a una lectura con orden. Si la lectura es
    /// demostrablemente anterior (mismo socket, `E` o `T` menor), se descarta.
    #[allow(clippy::too_many_arguments)]
    fn aplicar_posicion_con(
        &mut self,
        clave: &ClavePosicion,
        dato: &DatoPosicion,
        t: Option<u64>,
        v: &Veredicto,
        posterior: bool,
        anterior_demostrable: bool,
        origen: Option<Origen>,
    ) {
        let marca = self.marcas_pos.get(clave).copied().flatten();
        if matches!(v, Veredicto::Igual) && marca.is_some() && (t < marca || (t == marca && !posterior)) {
            if !anterior_demostrable {
                if let (DatoPosicion::Abierta { upnl, .. }, Some(a)) = (dato, self.abiertas.get_mut(clave)) {
                    if upnl.is_finite() && (*upnl < a.1 || !a.1.is_finite()) {
                        a.1 = *upnl;
                    }
                }
            }
            return;
        }
        if matches!(dato, DatoPosicion::Invalida) {
            return;
        }
        self.aplicar_posicion(clave, dato, t);
        let g = self.generacion;
        let ev = self.ev_pos.entry(clave.clone()).or_default();
        ev.generacion = g;
        if matches!(v, Veredicto::Aplicar) {
            ev.empate_desde = None;
            if t.is_some() {
                ev.origen = origen;
            }
        }
    }

    fn aplicar_posicion(&mut self, clave: &ClavePosicion, dato: &DatoPosicion, t: Option<u64>) {
        match dato {
            DatoPosicion::Abierta { cantidad, upnl } => {
                let cantidad = if cantidad.is_finite() {
                    *cantidad
                } else {
                    self.abiertas.get(clave).map_or(f64::NAN, |(c, _)| *c)
                };
                self.abiertas.insert(clave.clone(), (cantidad, *upnl));
            }
            DatoPosicion::Cerrada => {
                self.abiertas.remove(clave);
            }
            DatoPosicion::Invalida => return,
        }
        let previa = self.marcas_pos.get(clave).copied().flatten();
        self.marcas_pos.insert(clave.clone(), maxima(previa, t));
    }
}

/// Lo que el host escribe cada 60 s (solo medida).
#[derive(Debug, Clone, Default, PartialEq)]
pub struct DiagnosticoPuente {
    pub veta: bool,
    /// Por qué veta ahora (`SinSaldo`, `Reconexion`, `PerdidaSocket`,
    /// `Contradiccion`, `SinOrden`, `EvidenciaInvalida`, `EventoIlegible`).
    pub motivo_veto: Option<&'static str>,
    /// Rechazos desde la última toma: (fuente, motivo, cuántos), ordenados.
    pub rechazos: Vec<(&'static str, &'static str, u64)>,
    /// El último rechazo, con fuente, `T`, `E` y motivo.
    pub ultimo_rechazo: Option<String>,
}

type Observador = Box<dyn Fn(bool) + Send + Sync>;
type Publicador = Box<dyn Fn(&Publicacion) -> bool + Send + Sync>;
/// V4F (P1-c): la generación vigente del `SessionToken` del OMS.
type Testigo = Box<dyn Fn() -> u64 + Send + Sync>;

/// V4E (P1-b, INV-5): lo que el puente entrega a su publicador tras cada
/// cambio, bajo su mutex.
///
/// Contrato del publicador (el host): con `veta`, alzar el veto. Sin `veta` y
/// con `saldo`, es el ALZA: publicar ese saldo dimensionable y DESPUÉS, en la
/// misma llamada, soltar el veto (`Release`); devolver `true` sólo si lo soltó
/// (p. ej. `false` si la sesión OMS ya no es `sesion`: el veto sigue y el
/// puente vuelve a ofrecer el alza en el cambio siguiente). Sin `veta` y sin
/// `saldo`, el veto ya estaba suelto. V4G (INV-B): sin `veta`, `vigente` trae
/// siempre el saldo aceptado y su marca, y el publicador lo entrega al saldo
/// dimensionable en esta misma llamada (o veta si no puede). No debe reentrar
/// en el puente.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Publicacion {
    pub era: u64,
    /// Sesión OMS de la era (P1-c): el publicador sólo publica en ella.
    pub sesion: u64,
    pub veta: bool,
    /// `Some` sólo en el alza.
    pub saldo: Option<f64>,
    /// V4G (INV-B): el saldo aceptado vigente de la era y su marca del exchange, en CADA cambio
    /// en que el puente no veta (también con la admisión ya abierta). El publicador lo entrega al
    /// dimensionable en este mismo paso; si no puede (otra sesión OMS), veta y devuelve `false`,
    /// y el puente lo vuelve a ofrecer en el cambio siguiente.
    pub vigente: Option<(f64, Option<u64>)>,
}

/// El puente. Un solo `Mutex`: las decisiones de REST y del stream se
/// serializan, y lo publicado sale de la misma sección crítica.
pub struct PuenteCuenta {
    estado: Mutex<Estado>,
    /// V4E: observador de SOLO veto. Recibe `true` en cada cambio en que el
    /// puente veta y nunca `false`: soltar el veto exige publicar antes el
    /// saldo dimensionable (INV-5), y este observador no lo publica.
    veto: OnceLock<Observador>,
    /// V4E (P1-b): el publicador del saldo dimensionable y del veto.
    publicador: OnceLock<Publicador>,
    /// V4F (P1-c): testigo de la sesión OMS vigente. Se consulta bajo el mutex
    /// del puente, en el mismo paso que cada escritura ([`Self::vigente`]).
    testigo: OnceLock<Testigo>,
    /// V4E: el publicador soltó el veto en su última llamada (sólo se escribe
    /// bajo el mutex de `estado`).
    admitido: AtomicBool,
    /// CAPITAL-UNICO V4J (punto 5): (cuenta de la era vigente, cuenta anunciada
    /// para la era siguiente). Una era de OTRA cuenta no hereda la equity
    /// anterior para el freno. Candado propio (se toma después de `estado`).
    cuentas: Mutex<(Option<u64>, Option<u64>)>,
}

impl std::fmt::Debug for PuenteCuenta {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("PuenteCuenta").field("estado", &*self.estado()).finish_non_exhaustive()
    }
}

impl Default for PuenteCuenta {
    fn default() -> Self {
        Self::new()
    }
}

impl PuenteCuenta {
    /// Era 1 de la sesión OMS 0, sin saldo observado: veta entradas hasta la
    /// primera lectura.
    pub fn new() -> Self {
        Self::con_sesion(0)
    }

    /// V4E (P1-c): era 1 ligada a la sesión OMS `sesion` (la generación del
    /// `SessionToken` vigente al arrancar).
    pub fn con_sesion(sesion: u64) -> Self {
        Self {
            estado: Mutex::new(Estado {
                era: 1,
                sesion,
                completa: false,
                ..Estado::default()
            }),
            veto: OnceLock::new(),
            publicador: OnceLock::new(),
            testigo: OnceLock::new(),
            admitido: AtomicBool::new(false),
            cuentas: Mutex::new((None, None)),
        }
    }

    /// CAPITAL-UNICO V4J (punto 5): la cuenta (identidad de venue y clave
    /// pública, la misma que fija el libro) de la era que abra el próximo relevo,
    /// o de la era vigente si aún no tiene. El host la anuncia al arrancar y
    /// dentro del relevo (no toma el mutex de `estado`: se puede llamar desde el
    /// cierre de [`relevar_sesion`](Self::relevar_sesion)).
    pub fn anunciar_cuenta(&self, cuenta: u64) {
        let mut c = self.cuentas.lock().unwrap_or_else(|p| p.into_inner());
        c.1 = Some(cuenta);
        if c.0.is_none() {
            c.0 = Some(cuenta);
        }
    }

    fn estado(&self) -> std::sync::MutexGuard<'_, Estado> {
        self.estado.lock().unwrap_or_else(|p| p.into_inner())
    }

    /// V4F (P1-c): EL punto por el que pasa toda escritura del puente (saldo,
    /// equity del freno, posiciones, veto, socket, generación) y toda
    /// publicación al espejo con [`con_era_vigente`](Self::con_era_vigente).
    /// Se llama con el mutex tomado y la escritura sigue en esa misma sección
    /// crítica: entre comprobar y escribir no hay hueco.
    ///
    /// - `era` debe ser la vigente (`OtraEra`).
    /// - Con testigo vinculado, la sesión OMS vigente debe ser la de la era
    ///   (`OtraSesion { sesion: la de la era, vigente: la del OMS }`). Es la
    ///   ventana entre el relevo del OMS y el cambio de era: ahí nada de la
    ///   sesión anterior escribe. El host la cierra además con
    ///   [`relevar_sesion`](Self::relevar_sesion); el testigo cubre cualquier
    ///   otro relevo.
    fn vigente(&self, e: &Estado, era: u64) -> Result<(), Rechazo> {
        if era != e.era {
            return Err(Rechazo::OtraEra);
        }
        match self.testigo.get() {
            Some(testigo) => {
                let oms = testigo();
                if oms == e.sesion {
                    Ok(())
                } else {
                    Err(Rechazo::OtraSesion {
                        sesion: e.sesion,
                        vigente: oms,
                    })
                }
            }
            None => Ok(()),
        }
    }

    /// V4F (P1-c): vincula el testigo de la sesión OMS (la generación del
    /// `SessionToken` vigente). Corre bajo el mutex del puente (orden de
    /// candados: puente y después la sesión OMS, el mismo del publicador) y no
    /// debe reentrar en él. Sólo el primero vale.
    pub fn vincular_testigo_de_sesion(&self, f: impl Fn() -> u64 + Send + Sync + 'static) {
        let _e = self.estado();
        let _ = self.testigo.set(Box::new(f));
    }

    /// V4F (P1-c): relevo de la sesión OMS y era nueva del puente en un solo
    /// paso, bajo el mutex del puente. `relevo` publica el ejecutor nuevo (en el
    /// host, `OmsSession::paso_transicion`) y devuelve la generación publicada,
    /// o `None` si no publicó. Con `Some(sesion)`, sin soltar el mutex, se abre
    /// la era de esa sesión y nace sin socket privado (como
    /// [`sin_socket`](Self::sin_socket)). Ninguna lectura ni evento de la sesión
    /// anterior se cuela entre el relevo y el cambio de era: espera al mutex y
    /// después encuentra otra era. Devuelve lo de `relevo` y la era nueva.
    /// `relevo` no debe reentrar en el puente.
    pub fn relevar_sesion<R>(&self, relevo: impl FnOnce() -> (R, Option<u64>)) -> (R, Option<u64>) {
        let mut e = self.estado();
        let (r, sesion) = relevo();
        let era = sesion.map(|sesion| {
            let era = self.abrir_era(&mut e, sesion);
            Self::cerrar_socket(&mut e, "SinSocket");
            self.notificar(&e);
            era
        });
        (r, era)
    }

    fn cerrar_socket(e: &mut Estado, motivo: &'static str) {
        e.socket += 1;
        e.socket_cerrado = true;
        e.anclar(motivo);
    }

    fn notificar(&self, e: &Estado) {
        let veta = e.veta_entradas();
        if veta {
            if let Some(f) = self.veto.get() {
                f(true);
            }
        }
        let Some(publicar) = self.publicador.get() else {
            return;
        };
        let alza = !veta && !self.admitido.load(Ordering::Relaxed);
        let admite = publicar(&Publicacion {
            era: e.era,
            sesion: e.sesion,
            veta,
            saldo: if alza { e.saldo } else { None },
            vigente: if veta { None } else { e.saldo.map(|s| (s, e.marca_saldo)) },
        });
        self.admitido.store(!veta && admite, Ordering::Relaxed);
    }

    /// V4E: vincula un observador de SOLO veto. Se llama en seguida si el
    /// puente veta, y después en cada cambio en que vete, bajo el mutex del
    /// puente. Nunca recibe `false` (ver [`Publicacion`]). Sólo el primero vale.
    pub fn vincular_veto(&self, f: impl Fn(bool) + Send + Sync + 'static) {
        let e = self.estado();
        if self.veto.set(Box::new(f)).is_ok() && e.veta_entradas() {
            if let Some(f) = self.veto.get() {
                f(true);
            }
        }
    }

    /// V4E (P1-b, INV-5): vincula el publicador del saldo dimensionable y del
    /// veto (contrato en [`Publicacion`]). Se llama en seguida con el estado
    /// actual y después tras cada cambio, bajo el mutex del puente. Sólo el
    /// primero vale.
    pub fn vincular_publicador(&self, f: impl Fn(&Publicacion) -> bool + Send + Sync + 'static) {
        let e = self.estado();
        if self.publicador.set(Box::new(f)).is_ok() {
            self.admitido.store(false, Ordering::Relaxed);
            self.notificar(&e);
        }
    }

    /// V4E (P1-c): sesión OMS de la era vigente.
    pub fn sesion(&self) -> u64 {
        self.estado().sesion
    }

    /// Era vigente (la del streamer que la reciba).
    pub fn era(&self) -> u64 {
        self.estado().era
    }

    /// V4B: número del socket del stream vigente.
    pub fn socket(&self) -> u64 {
        self.estado().socket
    }

    /// Sesión de cuenta nueva (transición): el saldo queda sin observar, la
    /// equity incompleta y las entradas vetadas hasta una lectura aceptada de
    /// esta era. Lo que llegue con otra era se rechaza. La equity de la era
    /// anterior queda sólo como evidencia para el freno. Devuelve la era nueva.
    ///
    /// V4E (P1-c): la era nueva es de la sesión OMS siguiente (la que publica
    /// `replace_executor`: generación + 1). El host usa
    /// [`nueva_era_de_sesion`](Self::nueva_era_de_sesion) con la generación
    /// publicada.
    pub fn nueva_era(&self) -> u64 {
        let mut e = self.estado();
        let sesion = e.sesion + 1;
        self.abrir_era(&mut e, sesion)
    }

    /// V4E (P1-c): como [`nueva_era`](Self::nueva_era), ligada a la sesión OMS
    /// `sesion` (la generación de `ResultadoTransicion::Publicada`).
    pub fn nueva_era_de_sesion(&self, sesion: u64) -> u64 {
        let mut e = self.estado();
        self.abrir_era(&mut e, sesion)
    }

    fn abrir_era(&self, e: &mut Estado, sesion: u64) -> u64 {
        let era = e.era + 1;
        // V4J (punto 5): la equity de la era anterior solo vale como evidencia
        // conservada para el freno de la MISMA cuenta.
        let misma_cuenta = {
            let mut c = self.cuentas.lock().unwrap_or_else(|p| p.into_inner());
            let misma = c.0.is_none() || c.1.is_none() || c.0 == c.1;
            if c.1.is_some() {
                c.0 = c.1;
            }
            misma
        };
        let anterior = if misma_cuenta {
            e.observacion().map(|o| o.equity).or(e.anterior)
        } else {
            None
        };
        // El número de socket sigue subiendo: nada de la era anterior coincide.
        let socket = e.socket + 1;
        *e = Estado {
            era,
            sesion,
            socket,
            completa: false,
            anterior,
            ..Estado::default()
        };
        self.notificar(e);
        era
    }

    /// Ticket de la era vigente: tomarlo ANTES de pedir la lectura.
    pub fn ticket(&self) -> Ticket {
        let e = self.estado();
        Ticket {
            era: e.era,
            sesion: e.sesion,
            socket: e.socket,
            generacion: e.generacion,
        }
    }

    /// Ticket de `era`, si sigue vigente.
    pub fn ticket_de_era(&self, era: u64) -> Option<Ticket> {
        let e = self.estado();
        (e.era == era).then_some(Ticket {
            era,
            sesion: e.sesion,
            socket: e.socket,
            generacion: e.generacion,
        })
    }

    /// Corre `f` bajo el mutex del puente si `era` sigue vigente: ninguna
    /// transición se cuela entre la comprobación y la publicación.
    ///
    /// V4F (P1-c): con el mismo punto que las escrituras ([`Self::vigente`]):
    /// si la sesión OMS ya no es la de la era, `f` no corre (el espejo no se
    /// escribe con lo de la sesión anterior).
    pub fn con_era_vigente<R>(&self, era: u64, f: impl FnOnce(Option<Observacion>) -> R) -> Option<R> {
        let e = self.estado();
        self.vigente(&e, era).ok().map(|()| f(e.observacion()))
    }

    /// Lectura REST. Entera o nada: si un componente se rechaza, no cambia
    /// nada (salvo, en una contradicción, dejar el empate señalado y las
    /// entradas vetadas).
    ///
    /// V4E (P1-c): equivale a [`observar_rest_de_sesion`](Self::observar_rest_de_sesion)
    /// con la sesión del ticket (quien llama declara que el ejecutor es de esa
    /// sesión). Los caminos de producción usan la forma con testigo.
    pub fn observar_rest<L: FuenteRest + ?Sized>(
        &self,
        ticket: Ticket,
        lectura: &L,
    ) -> Result<Observacion, Rechazo> {
        self.observar_rest_de_sesion(ticket, ticket.sesion, lectura)
    }

    /// V4E (P1-c, INV-6): lectura REST hecha con el ejecutor de la sesión OMS
    /// `sesion` (la generación del `SessionToken` del que salió el ejecutor).
    /// Si no es la sesión de la era vigente, se rechaza bajo el mutex sin tocar
    /// nada ([`Rechazo::OtraSesion`]): ni autoriza ni publica.
    pub fn observar_rest_de_sesion<L: FuenteRest + ?Sized>(
        &self,
        ticket: Ticket,
        sesion: u64,
        lectura: &L,
    ) -> Result<Observacion, Rechazo> {
        let l = lectura.lectura_cuenta();
        let mut e = self.estado();
        if ticket.era != e.era {
            return Err(Rechazo::OtraEra);
        }
        let fuente = "rest";
        if sesion != e.sesion || ticket.sesion != e.sesion {
            let r = Rechazo::OtraSesion {
                sesion,
                vigente: e.sesion,
            };
            e.registrar(fuente, l.tiempo_saldo_ms, None, &r);
            return Err(r);
        }
        // V4F (P1-c): el lote entero se decide aquí, con el testigo del OMS y
        // bajo el mismo mutex que lo escribe. Tras el relevo del OMS, una
        // lectura de la sesión anterior no toca saldo, equity, veto ni espejo.
        if let Err(r) = self.vigente(&e, ticket.era) {
            e.registrar(fuente, l.tiempo_saldo_ms, None, &r);
            return Err(r);
        }
        // V4B: una lectura pedida antes de un corte no cubre los eventos que
        // ese corte pudo perder.
        if ticket.socket != e.socket {
            let r = Rechazo::OtroSocket {
                socket: ticket.socket,
                vigente: e.socket,
            };
            e.registrar(fuente, l.tiempo_saldo_ms, None, &r);
            return Err(r);
        }
        // Puede soltar el veto y resolver un empate: ticket del socket abierto
        // y posterior al último anclaje.
        // V4C (F2): y pedida después del último evento aceptado del stream.
        let anclada = !e.socket_cerrado
            && ticket.generacion >= e.ancla
            && e.ultimo_evento.map_or(true, |g| ticket.generacion > g);
        // V4C (F3): las filas de esta lectura, para la cota del freno si se
        // rechaza.
        let filas: Vec<(ClavePosicion, DatoPosicion)> = l
            .posiciones
            .iter()
            .flatten()
            .map(|(c, p)| (c.clone(), p.dato))
            .collect();
        let completa_l = l.posiciones.is_some();
        // Contradicción de REST: se señala el empate del componente y se veta;
        // el freno ve el saldo menor y el PnL no realizado menor por posición.
        let contradiccion = |e: &mut Estado, r: Rechazo, pos: Option<&ClavePosicion>, t| {
            if matches!(r, Rechazo::Contradiccion { .. }) {
                let g = e.anclar("Contradiccion");
                match pos {
                    None => e.ev_saldo.empate_desde = Some(g),
                    Some(c) => e.ev_pos.entry(c.clone()).or_default().empate_desde = Some(g),
                }
                e.bajar_techo_en_conflicto(Some(l.saldo), &filas, completa_l);
                self.notificar(e);
            }
            e.registrar(fuente, t, None, &r);
            r
        };
        let saldo = match decidir(
            || "saldo".to_string(),
            e.saldo.is_some(),
            e.saldo.is_some_and(|s| mismo_saldo(s, l.saldo)),
            l.tiempo_saldo_ms,
            e.marca_saldo,
            e.ev_saldo.empate_rest(&ticket, anclada),
        ) {
            Ok(v) => v,
            Err(r) => return Err(contradiccion(&mut *e, r, None, l.tiempo_saldo_ms)),
        };
        // V4C (riesgo de F2): con la misma `T` que el último saldo del stream y
        // otro valor, no hay forma de saber si la foto es posterior o una foto
        // atrasada (el saldo intermedio entre dos fills de esa `T`). No se
        // aplica, no cierra el empate ni suelta el veto, y el freno ve el
        // menor de los dos saldos. La cierra una REST que coincida con el
        // stream o una lectura con marca más nueva.
        if let (Some(t), Some((ts, _, vs))) = (l.tiempo_saldo_ms, e.ultimo_saldo_stream) {
            // V4E (INV-7): vale cualquiera de los saldos que el stream entregó
            // con esa `T` (el aceptado o uno rechazado por contradicción).
            if t == ts && Some(t) == e.marca_saldo && !e.saldo_del_stream_en_esa_marca(l.saldo) {
                let r = contradiccion(
                    &mut *e,
                    Rechazo::Contradiccion {
                        componente: "saldo".to_string(),
                        tiempo_ms: t,
                    },
                    None,
                    Some(t),
                );
                e.bajar_techo_en_conflicto(Some(vs.min(l.saldo)), &filas, completa_l);
                self.notificar(&e);
                return Err(r);
            }
        }
        let mut incompleta = l.posiciones.is_none();
        let mut plan: Vec<(ClavePosicion, DatoPosicion, Option<u64>, Veredicto)> = Vec::new();
        if let Some(m) = &l.posiciones {
            for (clave, p) in m {
                if p.dato == DatoPosicion::Invalida {
                    incompleta = true;
                    continue;
                }
                let empate = e.evidencia_pos(clave).empate_rest(&ticket, anclada);
                match e.decidir_posicion(clave, &p.dato, p.tiempo_ms, empate) {
                    Ok(v) => plan.push((clave.clone(), p.dato, p.tiempo_ms, v)),
                    Err(r) => return Err(contradiccion(&mut *e, r, Some(clave), p.tiempo_ms)),
                }
            }
            // Una abierta que la foto completa no trae: cierre sin marca.
            let ausentes: Vec<ClavePosicion> =
                e.abiertas.keys().filter(|c| !m.contains_key(*c)).cloned().collect();
            for clave in ausentes {
                let empate = e.evidencia_pos(&clave).empate_rest(&ticket, anclada);
                match e.decidir_posicion(&clave, &DatoPosicion::Cerrada, None, empate) {
                    Ok(v) => plan.push((clave, DatoPosicion::Cerrada, None, v)),
                    Err(r) => return Err(contradiccion(&mut *e, r, Some(&clave), None)),
                }
            }
        }
        // El techo se toma con la evidencia de ANTES de esta lectura.
        if incompleta {
            e.incompleta("EvidenciaInvalida");
        }
        if let Veredicto::Aplicar = saldo {
            e.saldo = Some(l.saldo);
            e.ev_saldo.generacion = e.generacion;
            e.ev_saldo.empate_desde = None;
        }
        if l.tiempo_saldo_ms.is_some() && l.tiempo_saldo_ms >= e.marca_saldo {
            // V4C (F1): una foto que sólo confirma el saldo con la misma marca
            // no le quita al stream el orden de su socket (el fill siguiente
            // con la misma `T` y `E` mayor sigue aplicando).
            if l.tiempo_saldo_ms > e.marca_saldo || matches!(saldo, Veredicto::Aplicar) {
                e.ev_saldo.origen = Some(Origen::Rest);
            }
            if anclada && ticket.generacion >= e.ev_saldo.empate_desde.unwrap_or(0) {
                // Una foto anclada con el mismo saldo confirma el empate.
                e.ev_saldo.empate_desde = None;
            }
        }
        e.marca_saldo = maxima(e.marca_saldo, l.tiempo_saldo_ms);
        for (clave, dato, t, v) in &plan {
            let ev = e.evidencia_pos(clave);
            // La foto se pidió después de la última escritura de esta posición.
            let posterior = ticket.generacion > ev.generacion || !e.ev_pos.contains_key(clave);
            let marca_antes = e.marcas_pos.get(clave).copied().flatten();
            // V4C (F3): entre una foto REST y la evidencia vigente no hay orden
            // demostrable del PnL no realizado cuando la fila trae marca vieja.
            e.aplicar_posicion_con(clave, dato, *t, v, posterior, false, Some(Origen::Rest));
            let marca = e.marcas_pos.get(clave).copied().flatten();
            if t.is_some() && *t >= marca {
                let ev = e.ev_pos.entry(clave.clone()).or_default();
                // V4C (F1): como en el saldo, confirmar con la misma marca no
                // le quita el origen al stream.
                if *t > marca_antes || matches!(v, Veredicto::Aplicar) {
                    ev.origen = Some(Origen::Rest);
                }
                if anclada && ev.empate_desde.is_some_and(|g| ticket.generacion >= g) {
                    ev.empate_desde = None;
                }
            }
        }
        // Una posición en empate que la foto completa no trae y que el puente
        // tiene cerrada: la foto confirma el cierre.
        if let (Some(m), true) = (&l.posiciones, anclada) {
            let Estado { ev_pos, abiertas, .. } = &mut *e;
            for (clave, ev) in ev_pos.iter_mut() {
                if !m.contains_key(clave)
                    && !abiertas.contains_key(clave)
                    && ev.empate_desde.is_some_and(|g| ticket.generacion >= g)
                {
                    ev.empate_desde = None;
                }
            }
        }
        if incompleta {
            // Con la suma del activo, una posición sin observar tampoco cuenta 0.
            if let Some(u) = l.upnl_del_activo {
                let cota = (l.saldo + u).max(0.0);
                if cota.is_finite() {
                    e.techo = Some(e.techo.map_or(cota, |t| t.min(cota)));
                }
            }
        } else if anclada {
            // V4B: sólo una lectura completa, del socket abierto y pedida
            // después del último anclaje suelta el veto.
            let pendiente = e.ev_saldo.empate_desde.is_some() || e.ev_pos.values().any(|v| v.empate_desde.is_some());
            if !pendiente {
                e.completa = true;
                e.techo = None;
                e.motivo = None;
            }
        }
        // V4D (P1-b): el cero de 3a llega al freno, pero no admite entradas.
        // V4E (INV-2, dictamen de Codex QRA sobre V4D): una REST sin rechazo sólo
        // despeja 3a si su ticket puede soltar el veto (posterior al último
        // anclaje y al último evento aceptado, socket abierto).
        e.rechazo_3a = l.rechazo_arranque.is_some() || (e.rechazo_3a && !anclada);
        e.generacion += 1;
        self.notificar(&e);
        Ok(e.observacion().expect("saldo recién fijado"))
    }

    /// Evento del stream de la era `era` (socket vigente, sin `E`). `Ok(None)`
    /// = aceptado, pero todavía no hay saldo observado (nada que publicar).
    pub fn observar_stream(&self, era: u64, ev: &EventoStream) -> Result<Option<Observacion>, Rechazo> {
        self.observar_evento(era, None, None, ev, |_| {})
    }

    /// Como [`observar_stream`](Self::observar_stream), y `publicar` corre con
    /// la observación bajo el mismo mutex (no debe reentrar en el puente).
    /// Entero o nada. Un rechazo por contradicción o por falta de marca deja
    /// la equity incompleta (hace falta una lectura REST).
    pub fn observar_stream_y_publicar(
        &self,
        era: u64,
        ev: &EventoStream,
        publicar: impl FnOnce(&Observacion),
    ) -> Result<Option<Observacion>, Rechazo> {
        self.observar_evento(era, None, None, ev, publicar)
    }

    /// V4B: evento del socket `socket` con su hora de evento `E`
    /// (`hora_evento_ms`); `ev.tiempo_ms` lleva la `T`. Un evento de otro
    /// socket se rechaza sin tocar nada.
    pub fn observar_stream_del_socket(
        &self,
        era: u64,
        socket: u64,
        hora_evento_ms: Option<u64>,
        ev: &EventoStream,
        publicar: impl FnOnce(&Observacion),
    ) -> Result<Option<Observacion>, Rechazo> {
        self.observar_evento(era, Some(socket), hora_evento_ms, ev, publicar)
    }

    fn observar_evento(
        &self,
        era: u64,
        socket: Option<u64>,
        hora: Option<u64>,
        ev: &EventoStream,
        publicar: impl FnOnce(&Observacion),
    ) -> Result<Option<Observacion>, Rechazo> {
        let mut e = self.estado();
        let t = ev.tiempo_ms;
        let fuente = "stream";
        // V4F (P1-c): era y testigo de sesión en el mismo paso que la escritura;
        // el `publicar` del evento (espejo del host) tampoco corre si se rechaza.
        match self.vigente(&e, era) {
            Ok(()) => {}
            Err(Rechazo::OtraEra) => return Err(Rechazo::OtraEra),
            Err(r) => {
                e.registrar(fuente, t, hora, &r);
                return Err(r);
            }
        }
        let socket = socket.unwrap_or(e.socket);
        if socket != e.socket || e.socket_cerrado {
            let r = Rechazo::OtroSocket {
                socket,
                vigente: e.socket,
            };
            e.registrar(fuente, t, hora, &r);
            return Err(r);
        }
        // Atrasada (incluida la regresión de `E`) se descarta sin más; una
        // contradicción o un dato sin orden dejan la equity incompleta y un
        // anclaje nuevo; la contradicción además señala el empate.
        let rechazo = |e: &mut Estado, r: Rechazo, pos: Option<&ClavePosicion>| {
            match &r {
                Rechazo::Contradiccion { .. } => {
                    let g = e.anclar("Contradiccion");
                    match pos {
                        None => e.ev_saldo.empate_desde = Some(g),
                        Some(c) => e.ev_pos.entry(c.clone()).or_default().empate_desde = Some(g),
                    }
                    // V4C (F3): el freno ve también las pérdidas del evento rechazado.
                    e.bajar_techo_en_conflicto(ev.saldo, &ev.posiciones, false);
                }
                Rechazo::SinOrden { .. } => {
                    e.anclar("SinOrden");
                }
                _ => {}
            }
            e.registrar(fuente, t, hora, &r);
            r
        };
        let saldo = match ev.saldo {
            Some(s) => match decidir(
                || "saldo".to_string(),
                e.saldo.is_some(),
                e.saldo.is_some_and(|v| mismo_saldo(v, s)),
                t,
                e.marca_saldo,
                e.ev_saldo.empate_stream(socket, hora),
            ) {
                // V4D (P1-d): el saldo aceptado se registra como último saldo
                // del stream sólo cuando el lote entero se aplica (abajo); un
                // lote rechazado por otro componente no deja ese testigo.
                Ok(v) => Some((s, v)),
                Err(r) => {
                    if matches!(r, Rechazo::Contradiccion { .. }) {
                        e.registrar_saldo_stream(t, hora, s, false);
                    }
                    let r = rechazo(&mut *e, r, None);
                    self.notificar(&e);
                    return Err(r);
                }
            },
            None => None,
        };
        let mut incompleta = false;
        let mut veredictos: Vec<Option<(Veredicto, bool, bool)>> = Vec::with_capacity(ev.posiciones.len());
        for (clave, dato) in &ev.posiciones {
            if *dato == DatoPosicion::Invalida {
                incompleta = true;
                veredictos.push(None);
                continue;
            }
            let evidencia = e.evidencia_pos(clave);
            let empate = evidencia.empate_stream(socket, hora);
            // V4C (F3): sólo el mismo socket ordena dos lecturas del stream
            // (`E` menor con la misma `T`, o `T` menor): esa es anterior.
            let marca = e.marcas_pos.get(clave).copied().flatten();
            let mismo_socket = matches!(evidencia.origen, Some(Origen::Stream { socket: s, .. }) if s == socket);
            let anterior = mismo_socket && (empate == Empate::Anterior || (t.is_some() && t < marca));
            match e.decidir_posicion(clave, dato, t, empate) {
                Ok(v) => veredictos.push(Some((v, empate == Empate::Posterior, anterior))),
                Err(r) => {
                    let r = rechazo(&mut *e, r, Some(clave));
                    self.notificar(&e);
                    return Err(r);
                }
            }
        }
        // El techo se toma con la evidencia de ANTES de este evento.
        if incompleta {
            e.anclar("EvidenciaInvalida");
        }
        let empate_saldo = e.ev_saldo.empate_stream(socket, hora);
        match saldo {
            Some((s, Veredicto::Aplicar)) => {
                e.saldo = Some(s);
                // V4D (P1-b): un saldo positivo ya no es el cero de 3a.
                if s > 0.0 {
                    e.rechazo_3a = false;
                }
                e.ev_saldo.generacion = e.generacion;
                e.ev_saldo.empate_desde = None;
                // Sólo un dato que se aplicó hace del stream el origen de la
                // marca; un igual (p. ej. un repetido tras reconectar) no.
                if t.is_some() {
                    e.ev_saldo.origen = con_e_mayor(e.ev_saldo.origen, socket, hora);
                }
            }
            Some((_, Veredicto::Igual)) if t.is_some() && t == e.marca_saldo && empate_saldo == Empate::Posterior => {
                // Duplicado posterior del mismo socket: guarda la `E` mayor.
                e.ev_saldo.origen = con_e_mayor(e.ev_saldo.origen, socket, hora);
            }
            _ => {}
        }
        if let Some((s, _)) = saldo {
            e.registrar_saldo_stream(t, hora, s, true);
            e.marca_saldo = maxima(e.marca_saldo, t);
        }
        for ((clave, dato), v) in ev.posiciones.iter().zip(&veredictos) {
            let Some((v, posterior, anterior)) = v else { continue };
            let origen = con_e_mayor(e.evidencia_pos(clave).origen, socket, hora);
            e.aplicar_posicion_con(clave, dato, t, v, *posterior, *anterior, origen);
            if matches!(v, Veredicto::Igual) && *posterior && t.is_some() {
                e.ev_pos.entry(clave.clone()).or_default().origen = origen;
            }
        }
        // V4C (F2): una REST pedida antes de este evento ya no suelta el veto.
        e.ultimo_evento = Some(e.generacion);
        e.generacion += 1;
        let obs = e.observacion();
        if let Some(o) = &obs {
            publicar(o);
        }
        self.notificar(&e);
        Ok(obs)
    }

    /// Apertura de un socket nuevo del stream de la era `era` (reconexión):
    /// pudo perderse un evento. La evidencia se conserva, la equity queda
    /// incompleta (con techo en la actual) y se vetan las entradas hasta una
    /// REST pedida después de esta apertura. Sube el número de socket: la
    /// procedencia y los tickets del socket anterior dejan de valer. La
    /// reconexión de otra era no toca nada.
    pub fn reconexion(&self, era: u64) -> Result<(), Rechazo> {
        self.abrir_socket(era).map(|_| ())
    }

    /// Como [`reconexion`](Self::reconexion); devuelve el número del socket nuevo.
    pub fn abrir_socket(&self, era: u64) -> Result<u64, Rechazo> {
        let mut e = self.estado();
        self.vigente(&e, era)?;
        e.socket += 1;
        e.socket_cerrado = false;
        e.anclar("Reconexion");
        self.notificar(&e);
        Ok(e.socket)
    }

    /// V4B: el socket del stream de la era `era` se perdió (fin del lazo de
    /// lectura). Desde este instante se vetan las entradas; una REST pedida
    /// durante el hueco aplica sus datos (el freno los ve) pero no suelta el
    /// veto: sólo lo hace una REST pedida después de abrir el socket nuevo.
    pub fn perdida_socket(&self, era: u64) -> Result<(), Rechazo> {
        let mut e = self.estado();
        self.vigente(&e, era)?;
        Self::cerrar_socket(&mut e, "PerdidaSocket");
        self.notificar(&e);
        Ok(())
    }

    /// V4D (P1-a): el streamer de la era `era` arranca y todavía no abrió su
    /// primer socket. Como en [`perdida_socket`](Self::perdida_socket): socket
    /// cerrado, número nuevo (una REST pedida antes no suelta el veto) y
    /// entradas vetadas hasta una REST pedida después de abrirlo. Una REST de
    /// este hueco aplica sus datos al freno. Los cierres no lo consultan.
    pub fn sin_socket(&self, era: u64) -> Result<(), Rechazo> {
        let mut e = self.estado();
        self.vigente(&e, era)?;
        Self::cerrar_socket(&mut e, "SinSocket");
        self.notificar(&e);
        Ok(())
    }

    /// V4B: un `ACCOUNT_UPDATE` ilegible del socket vigente: evidencia perdida
    /// sin corte. Equity incompleta y anclaje nuevo; el socket sigue.
    pub fn evidencia_perdida(&self, era: u64) -> Result<(), Rechazo> {
        let mut e = self.estado();
        self.vigente(&e, era)?;
        e.anclar("EventoIlegible");
        self.notificar(&e);
        Ok(())
    }

    /// `true` sin saldo observado en esta era o con la equity incompleta.
    pub fn necesita_resincronizar(&self) -> bool {
        self.estado().necesita_lectura()
    }

    /// Ninguna entrada nueva: sin saldo vigente de la era, saldo cero (V4D)
    /// o equity incompleta.
    /// Los cierres no lo consultan.
    pub fn veta_entradas(&self) -> bool {
        self.estado().veta_entradas()
    }

    /// Lo último de esta era (`None` = sin saldo observado).
    pub fn observacion(&self) -> Option<Observacion> {
        self.estado().observacion()
    }

    /// Equity para el freno: la de esta era o, si aún no hay saldo, la última
    /// de la era anterior (evidencia conservada; `None` si nunca hubo).
    pub fn equity_del_freno(&self) -> Option<f64> {
        let e = self.estado();
        e.observacion().map(|o| o.equity).or(e.anterior)
    }

    /// Posiciones abiertas observadas.
    pub fn posiciones_en_cache(&self) -> usize {
        self.estado().abiertas.len()
    }

    /// Diagnóstico (solo medida): veto y su motivo, rechazos por fuente y
    /// motivo desde la última toma, y el último rechazo. Vacía los contadores.
    pub fn tomar_diagnostico(&self) -> DiagnosticoPuente {
        let mut e = self.estado();
        let mut rechazos: Vec<(&'static str, &'static str, u64)> =
            e.diag.rechazos.drain().map(|((f, m), n)| (f, m, n)).collect();
        rechazos.sort();
        DiagnosticoPuente {
            veta: e.veta_entradas(),
            motivo_veto: e.motivo_veto(),
            rechazos,
            ultimo_rechazo: e.diag.ultimo.take(),
        }
    }
}

/// Activo de margen de una posición del stream: el campo `ma` si viene; si
/// no, el sufijo del símbolo (USDⓈ-M: `BTCUSDT` → USDT, `BTCUSDC` → USDC).
pub fn margen_en_activo(simbolo: &str, ma: Option<&str>, activo: &str) -> bool {
    match ma {
        Some(m) if !m.is_empty() => m == activo,
        _ => simbolo.ends_with(activo),
    }
}
