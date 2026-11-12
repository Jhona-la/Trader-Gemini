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
//! El módulo sólo usa `std` (se compila solo en las pruebas de auditoría).
//! No toca d*, `capital_del_freno` ni ningún umbral.

use std::collections::HashMap;
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

/// Se toma ANTES de pedir la lectura REST: ata la lectura a la era.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Ticket {
    era: u64,
    generacion: u64,
}

impl Ticket {
    pub fn era(&self) -> u64 {
        self.era
    }

    /// Cuántas observaciones había aceptado el puente al tomarlo (informativo).
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
}

impl std::fmt::Display for Rechazo {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Rechazo::OtraEra => write!(f, "observación de otra sesión de cuenta (era)"),
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

enum Veredicto {
    Aplicar,
    Igual,
}

/// Regla única de frescura de un componente.
fn decidir(
    componente: impl Fn() -> String,
    observado: bool,
    igual: bool,
    t: Option<u64>,
    marca: Option<u64>,
) -> Result<Veredicto, Rechazo> {
    if !observado {
        return Ok(Veredicto::Aplicar);
    }
    match (t, marca) {
        (Some(t), Some(m)) if t > m => Ok(Veredicto::Aplicar),
        (Some(_), None) => Ok(Veredicto::Aplicar),
        _ if igual => Ok(Veredicto::Igual),
        (Some(t), Some(m)) if t == m => Err(Rechazo::Contradiccion {
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

#[derive(Debug, Default)]
struct Estado {
    era: u64,
    generacion: u64,
    saldo: Option<f64>,
    marca_saldo: Option<u64>,
    /// Posiciones abiertas observadas: (cantidad, PnL no realizado).
    abiertas: HashMap<ClavePosicion, (f64, f64)>,
    /// Marca de cada posición observada (abierta o cerrada).
    marcas_pos: HashMap<ClavePosicion, Option<u64>>,
    completa: bool,
    /// Mientras la equity está incompleta: la última completa (techo).
    techo: Option<f64>,
    /// Equity de la era anterior (sólo para el freno, nunca para entradas).
    anterior: Option<f64>,
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

    fn veta_entradas(&self) -> bool {
        self.saldo.is_none() || !self.completa
    }

    /// Deja la equity incompleta con techo en la equity actual.
    fn incompleta(&mut self) {
        let actual = self.observacion().map(|o| o.equity);
        self.techo = match (self.techo, actual) {
            (Some(t), Some(a)) => Some(t.min(a)),
            (t, a) => t.or(a),
        };
        self.completa = false;
    }

    /// Veredicto de una posición contra lo observado.
    fn decidir_posicion(
        &self,
        clave: &ClavePosicion,
        dato: &DatoPosicion,
        t: Option<u64>,
    ) -> Result<Veredicto, Rechazo> {
        let observado = self.marcas_pos.contains_key(clave) || self.abiertas.contains_key(clave);
        let igual = match (dato, self.abiertas.get(clave)) {
            (DatoPosicion::Abierta { cantidad, .. }, Some((c, _))) => mismo_saldo(*cantidad, *c),
            (DatoPosicion::Cerrada, None) => observado,
            _ => false,
        };
        let marca = self.marcas_pos.get(clave).copied().flatten();
        decidir(|| format!("posición {} {}", clave.0, clave.1), observado, igual, t, marca)
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

type Observador = Box<dyn Fn(bool) + Send + Sync>;

/// El puente. Un solo `Mutex`: las decisiones de REST y del stream se
/// serializan, y lo publicado sale de la misma sección crítica.
pub struct PuenteCuenta {
    estado: Mutex<Estado>,
    /// Recibe [`PuenteCuenta::veta_entradas`] tras cada cambio, bajo el mutex.
    veto: OnceLock<Observador>,
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
    /// Era 1, sin saldo observado: veta entradas hasta la primera lectura.
    pub fn new() -> Self {
        Self {
            estado: Mutex::new(Estado {
                era: 1,
                completa: false,
                ..Estado::default()
            }),
            veto: OnceLock::new(),
        }
    }

    fn estado(&self) -> std::sync::MutexGuard<'_, Estado> {
        self.estado.lock().unwrap_or_else(|p| p.into_inner())
    }

    fn notificar(&self, e: &Estado) {
        if let Some(f) = self.veto.get() {
            f(e.veta_entradas());
        }
    }

    /// Vincula el veto de entradas (el host escribe la bandera del arena). Se
    /// llama en seguida con el estado actual y después tras cada cambio, bajo
    /// el mismo mutex del puente. Sólo el primero vale.
    pub fn vincular_veto(&self, f: impl Fn(bool) + Send + Sync + 'static) {
        let e = self.estado();
        if self.veto.set(Box::new(f)).is_ok() {
            self.notificar(&e);
        }
    }

    /// Era vigente (la del streamer que la reciba).
    pub fn era(&self) -> u64 {
        self.estado().era
    }

    /// Sesión de cuenta nueva (transición): el saldo queda sin observar, la
    /// equity incompleta y las entradas vetadas hasta una lectura aceptada de
    /// esta era. Lo que llegue con otra era se rechaza. La equity de la era
    /// anterior queda sólo como evidencia para el freno. Devuelve la era nueva.
    pub fn nueva_era(&self) -> u64 {
        let mut e = self.estado();
        let era = e.era + 1;
        let anterior = e.observacion().map(|o| o.equity).or(e.anterior);
        *e = Estado {
            era,
            completa: false,
            anterior,
            ..Estado::default()
        };
        self.notificar(&e);
        era
    }

    /// Ticket de la era vigente: tomarlo ANTES de pedir la lectura.
    pub fn ticket(&self) -> Ticket {
        let e = self.estado();
        Ticket {
            era: e.era,
            generacion: e.generacion,
        }
    }

    /// Ticket de `era`, si sigue vigente.
    pub fn ticket_de_era(&self, era: u64) -> Option<Ticket> {
        let e = self.estado();
        (e.era == era).then_some(Ticket {
            era,
            generacion: e.generacion,
        })
    }

    /// Corre `f` bajo el mutex del puente si `era` sigue vigente: ninguna
    /// transición se cuela entre la comprobación y la publicación.
    pub fn con_era_vigente<R>(&self, era: u64, f: impl FnOnce(Option<Observacion>) -> R) -> Option<R> {
        let e = self.estado();
        (e.era == era).then(|| f(e.observacion()))
    }

    /// Lectura REST. Entera o nada: si un componente se rechaza, no cambia nada.
    pub fn observar_rest<L: FuenteRest + ?Sized>(
        &self,
        ticket: Ticket,
        lectura: &L,
    ) -> Result<Observacion, Rechazo> {
        let l = lectura.lectura_cuenta();
        let mut e = self.estado();
        if ticket.era != e.era {
            return Err(Rechazo::OtraEra);
        }
        let saldo = decidir(
            || "saldo".to_string(),
            e.saldo.is_some(),
            e.saldo.is_some_and(|s| mismo_saldo(s, l.saldo)),
            l.tiempo_saldo_ms,
            e.marca_saldo,
        )?;
        let mut incompleta = l.posiciones.is_none();
        let mut plan: Vec<(ClavePosicion, DatoPosicion, Option<u64>)> = Vec::new();
        if let Some(m) = &l.posiciones {
            for (clave, p) in m {
                if p.dato == DatoPosicion::Invalida {
                    incompleta = true;
                    continue;
                }
                e.decidir_posicion(clave, &p.dato, p.tiempo_ms)?;
                plan.push((clave.clone(), p.dato, p.tiempo_ms));
            }
            // Una abierta que la foto completa no trae: cierre sin marca.
            for clave in e.abiertas.keys() {
                if !m.contains_key(clave) {
                    e.decidir_posicion(clave, &DatoPosicion::Cerrada, None)?;
                    plan.push((clave.clone(), DatoPosicion::Cerrada, None));
                }
            }
        }
        // El techo se toma con la evidencia de ANTES de esta lectura.
        if incompleta {
            e.incompleta();
        }
        if let Veredicto::Aplicar = saldo {
            e.saldo = Some(l.saldo);
        }
        e.marca_saldo = maxima(e.marca_saldo, l.tiempo_saldo_ms);
        for (clave, dato, t) in &plan {
            e.aplicar_posicion(clave, dato, *t);
        }
        if incompleta {
            // Con la suma del activo, una posición sin observar tampoco cuenta 0.
            if let Some(u) = l.upnl_del_activo {
                let cota = (l.saldo + u).max(0.0);
                if cota.is_finite() {
                    e.techo = Some(e.techo.map_or(cota, |t| t.min(cota)));
                }
            }
        } else {
            e.completa = true;
            e.techo = None;
        }
        e.generacion += 1;
        self.notificar(&e);
        Ok(e.observacion().expect("saldo recién fijado"))
    }

    /// Evento del stream de la era `era`. `Ok(None)` = aceptado, pero todavía
    /// no hay saldo observado (nada que publicar).
    pub fn observar_stream(&self, era: u64, ev: &EventoStream) -> Result<Option<Observacion>, Rechazo> {
        self.observar_stream_y_publicar(era, ev, |_| {})
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
        let mut e = self.estado();
        if era != e.era {
            return Err(Rechazo::OtraEra);
        }
        let t = ev.tiempo_ms;
        let rechazo = |e: &mut Estado, r: Rechazo| {
            if !matches!(r, Rechazo::Atrasada { .. }) {
                e.incompleta();
            }
            r
        };
        let saldo = match ev.saldo {
            Some(s) => match decidir(
                || "saldo".to_string(),
                e.saldo.is_some(),
                e.saldo.is_some_and(|v| mismo_saldo(v, s)),
                t,
                e.marca_saldo,
            ) {
                Ok(v) => Some((s, v)),
                Err(r) => {
                    let r = rechazo(&mut *e, r);
                    self.notificar(&e);
                    return Err(r);
                }
            },
            None => None,
        };
        let mut incompleta = false;
        for (clave, dato) in &ev.posiciones {
            if *dato == DatoPosicion::Invalida {
                incompleta = true;
                continue;
            }
            if let Err(r) = e.decidir_posicion(clave, dato, t) {
                let r = rechazo(&mut *e, r);
                self.notificar(&e);
                return Err(r);
            }
        }
        // El techo se toma con la evidencia de ANTES de este evento.
        if incompleta {
            e.incompleta();
        }
        if let Some((s, Veredicto::Aplicar)) = saldo {
            e.saldo = Some(s);
        }
        if saldo.is_some() {
            e.marca_saldo = maxima(e.marca_saldo, t);
        }
        for (clave, dato) in &ev.posiciones {
            e.aplicar_posicion(clave, dato, t);
        }
        e.generacion += 1;
        let obs = e.observacion();
        if let Some(o) = &obs {
            publicar(o);
        }
        self.notificar(&e);
        Ok(obs)
    }

    /// Reconexión del stream de la era `era`: pudo perderse un evento. La
    /// evidencia se conserva, la equity queda incompleta (con techo en la
    /// actual) y se vetan las entradas hasta resincronizar. La reconexión de
    /// otra era no toca nada.
    pub fn reconexion(&self, era: u64) -> Result<(), Rechazo> {
        let mut e = self.estado();
        if era != e.era {
            return Err(Rechazo::OtraEra);
        }
        e.incompleta();
        e.generacion += 1;
        self.notificar(&e);
        Ok(())
    }

    /// `true` sin saldo observado en esta era o con la equity incompleta.
    pub fn necesita_resincronizar(&self) -> bool {
        self.estado().veta_entradas()
    }

    /// Ninguna entrada nueva: sin saldo vigente de la era o equity incompleta.
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
}

/// Activo de margen de una posición del stream: el campo `ma` si viene; si
/// no, el sufijo del símbolo (USDⓈ-M: `BTCUSDT` → USDT, `BTCUSDC` → USDC).
pub fn margen_en_activo(simbolo: &str, ma: Option<&str>, activo: &str) -> bool {
    match ma {
        Some(m) if !m.is_empty() => m == activo,
        _ => simbolo.ends_with(activo),
    }
}
