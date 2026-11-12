//! CAPITAL-UNICO V2 — puente de la cuenta: DOS magnitudes con nombre, cada
//! una definida igual en REST (`/fapi/v2/account`) y en el user-data stream
//! (`ACCOUNT_UPDATE`). Decisión 1 del coordinador (2026-11-11) y P1.1/P1.4 de
//! la revisión de Codex QRA.
//!
//! - **SALDO** = `walletBalance` del activo de margen
//!   ([`ACTIVO_DE_MARGEN`](crate::execution_evidence::ACTIVO_DE_MARGEN), USDT).
//!   Alimenta el arena y el tamaño. Nunca lleva PnL no realizado. Es lo que el
//!   stream publica por `AccountSink::on_capital`.
//! - **EQUITY** = saldo + PnL no realizado de las posiciones con margen en
//!   USDT. Observación separada (`AccountSink::on_equity`) para el freno y las
//!   guardas con posición abierta: el freno sigue viendo las pérdidas abiertas.
//!   `marginBalance` de REST NO siembra el arena.
//!
//! Concurrencia y recuperación (P1.4):
//! - **Era**: una por cuenta. La transición a otra cuenta abre era nueva
//!   ([`PuenteCuenta::nueva_era`]); lo que llegue de la era anterior se rechaza.
//! - **Frescura**: el stream se ordena por el tiempo del exchange (`T`/`E`); un
//!   evento más viejo que el último aceptado se rechaza. Una lectura REST lleva
//!   el [`Ticket`] tomado ANTES de pedirla: si desde entonces el puente aceptó
//!   otra observación, la lectura sólo vale si su propio tiempo
//!   (`updateTime`) no es anterior a la última aceptada. Así una respuesta REST
//!   atrasada no pisa un estado más nuevo.
//! - **Ausente no es cero**: un evento sin la fila de USDT, o con la fila sin
//!   `wb`, no cambia el saldo; `wb = 0` es un cero real y se publica (el freno
//!   por capital ≤ 0 lo recibe). Una actualización sólo de posiciones no
//!   cambia el saldo: sólo la equity.
//! - **USDT duplicado** en `B[]`: el evento se descarta para el capital, con
//!   el mismo criterio que REST (`parse_account_capital`).
//! - **Otro activo**: las posiciones con margen en otro activo (p. ej. USDC)
//!   no entran en la caché ni en la equity.
//! - **Reconexión**: la caché de PnL no realizado se vacía (pudo perderse el
//!   cierre de una posición) y la equity queda *incompleta*: mientras lo esté,
//!   la equity publicada no sube por encima de la última completa (no se
//!   olvida una pérdida abierta por haber perdido eventos). Una lectura REST
//!   aceptada la completa otra vez ([`PuenteCuenta::necesita_resincronizar`]).
//!
//! No toca d*, `capital_del_freno` ni ningún umbral.

use std::collections::HashMap;
use std::sync::Mutex;

/// (símbolo, `positionSide`).
pub type ClavePosicion = (String, String);

/// Lo que el puente publica: las dos magnitudes, coherentes entre sí.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Observacion {
    pub saldo: f64,
    pub equity: f64,
}

/// Lectura REST de `/fapi/v2/account` (ver
/// [`parse_account_lectura`](crate::execution_evidence::parse_account_lectura)).
#[derive(Debug, Clone, PartialEq)]
pub struct LecturaRest {
    /// `walletBalance` de USDT.
    pub saldo: f64,
    /// PnL no realizado por posición con margen en USDT. `None` si la
    /// respuesta no trae `positions[]` (la equity queda incompleta).
    pub upnl: Option<HashMap<ClavePosicion, f64>>,
    /// `unrealizedProfit` del activo USDT (sólo se usa si falta `positions[]`).
    pub upnl_del_activo: f64,
    /// El `updateTime` más nuevo de la fila de USDT y de las posiciones.
    pub tiempo_ms: Option<u64>,
}

impl LecturaRest {
    /// La equity de esta lectura, con la misma definición que el stream.
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

/// Un `ACCOUNT_UPDATE` ya filtrado al activo de margen.
#[derive(Debug, Clone, PartialEq, Default)]
pub struct EventoStream {
    /// `T` (o `E`) del exchange.
    pub tiempo_ms: Option<u64>,
    /// `wb` de USDT. `None` = la fila falta o no trae `wb` (no es un cero).
    pub saldo: Option<f64>,
    /// Posiciones con margen en USDT: `Some(upnl)` abierta, `None` cerrada.
    pub posiciones: Vec<(ClavePosicion, Option<f64>)>,
}

/// Se toma ANTES de pedir la lectura REST.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Ticket {
    era: u64,
    generacion: u64,
}

/// Por qué no se aceptó una observación.
#[derive(Debug, Clone, PartialEq)]
pub enum Rechazo {
    /// De otra cuenta (era anterior a la transición).
    OtraEra,
    /// Más vieja que la última observación aceptada.
    Atrasada { tiempo_ms: Option<u64>, ultimo_ms: u64 },
}

impl std::fmt::Display for Rechazo {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Rechazo::OtraEra => write!(f, "observación de otra cuenta (era anterior)"),
            Rechazo::Atrasada { tiempo_ms, ultimo_ms } => write!(
                f,
                "observación atrasada (tiempo {}, última aceptada {ultimo_ms})",
                tiempo_ms.map_or_else(|| "N/D".to_string(), |t| t.to_string())
            ),
        }
    }
}

#[derive(Debug, Default)]
struct Estado {
    era: u64,
    generacion: u64,
    ultimo_ms: u64,
    saldo: Option<f64>,
    upnl: HashMap<ClavePosicion, f64>,
    completa: bool,
    /// Mientras la equity está incompleta: la última completa (techo).
    techo: Option<f64>,
}

impl Estado {
    fn equity_cruda(&self) -> Option<f64> {
        let s = self.saldo?;
        let e = s + self.upnl.values().sum::<f64>();
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
}

/// El puente. Un solo `Mutex`: las decisiones de REST y del stream se
/// serializan, y lo publicado sale de la misma sección crítica.
#[derive(Debug)]
pub struct PuenteCuenta {
    estado: Mutex<Estado>,
}

impl Default for PuenteCuenta {
    fn default() -> Self {
        Self::new()
    }
}

impl PuenteCuenta {
    pub fn new() -> Self {
        Self {
            estado: Mutex::new(Estado {
                era: 1,
                completa: true,
                ..Estado::default()
            }),
        }
    }

    fn estado(&self) -> std::sync::MutexGuard<'_, Estado> {
        self.estado.lock().unwrap_or_else(|p| p.into_inner())
    }

    /// Era vigente (la del streamer que la reciba).
    pub fn era(&self) -> u64 {
        self.estado().era
    }

    /// Transición a otra cuenta: todo vuelve a «no observado» y lo que llegue
    /// con la era anterior se rechaza. Devuelve la era nueva.
    pub fn nueva_era(&self) -> u64 {
        let mut e = self.estado();
        let era = e.era + 1;
        *e = Estado {
            era,
            completa: true,
            ..Estado::default()
        };
        era
    }

    /// Ticket para una lectura REST: tomarlo ANTES de pedirla.
    pub fn ticket(&self) -> Ticket {
        let e = self.estado();
        Ticket {
            era: e.era,
            generacion: e.generacion,
        }
    }

    /// Lectura REST completa (saldo, PnL no realizado por posición). Si se
    /// acepta, reemplaza la caché y completa la equity.
    pub fn observar_rest(&self, ticket: Ticket, lectura: &LecturaRest) -> Result<Observacion, Rechazo> {
        let mut e = self.estado();
        if ticket.era != e.era {
            return Err(Rechazo::OtraEra);
        }
        if ticket.generacion != e.generacion {
            match lectura.tiempo_ms {
                Some(t) if t >= e.ultimo_ms => {}
                t => {
                    return Err(Rechazo::Atrasada {
                        tiempo_ms: t,
                        ultimo_ms: e.ultimo_ms,
                    })
                }
            }
        }
        e.saldo = Some(lectura.saldo);
        match &lectura.upnl {
            Some(m) => {
                e.upnl = m.clone();
                e.completa = true;
                e.techo = None;
            }
            None => {
                // Sin `positions[]` la caché no se puede reconstruir: equity
                // incompleta con techo en la del activo.
                e.upnl.clear();
                e.completa = false;
                e.techo = Some(lectura.equity());
            }
        }
        if let Some(t) = lectura.tiempo_ms {
            e.ultimo_ms = e.ultimo_ms.max(t);
        }
        e.generacion += 1;
        Ok(e.observacion().expect("saldo recién fijado"))
    }

    /// Evento del stream de la era `era`. `Ok(None)` = aceptado, pero todavía
    /// no hay saldo observado (nada que publicar).
    pub fn observar_stream(&self, era: u64, ev: &EventoStream) -> Result<Option<Observacion>, Rechazo> {
        let mut e = self.estado();
        if era != e.era {
            return Err(Rechazo::OtraEra);
        }
        if let Some(t) = ev.tiempo_ms {
            if t < e.ultimo_ms {
                return Err(Rechazo::Atrasada {
                    tiempo_ms: Some(t),
                    ultimo_ms: e.ultimo_ms,
                });
            }
            e.ultimo_ms = t;
        }
        if let Some(s) = ev.saldo {
            e.saldo = Some(s);
        }
        for (clave, upnl) in &ev.posiciones {
            match upnl {
                Some(u) => {
                    e.upnl.insert(clave.clone(), *u);
                }
                None => {
                    e.upnl.remove(clave);
                }
            }
        }
        e.generacion += 1;
        Ok(e.observacion())
    }

    /// Reconexión del stream: pudo perderse el cierre de una posición. La
    /// caché se vacía y la equity queda incompleta (con techo en la última).
    pub fn reconexion(&self) {
        let mut e = self.estado();
        let actual = e.observacion().map(|o| o.equity);
        e.techo = match (e.techo, actual) {
            (Some(t), Some(a)) => Some(t.min(a)),
            (t, a) => t.or(a),
        };
        e.upnl.clear();
        e.completa = false;
        e.generacion += 1;
    }

    /// `true` tras una reconexión hasta la próxima lectura REST aceptada.
    pub fn necesita_resincronizar(&self) -> bool {
        !self.estado().completa
    }

    /// Lo último publicado (`None` = sin saldo observado).
    pub fn observacion(&self) -> Option<Observacion> {
        self.estado().observacion()
    }

    /// Posiciones en la caché de PnL no realizado.
    pub fn posiciones_en_cache(&self) -> usize {
        self.estado().upnl.len()
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
