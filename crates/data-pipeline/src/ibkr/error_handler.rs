//! IBKR-I7 (integrador, 2026-11-16): freno automatico ante los errores de
//! ritmo del IB Gateway.
//!
//! Codigos (https://interactivebrokers.github.io/tws-api/message_codes.html):
//! - 100 = "Max rate of messages per second has been exceeded" (saturacion de
//!   mensajes; otro problema, no lo trata este freno).
//! - 101 = "Max number of tickers has been reached" (exceso de lineas streaming).
//! - 162 = "Historical market data Service error message" (canal general: el
//!   texto dice la causa: pacing, permisos o sin datos).
//! - 420 = pacing de streaming.
//!
//! Regla: ante pacing (162 de pacing o de causa desconocida, y 420) se para
//! todo durante `IBKR_PACING_BREAKER_COOLDOWN_SECS` y despues se reanuda a la
//! mitad de la tasa durante `IBKR_PACING_RECOVERY_HALF_RATE_SECS`. Un 162 de
//! causa desconocida frena (falla cerrado). Un 162 de permisos o sin datos no
//! frena: no es ritmo y frenar no lo arregla.

use std::time::{Duration, Instant};

use super::pacing::IBKR_HISTORICAL_WINDOW_SECS;
use super::IBKR_PACING_BREAKER_COOLDOWN_SECS;

/// Reanudacion a media tasa tras el enfriamiento: T_m = W = 600 s.
/// IBKR-I9. Notacion: C = IBKR_HISTORICAL_MAX_PER_10MIN = 60,
/// W = IBKR_HISTORICAL_WINDOW_SECS = 600 s, r = W/C = 10 s (reposicion de la cubeta de I4),
/// g = IBKR_MIN_PACING_INTERVAL_MS = 2,5 s, T_f = IBKR_PACING_BREAKER_COOLDOWN_SECS = 65 s.
///   N_max(W) = C + ceil(W/r) - 1 = 60 + 60 - 1 = 119 > C   (cubeta de I4 llena al empezar; g <= r)
///   dN_max/dT_m = 0                                        (la cubeta de I4 y PacingGate no leen factor_tasa_at)
///   => no existe T_m con N_max(W) <= C; la condicion es de la cubeta: C_b + ceil(W/r_b) - 1 <= C.
///   T_m := W  (respaldo del pedido I9 sin supuestos; NO basta para la invariante).
/// IBKR-I10: la admision historica pasa a `pacing::RegistroVentana` (ventana deslizante
/// exacta), con N_max(W) = C para todo T_m. La invariante ya no depende de la media tasa,
/// que se conserva como reaccion a la senal de IBKR (162/420).
/// IBKR-I11: la cubeta de I4 y su reposicion r salen de produccion; viven solo en
/// tests/ibkr_compile_contract.rs como control de 119 (i10_control_cubeta_vieja_119).
/// Pruebas: tests/ibkr_compile_contract.rs, i9_* e i10_*.
pub const IBKR_PACING_RECOVERY_HALF_RATE_SECS: u64 = IBKR_HISTORICAL_WINDOW_SECS;

/// Codigo de exceso de lineas streaming (OFICIAL, message_codes.html).
pub const IBKR_ERR_MAX_TICKERS: i32 = 101;
/// Codigo del servicio historico (OFICIAL, message_codes.html).
pub const IBKR_ERR_HISTORICAL_SERVICE: i32 = 162;
/// Codigo de pacing de streaming (pedido I7).
pub const IBKR_ERR_STREAMING_PACING: i32 = 420;

/// Causa de un error que el freno reconoce.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CausaIbkr {
    /// 162 con texto de pacing.
    PacingHistorico,
    /// 162 con texto que no se reconoce: se trata como pacing (falla cerrado).
    HistoricoDesconocido,
    /// 162 por falta de permisos de datos.
    SinPermisos,
    /// 162 porque la consulta no devolvio datos.
    SinDatos,
    /// 101: demasiadas lineas streaming.
    ExcesoLineas,
    /// 420: pacing de streaming.
    PacingStreaming,
    /// Cualquier otro codigo: el freno no actua.
    Otro,
}

impl CausaIbkr {
    /// `true` si la causa activa el enfriamiento.
    pub fn frena(self) -> bool {
        matches!(self, Self::PacingHistorico | Self::HistoricoDesconocido | Self::PacingStreaming)
    }
}

/// Clasifica un error del Gateway por codigo y texto.
pub fn clasificar(codigo: i32, texto: &str) -> CausaIbkr {
    let t = texto.to_ascii_lowercase();
    match codigo {
        IBKR_ERR_HISTORICAL_SERVICE => {
            if t.contains("pacing") {
                CausaIbkr::PacingHistorico
            } else if t.contains("permission") {
                CausaIbkr::SinPermisos
            } else if t.contains("no data") || t.contains("returned no") {
                CausaIbkr::SinDatos
            } else {
                CausaIbkr::HistoricoDesconocido
            }
        }
        IBKR_ERR_MAX_TICKERS => CausaIbkr::ExcesoLineas,
        IBKR_ERR_STREAMING_PACING => CausaIbkr::PacingStreaming,
        _ => CausaIbkr::Otro,
    }
}

/// Estado del freno en un instante.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum EstadoFreno {
    /// Tasa normal.
    Normal,
    /// Todo parado.
    Cooldown,
    /// Reanudado a media tasa.
    MediaTasa,
}

/// Evento que el freno registro (y escribio como WARN).
#[derive(Debug, Clone, PartialEq)]
pub struct EventoFreno {
    pub codigo: i32,
    pub causa: CausaIbkr,
    pub texto: String,
    pub enfriamiento: Duration,
}

/// Freno de errores de IBKR con reloj inyectable.
#[derive(Debug, Clone)]
pub struct ErrorHandler {
    enfriamiento: Duration,
    media_tasa: Duration,
    fin_enfriamiento: Option<Instant>,
    lineas_excedidas: u32,
    eventos: Vec<EventoFreno>,
}

impl Default for ErrorHandler {
    fn default() -> Self {
        Self::new()
    }
}

impl ErrorHandler {
    /// Freno con los tiempos del pedido.
    pub fn new() -> Self {
        Self {
            enfriamiento: Duration::from_secs(IBKR_PACING_BREAKER_COOLDOWN_SECS),
            media_tasa: Duration::from_secs(IBKR_PACING_RECOVERY_HALF_RATE_SECS),
            fin_enfriamiento: None,
            lineas_excedidas: 0,
            eventos: Vec::new(),
        }
    }

    /// Recibe un error en el instante actual.
    pub fn receive_error(&mut self, codigo: i32, texto: &str) -> CausaIbkr {
        self.receive_error_at(codigo, texto, Instant::now())
    }

    /// Recibe un error en `ahora`.
    pub fn receive_error_at(&mut self, codigo: i32, texto: &str, ahora: Instant) -> CausaIbkr {
        let causa = clasificar(codigo, texto);
        if causa == CausaIbkr::ExcesoLineas {
            self.lineas_excedidas += 1;
        }
        if causa.frena() {
            let fin = ahora + self.enfriamiento;
            // Un error nuevo durante el enfriamiento lo alarga, nunca lo acorta.
            if self.fin_enfriamiento.map_or(true, |f| fin > f) {
                self.fin_enfriamiento = Some(fin);
            }
        }
        if causa != CausaIbkr::Otro {
            let enfriamiento = if causa.frena() { self.enfriamiento } else { Duration::ZERO };
            eprintln!(
                "WARN ibkr_freno codigo={codigo} causa={causa:?} enfriamiento_s={} media_tasa_s={} texto={texto:?}",
                enfriamiento.as_secs(),
                if causa.frena() { self.media_tasa.as_secs() } else { 0 },
            );
            self.eventos.push(EventoFreno { codigo, causa, texto: texto.to_string(), enfriamiento });
        }
        causa
    }

    /// Estado del freno en `ahora`.
    pub fn estado_at(&self, ahora: Instant) -> EstadoFreno {
        match self.fin_enfriamiento {
            Some(fin) if ahora < fin => EstadoFreno::Cooldown,
            Some(fin) if ahora < fin + self.media_tasa => EstadoFreno::MediaTasa,
            _ => EstadoFreno::Normal,
        }
    }

    /// Fraccion de la tasa normal permitida en `ahora` (0, 1/2 o 1).
    pub fn factor_tasa_at(&self, ahora: Instant) -> f64 {
        match self.estado_at(ahora) {
            EstadoFreno::Cooldown => 0.0,
            EstadoFreno::MediaTasa => 0.5,
            EstadoFreno::Normal => 1.0,
        }
    }

    pub fn is_cooling_down_at(&self, ahora: Instant) -> bool {
        self.estado_at(ahora) == EstadoFreno::Cooldown
    }

    pub fn is_cooling_down(&self) -> bool {
        self.is_cooling_down_at(Instant::now())
    }

    /// Duracion del enfriamiento.
    pub fn cooldown_duration(&self) -> Duration {
        self.enfriamiento
    }

    /// Duracion de la reanudacion a media tasa.
    pub fn half_rate_duration(&self) -> Duration {
        self.media_tasa
    }

    /// Cuantos 101 (exceso de lineas) se recibieron.
    pub fn lineas_excedidas(&self) -> u32 {
        self.lineas_excedidas
    }

    pub fn eventos(&self) -> &[EventoFreno] {
        &self.eventos
    }
}
