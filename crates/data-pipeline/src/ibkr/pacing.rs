//! IBKR-I4 (integrador, 2026-11-16): compuerta de ritmo de las peticiones al
//! IB Gateway. Todo con reloj inyectado (`*_at(Instant)`) para probar sin
//! esperas reales.
//!
//! Fuentes oficiales:
//! - Limites historicos: https://www.interactivebrokers.com/docs/tws-api/doc/md/market-data-limitations
//! - Lineas streaming: https://interactivebrokers.github.io/tws-api/market_data.html
//! - Codigos de error: https://interactivebrokers.github.io/tws-api/message_codes.html

use std::collections::{HashMap, VecDeque};
use std::time::{Duration, Instant};

use super::{IBKR_HISTORICAL_MAX_PER_10MIN, IBKR_MAX_STREAMING_LINES_BUDGET, IBKR_MIN_PACING_INTERVAL_MS};

/// Ventana oficial del limite historico: 10 minutos. OFICIAL (misma fuente que
/// `IBKR_HISTORICAL_MAX_PER_10MIN`).
pub const IBKR_HISTORICAL_WINDOW_SECS: u64 = 600;

/// Reposicion de una ficha: 10 s. DERIVADO: 600 s / 60 peticiones.
/// Desde I10 la admision historica no la usa (`RegistroVentana` solo lee C y W).
pub const IBKR_HISTORICAL_REFILL_SECS: u64 =
    IBKR_HISTORICAL_WINDOW_SECS / IBKR_HISTORICAL_MAX_PER_10MIN as u64;

/// Maximo de peticiones historicas simultaneas en vuelo: 50. OFICIAL segun la
/// especificacion de AGY (seccion 2.1), que cita
/// https://interactivebrokers.github.io/tws-api/historical_limitations.html
/// (consulta 2026-11-16; esa URL esta deprecada y el integrador no la
/// reverifico porque el pedido se ejecuta sin red).
pub const IBKR_MAX_IN_FLIGHT_HISTORICAL: usize = 50;

/// IBKR-I10: admision de las peticiones historicas por ventana deslizante
/// exacta. Guarda las horas de envio de las ultimas `capacidad` peticiones
/// admitidas (anillo de capacidad C) y admite una peticion en `t` solo si hay
/// menos de C marcas en (t - W, t]. Con C = `IBKR_HISTORICAL_MAX_PER_10MIN` y
/// W = `IBKR_HISTORICAL_WINDOW_SECS` cumple sin supuestos la invariante
/// "nunca mas de C envios en cualquier ventana de W" (la cubeta de I4 dejaba
/// pasar C + ceil(W/r) - 1 = 119).
///
/// Falla cerrado, sin panico (RB-P1): C = 0 o W = 0 no admiten nada; un `ahora`
/// anterior a la ultima marca (reloj hacia atras) no admite.
#[derive(Debug, Clone)]
pub struct RegistroVentana {
    capacidad: usize,
    ventana: Duration,
    /// Horas de envio admitidas, en orden no decreciente; nunca mas de `capacidad`.
    envios: VecDeque<Instant>,
}

impl RegistroVentana {
    pub fn new(capacidad: usize, ventana: Duration) -> Self {
        Self { capacidad, ventana, envios: VecDeque::new() }
    }

    /// Limites oficiales de IBKR: C = 60 peticiones en W = 600 s.
    pub fn ibkr_historico() -> Self {
        Self::new(IBKR_HISTORICAL_MAX_PER_10MIN, Duration::from_secs(IBKR_HISTORICAL_WINDOW_SECS))
    }

    /// Retira las marcas que ya no estan en (ahora - W, ahora].
    fn purgar(&mut self, ahora: Instant) {
        // Si ahora - W no es representable, ninguna marca puede haber salido.
        let Some(borde) = ahora.checked_sub(self.ventana) else {
            return;
        };
        while self.envios.front().is_some_and(|m| *m <= borde) {
            self.envios.pop_front();
        }
    }

    fn cerrado(&self) -> bool {
        self.capacidad == 0 || self.ventana.is_zero()
    }

    /// Admite y registra una peticion en `ahora` si la ventana tiene hueco.
    pub fn try_acquire_at(&mut self, ahora: Instant) -> bool {
        if self.cerrado() || self.envios.back().is_some_and(|u| ahora < *u) {
            return false;
        }
        self.purgar(ahora);
        if self.envios.len() >= self.capacidad {
            return false;
        }
        self.envios.push_back(ahora);
        true
    }

    /// Peticiones que se admitirian en `ahora` (0 si falla cerrado).
    pub fn disponibles_at(&mut self, ahora: Instant) -> usize {
        if self.cerrado() || self.envios.back().is_some_and(|u| ahora < *u) {
            return 0;
        }
        self.purgar(ahora);
        self.capacidad.saturating_sub(self.envios.len())
    }

    pub fn capacidad(&self) -> usize {
        self.capacidad
    }

    pub fn ventana(&self) -> Duration {
        self.ventana
    }
}

/// Cubeta de fichas: `capacidad` fichas, una ficha nueva cada `reposicion`.
///
/// IBKR-I10: ya NO es la admision de las peticiones historicas (su peor caso
/// es C + ceil(W/r) - 1 = 119 por ventana de 600 s); esa es `RegistroVentana`.
/// Se conserva como cubeta generica y como control de las pruebas i9/i10.
#[derive(Debug, Clone)]
pub struct TokenBucket {
    capacidad: usize,
    reposicion: Duration,
    fichas: usize,
    /// Instante desde el que se cuenta la proxima reposicion.
    ultima: Option<Instant>,
}

impl TokenBucket {
    /// Cubeta llena con `capacidad` fichas. Con `reposicion` cero no repone
    /// nunca (falla cerrado; sin panico: release usa panic = abort, RB-P1).
    pub fn new(capacidad: usize, reposicion: Duration) -> Self {
        Self { capacidad, reposicion, fichas: capacidad, ultima: None }
    }

    fn reponer(&mut self, ahora: Instant) {
        let Some(ultima) = self.ultima else {
            self.ultima = Some(ahora);
            return;
        };
        if ahora <= ultima {
            return;
        }
        let paso = self.reposicion.as_nanos();
        if paso == 0 {
            return;
        }
        let transcurrido = ahora - ultima;
        let nuevas = usize::try_from(transcurrido.as_nanos() / paso).unwrap_or(usize::MAX);
        if nuevas == 0 {
            return;
        }
        self.fichas = self.fichas.saturating_add(nuevas).min(self.capacidad);
        let avance = u32::try_from(nuevas).ok().and_then(|n| self.reposicion.checked_mul(n));
        let siguiente = avance.and_then(|a| ultima.checked_add(a));
        if self.fichas == self.capacidad || siguiente.is_none() {
            self.ultima = Some(ahora);
        } else {
            self.ultima = siguiente;
        }
    }

    /// Toma una ficha en `ahora` si hay. Devuelve `false` sin tomar nada si la
    /// cubeta esta vacia.
    pub fn try_acquire_at(&mut self, ahora: Instant) -> bool {
        self.reponer(ahora);
        if self.fichas == 0 {
            return false;
        }
        self.fichas -= 1;
        true
    }

    /// Fichas disponibles en `ahora`.
    pub fn disponibles_at(&mut self, ahora: Instant) -> usize {
        self.reponer(ahora);
        self.fichas
    }

    pub fn capacidad(&self) -> usize {
        self.capacidad
    }
}

/// Espaciador: exige al menos `min_interval_ms` entre dos envios.
#[derive(Debug, Clone)]
pub struct PacingGate {
    min_interval: Duration,
    ultimo_envio: Option<Instant>,
}

impl PacingGate {
    pub fn new(min_interval_ms: u64) -> Self {
        Self { min_interval: Duration::from_millis(min_interval_ms), ultimo_envio: None }
    }

    /// Espaciador de IBKR: `IBKR_MIN_PACING_INTERVAL_MS`.
    pub fn ibkr() -> Self {
        Self::new(IBKR_MIN_PACING_INTERVAL_MS)
    }

    pub fn min_interval_ms(&self) -> u64 {
        self.min_interval.as_millis() as u64
    }

    /// Registra un envio en `ahora` si ya paso el intervalo minimo.
    pub fn try_send_at(&mut self, ahora: Instant) -> bool {
        if let Some(previo) = self.ultimo_envio {
            if ahora < previo || ahora - previo < self.min_interval {
                return false;
            }
        }
        self.ultimo_envio = Some(ahora);
        true
    }
}

/// Contador de peticiones en vuelo con techo.
#[derive(Debug, Clone)]
pub struct InFlightCounter {
    maximo: usize,
    en_vuelo: usize,
}

impl InFlightCounter {
    pub fn new(maximo: usize) -> Self {
        Self { maximo, en_vuelo: 0 }
    }

    /// Techo de IBKR: `IBKR_MAX_IN_FLIGHT_HISTORICAL`.
    pub fn ibkr() -> Self {
        Self::new(IBKR_MAX_IN_FLIGHT_HISTORICAL)
    }

    /// Abre una peticion si queda hueco.
    pub fn try_begin(&mut self) -> bool {
        if self.en_vuelo >= self.maximo {
            return false;
        }
        self.en_vuelo += 1;
        true
    }

    /// Cierra una peticion (respuesta, error o cancelacion).
    pub fn finish(&mut self) {
        self.en_vuelo = self.en_vuelo.saturating_sub(1);
    }

    pub fn en_vuelo(&self) -> usize {
        self.en_vuelo
    }
}

/// Registro de lineas streaming abiertas, con desalojo de la menos usada
/// (LRU) al pasar el techo.
#[derive(Debug, Clone)]
pub struct StreamingLines {
    maximo: usize,
    /// clave de la linea -> ultimo uso
    lineas: HashMap<String, Instant>,
}

impl StreamingLines {
    /// Con `maximo` cero no abre ninguna linea (falla cerrado, sin panico).
    pub fn new(maximo: usize) -> Self {
        Self { maximo, lineas: HashMap::new() }
    }

    /// Techo de IBKR: `IBKR_MAX_STREAMING_LINES_BUDGET`.
    pub fn ibkr() -> Self {
        Self::new(IBKR_MAX_STREAMING_LINES_BUDGET)
    }

    /// Abre (o refresca) la linea `clave` en `ahora`. Si abrirla pasa el
    /// techo, devuelve la linea desalojada, que el llamador debe cancelar en
    /// el Gateway.
    pub fn open_at(&mut self, clave: &str, ahora: Instant) -> Option<String> {
        if let Some(uso) = self.lineas.get_mut(clave) {
            *uso = ahora;
            return None;
        }
        if self.maximo == 0 {
            // Sin techo util: la propia linea pedida es la que hay que cancelar.
            return Some(clave.to_string());
        }
        let mut desalojada = None;
        if self.lineas.len() >= self.maximo {
            let vieja = self
                .lineas
                .iter()
                .min_by(|a, b| a.1.cmp(b.1).then_with(|| a.0.cmp(b.0)))
                .map(|(k, _)| k.clone());
            if let Some(vieja) = vieja {
                self.lineas.remove(&vieja);
                desalojada = Some(vieja);
            }
        }
        self.lineas.insert(clave.to_string(), ahora);
        desalojada
    }

    /// Cierra la linea `clave`.
    pub fn close(&mut self, clave: &str) -> bool {
        self.lineas.remove(clave).is_some()
    }

    pub fn abiertas(&self) -> usize {
        self.lineas.len()
    }

    pub fn contiene(&self, clave: &str) -> bool {
        self.lineas.contains_key(clave)
    }
}
