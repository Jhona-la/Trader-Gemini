//! IBKR-I5 (integrador, 2026-11-16): cache de deduplicacion de peticiones
//! historicas. Dos peticiones con la misma clave de 6 campos dentro del TTL se
//! responden desde memoria y no salen al Gateway (la regla oficial prohibe
//! peticiones identicas a menos de 15 s).
//!
//! Sin techo de entradas propio: una entrada vive como mucho el TTL y solo
//! entra tras una peticion admitida por la cubeta de `pacing`, asi que el
//! tamano queda acotado por esa cubeta (DERIVADO; no hace falta otro numero).

use std::collections::HashMap;
use std::time::{Duration, Instant};

use super::IBKR_DEDUP_CACHE_TTL_SECS;

/// Clave de deduplicacion:
/// `(Symbol, SecType, Exchange, BarSize, EndDateTime, WhatToShow)`.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct ClaveCache {
    pub symbol: String,
    pub sec_type: String,
    pub exchange: String,
    pub bar_size: String,
    pub end_date_time: String,
    pub what_to_show: String,
}

impl<'a> From<(&'a str, &'a str, &'a str, &'a str, &'a str, &'a str)> for ClaveCache {
    fn from(c: (&'a str, &'a str, &'a str, &'a str, &'a str, &'a str)) -> Self {
        Self {
            symbol: c.0.to_string(),
            sec_type: c.1.to_string(),
            exchange: c.2.to_string(),
            bar_size: c.3.to_string(),
            end_date_time: c.4.to_string(),
            what_to_show: c.5.to_string(),
        }
    }
}

/// Cache con TTL y reloj inyectado.
#[derive(Debug, Clone)]
pub struct IbkrCache<V> {
    ttl: Duration,
    entradas: HashMap<ClaveCache, (Instant, V)>,
}

impl<V> IbkrCache<V> {
    pub fn new(ttl: Duration) -> Self {
        Self { ttl, entradas: HashMap::new() }
    }

    /// Cache con el TTL de IBKR: `IBKR_DEDUP_CACHE_TTL_SECS`.
    pub fn ibkr() -> Self {
        Self::new(Duration::from_secs(IBKR_DEDUP_CACHE_TTL_SECS))
    }

    pub fn ttl(&self) -> Duration {
        self.ttl
    }

    fn vigente(&self, guardada: Instant, ahora: Instant) -> bool {
        ahora.checked_duration_since(guardada).is_some_and(|edad| edad <= self.ttl)
    }

    /// Guarda `valor` para `clave` en `ahora` y retira lo vencido.
    pub fn insert_at(&mut self, clave: impl Into<ClaveCache>, valor: V, ahora: Instant) {
        let ttl = self.ttl;
        self.entradas.retain(|_, (t, _)| ahora.checked_duration_since(*t).map_or(true, |e| e <= ttl));
        self.entradas.insert(clave.into(), (ahora, valor));
    }

    /// Respuesta en memoria para `clave` si sigue dentro del TTL en `ahora`.
    pub fn get_at(&self, clave: impl Into<ClaveCache>, ahora: Instant) -> Option<&V> {
        let clave = clave.into();
        match self.entradas.get(&clave) {
            Some((t, v)) if self.vigente(*t, ahora) => Some(v),
            _ => None,
        }
    }

    pub fn len(&self) -> usize {
        self.entradas.len()
    }

    pub fn is_empty(&self) -> bool {
        self.entradas.is_empty()
    }
}
