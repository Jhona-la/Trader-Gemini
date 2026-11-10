//! QS-R4a — LIBRO CONTRAFACTUAL EN SOMBRA DE LAS INTENCIONES VETADAS.
//!
//! El sistema sólo aprende de lo que ejecuta. Una puerta que veta no recibe
//! nunca la prueba de si vetó bien: si una calibración baja la probabilidad,
//! el EV veta, no hay operaciones y la probabilidad no se corrige (estado
//! absorbente; hallazgo 1 de la auditoría del PR #5). Y un veto sin medir
//! puede estar bloqueando justo las operaciones que pagan.
//!
//! Este libro registra cada intención vetada con la geometría que habría
//! llevado (entrada, TP, SL, τ) y la resuelve con los precios que llegan
//! después, por PRIMER TOQUE: TP, SL o fin de τ (cierre al último precio).
//! El resultado se expresa en R neto de fricción:
//!
//! ```text
//! r = (+tp | −sl | retorno al vencer) − fricción_ida_y_vuelta, todo / sl
//! ```
//!
//! Por cada fuente de veto acumula n, toques y los dos primeros momentos de r.
//! El veredicto compara la media con su error estándar:
//! - la población bloqueada pierde con confianza ⇒ el veto ahorra dinero;
//! - gana con confianza ⇒ el veto bloquea operaciones que pagan;
//! - si no, no hay evidencia todavía.
//!
//! RB-V1a añade `veredicto_hoeffding`: el mismo juicio con el radio de
//! Hoeffding sobre el ancho conocido de cada resultado, válido para todo n
//! (el de z·SE es asintótico y necesita un `n_min`).
//!
//! Límites conocidos (deliberados):
//! - Resolución al ritmo de los precios observados: un toque entre dos
//!   precios se detecta en el siguiente (igual que la gestión viva).
//! - Sin impacto de mercado ni rechazo de la orden: con órdenes del tamaño
//!   mínimo es despreciable, no con tamaños grandes.
//! - Las intenciones de una misma fuente, moneda y lado se solapan y no son
//!   independientes. Se admite UNA abierta por (fuente, moneda, lado): las
//!   repeticiones mientras está abierta se cuentan como `descartadas_solape`.
//! - Memoria fija (`CAPACIDAD`), sin asignaciones tras la construcción: si
//!   está lleno se descarta y se cuenta (`descartadas_llenas`).

/// Intenciones abiertas a la vez como máximo.
pub const CAPACIDAD: usize = 512;
/// Fuentes de veto distinguibles (ids `0..MAX_FUENTES`). El llamador asigna
/// los ids (p. ej. los `REJ_*` de risk-engine y, por encima, las puertas del
/// núcleo y el consejo).
pub const MAX_FUENTES: usize = 64;

/// Una intención vetada, con la geometría que habría llevado.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct IntencionVetada {
    pub fuente: u8,
    pub moneda: u16,
    pub es_largo: bool,
    /// Precio de entrada que habría tenido (mid al vetar).
    pub entrada: f64,
    /// Distancias como fracción del precio de entrada (> 0).
    pub tp_pct: f64,
    pub sl_pct: f64,
    /// Fricción de ida y vuelta como fracción del precio (≥ 0).
    pub friccion_pct: f64,
    /// Horizonte de la intención en ms (> 0).
    pub tau_ms: u64,
    /// Instante del veto en ms.
    pub t0_ms: u64,
}

impl IntencionVetada {
    fn valida(&self) -> bool {
        (self.fuente as usize) < MAX_FUENTES
            && self.entrada.is_finite()
            && self.entrada > 0.0
            && self.tp_pct.is_finite()
            && self.tp_pct > 0.0
            && self.sl_pct.is_finite()
            && self.sl_pct > 0.0
            && self.friccion_pct.is_finite()
            && self.friccion_pct >= 0.0
            && self.tau_ms > 0
    }

    /// Retorno a favor de la intención con el precio `p`, en fracción.
    fn a_favor(&self, p: f64) -> f64 {
        let r = (p - self.entrada) / self.entrada;
        if self.es_largo {
            r
        } else {
            -r
        }
    }
}

/// Cómo se resolvió una intención.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum Resolucion {
    Tp,
    Sl,
    /// Venció τ sin tocar ninguna barrera; lleva el retorno a favor al vencer.
    Vencida(f64),
}

/// Acumulados de una fuente de veto.
#[derive(Debug, Clone, Copy, Default, PartialEq)]
pub struct EstadisticaFuente {
    pub n: u64,
    pub tp: u64,
    pub sl: u64,
    pub vencidas: u64,
    pub suma_r: f64,
    pub suma_r2: f64,
    /// RB-V1a — suma de los cuadrados del ANCHO de cada resultado posible:
    /// r vive en `[−(sl+f)/sl, (tp−f)/sl]`, de ancho `(tp+sl)/sl`, fijado al
    /// registrar la intención (no depende de cómo se resuelva).
    pub suma_ancho2: f64,
}

/// Qué dice la evidencia acumulada de una fuente.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum Veredicto {
    /// Menos de `n_min` intenciones resueltas, o la media no se separa de 0.
    SinEvidencia,
    /// La población bloqueada pierde con confianza: el veto ahorra dinero.
    AhorraDinero,
    /// La población bloqueada gana con confianza: el veto bloquea operaciones
    /// que pagan.
    BloqueaGanadoras,
}

impl EstadisticaFuente {
    /// R medio neto de fricción de lo que la fuente bloqueó.
    pub fn media_r(&self) -> Option<f64> {
        (self.n > 0).then(|| self.suma_r / self.n as f64)
    }

    /// Error estándar de la media (varianza muestral insesgada).
    pub fn error_estandar_r(&self) -> Option<f64> {
        if self.n < 2 {
            return None;
        }
        let n = self.n as f64;
        let media = self.suma_r / n;
        let var = ((self.suma_r2 - n * media * media) / (n - 1.0)).max(0.0);
        Some((var / n).sqrt())
    }

    /// Veredicto con `z` desviaciones del error estándar y un mínimo de
    /// `n_min` resueltas. Entradas no finitas o fuera de dominio ⇒ sin
    /// evidencia (nunca un veredicto por defecto).
    pub fn veredicto(&self, z: f64, n_min: u64) -> Veredicto {
        if !z.is_finite() || z <= 0.0 || self.n < n_min.max(2) {
            return Veredicto::SinEvidencia;
        }
        let (Some(m), Some(se)) = (self.media_r(), self.error_estandar_r()) else {
            return Veredicto::SinEvidencia;
        };
        if !m.is_finite() || !se.is_finite() {
            return Veredicto::SinEvidencia;
        }
        if m + z * se < 0.0 {
            Veredicto::AhorraDinero
        } else if m - z * se > 0.0 {
            Veredicto::BloqueaGanadoras
        } else {
            Veredicto::SinEvidencia
        }
    }

    /// RB-V1a — radio de confianza de Hoeffding para la media de R, con el
    /// ancho CONOCIDO de cada resultado (desigualdad de Hoeffding para
    /// variables acotadas no idénticas): con probabilidad ≥ 1 − `delta_cola`,
    /// `media − E[r] ≤ √(ln(1/δ)·Σ ancho²/2) / n` (y lo mismo en la otra cola).
    /// Vale para TODO n, sin aproximación normal ni varianza estimada.
    /// Supone resultados independientes: el libro admite una intención
    /// abierta por (fuente, moneda, lado), pero monedas correlacionadas siguen
    /// sin ser independientes (límite declarado).
    pub fn radio_hoeffding(&self, delta_cola: f64) -> Option<f64> {
        if self.n == 0
            || !delta_cola.is_finite()
            || delta_cola <= 0.0
            || delta_cola >= 1.0
            || !self.suma_ancho2.is_finite()
            || self.suma_ancho2 <= 0.0
        {
            return None;
        }
        let radio = ((1.0 / delta_cola).ln() * self.suma_ancho2 / 2.0).sqrt() / self.n as f64;
        radio.is_finite().then_some(radio)
    }

    /// RB-V1a — veredicto distribución-libre con el radio de Hoeffding: la
    /// fuente ahorra dinero si TODO el intervalo queda bajo 0 y bloquea
    /// ganadoras si queda sobre 0. Cada cola se contrasta con `delta_cola`.
    pub fn veredicto_hoeffding(&self, delta_cola: f64) -> Veredicto {
        let (Some(m), Some(radio)) = (self.media_r(), self.radio_hoeffding(delta_cola)) else {
            return Veredicto::SinEvidencia;
        };
        if !m.is_finite() {
            Veredicto::SinEvidencia
        } else if m + radio < 0.0 {
            Veredicto::AhorraDinero
        } else if m - radio > 0.0 {
            Veredicto::BloqueaGanadoras
        } else {
            Veredicto::SinEvidencia
        }
    }
}

/// El libro. Memoria fija; ni `registrar` ni `observar` asignan.
///
/// RB-V1a: las intenciones abiertas ocupan el prefijo denso
/// `abiertas[..n_abiertas]`, y la que se resuelve deja su hueco a la última
/// abierta. `registrar` y `observar` recorren sólo las abiertas (antes, las
/// `CAPACIDAD` casillas, ~29 KB, en cada llamada) y `abiertas()` es O(1).
pub struct LibroSombra {
    abiertas: Box<[Option<IntencionVetada>; CAPACIDAD]>,
    n_abiertas: usize,
    stats: [EstadisticaFuente; MAX_FUENTES],
    pub descartadas_llenas: u64,
    pub descartadas_solape: u64,
    pub descartadas_invalidas: u64,
}

impl Default for LibroSombra {
    fn default() -> Self {
        Self::new()
    }
}

impl LibroSombra {
    pub fn new() -> Self {
        Self {
            abiertas: Box::new([None; CAPACIDAD]),
            n_abiertas: 0,
            stats: [EstadisticaFuente::default(); MAX_FUENTES],
            descartadas_llenas: 0,
            descartadas_solape: 0,
            descartadas_invalidas: 0,
        }
    }

    /// Registra una intención vetada. Devuelve `true` si queda abierta.
    pub fn registrar(&mut self, i: IntencionVetada) -> bool {
        if !i.valida() {
            self.descartadas_invalidas += 1;
            return false;
        }
        let solapa = self
            .abiertas
            .iter()
            .take(self.n_abiertas)
            .flatten()
            .any(|a| a.fuente == i.fuente && a.moneda == i.moneda && a.es_largo == i.es_largo);
        if solapa {
            self.descartadas_solape += 1;
            return false;
        }
        match self.abiertas.get_mut(self.n_abiertas) {
            Some(libre) => {
                *libre = Some(i);
                self.n_abiertas += 1;
                true
            }
            None => {
                self.descartadas_llenas += 1;
                false
            }
        }
    }

    /// Avanza las intenciones abiertas de `moneda` con el precio `p` en el
    /// instante `t_ms`. Resuelve las que tocan TP/SL o vencen. Precios no
    /// finitos o no positivos se ignoran. Devuelve cuántas se resolvieron.
    pub fn observar(&mut self, moneda: u16, p: f64, t_ms: u64) -> usize {
        if !p.is_finite() || p <= 0.0 {
            return 0;
        }
        let mut resueltas = 0;
        let mut k = 0;
        while k < self.n_abiertas {
            let Some(i) = self.abiertas.get(k).copied().flatten() else {
                k += 1;
                continue;
            };
            if i.moneda != moneda || t_ms < i.t0_ms {
                k += 1;
                continue;
            }
            let x = i.a_favor(p);
            let res = if x >= i.tp_pct {
                Resolucion::Tp
            } else if x <= -i.sl_pct {
                Resolucion::Sl
            } else if t_ms.saturating_sub(i.t0_ms) >= i.tau_ms {
                Resolucion::Vencida(x)
            } else {
                k += 1;
                continue;
            };
            let bruto = match res {
                Resolucion::Tp => i.tp_pct,
                Resolucion::Sl => -i.sl_pct,
                Resolucion::Vencida(x) => x,
            };
            if let Some(s) = self.stats.get_mut(usize::from(i.fuente)) {
                let r = (bruto - i.friccion_pct) / i.sl_pct;
                s.n += 1;
                match res {
                    Resolucion::Tp => s.tp += 1,
                    Resolucion::Sl => s.sl += 1,
                    Resolucion::Vencida(_) => s.vencidas += 1,
                }
                s.suma_r += r;
                s.suma_r2 += r * r;
                let ancho = (i.tp_pct + i.sl_pct) / i.sl_pct;
                s.suma_ancho2 += ancho * ancho;
            }
            // La última abierta ocupa el hueco; `k` no avanza porque esa
            // casilla trae ahora una intención todavía sin revisar.
            self.n_abiertas -= 1;
            let ultima = self
                .abiertas
                .get_mut(self.n_abiertas)
                .and_then(Option::take);
            if k < self.n_abiertas {
                if let Some(hueco) = self.abiertas.get_mut(k) {
                    *hueco = ultima;
                }
            }
            resueltas += 1;
        }
        resueltas
    }

    pub fn estadistica(&self, fuente: u8) -> Option<&EstadisticaFuente> {
        self.stats.get(fuente as usize)
    }

    pub fn abiertas(&self) -> usize {
        self.n_abiertas
    }
}
