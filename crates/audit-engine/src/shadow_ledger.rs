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
//! RB-V1b sustituye el veredicto de Hoeffding de RB-V1a por `veredicto_ville`.
//! Hoeffding vale para un n FIJADO de antemano, pero el núcleo vuelve a
//! evaluar el veredicto en cada resolución y lo anuncia en cuanto cruza. Con
//! esa mirada repetida, la probabilidad de un veredicto falso supera la
//! nominal (por la ley del logaritmo iterado, acaba cruzando con probabilidad
//! uno). El contrato `rb_v1b_ville_sombra_contract` lo mide con un juego justo:
//! con δ = 0,05/6 por cola y mirando en cada n hasta 4 000, Hoeffding se
//! equivoca en el 3,1 % ± 0,3 % frente al 1,7 % nominal, y el veredicto de
//! Ville en el 0,3 %. `veredicto_ville` apuesta contra cada hipótesis
//! nula con un e-proceso (supermartingala no negativa) y, por la desigualdad
//! maximal de Ville, P(∃n: M_n ≥ 1/α) ≤ α: el veredicto es válido en
//! CUALQUIER momento, se mire cuantas veces se mire (ver `log_evidencia`).
//! El de z·SE (QS-R4) es asintótico y necesita un `n_min`.
//!
//! RB-V1c añade la PROMESA: cada intención puede llevar el EV neto que el
//! riesgo le prometió (`promesa_r`, en R). Por fuente se acumula la brecha
//! realizado − prometido con otro e-proceso, y `veredicto_promesa` certifica,
//! en cualquier momento, si el riesgo promete de más (su EV valida operaciones
//! que no pagan lo que dice) o de menos.
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
    /// RB-V1c — EV neto de fricción que el riesgo prometió para la intención,
    /// en R (`(EV bruto − f)/sl`); `None` si la evaluación no llegó a
    /// calcularlo.
    pub promesa_r: Option<f64>,
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
            && self.promesa_r.is_none_or(f64::is_finite)
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

/// Apuestas de la rejilla del e-proceso de Ville (ver `log_evidencia`).
pub const APUESTAS: usize = 8;
/// Apuesta mayor de la rejilla. Con `x = r/ancho` en un intervalo de ancho 1,
/// la apuesta de crecimiento óptimo para una media x̄ es `u* = 4·x̄` (máximo de
/// `u·x̄ − u²/8`), y |x̄| < 1: ninguna apuesta útil supera 4.
pub const APUESTA_MAX: f64 = 4.0;

/// `u_j = 4·2^(−j)`, j = 0..8: cubre las medias normalizadas desde su máximo
/// hasta x̄ ≈ 1/128, por debajo de la cual certificar exigiría más de 10⁴
/// resoluciones. Con un paso de factor 2, la mejor apuesta de la rejilla
/// conserva al menos el 83 % del crecimiento óptimo `2·x̄²` por resolución.
pub const REJILLA_APUESTAS: [f64; APUESTAS] = {
    let mut u = [0.0; APUESTAS];
    let mut j = 0;
    let mut v = APUESTA_MAX;
    while j < APUESTAS {
        u[j] = v;
        v *= 0.5;
        j += 1;
    }
    u
};

/// RB-V1b/RB-V1c — e-proceso de Ville sobre observaciones `x` de un intervalo
/// de ancho 1 conocido antes de verlas: la riqueza de cada apuesta `u_j` de la
/// rejilla contra «media ≤ 0» y contra «media ≥ 0» (ver `log_evidencia`).
#[derive(Debug, Clone, Copy, Default, PartialEq)]
pub struct EProcesoVille {
    /// `ln` de la riqueza contra H0⁺ (media ≤ 0): `Σ_i (u_j·x_i − u_j²/8)`.
    pub log_riqueza_positiva: [f64; APUESTAS],
    /// `ln` de la riqueza contra H0⁻ (media ≥ 0): `Σ_i (−u_j·x_i − u_j²/8)`.
    pub log_riqueza_negativa: [f64; APUESTAS],
}

/// Signo de una media que un e-proceso certifica.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Signo {
    Positivo,
    Negativo,
}

impl EProcesoVille {
    /// Apuesta una observación `x` de un intervalo de ancho 1. El intervalo y
    /// la apuesta se fijan antes de ver `x`: la apuesta es predecible.
    pub fn apostar(&mut self, x: f64) {
        let riquezas = self
            .log_riqueza_positiva
            .iter_mut()
            .zip(self.log_riqueza_negativa.iter_mut());
        for ((positiva, negativa), u) in riquezas.zip(REJILLA_APUESTAS) {
            let castigo = u * u / 8.0;
            *positiva += u * x - castigo;
            *negativa -= u * x + castigo;
        }
    }

    /// `(ln M⁺, ln M⁻)`: el logaritmo de la evidencia (e-valor) acumulada
    /// contra «la media no es positiva» y contra «la media no es negativa».
    ///
    /// `M⁺_n = (1/J)·Σ_j Π_i exp(u_j·x_i − u_j²/8)`, y `M⁻` con `−x_i`. Por el
    /// lema de Hoeffding, una variable de ancho 1 con media condicional
    /// `E[x_i | pasado] ≤ 0` cumple `E[exp(u·x_i − u²/8) | pasado] ≤ 1` para
    /// toda `u ≥ 0`. Así `M⁺` es una supermartingala no negativa con
    /// `M⁺_0 = 1` bajo H0⁺, y la mezcla de apuestas conserva la propiedad. La
    /// desigualdad maximal de Ville da `P(∃n: M⁺_n ≥ 1/α) ≤ α`, válida mirando
    /// en cada resolución. Supone resultados condicionalmente independientes
    /// entre intenciones (límite declarado: monedas correlacionadas no lo son).
    pub fn log_evidencia(&self) -> (f64, f64) {
        (
            log_media_exp(&self.log_riqueza_positiva),
            log_media_exp(&self.log_riqueza_negativa),
        )
    }

    /// El signo de la media si su evidencia alcanza `1/alfa_cola` y la del
    /// contrario no (las dos a la vez sería contradictorio). `None` si no hay
    /// evidencia o `alfa_cola` cae fuera de (0, 1).
    pub fn signo(&self, alfa_cola: f64) -> Option<Signo> {
        if !alfa_cola.is_finite() || alfa_cola <= 0.0 || alfa_cola >= 1.0 {
            return None;
        }
        let umbral = (1.0 / alfa_cola).ln();
        let (positiva, negativa) = self.log_evidencia();
        if !positiva.is_finite() || !negativa.is_finite() {
            return None;
        }
        match (positiva >= umbral, negativa >= umbral) {
            (true, false) => Some(Signo::Positivo),
            (false, true) => Some(Signo::Negativo),
            _ => None,
        }
    }
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
    /// RB-V1b — evidencia sobre el signo de E[r], con `x = r/ancho`.
    pub evidencia_r: EProcesoVille,
    /// RB-V1c — resoluciones de intenciones con promesa del riesgo.
    pub n_promesa: u64,
    /// RB-V1c — Σ promesa (R) de esas resoluciones.
    pub suma_promesa_r: f64,
    /// RB-V1c — Σ R realizado de esas mismas resoluciones.
    pub suma_r_prometida: f64,
    /// RB-V1c — evidencia sobre el signo de E[r − promesa], con
    /// `x = (r − promesa)/ancho` (la promesa es fija al registrar, así que
    /// `r − promesa` conserva el ancho de `r`).
    pub evidencia_brecha: EProcesoVille,
}

/// `ln((1/J)·Σ_j e^(l_j))` sin desbordamiento (se resta el máximo).
fn log_media_exp(l: &[f64; APUESTAS]) -> f64 {
    let max = l.iter().copied().fold(f64::NEG_INFINITY, f64::max);
    if !max.is_finite() {
        return f64::NAN;
    }
    let suma: f64 = l.iter().map(|&v| (v - max).exp()).sum();
    max + (suma / APUESTAS as f64).ln()
}

/// RB-V1c — qué dice la evidencia sobre la promesa del riesgo.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Promesa {
    SinEvidencia,
    /// Lo realizado queda por debajo de lo prometido: el EV del riesgo es
    /// optimista y valida operaciones que no pagan lo que dice.
    PrometeDeMas,
    /// Lo realizado supera lo prometido: el EV del riesgo es conservador.
    PrometeDeMenos,
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

    /// RB-V1b — `(ln M⁺, ln M⁻)` de E[r]: la evidencia contra «lo vetado no
    /// paga» y contra «lo vetado no pierde» (ver `EProcesoVille`).
    pub fn log_evidencia(&self) -> (f64, f64) {
        self.evidencia_r.log_evidencia()
    }

    /// RB-V1b — veredicto siempre válido: la fuente bloquea ganadoras si la
    /// evidencia contra «no paga» alcanza `1/alfa_cola`, ahorra dinero si la
    /// alcanza la de «no pierde», y no hay veredicto si ninguna.
    pub fn veredicto_ville(&self, alfa_cola: f64) -> Veredicto {
        match self.evidencia_r.signo(alfa_cola) {
            Some(Signo::Positivo) => Veredicto::BloqueaGanadoras,
            Some(Signo::Negativo) => Veredicto::AhorraDinero,
            None => Veredicto::SinEvidencia,
        }
    }

    /// RB-V1c — EV medio que prometió el riesgo, en R, sobre las resoluciones
    /// con promesa.
    pub fn promesa_media_r(&self) -> Option<f64> {
        (self.n_promesa > 0).then(|| self.suma_promesa_r / self.n_promesa as f64)
    }

    /// RB-V1c — brecha media realizado − prometido, en R (negativa: el riesgo
    /// promete de más).
    pub fn brecha_media_r(&self) -> Option<f64> {
        (self.n_promesa > 0)
            .then(|| (self.suma_r_prometida - self.suma_promesa_r) / self.n_promesa as f64)
    }

    /// RB-V1c — `(ln M⁺, ln M⁻)` de E[r − promesa]: la evidencia contra «el
    /// riesgo no promete de menos» y contra «no promete de más».
    pub fn log_evidencia_promesa(&self) -> (f64, f64) {
        self.evidencia_brecha.log_evidencia()
    }

    /// RB-V1c — veredicto siempre válido sobre la promesa del riesgo.
    pub fn veredicto_promesa(&self, alfa_cola: f64) -> Promesa {
        match self.evidencia_brecha.signo(alfa_cola) {
            Some(Signo::Negativo) => Promesa::PrometeDeMas,
            Some(Signo::Positivo) => Promesa::PrometeDeMenos,
            None => Promesa::SinEvidencia,
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
                s.evidencia_r.apostar(r / ancho);
                if let Some(promesa) = i.promesa_r {
                    s.n_promesa += 1;
                    s.suma_promesa_r += promesa;
                    s.suma_r_prometida += r;
                    s.evidencia_brecha.apostar((r - promesa) / ancho);
                }
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
