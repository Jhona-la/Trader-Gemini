//! DEMO-B (pedido «Demo en Testnet sin ambiguedad»; especificacion Demo §B, formulas de las
//! lineas 88-95; revision N1 de NVIDIA, cambio 4): PERDIDA DIARIA DERIVADA.
//!
//! Nucleo puro (aritmetica y transicion) y su persistencia. Ninguna cifra de la ruta de decision
//! esta escrita a mano: todo sale de las constantes y funciones del motor.
//!
//! - `q_D = max_i conservative_loss_q(WR_i, n_i)` del roster al corte ([`q_del_roster`]).
//! - `h = expected_loss_streak(q_D)`, `f_sizing = clamp_ruin(1, q_D)` (`ruin.rs`).
//! - `d_duro` = el tope patrimonial del freno duro vigente ([`tope_capital_del_freno`], la misma
//!   expresion que `drawdown::freno_duro`: `min(global_max_drawdown, drawdown_de_falsacion(c, α))`).
//! - `d_unidad = 1 - (1 - d_duro)^(1/h)`, `d_restante = max(0, 1 - P0 (1 - d_duro) / E0)`,
//!   `d_D = min(f_sizing, d_unidad, d_restante)`, `B_D = E0 d_D`.
//! - Perdida observada: `L_D(t) = max(0, E0 - (E(t) - F_D(t)))`, con `F_D` la suma de flujos
//!   externos USDT del dia (deposito +, retiro -), una vez por identidad del venue.
//! - `L_D >= B_D` (inclusive) enclava `DAY_LOSS_LOCKED` hasta el siguiente dia UTC acreditado:
//!   rebote, deposito, reinicio, cambio de llave o reloj regresivo no lo rearman. `B_D = 0`
//!   bloquea desde que se publica el dia.
//!
//! El ancla del dia (E0, P0) sale de dos observaciones CONSECUTIVAS del mismo lazo y la misma
//! cuenta que bracketean el corte 00:00:00 UTC: `E0 = max(anterior, siguiente)` (el mayor de los
//! dos lados del corte: la perdida del dia se mide desde arriba). Sin esas dos observaciones (el
//! proceso arranco despues del corte sin estado persistido de hoy, o la ultima observacion era
//! invalida) el dia queda `DAY_EQUITY_ND`: aperturas retenidas hasta el siguiente corte acreditado.

use crate::drawdown::{drawdown_de_falsacion, ALFA_FALSACION, FRACCION_KELLY_MAXIMA};
use crate::ruin::{clamp_ruin, conservative_loss_q, expected_loss_streak};
use std::collections::BTreeMap;

/// Duracion de un dia UTC en milisegundos (definicion del calendario, no un parametro).
pub const MS_POR_DIA_UTC: u64 = 24 * 60 * 60 * 1000;

/// Activo de la equity de la cuenta USD-M.
pub const ACTIVO_DE_LA_CUENTA: &str = "USDT";

/// Codigos de la decision (los que el host escribe en el rechazo de la entrada).
pub const CODIGO_DAY_LOSS_LOCKED: &str = "DAY_LOSS_LOCKED";
pub const CODIGO_DAY_EQUITY_ND: &str = "DAY_EQUITY_ND";

/// Dia UTC (dias desde la epoca) de un instante en ms.
#[inline]
pub fn dia_utc(ms: u64) -> u64 {
    ms / MS_POR_DIA_UTC
}

/// Por que el presupuesto o el estado del dia no se pueden acreditar (N/D).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CausaNd {
    EquityInvalida,
    PicoInvalido,
    PicoMenorQueEquity,
    QInvalido,
    SinRoster,
    SinAncla,
    SinEvidencia,
    FlujoContradictorio,
    FlujoOtroActivo,
    FlujoSinIdentidad,
    FlujoInvalido,
    CuentaDistinta,
    DiaDistinto,
    EstadoIlegible,
}

impl CausaNd {
    pub fn nombre(self) -> &'static str {
        match self {
            CausaNd::EquityInvalida => "EQUITY_INVALIDA",
            CausaNd::PicoInvalido => "PICO_INVALIDO",
            CausaNd::PicoMenorQueEquity => "PICO_MENOR_QUE_EQUITY",
            CausaNd::QInvalido => "Q_INVALIDO",
            CausaNd::SinRoster => "SIN_ROSTER",
            CausaNd::SinAncla => "SIN_ANCLA",
            CausaNd::SinEvidencia => "SIN_EVIDENCIA",
            CausaNd::FlujoContradictorio => "FLUJO_CONTRADICTORIO",
            CausaNd::FlujoOtroActivo => "FLUJO_OTRO_ACTIVO",
            CausaNd::FlujoSinIdentidad => "FLUJO_SIN_IDENTIDAD",
            CausaNd::FlujoInvalido => "FLUJO_INVALIDO",
            CausaNd::CuentaDistinta => "CUENTA_DISTINTA",
            CausaNd::DiaDistinto => "DIA_DISTINTO",
            CausaNd::EstadoIlegible => "ESTADO_ILEGIBLE",
        }
    }
}

/// El tope patrimonial del freno duro (`d*` acotado por el gen): la MISMA expresion que
/// `drawdown::freno_duro` (`tope_capital`).
pub fn tope_capital_del_freno(global_max_drawdown: f64) -> f64 {
    let tope_d3 = drawdown_de_falsacion(FRACCION_KELLY_MAXIMA, ALFA_FALSACION);
    if global_max_drawdown.is_finite() && global_max_drawdown > 0.0 && global_max_drawdown < 1.0 {
        global_max_drawdown.min(tope_d3)
    } else {
        tope_d3
    }
}

/// `q_D`: el mayor `conservative_loss_q` del roster al corte (racha mas larga, presupuesto
/// menor). Un simbolo sin cierres usa el caso frio del motor. Roster vacio o entradas no finitas:
/// N/D.
pub fn q_del_roster(evidencia: &[(f64, f64)]) -> Result<f64, CausaNd> {
    if evidencia.is_empty() {
        return Err(CausaNd::SinRoster);
    }
    let mut q = f64::NEG_INFINITY;
    for &(wr, n) in evidencia {
        if !n.is_finite() || n < 0.0 || (n >= 1.0 && !wr.is_finite()) {
            return Err(CausaNd::QInvalido);
        }
        q = q.max(conservative_loss_q(wr, n));
    }
    if q.is_finite() && q > 0.0 && q < 1.0 {
        Ok(q)
    } else {
        Err(CausaNd::QInvalido)
    }
}

/// El presupuesto de un dia, con cada termino de la derivacion (para el recibo).
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct PresupuestoDiario {
    pub q_d: f64,
    pub h: f64,
    pub f_sizing: f64,
    pub d_duro: f64,
    pub d_unidad: f64,
    pub d_restante: f64,
    /// Fraccion de E0 presupuestada para el dia.
    pub d_d: f64,
    /// Presupuesto en USDT: `E0 * d_D`.
    pub b_d: f64,
}

/// LA FUNCION PURA de las lineas 88-95 de la especificacion.
pub fn presupuesto_diario(e0: f64, p0: f64, q_d: f64, global_max_drawdown: f64) -> Result<PresupuestoDiario, CausaNd> {
    if !(e0.is_finite() && e0 > 0.0) {
        return Err(CausaNd::EquityInvalida);
    }
    if !p0.is_finite() || p0 <= 0.0 {
        return Err(CausaNd::PicoInvalido);
    }
    if p0 < e0 {
        return Err(CausaNd::PicoMenorQueEquity);
    }
    if !(q_d.is_finite() && q_d > 0.0 && q_d < 1.0) {
        return Err(CausaNd::QInvalido);
    }
    let h = expected_loss_streak(q_d);
    let f_sizing = clamp_ruin(1.0, q_d);
    let d_duro = tope_capital_del_freno(global_max_drawdown);
    let d_unidad = 1.0 - (1.0 - d_duro).powf(1.0 / h);
    let d_restante = (1.0 - p0 * (1.0 - d_duro) / e0).max(0.0);
    let d_d = f_sizing.min(d_unidad).min(d_restante);
    if !(d_d.is_finite() && d_d >= 0.0) {
        return Err(CausaNd::QInvalido);
    }
    Ok(PresupuestoDiario { q_d, h, f_sizing, d_duro, d_unidad, d_restante, d_d, b_d: e0 * d_d })
}

/// La decision de la guarda diaria para una apertura.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DecisionDiaria {
    Permitida,
    DayLossLocked,
    DayEquityNd(CausaNd),
}

impl DecisionDiaria {
    pub fn codigo(self) -> Option<&'static str> {
        match self {
            DecisionDiaria::Permitida => None,
            DecisionDiaria::DayLossLocked => Some(CODIGO_DAY_LOSS_LOCKED),
            DecisionDiaria::DayEquityNd(_) => Some(CODIGO_DAY_EQUITY_ND),
        }
    }
}

/// El estado de un dia de una cuenta: ancla, presupuesto congelado, flujos y latch.
#[derive(Debug, Clone, PartialEq)]
pub struct EstadoDiario {
    pub cuenta: u64,
    pub dia: u64,
    pub e0: f64,
    pub p0: f64,
    pub d_d: f64,
    pub b_d: f64,
    /// Flujos externos USDT del dia por identidad del venue (`tranId`).
    pub flujos: BTreeMap<u64, f64>,
    pub bloqueado: bool,
    /// Un flujo que no se pudo clasificar deja el dia en N/D (aperturas retenidas).
    pub nd: Option<CausaNd>,
}

impl EstadoDiario {
    /// Abre el dia con su ancla y congela el presupuesto. `B_D = 0` bloquea desde ya.
    pub fn abrir(cuenta: u64, dia: u64, e0: f64, p0: f64, q_d: f64, global_max_drawdown: f64) -> Result<(Self, PresupuestoDiario), CausaNd> {
        let p = presupuesto_diario(e0, p0, q_d, global_max_drawdown)?;
        Ok((
            Self {
                cuenta,
                dia,
                e0,
                p0,
                d_d: p.d_d,
                b_d: p.b_d,
                flujos: BTreeMap::new(),
                bloqueado: p.b_d <= 0.0,
                nd: None,
            },
            p,
        ))
    }

    /// `F_D`: suma de los flujos del dia.
    pub fn flujo_neto(&self) -> f64 {
        self.flujos.values().sum()
    }

    /// `L_D(E) = max(0, E0 - (E - F_D))`; `None` con equity no finita.
    pub fn perdida(&self, equity: f64) -> Option<f64> {
        if !equity.is_finite() {
            return None;
        }
        Some((self.e0 - (equity - self.flujo_neto())).max(0.0))
    }

    /// Un flujo externo del dia. Misma identidad y mismo importe: idempotente. Identidad
    /// reutilizada con otro importe, sin identidad, otro activo o importe invalido: N/D.
    pub fn registrar_flujo(&mut self, id: u64, activo: &str, importe: f64) -> Result<(), CausaNd> {
        let causa = if id == 0 {
            Some(CausaNd::FlujoSinIdentidad)
        } else if !activo.eq_ignore_ascii_case(ACTIVO_DE_LA_CUENTA) {
            Some(CausaNd::FlujoOtroActivo)
        } else if !(importe.is_finite() && importe != 0.0) {
            Some(CausaNd::FlujoInvalido)
        } else {
            match self.flujos.get(&id) {
                Some(previo) if previo.to_bits() == importe.to_bits() => return Ok(()),
                Some(_) => Some(CausaNd::FlujoContradictorio),
                None => {
                    self.flujos.insert(id, importe);
                    return Ok(());
                }
            }
        };
        match causa {
            Some(c) => {
                self.nd.get_or_insert(c);
                Err(c)
            }
            None => Ok(()),
        }
    }

    /// Decide con la equity de ahora. El latch, una vez puesto, no se levanta en el dia.
    pub fn evaluar(&mut self, equity: f64) -> DecisionDiaria {
        if self.bloqueado {
            return DecisionDiaria::DayLossLocked;
        }
        if let Some(c) = self.nd {
            return DecisionDiaria::DayEquityNd(c);
        }
        match self.perdida(equity) {
            None => DecisionDiaria::DayEquityNd(CausaNd::EquityInvalida),
            Some(l) if l >= self.b_d => {
                self.bloqueado = true;
                DecisionDiaria::DayLossLocked
            }
            Some(_) => DecisionDiaria::Permitida,
        }
    }

    /// Texto persistido: una clave por linea; los f64 por sus bits (ida y vuelta exacta).
    pub fn serializar(&self) -> String {
        let mut s = String::from("perdida_diaria v1\n");
        s.push_str(&format!("cuenta={:016x}\n", self.cuenta));
        s.push_str(&format!("dia={}\n", self.dia));
        for (k, v) in [("e0", self.e0), ("p0", self.p0), ("d_d", self.d_d), ("b_d", self.b_d)] {
            s.push_str(&format!("{k}={:016x}\n", v.to_bits()));
        }
        s.push_str(&format!("bloqueado={}\n", u8::from(self.bloqueado)));
        s.push_str(&format!("nd={}\n", self.nd.map_or("-", CausaNd::nombre)));
        for (id, v) in &self.flujos {
            s.push_str(&format!("flujo={id}:{:016x}\n", v.to_bits()));
        }
        s
    }

    pub fn deserializar(texto: &str) -> Result<Self, CausaNd> {
        let ilegible = CausaNd::EstadoIlegible;
        let mut lineas = texto.lines();
        if lineas.next() != Some("perdida_diaria v1") {
            return Err(ilegible);
        }
        let bits = |v: &str| u64::from_str_radix(v, 16).map(f64::from_bits).map_err(|_| ilegible);
        let (mut cuenta, mut dia, mut e0, mut p0, mut d_d, mut b_d, mut bloqueado, mut nd) =
            (None, None, None, None, None, None, None, None);
        let mut flujos = BTreeMap::new();
        for l in lineas {
            let (k, v) = l.split_once('=').ok_or(ilegible)?;
            match k {
                "cuenta" => cuenta = Some(u64::from_str_radix(v, 16).map_err(|_| ilegible)?),
                "dia" => dia = Some(v.parse::<u64>().map_err(|_| ilegible)?),
                "e0" => e0 = Some(bits(v)?),
                "p0" => p0 = Some(bits(v)?),
                "d_d" => d_d = Some(bits(v)?),
                "b_d" => b_d = Some(bits(v)?),
                "bloqueado" => {
                    bloqueado = Some(match v {
                        "0" => false,
                        "1" => true,
                        _ => return Err(ilegible),
                    })
                }
                "nd" => {
                    nd = Some(if v == "-" {
                        None
                    } else {
                        Some(
                            [
                                CausaNd::FlujoContradictorio,
                                CausaNd::FlujoOtroActivo,
                                CausaNd::FlujoSinIdentidad,
                                CausaNd::FlujoInvalido,
                            ]
                            .into_iter()
                            .find(|c| c.nombre() == v)
                            .ok_or(ilegible)?,
                        )
                    })
                }
                "flujo" => {
                    let (id, f) = v.split_once(':').ok_or(ilegible)?;
                    let id = id.parse::<u64>().map_err(|_| ilegible)?;
                    if flujos.insert(id, bits(f)?).is_some() {
                        return Err(ilegible);
                    }
                }
                _ => return Err(ilegible),
            }
        }
        let estado = Self {
            cuenta: cuenta.ok_or(ilegible)?,
            dia: dia.ok_or(ilegible)?,
            e0: e0.ok_or(ilegible)?,
            p0: p0.ok_or(ilegible)?,
            d_d: d_d.ok_or(ilegible)?,
            b_d: b_d.ok_or(ilegible)?,
            flujos,
            bloqueado: bloqueado.ok_or(ilegible)?,
            nd: nd.ok_or(ilegible)?,
        };
        let finitos = [estado.e0, estado.p0, estado.d_d, estado.b_d].iter().all(|v| v.is_finite());
        if !finitos || estado.e0 <= 0.0 || estado.b_d < 0.0 || estado.flujos.values().any(|v| !v.is_finite()) {
            return Err(ilegible);
        }
        Ok(estado)
    }
}

/// Escritura atomica e idempotente: temporal en la misma carpeta y renombrado. El mismo estado
/// produce los mismos bytes.
pub fn guardar_estado(ruta: &std::path::Path, estado: &EstadoDiario) -> std::io::Result<()> {
    if let Some(dir) = ruta.parent() {
        std::fs::create_dir_all(dir)?;
    }
    let tmp = ruta.with_extension("tmp");
    std::fs::write(&tmp, estado.serializar())?;
    std::fs::rename(&tmp, ruta)
}

/// Lo que hay en disco para esta cuenta y este dia. Otra cuenta u otro dia nunca se adoptan.
pub fn cargar_estado(ruta: &std::path::Path, cuenta: u64, dia: u64) -> Result<Option<EstadoDiario>, CausaNd> {
    let texto = match std::fs::read_to_string(ruta) {
        Ok(t) => t,
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => return Ok(None),
        Err(_) => return Err(CausaNd::EstadoIlegible),
    };
    let estado = EstadoDiario::deserializar(&texto)?;
    if estado.cuenta != cuenta {
        return Err(CausaNd::CuentaDistinta);
    }
    if estado.dia != dia {
        return Ok(None);
    }
    Ok(Some(estado))
}

/// Lo que paso al observar.
#[derive(Debug, Clone, PartialEq)]
pub enum Transicion {
    /// Sin cambio de estado.
    Nada,
    /// Dia nuevo acreditado: presupuesto congelado (para el recibo).
    DiaAbierto(PresupuestoDiario),
    /// El latch se puso en esta observacion.
    Bloqueado,
    /// El dia empezo sin ancla acreditable (o el ancla no se pudo abrir).
    SinAncla(CausaNd),
    /// Reloj regresivo: la observacion se ignora y el estado se conserva.
    RelojRegresivo,
}

/// La guarda de una cuenta: el dia vigente, la ultima observacion del lazo y la ultima evidencia
/// del roster publicada por el nucleo.
#[derive(Debug, Clone)]
pub struct GuardaDiaria {
    pub cuenta: u64,
    pub estado: Option<EstadoDiario>,
    /// El estado guardado de hoy era ilegible: el dia queda N/D (no se sobrescribe hasta el
    /// siguiente corte acreditado).
    pub ilegible_hoy: Option<u64>,
    ultima: Option<(u64, f64)>,
    evidencia: Option<Vec<(f64, f64)>>,
}

impl GuardaDiaria {
    /// Recupera el estado de hoy de esta cuenta (reinicio a mitad del dia: misma base y latch).
    pub fn recuperar(cuenta: u64, ruta: &std::path::Path, ahora_ms: u64) -> Self {
        let dia = dia_utc(ahora_ms);
        let (estado, ilegible_hoy) = match cargar_estado(ruta, cuenta, dia) {
            Ok(e) => (e, None),
            // Otra cuenta en la ruta de esta: nada se adopta y el dia queda sin ancla.
            Err(CausaNd::CuentaDistinta) => (None, None),
            Err(_) => (None, Some(dia)),
        };
        Self { cuenta, estado, ilegible_hoy, ultima: None, evidencia: None }
    }

    /// Evidencia del roster (win rate, n) publicada por el nucleo; se congela al abrir el dia.
    pub fn publicar_evidencia(&mut self, evidencia: Vec<(f64, f64)>) {
        self.evidencia = Some(evidencia);
    }

    /// Una observacion del lazo del inmune: equity del freno y pico del mismo paso. Abre el dia
    /// si esta observacion y la anterior bracketean el corte; despues evalua el latch.
    pub fn observar(&mut self, ahora_ms: u64, equity: f64, pico: f64, global_max_drawdown: f64) -> Transicion {
        if let Some((t_previo, _)) = self.ultima {
            if ahora_ms < t_previo {
                return Transicion::RelojRegresivo;
            }
        }
        let dia = dia_utc(ahora_ms);
        let mut transicion = Transicion::Nada;
        let vigente = self.estado.as_ref().is_some_and(|e| e.dia == dia);
        if !vigente {
            // Un dia viejo nunca rige el de hoy: sin ancla nueva, N/D.
            self.estado = None;
            let anterior = self.ultima.filter(|(t, e)| dia_utc(*t) + 1 == dia && e.is_finite());
            transicion = match (anterior, self.ilegible_hoy == Some(dia)) {
                (_, true) => Transicion::SinAncla(CausaNd::EstadoIlegible),
                (None, false) => Transicion::SinAncla(CausaNd::SinAncla),
                (Some((_, e_previa)), false) => {
                    if !equity.is_finite() {
                        Transicion::SinAncla(CausaNd::EquityInvalida)
                    } else {
                        let e0 = e_previa.max(equity);
                        match self
                            .evidencia
                            .as_deref()
                            .ok_or(CausaNd::SinEvidencia)
                            .and_then(q_del_roster)
                            .and_then(|q| EstadoDiario::abrir(self.cuenta, dia, e0, pico, q, global_max_drawdown))
                        {
                            Ok((estado, p)) => {
                                self.estado = Some(estado);
                                Transicion::DiaAbierto(p)
                            }
                            Err(c) => Transicion::SinAncla(c),
                        }
                    }
                }
            };
        }
        self.ultima = Some((ahora_ms, equity));
        if let Some(estado) = self.estado.as_mut() {
            let antes = estado.bloqueado;
            if estado.evaluar(equity) == DecisionDiaria::DayLossLocked && !antes && transicion == Transicion::Nada {
                transicion = Transicion::Bloqueado;
            }
        }
        transicion
    }

    /// Un flujo externo de la cuenta con su instante. Los del corte o anteriores pertenecen al
    /// ancla; sin dia vigente no hay ledger que ajustar.
    pub fn registrar_flujo(&mut self, tiempo_ms: u64, id: u64, activo: &str, importe: f64) -> Result<(), CausaNd> {
        match self.estado.as_mut() {
            Some(e) if tiempo_ms > e.dia * MS_POR_DIA_UTC && dia_utc(tiempo_ms) == e.dia => {
                e.registrar_flujo(id, activo, importe)
            }
            _ => Ok(()),
        }
    }

    /// La decision para una apertura ahora, con la equity de ahora. Enclava si toca el suelo.
    pub fn permite_apertura(&mut self, ahora_ms: u64, equity: f64) -> DecisionDiaria {
        let dia = dia_utc(ahora_ms);
        match self.estado.as_mut() {
            Some(e) if e.dia == dia => e.evaluar(equity),
            Some(_) => DecisionDiaria::DayEquityNd(CausaNd::DiaDistinto),
            None if self.ilegible_hoy == Some(dia) => DecisionDiaria::DayEquityNd(CausaNd::EstadoIlegible),
            None => DecisionDiaria::DayEquityNd(CausaNd::SinAncla),
        }
    }
}
