//! CL-108 (C-F8 en su parte de estado; decisión del dueño «Guardar todo»,
//! 2026-10-12) — EL ESTADO APRENDIDO SOBREVIVE AL REINICIO.
//!
//! Antes, cada reinicio del host empezaba de cero:
//! - el recuento de cierres por moneda (`trade_count`): todas las monedas,
//!   también las que tienen modelo `_MOTOR`, volvían a la fase de sonda de
//!   D4, y la cartera abría una sola posición nueva a la vez durante unos 8
//!   días con 26 monedas;
//! - la evidencia por moneda (acierto, PF, Kelly, PnL);
//! - los calibradores de Platt (por moneda y global), que deciden la
//!   probabilidad con la que el riesgo juzga el valor esperado;
//! - el freno de la estrategia (proceso e de Ville, CL-107): un reinicio
//!   borraba la evidencia contra H0, o soltaba un freno ya echado;
//! - el riesgo y la geometría medidos de las órdenes, de los que el freno
//!   duro (CL-106) deriva su umbral.
//!
//! Qué se guarda y cómo:
//! - por moneda, por SÍMBOLO (el universo puede cambiar de orden entre
//!   arranques): las métricas y las últimas `calibration::WINDOW`
//!   observaciones del calibrador;
//! - las del calibrador global, el freno de la estrategia y lo medido.
//!
//! Los calibradores de Platt guardan sus datos en campos privados
//! (`calibration.rs`, que no es de la Línea C). Por eso el núcleo los
//! alimenta por [`GodEngineCore::alimentar_calibradores`], que además anota
//! cada observación. Al restaurar se reproducen en el mismo orden: el ajuste
//! de Platt (logística con cresta) es convexo, así que los coeficientes
//! salen los mismos salvo la tolerancia de su Newton.
//!
//! El host guarda el estado en cada cierre (`data/{demo|prod}/
//! estado_aprendido.json`, escritura atómica) y lo restaura al arrancar,
//! antes de operar. Para empezar de cero (otra estrategia es otra
//! hipótesis), se borra el archivo.
//!
//! No se guarda todavía: la evidencia por rama (`TasaAcierto`), los
//! calibradores conformales, los pesos del ensamble ni los del perceptrón.
//! Tampoco cambia el ML-GATE: una moneda sin `_MOTOR` sigue necesitando un
//! modelo después de su primera sonda.

use crate::calibration::{PlattCalibrator, WINDOW};
use crate::GodEngineCore;
use serde::{Deserialize, Serialize};
use std::collections::VecDeque;
use std::sync::atomic::Ordering::Relaxed;

/// Versión del formato. Un archivo de otra versión no se restaura.
pub const VERSION_ESTADO: u32 = 1;

/// Una observación del calibrador: (puntuación cruda, ganó, instante de evento).
pub type Observacion = (f64, bool, u64);

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct MetricasGuardadas {
    pub trade_count: usize,
    pub win_rate: f64,
    pub profit_factor: f64,
    pub kelly_fraction: f64,
    pub gross_wins: f64,
    pub gross_losses: f64,
    pub pnl_realized: f64,
    pub pnl_gross: f64,
    pub roi_pre_fee: f64,
    pub roi_post_fee: f64,
    pub zombie_promotions: usize,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct MonedaGuardada {
    pub simbolo: String,
    pub metricas: MetricasGuardadas,
    pub calibrador: Vec<Observacion>,
}

/// El proceso e del freno de la estrategia y si ya rechazó.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct FrenoGuardado {
    pub e_value: f64,
    pub log_wealth: f64,
    pub peak_e_value: f64,
    pub count: usize,
    pub running_mean: f64,
    pub running_sq_mean: f64,
    pub last_lambda: f64,
    pub rechazada: bool,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct EstadoAprendido {
    pub version: u32,
    pub monedas: Vec<MonedaGuardada>,
    pub calibrador_global: Vec<Observacion>,
    pub freno_estrategia: FrenoGuardado,
    pub riesgo_por_operacion: f64,
    pub ganancia_en_stops: f64,
    pub perdida_en_stops: f64,
}

/// Anota una observación en un registro acotado a la ventana del calibrador.
fn anotar(registro: &mut VecDeque<Observacion>, obs: Observacion) {
    if registro.len() == WINDOW {
        registro.pop_front();
    }
    registro.push_back(obs);
}

/// La misma validación que `PlattCalibrator::update_at`: lo que el
/// calibrador descarta no se anota.
fn puntuacion_valida(score: f64) -> bool {
    score.is_finite() && (0.0..=1.0).contains(&score)
}

/// [`GodEngineCore::alimentar_calibradores`] sobre los campos sueltos, para
/// el camino de cierre del núcleo, que mantiene prestado el arena.
pub(crate) fn alimentar(
    por_moneda: &mut [PlattCalibrator],
    registro: &mut [VecDeque<Observacion>],
    global: &mut PlattCalibrator,
    registro_global: &mut VecDeque<Observacion>,
    (coin_id, score, won, ts_ms): (usize, f64, bool, u64),
) {
    if !puntuacion_valida(score) {
        return;
    }
    if let Some(c) = por_moneda.get_mut(coin_id) {
        c.update_at(score, won, ts_ms);
        if let Some(r) = registro.get_mut(coin_id) {
            anotar(r, (score, won, ts_ms));
        }
    }
    global.update_at(score, won, ts_ms);
    anotar(registro_global, (score, won, ts_ms));
}

impl GodEngineCore {
    /// Alimenta el calibrador de la moneda y el global con un resultado, y
    /// anota la observación para poder restaurarlos tras un reinicio.
    pub fn alimentar_calibradores(&mut self, coin_id: usize, score: f64, won: bool, ts_ms: u64) {
        alimentar(
            &mut self.calibrator_by_coin,
            &mut self.registro_calibrador,
            &mut self.confidence_calibrator,
            &mut self.registro_calibrador_global,
            (coin_id, score, won, ts_ms),
        );
    }

    /// Foto del estado aprendido, por símbolo.
    pub fn estado_aprendido(&self) -> EstadoAprendido {
        let mut monedas = Vec::new();
        for (coin_id, coin) in self.arena.coins.iter().enumerate() {
            let Some(simbolo) = quantum_arena::symbol_registry::try_symbol(coin_id) else {
                continue;
            };
            let m = &coin.metrics;
            monedas.push(MonedaGuardada {
                simbolo,
                metricas: MetricasGuardadas {
                    trade_count: m.trade_count.load(Relaxed),
                    win_rate: m.win_rate.load(Relaxed),
                    profit_factor: m.profit_factor.load(Relaxed),
                    kelly_fraction: m.kelly_fraction.load(Relaxed),
                    gross_wins: m.gross_wins.load(Relaxed),
                    gross_losses: m.gross_losses.load(Relaxed),
                    pnl_realized: m.pnl_realized.load(Relaxed),
                    pnl_gross: m.pnl_gross.load(Relaxed),
                    roi_pre_fee: m.roi_pre_fee.load(Relaxed),
                    roi_post_fee: m.roi_post_fee.load(Relaxed),
                    zombie_promotions: m.zombie_promotions.load(Relaxed),
                },
                calibrador: self
                    .registro_calibrador
                    .get(coin_id)
                    .map(|r| r.iter().copied().collect())
                    .unwrap_or_default(),
            });
        }
        let v = &self.risk_engine.freno_estrategia.ville;
        EstadoAprendido {
            version: VERSION_ESTADO,
            monedas,
            calibrador_global: self.registro_calibrador_global.iter().copied().collect(),
            freno_estrategia: FrenoGuardado {
                e_value: v.e_value,
                log_wealth: v.log_wealth,
                peak_e_value: v.peak_e_value,
                count: v.count,
                running_mean: v.running_mean,
                running_sq_mean: v.running_sq_mean,
                last_lambda: v.last_lambda,
                rechazada: self.risk_engine.freno_estrategia.rechazada,
            },
            riesgo_por_operacion: self.arena.riesgo_por_operacion.load(Relaxed),
            ganancia_en_stops: self.arena.ganancia_en_stops.load(Relaxed),
            perdida_en_stops: self.arena.perdida_en_stops.load(Relaxed),
        }
    }

    /// Restaura un estado guardado. Las monedas se buscan por símbolo en el
    /// universo de este arranque; las que ya no están se ignoran. Devuelve
    /// cuántas monedas se restauraron.
    pub fn restaurar_estado_aprendido(&mut self, e: &EstadoAprendido) -> usize {
        if e.version != VERSION_ESTADO {
            return 0;
        }
        let mut restauradas = 0;
        for g in &e.monedas {
            let Some(coin_id) = quantum_arena::symbols::get_coin_id(&g.simbolo) else {
                continue;
            };
            let Some(coin) = self.arena.coins.get(coin_id) else {
                continue;
            };
            let m = &coin.metrics;
            let s = &g.metricas;
            m.trade_count.store(s.trade_count, Relaxed);
            m.win_rate.store(s.win_rate, Relaxed);
            m.profit_factor.store(s.profit_factor, Relaxed);
            m.kelly_fraction.store(s.kelly_fraction, Relaxed);
            m.gross_wins.store(s.gross_wins, Relaxed);
            m.gross_losses.store(s.gross_losses, Relaxed);
            m.pnl_realized.store(s.pnl_realized, Relaxed);
            m.pnl_gross.store(s.pnl_gross, Relaxed);
            m.roi_pre_fee.store(s.roi_pre_fee, Relaxed);
            m.roi_post_fee.store(s.roi_post_fee, Relaxed);
            m.zombie_promotions.store(s.zombie_promotions, Relaxed);
            if let Some(c) = self.calibrator_by_coin.get_mut(coin_id) {
                *c = PlattCalibrator::new();
                let r = &mut self.registro_calibrador[coin_id];
                r.clear();
                for &(score, won, ts) in g.calibrador.iter().filter(|o| puntuacion_valida(o.0)) {
                    c.update_at(score, won, ts);
                    anotar(r, (score, won, ts));
                }
            }
            restauradas += 1;
        }
        self.confidence_calibrator = PlattCalibrator::new();
        self.registro_calibrador_global.clear();
        for &(score, won, ts) in e.calibrador_global.iter().filter(|o| puntuacion_valida(o.0)) {
            self.confidence_calibrator.update_at(score, won, ts);
            anotar(&mut self.registro_calibrador_global, (score, won, ts));
        }
        let f = &e.freno_estrategia;
        let freno = &mut self.risk_engine.freno_estrategia;
        freno.ville.e_value = f.e_value;
        freno.ville.log_wealth = f.log_wealth;
        freno.ville.peak_e_value = f.peak_e_value;
        freno.ville.count = f.count;
        freno.ville.running_mean = f.running_mean;
        freno.ville.running_sq_mean = f.running_sq_mean;
        freno.ville.last_lambda = f.last_lambda;
        freno.rechazada = f.rechazada;
        for (celda, valor) in [
            (&self.arena.riesgo_por_operacion, e.riesgo_por_operacion),
            (&self.arena.ganancia_en_stops, e.ganancia_en_stops),
            (&self.arena.perdida_en_stops, e.perdida_en_stops),
        ] {
            if valor.is_finite() && valor > 0.0 {
                celda.store(valor, Relaxed);
            }
        }
        restauradas
    }
}

/// Guarda el estado (escritura atómica: tmp + rename).
pub fn guardar(ruta: &std::path::Path, e: &EstadoAprendido) -> std::io::Result<()> {
    let texto = serde_json::to_string(e).map_err(std::io::Error::other)?;
    if let Some(dir) = ruta.parent() {
        std::fs::create_dir_all(dir)?;
    }
    let tmp = ruta.with_extension("tmp");
    std::fs::write(&tmp, texto)?;
    std::fs::rename(&tmp, ruta)
}

/// Lee un estado guardado; `None` si no existe, está corrupto o es de otra
/// versión.
pub fn cargar(ruta: &std::path::Path) -> Option<EstadoAprendido> {
    let texto = std::fs::read_to_string(ruta).ok()?;
    let e: EstadoAprendido = serde_json::from_str(&texto).ok()?;
    (e.version == VERSION_ESTADO).then_some(e)
}
