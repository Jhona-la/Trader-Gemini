//! CL-108 (C-F8 en su parte de estado; decisión del dueño «Guardar todo»,
//! 2026-10-12) — EL ESTADO APRENDIDO SOBREVIVE AL REINICIO.
//!
//! Antes, cada reinicio del host empezaba de cero:
//! - el recuento de cierres por moneda (`trade_count`): todas las monedas,
//!   también las que tienen modelo `_MOTOR`, volvían a la fase de sonda de
//!   D4, y la cartera abría una sola posición nueva a la vez durante unos 8
//!   días con 26 monedas;
//! - la evidencia por moneda (acierto, PF, Kelly, PnL);
//!   el PnL de las sesiones anteriores se conserva aparte (CL-108d): los
//!   acumuladores `pnl_realized`/`pnl_gross` son de la sesión;
//! - los calibradores de Platt (por moneda y global), que deciden la
//!   probabilidad con la que el riesgo juzga el valor esperado;
//! - el freno de la estrategia (proceso e de Ville, CL-107): un reinicio
//!   borraba la evidencia contra H0, o soltaba un freno ya echado;
//! - el riesgo y la geometría medidos de las órdenes, de los que el freno
//!   duro (CL-106) deriva su umbral, y la caída del freno duro en stops de
//!   sus posiciones (CL-130).
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
//! salen los mismos salvo la tolerancia de su Newton. LH-12 conserva además
//! el reloj máximo: un cierre tardío puede expulsar de la ventana la fila
//! que lo fijó, sin hacer retroceder el reloj del calibrador.
//!
//! El host guarda el estado en cada cierre (`data/{demo|prod}/
//! estado_aprendido.json`, escritura atómica y sincronizada) y lo restaura
//! al arrancar, antes de operar. Para empezar de cero (otra estrategia es
//! otra hipótesis), se borra el archivo. Un archivo ilegible o de otra
//! versión NO es un archivo ausente (CL-108f): el freno de la estrategia
//! queda echado y la sesión no lo sobrescribe ([`Arranque::Bloqueado`]).
//!
//! También, por moneda (CL-108e): el sesgo y el umbral epigenéticos con su
//! ancla (el último cierre) y el anillo de siniestros de Cramér-Lundberg; y
//! (LG-27) el detector e-CUSUM de pérdida de ventaja (LG-22). Límites del
//! detector guardado: un binario anterior a LG-27 lee el archivo e ignora el
//! campo, y su primer guardado lo borra; una moneda que sale del universo
//! pierde su estado al primer guardado (como todo lo de CL-108); el host sólo
//! guarda tras cierres con la operación permitida.
//!
//! No se guarda todavía: la evidencia por rama (`TasaAcierto`), los
//! calibradores conformales, la ganancia epigenética de las 32 escalas del
//! espectro, los pesos del ensamble ni los del perceptrón.
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
    /// PnL realizado y bruto de TODAS las sesiones (CL-108d).
    pub pnl_realized: f64,
    pub pnl_gross: f64,
    pub roi_pre_fee: f64,
    pub roi_post_fee: f64,
    pub zombie_promotions: usize,
}

/// CL-108e (T2-2 de la Línea G) — lo que el cierre enseña a la epigenética
/// de la moneda: el sesgo (multiplica la exposición en el riesgo y no se
/// olvida con el tiempo), el umbral (se relaja hacia 1 desde el último
/// cierre, así que va con su ancla) y el último cierre (también lo usa la
/// guarda anti-latigazo, D-463).
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct EpigeneticaGuardada {
    pub sesgo: f64,
    pub umbral: f64,
    pub ultimo_cierre_ms: u64,
    pub ultimo_cierre_largo: bool,
    pub ultimo_cierre_gano: bool,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct MonedaGuardada {
    pub simbolo: String,
    pub metricas: MetricasGuardadas,
    pub calibrador: Vec<Observacion>,
    /// LH-12: máximo conocido, no necesariamente presente en la ventana.
    /// Estados anteriores sin este campo infieren sólo el máximo retenido.
    #[serde(default)]
    pub calibrador_reloj_ms: Option<u64>,
    pub epigenetica: EpigeneticaGuardada,
    /// CL-108e — el anillo de siniestros de Cramér-Lundberg (retorno neto por
    /// nocional de cada cierre), del más viejo al más nuevo.
    pub siniestros: Vec<f64>,
    /// LG-27 (B4 de la revisión del PR #62) — el detector e-CUSUM de pérdida
    /// de ventaja de la moneda (LG-22). Estados anteriores sin este campo
    /// arrancan el detector de cero, como antes.
    #[serde(default)]
    pub detector_lg22: Option<crate::resultado_orden::DetectorGuardado>,
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
    #[serde(default)]
    pub calibrador_global_reloj_ms: Option<u64>,
    pub freno_estrategia: FrenoGuardado,
    pub riesgo_por_operacion: f64,
    pub riesgo_por_operacion_usd: f64,
    pub ganancia_en_stops: f64,
    pub perdida_en_stops: f64,
    /// CL-130 — la caída del freno duro en stops de sus posiciones: la suma
    /// de los resultados de los cierres y su máximo. Un estado anterior sin
    /// ellos empieza con caída 0 en stops.
    #[serde(default)]
    pub resultado_en_stops: f64,
    #[serde(default)]
    pub pico_en_stops: f64,
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

/// Cota inferior de R de Lundberg que se publica: z = 1,645 (95 %
/// unilateral, R7-R2-A-3), para que una muestra corta con racha ganadora no
/// infle R ni afloje el margen de ruina del grupo.
pub const LUNDBERG_Z: f64 = 1.645;

/// Un cierre alimenta el estimador de Cramér-Lundberg de su moneda y se
/// publica la cota (o su retirada, MG02). Es la misma función para el cierre
/// del núcleo y para la restauración.
pub(crate) fn observar_siniestro(
    est: &mut risk_engine::cramer_lundberg::EstimadorSiniestros,
    registry: &omniscient_registry::OmniscientRegistry,
    coin_id: usize,
    y: f64,
) {
    est.observar(y);
    crate::evidence_publication::publicar_lundberg(registry, coin_id, est.lundberg_lcb(LUNDBERG_Z));
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

    /// Registra el retorno neto por nocional de un cierre en el estimador de
    /// Cramér-Lundberg de la moneda y publica su cota (lo mismo que hace el
    /// cierre del núcleo).
    pub fn observar_siniestro(&mut self, coin_id: usize, y: f64) {
        if let Some(est) = self.lundberg_siniestros.get_mut(coin_id) {
            observar_siniestro(est, &self.arena.registry, coin_id, y);
        }
    }

    /// Foto del estado aprendido, por símbolo.
    pub fn estado_aprendido(&self) -> EstadoAprendido {
        let mut monedas = Vec::new();
        for (coin_id, coin) in self.arena.coins.iter().enumerate() {
            let Some(simbolo) = quantum_arena::symbol_registry::try_symbol(coin_id) else {
                continue;
            };
            let m = &coin.metrics;
            let (pnl_previo, pnl_bruto_previo) =
                self.pnl_previo.get(coin_id).copied().unwrap_or((0.0, 0.0));
            monedas.push(MonedaGuardada {
                simbolo,
                metricas: MetricasGuardadas {
                    trade_count: m.trade_count.load(Relaxed),
                    win_rate: m.win_rate.load(Relaxed),
                    profit_factor: m.profit_factor.load(Relaxed),
                    kelly_fraction: m.kelly_fraction.load(Relaxed),
                    gross_wins: m.gross_wins.load(Relaxed),
                    gross_losses: m.gross_losses.load(Relaxed),
                    pnl_realized: pnl_previo + m.pnl_realized.load(Relaxed),
                    pnl_gross: pnl_bruto_previo + m.pnl_gross.load(Relaxed),
                    roi_pre_fee: m.roi_pre_fee.load(Relaxed),
                    roi_post_fee: m.roi_post_fee.load(Relaxed),
                    zombie_promotions: m.zombie_promotions.load(Relaxed),
                },
                calibrador: self
                    .registro_calibrador
                    .get(coin_id)
                    .map(|r| r.iter().copied().collect())
                    .unwrap_or_default(),
                calibrador_reloj_ms: self.calibrator_by_coin.get(coin_id)
                    .and_then(PlattCalibrator::evidence_clock_ms),
                epigenetica: EpigeneticaGuardada {
                    sesgo: coin.epigenetic_bias.load(Relaxed),
                    umbral: coin.epigenetic_threshold_modifier.load(Relaxed),
                    ultimo_cierre_ms: coin.last_close_ts.load(Relaxed),
                    ultimo_cierre_largo: coin.last_close_is_long.load(Relaxed),
                    ultimo_cierre_gano: coin.last_close_was_win.load(Relaxed),
                },
                siniestros: self
                    .lundberg_siniestros
                    .get(coin_id)
                    .map(|e| e.observaciones())
                    .unwrap_or_default(),
                detector_lg22: self.lg22_detectores.get(coin_id).map(|d| d.guardado()),
            });
        }
        let v = &self.risk_engine.freno_estrategia.ville;
        EstadoAprendido {
            version: VERSION_ESTADO,
            monedas,
            calibrador_global: self.registro_calibrador_global.iter().copied().collect(),
            calibrador_global_reloj_ms: self.confidence_calibrator.evidence_clock_ms(),
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
            riesgo_por_operacion_usd: self.arena.riesgo_por_operacion_usd.load(Relaxed),
            ganancia_en_stops: self.arena.ganancia_en_stops.load(Relaxed),
            perdida_en_stops: self.arena.perdida_en_stops.load(Relaxed),
            resultado_en_stops: self.arena.resultado_en_stops.load(Relaxed),
            pico_en_stops: self.arena.pico_en_stops.load(Relaxed),
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
            // CL-108d: el PnL de las sesiones anteriores no entra en los
            // acumuladores de la sesión; se conserva aparte y se suma al
            // guardar.
            if let Some(p) = self.pnl_previo.get_mut(coin_id) {
                *p = (s.pnl_realized, s.pnl_gross);
            }
            m.roi_pre_fee.store(s.roi_pre_fee, Relaxed);
            m.roi_post_fee.store(s.roi_post_fee, Relaxed);
            m.zombie_promotions.store(s.zombie_promotions, Relaxed);
            let ep = &g.epigenetica;
            coin.epigenetic_bias.store(ep.sesgo, Relaxed);
            coin.epigenetic_threshold_modifier.store(ep.umbral, Relaxed);
            coin.last_close_ts.store(ep.ultimo_cierre_ms, Relaxed);
            coin.last_close_is_long.store(ep.ultimo_cierre_largo, Relaxed);
            coin.last_close_was_win.store(ep.ultimo_cierre_gano, Relaxed);
            if let Some(est) = self.lundberg_siniestros.get_mut(coin_id) {
                *est = risk_engine::cramer_lundberg::EstimadorSiniestros::new();
                for &y in &g.siniestros {
                    est.observar(y);
                }
                crate::evidence_publication::publicar_lundberg(
                    &self.arena.registry,
                    coin_id,
                    est.lundberg_lcb(LUNDBERG_Z),
                );
            }
            // LG-27: se continúa el mismo proceso sólo si el guardado es un
            // estado alcanzable y mide contra la misma H0 (la referencia μ);
            // si no, el detector de la moneda empieza de cero y queda constancia
            // en `lg22_detector_descartado`. Un detector inservible no bloquea
            // el resto del estado (no es un freno echado: tras una alarma el
            // detector ya se reinicia, y su efecto —el sesgo ×0,5— se guarda
            // aparte en la epigenética).
            if let (Some(guardado), Some(det)) =
                (&g.detector_lg22, self.lg22_detectores.get_mut(coin_id))
            {
                let mu = det.referencia();
                let continuado = crate::resultado_orden::DetectorPerdidaDeVentaja::desde_guardado(
                    guardado,
                )
                .filter(|r| (r.referencia() - mu).abs() <= 1e-12 * mu.abs().max(1.0));
                let descartado = continuado.is_none();
                *det = continuado.unwrap_or_else(|| {
                    crate::resultado_orden::DetectorPerdidaDeVentaja::con_referencia(mu)
                });
                let registro = &self.arena.registry;
                registro.set_for_coin(coin_id, "lg22_estadistico", det.estadistico());
                registro.set_for_coin(coin_id, "lg22_cierres", det.cierres_totales as f64);
                registro.set_for_coin(
                    coin_id,
                    "lg22_detector_descartado",
                    if descartado { 1.0 } else { 0.0 },
                );
            }
            if let Some(c) = self.calibrator_by_coin.get_mut(coin_id) {
                *c = PlattCalibrator::new();
                let r = &mut self.registro_calibrador[coin_id];
                r.clear();
                for &(score, won, ts) in g.calibrador.iter().filter(|o| puntuacion_valida(o.0)) {
                    c.update_at(score, won, ts);
                    anotar(r, (score, won, ts));
                }
                if let Some(clock) = g.calibrador_reloj_ms {
                    c.restore_evidence_clock_ms(clock);
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
        if let Some(clock) = e.calibrador_global_reloj_ms {
            self.confidence_calibrator.restore_evidence_clock_ms(clock);
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
            (&self.arena.riesgo_por_operacion_usd, e.riesgo_por_operacion_usd),
            (&self.arena.ganancia_en_stops, e.ganancia_en_stops),
            (&self.arena.perdida_en_stops, e.perdida_en_stops),
        ] {
            if valor.is_finite() && valor > 0.0 {
                celda.store(valor, Relaxed);
            }
        }
        // `validar` ya exigió los dos finitos y pico ≥ máx(0, suma).
        self.arena.resultado_en_stops.store(e.resultado_en_stops, Relaxed);
        self.arena.pico_en_stops.store(e.pico_en_stops, Relaxed);
        restauradas
    }

    /// CL-134 (R6 de la revisión del PR #62) — los frenos empiezan de cero
    /// con `capital`: el pico del arena y del riesgo, lo medido de las
    /// órdenes, la caída en stops y el freno de la estrategia. Lo usa el
    /// bosque sombra al replantar un universo, que es otra hipótesis.
    pub fn reiniciar_frenos(&mut self, capital: f64) {
        risk_engine::drawdown::reiniciar_la_cuenta(&self.arena, capital);
        self.risk_engine.reset(capital);
        // LG-27c: el detector de pérdida de ventaja (LG-22) también es un
        // freno: la evidencia contra la hipótesis vieja no se acredita a la
        // nueva. Conserva su referencia μ.
        for det in &mut self.lg22_detectores {
            *det = crate::resultado_orden::DetectorPerdidaDeVentaja::con_referencia(det.referencia());
        }
    }
}

/// CL-108f (T2-1 de la Línea G) — comprueba que un estado se puede guardar
/// y restaurar: todo número finito y dentro del rango que le da su escritor.
/// `Err` nombra el primer campo que falla.
pub fn validar(e: &EstadoAprendido) -> Result<(), String> {
    fn rango(nombre: &str, v: f64, lo: f64, hi: f64) -> Result<(), String> {
        if v.is_finite() && v >= lo && v <= hi {
            Ok(())
        } else {
            Err(format!("{nombre} = {v} fuera de [{lo}, {hi}]"))
        }
    }
    let todo = f64::MAX;
    for g in &e.monedas {
        let m = &g.metricas;
        let s = &g.simbolo;
        rango(&format!("{s}.win_rate"), m.win_rate, 0.0, 1.0)?;
        rango(&format!("{s}.profit_factor"), m.profit_factor, 0.0, todo)?;
        rango(&format!("{s}.gross_wins"), m.gross_wins, 0.0, todo)?;
        rango(&format!("{s}.gross_losses"), m.gross_losses, 0.0, todo)?;
        for (n, v) in [
            ("kelly_fraction", m.kelly_fraction),
            ("pnl_realized", m.pnl_realized),
            ("pnl_gross", m.pnl_gross),
            ("roi_pre_fee", m.roi_pre_fee),
            ("roi_post_fee", m.roi_post_fee),
        ] {
            rango(&format!("{s}.{n}"), v, -todo, todo)?;
        }
        // Los rangos que impone su escritor
        // (`CoinArena::apply_spectral_epigenetic_feedback_with_time`).
        rango(&format!("{s}.sesgo"), g.epigenetica.sesgo, 0.5, 2.0)?;
        rango(&format!("{s}.umbral"), g.epigenetica.umbral, 0.80, 1.25)?;
        if g.calibrador.iter().any(|o| !o.0.is_finite()) {
            return Err(format!("{s}.calibrador con una puntuación no finita"));
        }
        if g.calibrador_reloj_ms.is_some_and(|t| g.calibrador.iter().any(|o| o.2 > t)) {
            return Err(format!("{s}.calibrador_reloj_ms anterior a una observación"));
        }
        if g.siniestros.iter().any(|y| !y.is_finite()) {
            return Err(format!("{s}.siniestros con un valor no finito"));
        }
        // LG-27: el detector no se valida aquí. Uno inservible se descarta al
        // restaurar (la moneda empieza ese detector de cero) en vez de
        // bloquear el estado entero.
    }
    if e.calibrador_global.iter().any(|o| !o.0.is_finite()) {
        return Err("calibrador_global con una puntuación no finita".into());
    }
    if e.calibrador_global_reloj_ms.is_some_and(|t| e.calibrador_global.iter().any(|o| o.2 > t)) {
        return Err("calibrador_global_reloj_ms anterior a una observación".into());
    }
    let f = &e.freno_estrategia;
    rango("freno.e_value", f.e_value, 0.0, todo)?;
    rango("freno.peak_e_value", f.peak_e_value, 0.0, todo)?;
    for (n, v) in [
        ("freno.log_wealth", f.log_wealth),
        ("freno.running_mean", f.running_mean),
        ("freno.running_sq_mean", f.running_sq_mean),
        ("freno.last_lambda", f.last_lambda),
    ] {
        rango(n, v, -todo, todo)?;
    }
    for (n, v) in [
        ("riesgo_por_operacion", e.riesgo_por_operacion),
        ("riesgo_por_operacion_usd", e.riesgo_por_operacion_usd),
        ("ganancia_en_stops", e.ganancia_en_stops),
        ("perdida_en_stops", e.perdida_en_stops),
    ] {
        rango(n, v, 0.0, todo)?;
    }
    rango("resultado_en_stops", e.resultado_en_stops, -todo, todo)?;
    rango("pico_en_stops", e.pico_en_stops, e.resultado_en_stops.max(0.0), todo)?;
    Ok(())
}

/// Guarda el estado: comprueba que se podrá leer ([`validar`]), lo escribe
/// en un temporal sincronizado con el disco y lo renombra encima (un corte
/// de luz no deja el archivo a medias). Un estado inválido no se escribe:
/// el archivo anterior queda como estaba.
pub fn guardar(ruta: &std::path::Path, e: &EstadoAprendido) -> std::io::Result<()> {
    use std::io::Write;
    validar(e).map_err(|m| std::io::Error::new(std::io::ErrorKind::InvalidData, m))?;
    let texto = serde_json::to_string(e).map_err(std::io::Error::other)?;
    if let Some(dir) = ruta.parent() {
        std::fs::create_dir_all(dir)?;
    }
    let tmp = ruta.with_extension("tmp");
    {
        let mut f = std::fs::File::create(&tmp)?;
        f.write_all(texto.as_bytes())?;
        f.sync_all()?;
    }
    std::fs::rename(&tmp, ruta)
}

/// CL-108f (T2-1 de la Línea G) — lo que se encontró al leer el estado.
/// Un archivo ilegible no es un archivo ausente: tratarlo como ausente
/// soltaba el freno de la estrategia y, en el primer cierre, sobrescribía la
/// única copia de la evidencia.
#[derive(Debug)]
pub enum Carga {
    /// No hay archivo: primer arranque, o el operador lo borró para empezar
    /// de cero.
    Ausente,
    /// El archivo existe pero no se puede leer, está truncado o corrupto, o
    /// tiene un valor fuera de rango.
    Ilegible(String),
    /// El archivo es de otra versión del formato.
    OtraVersion(u32),
    Leido(EstadoAprendido),
}

/// Lee el estado guardado.
pub fn cargar(ruta: &std::path::Path) -> Carga {
    #[derive(Deserialize)]
    struct SoloVersion {
        version: u32,
    }
    let texto = match std::fs::read_to_string(ruta) {
        Ok(t) => t,
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => return Carga::Ausente,
        Err(e) => return Carga::Ilegible(format!("no se puede leer: {e}")),
    };
    match serde_json::from_str::<SoloVersion>(&texto) {
        Err(e) => return Carga::Ilegible(format!("JSON inválido: {e}")),
        Ok(v) if v.version != VERSION_ESTADO => return Carga::OtraVersion(v.version),
        Ok(_) => {}
    }
    let e: EstadoAprendido = match serde_json::from_str(&texto) {
        Ok(e) => e,
        Err(err) => return Carga::Ilegible(format!("JSON inválido: {err}")),
    };
    match validar(&e) {
        Ok(()) => Carga::Leido(e),
        Err(m) => Carga::Ilegible(m),
    }
}

/// Qué hizo el arranque con el estado guardado.
#[derive(Debug, PartialEq)]
pub enum Arranque {
    Restaurado { monedas: usize, freno_echado: bool },
    /// No había estado: se empieza de cero.
    SinEstado,
    /// Estado ilegible o de otra versión (fail-closed): el freno de la
    /// estrategia queda echado (no entran posiciones nuevas; las abiertas se
    /// siguen gestionando) y no se guarda nada en esta sesión, así que el
    /// archivo queda intacto hasta que el operador lo repare o lo retire.
    Bloqueado(String),
}

impl Arranque {
    /// ¿Se puede guardar el estado en esta sesión?
    pub fn permite_guardar(&self) -> bool {
        !matches!(self, Arranque::Bloqueado(_))
    }
}

impl GodEngineCore {
    /// Aplica lo leído al arrancar (CL-108f): restaura un estado legible y,
    /// ante uno ilegible o de otra versión, echa el freno de la estrategia.
    /// No escribe nada en disco.
    pub fn arrancar_con_estado(&mut self, carga: Carga) -> Arranque {
        match carga {
            Carga::Ausente => Arranque::SinEstado,
            Carga::Leido(e) => {
                let monedas = self.restaurar_estado_aprendido(&e);
                Arranque::Restaurado {
                    monedas,
                    freno_echado: self.risk_engine.freno_estrategia.rechazada(),
                }
            }
            Carga::Ilegible(m) => {
                self.risk_engine.freno_estrategia.rechazada = true;
                Arranque::Bloqueado(m)
            }
            Carga::OtraVersion(v) => {
                self.risk_engine.freno_estrategia.rechazada = true;
                Arranque::Bloqueado(format!(
                    "versión {v} del formato; este binario lee la {VERSION_ESTADO}"
                ))
            }
        }
    }
}
