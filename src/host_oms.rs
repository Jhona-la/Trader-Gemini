//! OMS authorization used by the host, including its executor replacement and entry path.
//!
//! A successful regular-open-orders query is not proof that orphan orders are resolved.
//! Session replacement and snapshot application share a short synchronous lock; no network
//! future holds that lock. Admission is checked immediately before each entry transport stage.
//! Revocation cannot retract a request already admitted to the transport or an exchange fill.
use arc_swap::ArcSwap;
use execution_engine::{
    entry_dispatch::{self, EntryRequest, EntryTransport},
    executor::{ExecutionProvider, OrderExecutor},
    order_registry::TrackedOrder,
    order_types::OrderAck,
    reconciliation::{self, PositionRiskEntry},
};
use god_engine_core::close_accounting::LocalCloseAccounting;
use god_engine_core::entry_reservation::{EntryReservation, ReservationError};
use quantum_arena::{FeeSchedule, GlobalArena, SymbolFees};
use std::collections::{HashMap, HashSet};
use std::sync::{
    Arc, Mutex,
    atomic::{AtomicBool, Ordering},
};
use tokio::sync::Notify;

/// LG-32: consultas REST por refresco para intenciones ausentes de openOrders (peso 1
/// cada una). Una por ranura de entrada del arena; el resto espera al refresco siguiente.
pub const MAX_CONSULTAS_INTENCIONES_AUSENTES: usize =
    quantum_arena::state::MAX_COINS * quantum_arena::position::MAX_SPECTRAL_SLOTS;

/// Motivo con el que la sesión rehúsa sustituir su ejecutor: un transporte de entrada o una
/// ranura local sin confirmar por el exchange siguen siendo de la sesión vigente.
pub const OMS_EXECUTOR_BUSY: &str =
    "OMS_EXECUTOR_BUSY: entry transport or settlement still owns this session";

pub struct OmsSession {
    executors: Arc<ArcSwap<OrderExecutor>>,
    ready: Arc<AtomicBool>,
    state: Mutex<SessionState>,
    quiescent: Notify,
    refresh_serial: tokio::sync::Mutex<()>,
    /// CAPITAL-UNICO V4H (INV-G): lo que el host hace cuando una pasada detecta un cierre que el
    /// exchange aplico sin evento local (pide evidencia perdida al puente: veta hasta una REST
    /// pedida despues). Se llama sin candados del OMS.
    al_cierre_externo: std::sync::OnceLock<Box<dyn Fn() + Send + Sync>>,
    /// DEMO-B: la guarda de perdida diaria del perfil `demo-testnet` (una vez). Veta aperturas en
    /// la entrada del OMS y en cada cruce de la frontera del transporte; nunca cierres.
    guarda_diaria: std::sync::OnceLock<Arc<GuardaDiariaHost>>,
}

/// DEMO-B (especificacion Demo §B.3): la guarda de perdida diaria de una cuenta, compartida por
/// el lazo del inmune (observa equity y pico), el lazo del income (flujos externos), el nucleo
/// (evidencia del roster) y el OMS (decide cada apertura con la equity de ese instante). El
/// estado se persiste de forma atomica en cada transicion (dia abierto o latch).
pub struct GuardaDiariaHost {
    guarda: Mutex<risk_engine::perdida_diaria::GuardaDiaria>,
    ruta: std::path::PathBuf,
    reloj_ms: fn() -> u64,
    ultimo_aviso: Mutex<Option<String>>,
}

/// Reloj de pared UTC en ms (el dia UTC de la guarda).
pub fn reloj_utc_ms() -> u64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_millis() as u64)
        .unwrap_or(0)
}

impl GuardaDiariaHost {
    /// Recupera el estado de hoy de `cuenta` desde `ruta` (reinicio: misma base y latch).
    pub fn recuperar(cuenta: u64, ruta: std::path::PathBuf, reloj_ms: fn() -> u64) -> Self {
        let guarda = risk_engine::perdida_diaria::GuardaDiaria::recuperar(cuenta, &ruta, reloj_ms());
        Self { guarda: Mutex::new(guarda), ruta, reloj_ms, ultimo_aviso: Mutex::new(None) }
    }

    fn persistir(&self, guarda: &risk_engine::perdida_diaria::GuardaDiaria) -> Result<(), String> {
        match guarda.estado.as_ref() {
            Some(estado) => risk_engine::perdida_diaria::guardar_estado(&self.ruta, estado)
                .map_err(|e| format!("estado diario no persistido: {e}")),
            None => Ok(()),
        }
    }

    /// Una observacion del lazo del inmune. Persiste si abrio el dia o puso el latch.
    pub fn observar(&self, equity: f64, pico: f64, global_max_drawdown: f64) -> (risk_engine::perdida_diaria::Transicion, Result<(), String>) {
        let mut g = self.guarda.lock().unwrap_or_else(|p| p.into_inner());
        let t = g.observar((self.reloj_ms)(), equity, pico, global_max_drawdown);
        let persistido = match t {
            risk_engine::perdida_diaria::Transicion::DiaAbierto(_)
            | risk_engine::perdida_diaria::Transicion::Bloqueado => self.persistir(&g),
            _ => Ok(()),
        };
        (t, persistido)
    }

    /// `observar` y la linea de telemetria si el estado cambio (una por cambio, no por ciclo).
    pub fn observar_con_aviso(&self, equity: f64, pico: f64, global_max_drawdown: f64) -> Option<String> {
        use risk_engine::perdida_diaria::Transicion;
        let (t, persistido) = self.observar(equity, pico, global_max_drawdown);
        let linea = match &t {
            Transicion::DiaAbierto(p) => Some(format!(
                "DIA_ABIERTO q_D={:.6} h={:.6} f_sizing={:.6} d_duro={:.6} d_unidad={:.10} d_restante={:.10} d_D={:.10} B_D={:.8} USDT",
                p.q_d, p.h, p.f_sizing, p.d_duro, p.d_unidad, p.d_restante, p.d_d, p.b_d
            )),
            Transicion::Bloqueado => Some(format!("{} hasta el siguiente dia UTC acreditado", risk_engine::perdida_diaria::CODIGO_DAY_LOSS_LOCKED)),
            Transicion::SinAncla(c) => Some(format!("{} causa={}: aperturas retenidas", risk_engine::perdida_diaria::CODIGO_DAY_EQUITY_ND, c.nombre())),
            Transicion::RelojRegresivo => Some("RELOJ_REGRESIVO: observacion ignorada, estado conservado".to_string()),
            Transicion::Nada => None,
        };
        let linea = match (linea, persistido) {
            (Some(l), Err(e)) => Some(format!("{l} ({e})")),
            (l, _) => l,
        };
        let mut ultimo = self.ultimo_aviso.lock().unwrap_or_else(|p| p.into_inner());
        match linea {
            Some(l) if ultimo.as_deref() != Some(l.as_str()) => {
                *ultimo = Some(l.clone());
                Some(l)
            }
            _ => None,
        }
    }

    /// Flujo externo de la cuenta (`/fapi/v1/income`), con su instante del venue.
    pub fn registrar_flujo(&self, tiempo_ms: u64, id: u64, activo: &str, importe: f64) -> Result<(), String> {
        let mut g = self.guarda.lock().unwrap_or_else(|p| p.into_inner());
        let r = g.registrar_flujo(tiempo_ms, id, activo, importe);
        let p = self.persistir(&g);
        r.map_err(|c| c.nombre().to_string()).and(p)
    }

    /// Evidencia del roster (win rate, n) que el nucleo publica; se congela al abrir el dia.
    pub fn publicar_evidencia(&self, evidencia: Vec<(f64, f64)>) {
        self.guarda.lock().unwrap_or_else(|p| p.into_inner()).publicar_evidencia(evidencia);
    }

    /// La decision para una apertura con la equity del freno de este instante. Persiste el latch.
    pub fn permite_apertura(&self, arena: &GlobalArena) -> Result<(), String> {
        let equity = arena.capital_exchange.load(Ordering::Acquire);
        let mut g = self.guarda.lock().unwrap_or_else(|p| p.into_inner());
        let antes = g.estado.as_ref().is_some_and(|e| e.bloqueado);
        let decision = g.permite_apertura((self.reloj_ms)(), equity);
        if !antes && g.estado.as_ref().is_some_and(|e| e.bloqueado) {
            // El latch vale aunque la escritura falle (en memoria); el error se informa.
            let _ = self.persistir(&g);
        }
        match decision {
            risk_engine::perdida_diaria::DecisionDiaria::Permitida => Ok(()),
            risk_engine::perdida_diaria::DecisionDiaria::DayLossLocked => {
                Err(risk_engine::perdida_diaria::CODIGO_DAY_LOSS_LOCKED.to_string())
            }
            risk_engine::perdida_diaria::DecisionDiaria::DayEquityNd(c) => Err(format!(
                "{} causa={}",
                risk_engine::perdida_diaria::CODIGO_DAY_EQUITY_ND,
                c.nombre()
            )),
        }
    }

    /// Copia del estado del dia (recibo y pruebas).
    pub fn estado(&self) -> Option<risk_engine::perdida_diaria::EstadoDiario> {
        self.guarda.lock().unwrap_or_else(|p| p.into_inner()).estado.clone()
    }
}

struct SessionState {
    generation: u64,
    approved_universe: Option<Arc<Vec<String>>>,
    in_flight: usize,
}

#[derive(Clone)]
pub struct SessionToken {
    generation: u64,
    executor: Arc<OrderExecutor>,
}

impl SessionToken {
    pub fn generation(&self) -> u64 {
        self.generation
    }

    pub fn executor(&self) -> &Arc<OrderExecutor> {
        &self.executor
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SnapshotStatus {
    Ready,
    Blocked,
    Stale,
}

#[derive(Debug)]
pub struct SnapshotOutcome {
    pub status: SnapshotStatus,
    pub adopted: usize,
    pub adjustments: usize,
    pub unresolved: usize,
    pub orphan_orders: usize,
    pub unresolved_orders: usize,
    pub pending_regular_orders: usize,
    pub pruned: usize,
    pub error: Option<String>,
    /// CAPITAL-UNICO V4H (INV-G): la pasada detecto un cierre que el exchange aplico sin evento
    /// local; el libro quedo vetado hasta un saldo pedido despues.
    pub cierre_externo: bool,
}

impl SnapshotOutcome {
    fn empty(status: SnapshotStatus) -> Self {
        Self {
            status,
            adopted: 0,
            adjustments: 0,
            unresolved: 0,
            orphan_orders: 0,
            unresolved_orders: 0,
            pending_regular_orders: 0,
            pruned: 0,
            error: None,
            cierre_externo: false,
        }
    }
}

/// I-12 (Línea I, N48): estado del reintento de la transición del ejecutor del host.
/// A lo sumo un refresco de reintento en curso, y un solo aviso por aplazamiento.
#[derive(Debug, Default)]
pub struct ReintentoTransicion {
    refresco_en_curso: Arc<AtomicBool>,
    aplazamiento_avisado: AtomicBool,
}

/// Testigo de un refresco de reintento en curso. La tarea que corre el refresco lo retiene;
/// al soltarse (al terminar, o si la tarea se cancela o falla) habilita el siguiente intento.
#[derive(Debug)]
#[must_use = "soltar el testigo habilita el siguiente intento de transición"]
pub struct RefrescoDeReintento {
    refresco_en_curso: Arc<AtomicBool>,
}

impl Drop for RefrescoDeReintento {
    fn drop(&mut self) {
        self.refresco_en_curso.store(false, Ordering::Release);
    }
}

impl ReintentoTransicion {
    pub fn new() -> Self {
        Self::default()
    }

    /// ¿Sigue en curso el refresco de reintento entregado tras un rechazo?
    pub fn refresco_en_curso(&self) -> bool {
        self.refresco_en_curso.load(Ordering::Acquire)
    }

    fn tomar_refresco(&self) -> Option<RefrescoDeReintento> {
        (!self.refresco_en_curso.swap(true, Ordering::AcqRel)).then(|| RefrescoDeReintento {
            refresco_en_curso: Arc::clone(&self.refresco_en_curso),
        })
    }

    fn primer_aviso_de_aplazamiento(&self) -> bool {
        !self.aplazamiento_avisado.swap(true, Ordering::AcqRel)
    }

    fn rearmar_aviso(&self) {
        self.aplazamiento_avisado.store(false, Ordering::Release);
    }
}

/// Lo que hizo un paso de la transición del ejecutor del host.
pub enum ResultadoTransicion {
    /// El ejecutor nuevo quedó publicado con esta generación de la sesión.
    Publicada {
        generacion: u64,
        ejecutor: Arc<OrderExecutor>,
    },
    /// Un refresco de reintento sigue en curso: no se construyó ejecutor.
    Esperando,
    /// La sesión sigue ocupada (transporte de entrada o ranura sin confirmar): no se construyó
    /// ejecutor ni se revocó la sesión. `primer_aviso` es verdadero una vez por aplazamiento.
    Ocupada { primer_aviso: bool },
    /// La sustitución rehusó y ya revocó la sesión. Con `Some`, el host corre el refresco
    /// reteniendo el testigo hasta que termine.
    Rechazada {
        motivo: String,
        refresco: Option<RefrescoDeReintento>,
    },
}

impl std::fmt::Debug for ResultadoTransicion {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Publicada { generacion, .. } => f
                .debug_struct("Publicada")
                .field("generacion", generacion)
                .finish_non_exhaustive(),
            Self::Esperando => f.write_str("Esperando"),
            Self::Ocupada { primer_aviso } => f
                .debug_struct("Ocupada")
                .field("primer_aviso", primer_aviso)
                .finish(),
            Self::Rechazada { motivo, refresco } => f
                .debug_struct("Rechazada")
                .field("motivo", motivo)
                .field("refresco", &refresco.is_some())
                .finish(),
        }
    }
}

/// Un paso de la transición: su resultado y, si la reserva del tick no se pudo soltar, el
/// motivo (la protección ya quedó marcada sucia para el vigilante).
#[derive(Debug)]
pub struct PasoDeTransicion {
    pub resultado: ResultadoTransicion,
    pub reserva_no_cancelable: Option<ReservationError>,
}

impl OmsSession {
    pub fn new(executors: Arc<ArcSwap<OrderExecutor>>, ready: Arc<AtomicBool>) -> Self {
        // Demo/Testnet is a live exchange session. Only explicit local paper needs no REST proof.
        ready.store(executors.load().is_paper_trading(), Ordering::Release);
        Self {
            executors,
            ready,
            state: Mutex::new(SessionState {
                generation: 0,
                approved_universe: None,
                in_flight: 0,
            }),
            quiescent: Notify::new(),
            refresh_serial: tokio::sync::Mutex::new(()),
            al_cierre_externo: std::sync::OnceLock::new(),
            guarda_diaria: std::sync::OnceLock::new(),
        }
    }

    /// CAPITAL-UNICO V4H (INV-G): vincula la accion del host ante un cierre externo (una vez).
    pub fn vincular_cierre_externo(&self, f: impl Fn() + Send + Sync + 'static) {
        let _ = self.al_cierre_externo.set(Box::new(f));
    }

    /// DEMO-B: vincula la guarda de perdida diaria (una vez; `false` si ya habia una).
    pub fn vincular_guarda_diaria(&self, guarda: Arc<GuardaDiariaHost>) -> bool {
        self.guarda_diaria.set(guarda).is_ok()
    }

    /// DEMO-B: el veto de la guarda diaria para una apertura (sin guarda: no veta).
    fn veto_diario(&self, arena: &GlobalArena) -> Result<(), String> {
        match self.guarda_diaria.get() {
            Some(g) => g.permite_apertura(arena),
            None => Ok(()),
        }
    }

    pub fn snapshot_token(&self) -> SessionToken {
        let state = self.state.lock().unwrap_or_else(|e| e.into_inner());
        SessionToken {
            generation: state.generation,
            executor: self.executors.load_full(),
        }
    }

    fn token_is_current(&self, generation: u64, token: &SessionToken) -> bool {
        generation == token.generation && Arc::ptr_eq(&self.executors.load_full(), &token.executor)
    }

    /// Called immediately after the host creates its arena, before any core can use it.
    pub fn initialize_arena(&self, arena: &GlobalArena) {
        let mut state = self.state.lock().unwrap_or_else(|e| e.into_inner());
        state.approved_universe = None;
        let paper = self.executors.load().is_paper_trading();
        self.reset_fee_requirement(arena, paper);
        self.ready.store(paper, Ordering::Release);
    }

    fn reset_fee_requirement(&self, arena: &GlobalArena, paper: bool) {
        arena
            .config
            .entry_publication_paused
            .store(!paper, Ordering::Release);
        arena
            .config
            .observed_fees_required
            .store(!paper, Ordering::Release);
        arena.config.live_fee_schedule.store(None);
    }

    /// Every host executor replacement goes through this operation before publishing the executor.
    pub fn replace_executor(
        &self,
        executor: Arc<OrderExecutor>,
        arena: &GlobalArena,
    ) -> Result<u64, String> {
        let mut state = self.state.lock().unwrap_or_else(|e| e.into_inner());
        // Also serialize with a core proposal not yet handed to its queued task.
        let _publication = arena
            .config
            .entry_publication_gate
            .lock()
            .unwrap_or_else(|e| e.into_inner());
        self.ready.store(false, Ordering::Release);
        state.approved_universe = None;
        self.reset_fee_requirement(arena, self.executors.load().is_paper_trading());
        arena
            .config
            .entry_publication_paused
            .store(true, Ordering::Release);
        if Self::ocupada_locked(&state, arena) {
            return Err(OMS_EXECUTOR_BUSY.into());
        }
        let next = state
            .generation
            .checked_add(1)
            .ok_or_else(|| "OMS_GENERATION_EXHAUSTED".to_string())?;
        state.generation = next;
        let paper = executor.is_paper_trading();
        self.reset_fee_requirement(arena, paper);
        self.executors.store(executor);
        self.ready.store(paper, Ordering::Release);
        self.quiescent.notify_waiters();
        Ok(next)
    }

    /// El predicado con el que `replace_executor` rehúsa (`OMS_EXECUTOR_BUSY`): una entrada en
    /// transporte o una ranura abierta que el exchange todavía no confirmó.
    fn ocupada_locked(state: &SessionState, arena: &GlobalArena) -> bool {
        state.in_flight > 0
            || arena.coins.iter().any(|coin| {
                coin.positions
                    .slots()
                    .iter()
                    .any(|slot| slot.is_open() && !slot.exchange_confirmed.load(Ordering::Acquire))
            })
    }

    /// I-12: el mismo predicado que `replace_executor`, sin revocar nada. Toma los mismos
    /// candados (la sesión y la puerta de publicación) para leer lo mismo que leería la
    /// sustitución.
    pub fn ocupada(&self, arena: &GlobalArena) -> bool {
        let state = self.state.lock().unwrap_or_else(|e| e.into_inner());
        let _publication = arena
            .config
            .entry_publication_gate
            .lock()
            .unwrap_or_else(|e| e.into_inner());
        Self::ocupada_locked(&state, arena)
    }

    /// I-12 (Línea I, fila N48 del censo C-F7b): un paso de la transición del ejecutor del
    /// host (fin del calentamiento).
    ///
    /// 1. La propuesta de entrada de este mismo tick se suelta antes que nada: su reserva es
    ///    una ranura abierta sin confirmar, y con ella `replace_executor` rehúsa siempre. Antes
    ///    la sustitución fallaba en cada tick en que el núcleo proponía una entrada, así que la
    ///    transición se aplazaba mientras durara la señal. El tick de la transición no abre
    ///    exposición (tampoco la abría antes: la reserva se cancelaba tras el rechazo).
    /// 2. Con un refresco de reintento en curso no se construye nada (`Esperando`).
    /// 3. Con la sesión ocupada por otra causa no se construye ejecutor ni se revoca nada
    ///    (`Ocupada`): se espera a que termine el transporte o a la reconciliación periódica.
    /// 4. Sólo si la sustitución, con la sesión libre, aun así rehúsa (`Rechazada`) se entrega
    ///    a lo sumo un testigo de refresco; el siguiente intento espera a que se suelte.
    ///
    /// Antes, cada mensaje WS posterior a un rechazo construía otro ejecutor (cliente reqwest
    /// nuevo), revocaba la sesión y encolaba otro `refresh` (instantáneas REST) en
    /// `refresh_serial`, sin cota mientras durara la causa.
    pub fn paso_transicion<F>(
        &self,
        arena: &GlobalArena,
        reserva_del_tick: Option<&EntryReservation>,
        reintento: &ReintentoTransicion,
        construir: F,
    ) -> PasoDeTransicion
    where
        F: FnOnce() -> Arc<OrderExecutor>,
    {
        let reserva_no_cancelable =
            reserva_del_tick.and_then(|reserva| reserva.cancel(arena).err());
        if reserva_no_cancelable.is_some() {
            quantum_arena::protection_health::mark_dirty();
        }
        let resultado = if reintento.refresco_en_curso() {
            ResultadoTransicion::Esperando
        } else if self.ocupada(arena) {
            ResultadoTransicion::Ocupada {
                primer_aviso: reintento.primer_aviso_de_aplazamiento(),
            }
        } else {
            let ejecutor = construir();
            match self.replace_executor(Arc::clone(&ejecutor), arena) {
                Ok(generacion) => {
                    reintento.rearmar_aviso();
                    ResultadoTransicion::Publicada {
                        generacion,
                        ejecutor,
                    }
                }
                Err(motivo) => ResultadoTransicion::Rechazada {
                    motivo,
                    refresco: reintento.tomar_refresco(),
                },
            }
        };
        PasoDeTransicion {
            resultado,
            reserva_no_cancelable,
        }
    }

    pub fn check_entry(&self, token: &SessionToken, arena: &GlobalArena) -> Result<(), String> {
        let mut state = self.state.lock().unwrap_or_else(|e| e.into_inner());
        self.check_entry_locked(&mut state, token, arena)
    }

    fn check_entry_locked(
        &self,
        state: &mut SessionState,
        token: &SessionToken,
        arena: &GlobalArena,
    ) -> Result<(), String> {
        if !self.token_is_current(state.generation, token) {
            return Err("OMS_ENTRY_WITHHELD: stale executor session".into());
        }
        // DEMO-B: perdida diaria (DAY_LOSS_LOCKED / DAY_EQUITY_ND), antes de cualquier etapa de
        // la apertura. Los cierres no pasan por aqui.
        self.veto_diario(arena)
            .map_err(|motivo| format!("OMS_ENTRY_WITHHELD: {motivo}"))?;
        // CAPITAL-UNICO V3 (P1.2): ninguna entrada hasta tener saldo vigente de
        // la era del puente de la cuenta (y equity completa). Sólo entradas:
        // los cierres no pasan por aquí.
        if arena.capital_incompleto.load(Ordering::Acquire) {
            // V4B: solo medida (el host lo escribe cada 60 s).
            arena.vetos_capital_oms.fetch_add(1, Ordering::Relaxed);
            return Err("OMS_ENTRY_WITHHELD: account balance not current for this session".into());
        }
        let current = self.ready.load(Ordering::Acquire)
            && !arena
                .config
                .entry_publication_paused
                .load(Ordering::Acquire);
        let fee_evidence = if token.executor.is_paper_trading() {
            !arena.config.observed_fees_required.load(Ordering::Acquire)
        } else {
            let universe = quantum_arena::symbols::get_active_universe_ref();
            state
                .approved_universe
                .as_ref()
                .is_some_and(|approved| Arc::ptr_eq(approved, &universe))
                && arena
                    .config
                    .live_fee_schedule
                    .load()
                    .as_ref()
                    .is_some_and(|book| {
                        book.generation() == token.generation
                            && book.rows().len() == universe.len()
                            && !universe.is_empty()
                            && book.rows().iter().zip(universe.iter()).enumerate().all(
                                |(i, (row, symbol))| {
                                    row.symbol.eq_ignore_ascii_case(symbol)
                                        && arena.config.fee_rates_for(i)
                                            == Some((row.maker, row.taker))
                                },
                            )
                    })
        };
        if current && fee_evidence {
            Ok(())
        } else {
            if !token.executor.is_paper_trading() && !fee_evidence {
                state.approved_universe = None;
                self.ready.store(false, Ordering::Release);
                self.reset_fee_requirement(arena, false);
            }
            Err("OMS_ENTRY_WITHHELD: session is stale or unreconciled".into())
        }
    }

    /// Freeze new admission before any potentially slow private query. An old task may
    /// finish its already admitted transport, but cannot submit another entry stage.
    pub fn begin_refresh(&self, token: &SessionToken, arena: &GlobalArena) -> bool {
        let mut state = self.state.lock().unwrap_or_else(|e| e.into_inner());
        if !self.token_is_current(state.generation, token) {
            return false;
        }
        self.ready.store(false, Ordering::Release);
        state.approved_universe = None;
        self.reset_fee_requirement(arena, token.executor.is_paper_trading());
        arena
            .config
            .entry_publication_paused
            .store(true, Ordering::Release);
        true
    }

    /// The callback is synchronous and must not reenter this session. Current zero is a valid
    /// account balance; an old executor's observation never writes the new session's capital.
    pub fn publish_current_balance<F: FnOnce(f64)>(
        &self,
        token: &SessionToken,
        balance: f64,
        publish: F,
    ) -> bool {
        let state = self.state.lock().unwrap_or_else(|e| e.into_inner());
        if self.token_is_current(state.generation, token) && balance.is_finite() && balance >= 0.0 {
            publish(balance);
            true
        } else {
            false
        }
    }

    pub async fn await_quiescence(&self, token: &SessionToken) -> bool {
        loop {
            let notified = self.quiescent.notified();
            tokio::pin!(notified);
            notified.as_mut().enable();
            {
                let state = self.state.lock().unwrap_or_else(|e| e.into_inner());
                if !self.token_is_current(state.generation, token) {
                    return false;
                }
                if state.in_flight == 0 {
                    return true;
                }
            }
            notified.await;
        }
    }

    /// CAPITAL-UNICO V4J (punto 1): el validador del cruce de la frontera del transporte. Con el
    /// estado de la sesion tomado y, sin soltarlo, la puerta del libro (el orden de candados de
    /// `replace_executor` y `ocupada`): testigo de la sesion vigente, sesion lista y publicacion no
    /// pausada (un refresco o un relevo), veto del puente y, con reserva, veto del libro, revision
    /// e importe. Los dos candados se sueltan al devolver: ninguno se sostiene durante la red.
    pub fn entrega_vigente(
        &self,
        token: &SessionToken,
        arena: &GlobalArena,
        reserva: Option<quantum_arena::dimensionable::ClaveReserva>,
    ) -> Result<(), String> {
        let state = self.state.lock().unwrap_or_else(|e| e.into_inner());
        if !self.token_is_current(state.generation, token) {
            return Err("stale executor session".into());
        }
        if !self.ready.load(Ordering::Acquire)
            || arena
                .config
                .entry_publication_paused
                .load(Ordering::Acquire)
        {
            return Err("session is refreshing or unreconciled".into());
        }
        // DEMO-B: la perdida diaria se revalida en cada cruce de la frontera del transporte, con
        // la equity de ese instante (un cruce del suelo en el mismo poll retiene el POST).
        self.veto_diario(arena)?;
        let mut libro = arena
            .config
            .entry_publication_gate
            .lock()
            .unwrap_or_else(|e| e.into_inner());
        if quantum_arena::dimensionable::LibroDimensionable::veto_de_admision(arena) {
            return Err("account balance not current for this session".into());
        }
        if let Some(clave) = reserva {
            if !libro.envio_valido_en(arena, clave) {
                return Err("capital revision changed since the decision".into());
            }
        }
        // CAPITAL-UNICO V4L (L-A1): en el compromiso local (el cliente HTTP, con la solicitud ya
        // construida) la aceptacion se registra AQUI, con la sesion y la puerta aun tomadas: es el
        // mismo paso ordenado por los escritores. Desde aqui el paquete es irrevocable y pertenece
        // a esta revision; nada lo vuelve a comprobar antes de la entrega al transporte.
        if let Some(fase) = quantum_arena::dimensionable::fase_en_compromiso() {
            libro.comprometer(state.generation, reserva, fase);
        }
        drop(libro);
        drop(state);
        Ok(())
    }

    fn admit_transport<'a>(
        &'a self,
        token: &SessionToken,
        arena: &GlobalArena,
    ) -> Result<TransportPermit<'a>, String> {
        let mut state = self.state.lock().unwrap_or_else(|e| e.into_inner());
        self.check_entry_locked(&mut state, token, arena)?;
        state.in_flight = state
            .in_flight
            .checked_add(1)
            .ok_or("OMS_IN_FLIGHT_OVERFLOW")?;
        Ok(TransportPermit { session: self })
    }

    /// Held by the complete host task, including post-ack fill attribution, protection and
    /// rollback. Executor replacement cannot move that settlement onto a different account.
    pub fn begin_entry_task(
        self: &Arc<Self>,
        token: &SessionToken,
        arena: &GlobalArena,
    ) -> Result<OwnedEntryPermit, String> {
        let mut state = self.state.lock().unwrap_or_else(|e| e.into_inner());
        self.check_entry_locked(&mut state, token, arena)?;
        state.in_flight = state
            .in_flight
            .checked_add(1)
            .ok_or("OMS_IN_FLIGHT_OVERFLOW")?;
        Ok(OwnedEntryPermit {
            session: Arc::clone(self),
        })
    }

    fn release_admission(&self) {
        let mut state = self.state.lock().unwrap_or_else(|e| e.into_inner());
        state.in_flight -= 1;
        if state.in_flight == 0 {
            self.quiescent.notify_waiters();
        }
    }

    /// Applies only an observation obtained with the current executor. Stale successes AND
    /// stale errors leave the new session, registry and arena untouched.
    pub fn apply_snapshot(
        &self,
        token: &SessionToken,
        positions: Result<Vec<PositionRiskEntry>, String>,
        orders: Result<Vec<OrderAck>, String>,
        fees: Result<FeeSchedule, String>,
        arena: &GlobalArena,
        now_ms: u64,
    ) -> SnapshotOutcome {
        let mut state = self.state.lock().unwrap_or_else(|e| e.into_inner());
        if !self.token_is_current(state.generation, token) {
            return SnapshotOutcome::empty(SnapshotStatus::Stale);
        }
        // Lock order is always session -> publication. Core uses only publication, and
        // rechecks the pause/book under it before publishing a position or debiting capital.
        let mut _publication = arena
            .config
            .entry_publication_gate
            .lock()
            .unwrap_or_else(|e| e.into_inner());
        state.approved_universe = None;
        self.ready.store(false, Ordering::Release);
        if state.in_flight > 0 {
            self.reset_fee_requirement(arena, token.executor.is_paper_trading());
            let mut outcome = SnapshotOutcome::empty(SnapshotStatus::Blocked);
            outcome.error = Some("OMS_TRANSPORT_STILL_IN_FLIGHT".into());
            return outcome;
        }
        if token.executor.is_paper_trading() {
            self.reset_fee_requirement(arena, true);
            self.ready.store(true, Ordering::Release);
            return SnapshotOutcome::empty(SnapshotStatus::Ready);
        }
        self.ready.store(false, Ordering::Release);
        self.reset_fee_requirement(arena, false);
        let (positions, orders, fees) = match (positions, orders, fees) {
            (Ok(positions), Ok(orders), Ok(fees)) => (positions, orders, fees),
            (Err(error), _, _) | (_, Err(error), _) | (_, _, Err(error)) => {
                let mut outcome = SnapshotOutcome::empty(SnapshotStatus::Blocked);
                outcome.error = Some(error);
                return outcome;
            }
        };
        let universe = quantum_arena::symbols::get_active_universe_ref();
        let expected: HashSet<_> = universe.iter().map(|s| s.to_ascii_uppercase()).collect();
        if fees.generation() != token.generation
            || expected.is_empty()
            || expected.len() != universe.len()
            || fees.rows().len() != universe.len()
            || fees
                .rows()
                .iter()
                .zip(universe.iter())
                .any(|(row, symbol)| !row.symbol.eq_ignore_ascii_case(symbol))
        {
            let mut outcome = SnapshotOutcome::empty(SnapshotStatus::Blocked);
            outcome.error = Some("OMS_FEES_INCOMPLETE_OR_WRONG_SESSION".into());
            return outcome;
        }
        // Publish this session's complete symbol schedule before authorization. Global legacy
        // maker/taker scalars are not evidence that every symbol has those account rates.
        arena.config.live_fee_schedule.store(Some(Arc::new(fees)));
        let registry = token.executor.registry();
        let report = reconciliation::reconcile_with_orders(&positions, &orders, &registry);
        let remote_ids: HashSet<_> = orders.iter().map(|o| o.client_order_id.as_str()).collect();
        // Known resting orders may have no position yet. An active local intent absent
        // from openOrders instead has unknown terminal status: retain it and block entries.
        let unresolved_orders = registry
            .active_orders()
            .iter()
            .filter(|o| !remote_ids.contains(o.client_order_id.as_str()))
            .count();
        // This endpoint lacks a proven pending-entry margin reservation contract. Do not
        // close a flat remote position's local reservation while a resting order may fill.
        // Conditional algo protection is a separate endpoint; no regular order is canceled here.
        if !orders.is_empty() || unresolved_orders > 0 {
            let mut outcome = SnapshotOutcome::empty(SnapshotStatus::Blocked);
            outcome.orphan_orders = report.orphan_remote_orders.len();
            outcome.unresolved_orders = unresolved_orders;
            outcome.pending_regular_orders = orders.len();
            return outcome;
        }
        // I-15: una reserva que espera su llenado bloquea ANTES de reconciliar.
        // Tras una IOC llenada, positionRisk puede tardar en mostrar la
        // posición: con una fila plana, reconcile_arena_checked cerraría la
        // reserva como fantasma, liberaría su margen y el núcleo podría abrir
        // otra entrada encima de la posición real. GLM-210 movió este gate
        // detrás de la reconciliación para curar el bloqueo de la corrida A3,
        // pero aquella ranura no esperaba ningún llenado: el aplanado del
        // sistema inmune le había consumido la confirmación (CL-44b) y ya la
        // había cerrado en el exchange. Esas ranuras (`confirmacion_consumida`)
        // sí pasan a la reconciliación, que las cierra con la fila plana.
        if arena
            .coins
            .iter()
            .any(|coin| coin.positions.slots().iter().any(|p| p.espera_llenado()))
        {
            let mut outcome = SnapshotOutcome::empty(SnapshotStatus::Blocked);
            outcome.error = Some("OMS_UNCONFIRMED_LOCAL_RESERVATION".into());
            return outcome;
        }
        let adopted = report.apply_to_registry_with_rates(&registry, |symbol| {
            quantum_arena::symbols::get_coin_id(symbol)
                .and_then(|ci| arena.config.fee_rates_for(ci))
                .map(|(_, taker)| taker)
        });
        let arena_report = reconciliation::reconcile_arena_checked(&positions, arena, now_ms);
        // CAPITAL-UNICO V4H (INV-G): un cierre que el exchange aplico sin evento local (la
        // reconciliacion cierra una ranura ya plana en el exchange) veta la admision hasta el
        // primer saldo posterior; una moneda plana retira el estimado de su cierre local (ya no
        // llegara fill que lo reemplace) y veta igual. Bajo la misma puerta.
        let mut cierre_externo = arena_report.cierres_externos > 0;
        if cierre_externo {
            _publication.cierre_externo(arena);
        }
        for &coin_id in &arena_report.monedas_planas {
            cierre_externo |= _publication.moneda_plana(arena, coin_id);
        }
        // GLM-210 (A3, 2026-11-06): tras reconciliar, una ranura que SIGUE
        // abierta y sin confirmar bloquea la admisión. Con I-15 sólo puede
        // ser una cuya confirmación consumió el aplanado total y que la fila
        // remota no permitió cerrar (sin fila, o el exchange aún no plano).
        // Su cierre por fila plana es el que curó el bloqueo de la corrida A3
        // (574 snapshots OMS_UNCONFIRMED_LOCAL_RESERVATION seguidos).
        let reserva_no_confirmada = arena.coins.iter().any(|coin| {
            coin.positions
                .slots()
                .iter()
                .any(|p| p.is_open() && !p.exchange_confirmed.load(Ordering::Acquire))
        });
        // Do not manufacture CANCELED acknowledgements merely because no position exists:
        // a regular resting order can be live before it creates a position.
        let pruned = registry.prune_terminated(now_ms.saturating_sub(600_000));
        let ready = arena_report.unresolved.is_empty()
            && report.orphan_remote_orders.is_empty()
            && unresolved_orders == 0
            && !reserva_no_confirmada;
        if ready {
            state.approved_universe = Some(universe);
        }
        arena
            .config
            .entry_publication_paused
            .store(!ready, Ordering::Release);
        self.ready.store(ready, Ordering::Release);
        SnapshotOutcome {
            status: if ready {
                SnapshotStatus::Ready
            } else {
                SnapshotStatus::Blocked
            },
            adopted,
            adjustments: arena_report.adjustments,
            unresolved: arena_report.unresolved.len(),
            orphan_orders: report.orphan_remote_orders.len(),
            unresolved_orders,
            pending_regular_orders: orders.len(),
            pruned,
            error: if reserva_no_confirmada {
                Some("OMS_UNCONFIRMED_LOCAL_RESERVATION".into())
            } else {
                None
            },
            cierre_externo,
        }
    }

    pub async fn refresh(&self, arena: &GlobalArena) -> SnapshotOutcome {
        // Boot/periodic/transition refreshes must not overlap and publish a ready flag
        // while another pass still holds an older exposure snapshot.
        let _refresh = self.refresh_serial.lock().await;
        let token = self.snapshot_token();
        if !self.begin_refresh(&token, arena) || !self.await_quiescence(&token).await {
            return SnapshotOutcome::empty(SnapshotStatus::Stale);
        }
        if token.executor.is_paper_trading() {
            return self.apply_snapshot(
                &token,
                Ok(Vec::new()),
                Ok(Vec::new()),
                Err("local paper scenario".into()),
                arena,
                now_ms(),
            );
        }
        // Position-side assumptions are authorization evidence too. An error leaves
        // the account mode unknown and cannot be repaired by successful fee/position queries.
        if let Err(error) = token.executor.ensure_hedge_mode().await {
            return self.apply_snapshot(
                &token,
                Ok(Vec::new()),
                Ok(Vec::new()),
                Err(format!("OMS_MODE_UNCONFIRMED: {error}")),
                arena,
                now_ms(),
            );
        }
        // Obtain the potentially long per-symbol schedule before the final exposure snapshot.
        let universe = quantum_arena::symbols::get_active_universe_ref();
        let mut rows = Vec::with_capacity(universe.len());
        for symbol in universe.iter() {
            match token.executor.fetch_commission_rate(symbol).await {
                Ok((maker, taker)) => rows.push(SymbolFees {
                    symbol: symbol.to_string(),
                    maker,
                    taker,
                }),
                Err(error) => {
                    return self.apply_snapshot(
                        &token,
                        Ok(Vec::new()),
                        Ok(Vec::new()),
                        Err(format!("OMS_FEES {symbol}: {error}")),
                        arena,
                        now_ms(),
                    );
                }
            }
        }
        let fees = FeeSchedule::new(token.generation, rows).map_err(|e| format!("OMS_FEES: {e:?}"));
        if fees.is_err() {
            return self.apply_snapshot(
                &token,
                Ok(Vec::new()),
                Ok(Vec::new()),
                fees,
                arena,
                now_ms(),
            );
        }
        let positions = token.executor.fetch_position_risk().await;
        if positions.is_err() {
            return self.apply_snapshot(&token, positions, Ok(Vec::new()), fees, arena, now_ms());
        }
        let orders = token.executor.fetch_all_open_orders().await;
        // LG-32: una intención local activa ausente de openOrders se consulta por REST
        // antes de decidir. Sin esto, una orden que el exchange nunca aceptó dejaba
        // `unresolved_orders > 0` para siempre y el OMS no volvía a autorizar entradas.
        if let Ok(abiertas) = &orders {
            token
                .executor
                .resolver_intenciones_ausentes(abiertas, MAX_CONSULTAS_INTENCIONES_AUSENTES)
                .await;
        }
        let outcome = self.apply_snapshot(&token, positions, orders, fees, arena, now_ms());
        // V4H (INV-G): fuera de los candados del OMS y de la puerta (orden puente, sesion, puerta).
        if outcome.cierre_externo {
            if let Some(f) = self.al_cierre_externo.get() {
                f();
            }
        }
        outcome
    }

    pub async fn dispatch_entry<T: EntryTransport>(
        &self,
        token: &SessionToken,
        transport: &T,
        request: &EntryRequest<'_>,
        arena: &GlobalArena,
    ) -> Result<(), String> {
        self.dispatch_entry_reservada(token, transport, request, arena, None).await
    }

    /// CAPITAL-UNICO V4H (INV-F): como [`dispatch_entry`](Self::dispatch_entry), con la reserva
    /// del libro del dimensionable que respalda esta entrada. Justo antes de cada llamada al
    /// exchange el guardia revalida contra el libro: la reserva sigue viva, el libro no veta y
    /// el disponible para ella cubre aun el dimensionable con el que Risk la decidio.
    pub async fn dispatch_entry_reservada<T: EntryTransport>(
        &self,
        token: &SessionToken,
        transport: &T,
        request: &EntryRequest<'_>,
        arena: &GlobalArena,
        reserva: Option<quantum_arena::dimensionable::ClaveReserva>,
    ) -> Result<(), String> {
        // V4I (INV-E): la orden que respalda la reserva, para atribuirle cada fill del stream.
        // V4J (punto 3): con su cantidad (la misma unidad que la `l` de cada fill).
        if let Some(clave) = reserva {
            arena
                .config
                .entry_publication_gate
                .lock()
                .unwrap_or_else(|e| e.into_inner())
                .vincular_orden_con_cantidad(clave, request.client_order_id, request.quantity);
        }
        let guarded = GuardedEntry {
            session: self,
            token,
            transport,
            arena,
            reserva,
        };
        entry_dispatch::dispatch_entry(&guarded, request).await
    }
}

type SettlementSide = (String, bool);

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
struct SettlementId {
    symbol: String,
    coin_id: usize,
    slot: usize,
    generation: u64,
}

static SETTLEMENT_SIDES: std::sync::LazyLock<
    Mutex<HashMap<SettlementSide, Arc<tokio::sync::Mutex<()>>>>,
> = std::sync::LazyLock::new(|| Mutex::new(HashMap::new()));
static PENDING_SETTLEMENTS: std::sync::LazyLock<
    Mutex<HashMap<SettlementId, tokio::sync::OwnedMutexGuard<()>>>,
> = std::sync::LazyLock::new(|| Mutex::new(HashMap::new()));

/// A pending guard is owned by this registry, not by the async task stack.
/// Dropping the handle without release therefore cannot open the side early.
pub struct EntrySettlement {
    reservation: EntryReservation,
}

impl EntrySettlement {
    pub async fn acquire(reservation: EntryReservation, is_long: bool) -> Self {
        let side = (reservation.symbol.to_ascii_uppercase(), is_long);
        let lock = {
            let mut sides = SETTLEMENT_SIDES.lock().unwrap_or_else(|e| e.into_inner());
            Arc::clone(
                sides
                    .entry(side)
                    .or_insert_with(|| Arc::new(tokio::sync::Mutex::new(()))),
            )
        };
        let permit = lock.lock_owned().await;
        let id = SettlementId {
            symbol: reservation.symbol.clone(),
            coin_id: reservation.coin_id,
            slot: reservation.slot,
            generation: reservation.generation,
        };
        PENDING_SETTLEMENTS
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .insert(id, permit);
        Self { reservation }
    }

    pub fn release(self) {
        Self::release_for(&self.reservation);
    }

    pub fn release_for(reservation: &EntryReservation) -> bool {
        let id = SettlementId {
            symbol: reservation.symbol.clone(),
            coin_id: reservation.coin_id,
            slot: reservation.slot,
            generation: reservation.generation,
        };
        PENDING_SETTLEMENTS
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .remove(&id)
            .is_some()
    }
}

pub fn release_confirmed_side(
    arena: &GlobalArena,
    symbol: &str,
    is_long: bool,
    protection_complete: bool,
) -> usize {
    if !protection_complete {
        return 0;
    }
    let mut released = 0;
    PENDING_SETTLEMENTS
        .lock()
        .unwrap_or_else(|e| e.into_inner())
        .retain(|id, _| {
            let confirmed = id.symbol.eq_ignore_ascii_case(symbol)
                && arena.coins.get(id.coin_id).is_some_and(|coin| {
                    let position = coin.positions.get_slot(id.slot);
                    position.is_open()
                        && position.generation.load(Ordering::Acquire) == id.generation
                        && position.exchange_confirmed.load(Ordering::Acquire)
                        && position.is_long.load(Ordering::Relaxed) == is_long
                });
            if confirmed {
                released += 1;
            }
            !confirmed
        });
    released
}

pub fn cancel_entry_after_deadline(
    session: &OmsSession,
    token: &SessionToken,
    arena: &GlobalArena,
    reservation: &EntryReservation,
) -> bool {
    if !session.begin_refresh(token, arena) {
        return false;
    }
    quantum_arena::protection_health::mark_dirty();
    match reservation.cancel(arena) {
        Ok(()) => {
            EntrySettlement::release_for(reservation);
            true
        }
        Err(_) => false,
    }
}

pub fn entry_execution_for_host(
    order: Option<&TrackedOrder>,
    paper: bool,
    requested_quantity: f64,
) -> Option<f64> {
    if paper {
        Some(requested_quantity.abs())
    } else {
        execution_engine::ioc_evidence::cantidad_ejecutada_terminal(order)
    }
}

/// CAPITAL-UNICO V4L (L-A1H): lo que el host hizo con la reserva y el settlement de una entrada
/// tras el resultado del envio. Lo devuelve [`resolver_resultado_de_entrada`], la resolucion que
/// corre el host (`god_engine.rs`) y que las pruebas llaman tal cual.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum DesenlaceDeEntrada {
    /// Rechazo firme o terminal sin ejecucion: la generacion exacta se cancelo una vez (devuelve
    /// solo lo aun reservado) y su settlement quedo suelto.
    Revertida,
    /// La reserva ya no es cancelable (otra generacion, ya confirmada o ya cancelada): no se toca
    /// nada mas; la proteccion queda sucia.
    ReversionImposible,
    /// Ejecucion acreditada por evidencia terminal de ESTA orden, confirmada con su cantidad.
    /// `envio_ambiguo`: el envio fue ambiguo y la consulta lo resolvio (la proteccion la pone el
    /// vigilante); si no, el host sigue con su bracket.
    Confirmada { ejecutada: f64, envio_ambiguo: bool },
    /// La confirmacion no cuadro con la reserva (otra generacion): se reconcilia.
    ConfirmacionRechazada,
    /// Sin evidencia concluyente (o con evidencia contradictoria): reserva, cargo, settlement y
    /// proteccion se conservan hasta reconciliar. Nada se reenvia.
    Conservada,
    /// El plazo de recuperacion vencio: admision revocada y reserva local cancelada sin certificar.
    CanceladaPorDeadline,
}

/// CAPITAL-UNICO V4L (L-A1H): la reversion de una entrada desde el host. Cancela la generacion
/// EXACTA de la reserva por su mecanismo productivo (`EntryReservation::cancel`, que cancela solo
/// esa generacion, devuelve su margen y quita su reserva del libro bajo la puerta) y, solo si esa
/// cancelacion ocurrio, suelta su settlement. Repetida: la segunda no cancela ni suelta nada.
pub fn revertir_entrada(arena: &GlobalArena, reservation: &EntryReservation) -> bool {
    match reservation.cancel(arena) {
        Ok(()) => {
            EntrySettlement::release_for(reservation);
            true
        }
        Err(_) => {
            quantum_arena::protection_health::mark_dirty();
            false
        }
    }
}

/// V4L (L-A1H, borde de A2): hay evidencia positiva de ejecucion de esta orden (el stream la
/// atribuyo a la reserva, o el registro la vio ejecutar). Un rechazo o un terminal sin ejecucion
/// que lo contradiga no suelta la reserva solo porque el callback del fill llego tarde.
/// El host (`god_engine.rs`, brazos `Revertir` y rechazo firme) y [`resolver_resultado_de_entrada`]
/// lo consultan antes de revertir. En papel no aplica (no hay exchange).
pub fn entrada_con_ejecucion_acreditada(arena: &GlobalArena, executor: &OrderExecutor, reservation: &EntryReservation, client_id: &str) -> bool {
    if executor.is_paper_trading() {
        return false;
    }
    let libro = arena
        .config
        .entry_publication_gate
        .lock()
        .unwrap_or_else(|e| e.into_inner())
        .ejecucion_observada(reservation.clave());
    libro
        || executor
            .registry()
            .get(client_id)
            .is_some_and(|o| !(o.executed_qty.is_finite() && o.executed_qty <= 0.0))
}

/// CAPITAL-UNICO V4L (L-A1H): la resolucion PRODUCTIVA del resultado de una entrada del host, con
/// su reserva y su settlement reales. El host la llama con el resultado de
/// [`OmsSession::dispatch_entry_reservada`]; las pruebas, igual.
/// - Ambiguo (o maker sin verificar): proteccion sucia y consulta de ESTA identidad
///   (`resolve_terminal_entry_until`, sin reenvio); el padre no resuelve al hijo maker.
/// - Rechazo firme sin ejecucion: reversion exacta una vez ([`revertir_entrada`]).
/// - `Ok`: la cantidad del terminal de esta orden; sin terminal, la misma consulta.
/// - Con ejecucion acreditada por el stream o el registro, un rechazo o un terminal sin
///   ejecucion no revierte: se conserva hasta reconciliar.
pub async fn resolver_resultado_de_entrada(
    resultado: Result<(), String>,
    session: &OmsSession,
    token: &SessionToken,
    arena: &GlobalArena,
    executor: &OrderExecutor,
    reservation: &EntryReservation,
    client_id: &str,
    cantidad_pedida: f64,
    plazo: std::time::Duration,
) -> DesenlaceDeEntrada {
    use execution_engine::ioc_evidence::DestinoReserva;
    let (mut destino, envio_ambiguo) = match resultado {
        Err(e) if e.starts_with("AMBIGUOUS") || e.starts_with("MAKER_CHASE_UNVERIFIED") => {
            quantum_arena::protection_health::mark_dirty();
            let d = execution_engine::entry_dispatch::resolve_terminal_entry_until(
                executor,
                &reservation.symbol,
                client_id,
                &e,
                plazo,
            )
            .await;
            (d, true)
        }
        Err(_) => (DestinoReserva::Revertir, false),
        Ok(()) => {
            match entry_execution_for_host(executor.registry().get(client_id).as_ref(), executor.is_paper_trading(), cantidad_pedida) {
                Some(ejecutada) => (DestinoReserva::Confirmar { ejecutada }, false),
                None => {
                    quantum_arena::protection_health::mark_dirty();
                    let d = execution_engine::entry_dispatch::resolve_terminal_entry_until(
                        executor,
                        &reservation.symbol,
                        client_id,
                        "AMBIGUOUS: ENTRY_TERMINAL_PENDING",
                        plazo,
                    )
                    .await;
                    (d, false)
                }
            }
        }
    };
    // CAPITAL-UNICO V4M (L-A1H): ruta maker con hija (remanente `mcT_`). El padre no resuelve a la
    // hija ni al reves: cada una se resuelve por SU identidad y su evidencia terminal; solo con las
    // dos terminales se confirma lo ejecutado acreditado (padre + hija). Si alguna sigue sin
    // evidencia (ACK de la hija perdido sin terminal, activa o plazo vencido), se conserva todo.
    if !executor.is_paper_trading() {
        // Una hija retenida en la frontera nunca salio (rechazo local: sin orderId ni ejecucion):
        // el remanente quedo retenido y la resolucion previa (conservar) se mantiene.
        let hija_enviada = executor.hija_del_maker(client_id).filter(|h| {
            !executor.registry().get(h).is_some_and(|o| {
                o.status == execution_engine::order_registry::OrderStatus::Rejected && o.order_id == 0 && o.executed_qty == 0.0
            })
        });
        if let Some(hija) = hija_enviada {
            quantum_arena::protection_health::mark_dirty();
            let consultar = |id: String, motivo: &'static str| async move {
                execution_engine::entry_dispatch::resolve_terminal_entry_until(executor, &reservation.symbol, &id, motivo, plazo).await
            };
            let padre = consultar(client_id.to_string(), "AMBIGUOUS: MAKER_PARENT_TERMINAL").await;
            let hijo = consultar(hija, "AMBIGUOUS: MAKER_CHILD_TERMINAL").await;
            let ejecutada = |d: DestinoReserva| match d {
                DestinoReserva::Confirmar { ejecutada } => Some(ejecutada),
                DestinoReserva::Revertir => Some(0.0),
                _ => None,
            };
            destino = match (ejecutada(padre), ejecutada(hijo)) {
                (Some(p), Some(h)) if p + h > 0.0 => DestinoReserva::Confirmar { ejecutada: p + h },
                (Some(_), Some(_)) => DestinoReserva::Revertir,
                _ => DestinoReserva::Conservar,
            };
        }
    }
    match destino {
        DestinoReserva::Revertir => {
            if entrada_con_ejecucion_acreditada(arena, executor, reservation, client_id) {
                quantum_arena::protection_health::mark_dirty();
                return DesenlaceDeEntrada::Conservada;
            }
            if revertir_entrada(arena, reservation) {
                DesenlaceDeEntrada::Revertida
            } else {
                DesenlaceDeEntrada::ReversionImposible
            }
        }
        DestinoReserva::Confirmar { ejecutada } => {
            let marca = executor.registry().marca_terminal(client_id);
            match reservation.confirmar_llenado_con_marca(arena, Some(ejecutada), marca) {
                Ok(()) => DesenlaceDeEntrada::Confirmada { ejecutada, envio_ambiguo },
                Err(_) => {
                    quantum_arena::protection_health::mark_dirty();
                    DesenlaceDeEntrada::ConfirmacionRechazada
                }
            }
        }
        DestinoReserva::CancelarPorDeadline => {
            let _ = cancel_entry_after_deadline(session, token, arena, reservation);
            DesenlaceDeEntrada::CanceladaPorDeadline
        }
        DestinoReserva::Conservar => {
            quantum_arena::protection_health::mark_dirty();
            DesenlaceDeEntrada::Conservada
        }
    }
}

fn now_ms() -> u64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap_or_default()
        .as_millis() as u64
}

/// A local proposal receipt belongs only to the exact event and quantity that produced it.
/// It does not assert that the exchange settled this close.
pub fn matching_local_close(
    receipt: Option<LocalCloseAccounting>,
    event_time_ms: u64,
    quantity: f64,
) -> Option<LocalCloseAccounting> {
    receipt.filter(|r| {
        r.event_time_ms == event_time_ms
            && r.quantity == quantity
            && quantity.is_finite()
            && quantity > 0.0
            && r.entry_price.is_finite()
            && r.entry_price > 0.0
            && r.exit_price.is_finite()
            && r.exit_price > 0.0
            && r.gross_pnl.is_finite()
            && r.entry_fee.is_finite()
            && r.exit_fee.is_finite()
            && r.total_fees().is_finite()
    })
}

/// Binance income is cash flow: negative COMMISSION is a cost, positive is a rebate.
pub fn signed_commission_cost(income: f64) -> Option<f64> {
    income.is_finite().then_some(-income)
}

/// SIGNO-COMISION (pedido 20261117-2355, hallazgo `codex_app_demo_testnet.md` §4): agregado de
/// CUENTA de una pasada del income. Por simbolo llegan sus filas REALIZED_PNL `(t, pnl)`, el
/// funding como FLUJO de caja (+ cobrado, - pagado) y la comision como COSTO
/// (`signed_commission_cost`: + pagada, - rebate). Devuelve `(bruto, flujo_de_costos, filas,
/// ganadoras)` con `flujo_de_costos = funding - costo`, firmado como lo leen el profiler, el
/// breaker de Produccion y el certificado demo (negativo = gasto): `bruto + flujo_de_costos` es
/// el neto, el mismo que el calculo por simbolo (`bruto + funding - costo`). Antes el host sumaba
/// `costo + funding`, y una comision pagada subia el neto.
/// `filas` cuenta filas REALIZED_PNL, no ciclos: un cierre en varios llenados cuenta varias
/// (pendiente separado, documentado en la respuesta del pedido).
pub fn agregado_income_de_cuenta<'a>(
    simbolos: impl IntoIterator<Item = (&'a [(u64, f64)], f64, f64)>,
) -> (f64, f64, u64, u64) {
    let mut bruto = 0.0;
    let mut flujo_de_costos = 0.0;
    let mut filas: u64 = 0;
    let mut ganadoras: u64 = 0;
    for (registros, funding, costo_comision) in simbolos {
        flujo_de_costos += funding - costo_comision;
        for (_, pnl) in registros {
            filas += 1;
            if *pnl > 0.0 {
                ganadoras += 1;
            }
            bruto += pnl;
        }
    }
    (bruto, flujo_de_costos, filas, ganadoras)
}

/// EV neto por fila del certificado demo y del breaker de Produccion. N/D (`None`), nunca 0, sin
/// filas o con bruto, flujo o cociente no finito.
pub fn ev_neto_por_cierre(cierres: u64, bruto: f64, flujo_de_costos: f64) -> Option<f64> {
    if cierres == 0 || !bruto.is_finite() || !flujo_de_costos.is_finite() {
        return None;
    }
    let ev = (bruto + flujo_de_costos) / cierres as f64;
    ev.is_finite().then_some(ev)
}

/// GLM-219: el breaker Prod->Demo con el income de la cuenta (trasladado del bin por
/// SIGNO-COMISION para probarlo sin red). Sin filas no decide; un EV N/D llega como NaN y el
/// orquestador lo trata como no finito (dispara), como antes.
pub fn breaker_de_ingreso(
    orch: &mut god_engine_core::orchestrator::PhaseOrchestrator,
    cierres: u64,
    bruto: f64,
    flujo_de_costos: f64,
) -> bool {
    if cierres == 0 {
        return false;
    }
    orch.registrar_desempeno_prod(cierres, ev_neto_por_cierre(cierres, bruto, flujo_de_costos).unwrap_or(f64::NAN))
}

struct GuardedEntry<'a, T> {
    session: &'a OmsSession,
    token: &'a SessionToken,
    transport: &'a T,
    arena: &'a GlobalArena,
    /// V4H (INV-F): la reserva del libro que respalda esta entrada.
    reserva: Option<quantum_arena::dimensionable::ClaveReserva>,
}

impl<T> GuardedEntry<'_, T> {
    /// V4H (INV-F): la decision sigue siendo la misma revision del capital al enviar.
    fn revalidar_capital(&self) -> Result<(), String> {
        // El veto del PUENTE pudo entrar despues del permiso (el publicador veta antes de
        // esperar la puerta): el libro lo vuelve a mirar aqui, junto al suyo.
        let Some(clave) = self.reserva else { return Ok(()) };
        let valido = self
            .arena
            .config
            .entry_publication_gate
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .envio_valido_en(self.arena, clave);
        if valido {
            Ok(())
        } else {
            Err("OMS_ENTRY_WITHHELD: capital revision changed since the decision".into())
        }
    }
}

struct TransportPermit<'a> {
    session: &'a OmsSession,
}

#[must_use = "retain this permit until the whole entry task, including settlement, finishes"]
pub struct OwnedEntryPermit {
    session: Arc<OmsSession>,
}

impl Drop for OwnedEntryPermit {
    fn drop(&mut self) {
        self.session.release_admission();
    }
}

impl Drop for TransportPermit<'_> {
    fn drop(&mut self) {
        self.session.release_admission();
    }
}

impl<T: EntryTransport> EntryTransport for GuardedEntry<'_, T> {
    fn supports_native_iceberg(&self) -> bool {
        self.transport.supports_native_iceberg()
    }

    async fn configure_leverage(&self, symbol: &str, leverage: u32) -> Result<(), String> {
        let _permit = self.session.admit_transport(self.token, self.arena)?;
        // V4I (INV-F, punto 5c): una propuesta revocada no configura apalancamiento.
        self.revalidar_capital()?;
        // V4J (punto 1): el POST de /leverage tambien cruza la frontera con el validador.
        let frontera = quantum_arena::dimensionable::FronteraDeEnvio::nueva();
        let validar = || self.session.entrega_vigente(self.token, self.arena, self.reserva);
        let mut cambio = std::pin::pin!(async { self.transport.configure_leverage(symbol, leverage).await });
        std::future::poll_fn(|cx| frontera.paso(&validar, || std::future::Future::poll(cambio.as_mut(), cx))).await
    }

    async fn submit_entry(&self, request: &EntryRequest<'_>) -> Result<(), String> {
        // Leverage confirmation may have awaited while OMS was revoked or replaced.
        let _permit = self.session.admit_transport(self.token, self.arena)?;
        self.revalidar_capital()?;
        // V4I (INV-F): la frontera es la entrega al transporte, no esta linea. Hasta que el
        // ejecutor entrega su primer POST, una revision vencida al empezar un paso abandona la
        // entrega (0 llamadas). V4J (punto 1): cada cruce de la frontera EJECUTA el validador
        // (`OmsSession::entrega_vigente`: sesion, vetos, revision e importe, bajo la sesion y la
        // puerta), asi que un cambio dentro del mismo `poll`, despues de la revalidacion del
        // inicio del paso, tambien retiene el POST. Los candados se sueltan en el cruce: ninguno
        // durante la red.
        let frontera = quantum_arena::dimensionable::FronteraDeEnvio::nueva();
        let validar = || self.session.entrega_vigente(self.token, self.arena, self.reserva);
        let mut envio = std::pin::pin!(async { self.transport.submit_entry(request).await });
        std::future::poll_fn(|cx| {
            if !frontera.cruzada() {
                if let Err(e) = self.revalidar_capital() {
                    return std::task::Poll::Ready(Err(e));
                }
            }
            frontera.paso(&validar, || std::future::Future::poll(envio.as_mut(), cx))
        })
        .await
    }
}

/// GLM-216 (bloqueante #2), trasladado por V4M (F8): en demo, persiste el certificado de
/// rentabilidad que la puerta a Produccion exige. Sin cambio de logica respecto del bin.
pub fn persistir_evidencia_demo_en_modo(
    is_demo_mode: bool,
    ruta: &std::path::Path,
    cierres: u64,
    bruto: f64,
    fees: f64,
    wins: u64,
) -> Result<bool, String> {
    if !is_demo_mode {
        return Ok(false);
    }
    let Some(ev_neto_por_trade) = ev_neto_por_cierre(cierres, bruto, fees) else {
        return Ok(false);
    };
    let evidencia = god_engine_core::orchestrator::EvidenciaDemo { cierres, ev_neto_por_trade, wins };
    let texto = serde_json::to_string_pretty(&evidencia).map_err(|e| e.to_string())?;
    std::fs::write(ruta, texto).map_err(|e| e.to_string())?;
    Ok(true)
}

/// CAPITAL-UNICO V4M (F8, dictamen V4L :65-73): el certificado demo se publica con el MISMO testigo
/// de la consulta de income (la era del puente, tomada antes del ejecutor; su ambito es la cuenta y
/// la sesion de esa era). La escritura ocurre bajo el mutex del puente, el mismo del relevo: si el
/// relevo gano, no se escribe nada (`None`). Es una escritura local pequena, no red.
#[allow(clippy::too_many_arguments)]
pub fn certificado_demo_de_era(
    puente: &execution_engine::puente_cuenta::PuenteCuenta,
    era: u64,
    is_demo_mode: bool,
    ruta: &std::path::Path,
    cierres: u64,
    bruto: f64,
    fees: f64,
    wins: u64,
) -> Option<Result<bool, String>> {
    puente.con_era_vigente(era, |_| persistir_evidencia_demo_en_modo(is_demo_mode, ruta, cierres, bruto, fees, wins))
}

/// CAPITAL-UNICO V4K (F8): el registro de los flujos externos de una respuesta del historial de
/// ingresos (`/fapi/v1/income`), el MISMO que corre el lazo del host. Testigo de era: con la era de
/// la consulta vencida (una respuesta retenida que llega despues del relevo) no registra nada y
/// devuelve `None`. El ambito es la cuenta vigente del libro (bajo el mutex del puente y despues la
/// puerta); el activo, el de cada movimiento; un `tranId` repetido (replay) no cuenta otra vez.
/// Solo atribucion: no ajusta pico ni freno (no hay politica de transferencias).
pub fn registrar_flujos_externos_de_era(
    puente: &execution_engine::puente_cuenta::PuenteCuenta,
    era: u64,
    arena: &GlobalArena,
    flujos: &std::sync::Mutex<risk_engine::drawdown::FlujosExternos>,
    entries: &[execution_engine::order_types::IncomeEntry],
) -> Option<Vec<(u64, risk_engine::drawdown::RegistroDeFlujo)>> {
    puente.con_era_vigente(era, |_| {
        let cuenta = arena
            .config
            .entry_publication_gate
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .cuenta()
            .unwrap_or(0);
        let mut flujos = flujos.lock().unwrap_or_else(|p| p.into_inner());
        let mut registros = Vec::new();
        for e in entries {
            if risk_engine::drawdown::es_flujo_externo(&e.income_type) {
                // V4L (L-F8): sin identidad nativa (`tranId` 0) no se sintetiza una por tipo y hora:
                // la atribucion queda incierta, sin cifra ni deduplicacion inferidas (id 0).
                if e.tran_id == 0 {
                    registros.push((0, flujos.registrar_sin_identidad(cuenta, e.income)));
                    continue;
                }
                let id = e.tran_id;
                registros.push((id, flujos.registrar(cuenta, id, &e.asset, e.income, era)));
            }
        }
        registros
    })
}

/// CAPITAL-UNICO V4N (F2/F5): el estado del pico que el lazo del inmune lleva de un ciclo al
/// siguiente (antes, variables locales de la tarea del binario).
#[derive(Debug, Clone)]
pub struct PicoDelInmune {
    /// El pico guardado del espacio es ilegible (CL-133): no se sobrescribe.
    pub pico_ilegible: bool,
    /// La marca de la definicion se escribe tras el primer pico guardado.
    pub marca_pendiente: bool,
    pub ruta_pico: std::path::PathBuf,
    pub ruta_marca: std::path::PathBuf,
    /// Ultimo pico persistido en este espacio (0 ⇒ el primer ciclo guarda).
    pub pico_guardado: f64,
}

/// CAPITAL-UNICO V4N (F2/F5): lo que decide un paso del lazo del inmune: la magnitud del freno, su
/// pico, el ambito y la revision del plano en que se escribieron (el testigo con que Risk y el host
/// comparten la decision). `error_de_marca`: la marca de la definicion no se pudo escribir.
#[derive(Debug, Clone, PartialEq)]
pub struct PasoDelFreno {
    pub cap: f64,
    pub pico: f64,
    pub ruta_del_pico: std::path::PathBuf,
    pub ambito: u64,
    pub revision: u64,
    pub error_de_marca: Option<String>,
}

/// CAPITAL-UNICO V4N (F2/F5, dictamen V4L :83 y :118): el paso del host en el ciclo del inmune,
/// el MISMO que llama el binario (antes en linea en `god_engine.rs` y copiado por las pruebas como
/// adaptador). En UN paso bajo el mutex del puente y dentro de UNA escritura del plano: adopta el
/// espacio nuevo del pico si el relevo lo dejo en el buzon (`aplicar_arranque`, la funcion del
/// binario), publica la equity del puente como magnitud del freno (ausencia mientras el pico del
/// ambito esta pendiente), registra el pico y toma la revision que la escritura deja al cerrarse.
/// Fuera del mutex persiste el pico (y la marca tras el primero) salvo pico ilegible. No cambia
/// ningun umbral ni formula: solo traslada el paso.
pub fn paso_del_freno_del_inmune(
    puente: &execution_engine::puente_cuenta::PuenteCuenta,
    arena: &GlobalArena,
    buzon: &Mutex<Option<risk_engine::drawdown::RutasPico>>,
    flujos: &Mutex<risk_engine::drawdown::FlujosExternos>,
    estado: &mut PicoDelInmune,
    aplicar_arranque: impl FnOnce(&GlobalArena, &risk_engine::drawdown::RutasPico, bool) -> (bool, bool),
) -> PasoDelFreno {
    let mut revision = 0u64;
    let (cap, pico, ruta_del_pico, ambito) = puente.con_equity_del_freno(|equity| {
        // V4K (F2): este paso es UNA revision del plano del freno (Risk no combina la equity de
        // este paso con el pico de otro).
        let plano = risk_engine::drawdown::EscrituraDelPlano::abrir(arena);
        let espacio_nuevo = buzon.lock().unwrap_or_else(|p| p.into_inner()).take();
        if let Some(rutas) = espacio_nuevo {
            // V4J (punto 5): nada del ambito anterior mide el pico nuevo.
            arena.capital_exchange.store(f64::NAN, Ordering::Release);
            *flujos.lock().unwrap_or_else(|p| p.into_inner()) = risk_engine::drawdown::FlujosExternos::default();
            (estado.pico_ilegible, estado.marca_pendiente) = aplicar_arranque(arena, &rutas, true);
            estado.ruta_pico = rutas.pico.clone();
            estado.ruta_marca = rutas.marca.clone();
            estado.pico_guardado = 0.0;
        }
        let cap_exchange = equity.unwrap_or(f64::NAN);
        // V4K (F6): con el pico del ambito aun pendiente, la equity no es evidencia lista:
        // ausencia (no escribe pico; el freno la distingue).
        let pendiente = arena.pico_pendiente.load(Ordering::Acquire);
        arena.capital_exchange.store(
            if cap_exchange.is_finite() && !pendiente { cap_exchange } else { f64::NAN },
            Ordering::Release,
        );
        let cap = risk_engine::drawdown::capital_del_freno(arena);
        let pico = risk_engine::drawdown::registrar_pico(&arena.pico_capital, cap);
        // V4L (L-F2): el testigo de este paso es la revision que deja al cerrarse.
        revision = plano.revision_al_cerrar();
        (cap, pico, estado.ruta_pico.clone(), risk_engine::drawdown::ambito_del_plano(arena))
    });
    let mut error_de_marca = None;
    if !estado.pico_ilegible
        && pico > estado.pico_guardado
        && risk_engine::drawdown::guardar_pico(&ruta_del_pico, pico).is_ok()
    {
        estado.pico_guardado = pico;
        // CAPITAL-UNICO V2 (7a): la marca de la definicion se escribe DESPUES del primer pico
        // guardado en esta definicion; desde entonces, un pico ausente es un retiro del operador.
        if estado.marca_pendiente {
            match risk_engine::drawdown::guardar_marca_definicion(&estado.ruta_marca) {
                Ok(()) => estado.marca_pendiente = false,
                Err(e) => error_de_marca = Some(format!("{}: {}", estado.ruta_marca.display(), e)),
            }
        }
    }
    PasoDelFreno { cap, pico, ruta_del_pico, ambito, revision, error_de_marca }
}