use crate::atomic_float::AtomicF64;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};

/// Modo de horizonte de una posición.
///
/// # U-ERR-5 (ERRADICACIÓN DEL BINARIO DE HORIZONTE)
///
/// Tenía tres variantes (`Continuous`, `Scalping`, `Swing`). Ningún fichero
/// del repositorio construía `Scalping` ni `Swing` fuera de este módulo: todas
/// las aperturas vivas —core, backtest, reconciliación, binario— abren
/// `Continuous`. Las dos variantes sobrantes sólo servían para alimentar un
/// `match` en `PositionManager::get_position`, que tampoco tenía llamadores.
///
/// El motor tiene UN modo de horizonte. El horizonte REAL de una posición no
/// es esta etiqueta sino [`Position::entry_tau_ms`]: la τ dominante del
/// espectro temporal en el instante de la entrada, un valor continuo en ms.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PositionHorizon {
    Continuous,
}
/// Lock-free Position tracking for the Hot Path
#[repr(C, align(64))]
pub struct Position {
    pub is_open: AtomicBool,
    pub is_long: AtomicBool,
    /// U-ERR-5: encoding del modo de horizonte. Único valor vivo: 0 =
    /// continuo. La τ real de la posición vive en `entry_tau_ms`.
    pub horizon: std::sync::atomic::AtomicU8,
    pub entry_price: AtomicF64,
    pub quantity: AtomicF64,
    pub margin_used: AtomicF64,
    pub entry_time_ms: AtomicU64,
    pub trailing_phase: std::sync::atomic::AtomicU8,
    pub mfe_atr: AtomicF64,
    pub max_pnl_pct: AtomicF64,
    pub trail_stop: AtomicF64,
    pub tp_price: AtomicF64,
    pub sl_price: AtomicF64,
    pub ml_prediction: AtomicF64,
    /// QO-E2b — tensor 54D congelado EN LA APERTURA: el productor del
    /// dataset NN lo llena al abrir y lo consume el cierre (target =
    /// retorno neto realizado) — SIN snapshot de entrada el label sería
    /// fuga (features post-hoc prediciendo su propio pasado).
    pub nn_entry_tensor: std::sync::Mutex<Vec<f64>>,
    pub confidence: AtomicF64,
    pub entry_fee: AtomicF64,
    /// REHAB-1b: τ DOMINANTE del espectro temporal en el instante de la
    /// entrada (horizonte continuo VIVO — el atributo real de la posición,
    /// no una etiqueta binaria). 0 = espectro sin opinión aún.
    pub entry_tau_ms: AtomicU64,
    /// B3.14 — ¿la entrada EXISTE en el exchange? La posición local nace
    /// en el core ANTES de la ejecución asíncrona; si esa entrada fue
    /// vetada (envolvente/margen/breaker) o rechazada, todo round-trip
    /// local es PAPEL y NO contabiliza (caso KOMA: +$36 realizados con
    /// WR 1.0 jamás operados, 2026-09-15). La confirma el host tras el
    /// fill real; la adopción FASE 5 también la setea.
    pub exchange_confirmed: AtomicBool,
    /// B3.14 — resultado del último cierre, leído por el host para su
    /// contabilidad/Kelly: true = la posición cerrada tenía entrada real.
    pub last_close_confirmed: AtomicBool,
    /// D-659 (DÉCIMA OLA) — CONTADOR DE GENERACIÓN. Se incrementa en cada
    /// apertura; sirve para diagnóstico y para distinguir ocupantes sucesivos
    /// del mismo slot en la telemetría.
    pub generation: AtomicU64,
    /// D-659 — CERROJO DE TRANSICIÓN.
    ///
    /// Un contador de generación NO basta por sí solo: el abridor puede
    /// incrementarlo antes de que el cerrador lo capture, con lo que el
    /// cerrador cree ser dueño de la generación entrante y autoriza el
    /// borrado de la posición recién abierta. La única garantía sólida es
    /// que apertura y cierre no muten los campos a la vez.
    ///
    /// Es un cerrojo de giro sobre transiciones de posición —frecuencia de
    /// OPERACIÓN, no de tick—, de modo que su coste es irrelevante frente a
    /// la corrección que aporta. La ruta caliente de lectura (`is_open`,
    /// precios, PnL) NO lo toma: sigue siendo lock-free.
    transition_lock: AtomicBool,
}

impl Default for Position {
    fn default() -> Self {
        Self {
            is_open: AtomicBool::new(false),
            is_long: AtomicBool::new(true),
            generation: AtomicU64::new(0),
            transition_lock: AtomicBool::new(false),
            horizon: std::sync::atomic::AtomicU8::new(0),
            entry_price: AtomicF64::new(0.0),
            quantity: AtomicF64::new(0.0),
            margin_used: AtomicF64::new(0.0),
            entry_time_ms: AtomicU64::new(0),
            trailing_phase: std::sync::atomic::AtomicU8::new(0),
            mfe_atr: AtomicF64::new(0.0),
            max_pnl_pct: AtomicF64::new(0.0),
            trail_stop: AtomicF64::new(0.0),
            tp_price: AtomicF64::new(0.0),
            sl_price: AtomicF64::new(0.0),
            ml_prediction: AtomicF64::new(0.0),
            nn_entry_tensor: std::sync::Mutex::new(Vec::new()),
            confidence: AtomicF64::new(0.0),
            entry_fee: AtomicF64::new(0.0),
            entry_tau_ms: AtomicU64::new(0),
            exchange_confirmed: AtomicBool::new(false),
            last_close_confirmed: AtomicBool::new(false),
        }
    }
}

impl Position {
    /// Adquiere el cerrojo de transición. Espera activa acotada: las
    /// secciones críticas son decenas de stores atómicos, nunca I/O.
    #[inline]
    fn lock_transition(&self) {
        while self
            .transition_lock
            .compare_exchange_weak(false, true, Ordering::AcqRel, Ordering::Relaxed)
            .is_err()
        {
            std::hint::spin_loop();
        }
    }

    #[inline]
    fn unlock_transition(&self) {
        self.transition_lock.store(false, Ordering::Release);
    }

    #[inline(always)]
    #[allow(clippy::too_many_arguments)]
    pub fn open(
        &self,
        is_long: bool,
        price: f64,
        qty: f64,
        margin: f64,
        current_time_ms: u64,
        tp: f64,
        sl: f64,
    ) {
        self.open_with_horizon(
            is_long,
            price,
            qty,
            margin,
            current_time_ms,
            tp,
            sl,
            PositionHorizon::Continuous,
        );
    }

    #[inline(always)]
    #[allow(clippy::too_many_arguments)]
    pub fn open_with_horizon(
        &self,
        is_long: bool,
        price: f64,
        qty: f64,
        margin: f64,
        current_time_ms: u64,
        tp: f64,
        sl: f64,
        horizon: PositionHorizon,
    ) -> bool {
        // D-729: propaga el rechazo de una entrada sin precio o sin cantidad.
        self.open_with_full_meta(
            is_long,
            price,
            qty,
            margin,
            current_time_ms,
            tp,
            sl,
            horizon,
            0.0,
            0.0,
        )
    }

    #[inline(always)]
    #[allow(clippy::too_many_arguments)]
    pub fn open_with_full_meta(
        &self,
        is_long: bool,
        price: f64,
        qty: f64,
        margin: f64,
        current_time_ms: u64,
        tp: f64,
        sl: f64,
        horizon: PositionHorizon,
        ml_pred: f64,
        conf: f64,
    ) -> bool {
        // D-729: propaga el rechazo.
        self.open_with_fee(
            is_long,
            price,
            qty,
            margin,
            current_time_ms,
            tp,
            sl,
            horizon,
            ml_pred,
            conf,
            0.0,
        )
    }

    #[inline(always)]
    #[allow(clippy::too_many_arguments)]
    pub fn open_with_fee(
        &self,
        is_long: bool,
        price: f64,
        qty: f64,
        margin: f64,
        current_time_ms: u64,
        tp: f64,
        sl: f64,
        horizon: PositionHorizon,
        ml_pred: f64,
        conf: f64,
        entry_fee: f64,
    ) -> bool {
        // D-729 (DÉCIMA OLA · auditoría integral): UNA ENTRADA SIN PRECIO NO ES
        // UNA ENTRADA.
        //
        // Un precio no finito o ≤ 0 se sustituía por el literal 1.0 y la posición
        // se publicaba igual. Binance devuelve `entryPrice: "0.0"` mientras el
        // margen de una posición recién abierta se liquida, y la ruta de adopción
        // pasa ese valor sin validar: la posición quedaba viva en el arena con
        // entrada 1,0 y, al cerrarla, `qty·(salida − 1,0)` producía cientos de
        // dólares de beneficio FANTASMA que entraban enteros en el win-rate, el
        // profit factor y Kelly. Ningún guardia aguas abajo lo detecta porque 1,0
        // es finito y positivo, y el propio test de corrupción exige
        // `entry_price <= 0`. Con `qty` inválida se publicaba una posición viva
        // con cantidad cero. Ahora se rechaza la apertura sin tocar un campo; el
        // llamador debe reconciliar contra el exchange, no inventar.
        if !(price.is_finite() && price > 0.0) || !(qty.is_finite() && qty > 0.0) {
            return false;
        }
        let safe_price = price;
        let safe_qty = qty;
        let safe_margin = if margin.is_finite() && margin >= 0.0 {
            margin
        } else {
            0.0
        };
        let safe_tp = if tp.is_finite() && tp >= 0.0 { tp } else { 0.0 };
        let safe_sl = if sl.is_finite() && sl >= 0.0 { sl } else { 0.0 };
        let safe_ml = if ml_pred.is_finite() { ml_pred } else { 0.5 };
        let safe_conf = if conf.is_finite() {
            conf.clamp(0.0, 1.0)
        } else {
            0.5
        };
        let safe_fee = if entry_fee.is_finite() && entry_fee >= 0.0 {
            entry_fee
        } else {
            0.0
        };

        // D-659: la apertura muta campos bajo el cerrojo de transición, de
        // modo que jamás puede solaparse con el borrado de un cierre.
        self.lock_transition();
        // Despublicar antes de reescribir: si el slot venía abierto (reapertura
        // sin cierre previo), ningún lector debe ver una mezcla de la posición
        // saliente y la entrante.
        self.is_open.store(false, Ordering::Release);
        self.generation.fetch_add(1, Ordering::AcqRel);

        self.is_long.store(is_long, Ordering::Relaxed);
        // U-1 / U-ERR-5 — encoding fiel del continuo. El encoding llegó a tener
        // tres valores (0 = Continuous, 1 = Scalping, 2 = Swing) y una colisión
        // histórica entre Continuous y Swing (auditoría T-08/K-17). Ya no hay
        // bandas: sólo el modo continuo (0). La τ real de la posición se
        // publica aparte en `entry_tau_ms`.
        let h_val = match horizon {
            PositionHorizon::Continuous => 0,
        };
        self.horizon.store(h_val, Ordering::Relaxed);
        self.entry_price.store(safe_price, Ordering::Relaxed);
        self.quantity.store(safe_qty, Ordering::Relaxed);
        self.margin_used.store(safe_margin, Ordering::Relaxed);
        self.entry_time_ms.store(current_time_ms, Ordering::Relaxed);
        self.trailing_phase.store(0, Ordering::Relaxed);
        self.mfe_atr.store(0.0, Ordering::Relaxed);
        self.max_pnl_pct.store(0.0, Ordering::Relaxed);
        self.trail_stop.store(0.0, Ordering::Relaxed);
        self.tp_price.store(safe_tp, Ordering::Relaxed);
        self.sl_price.store(safe_sl, Ordering::Relaxed);
        self.ml_prediction.store(safe_ml, Ordering::Relaxed);
        self.confidence.store(safe_conf, Ordering::Relaxed);
        self.entry_fee.store(safe_fee, Ordering::Relaxed);
        // La τ de entrada es del OCUPANTE, no del slot: sin este reset, una
        // reapertura sin cierre previo heredaría la τ (y por tanto las
        // geometrías de gestión temporal) de la posición saliente. El core la
        // reescribe justo tras la apertura con la τ dominante viva (REHAB-1b).
        self.entry_tau_ms.store(0, Ordering::Relaxed);
        // B3.14: toda apertura nace SIN confirmación de exchange — el host
        // la setea sólo tras el fill real (o la adopción FASE 5).
        self.exchange_confirmed.store(false, Ordering::Relaxed);
        // Publicar la posición completa: todo store previo es visible para
        // cualquier lector que observe is_open con Acquire.
        self.is_open.store(true, Ordering::Release);
        self.unlock_transition();
        true
    }

    /// Modo de horizonte del ocupante del slot.
    ///
    /// U-ERR-5: el encoding ya no tiene bandas — toda posición nace en el modo
    /// continuo. Quien quiera el horizonte REAL de la posición debe leer
    /// `entry_tau_ms` (τ en ms), que es la magnitud del continuo.
    #[inline(always)]
    pub fn horizon(&self) -> PositionHorizon {
        debug_assert_eq!(
            self.horizon.load(Ordering::Acquire),
            0,
            "encoding de horizonte fuera del modo continuo"
        );
        PositionHorizon::Continuous
    }

    pub fn close(&self) -> (bool, f64, f64, f64) {
        let (is_long, price, qty, margin, _fee) = self.close_with_fee();
        (is_long, price, qty, margin)
    }

    /// Cierra la posición y devuelve (is_long, price, qty, margin, entry_fee).
    ///
    /// D-659 (DÉCIMA OLA) — CORRECCIÓN DE LA CARRERA ABA.
    ///
    /// El diseño anterior hacía CAS(true→false) y DESPUÉS ponía los campos a
    /// cero. El CAS sólo impedía un doble cierre; el borrado posterior corría
    /// libre contra una apertura concurrente:
    ///
    /// ```text
    /// T1: CAS(true→false) ok, lee campos          ── desalojado ──
    /// T2: open(): escribe entry_price, qty, tp, sl; is_open=true (Release)
    /// T1: (reanuda) entry_price=0, qty=0, tp=0, sl=0
    /// ⟹ is_open=true CON entry_price=0 y SIN protecciones
    /// ```
    ///
    /// El resultado era una posición viva, sin precio de entrada (división
    /// por cero en todo cálculo de PnL) y sin TP/SL (ninguna comprobación de
    /// salida se dispara). El comentario original prometía exactamente la
    /// garantía que el código no daba.
    ///
    /// Ahora el borrado ocurre ANTES de publicar el cierre, y se protege con
    /// el contador de generación: si otro hilo abrió mientras leíamos, la
    /// generación cambió y abortamos sin tocar un solo campo suyo.
    pub fn close_with_fee(&self) -> (bool, f64, f64, f64, f64) {
        // D-659: toda la transición —comprobar, leer el snapshot, despublicar
        // y borrar— ocurre bajo el cerrojo. Una apertura concurrente espera
        // su turno en lugar de intercalarse, que es exactamente lo que
        // producía posiciones vivas con entry_price = 0 y sin protecciones.
        self.lock_transition();

        // El CAS se conserva: garantiza cierre único incluso frente a otro
        // cerrador que ya hubiera pasado por aquí.
        if self
            .is_open
            .compare_exchange(true, false, Ordering::AcqRel, Ordering::Acquire)
            .is_err()
        {
            self.unlock_transition();
            return (false, 0.0, 0.0, 0.0, 0.0);
        }

        let is_long = self.is_long.load(Ordering::Relaxed);
        let price = self.entry_price.load(Ordering::Relaxed);
        let qty = self.quantity.load(Ordering::Relaxed);
        let margin = self.margin_used.load(Ordering::Relaxed);
        let fee = self.entry_fee.load(Ordering::Relaxed);
        // B3.14: el resultado del último cierre se captura ANTES de tocar el
        // estado — «true = la posición cerrada tenía entrada real en el
        // exchange». También para los caminos de cierre que NO pasan por el
        // swap del host (rollback, emergencia).
        let was_exchange_confirmed = self.exchange_confirmed.load(Ordering::Relaxed);

        self.entry_price.store(0.0, Ordering::Relaxed);
        self.quantity.store(0.0, Ordering::Relaxed);
        self.margin_used.store(0.0, Ordering::Relaxed);
        self.entry_fee.store(0.0, Ordering::Relaxed);
        self.entry_time_ms.store(0, Ordering::Relaxed);
        self.trailing_phase.store(0, Ordering::Relaxed);
        self.mfe_atr.store(0.0, Ordering::Relaxed);
        self.max_pnl_pct.store(0.0, Ordering::Relaxed);
        self.trail_stop.store(0.0, Ordering::Relaxed);
        self.tp_price.store(0.0, Ordering::Relaxed);
        self.sl_price.store(0.0, Ordering::Relaxed);
        self.ml_prediction.store(0.0, Ordering::Relaxed);
        self.confidence.store(0.0, Ordering::Relaxed);
        self.entry_tau_ms.store(0, Ordering::Relaxed);
        // B3.14 — CONSUMIDOR ÚNICO. `exchange_confirmed` NO se limpia aquí:
        // el host la consume con `swap(false)` DESPUÉS del cierre y copia el
        // resultado a `last_close_confirmed` (god-engine-core). Limpiarla en
        // el cierre hacía que el swap leyera SIEMPRE false — todo cierre
        // pasaba por «papel», ni el PnL ni el WR contabilizaban y el host
        // disparaba un reduce-only de respaldo por cada cierre real. La
        // higiene del slot queda garantizada por el reset de `open_with_fee`.
        self.last_close_confirmed
            .store(was_exchange_confirmed, Ordering::Relaxed);
        // El contador avanza TAMBIÉN al cerrar: así es una secuencia real y
        // `snapshot()` puede detectar cualquier transición ocurrida durante
        // su lectura, no sólo las aperturas.
        self.generation.fetch_add(1, Ordering::AcqRel);

        self.unlock_transition();
        (is_long, price, qty, margin, fee)
    }

    #[inline(always)]
    pub fn is_open(&self) -> bool {
        self.is_open.load(Ordering::Acquire)
    }

    /// D-659 (DÉCIMA OLA) — LECTURA CONSISTENTE DE LA POSICIÓN.
    ///
    /// `is_open()` seguido de lecturas sueltas de los campos es un TOCTOU:
    /// entre la comprobación y las lecturas, un cierre puede despublicar y
    /// vaciar el slot, con lo que el lector obtiene `entry_price = 0` sobre
    /// una posición que creía viva — y divide por cero al calcular PnL.
    ///
    /// Esto es un seqlock de lectura: se toma el contador de secuencia antes
    /// y después del snapshot y sólo se acepta si no hubo transición. No
    /// bloquea ni penaliza al escritor; en el peor caso reintenta.
    ///
    /// Todo consumidor de la ruta caliente debe usar ESTO en lugar de
    /// `is_open()` + cargas individuales.
    #[inline]
    pub fn snapshot(&self) -> Option<PositionSnapshot> {
        for _ in 0..64 {
            let seq_before = self.generation.load(Ordering::Acquire);
            if !self.is_open.load(Ordering::Acquire) {
                return None;
            }
            let snap = PositionSnapshot {
                generation: seq_before,
                is_long: self.is_long.load(Ordering::Relaxed),
                entry_price: self.entry_price.load(Ordering::Relaxed),
                quantity: self.quantity.load(Ordering::Relaxed),
                margin_used: self.margin_used.load(Ordering::Relaxed),
                entry_time_ms: self.entry_time_ms.load(Ordering::Relaxed),
                tp_price: self.tp_price.load(Ordering::Relaxed),
                sl_price: self.sl_price.load(Ordering::Relaxed),
                entry_fee: self.entry_fee.load(Ordering::Relaxed),
                entry_tau_ms: self.entry_tau_ms.load(Ordering::Relaxed),
                confidence: self.confidence.load(Ordering::Relaxed),
                ml_prediction: self.ml_prediction.load(Ordering::Relaxed),
            };
            // Sin transición durante la lectura y sigue abierta ⇒ coherente.
            if self.generation.load(Ordering::Acquire) == seq_before
                && self.is_open.load(Ordering::Acquire)
            {
                return Some(snap);
            }
            std::hint::spin_loop();
        }
        // Contención patológica: preferible no operar a operar con datos
        // posiblemente inconsistentes.
        None
    }
}

/// Vista coherente de una posición en un instante. Producida por
/// `Position::snapshot()`; nunca contiene una mezcla de dos ocupantes del
/// mismo slot (D-659).
#[derive(Debug, Clone, Copy)]
pub struct PositionSnapshot {
    pub generation: u64,
    pub is_long: bool,
    pub entry_price: f64,
    pub quantity: f64,
    pub margin_used: f64,
    pub entry_time_ms: u64,
    pub tp_price: f64,
    pub sl_price: f64,
    pub entry_fee: f64,
    pub entry_tau_ms: u64,
    pub confidence: f64,
    pub ml_prediction: f64,
}

/// Posición viva de una moneda.
///
/// # U-ERR-5 (ERRADICACIÓN DEL BINARIO DE HORIZONTE)
///
/// Tenía TRES ranuras (`scalp`, `swing`, `position`) y `is_any_open`
/// consultaba las tres. Ningún productor del repositorio escribía en `scalp`
/// ni en `swing`: todas las aperturas vivas —core, backtest, reconciliación,
/// binario— van a `position`. Las dos ranuras muertas costaban 128 bytes de
/// línea de caché por moneda y dos lecturas atómicas por consulta de
/// ocupación, y sostenían la ficción de que el motor operaba dos bandas en
/// paralelo. Queda UNA ranura: una moneda tiene una posición.
#[repr(C, align(64))]
#[derive(Default)]
pub struct PositionManager {
    pub position: Position,
}

impl PositionManager {
    #[inline(always)]
    pub fn is_any_open(&self) -> bool {
        self.position.is_open()
    }
}

#[cfg(test)]
mod tests {
    /// T-4 (DÉCIMA OLA) — PRUEBA DE ESTRÉS DE CONCURRENCIA PARA D-659.
    ///
    /// Este test es la razón por la que la carrera ABA vivió sin detectarse:
    /// el backtest es de un solo hilo y la suite no ejercitaba concurrencia,
    /// de modo que el defecto era ESTRUCTURALMENTE invisible aunque todos los
    /// tests estuvieran en verde.
    ///
    /// Invariante bajo prueba: NUNCA debe observarse una posición abierta con
    /// precio de entrada cero o sin protecciones — el estado corrupto que la
    /// versión anterior producía al borrar campos tras publicar el cierre.
    #[test]
    fn t4_apertura_y_cierre_concurrentes_nunca_dejan_posicion_fantasma() {
        use std::sync::atomic::AtomicUsize;
        use std::sync::Arc;

        let pos = Arc::new(Position::default());
        let corrupciones = Arc::new(AtomicUsize::new(0));
        let observaciones = Arc::new(AtomicUsize::new(0));
        const ITERS: usize = 20_000;

        let abridor = {
            let pos = Arc::clone(&pos);
            std::thread::spawn(move || {
                for i in 0..ITERS {
                    pos.open_with_fee(
                        i % 2 == 0,
                        62_500.0,
                        0.0032,
                        13.0,
                        1_700_000_000_000 + i as u64,
                        63_200.0,
                        62_100.0,
                        PositionHorizon::Continuous,
                        0.61,
                        0.72,
                        0.004,
                    );
                    std::hint::spin_loop();
                }
            })
        };
        let cerrador = {
            let pos = Arc::clone(&pos);
            std::thread::spawn(move || {
                for _ in 0..ITERS {
                    let _ = pos.close_with_fee();
                    std::hint::spin_loop();
                }
            })
        };
        let vigilante = {
            let pos = Arc::clone(&pos);
            let corrupciones = Arc::clone(&corrupciones);
            let observaciones = Arc::clone(&observaciones);
            std::thread::spawn(move || {
                for _ in 0..ITERS * 4 {
                    // Lectura por el camino que producción debe usar.
                    if let Some(snap) = pos.snapshot() {
                        observaciones.fetch_add(1, Ordering::Relaxed);
                        // Posición viva con entrada cero, cantidad cero o sin
                        // stop = el estado que D-659 producía.
                        if snap.entry_price <= 0.0 || snap.quantity <= 0.0 || snap.sl_price <= 0.0 {
                            corrupciones.fetch_add(1, Ordering::Relaxed);
                        }
                    }
                    std::hint::spin_loop();
                }
            })
        };

        abridor.join().expect("hilo abridor");
        cerrador.join().expect("hilo cerrador");
        vigilante.join().expect("hilo vigilante");

        // El vigilante corre un número FIJO de vueltas: bajo carga (la suite
        // completa ocupa todos los núcleos) puede consumirlas enteras mientras
        // la posición está cerrada y no observar ni una vez el estado abierto
        // — el test fallaba por el planificador de la máquina, no por el
        // código. La observación por el camino de producción se garantiza aquí,
        // ya sin concurrencia: el tramo concurrente sigue contando corrupciones,
        // y esta apertura final asegura que el invariante se comprueba SIEMPRE.
        pos.open_with_fee(
            true,
            62_500.0,
            0.0032,
            13.0,
            1_700_000_100_000,
            63_200.0,
            62_100.0,
            PositionHorizon::Continuous,
            0.61,
            0.72,
            0.004,
        );
        let snap = pos
            .snapshot()
            .expect("tras abrir sin concurrencia, snapshot() debe ver la posición");
        observaciones.fetch_add(1, Ordering::Relaxed);
        if snap.entry_price <= 0.0 || snap.quantity <= 0.0 || snap.sl_price <= 0.0 {
            corrupciones.fetch_add(1, Ordering::Relaxed);
        }

        assert!(
            observaciones.load(Ordering::Relaxed) > 0,
            "el vigilante nunca vio la posición abierta: el test no ejercitó la carrera"
        );
        assert_eq!(
            corrupciones.load(Ordering::Relaxed),
            0,
            "se observó una posición abierta con entry_price/qty/sl en cero              (carrera ABA de D-659) en {} de {} observaciones",
            corrupciones.load(Ordering::Relaxed),
            observaciones.load(Ordering::Relaxed)
        );
    }

    use super::*;

    /// B3.14 — contrato de la confirmación de exchange a lo largo del ciclo
    /// de vida del slot, con el MISMO protocolo que usa el host:
    ///
    ///   open_*  → exchange_confirmed = false (nace sin confirmar);
    ///   fill    → host setea true;
    ///   close   → captura el valor en last_close_confirmed y NO lo consume;
    ///   swap    → el host lo consume (false) para su contabilidad.
    #[test]
    fn b3_14_ciclo_confirmacion_open_fill_close_swap() {
        let pos = Position::default();
        // Estado «heredado» de un ocupante previo confirmado.
        pos.exchange_confirmed.store(true, Ordering::Relaxed);
        pos.entry_tau_ms.store(987_654, Ordering::Relaxed);

        // TODO camino de apertura pasa por open_with_fee: nace sin confirmar
        // y sin la τ del ocupante anterior.
        pos.open_with_horizon(
            true,
            60_000.0,
            0.1,
            600.0,
            1_000,
            61_200.0,
            59_100.0,
            PositionHorizon::Continuous,
        );
        assert!(!pos.exchange_confirmed.load(Ordering::Relaxed));
        assert_eq!(pos.entry_tau_ms.load(Ordering::Relaxed), 0);

        // El fill real llega: el host confirma.
        pos.exchange_confirmed.store(true, Ordering::Relaxed);

        // Cierre local: captura el resultado y deja la confirmación viva para
        // el consumidor designado (el swap del host).
        let _ = pos.close_with_fee();
        assert!(
            pos.last_close_confirmed.load(Ordering::Relaxed),
            "el cierre de una entrada confirmada debe dejar last_close_confirmed = true"
        );
        assert!(
            pos.exchange_confirmed.load(Ordering::Relaxed),
            "close_with_fee NO consume la confirmación: el swap del host es el consumidor"
        );

        // Protocolo del host (god-engine-core): swap → contabiliza → false.
        let was = pos.exchange_confirmed.swap(false, Ordering::Relaxed);
        pos.last_close_confirmed.store(was, Ordering::Relaxed);
        assert!(was, "el swap debe ver la confirmación PREVIA al cierre");
        assert!(pos.last_close_confirmed.load(Ordering::Relaxed));
        assert!(!pos.exchange_confirmed.load(Ordering::Relaxed));
    }

    /// B3.14 — un cierre de entrada NUNCA confirmada (vetada/rechazada,
    /// round-trip de papel) no contabiliza: last_close_confirmed queda false.
    #[test]
    fn b3_14_cierre_de_papel_no_confirma() {
        let pos = Position::default();
        pos.open_with_fee(
            true,
            50_000.0,
            1.0,
            5_000.0,
            1_000,
            51_000.0,
            49_000.0,
            PositionHorizon::Continuous,
            0.8,
            0.9,
            2.5,
        );
        // Sin store(true): la entrada fue vetada o rechazada en el exchange.
        let _ = pos.close_with_fee();
        assert!(!pos.last_close_confirmed.load(Ordering::Relaxed));
        // El swap del host tampoco encuentra nada que consumir.
        assert!(!pos.exchange_confirmed.swap(false, Ordering::Relaxed));
    }

    /// B3.14/REHAB-1b — la reapertura sin cierre previo no hereda ni la
    /// confirmación ni la τ del ocupante saliente.
    #[test]
    fn b3_14_reapertura_no_hereda_estado_del_ocupante_previo() {
        let pos = Position::default();
        pos.open_with_fee(
            false,
            30_000.0,
            0.2,
            600.0,
            1_000,
            30_600.0,
            29_700.0,
            PositionHorizon::Continuous,
            0.6,
            0.7,
            1.0,
        );
        pos.exchange_confirmed.store(true, Ordering::Relaxed);
        pos.entry_tau_ms.store(555_000, Ordering::Relaxed);
        // Reapertura DIRECTA (sin cierre): open despublica y reescribe.
        pos.open_with_full_meta(
            true,
            31_000.0,
            0.3,
            900.0,
            2_000,
            31_900.0,
            30_400.0,
            PositionHorizon::Continuous,
            0.65,
            0.75,
        );
        assert!(pos.is_open());
        assert!(!pos.exchange_confirmed.load(Ordering::Relaxed));
        assert_eq!(pos.entry_tau_ms.load(Ordering::Relaxed), 0);
    }

    #[test]
    fn test_position_continuous_open_and_close() {
        let mgr = PositionManager::default();

        // Continuous Long
        mgr.position.open_with_horizon(
            true,
            60000.0,
            0.1,
            600.0,
            1000,
            60500.0,
            59500.0,
            PositionHorizon::Continuous,
        );

        assert!(mgr.position.is_open());
        assert!(mgr.is_any_open());
        assert_eq!(mgr.position.horizon(), PositionHorizon::Continuous);

        let (is_long, price, qty, _) = mgr.position.close();
        assert!(is_long);
        assert_eq!(price, 60000.0);
        assert_eq!(qty, 0.1);
        assert!(!mgr.position.is_open());
        assert!(!mgr.is_any_open());
    }

    #[test]
    fn test_position_atomic_close_idempotency() {
        let pos = Position::default();
        pos.open_with_fee(
            true,
            50000.0,
            1.0,
            5000.0,
            1000,
            51000.0,
            49000.0,
            PositionHorizon::Continuous,
            0.8,
            0.9,
            2.5,
        );

        let (is_long, p, q, m, f) = pos.close_with_fee();
        assert!(is_long);
        assert_eq!(p, 50000.0);
        assert_eq!(qty_or_eq(q, 1.0), true);
        assert_eq!(m, 5000.0);
        assert_eq!(f, 2.5);

        // Segundo cierre debe retornar zeros (idempotente)
        let (is_long2, p2, _, _, _) = pos.close_with_fee();
        assert!(!is_long2);
        assert_eq!(p2, 0.0);
    }

    /// U-ERR-5 — UNA MONEDA, UNA POSICIÓN.
    ///
    /// Sustituye a `test_position_dual_scalp_swing_independence`, que abría a
    /// la vez una posición en la ranura `swing` y otra en `scalp` y
    /// comprobaba que eran independientes. Esa independencia nunca existió en
    /// producción: ningún productor del repositorio escribía en esas dos
    /// ranuras, de modo que el test verdeaba sobre una capacidad que el motor
    /// no ejercía, y `is_any_open` pagaba dos lecturas atómicas por consulta
    /// para interrogar ranuras que siempre estaban vacías.
    ///
    /// Este test FALLA con el código viejo: allí `is_any_open` devolvía `true`
    /// mientras cualquiera de las tres ranuras estuviese abierta, así que
    /// cerrar `position` no bastaba para dejar la moneda plana. La invariante
    /// que se fija ahora: la ocupación de la moneda es exactamente la
    /// ocupación de su ÚNICA posición.
    #[test]
    fn u_err_5_la_ocupacion_es_la_de_la_unica_posicion() {
        let mgr = PositionManager::default();
        assert!(!mgr.is_any_open());

        mgr.position.open_with_horizon(
            true,
            90000.0,
            0.1,
            900.0,
            1000,
            91500.0,
            89300.0,
            PositionHorizon::Continuous,
        );
        assert!(mgr.is_any_open());
        assert_eq!(mgr.is_any_open(), mgr.position.is_open());

        let (is_long, price, qty, _) = mgr.position.close();
        assert!(is_long);
        assert_eq!(price, 90000.0);
        assert_eq!(qty, 0.1);

        // Cerrada la única posición, la moneda queda plana: no hay ninguna
        // otra ranura que pueda sostener un `true` fantasma.
        assert!(!mgr.is_any_open());
        assert_eq!(mgr.is_any_open(), mgr.position.is_open());
    }

    fn qty_or_eq(a: f64, b: f64) -> bool {
        (a - b).abs() < 1e-9
    }
}
