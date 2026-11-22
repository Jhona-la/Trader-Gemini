use super::*;

pub(super) const ARBITER_TICK: Duration = Duration::from_millis(25);

// -------------------------------------------------------------------------------------------
// R2: snapshot REST periodico por simbolo (red de seguridad contra huecos silenciosos).
//
// El gap del stream solo se detecta cuando llegan frames nuevos (offer/l2.rs), asi que un
// silencio con el socket vivo no dispara nunca el re-anclaje existente (public_l2.rs write:
// generation/needs_snapshot). R2 pone un temporizador: aunque el stream calle, cada simbolo
// recibe un ancla fresca a lo sumo cada `interval`, acotando la perdida a interval+epsilon.
//
// El intervalo SALE DE UNA CUENTA, no de una constante suelta:
//   T = max(max_gap, ceil(N * w * W / (f*L - N*w - base)))
// con L = limite REQUEST_WEIGHT de exchangeInfo (docs Binance USDT-M Futures, 2400/min por
// IP, https://developers.binance.com/docs/derivatives/usds-margined-futures/general-info;
// verificado en vivo 2026-11-17: exchangeInfo.rateLimits y cabecera X-MBX-USED-WEIGHT-1M),
// w = peso del snapshot /fapi/v1/depth limit=1000 = 20 (misma URL, endpoint Order Book;
// verificado en vivo 2026-11-17: used-weight 3 -> 23), N = simbolos depth del roster,
// f = 0.8 (regla anti-bloqueo del operador: quedarse bajo el 80 % del limite publicado),
// N*w = reserva de tormenta (un snapshot por simbolo tras un corte de red, misma ventana) y
// base = linea base ya consumida por la captura (1.2 snapshots/simbolo/h medidos en la
// prueba de 2 h: 95 snapshots, 40 simbolos). Con N=40, L=2400, w=20: presupuesto R2 =
// 1920 - 800 - 16 = 1104 peso/min -> piso 44 s; el objetivo pre-registrado (analisis del
// 17-nov §3: el techo de vacio interior es el propio intervalo) manda: T = 300 s =
// 160 peso/min = 8.3 % del tope anti-bloqueo. Todo se recalcula en vivo con los pesos que
// la puerta aprende de las cabeceras; defaults solo antes del primer exchangeInfo.
//
// v2 (N1 de WorkBuddy): cada exito rearma el temporizador con el intervalo del plan
// vigente y la fase solo se aplica al primer armado. La cinta registra el plan resuelto
// en la fila r2_plan al arrancar y cuando el shard observa un cambio. Con limite
// desconocido el plan usa defaults de docs. La puerta espera a conocer el limite
// antes de servir snapshots depth (gate.rs: limits.is_some).
// -------------------------------------------------------------------------------------------

/// Objetivo de garantia: techo de re-anclaje tras hueco silencioso (analisis 17-nov §3,
/// pre-registrado). Ajustable por argumento `r2_max_gap_s` sin recompilar disenos.
pub(super) const R2_MAX_GAP_DEFAULT: Duration = Duration::from_secs(300);
const R2_BUDGET_FACTOR: f64 = 0.8;
/// Snapshots/simbolo/hora que la captura ya emite hoy (arranques, rotaciones y huecos):
/// medido 95 snapshots / 2 h / 40 simbolos en la prueba de 2 h (END.json kinds.snapshot).
const R2_BASELINE_PER_SYMBOL_H: f64 = 1.2;
const R2_DEFAULT_LIMIT: u64 = 2400;
const R2_DEFAULT_WINDOW_S: u64 = 60;
const R2_DEFAULT_WEIGHT: u64 = 20;

/// Plan R2 resuelto. `defaulted` distingue el plan de arranque (limite/peso todavia no
/// aprendidos: defaults de docs) del plan que rige tras el primer exchangeInfo + pesos.
#[derive(PartialEq)]
pub(super) struct R2Plan {
    pub(super) interval: Duration,
    pub(super) symbols: usize,
    pub(super) limit: u64,
    pub(super) window_s: u64,
    pub(super) weight: u64,
    pub(super) defaulted: bool,
}

impl R2Plan {
    /// Fila `r2_plan` de la cinta (N1 cambio 2): T, N, L, w, ventana y motivo, para
    /// auditar con que intervalo se rearman los temporizadores.
    fn row(&self, max_gap: Duration, reason: &str) -> Value {
        let cycle = (self.symbols as u64 * self.weight) as f64;
        json!({
            "interval_ms": self.interval.as_millis() as u64,
            "symbols": self.symbols,
            "limit": self.limit,
            "window_s": self.window_s,
            "weight_depth": self.weight,
            "budget_factor": R2_BUDGET_FACTOR,
            "max_gap_ms": max_gap.as_millis() as u64,
            "periodic_weight_per_min": cycle * 60.0 / self.interval.as_secs_f64().max(1e-9),
            "defaulted": self.defaulted,
            "reason": reason,
        })
    }
}

pub(super) fn r2_plan(max_gap: Duration, symbols: usize, weight: u64, limit: u64, window_s: u64) -> R2Plan {
    let cycle = (symbols as u64 * weight) as f64;
    // Linea base por VENTANA: los snapshots/simbolo/hora ya emitidos escalados a la ventana
    // del limite (window_s=60 reproduce la cuenta original cycle*1.2/60; N1 cambio 5).
    let baseline = cycle * R2_BASELINE_PER_SYMBOL_H * window_s as f64 / 3600.0;
    // Presupuesto por ventana para R2: 80 % del limite, menos la tormenta de re-anclaje
    // (un snapshot por simbolo) y menos la linea base medida. floor(1): sin presupuesto,
    // R2 queda en la practica desactivado (intervalo enorme) en vez de arriesgar un 429.
    let budget = (limit as f64 * R2_BUDGET_FACTOR) - cycle - baseline;
    let floor_s = (cycle * window_s as f64 / budget.max(1.0)).ceil() as u64;
    R2Plan {
        interval: max_gap.max(Duration::from_secs(floor_s)),
        symbols, limit, window_s, weight,
        defaulted: false,
    }
}

/// Plan resuelto desde el estado publicado por la puerta (watch `info`: fila
/// REQUEST_WEIGHT del exchangeInfo + pesos aprendidos de las cabeceras). Watch sin
/// publicar o sin fila REQUEST_WEIGHT => defaults de docs con `defaulted=true`.
/// Interpreta SECOND/MINUTE/HOUR/DAY con el mismo criterio que la puerta (gate.rs).
pub(super) fn r2_plan_from_info(info: &Value, max_gap: Duration, symbols: usize) -> R2Plan {
    let row = &info["request_weight_limit"];
    let (mut limit, mut window_s, mut weight) = (R2_DEFAULT_LIMIT, R2_DEFAULT_WINDOW_S, R2_DEFAULT_WEIGHT);
    let mut defaulted = false;
    if let Some(n) = row["limit"].as_u64() {
        limit = n;
    } else {
        defaulted = true;
    }
    match (row["interval"].as_str(), row["intervalNum"].as_u64()) {
        (Some("SECOND"), Some(n)) => window_s = n,
        (Some("MINUTE"), Some(n)) => window_s = 60 * n,
        (Some("HOUR"), Some(n)) => window_s = 3600 * n,
        (Some("DAY"), Some(n)) => window_s = 86400 * n,
        _ => defaulted = true,
    }
    if let Some(w) = info["learned_weights"]["depth"].as_u64() {
        weight = w;
    } else {
        defaulted = true;
    }
    let mut plan = r2_plan(max_gap, symbols, weight, limit, window_s);
    plan.defaulted = defaulted;
    plan
}

/// Fase del simbolo con posicion `rank` en el roster (0..N): escalona los temporizadores
/// para no pedir todos los simbolos a la vez (un disparo cada T/N). Se aplica al ARMAR
/// (primer ancla): el primer periodico de cada simbolo cae entre T/N y T despues de su
/// ancla fresca, nunca inmediatamente (esa ancla acaba de llegar).
pub(super) fn r2_phase(rank: usize, total: usize, interval: Duration) -> Duration {
    if total == 0 {
        Duration::ZERO
    } else {
        interval * (((rank % total) + 1) as u32) / total as u32
    }
}

pub(super) struct R2Timer {
    next: Instant,
    interval: Duration,
}

impl R2Timer {
    fn arm(now: Instant, interval: Duration, phase: Duration) -> Self {
        Self { next: now + phase, interval }
    }

    fn due(&self, now: Instant) -> bool {
        now >= self.next
    }

    /// Rearme tras un exito: ADOPTA el intervalo recalculado del plan vigente (N1 cambio 1).
    /// La fase solo existe en el primer armado: el ancla acaba de llegar, el proximo
    /// periodico va a un intervalo completo con el T que rige AHORA, no con el congelado.
    fn reschedule(&mut self, now: Instant, interval: Duration) {
        self.interval = interval;
        self.next = now + interval;
    }
}

pub(super) struct PublicFrame {
    connection: u64,
    stamp: Stamp,
    stream: String,
    symbol: String,
    payload: String,
    ticker_id: Option<u64>,
}

impl PublicFrame {
    fn charge(&self) -> usize {
        self.payload.capacity()
            + self.stream.capacity()
            + self.symbol.capacity()
            + std::mem::size_of::<Self>()
    }

    fn emit(
        self,
        emitter: &Emitter,
        generation: Option<u64>,
        cont: Option<Cont>,
    ) -> CaptureResult<()> {
        emitter.emit(
            "frame", self.connection, self.stamp,
            json!({"stream":self.stream,"symbol":self.symbol,"payload":self.payload,"depth_generation":generation}),
            cont,
        )
    }
}

pub(super) struct DepthState {
    arbiter: l2::Depth<PublicFrame>,
    last_u: u64,
    generation: u64,
    needs_snapshot: bool,
    pending: Option<u64>,
    retry_at: Instant,
    r2: Option<R2Timer>,
}

impl DepthState {
    fn new(now: Instant) -> Self {
        Self {
            arbiter: l2::Depth::default(),
            last_u: 0,
            generation: 0,
            needs_snapshot: true,
            pending: None,
            retry_at: now,
            r2: None,
        }
    }

    /// Hay que pedir un snapshot ahora. Some(false) = arranque/hueco (prioridad 0/1),
    /// Some(true) = periodico R2 (prioridad 3, por debajo del relleno aggTrades).
    /// None mientras haya uno en vuelo o dentro del cooldown (429/418 incluyen aqui).
    fn snapshot_wanted(&self, now: Instant) -> Option<bool> {
        if self.pending.is_some() || now < self.retry_at {
            return None;
        }
        if self.needs_snapshot {
            return Some(false);
        }
        if self.r2.as_ref().is_some_and(|t| t.due(now)) {
            return Some(true);
        }
        None
    }

    /// Respuesta de la puerta para la generacion `generation` (antes cuerpo de
    /// ShardCtx::public_reply, aqui para poder probarlo sin levantar el shard).
    /// `arm` = (intervalo, fase) R2 del plan VIGENTE: se arma con la fase en el primer
    /// exito y se rearma en cada exito adoptando el intervalo recalculado (si el limite
    /// o el peso aprendido cambiaron, el nuevo T rige desde este rearme).
    fn reply(&mut self, generation: u64, ok: bool, cooldown: Duration, now: Instant, arm: Option<(Duration, Duration)>) {
        if self.pending != Some(generation) {
            return;
        }
        self.pending = None;
        if self.generation != generation {
            return;
        }
        self.needs_snapshot = !ok;
        self.retry_at = now + cooldown;
        if ok {
            match &mut self.r2 {
                Some(timer) => {
                    let interval = arm.map_or(timer.interval, |(i, _)| i);
                    timer.reschedule(now, interval);
                }
                None => {
                    if let Some((interval, phase)) = arm {
                        self.r2 = Some(R2Timer::arm(now, interval, phase));
                    }
                }
            }
        }
    }

    fn write(
        &mut self,
        emitter: &Emitter,
        updates: Vec<l2::DepthOutput<PublicFrame>>,
        now: Instant,
    ) -> CaptureResult<()> {
        for l2::DepthOutput { id, frame, gap } in updates {
            if self.generation == 0 || gap.is_some() {
                self.generation += 1;
                self.needs_snapshot = true;
                self.retry_at = now;
            }
            if let Some(gap) = gap {
                emitter.emit(
                    "depth_gap", frame.connection, frame.stamp,
                    json!({"symbol":frame.symbol,"stream":frame.stream,"previous_u":gap.expected,
                        "U":id.first,"u":id.last,"pu":id.previous,"depth_generation":self.generation,
                        "reason":gap.reason}),
                    None,
                )?;
            }
            self.last_u = id.last;
            let cont = Cont::Depth {
                symbol: frame.symbol.clone(),
                first: id.first,
                last: id.last,
                prev: id.previous,
            };
            frame.emit(emitter, Some(self.generation), Some(cont))?;
        }
        Ok(())
    }
}

fn write_ticker(emitter: &Emitter, output: Vec<PublicFrame>) -> CaptureResult<()> {
    for frame in output {
        let cont = frame.ticker_id.map(|u| Cont::Book {
            symbol: frame.symbol.clone(),
            u,
        });
        frame.emit(emitter, None, cont)?;
    }
    Ok(())
}

impl Conn {
    pub(super) fn public_frame(
        &self,
        payload: String,
        stamp: Stamp,
        stream: Option<String>,
        symbol: Option<String>,
        data: &Value,
    ) -> CaptureResult<bool> {
        let (Some(stream), Some(symbol)) = (stream, symbol) else {
            self.emitter.emit(
                "frame",
                self.id,
                stamp,
                json!({"payload":payload,"parse_error":true}),
                None,
            )?;
            return Ok(false);
        };
        if !self
            .l2_streams
            .get(&stream)
            .is_some_and(|expected| expected == &symbol)
        {
            self.emitter.emit(
                "frame",
                self.id,
                stamp,
                json!({"stream":stream,"symbol":symbol,"payload":payload,"parse_error":true}),
                None,
            )?;
            return Ok(false);
        }
        let is_depth = stream.ends_with("@depth@100ms");
        let id = if is_depth {
            let seq = data["U"]
                .as_u64()
                .zip(data["u"].as_u64())
                .zip(data["pu"].as_u64());
            let Some(((first, last), previous)) = seq.filter(|((f, l), p)| f <= l && p < l) else {
                self.emitter.emit(
                    "frame",
                    self.id,
                    stamp,
                    json!({"stream":stream,"symbol":symbol,
                    "payload":payload,"invalid_depth_sequence":true}),
                    None,
                )?;
                return Ok(false);
            };
            Some(l2::DepthId {
                first,
                last,
                previous,
            })
        } else {
            None
        };
        let now = Instant::now();
        let mut book = lock(&self.book);
        book.seen
            .entry(self.id)
            .or_default()
            .insert(stream.clone(), now);
        let ticker_id = if is_depth { None } else { data["u"].as_u64() };
        let frame = PublicFrame {
            connection: self.id,
            stamp,
            stream,
            symbol: symbol.clone(),
            payload,
            ticker_id,
        };
        let cost = frame.charge();
        if let Some(id) = id {
            let state = book
                .depth
                .entry(symbol)
                .or_insert_with(|| DepthState::new(now));
            let output = state.arbiter.offer(id, frame, cost, now);
            state.write(&self.emitter, output, now)?;
        } else if let Some(u) = ticker_id {
            let output = book
                .tickers
                .entry(symbol)
                .or_default()
                .offer(u, frame, cost, now);
            write_ticker(&self.emitter, output.ready)?;
            if let Some(frame) = output.retired {
                self.emitter.emit("book_gap", frame.connection, frame.stamp,
                    json!({"stream":frame.stream,"symbol":frame.symbol,"u":u,"payload":frame.payload,
                        "reason":"ID outside retained BookTicker history"}), None)?;
            }
        } else {
            frame.emit(&self.emitter, None, None)?;
        }
        Ok(true)
    }
}

impl ShardBook {
    fn drain_l2(&mut self, emitter: &Emitter, now: Instant, finish: bool) -> CaptureResult<()> {
        for state in self.depth.values_mut() {
            let output = if finish {
                state.arbiter.finish()
            } else {
                state.arbiter.expire(now)
            };
            state.write(emitter, output, now)?;
        }
        for ticker in self.tickers.values_mut() {
            let output = if finish {
                ticker.flush()
            } else {
                ticker.expire(now)
            };
            write_ticker(emitter, output)?;
        }
        Ok(())
    }

    fn recent_rotation_streams(
        &self,
        old: u64,
        new: u64,
        now: Instant,
    ) -> (Vec<String>, Vec<String>) {
        let recent = |connection| -> Vec<String> {
            let mut streams: Vec<_> = self
                .seen
                .get(&connection)
                .into_iter()
                .flatten()
                .filter(|(_, at)| now.saturating_duration_since(**at) <= OVERLAP_GRACE)
                .map(|(stream, _)| stream.clone())
                .collect();
            streams.sort_unstable();
            streams
        };
        let streams = recent(new);
        let missing = recent(old)
            .into_iter()
            .filter(|stream| streams.binary_search(stream).is_err())
            .collect();
        (missing, streams)
    }

    pub(super) fn l2_lost(&mut self, emitter: &Emitter, now: Instant) -> CaptureResult<()> {
        for state in self.depth.values_mut() {
            let output = state.arbiter.force_gap_on_next();
            state.write(emitter, output, now)?;
        }
        for ticker in self.tickers.values_mut() {
            write_ticker(emitter, ticker.flush())?;
        }
        Ok(())
    }
}

fn close(live: &mut Live, why: String) {
    if let Some(tx) = live.close.take() {
        let _ = tx.send(why);
    }
}

fn retry(delay: &mut Duration, lived: Duration, shard: usize) -> Duration {
    if lived >= BACKOFF_RESET_LIFE {
        *delay = BACKOFF_START;
    }
    let wait = *delay + jitter() + SHARD_OFFSET * shard as u32;
    *delay = (*delay * 2).min(BACKOFF_MAX);
    wait
}

impl ShardCtx {
    fn online(&self, live: &Live) -> bool {
        lock(&self.book).public_connections.contains(&live.id)
    }

    fn public_tick(
        &self,
        now: Instant,
        replies: &mpsc::UnboundedSender<gate::Reply>,
    ) -> CaptureResult<()> {
        let mut book = lock(&self.book);
        book.drain_l2(&self.emitter, now, false)?;
        for (symbol, state) in &mut book.depth {
            let Some(periodic) = state.snapshot_wanted(now) else {
                continue;
            };
            if !*self.stop.borrow() {
                self.gate
                    .send(gate::Request {
                        kind: gate::Kind::Depth,
                        priority: if periodic {
                            3 // R2: por debajo del relleno aggTrades (2) y del re-anclaje (0/1).
                        } else if state.generation > 1 {
                            0
                        } else {
                            1
                        },
                        shard: self.index,
                        symbol: symbol.clone(),
                        generation: state.generation,
                        trigger_u: state.last_u,
                        from_id: 0,
                        limit: 0,
                        periodic,
                        reply: Some(replies.clone()),
                    })
                    .map_err(|_| io::Error::other("REST gate unavailable for depth snapshot"))?;
                state.pending = Some(state.generation);
            }
        }
        Ok(())
    }

    fn public_reply(&self, rep: gate::Reply, plan: &R2Plan) {
        if let gate::Reply::Depth {
            symbol,
            generation,
            ok,
            cooldown,
        } = rep
        {
            let rank = self.r2_rank.get(&symbol).copied().unwrap_or(0);
            let phase = r2_phase(rank, self.r2_symbols_total, plan.interval);
            let mut book = lock(&self.book);
            if let Some(state) = book.depth.get_mut(&symbol) {
                state.reply(generation, ok, cooldown, Instant::now(), Some((plan.interval, phase)));
            }
        }
    }

    /// Plan R2 vigente: limite/ventana/peso leidos del estado publicado por la puerta
    /// (exchangeInfo + pesos aprendidos de X-MBX-USED-WEIGHT); defaults de docs mientras
    /// no llegue el primer exchangeInfo (plan marcado `defaulted`). Con limite N/D la
    /// puerta no sirve ningun depth hasta conocer el limite (gate.rs), asi que el plan
    /// de defaults solo gobierna temporizadores, nunca peticiones.
    fn current_r2_plan(&self) -> R2Plan {
        r2_plan_from_info(&self.gate_info.borrow(), self.r2_max_gap, self.r2_symbols_total)
    }

    // Cada shard registra el plan fuera del mutex del libro. Los temporizadores
    // ya armados lo adoptan cuando su siguiente snapshot tiene exito.
    fn refresh_r2_plan(&self, last_plan: &mut R2Plan) -> CaptureResult<()> {
        let plan = self.current_r2_plan();
        if plan != *last_plan {
            self.row("r2_plan", 0, plan.row(self.r2_max_gap, "cambio"))?;
            *last_plan = plan;
        }
        Ok(())
    }

    pub(super) async fn run_l2(self) -> CaptureResult<()> {
        let (life_tx, mut life) = mpsc::unbounded_channel();
        let (reply_tx, mut replies) = mpsc::unbounded_channel();
        let mut tasks = JoinSet::new();
        let mut stop = self.stop.clone();
        if *stop.borrow() {
            return Ok(());
        }
        tokio::select! {
            _ = stop.changed() => return Ok(()),
            _ = time::sleep(SHARD_OFFSET * self.index as u32) => {}
        }
        let mut r2_plan = self.current_r2_plan();
        self.row("r2_plan", 0, r2_plan.row(self.r2_max_gap, "arranque"))?;
        let mut active = Some(self.spawn_conn(&life_tx, &mut tasks));
        let mut backup: Option<Live> = None;
        let mut reconnect_at = None;
        let mut backup_at = None;
        let mut reconnect_delay = BACKOFF_START;
        let mut backup_delay = BACKOFF_START;
        let mut rotation_due = None;
        let mut first_rotation = true;
        let mut holding = false;
        let mut tick = time::interval(ARBITER_TICK);
        tick.set_missed_tick_behavior(MissedTickBehavior::Skip);

        while !*stop.borrow() {
            tokio::select! {
                _ = stop.changed() => {}
                Some(completed) = tasks.join_next(), if !tasks.is_empty() => {
                    completed.map_err(io::Error::other)?;
                }
                Some(ev) = life.recv() => match ev {
                    Life::Open(id) => {
                        let now = Instant::now();
                        if let Some(a) = active.as_mut().filter(|c| c.id == id) {
                            a.opened = Some(now);
                            // Desfasar los dos sockets al principio. Luego se rota por edad.
                            let span = if first_rotation {
                                first_rotation = false;
                                self.rotate_after.mul_f64(0.5 - 0.25 * self.index as f64 / self.total as f64)
                            } else {
                                self.rotate_after
                            };
                            rotation_due = Some(now + span);
                            if backup.is_none() {
                                backup = Some(self.spawn_conn(&life_tx, &mut tasks));
                                backup_at = None;
                            }
                        } else if let Some(b) = backup.as_mut().filter(|c| c.id == id) {
                            b.opened = Some(now);
                            self.row("backup_open", id, json!({"old_connection":active.as_ref().map(|a| a.id),"new_connection":id}))?;
                        }
                    }
                    Life::Stream(_) => {}
                    Life::Ended(_, _, _) if *stop.borrow() => {}
                    Life::Ended(id, why, coverage_lost) => {
                        let now = Instant::now();
                        if active.as_ref().is_some_and(|a| a.id == id) {
                            let lived = active.take().and_then(|a| a.opened).map(|x| now.duration_since(x)).unwrap_or_default();
                            holding = false;
                            if let Some(b) = backup.take() {
                                if self.online(&b) {
                                    let opened = b.opened.unwrap_or(now);
                                    self.row("backup_promoted", b.id, json!({"from_connection":id,"old_connection":id,"reason":why}))?;
                                    rotation_due = Some(opened + self.rotate_after);
                                    active = Some(b);
                                    backup = Some(self.spawn_conn(&life_tx, &mut tasks));
                                    backup_at = None;
                                } else if b.opened.is_none() {
                                    // Conserva el intento pendiente. Ended programará la
                                    // siguiente reconexión si este intento también falla.
                                    active = Some(b);
                                    rotation_due = None;
                                }
                            }
                            if active.is_none() {
                                let wait = retry(&mut reconnect_delay, lived, self.index);
                                reconnect_at = Some(now + wait);
                                rotation_due = None;
                                self.row("socket_gap", id, json!({"reason":why,"retry_ms":wait.as_millis() as u64,"coverage_lost":coverage_lost}))?;
                            } else if coverage_lost {
                                self.row("socket_gap", id, json!({"reason":why,"coverage_lost":true}))?;
                            }
                        } else if backup.as_ref().is_some_and(|b| b.id == id) {
                            let lived = backup.take().and_then(|b| b.opened).map(|x| now.duration_since(x)).unwrap_or_default();
                            let wait = retry(&mut backup_delay, lived, self.index);
                            backup_at = Some(now + wait);
                            if coverage_lost {
                                self.row("socket_gap", id, json!({"reason":why,"coverage_lost":true}))?;
                            } else {
                                self.row("backup_failed", id, json!({"reason":why,"retry_ms":wait.as_millis() as u64}))?;
                            }
                        } else if coverage_lost {
                            self.row("socket_gap", id, json!({"reason":why,"coverage_lost":true}))?;
                        }
                    }
                },
                Some(rep) = replies.recv() => {
                    self.refresh_r2_plan(&mut r2_plan)?;
                    self.public_reply(rep, &r2_plan);
                }
                _ = tick.tick() => {
                    self.refresh_r2_plan(&mut r2_plan)?;
                    let now = Instant::now();
                    if active.is_none() && reconnect_at.is_some_and(|t| now >= t) {
                        reconnect_at = None;
                        active = Some(self.spawn_conn(&life_tx, &mut tasks));
                    }
                    if backup.is_none() && active.as_ref().is_some_and(|a| a.opened.is_some())
                        && backup_at.is_some_and(|t| now >= t)
                    {
                        backup_at = None;
                        backup = Some(self.spawn_conn(&life_tx, &mut tasks));
                    }
                    if let (Some(a), Some(b), Some(due)) = (&active, &backup, rotation_due) {
                        if now >= due && self.online(b) {
                            if let Some(opened) = b.opened {
                                let (missing, streams) = lock(&self.book).recent_rotation_streams(a.id, b.id, now);
                                if missing.is_empty() && !streams.is_empty() {
                                    let (old_id, new_id) = (a.id, b.id);
                                    self.row("rotation_ready", new_id, json!({"old_connection":old_id,"new_connection":new_id,
                                        "overlap_ms":now.duration_since(opened).as_millis() as u64,
                                        "streams_on_new":streams,"missing":missing}))?;
                                    let mut old = active.take().expect("active checked");
                                    close(&mut old, format!("rotation to connection {new_id}"));
                                    self.row("rotation_close", old_id, json!({"old_connection":old_id,"new_connection":new_id,
                                        "old_lived_ms":old.opened.map(|o| o.elapsed().as_millis() as u64)}))?;
                                    active = backup.take();
                                    backup = Some(self.spawn_conn(&life_tx, &mut tasks));
                                    backup_at = None;
                                    rotation_due = Some(opened + self.rotate_after);
                                    holding = false;
                                } else if !holding && now.duration_since(due) >= OVERLAP_GRACE {
                                    holding = true;
                                    self.row("rotation_hold", b.id, json!({"old_connection":a.id,"new_connection":b.id,"missing":missing}))?;
                                }
                            }
                        }
                    }
                    self.public_tick(now, &reply_tx)?;
                }
            }
        }
        for mut live in [active, backup].into_iter().flatten() {
            close(&mut live, "stop".into());
        }
        while let Some(completed) = tasks.join_next().await {
            completed.map_err(io::Error::other)?;
        }
        lock(&self.book).drain_l2(&self.emitter, Instant::now(), true)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn planned_rotation_requires_recent_backup_activity() {
        let now = Instant::now();
        let stream = "oneusdt@depth@100ms".to_owned();
        let mut book = ShardBook::default();
        book.seen.entry(1).or_default().insert(stream.clone(), now);
        book.seen.entry(2).or_default().insert(
            stream.clone(),
            now - OVERLAP_GRACE - Duration::from_millis(1),
        );
        assert_eq!(
            book.recent_rotation_streams(1, 2, now),
            (vec![stream.clone()], vec![])
        );
        book.seen.entry(2).or_default().insert(stream.clone(), now);
        assert_eq!(
            book.recent_rotation_streams(1, 2, now),
            (vec![], vec![stream])
        );
    }

    fn frame(symbol: &str) -> PublicFrame {
        PublicFrame {
            connection: 1,
            stamp: Stamp { utc_ms: 0, mono_ns: 0 },
            stream: format!("{}@depth@100ms", symbol.to_ascii_lowercase()),
            symbol: symbol.to_owned(),
            payload: "{}".into(),
            ticker_id: None,
        }
    }

    fn anchored(now: Instant, interval: Duration, phase: Duration) -> DepthState {
        let mut state = DepthState::new(now);
        state.needs_snapshot = false;
        state.r2 = Some(R2Timer::arm(now, interval, phase));
        state
    }

    #[test]
    fn r2_interval_sale_de_la_cuenta_y_respeta_el_presupuesto() {
        // Con los numeros del roster de 40 simbolos el objetivo de 5 min cabe de sobra.
        let plan = r2_plan(Duration::from_secs(300), 40, 20, 2400, 60);
        assert_eq!(plan.interval, Duration::from_secs(300));
        let cycle = (40u64 * 20) as f64;
        let budget = 2400.0 * R2_BUDGET_FACTOR - cycle - cycle * R2_BASELINE_PER_SYMBOL_H / 60.0;
        // El ritmo R2 (peso de una vuelta completa por intervalo) no excede su presupuesto.
        assert!(cycle * 60.0 / plan.interval.as_secs() as f64 <= budget);
        // Con menos margen manda el piso de presupuesto, no el objetivo.
        let ajustado = r2_plan(Duration::from_secs(300), 40, 20, 1100, 60);
        assert_eq!(ajustado.interval, Duration::from_secs(750));
        // Sin presupuesto (limite absurdo) R2 se apaga en la practica antes que arriesgar un 429.
        let roto = r2_plan(Duration::from_secs(300), 40, 20, 300, 60);
        assert!(roto.interval.as_secs() > 3600);
        // Mas simbolos => intervalo mas largo (el presupuesto no crece con el roster).
        let grande = r2_plan(Duration::from_secs(300), 80, 20, 2400, 60);
        assert!(grande.interval > plan.interval);
    }

    #[test]
    fn r2_convierte_ventanas_y_escala_la_linea_base() {
        let max_gap = Duration::from_secs(300);
        let startup = r2_plan_from_info(&Value::Null, max_gap, 40);
        assert!(startup.defaulted);
        assert_eq!((startup.limit, startup.window_s, startup.weight), (2400, 60, 20));
        assert_eq!(startup.interval, max_gap);

        // 40 * 20 de peso por vuelta. Las bases por ventana son 8 y 46080.
        for (unit, count, limit, window_s, interval_s) in [
            ("SECOND", 30, 1060, 30, 600),
            ("DAY", 2, 59000, 172800, 432000),
        ] {
            let info = json!({
                "request_weight_limit": {"limit": limit, "interval": unit, "intervalNum": count},
                "learned_weights": {"depth": 20},
            });
            let plan = r2_plan_from_info(&info, max_gap, 40);
            assert!(!plan.defaulted);
            assert_eq!(plan.window_s, window_s);
            assert_eq!(plan.interval, Duration::from_secs(interval_s));
        }
    }

    #[test]
    fn r2_fases_escalonan_los_simbolos_sin_solaparse() {
        let interval = Duration::from_secs(300);
        let phases: Vec<_> = (0..40).map(|rank| r2_phase(rank, 40, interval).as_millis()).collect();
        let step = interval.as_millis() / 40;
        assert!(phases.iter().all(|p| *p >= step && *p <= interval.as_millis()));
        assert_eq!(phases.iter().collect::<std::collections::BTreeSet<_>>().len(), 40);
        assert_eq!(phases[0], step);
        assert_eq!(phases[39], interval.as_millis());
        assert_eq!(phases[1] - phases[0], step);
    }

    #[test]
    fn r2_periodica_dispara_solo_cuando_toca_y_se_rearma_con_exito() {
        let now = Instant::now();
        let interval = Duration::from_secs(300);
        let mut state = anchored(now, interval, Duration::ZERO);
        assert_eq!(state.snapshot_wanted(now), Some(true));
        state.pending = Some(0);
        assert_eq!(state.snapshot_wanted(now), None);
        state.reply(0, true, Duration::from_secs(5), now, Some((interval, Duration::ZERO)));
        assert_eq!(state.snapshot_wanted(now + Duration::from_secs(299)), None);
        assert_eq!(state.snapshot_wanted(now + Duration::from_secs(300)), Some(true));
    }

    #[test]
    fn r2_adopta_el_intervalo_nuevo_y_no_repite_la_fase() {
        let now = Instant::now();
        let initial = r2_plan(Duration::from_secs(300), 40, 20, 2400, 60);
        let updated = r2_plan(Duration::from_secs(300), 40, 20, 1200, 60);
        let phase = r2_phase(0, 40, initial.interval);
        let mut state = anchored(now, initial.interval, phase);
        let reply_at = now + phase;
        assert_eq!(initial.interval, Duration::from_secs(300));
        assert_eq!(updated.interval, Duration::from_secs(334));
        assert_eq!(state.snapshot_wanted(reply_at), Some(true));
        state.pending = Some(state.generation);
        let updated_phase = r2_phase(0, 40, updated.interval);
        state.reply(
            state.generation,
            true,
            Duration::ZERO,
            reply_at,
            Some((updated.interval, updated_phase)),
        );
        assert_eq!(state.snapshot_wanted(reply_at + Duration::from_secs(333)), None);
        assert_eq!(state.snapshot_wanted(reply_at + Duration::from_secs(334)), Some(true));
    }

    #[test]
    fn r2_se_arma_con_la_fase_del_simbolo_en_el_primer_exito() {
        let now = Instant::now();
        let interval = Duration::from_secs(300);
        let mut state = DepthState::new(now);
        state.pending = Some(0);
        state.reply(0, true, Duration::from_secs(5), now, Some((interval, Duration::from_secs(120))));
        assert_eq!(state.snapshot_wanted(now + Duration::from_secs(119)), None);
        assert_eq!(state.snapshot_wanted(now + Duration::from_secs(120)), Some(true));
    }

    // Prueba de ESTADO DEL SHARD (la puerta se prueba aparte, en banco_local): aqui se
    // inyecta a mano el cooldown que la puerta entrega tras un 429 real.
    #[test]
    fn r2_429_en_el_shard_retrasa_los_snapshots_sin_tocar_el_websocket() {
        let now = Instant::now();
        let interval = Duration::from_secs(300);
        let mut state = anchored(now, interval, Duration::ZERO);
        state.pending = Some(0);
        // 429: la puerta pausa (max(Retry-After, fin de ventana)) y la respuesta final
        // llega con ok=false y cooldown >= Retry-After (gate.rs throttle/serve).
        state.reply(0, false, Duration::from_secs(60), now, Some((interval, Duration::ZERO)));
        assert_eq!(state.snapshot_wanted(now + Duration::from_secs(30)), None);
        assert_eq!(state.snapshot_wanted(now + Duration::from_secs(61)), Some(false));
        // El websocket no se entera del 429: el arbitro sigue comprometiendo frames.
        let out = state.arbiter.offer(
            l2::DepthId { first: 100, last: 100, previous: 99 },
            frame("BTCUSDT"),
            0,
            now + Duration::from_secs(30),
        );
        assert_eq!(out.len(), 1);
        assert!(out[0].gap.is_none());
    }

    #[test]
    fn r2_418_detiene_los_snapshots_una_hora_sin_tocar_el_websocket() {
        let now = Instant::now();
        let interval = Duration::from_secs(300);
        let mut state = anchored(now, interval, Duration::ZERO);
        state.pending = Some(0);
        // 418: puerta cerrada; rechaza con el tope de espera del kit v1 (gate.rs
        // CLOSED_COOLDOWN = 3600 s). Un intento por hora mientras siga cerrada.
        state.reply(0, false, Duration::from_secs(3600), now, Some((interval, Duration::ZERO)));
        assert_eq!(state.snapshot_wanted(now + Duration::from_secs(3599)), None);
        assert_eq!(state.snapshot_wanted(now + Duration::from_secs(3601)), Some(false));
        // El flujo publico sigue arbitrando durante el veto REST.
        let out = state.arbiter.offer(
            l2::DepthId { first: 7, last: 7, previous: 6 },
            frame("ETHUSDT"),
            0,
            now + Duration::from_secs(3599),
        );
        assert_eq!(out.len(), 1);
        assert!(out[0].gap.is_none());
    }
}
