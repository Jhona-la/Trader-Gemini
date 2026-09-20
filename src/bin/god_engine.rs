#[global_allocator]
static GLOBAL: mimalloc::MiMalloc = mimalloc::MiMalloc;

use futures_util::StreamExt;
use quantum_engine::parsers;
use std::env;
use std::sync::Arc;
use tokio::sync::mpsc;
use tokio::time::{sleep, Duration};

use god_engine_core::orchestrator::PhaseOrchestrator;

use execution_engine::executor::ExecutionProvider;
use redb::{Database, ReadableTable, TableDefinition};
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::Instant;
const CAPITAL_TABLE: TableDefinition<u32, f64> = TableDefinition::new("capital_state");

// Removed ActiveExecutor

use quantum_engine::config::TensorConfig;
use quantum_engine::env_manager::EnvManager;
use telemetry_engine::telemetry;

/// B1.2/B1.3: precios de protección desde las CURVAS del genoma — mismo
/// criterio del camino en vivo (X-030/X-016): curva de horizonte por dos
/// puntos (bases scalp/swing del arena.config, ambas genes) evaluada en la
/// τ de ENTRADA de la posición; τ desconocida (posición adoptada tras
/// reinicio) ⇒ ancla rápida (comportamiento conservador). Sustituye los
/// ±2.5%/±1.5% hardcodeados del restore de arranque.
fn genome_protection_prices(
    arena: &quantum_arena::GlobalArena,
    symbol: &str,
    is_long: bool,
    entry_price: f64,
    entry_tau_ms: u64,
) -> (f64, f64) {
    use quantum_arena::temporal_spectrum::{
        HorizonCurve, TAU_ANCHOR_FAST_MS, TAU_ANCHOR_SLOW_MS,
    };
    // C-05 (informe decimocuarto) — CLAMP AL DOMINIO DE LAS ANCLAS: la
    // fusión espectral viva puede degenerar (journal con tau_ms =
    // 4 611 686 018 427 ≈ 146 años) y `HorizonCurve::eval` extrapola
    // EXPONENCIALMENTE fuera de banda ⇒ brackets a +65%/−32%: el
    // invariante de protección producía desnudez. La curva sólo tiene
    // validez ENTRE sus anclas: τ≤0 (adoptada sin diario) colapsa al
    // ancla rápida — igual que antes — y τ degenerada colapsa al ancla
    // lenta (máximo de la curva, jamás más allá).
    let tau_eff = (entry_tau_ms as f64).clamp(TAU_ANCHOR_FAST_MS, TAU_ANCHOR_SLOW_MS);
    let o = Ordering::Relaxed;
    let tp_frac = HorizonCurve::through_two_points(
        TAU_ANCHOR_FAST_MS,
        arena.config.scalp_tp_base.load(o),
        TAU_ANCHOR_SLOW_MS,
        arena.config.swing_tp_base.load(o),
    )
    .eval(tau_eff);
    let sl_frac = HorizonCurve::through_two_points(
        TAU_ANCHOR_FAST_MS,
        arena.config.scalp_sl_base.load(o),
        TAU_ANCHOR_SLOW_MS,
        arena.config.swing_sl_base.load(o),
    )
    .eval(tau_eff);
    // B3.2: VIABILIDAD POR FRICCIÓN como invariante de TODA protección
    // (entrada-fallback, watchdog, restore, adopción). La curva del genoma
    // decide la forma; la fricción pone el suelo: stop ≥ mínimo viable y
    // TP ≥ stop · RR_mínimo(fee). Modelo D-645: taker en ambas piernas +
    // piso de slippage por lado — el mismo que usa el gate del risk-engine.
    // B3.19 — + término de LATENCIA (atr_pct·lat/ref): el gate lo incluye y
    // los brackets no (hallazgo de auditoría: brackets ligeramente menos
    // conservadores que el gate que los aprueba).
    let latency_ref_ms = arena
        .config
        .latency_ms_panic_threshold
        .load(o)
        .clamp(10.0, 5_000.0);
    let atr_pct = quantum_arena::symbol_registry::try_index(symbol)
        .and_then(|ci| arena.coins.get(ci))
        .map(|c| {
            let px = c.current_price.load(o).max(1e-12);
            c.current_atr.load(o) / px
        })
        .filter(|a| a.is_finite() && *a > 0.0)
        .unwrap_or(0.0);
    let latency_slip = (atr_pct
        * (arena.config.latency_penalty_ms.load(o).max(0.0) / latency_ref_ms))
        .clamp(0.0, 0.05);
    let fee_rt = 2.0 * arena.config.live_taker_fee.load(o)
        + 2.0 * (arena.config.base_slippage_floor.load(o).max(0.00001) + latency_slip);
    let (sl_frac, tp_frac) =
        quantum_arena::genome::SuperGenotype::friction_floors(fee_rt, sl_frac, tp_frac);
    let tp = if is_long {
        entry_price * (1.0 + tp_frac)
    } else {
        entry_price * (1.0 - tp_frac)
    };
    let sl = if is_long {
        entry_price * (1.0 - sl_frac)
    } else {
        entry_price * (1.0 + sl_frac)
    };
    (tp, sl)
}

/// Resultado de una auditoría de cobertura (B1.3 + HOST-024).
struct ProtectionAudit {
    /// Gap TP/SL restante tras el intento de top-up (unidades del activo).
    tp_gap: f64,
    sl_gap: f64,
    /// HOST-024 (informe decimocuarto) — piernas omitidas porque el gap
    /// está por debajo del minNotional del exchange: cobertura IMPOSIBLE
    /// de colocar (no es un blip de red — el exchange JAMÁS aceptará esa
    /// orden). Evidencia determinística de riesgo real que el watchdog
    /// debe contar para el streak de escalado: la posición queda
    /// parcialmente protegida de forma permanente.
    min_notional_skips: u8,
}

/// B1.3: audita la cobertura TP/SL de una posición viva contra las algo
/// orders REALES del exchange y hace top-up QUIRÚRGICO por lado (solo el
/// lado con shortfall — un OCO completo sobre-protegería el sano y el
/// exchange rechaza por qty > posición). Devuelve el estado de la
/// cobertura tras el intento. `tag` identifica el llamador en telemetría.
async fn ensure_position_protected(
    executor: &execution_engine::executor::OrderExecutor,
    arena: &quantum_arena::GlobalArena,
    symbol: &str,
    is_long: bool,
    pos_qty: f64,
    entry_price: f64,
    entry_tau_ms: u64,
    tag: &str,
) -> ProtectionAudit {
    let legs = executor.fetch_open_algo_orders(symbol).await.unwrap_or_default();
    let want_side = if is_long { "LONG" } else { "SHORT" };
    let mut tp_covered = 0.0f64;
    let mut sl_covered = 0.0f64;
    for a in &legs {
        let side_ok = a.position_side == want_side
            || a.position_side == "BOTH"
            || a.position_side.is_empty();
        let live = matches!(
            a.algo_status.as_str(),
            "NEW" | "TRIGGERING" | "TRIGGERED"
        );
        if !side_ok || !live {
            continue;
        }
        if a.order_type.contains("TAKE_PROFIT") {
            tp_covered += a.quantity;
        } else {
            // STOP_MARKET y TRAILING_STOP_MARKET cuentan como cobertura de pérdida.
            sl_covered += a.quantity;
        }
    }
    let Ok(f) = executor.get_symbol_filter(symbol).await else {
        return ProtectionAudit {
            tp_gap: pos_qty - tp_covered,
            sl_gap: pos_qty - sl_covered,
            min_notional_skips: 0,
        };
    };
    let tp_gap = (pos_qty - tp_covered).max(0.0);
    let sl_gap = (pos_qty - sl_covered).max(0.0);
    if tp_gap < f.step_size && sl_gap < f.step_size {
        return ProtectionAudit {
            tp_gap: 0.0,
            sl_gap: 0.0,
            min_notional_skips: 0,
        }; // protegida
    }
    if entry_price <= 0.0 {
        telemetry_server::telemetry_log!(
            "⚠️ [{}] {} sin entryPrice en positionRisk — no se puede re-bracketear",
            tag,
            symbol
        );
        return ProtectionAudit {
            tp_gap,
            sl_gap,
            min_notional_skips: 0,
        };
    }
    let (tp_price, sl_price) =
        genome_protection_prices(arena, symbol, is_long, entry_price, entry_tau_ms);
    let micros = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap_or_default()
        .as_micros();
    let min_notional = f.min_notional.max(5.0);
    let mut audit = ProtectionAudit {
        tp_gap,
        sl_gap,
        min_notional_skips: 0,
    };
    // Precio VIVO del arena para el reintento -2021: cuando la posición
    // deriva más allá del nivel del genoma calculado desde el ENTRY, el
    // trigger "dispararía inmediatamente". Reposicionar la MISMA fracción
    // del genoma sobre el precio actual mantiene la semántica (distancia
    // relativa) sin hardcodes ni clamps arbitrarios.
    let live_price = quantum_arena::symbol_registry::try_index(symbol)
        .and_then(|ci| arena.coins.get(ci))
        .map(|c| c.current_price.load(Ordering::Relaxed))
        .filter(|p| *p > 0.0);
    if tp_gap >= f.step_size {
        if tp_gap * tp_price < min_notional {
            // HOST-024 — el gap es real pero el exchange jamás aceptará la
            // orden (< minNotional): se cuenta como evidencia determinística
            // para el streak del watchdog (antes sólo warning ⇒ riesgo
            // perpetuo invisible al escalado).
            audit.min_notional_skips += 1;
            telemetry_server::telemetry_log!(
                "⚠️ [{}] top-up TP {} {:.4} < minNotional — posición parcialmente protegida (cuenta para escalado)",
                tag,
                symbol,
                tp_gap * tp_price
            );
        } else {
            let mut trig = tp_price;
            let frac_tp = ((tp_price - entry_price).abs() / entry_price).max(0.0015);
            for intent in 0..2 {
                let id = format!("wdTP_{}_{}", micros, intent);
                match executor
                    .place_algo_leg(
                        symbol,
                        is_long,
                        "TAKE_PROFIT_MARKET",
                        tp_gap,
                        trig,
                        f.step_size,
                        f.tick_size,
                        &id,
                    )
                    .await
                {
                    Ok(()) => {
                        telemetry_server::telemetry_log!(
                            "🛡️ [{}] TP re-armado para {} ({:.6} @ {:.4}) — cobertura era {:.6}/{:.6}",
                            tag, symbol, tp_gap, trig, tp_covered, pos_qty
                        );
                        audit.tp_gap = 0.0;
                        break;
                    }
                    Err(e) if intent == 0 && e.contains("-2021") => {
                        match live_price {
                            Some(cur) => {
                                trig = if is_long {
                                    cur * (1.0 + frac_tp)
                                } else {
                                    cur * (1.0 - frac_tp)
                                };
                                telemetry_server::telemetry_log!(
                                    "↪️ [{}] TP de {} reposicionado al precio vivo ({:.4} → {:.4})",
                                    tag, symbol, tp_price, trig
                                );
                            }
                            None => {
                                telemetry_server::telemetry_log!(
                                    "⚠️ [{}] top-up TP {} falló (-2021 sin precio vivo): {}",
                                    tag, symbol, e
                                );
                                break;
                            }
                        }
                    }
                    Err(e) => {
                        // B3.5b: evidencia positiva de rechazo para el
                        // escalado (los fallos de red no cuentan).
                        let _ = quantum_arena::protection_health::note_rejection(symbol, &e);
                        telemetry_server::telemetry_log!(
                            "⚠️ [{}] top-up TP {} falló: {}",
                            tag, symbol, e
                        );
                        break;
                    }
                }
            }
        }
    }
    if sl_gap >= f.step_size {
        if sl_gap * sl_price < min_notional {
            // HOST-024 — ídem TP: gap bajo minNotional = riesgo real que
            // ninguna orden puede cubrir; alimenta el streak del watchdog.
            audit.min_notional_skips += 1;
            telemetry_server::telemetry_log!(
                "⚠️ [{}] top-up SL {} {:.4} < minNotional — posición parcialmente protegida (cuenta para escalado)",
                tag,
                symbol,
                sl_gap * sl_price
            );
        } else {
            let mut trig = sl_price;
            let frac_sl = ((sl_price - entry_price).abs() / entry_price).max(0.0015);
            for intent in 0..2 {
                let id = format!("wdSL_{}_{}", micros, intent);
                match executor
                    .place_algo_leg(
                        symbol,
                        is_long,
                        "STOP_MARKET",
                        sl_gap,
                        trig,
                        f.step_size,
                        f.tick_size,
                        &id,
                    )
                    .await
                {
                    Ok(()) => {
                        telemetry_server::telemetry_log!(
                            "🛡️ [{}] SL re-armado para {} ({:.6} @ {:.4}) — cobertura era {:.6}/{:.6}",
                            tag, symbol, sl_gap, trig, sl_covered, pos_qty
                        );
                        audit.sl_gap = 0.0;
                        break;
                    }
                    Err(e) if intent == 0 && e.contains("-2021") => {
                        match live_price {
                            Some(cur) => {
                                trig = if is_long {
                                    cur * (1.0 - frac_sl)
                                } else {
                                    cur * (1.0 + frac_sl)
                                };
                                telemetry_server::telemetry_log!(
                                    "↪️ [{}] SL de {} reposicionado al precio vivo ({:.4} → {:.4})",
                                    tag, symbol, sl_price, trig
                                );
                            }
                            None => {
                                telemetry_server::telemetry_log!(
                                    "⚠️ [{}] top-up SL {} falló (-2021 sin precio vivo): {}",
                                    tag, symbol, e
                                );
                                break;
                            }
                        }
                    }
                    Err(e) => {
                        // B3.5b: evidencia positiva de rechazo para el
                        // escalado (los fallos de red no cuentan).
                        let _ = quantum_arena::protection_health::note_rejection(symbol, &e);
                        telemetry_server::telemetry_log!(
                            "⚠️ [{}] top-up SL {} falló: {}",
                            tag, symbol, e
                        );
                        break;
                    }
                }
            }
        }
    }
    audit
}

/// B3.6 — CIRCUIT BREAKER POR SÍMBOLO (anti-horno de comisiones).
///
/// Criterio (medido en testnet 2026-09-15: SOL 7 trades, 0 wins, fees
/// −$3.21 vs bruto −$1.23): un símbolo cuyas FEES superan su bruto
/// ganador con neto negativo NO PUEDE PAGAR SU PROPIA FRICCIÓN — cada
/// trade lo hunde más, sin importar qué tan bien apunte el modelo. Se
/// suspenden sus NUEVAS entradas 4h (las posiciones vivas siguen con sus
/// brackets intocados). Estado en memoria: se reinicia con el motor —
/// deliberado, el breaker protege la sesión, no es un veto permanente.
static SYMBOL_SUSPENDED_UNTIL: std::sync::LazyLock<
    std::sync::Mutex<std::collections::HashMap<String, u64>>,
> = std::sync::LazyLock::new(|| std::sync::Mutex::new(std::collections::HashMap::new()));

fn suspend_symbol_until(symbol: &str, until_ms: u64) {
    SYMBOL_SUSPENDED_UNTIL
        .lock()
        .unwrap()
        .insert(symbol.to_string(), until_ms);
}

fn is_symbol_suspended(symbol: &str) -> bool {
    let now_ms = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap_or_default()
        .as_millis() as u64;
    SYMBOL_SUSPENDED_UNTIL
        .lock()
        .unwrap()
        .get(symbol)
        .is_some_and(|until| *until > now_ms)
}

/// B3.6b (auditoría) — persiste las suspensiones VIVAS en
/// data/fee_breaker.json con escritura ATÓMICA (tmp + rename): un crash a
/// media escritura dejaba JSON corrupto que la restauración del arranque
/// descartaba EN SILENCIO — una mañana mala perdonada por corrupción.
/// Las entradas expiradas se purgan: el archivo no crece sin cota.
fn persist_fee_breaker(now_ms: u64) {
    let Ok(guard) = SYMBOL_SUSPENDED_UNTIL.lock() else {
        return;
    };
    let live: Vec<(&String, &u64)> = guard.iter().filter(|(_, u)| **u > now_ms).collect();
    let file_body = live
        .iter()
        .map(|(k, v)| format!("\"{k}\":{v}"))
        .collect::<Vec<_>>()
        .join(",");
    drop(guard);
    let tmp = "data/fee_breaker.json.tmp";
    if std::fs::write(tmp, format!("{{{file_body}}}")).is_ok() {
        let _ = std::fs::rename(tmp, "data/fee_breaker.json");
    }
}

/// HOST-005 (informe decimocuarto) — PERSISTENCIA DE LA ENVOLVENTE KELLY.
/// 32 reinicios = 32 bootstraps del posterior de Jeffreys: el motor vivía
/// re-aprendiendo su edge desde cero en cada lanzamiento (n<3 ⇒ f=0 ⇒
/// sesgo de sizing sistemático de arranque). Se persisten los contadores
/// que alimentan el posterior (α/β ≡ wins/losses sobre el prior de
/// Jeffreys) y los payoffs medios; `payoff_ratio` se re-deriva al restaurar
/// (es función de los anteriores). Escritura atómica (tmp + rename).
fn persist_kelly_envelope(env: &risk_engine::kelly_envelope::RiskEnvelope) {
    let body = serde_json::json!({
        "alpha": env.posterior.alpha,
        "beta": env.posterior.beta,
        "avg_win": env.avg_win,
        "avg_loss": env.avg_loss,
    });
    let tmp = "data/kelly_envelope.json.tmp";
    if std::fs::write(tmp, body.to_string()).is_ok() {
        let _ = std::fs::rename(tmp, "data/kelly_envelope.json");
    }
}

/// HOST-005 — restaura la envolvente desde data/kelly_envelope.json.
/// Tolerante a archivo ausente/corrupto/incompleto (false ⇒ el llamador
/// continúa con el posterior fresco de Jeffreys — degradación, no pánico).
fn restore_kelly_envelope(env: &mut risk_engine::kelly_envelope::RiskEnvelope) -> bool {
    let Ok(content) = std::fs::read_to_string("data/kelly_envelope.json") else {
        return false; // primera ejecución — nada que restaurar
    };
    let Ok(v) = serde_json::from_str::<serde_json::Value>(&content) else {
        telemetry_server::telemetry_log!(
            "⚠️ [KELLY] data/kelly_envelope.json corrupto — envolvente fresca"
        );
        return false;
    };
    let g = |k: &str| v.get(k).and_then(|x| x.as_f64());
    let (Some(alpha), Some(beta), Some(avg_win), Some(avg_loss)) =
        (g("alpha"), g("beta"), g("avg_win"), g("avg_loss"))
    else {
        telemetry_server::telemetry_log!(
            "⚠️ [KELLY] data/kelly_envelope.json incompleto — envolvente fresca"
        );
        return false;
    };
    // Saneo: el posterior jamás baja del prior de Jeffreys (0.5/0.5) y los
    // payoffs son magnitudes finitas no negativas — un archivo manipulado
    // o de otra era no inyecta un edge fantasma.
    if !alpha.is_finite()
        || !beta.is_finite()
        || alpha < 0.5
        || beta < 0.5
        || !avg_win.is_finite()
        || !avg_loss.is_finite()
        || avg_win < 0.0
        || avg_loss < 0.0
    {
        telemetry_server::telemetry_log!(
            "⚠️ [KELLY] data/kelly_envelope.json con valores inválidos — envolvente fresca"
        );
        return false;
    }
    env.posterior.alpha = alpha;
    env.posterior.beta = beta;
    env.avg_win = avg_win;
    env.avg_loss = avg_loss;
    env.payoff_ratio = (avg_win + 1e-3) / (avg_loss + 1e-3);
    true
}

/// HOST-016 (informe decimocuarto) — UN SOLO sincronizador NTP vivo.
/// El spawn de la transición demo→mainnet dejaba DOS loops NTP infinitos
/// compitiendo por `server_time_offset_ms` (el offset oscilaba cada 15s
/// según qué loop ganara el fetch). Generación monótona: al lanzar un
/// sincronizador nuevo, todos los anteriores observan su generación
/// obsoleta en su guardia y salen — el equivalente al
/// `if ntp_abort.load() { break; }` del loop, implementado como guardia
/// en el host porque el loop vive en execution-engine::ntp.
static NTP_SYNC_GENERATION: AtomicU64 = AtomicU64::new(0);

fn spawn_ntp_synchronizer(
    handle: &tokio::runtime::Handle,
    client: Arc<execution_engine::client::BinanceClient>,
    arena: Arc<quantum_arena::GlobalArena>,
) {
    let my_gen = NTP_SYNC_GENERATION.fetch_add(1, Ordering::AcqRel);
    if my_gen > 0 {
        telemetry_server::telemetry_log!(
            "🔁 [NTP-SYNC] Generación {} — sincronizadores anteriores abortados (HOST-016: un solo loop vivo)",
            my_gen + 1
        );
    }
    handle.spawn(async move {
        let sync = execution_engine::ntp::start_ntp_synchronizer(client, arena);
        let superseded = async {
            loop {
                if NTP_SYNC_GENERATION.load(Ordering::Acquire) > my_gen {
                    break;
                }
                tokio::time::sleep(std::time::Duration::from_millis(500)).await;
            }
        };
        tokio::select! {
            _ = sync => {}
            _ = superseded => {
                telemetry_server::telemetry_log!(
                    "🛑 [NTP-SYNC] Sincronizador de generación {} superseded — loop detenido (HOST-016)",
                    my_gen + 1
                );
            }
        }
    });
}

/// Contexto recuperado de una posición desde el diario de entradas.
struct RecoveredContext {
    tau_ms: u64,
    ml: f64,
    age_hours: f64,
}

/// B2.7: lee data/position_journal.jsonl y devuelve el ÚLTIMO registro que
/// matchea símbolo+lado — la τ espectral y la predicción ML que motivaron la
/// entrada. Tolerante a diario ausente/corrupto (None ⇒ el llamador usa el
/// piso espectral conservador).
fn recover_position_context(symbol: &str, is_long: bool) -> Option<RecoveredContext> {
    let content = std::fs::read_to_string("data/position_journal.jsonl").ok()?;
    let now_ms = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap_or_default()
        .as_millis() as u64;
    let mut best: Option<(u64, u64, f64)> = None; // (ts, tau, ml)
    for line in content.lines() {
        let Ok(v) = serde_json::from_str::<serde_json::Value>(line) else {
            continue;
        };
        let sym = v.get("sym").and_then(|x| x.as_str()).unwrap_or("");
        let long = v.get("long").and_then(|x| x.as_bool()).unwrap_or(false);
        if sym != symbol || long != is_long {
            continue;
        }
        let ts = v.get("ts").and_then(|x| x.as_u64()).unwrap_or(0);
        if best.map(|(b, _, _)| ts >= b).unwrap_or(true) {
            best = Some((
                ts,
                v.get("tau_ms").and_then(|x| x.as_u64()).unwrap_or(0),
                v.get("ml").and_then(|x| x.as_f64()).unwrap_or(0.5),
            ));
        }
    }
    best.map(|(ts, tau_ms, ml)| RecoveredContext {
        tau_ms,
        ml,
        age_hours: now_ms.saturating_sub(ts) as f64 / 3_600_000.0,
    })
}

/// B3.8 — compacta data/position_journal.jsonl al ÚLTIMO registro por
/// (símbolo, lado) — lo único que `recover_position_context` consulta. Sin
/// esto el diario crece sin cota y cada reconexión re-lee todo el historial.
/// Umbral conservador: compactar sólo cuando supera 500 líneas. Escritura
/// atómica (tmp + rename); líneas corruptas se descartan en la compactación
/// (la lectura ya era tolerante línea a línea).
/// B3.19 — CIERRE DE EMERGENCIA a la contabilidad: los reduce-only de
/// emergencia (naked-escalation B3.5, X-009 del OCO de entrada) cerraban
/// posiciones reales que ni el core ni B3.7 veían — invisibles al diario y
/// al posterior de Kelly (sólo el income del fee-breaker las registraba).
/// El precio de salida es el vivo del arena (aproximación documentada: el
/// market reduce-only ejecuta ~ahora); la entrada sale del contexto local,
/// que aún vive al momento de la llamada.
fn record_emergency_close(
    symbol: &str,
    was_long: bool,
    qty: f64,
    arena: &Arc<quantum_arena::GlobalArena>,
    trigger: &'static str,
) {
    let ci = quantum_arena::symbol_registry::try_index(symbol);
    let (entry_price, entry_fee, exit_price) = ci
        .and_then(|i| arena.coins.get(i))
        .map(|c| {
            let p = &c.positions.position;
            (
                p.entry_price.load(Ordering::Relaxed),
                p.entry_fee.load(Ordering::Relaxed),
                c.current_price.load(Ordering::Relaxed),
            )
        })
        .unwrap_or((0.0, 0.0, 0.0));
    let sign = if was_long { 1.0 } else { -1.0 };
    let pnl_gross = if entry_price > 0.0 && exit_price > 0.0 {
        (entry_price - exit_price) * qty * sign
    } else {
        0.0
    };
    execution_engine::trade_accounting::record_bracket_close(
        execution_engine::trade_accounting::BracketClose {
            ts_ms: std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap_or_default()
                .as_millis() as u64,
            symbol: symbol.to_string(),
            was_long,
            qty,
            entry_price,
            exit_price,
            stop_price: 0.0,
            pnl_gross,
            fees: entry_fee,
            trigger,
            slippage_bps: 0.0,
        },
    );
}

fn compact_position_journal() {
    const COMPACT_THRESHOLD: usize = 500;
    let path = "data/position_journal.jsonl";
    let Ok(content) = std::fs::read_to_string(path) else {
        return; // diario ausente: primera ejecución — nada que compactar
    };
    let lines: Vec<&str> = content.lines().filter(|l| !l.trim().is_empty()).collect();
    if lines.len() <= COMPACT_THRESHOLD {
        return;
    }
    let mut latest: std::collections::HashMap<(String, bool), (u64, &str)> =
        std::collections::HashMap::new();
    let mut discarded = 0usize;
    for line in &lines {
        let Ok(v) = serde_json::from_str::<serde_json::Value>(line) else {
            discarded += 1;
            continue;
        };
        let key = (
            v.get("sym").and_then(|x| x.as_str()).unwrap_or("").to_string(),
            v.get("long").and_then(|x| x.as_bool()).unwrap_or(false),
        );
        let ts = v.get("ts").and_then(|x| x.as_u64()).unwrap_or(0);
        if latest.get(&key).map(|(b, _)| ts >= *b).unwrap_or(true) {
            latest.insert(key, (ts, line));
        }
    }
    let tmp = "data/position_journal.jsonl.tmp";
    let Ok(mut f) = std::fs::File::create(tmp) else {
        return;
    };
    use std::io::Write;
    let mut kept = 0usize;
    // Orden por ts: el diario queda determinista y legible.
    let mut ordered: Vec<_> = latest.values().collect();
    ordered.sort_by_key(|(ts, _)| *ts);
    for (_, line) in ordered {
        if writeln!(f, "{line}").is_ok() {
            kept += 1;
        }
    }
    drop(f);
    if std::fs::rename(tmp, path).is_ok() {
        telemetry_server::telemetry_log!(
            "🧹 [DIARIO] compactado: {} → {} registros ({} corruptos descartados) — rotación B3.8",
            lines.len(),
            kept,
            discarded
        );
    }
}

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    // 🛠️ [BOOTLOADER] Inicializar el entorno desde .env
    dotenvy::dotenv().ok();

    // CERT-M4-H02 — SHUTDOWN GRACEFUL: Registrar un atexit handler que
    // mata el trainer hijo. En Windows, Ctrl+C pasa por el console handler
    // del OS que termina el proceso — este thread daemon detecta que el
    // padre está saliendo (cerrando) y limpia antes de morir.
    std::thread::Builder::new()
        .name("shutdown-cleanup".into())
        .spawn(|| {
            // Dormir indefinidamente: este thread muere cuando el proceso
            // termina (panic=abort o Ctrl+C). El trainer hijo se limpia
            // porque el supervisor thread lo espera (M4-H01) y al morir
            // el padre, Windows termina los hijos del JobObject.
            loop {
                std::thread::sleep(std::time::Duration::from_secs(3600));
            }
        })
        .ok();

    // os_guardian init happens inside init_guardian or similar, we just use the module's time_critical if needed
    // FASE 12: Zero-Latency Telemetry Engine
    telemetry_engine::init_telemetry(100_000);
    std::thread::spawn(|| {
        loop {
            if let Some(msg) = telemetry_engine::pop_telemetry() {
                telemetry_server::telemetry_log!("{}", msg); // Asynchronous non-blocking console write
            } else {
                std::thread::sleep(std::time::Duration::from_millis(1));
            }
        }
    });

    telemetry_engine::telemetry!("========================================================");
    telemetry_server::telemetry_log!(
        "🚀 GOD ENGINE - NATIVE RUST ORCHESTRATOR (UNIFIED SCALP/SWING)"
    );
    telemetry_server::telemetry_log!("⚡ Sub-Microsecond Execution Core initialized.");
    telemetry_server::telemetry_log!("🛡️ OS Guardian active: strict resource isolation.");
    telemetry_server::telemetry_log!("========================================================");

    // HOST-012 (INFORME DECIMOCUARTO): run_god_engine.ps1 inyecta
    // BINANCE_API_KEY="shadow" ANTES de arrancar; dotenvy NO sobreescribe
    // variables ya presentes en el proceso, así que la clave "shadow"
    // TIENE PRIORIDAD sobre el .env real. Identidad demo: si se detecta,
    // telemetría roja — jamás debe llegar a mainnet (para producción:
    // LAUNCH_PRODUCTION.bat sin ese script).
    if env::var("BINANCE_API_KEY").unwrap_or_default() == "shadow" {
        eprintln!("\x1b[31m🔴 [HOST-012][SHADOW-KEY] BINANCE_API_KEY=='shadow' detectada tras dotenvy — CLAVE 'shadow' CON PRIORIDAD SOBRE .env (identidad demo inyectada por run_god_engine.ps1). ESTE ENTORNO NO ES PRODUCCIÓN. NUNCA usar para mainnet: usar LAUNCH_PRODUCTION.bat sin ese script.\x1b[0m");
        telemetry_server::telemetry_log!(
            "🔴 [HOST-012][SHADOW-KEY] BINANCE_API_KEY=='shadow' tiene prioridad sobre .env — entorno DEMO inyectado por run_god_engine.ps1. NUNCA usar para mainnet: usar LAUNCH_PRODUCTION.bat sin ese script."
        );
    }

    telemetry_server::telemetry_log!("\n========================================================");
    let args: Vec<String> = env::args().collect();
    // FASE 28 + F5.2: El calentamiento cuántico es obligatorio. Siempre inicia en
    // Demo/Testnet salvo --force-live CON armado humano explícito.
    //
    // PUERTA HUMANA DE MAINNET (Plan Maestro 7.4): --force-live solo activa
    // producción si EXISTE el archivo físico `config_dir/MAINNET_ARMED`
    // (creado a mano por el operador tras la certificación demo) y NO existe
    // STOP_TRADING.LOCK. Un flag de CLI jamás volverá a ser suficiente para
    // tocar capital real — el audit halló mainnet alcanzable tras 120 ticks.
    let requested_live = args.contains(&"--force-live".to_string());
    // H-2: MAINNET_ARMED con path centralizado (antes relativo al CWD —
    // lanzar desde otro directorio degradaba silenciosamente a mainnet-sin-lock)
    let mainnet_armed = std::path::Path::new(
        &quantum_arena::paths::data_join("../../config_dir/MAINNET_ARMED")
            .replace("data/../../", ""),
    )
    .exists()
        || std::path::Path::new("config_dir/MAINNET_ARMED").exists();
    let emergency_lock = operator_lock_exists();
    let is_demo_mode = if requested_live && mainnet_armed && !emergency_lock {
        telemetry_server::telemetry_log!(
            "🔴 [MODO PRODUCCION ARMADO] MAINNET_ARMED presente — fuego real AUTORIZADO por el operador."
        );
        false
    } else if requested_live {
        if emergency_lock {
            telemetry_server::telemetry_log!(
                "🛑 [PUERTA MAINNET] STOP_TRADING.LOCK activo — --force-live IGNORADO. Modo demo."
            );
        } else {
            telemetry_server::telemetry_log!(
                "🛑 [PUERTA MAINNET] --force-live sin config_dir/MAINNET_ARMED — capital real PROHIBIDO hasta certificación (Plan 7.4). Cayendo a DEMO."
            );
        }
        true
    } else {
        true
    };

    // E3 — ENTORNO DE GENOMA por modo de operación: demo y producción
    // mantienen linajes SEPARADOS (y separados del backtest). Sin esto, el
    // active.json compartido cruzaba overfits de backtest hacia dinero real.
    unsafe {
        std::env::set_var("TG_GENOME_ENV", if is_demo_mode { "demo" } else { "prod" });
    }

    let darwin_approved = Arc::new(std::sync::atomic::AtomicBool::new(false));
    let orchestrator = Arc::new(std::sync::RwLock::new(PhaseOrchestrator::new(
        120,
        is_demo_mode,
        Arc::clone(&darwin_approved),
    )));

    if is_demo_mode {
        telemetry_server::telemetry_log!(
            "🔥 [MODO WARMUP/DEMO ACTIVO] El sistema inicia en Paper Trading para simulación."
        );
    } else {
        telemetry_server::telemetry_log!(
            "🔴 [MODO PRODUCCION ACTIVO] El sistema inicia en MAINNET con fuego real."
        );
    }

    // FASE 38: Gestión de Memoria Estricta para Host de 16GB (Cero Swapping)
    // Asignamos 6144 MB (6GB) como límite de JobObject y VirtualLock.
    // Esto garantiza 10GB libres para Windows, erradicando por completo el Page Swap.
    // (Movido a la inicialización de arena_real)

    // ==========================================
    // FASE 1 Y 2 (BOOTLOADER CUÁNTICO INTEGRADO)
    // ==========================================
    let (_ntp_offset_ms, symbols) = god_engine_core::bootloader::SystemDiagnostics::execute_phase_1_and_2(false).await.unwrap_or_else(|e| {
        panic!("⚠️ [CRITICAL] Fallo en la Secuencia de Lanzamiento: {}. El sistema no puede operar a ciegas sin datos de red o símbolos. Abortando.", e);
    });

    quantum_arena::symbols::update_dynamic_universe(symbols.clone());

    // B3.38 — IDENTIDAD coin_id↔símbolo. El registry arranca VACÍO
    // (get_default_specs = []) y nada lo poblaba con la lista del
    // bootloader: el primer update_registry en correr era el del daemon
    // (pool de momentum del escáner), cuyo ORDEN no tiene nada que ver con
    // `symbols_clone`/`symbol_to_id` del handler. Resultado medido en vivo
    // (v41-v46): el slot 2 streameaba bnbusdt pero el core creía que era
    // el 3er símbolo del pool — coin_model_key equivocado ⇒ forest jamás
    // encontrado ⇒ ml_prob clavado en 0.5000 y el gate B3.25 vetando TODO
    // con modelos sanos (el NN BTC-only y el lead-lag BTC/ETH también
    // leían identidades falsas). v40 funcionó por ALINEACIÓN LUCKY del
    // pool con la lista del bootloader. FIX: registrar la lista del
    // bootloader COMO LAS PRIMERAS ENTRADAS del registry, EN ORDEN — el
    // merge append-only (FIX #905) del daemon respeta esas posiciones.
    let boot_specs: Vec<quantum_arena::symbol_registry::SymbolSpec> = symbols
        .iter()
        .map(|s| quantum_arena::symbol_registry::get_official_binance_spec(&s.to_uppercase()))
        .collect();
    quantum_arena::symbol_registry::update_registry(boot_specs);
    telemetry_server::telemetry_log!(
        "🧬 [B3.38] Identidad registrada: {} símbolos del bootloader en orden de slot (core≡handler)",
        symbols.len()
    );

    let dynamic_config_bin_path = EnvManager::data_path("dynamic_config.bin");
    let dynamic_config_json_path = EnvManager::data_path("dynamic_config.json");
    let config_bytes = std::fs::read(&dynamic_config_bin_path).unwrap_or_default();
    let mut config: TensorConfig = bincode::deserialize(&config_bytes).unwrap_or_else(|_| {
        let config_str =
            std::fs::read_to_string(&dynamic_config_json_path).unwrap_or_else(|_| "".to_string());
        let cfg: TensorConfig =
            serde_json::from_str(&config_str).unwrap_or_else(|_| TensorConfig {
                symbols: symbols.clone(),
                is_testnet: false, // We always use mainnet data, paper trading intercepts execution
            });
        if let Ok(encoded) = bincode::serialize(&cfg) {
            let _ = std::fs::write(&dynamic_config_bin_path, encoded);
        }
        cfg
    });
    config.symbols = symbols.clone(); // Update config memory

    let (telemetry_tx, _) = tokio::sync::broadcast::channel(100);

    // let dashboard_tx = telemetry_tx.clone(); // Removed unused variable
    // [FASE 23] Deprecated old dashboard in favor of zero-copy telemetry_server
    /*
    tokio::spawn(async move {
        if let Err(e) = quantum_engine::dashboard::start_server(dashboard_tx).await {
            telemetry_server::telemetry_log!("❌ [DASHBOARD] Failed to start server: {}", e);
        }
    });
    */

    // Telemetry Async Formatter (Lock-Free Offload)
    // Telemetry Async Formatter (Lock-Free Offload)
    let (tx_log_worker, rx_log_worker) = crossbeam_channel::bounded::<(bool, bool, usize)>(1000);
    let dash_tx_clone = telemetry_tx.clone();
    let symbols_for_log = symbols.clone();
    std::thread::spawn(move || {
        while let Ok((is_scalp, is_long, coin_id)) = rx_log_worker.recv() {
            let side_str = if is_long { "LONG" } else { "SHORT" };
            // FIX #1518: Acceso seguro al símbolo por coin_id para evitar pánicos fuera de límites
            let default_sym = format!("COIN_{}", coin_id);
            let parsed_sym = symbols_for_log
                .get(coin_id)
                .map(|s| s.as_str())
                .unwrap_or(&default_sym);
            if is_scalp {
                let _ = dash_tx_clone.send(telemetry_server::TelemetryEvent::LogUpdate(
                    "success".to_string(),
                    format!("⚡ SCALP {} on {}", side_str, parsed_sym),
                ));
            } else {
                let _ = dash_tx_clone.send(telemetry_server::TelemetryEvent::LogUpdate(
                    "success".to_string(),
                    format!("🚀 SWING {} on {}", side_str, parsed_sym),
                ));
            }
        }
    });


    let mut streams = String::new();
    for (i, sym) in symbols.iter().enumerate() {
        // Flujo pesado para TODO el universo dinámico (30 activos), permitiendo paralelismo masivo
        streams.push_str(sym);
        streams.push_str("@trade/");
        streams.push_str(sym);
        streams.push_str("@depth5/");
        streams.push_str(sym);
        // D-705 (DÉCIMA OLA · auditoría integral): EL RELOJ DE CALIBRACIÓN ES EL
        // MISMO EN CALENTAMIENTO, VALIDACIÓN Y VIVO.
        //
        // La vela cerrada es, por diseño del núcleo, el reloj que evalúa las
        // predicciones del ensamble contra la dirección realizada del bar
        // (`update_with_outcome`, F4.7) y el que mueve las EMAs de kline del
        // escudo macro. El calentamiento las llena con velas de 1 MINUTO
        // (bootloader: `interval=1m`) y el forense marca frontera de vela cada
        // minuto, pero en vivo se suscribía `@kline_1h`: 24 muestras diarias por
        // símbolo, con los pesos del ensamble en su valor inicial [0,5; 0,5]
        // durante la primera hora de cada arranque —y los arranques son
        // frecuentes—. Desde B3.18 ese ensamble decide TODAS las entradas, así
        // que su calibración no puede ir 60 veces más lenta que la decisión, ni
        // el motor vivo puede alimentar con velas horarias unas EMAs calentadas
        // con velas de un minuto.
        streams.push_str("@kline_1m");

        if i < symbols.len() - 1 {
            streams.push('/');
        }
    }
    // P-4 (PREDICTORES): stream de LIQUIDACIONES de todo el mercado. Cada
    // forceOrder alimenta liquidation_feed::bump → dex_severity viva en el
    // tensor dark_alpha (decae con la constante genómica) + omni[10].
    streams.push_str("/!forceOrder@arr");
    let streams_str = streams.clone();
    let initial_ws_host = if let Ok(ep) = env::var("BEST_WS_ENDPOINT") {
        ep
    } else {
        let is_env_testnet = env::var("USE_TESTNET")
            .unwrap_or_default()
            .trim()
            .to_lowercase()
            == "true";
        if is_env_testnet {
            "stream.binancefuture.com".to_string()
        } else {
            let endpoints = vec![
                "fstream.binance.com",
                "fstream-auth.binance.com",
                "dstream.binance.com",
            ];
            let mut best_host = "fstream.binance.com".to_string();
            let mut best_latency = u128::MAX;

            let mut best_ip = String::new();
            for ep in endpoints {
                if let Ok(addrs) = tokio::net::lookup_host((ep, 443)).await {
                    for addr in addrs {
                        let start = tokio::time::Instant::now();
                        if let Ok(_) = tokio::time::timeout(
                            tokio::time::Duration::from_millis(500),
                            tokio::net::TcpStream::connect(addr),
                        )
                        .await
                        {
                            let latency = start.elapsed().as_millis();
                            if latency < best_latency {
                                best_latency = latency;
                                best_host = ep.to_string();
                                best_ip = addr.ip().to_string();
                            }
                        }
                    }
                }
            }
            if !best_ip.is_empty() {
                telemetry_server::telemetry_log!(
                    "🚀 [LATENCY ACCELERATOR] Selected WS IP {} (Host: {}) with {}ms latency",
                    best_ip,
                    best_host,
                    best_latency
                );
                best_host
            } else {
                best_host
            }
        }
    };
    let ws_url = Arc::new(arc_swap::ArcSwap::from_pointee(format!(
        "wss://{}/stream?streams={}",
        initial_ws_host, streams_str
    )));

    let (tx_ws_control, mut rx_ws_control) = tokio::sync::mpsc::channel::<()>(1);

    // Spawn Dynamic Symbol Manager (Top 10 Evolver)
    // QO-U1b — UNIVERSO FANTASMA: el WS jamás re-suscribía al rotar el
    // universo — los símbolos rotados entraban al arena/registry pero
    // NUNCA recibían ticks (streams congelados al arranque). Ahora el
    // daemon recibe el ArcSwap de la URL del WS y el canal de control:
    // al rotar, re-construye la lista de streams y fuerza re-conexión.
    {
        let ws_url_for_sm = Arc::clone(&ws_url);
        let ws_ctrl_for_sm = tx_ws_control.clone();
        tokio::spawn(async move {
            quantum_engine::symbol_manager::evolve_symbols_daemon_with_resubscribe(
                ws_url_for_sm,
                ws_ctrl_for_sm,
            )
            .await;
        });
    }

    let (tx_events, rx_events) = crossbeam_channel::bounded::<Vec<u8>>(5_000);
    let rx_events_dropper = rx_events.clone();
    // CERT-M4-H02 — SHUTDOWN GRACEFUL: ^C #1 despierta el event loop con un
    // centinela para que drene ordenado (flatten-all + persistir estado);
    // ^C #2 fuerza la salida. Antes: un Ctrl+C con posiciones abiertas
    // mataba el proceso y la posición quedaba huérfana hasta la
    // reconciliación del siguiente arranque.
    let shutdown_requested = Arc::new(std::sync::atomic::AtomicBool::new(false));
    {
        let flag = Arc::clone(&shutdown_requested);
        let wake = tx_events.clone();
        tokio::spawn(async move {
            loop {
                if tokio::signal::ctrl_c().await.is_ok() {
                    if flag.swap(true, std::sync::atomic::Ordering::SeqCst) {
                        telemetry_server::telemetry_log!(
                            "🛑 [SHUTDOWN] Segundo ^C — salida forzada."
                        );
                        std::process::exit(130);
                    }
                    telemetry_server::telemetry_log!(
                        "🛑 [SHUTDOWN] ^C recibido — drenando posiciones y estado..."
                    );
                    let _ = wake.try_send(vec![0xFF]); // despierta el event loop
                }
            }
        });
    }
    // CERT-M1-H01: contador de eventos descartados por backpressure —
    // antes los ticks se perdían silenciosamente sin evidencia forense.
    let dropped_events = Arc::new(std::sync::atomic::AtomicU64::new(0));
    {
        let dropped_events = Arc::clone(&dropped_events);
        std::thread::Builder::new().name("ws-backpressure-monitor".into()).spawn(move || {
            loop {
                std::thread::sleep(std::time::Duration::from_secs(60));
                let d = dropped_events.load(std::sync::atomic::Ordering::Relaxed);
                if d > 0 {
                    println!("⚠️ [BACKPRESSURE] {} eventos WS descartados (drop-oldest) — throughput del lector insuficiente", d);
                }
            }
        }).ok();
    }

    // FASE 3A: FETCH CLAVES Y CONEXIÓN API REST PARA CHEQUEO DE COMISIONES Y CAPITAL ANTES DEL WARMUP Y ENTRENAMIENTO
    let mut testnet_key = env::var("TESTNET_API_KEY").unwrap_or_default();
    let mut testnet_secret = env::var("TESTNET_SECRET_KEY").unwrap_or_default();
    let mut mainnet_key = env::var("MAINNET_API_KEY").unwrap_or_default();
    let mut mainnet_secret = env::var("MAINNET_SECRET_KEY").unwrap_or_default();

    // Auto-map TESTNET keys
    if testnet_key.is_empty() {
        testnet_key = env::var("BINANCE_TESTNET_API_KEY").unwrap_or_default();
        testnet_secret = env::var("BINANCE_TESTNET_SECRET_KEY").unwrap_or_default();
    }
    // Auto-map MAINNET keys from .env
    if mainnet_key.is_empty() {
        mainnet_key = env::var("BINANCE_API_KEY").unwrap_or_default();
        mainnet_secret = env::var("BINANCE_SECRET_KEY").unwrap_or_default();
    }

    if testnet_key.is_empty() || mainnet_key.is_empty() {
        telemetry_server::telemetry_log!("⚠️ [WARNING] TESTNET_API_KEY or MAINNET_API_KEY not found. Ensure both are set for seamless transition.");
    }

    let is_env_testnet = env::var("USE_TESTNET")
        .unwrap_or_default()
        .trim()
        .to_lowercase()
        == "true";
    let (active_key, active_secret, is_testnet) = if is_env_testnet {
        telemetry_server::telemetry_log!(
            "🌐 [ENV DETECTED] Using BINANCE TESTNET for API connections."
        );
        (testnet_key.clone(), testnet_secret.clone(), true)
    } else {
        telemetry_server::telemetry_log!(
            "🌐 [ENV DETECTED] Using BINANCE MAINNET for API connections."
        );
        (mainnet_key.clone(), mainnet_secret.clone(), false)
    };

    let mut order_executor =
        execution_engine::executor::OrderExecutor::new(active_key, active_secret, is_testnet);
    // D-700 (DÉCIMA OLA · auditoría integral): LA PUERTA HUMANA GOBIERNA EL
    // FUEGO REAL DESDE EL PRIMER EJECUTOR.
    //
    // `is_paper_trading` nace en `false` (executor.rs), y aquí la única línea que
    // lo tocaba era `if is_env_testnet { set_paper_trading(false) }`, que no hace
    // nada porque ya vale false. Resultado: arrancar SIN `--force-live`, o con él
    // pero sin `config_dir/MAINNET_ARMED`, imprimía «MODO DEMO» y sin embargo el
    // ejecutor inicial disparaba contra la cuenta REAL con las claves de mainnet
    // —`ensure_hedge_mode` cambia el modo de posición de la cuenta, y las rutas de
    // restauración y protección colocan órdenes— hasta la transición de fase.
    //
    // El discriminador correcto no es el entorno (testnet o mainnet), que es
    // ortogonal, sino la MISMA puerta de capital que decide `is_demo_mode`: en
    // testnet se opera nativamente contra su API; en mainnet sin armado humano,
    // papel local.
    order_executor.set_paper_trading(is_demo_mode && !is_env_testnet);
    if is_demo_mode && !is_env_testnet {
        telemetry_server::telemetry_log!(
            "🧻 [PUERTA MAINNET] Ejecutor en PAPEL local: sin MAINNET_ARMED no sale ninguna orden a la cuenta real."
        );
    }
    let exec = Arc::new(arc_swap::ArcSwap::from_pointee(order_executor));

    let initial_capital = loop {
        match exec.load().fetch_account_balance().await {
            Ok(bal) => {
                telemetry_server::telemetry_log!(
                    "🌍 [OMNI-AWARENESS] API Real Balance Extracted: ${:.4}",
                    bal
                );
                break bal;
            }
            Err(e) => {
                telemetry_server::telemetry_log!("⚠️ [CRITICAL] Failed to fetch balance from API (Testnet/Mainnet): {}. Retrying in 5s to enforce 100% Reality Parity...", e);
                tokio::time::sleep(tokio::time::Duration::from_secs(5)).await;
            }
        }
    };

    // Inject Dynamic Capital into Institutional Telemetry Projections
    audit_engine::telemetry::update_dynamic_capital(initial_capital);

    let (live_maker, live_taker) = if let Some(first_sym) = symbols.first() {
        loop {
            match exec.load().fetch_commission_rate(first_sym).await {
                Ok((m, t)) => {
                    telemetry_server::telemetry_log!("🌍 [OMNI-AWARENESS] API Real VIP Fees Extracted: Maker {:.4}%, Taker {:.4}%", m * 100.0, t * 100.0);
                    // B3.11 — PARIDAD BT/LIVE de fees: el simulador de
                    // backtest lee este archivo para cobrar las MISMAS
                    // comisiones que la cuenta viva (antes hardcodeaba VIP0
                    // 0.05% taker; esta cuenta paga 0.04% — los backtests
                    // sobreestimaban fricción ~20%).
                    let now_ms = std::time::SystemTime::now()
                        .duration_since(std::time::UNIX_EPOCH)
                        .unwrap_or_default()
                        .as_millis() as u64;
                    let _ = std::fs::write(
                        "data/live_fees.json",
                        format!("{{\"maker\":{m},\"taker\":{t},\"ts\":{now_ms}}}"),
                    );
                    break (m, t);
                }
                Err(e) => {
                    telemetry_server::telemetry_log!("⚠️ [CRITICAL] Failed to fetch fees from API: {}. Retrying in 5s to enforce 100% Reality Parity...", e);
                    tokio::time::sleep(tokio::time::Duration::from_secs(5)).await;
                }
            }
        }
    } else {
        panic!("⚠️ [CRITICAL] No symbols defined. Cannot extract fees.");
    };

    let initial_genome =
        quantum_arena::genome::SuperGenotype::load_or_baseline(live_maker, live_taker);
    telemetry_server::telemetry_log!(
        "🧬 [INIT] Genesis Genome Loaded. DarwinDaemon is ready to evolve."
    );

    let historical_klines = god_engine_core::bootloader::SystemDiagnostics::execute_phase_3_warmup(
        is_demo_mode,
        &symbols,
    )
    .await;

    let global_learning_rate = 0.001 * (1.0 + initial_genome.quantum_mutation_rate);
    let training_epochs = (10.0 * (1.0 + initial_genome.quantum_mutation_rate)).max(5.0) as usize;

    let _swing_nn = god_engine_core::bootloader::SystemDiagnostics::execute_phase_4_training(
        &symbols,
        &historical_klines,
        global_learning_rate,
        training_epochs,
    )
    .await;

    let dark_router = Arc::new(quantum_engine::dark_alpha_router::DarkAlphaRouter::new());

    // Spawn Hyperliquid DEX cascade sniffer
    quantum_engine::dark_alpha_sniffer::spawn_hyperliquid_sniffer(Arc::clone(&dark_router));

    // El genoma inicial fue validado. Aprobamos la transición al Orchestrator.
    darwin_approved.store(true, Ordering::Relaxed);

    let mut forest_timestamps: std::collections::HashMap<String, std::time::SystemTime> =
        std::collections::HashMap::new();

    // Initial load of all models (.json es la fuente de verdad; el .bin es
    // caché del propio loader — B3.19: cargar AMBOS por stem hacía doble
    // trabajo y dejaba el resultado al orden de read_dir)
    if let Ok(entries) = std::fs::read_dir("models") {
        for entry in entries.filter_map(|e| e.ok()) {
            let path = entry.path();
            let ext = path.extension().and_then(|s| s.to_str());
            // Un .bin sólo se carga si su .json hermano NO existe (stem
            // legacy sin json); con .json presente, el loader deriva la
            // pareja y regenera la caché él mismo.
            let json_hermano = path.with_extension("json");
            let cargable = ext == Some("json")
                || (ext == Some("bin") && !json_hermano.exists());
            if cargable {
                if let Some(file_stem) = path.file_stem().and_then(|s| s.to_str()) {
                    if let Ok(meta) = std::fs::metadata(&path) {
                        if let Ok(modified) = meta.modified() {
                            forest_timestamps.insert(file_stem.to_string(), modified);
                            let _ = god_engine_core::ml_inference::NanoForest::load_global(
                                file_stem,
                                path.to_str().unwrap(),
                            );
                            telemetry_server::telemetry_log!("🧠 Loaded ML Model: {}", file_stem);
                        }
                    }
                }
            }
        }
    }

    tokio::spawn(async move {
        telemetry_server::telemetry_log!(
            "👀 [HOT-RELOAD] Watcher started. Monitoring DNA and ML Weights..."
        );
        let dynamic_config_json_path = EnvManager::data_path("dynamic_config.json");
        let mut last_config_ts = std::fs::metadata(&dynamic_config_json_path)
            .and_then(|m| m.modified())
            .ok();

        loop {
            sleep(Duration::from_secs(10)).await;

            if let Ok(meta) = std::fs::metadata(&dynamic_config_json_path) {
                if let Ok(modified) = meta.modified() {
                    if Some(modified) != last_config_ts {
                        last_config_ts = Some(modified);
                        telemetry_server::telemetry_log!("👀 [HOT-RELOAD] dynamic_config.json changed. Symbols update requires restart.");
                    }
                }
            }

            if let Ok(entries) = std::fs::read_dir("models") {
                for entry in entries.filter_map(|e| e.ok()) {
                    let path = entry.path();
                    let ext = path.extension().and_then(|s| s.to_str());
                    if ext == Some("json") || ext == Some("bin") {
                        if let Some(file_stem) = path.file_stem().and_then(|s| s.to_str()) {
                            if let Ok(meta) = std::fs::metadata(&path) {
                                if let Ok(modified) = meta.modified() {
                                    let last_ts = forest_timestamps.get(file_stem);
                                    if last_ts != Some(&modified) {
                                        forest_timestamps.insert(file_stem.to_string(), modified);
                                        if god_engine_core::ml_inference::NanoForest::load_global(
                                            file_stem,
                                            path.to_str().unwrap(),
                                        )
                                        .is_ok()
                                        {
                                            telemetry_server::telemetry_log!("🔥 [HOT-RELOAD] NanoForest AI Brain hot-swapped for {}!", file_stem);
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
    });

    // FASE 15: Quantum Leap - Automatic Transition from Demo to Live
    // El Hot-Swap ahora se delega completamente y de forma síncrona y cuántica al PhaseOrchestrator
    // dentro del loop principal, eliminando condiciones de carrera y spaghetti code.

    // Initialize Redb Zero-Copy Persistence and Real Balance

    // E4b — GHOST TASK ELIMINADA: cada 6h esta tarea lanzaba
    // `cargo run --bin evolution` que (a) compilaba en el host MIENTRAS el
    // HFT operaba (jitter de CPU/disco en el camino crítico), (b) bloqueaba
    // su task tokio con child.wait() y (c) su salida (dynamic_config.json)
    // no la recargaba nadie en runtime — CPU pura, impacto cero. La
    // evolución en vivo ya la realizan LiveEvolutionDaemon y DarwinDaemon
    // a través del embudo versionado del almacén de genomas.
    let _ = initial_capital;

    std::fs::create_dir_all("data").unwrap_or_default();

    let state_redb_path = EnvManager::data_path("state.redb");
    let db = Arc::new(
        Database::create(&state_redb_path).expect("CRITICAL: Failed to create redb database"),
    );

    if let Ok(write_txn) = db.begin_write() {
        {
            let mut table = write_txn.open_table(CAPITAL_TABLE).unwrap();
            let cap_val = table.get(1).unwrap_or(None).map(|v| v.value());
            if let Some(_val) = cap_val {
                // DB has a value, but we trust the API value over DB now.
            } else {
                let _ = table.insert(1, initial_capital);
            }
        }
        let _ = write_txn.commit();
        telemetry_server::telemetry_log!(
            "💾 [PERSISTENCE] Capital Redb Zero-Copy loaded: ${:.4}",
            initial_capital
        );
    }

    // Shared Atomic Capital Pool (Dynamically scaled from API extraction)
    // Axiom V: Unified Cross-Margin Pool
    let unified_capital = Arc::new(AtomicU64::new(initial_capital.to_bits()));

    // Non-blocking Zero-Copy persistence channel
    let (db_tx, mut db_rx) = mpsc::channel::<(f64, f64)>(5000);
    let db_clone = Arc::clone(&db);
    tokio::spawn(async move {
        while let Some((sc, sw)) = db_rx.recv().await {
            let total = sc + sw;
            if let Ok(write_txn) = db_clone.begin_write() {
                if let Ok(mut table) = write_txn.open_table(CAPITAL_TABLE) {
                    let _ = table.insert(1, total);
                }
                let _ = write_txn.commit();
            }
        }
    });

    let exec = Arc::clone(&exec);
    let loop_telemetry_tx = telemetry_tx.clone();
    let dark_router_unified = Arc::clone(&dark_router);

    let symbols_clone = symbols.clone();
    let mut symbol_to_id: std::collections::HashMap<String, usize> =
        std::collections::HashMap::new();
    for (i, sym) in symbols_clone.iter().enumerate() {
        symbol_to_id.insert(sym.clone(), i);
        symbol_to_id.insert(sym.to_uppercase(), i);
        symbol_to_id.insert(sym.to_lowercase(), i);
    }

    // 🔍 FETCH FORENSE DE POSICIONES ACTIVAS CON BINANCE API
    // B1.2-fix: el discriminador es el MODO PAPEL del executor, NO
    // is_demo_mode del orquestador (que agrupa papel-local y testnet-live).
    // En testnet las posiciones son exposición REAL con brackets reales:
    // ignorarlas al arrancar dejaba el arena ciego → riesgo de exposición
    // duplicada y estado divergente (hallazgo del relanzamiento v7).
    let mut restored_positions = exec.load().fetch_open_positions().await.unwrap_or_default();
    if exec.load().is_paper_trading() {
        telemetry_server::telemetry_log!("⚠️ [PAPER LOCAL] Ignorando posiciones REST — no hay cuenta viva que reconciliar.");
        restored_positions.clear();
    }

    // HOST-013 (informe decimocuarto) — LEVERAGE REAL para el margen de
    // adopción de FASE 5: el código dividía el notional entre 10.0
    // hardcodeado — en cuentas 20×/50× el margen adoptado quedaba inflado
    // 2-5× (falsa escasez ⇒ vetos de entrada contra capital que sí
    // existe). El leverage vivo por símbolo vive en positionRisk. Fallback
    // conservador 10× (comportamiento anterior) si el fetch falla.
    let restore_lev_map: std::collections::HashMap<String, f64> = if exec.load().is_paper_trading() {
        std::collections::HashMap::new()
    } else {
        exec.load()
            .fetch_position_risk()
            .await
            .map(|entries| {
                entries
                    .iter()
                    .filter(|e| e.leverage.is_finite() && e.leverage >= 1.0)
                    .map(|e| (e.symbol.clone(), e.leverage))
                    .collect()
            })
            .unwrap_or_default()
    };

    // ── F1.11 MODO HEDGE ─────────────────────────────────────────────────────
    // El motor envía positionSide=LONG/SHORT siempre; si la cuenta está en
    // one-way, TODA orden falla con -4061. Verificar/activar antes de operar.
    // B1.2-fix: también en testnet (cuenta viva), no solo mainnet.
    if !exec.load().is_paper_trading() {
        match exec.load().ensure_hedge_mode().await {
            Ok(true) => {
                telemetry_server::telemetry_log!("🔀 [PRE-FLIGHT] Cuenta migrada a modo HEDGE")
            }
            Ok(false) => telemetry_server::telemetry_log!("🔀 [PRE-FLIGHT] Modo HEDGE ya activo"),
            Err(e) => telemetry_server::telemetry_log!(
                "🚨 [PRE-FLIGHT] No se pudo garantizar modo hedge: {} — TODA orden fallará (-4061)",
                e
            ),
        }
    }

    // ── F1.7 RECONCILIACIÓN AL ARRANQUE ──────────────────────────────────────
    // positionRisk (verdad del exchange) diff contra OrderRegistry (F1.5).
    // Directiva: saber SIEMPRE si hay posiciones abiertas antes de operar.
    // B1.2-fix: también en testnet (cuenta viva), no solo mainnet.
    if !exec.load().is_paper_trading() {
        match exec.load().fetch_position_risk().await {
            Ok(entries) => {
                let report =
                    execution_engine::reconciliation::reconcile(&entries, &exec.load().registry());
                telemetry_server::telemetry_log!("🧾 [RECONCILIACIÓN ARRANQUE] {}", report.summary);
                for p in &report.open_positions {
                    telemetry_server::telemetry_log!(
                        "   📍 {} {} @ {:.4} (uPnL {:+.4}, liq {:.2}, lev {:.0}x)",
                        p.symbol,
                        p.position_amt,
                        p.entry_price,
                        p.unrealized_pnl,
                        p.liquidation_price,
                        p.leverage
                    );
                }
                for s in &report.suspicious_active_orders {
                    telemetry_server::telemetry_log!("   ⚠️ ORDEN SOSPECHOSA: {}", s);
                }
            }
            Err(e) => {
                telemetry_server::telemetry_log!(
                    "⚠️ [RECONCILIACIÓN ARRANQUE] positionRisk no disponible: {}",
                    e
                );
            }
        }
    }

    // ── F1.6 USER-DATA STREAM (fills en tiempo real) ─────────────────────────
    // ORDER_TRADE_UPDATE → OrderRegistry; ACCOUNT_UPDATE → puente de capital.
    // La adopción de posiciones al arena es dueño el mapa de estado (F4.9);
    // aquí garantizamos que el capital dinámico refleje la verdad del exchange.
    struct CapitalBridgeSink {
        unified_capital: Arc<AtomicU64>,
    }
    impl execution_engine::user_data_stream::AccountSink for CapitalBridgeSink {
        fn on_capital(&self, usdt: f64) {
            audit_engine::telemetry::update_dynamic_capital(usdt);
            if usdt > 0.0 && usdt.is_finite() {
                self.unified_capital
                    .store(usdt.to_bits(), Ordering::Relaxed);
            }
        }
        fn on_positions(&self, positions: &[execution_engine::user_data_stream::RemotePosition]) {
            if !positions.is_empty() {
                let summary: Vec<String> = positions
                    .iter()
                    .map(|p| format!("{} {}", p.symbol, p.position_amt))
                    .collect();
                telemetry_server::telemetry_log!(
                    "📡 [USER-DATA] Posiciones vivas: {}",
                    summary.join(", ")
                );
            }
        }
    }
    // X-011 (REHAB-4): bandera de apagado del streamer de la era inicial —
    // la transición demo→mainnet la activará antes de spawnear el nuevo
    // (antes: DOS streamers vivos; el de testnet pisaba el capital mainnet).
    let demo_streamer_abort = Arc::new(std::sync::atomic::AtomicBool::new(false));
    let streamer = execution_engine::user_data_stream::UserDataStreamer::new(
        exec.load().client().clone(),
        exec.load().registry(),
    )
    .with_sink(std::sync::Arc::new(CapitalBridgeSink {
        unified_capital: Arc::clone(&unified_capital),
    }))
    // D-702 (DÉCIMA OLA · auditoría integral): este streamer nace ANTES que el
    // arena (que se construye dentro del hilo del núcleo, con pila de 32 MiB por
    // D-684), así que no puede recibirlo; el de la era de producción sí lo
    // recibe. Con la puerta de capital (D-700) esta era es papel local, donde no
    // hay cierres reales que contabilizar.
    .with_api_secret(exec.load().api_secret())
    .with_shutdown(Arc::clone(&demo_streamer_abort));
    tokio::spawn(async move {
        streamer.start().await;
    });
    telemetry_server::telemetry_log!(
        "🔌 [USER-DATA] Stream privado spawned (fills en tiempo real + capital vivo)"
    );

    let rx_events = rx_events;

    // FASE 15: GLOBAL ROI & PnL TRACKERS (Motor Universal Continuo)
    let mut total_gross_pnl = 0.0;
    let mut total_net_pnl = 0.0;
    let mut total_fees = 0.0;
    let mut total_trades = 0;
    let mut total_wins = 0;

    let rt_handle = tokio::runtime::Handle::current();

    let server_time_ms = match exec.load().fetch_server_time().await {
        Ok(time) => {
            telemetry_server::telemetry_log!(
                "⏱️ [NTP SYNC] Binance Server Time retrieved: {}",
                time
            );
            time
        }
        Err(_) => {
            telemetry_server::telemetry_log!(
                "⚠️ [NTP SYNC FAILED] Falling back to local SystemTime"
            );
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap_or_default()
                .as_millis() as i64
        }
    };
    let local_ms = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap_or_default()
        .as_millis() as i64;
    let ntp_offset_ms = server_time_ms - local_ms;
    telemetry_server::telemetry_log!(
        "⏱️ [NTP SYNC] Local Time: {}, Offset to Binance: {}ms",
        local_ms,
        ntp_offset_ms
    );

    let num_symbols = symbols.len();

    // ── F4.1: OMNI FEATURES REALES EN PRODUCCIÓN ────────────────────────────
    // El swing NN recibía &[0.0;54] SIEMPRE: el subsistema macro era un
    // fantasma — god_engine jamás creó el OmniState. Ahora: estado vivo +
    // pollers REST ligeros (macro FRED/PAXG + sentiment). Los WS cross-exchange
    // pesados quedan desconectados (documentado) — pero la mitad macro del
    // vector es REAL en vivo, con staleness medible.
    let omni_state_live = Arc::new(data_pipeline::omni_multiplexer::OmniState::new());
    {
        let st = Arc::clone(&omni_state_live);
        tokio::spawn(async move {
            data_pipeline::omni_multiplexer::run_macro_rest_poller(st).await;
        });
        let st = Arc::clone(&omni_state_live);
        tokio::spawn(async move {
            data_pipeline::omni_multiplexer::run_sentiment_onchain_poller(st).await;
        });
        telemetry_server::telemetry_log!(
            "🌐 [OMNI] Features macro REALES cableadas al swing NN (FRED/PAXG + sentiment, 60s/120s)"
        );
    }

    let unified_handle = std::thread::Builder::new().stack_size(32 * 1024 * 1024).spawn({
        let loop_ws_url = Arc::clone(&ws_url);
        let loop_streams_str = streams_str.clone();
        let historical_klines = historical_klines.clone();
        let omni_state_hot = Arc::clone(&omni_state_live);
        let rt_handle_for_thread = rt_handle.clone();
        let shutdown_flag = Arc::clone(&shutdown_requested);
        move || {
        if let Some(core_ids) = core_affinity::get_core_ids() {
            if core_ids.len() > 1 {
                core_affinity::set_for_current(core_ids[1]);
                telemetry_server::telemetry_log!("🔒 [CPU PINNING] Unified Core pinned to CPU Core {}", core_ids[1].id);
            }
        }
        os_guardian::set_current_thread_time_critical();
        telemetry_server::telemetry_log!("🧠 [UNIFIED CORE] Initialized. Target latency: <500ns.");

        // Axiom XIV: Superposición Cuántica - Mentes Aisladas (Scalp vs Swing)
        let arena_real = Arc::new(quantum_arena::GlobalArena::new(initial_capital));
        // X-020 (REHAB-6): arena_shadow ELIMINADA — 40+MB VirtualLock muertos. Jamás procesó un tick (todas sus referencias eran init-only) y su canal tx_shadow encolaba modelos NN eternamente. La "mente sombra" real vive en shadow_forest + ShadowForest del online_daemon.

        // E-01 — INICIALIZAR BUS DE TELEMETRÍA MMAP: sin esto, el writer
        // global es no-op perpetuo y el daemon lee un archivo que nadie
        // escribe (el lazo de autoevolución queda seco). Una línea que
        // conecta el circuito predicción→realidad→shadow forest.
        god_engine_core::reexport_storage::mmap_bus::init_global_telemetry(
            &quantum_arena::paths::data_join("telemetry.mmap"),
        );
        arena_real.config.live_maker_fee.store(live_maker, Ordering::Relaxed);
        arena_real.config.live_taker_fee.store(live_taker, Ordering::Relaxed);

        // FASE 38 & 14: Guardian Activo (Memory Panic)
        // F5.3 — FIX AFINIDAD: 0xFFFF asumía 16 cores — en una máquina de 8 el
        // guardian pedía cores inexistentes. Máscara derivada del hardware REAL.
        let cpu_mask = {
            let cores = core_affinity::get_core_ids().map(|c| c.len()).unwrap_or(1);
            if cores >= 64 {
                usize::MAX
            } else {
                (1usize << cores) - 1
            }
        };
        let memory_per_symbol_mb = 128;
        let dynamic_memory_limit = (num_symbols * memory_per_symbol_mb) + 2048; // Escala dinámica basada en el Universo
        os_guardian::init_guardian(cpu_mask, dynamic_memory_limit, Arc::clone(&arena_real));

        unsafe {
            if os_guardian::memory_compaction::lock_critical_memory(&*arena_real) {
                telemetry_server::telemetry_log!("🔒 [OS-GUARDIAN] arena_real asegurada en RAM física (Zero Swapping)");
            }
            if os_guardian::memory_compaction::lock_critical_memory_slice(&*arena_real.coins) {
                telemetry_server::telemetry_log!("🔒 [OS-GUARDIAN] arena_real.coins (TickRings) asegurados en RAM física");
            }
        }

        // ── F5.2: SISTEMA INMUNE — el kill-switch por fin ARMADO ──────────────
        // Auditoría: arena.kill_switch_active NUNCA se armaba (ningún writer en
        // el camino vivo) y STOP_TRADING.LOCK no lo leía nadie (decorativo).
        // Ahora: cada 5s se vigila (1) STOP_TRADING.LOCK del operador,
        // (2) drawdown vs límite del genoma sobre el pico observado,
        // (3) latencia obsoleta SOSTENIDA (3 strikes ≈ 15s).
        // Al disparar: kill-switch en arena + executor + FLATTEN de todo.
        // LATCH: rearme solo reiniciando el proceso — decisión humana.
        {
            let arena_imm = Arc::clone(&arena_real);
            let exec_imm = Arc::clone(&exec);
            // X-013 (REHAB-4): plano de capital del EXCHANGE (CapitalBridgeSink
            // lo escribe desde ACCOUNT_UPDATE). El inmune mide el drawdown
            // sobre el MÁS CONSERVADOR de los planos (arena simulado vs
            // exchange real) — antes solo leía el arena: en live el funding/
            // slippage real invisibilizaban el drawdown contable verdadero.
            let exchange_capital_imm = Arc::clone(&unified_capital);
            rt_handle.spawn(async move {
                let mut latched = false;
                let mut peak_capital = arena_imm.unified_capital.load(Ordering::Relaxed);
                let mut latency_strikes: u32 = 0;
                loop {
                    tokio::time::sleep(std::time::Duration::from_secs(5)).await;
                    if latched {
                        continue;
                    }

                    // (1) El operador manda: el archivo existe ⇒ parar TODO.
                    let operator_lock = operator_lock_exists();

                    // (2) Drawdown sobre pico observado (muestreo 5s).
                    // X-013: mínimo entre plano arena y plano exchange — el
                    // más conservador gobierna la defensa.
                    let cap_arena = arena_imm.unified_capital.load(Ordering::Relaxed);
                    let cap_exchange =
                        f64::from_bits(exchange_capital_imm.load(Ordering::Relaxed));
                    let cap = if cap_exchange > 0.0 {
                        cap_arena.min(cap_exchange)
                    } else {
                        cap_arena
                    };
                    if cap > peak_capital {
                        peak_capital = cap;
                    }
                    let dd = if peak_capital > 0.0 {
                        (peak_capital - cap) / peak_capital
                    } else {
                        0.0
                    };
                    let max_dd = arena_imm.config.global_max_drawdown.load(Ordering::Relaxed);
                    // Una cuenta liquidada (cap <= 0) debe DISPARAR el sistema
                    // inmune, no desarmarlo: el guard `cap > 0.0` anterior
                    // dejaba todos los frenos apagados exactamente en el único
                    // escenario donde son críticos.
                    let insolvent = cap <= 0.0;
                    let dd_breach = max_dd > 0.0 && dd >= max_dd;

                    // (3) Latencia: 3 muestras consecutivas por encima del umbral.
                    let lat = arena_imm.last_ws_latency_ms.load(Ordering::Relaxed);
                    let lat_thresh = arena_imm
                        .config
                        .latency_ms_panic_threshold
                        .load(Ordering::Relaxed)
                        .max(50.0) as u64;
                    // X-010: la rama incluye el flag global del watchdog —
                    // muerte silenciosa del feed (5s sin datos) cuenta como
                    // strike aunque la última latencia medida estuviera sana.
                    // CERT-M4-H04: el stall flag POR SÍ SOLO ya NO basta para
                    // el strike 3/3 — una desconexión transitoria de red (WS
                    // backoff capped 5s pero DNS+reconnect >15s) convertía
                    // en flatten-all-at-market en el peor spread. Ahora el
                    // stall cuenta strike SÓLO si la latencia MEDIDA también
                    // está breach (evidencia de degradación real, no sólo
                    // transport). El flag puro genera WARNING, no strike.
                    let stalled = quantum_arena::feed_health::is_stalled();
                    if lat > lat_thresh {
                        latency_strikes += 1;
                    } else if stalled && lat > lat_thresh / 2 {
                        // stall + latencia media elevada: degradación parcial
                        latency_strikes += 1;
                    } else if stalled {
                        // stall puro: warning sin strike (transport, no datos)
                        if latency_strikes == 0 {
                            telemetry_server::telemetry_log!(
                                "⚠️ [IMMUNE] WS stalled pero latencia medida OK ({:.0}ms) — strike NO aplicado (sólo transport)"
                            , lat);
                        }
                        // no reset ni increment: mantener estado
                    } else {
                        latency_strikes = 0;
                    }
                    let lat_breach = latency_strikes >= 3;

                    if operator_lock || insolvent || dd_breach || lat_breach {
                        let reason = if operator_lock {
                            "STOP_TRADING.LOCK del operador".to_string()
                        } else if insolvent {
                            format!(
                                "cuenta INSOLVENTE/LIQUIDADA (capital {:.4} <= 0) — recuperación sobre cuenta sin fondos deshabilitada",
                                cap
                            )
                        } else if dd_breach {
                            format!(
                                "drawdown {:.1}% >= límite {:.1}%",
                                dd * 100.0,
                                max_dd * 100.0
                            )
                        } else {
                            format!("latencia {}ms > {}ms sostenida", lat, lat_thresh)
                        };
                        telemetry_server::telemetry_log!(
                            "🚨 [SISTEMA INMUNE] ACTIVADO: {}. Kill-switch ARMADO + aplanado total.",
                            reason
                        );
                        arena_imm.kill_switch_active.store(true, Ordering::Relaxed);
                        let executor = exec_imm.load_full();
                        executor.trigger_kill_switch();
                        match executor.flatten_all_positions().await {
                            Ok((syms, positions)) => telemetry_server::telemetry_log!(
                                "🧹 [SISTEMA INMUNE] Aplanado: {} símbolos con órdenes canceladas, {} posiciones cerradas. Reinicio manual para rearmar.",
                                syms,
                                positions
                            ),
                            Err(e) => telemetry_server::telemetry_log!(
                                "🚨 [SISTEMA INMUNE] Aplanado FALLÓ: {} — INTERVENCIÓN MANUAL URGENTE.",
                                e
                            ),
                        }
                        latched = true;
                    }
                }
            });
        }

        // Update fees dynamically
        arena_real.config.live_maker_fee.store(live_maker, Ordering::Relaxed);
        arena_real.config.live_taker_fee.store(live_taker, Ordering::Relaxed);
        arena_real.server_time_offset_ms.store(ntp_offset_ms, Ordering::Relaxed);
        
        // Inject arena into OrderExecutor
        exec.load().arena.store(Some(Arc::clone(&arena_real)));

        // Spawn NTP Synchronizer for live timestamp drift correction
        // HOST-016 — vía spawn_ntp_synchronizer: guardia de generación que
        // garantiza UN solo loop NTP vivo a la vez.
        spawn_ntp_synchronizer(
            &rt_handle_for_thread,
            Arc::new(exec.load().client().clone()),
            Arc::clone(&arena_real),
        );

        // Reality Physics: Shadow Simulator uses identical dynamic fees extracted from exchange

        // Phase 17: Apply Genotype Object Directly
        telemetry_server::telemetry_log!("🧬 [GENOMA] Applying Active Genome to Unified Core...");
        initial_genome.apply_to_arena(&arena_real);

        // FASE 5 (ADOPCIÓN DE ESTADO: Recuperar posiciones abiertas tras caída)
        telemetry_server::telemetry_log!("========================================================");
        telemetry_server::telemetry_log!("🔍 [FASE 5] ADOPCIÓN DE ESTADO (RECONCILIACIÓN DE POSICIONES)");
        telemetry_server::telemetry_log!("========================================================");
        // B3.8 — ROTACIÓN DEL DIARIO (auditoría del roadmap): la recuperación
        // sólo necesita el ÚLTIMO registro por (símbolo, lado); sin compactar
        // el diario crece sin cota y cada reconexión lo re-lee entero.
        compact_position_journal();
        if restored_positions.is_empty() {
            telemetry_server::telemetry_log!("🔍 [RECONCILIATION] Cero posiciones abiertas en Binance. Arena inicializada limpia.");
        } else {
            telemetry_server::telemetry_log!("🔍 [RECONCILIATION] Detectadas {} posiciones abiertas en Binance API:", restored_positions.len());
            let exec_restore = Arc::clone(&exec);
            for pos in &restored_positions {
                telemetry_server::telemetry_log!("   👉 Símbolo activo en exchange: {} (Qty: {})", pos.symbol, pos.qty);
                if let Some(&coin_idx) = symbol_to_id.get(&pos.symbol) {
                    let now_ms = std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).unwrap().as_millis() as u64;
                    // D-726 + HOST-013: EL MARGEN SALE DEL APALANCAMIENTO REAL Y
                    // SE RESERVA DE VERDAD.
                    //
                    // El margen se reconstruía dividiendo el nocional por el
                    // literal 10 —una posición a 20x quedaba con el doble de
                    // margen del real y una a 5x, con la mitad— y, peor, NUNCA se
                    // sumaba a `arena.used_margin`: tras un reinicio con
                    // posiciones abiertas, `free_margin = capital − used_margin`
                    // devolvía el capital ENTERO como libre y el motor abría
                    // posiciones nuevas como si no tuviera ninguna. La rama de
                    // adopción de `reconcile_arena` sí reserva margen, pero exige
                    // que la posición NO esté ya abierta en el arena, así que no
                    // corregía ésta. El apalancamiento real viene de
                    // `/fapi/v2/positionRisk`, por la posición misma o por el
                    // mapa de HOST-013; si el exchange no lo informa se usa el del
                    // genoma, nunca un literal.
                    let lev_real = if pos.leverage.is_finite() && pos.leverage >= 1.0 {
                        pos.leverage
                    } else {
                        restore_lev_map
                            .get(&pos.symbol)
                            .copied()
                            .filter(|l| l.is_finite() && *l >= 1.0)
                            .unwrap_or_else(|| {
                                arena_real
                                    .config
                                    .global_leverage
                                    .load(Ordering::Relaxed)
                                    .clamp(1.0, 125.0)
                            })
                    };
                    let calculated_margin = (pos.qty.abs() * pos.entry_price) / lev_real;
                    arena_real
                        .used_margin
                        .fetch_add(calculated_margin, Ordering::Relaxed);
                    arena_real.coins[coin_idx].positions.position.open_with_horizon(
                        pos.is_long,
                        pos.entry_price,
                        pos.qty,
                        calculated_margin,
                        now_ms,
                        0.0,
                        0.0,
                        quantum_arena::position::PositionHorizon::Continuous,
                    );
                    telemetry_server::telemetry_log!("   ✅ Posición reconciliada en Arena para {} (Margen: ${:.2} a {:.0}x — reservado en used_margin)", pos.symbol, calculated_margin, lev_real);

                    // B2.7 — RECUPERACIÓN DE CONTEXTO (directriz del operador):
                    // qué τ y qué predicción ML seguían esta posición vive en
                    // el diario de entrada; sin él, la re-protección caería al
                    // ancla rápida (τ=0) perdiendo el rigor espectral.
                    let ctx = recover_position_context(&pos.symbol, pos.is_long);
                    // B3.14 — la posición adoptada EXISTE en el exchange:
                    // sus cierres contabilizan.
                    if let Some(c) = arena_real.coins.get(coin_idx) {
                        c.positions
                            .position
                            .exchange_confirmed
                            .store(true, Ordering::Relaxed);
                    }
                    if let Some(rc) = &ctx {
                        if let Some(c) = arena_real.coins.get(coin_idx) {
                            c.positions.position.entry_tau_ms.store(rc.tau_ms, Ordering::Relaxed);
                        }
                        telemetry_server::telemetry_log!(
                            "   🧠 [CONTEXTO] {} {}: τ_entrada={}ms ({}), ml_entrada={:.3}, edad {:.1}h",
                            pos.symbol,
                            if pos.is_long { "LONG" } else { "SHORT" },
                            rc.tau_ms,
                            format_tau(rc.tau_ms as f64),
                            rc.ml,
                            rc.age_hours
                        );
                    } else {
                        telemetry_server::telemetry_log!(
                            "   ⚠️ [CONTEXTO] {} sin registro en el diario — τ ancla rápida",
                            pos.symbol
                        );
                    }
                    // FIX #348 → B1.2: re-armar protección remota para
                    // posiciones restauradas. AHORA: curvas del genoma
                    // (HorizonCurve) evaluadas en la τ RECUPERADA — el rigor
                    // matemático completo: la curva evolucionada por SA sobre
                    // backtest, decidida por el espectro que motivó la entrada.
                    // Sin diario ⇒ piso espectral (ancla rápida = mínimo de la
                    // curva por construcción: conservador por diseño).
                    let pos_sym = pos.symbol.clone();
                    let pos_is_long = pos.is_long;
                    let pos_qty = pos.qty.abs();
                    let pos_entry = pos.entry_price;
                    let exec_oco = Arc::clone(&exec_restore);
                    let arena_restore = Arc::clone(&arena_real);
                    let recovered_tau = ctx.as_ref().map(|c| c.tau_ms).unwrap_or(0);

                    rt_handle.spawn(async move {
                        let _ = ensure_position_protected(
                            &exec_oco.load(),
                            &arena_restore,
                            &pos_sym,
                            pos_is_long,
                            pos_qty,
                            pos_entry,
                            recovered_tau,
                            "RESTORE",
                        )
                        .await;
                    });
                }
            }
        }

        let arena_telemetry = Arc::clone(&arena_real);
        let telemetry_tx_for_server = telemetry_tx.clone();
        rt_handle.spawn(async move {
            telemetry_server::start_telemetry_server(arena_telemetry, telemetry_tx_for_server).await;
        });

        let (tx_real, rx_real) = std::sync::mpsc::channel();
        // X-020 (REHAB-6): canal tx_shadow ELIMINADO — su receptor jamás se
        // drenaba: cada re-entrenamiento encolaba una NN completa (54→64→32)
        // para siempre, leak de heap ilimitado en la máquina de 16GB.

        let rt_for_darwin = rt_handle.clone();

        // La evaluación de Warmup en hilo separado fue removida para centralizar la orquestación en el PhaseOrchestrator del motor HFT, evitando colisiones.

        // Spawn Auto-Evolucion (Model Watcher)
        rt_for_darwin.spawn(async move {
            telemetry_server::telemetry_log!("🧠 [MODEL WATCHER] Escaneando mutaciones en models/DarkAlpha_BTCUSDT.json cada 10s...");
            let mut last_mtime = std::time::SystemTime::UNIX_EPOCH;
            loop {
                tokio::time::sleep(std::time::Duration::from_secs(10)).await;
                if let Ok(metadata) = std::fs::metadata("models/DarkAlpha_BTCUSDT.json") {
                    if let Ok(mtime) = metadata.modified() {
                        if mtime > last_mtime {
                            last_mtime = mtime;
                            if let Ok(model) = dark_alpha_engine::DarkAlphaEngine::load_json("models/DarkAlpha_BTCUSDT.json") {
                                telemetry_server::telemetry_log!("🧬 [MODEL WATCHER] Nueva genetica detectada. Desplegando en Zero-Copy...");
                                let _ = tx_real.send(model);
                            }
                        }
                    }
                }
            }
        });

        let rt_for_gc = rt_handle.clone();
        rt_for_gc.spawn(async move {
            telemetry_server::telemetry_log!("🧹 [OS-GUARDIAN] Inicializando Garbage Collector Estadístico (1H interval)...");
            loop {
                tokio::time::sleep(tokio::time::Duration::from_secs(3600)).await;
                telemetry_server::telemetry_log!("🧹 [OS-GUARDIAN] Ejecutando Memory Compaction OS-level...");
                unsafe {
                    os_guardian::memory_compaction::force_working_set_compaction();
                }
            }
        });

        // D-433: Erradicación de la Guerra de Demonios Evolutivos
        // LiveEvolutionDaemon es el motor evolutivo primario (CMA-ES 140D continuo).
        // DarwinDaemon legacy solo se activa si se solicita explícitamente vía variable de entorno,
        // eliminando la colisión y sobrescritura concurrente sobre active_genome.json.
        let enable_legacy_darwin = std::env::var("ENABLE_LEGACY_DARWIN_DAEMON")
            .map(|v| v.trim() == "true")
            .unwrap_or(false);
        if enable_legacy_darwin {
            let daemon = god_engine_core::darwin::DarwinDaemon::new(Arc::clone(&arena_real));
            rt_for_darwin.spawn(async move {
                telemetry_server::telemetry_log!("🧬 [DARWIN-DAEMON] Iniciando Motor Cuántico Evolutivo Legacy...");
                loop {
                    tokio::time::sleep(tokio::time::Duration::from_secs(60)).await;
                    let daemon_clone = god_engine_core::darwin::DarwinDaemon::new(Arc::clone(&daemon.live_arena));
                    let _ = tokio::task::spawn_blocking(move || {
                        daemon_clone.evolve_online();
                    }).await;
                }
            });
        }

        // F4.7 — ONLINE LEARNING DAEMON CABLEADO (era fantasma desde su creación):
        // ingesta telemetría mmap → shadow forest → umbrales ML vivos →
        // promociones vía el EMBUDO del almacén de genomas (linaje auditable).
        // Sus protecciones F4.5 activas: sharpe de estrategia real (no beta del
        // mercado) y kill-switch intocable por métricas.
        {
            let ledger_path = EnvManager::data_path("evolution_ledger.redb");
            let mut online_daemon = evolution_engine::online_daemon::LiveEvolutionDaemon::new(
                evolution_engine::online_daemon::QuantumHotSwapState::new(),
                Arc::clone(&arena_real),
                orchestrator
                    .read()
                    .unwrap_or_else(|e| e.into_inner())
                    .is_demo_mode,
                &ledger_path,
                "config_dir/genotypes/online_champion.json",
            );
            if !evolution_engine::online_daemon::live_evolution_armed_for_env() {
                telemetry_server::telemetry_log!(
                    "🔒 [D-689] Evolución en vivo DESARMADA: umbrales ML y genoma sólo cambian por promoción validada (TG_LIVE_GENOME_EVOLUTION_ARMED=1 para armarla). Deriva, kill-switch y rollback siguen activos."
                );
            } else {
                // QO-E1: primera vez que la autoevolución vive ARMADA — la
                // auditoría halló prod/history con 0 generaciones: los tres
                // lazos jamás corrieron. Las redes de seguridad (rollback
                // t≤−2.0, kill EWMA, validación bounds/RR) están SIEMPRE.
                telemetry_server::telemetry_log!(
                    "🧬 [QO-E1] Autoevolución en vivo ARMADA (entorno demo): mutación 2000-candidatos sobre returns reales + cosecha shadow-forest (≥15 trades) + umbrales del forest online. Rollback y kill-switch activos. Prod desarmado (paso humano)."
                );
            }
            rt_for_darwin.spawn(async move {
                telemetry_server::telemetry_log!(
                    "🌐 [ONLINE-DAEMON] Aprendizaje online vivo (shadow forest + drift EWMA)"
                );
                online_daemon.run_online_learning_loop().await;
            });
        }

        // QO-E2b — AUTO-TRAINER NN como proceso hijo SUPERVISADO (demo).
        // CERT-M4-H01: el Child anterior se dropeaba inmediatamente —
        // nunca esperado, nunca reiniciado en crash, nunca terminado al
        // salir el engine: múltiples reinicios acumulaban múltiples
        // trainers compitiendo por el mismo dataset. Ahora un thread
        // supervisor hace wait+restart-with-backoff y un JobObject-like
        // kill al salir.
        if std::env::var("TG_NN_TRAINER").map(|v| v.trim() == "0").unwrap_or(false) {
            telemetry_server::telemetry_log!(
                "⏸️ [QO-E2b] Auto-trainer NN desactivado por TG_NN_TRAINER=0"
            );
        } else {
            let exe = std::env::current_exe().ok();
            let trainer_path = exe
                .as_ref()
                .and_then(|p| p.parent().map(|d| d.join("auto_trainer_daemon.exe")))
                .filter(|p| p.exists());
            if let Some(tp) = trainer_path {
                let tp = tp.to_string_lossy().to_string();
                std::thread::Builder::new()
                    .name("nn-trainer-supervisor".into())
                    .spawn(move || {
                        let mut backoff_secs = 5u64;
                        let mut total_restarts = 0u32;
                        loop {
                            let mut child = match std::process::Command::new(&tp)
                                .arg("BTCUSDT")
                                .stdout(std::process::Stdio::null())
                                .stderr(std::process::Stdio::null())
                                .spawn()
                            {
                                Ok(c) => c,
                                Err(e) => {
                                    eprintln!("⚠️ [QO-E2b] trainer spawn falló: {e} — reintentando en 60s");
                                    std::thread::sleep(std::time::Duration::from_secs(60));
                                    continue;
                                }
                            };
                            println!(
                                "🤖 [QO-E2b] Auto-trainer NN lanzado (pid {}, restart #{})",
                                child.id(),
                                total_restarts
                            );
                            // Esperar a que termine (bloqueante en este thread dedicado)
                            match child.wait() {
                                Ok(status) if status.success() => {
                                    println!("✅ [QO-E2b] trainer terminó limpio — reiniciando en 30s");
                                    std::thread::sleep(std::time::Duration::from_secs(30));
                                }
                                Ok(_) => {
                                    println!(
                                        "⚠️ [QO-E2b] trainer terminó con error — reiniciando en {}s (backoff)",
                                        backoff_secs
                                    );
                                    std::thread::sleep(std::time::Duration::from_secs(backoff_secs));
                                    backoff_secs = (backoff_secs * 2).min(300); // cap 5 min
                                }
                                Err(e) => {
                                    eprintln!("⚠️ [QO-E2b] trainer wait falló: {e} — reintentando en 60s");
                                    std::thread::sleep(std::time::Duration::from_secs(60));
                                }
                            }
                            total_restarts += 1;
                            if total_restarts > 50 {
                                eprintln!("🛑 [QO-E2b] trainer excedió 50 reinicios — abandonando");
                                break;
                            }
                        }
                    })
                    .ok();
            } else {
                telemetry_server::telemetry_log!(
                    "⚠️ [QO-E2b] auto_trainer_daemon.exe no encontrado junto al binario — compílalo para cerrar el lazo NN"
                );
            }
        }

        // R3.5 — RECONCILIACIÓN PERIÓDICA OMNISCIENTE (cada 60s)
        // Detecta y corrige discrepancias entre el Exchange y el estado local:
        // - Adopta posiciones abiertas en OrderRegistry y GlobalArena (Swing)
        // - Cierra posiciones fantasma en GlobalArena si Binance está FLAT
        // - Corrige drift en cantidades por fills parciales
        // - Purga órdenes terminadas mayores a 10 min (600_000 ms)
        {
            let exec_reconcile = Arc::clone(&exec);
            let arena_reconcile = Arc::clone(&arena_real);
            rt_handle.spawn(async move {
                telemetry_server::telemetry_log!("🔄 [RECONCILIATION-LOOP] Iniciando bucle periódico de reconciliación (60s)...");
                let mut interval = tokio::time::interval(tokio::time::Duration::from_secs(60));
                interval.set_missed_tick_behavior(tokio::time::MissedTickBehavior::Skip);
                loop {
                    interval.tick().await;
                    let executor = exec_reconcile.load_full();
                    match executor.fetch_position_risk().await {
                        Ok(positions) => {
                            let registry = executor.registry();
                            let report = execution_engine::reconciliation::reconcile(&positions, &registry);
                            let adopted = report.apply_to_registry(
                                &registry,
                                arena_reconcile
                                    .config
                                    .live_taker_fee
                                    .load(std::sync::atomic::Ordering::Relaxed),
                            );
                            let now_ms = std::time::SystemTime::now()
                                .duration_since(std::time::UNIX_EPOCH)
                                .unwrap_or_default()
                                .as_millis() as u64;
                            let arena_adj = execution_engine::reconciliation::reconcile_arena(&positions, &arena_reconcile, now_ms);
                            let pruned = registry.prune_terminated(now_ms.saturating_sub(600_000));
                            if !report.suspicious_active_orders.is_empty() || adopted > 0 || arena_adj > 0 {
                                telemetry_server::telemetry_log!(
                                    "🔄 [RECONCILIATION] {} | Adopted Reg: {}, Adj Arena: {}, Pruned Reg: {}",
                                    report.summary,
                                    adopted,
                                    arena_adj,
                                    pruned
                                );
                            }
                        }
                        Err(e) => {
                            telemetry_server::telemetry_log!("⚠️ [RECONCILIATION] Error consultando positionRisk: {}", e);
                        }
                    }
                }
            });
        }

        // B3.6 — EVALUADOR DEL BREAKER DE FEES: cada 5 min, contabilidad
        // REAL del exchange (/fapi/v1/income). Un símbolo con ≥3 cierres,
        // neto negativo y fees > bruto ganador queda suspendido 4h para
        // nuevas entradas (veto en la ruta de entrada). Detecta tanto el
        // churn sin edge (bruto plano, fees acumulando) como el anti-edge
        // (bruto negativo). Medición que lo motiva: SOL 7 trades 0 wins —
        // fees −$3.21 dominaron el −$4.44 neto del día.
        // B3.6b — VENTANA RODANTE 24h (antes: desde el arranque — un símbolo
        // que quemó fees ayer amanecía limpio tras cada reinicio) y
        // SUSPENSIONES PERSISTIDAS en data/fee_breaker.json: sobreviven
        // reinicios del motor. Una mañana mala ahora cuesta el día entero,
        // no un arranque.
        {
            let exec_brk = Arc::clone(&exec);
            let arena_brk = Arc::clone(&arena_real);
            rt_handle.spawn(async move {
                let now0 = std::time::SystemTime::now()
                    .duration_since(std::time::UNIX_EPOCH)
                    .unwrap_or_default()
                    .as_millis() as u64;
                // Restaurar suspensiones vivas del disco.
                if let Ok(content) = std::fs::read_to_string("data/fee_breaker.json") {
                    match serde_json::from_str::<serde_json::Value>(&content) {
                        Ok(v) if v.as_object().is_some() => {
                            let map = v.as_object().unwrap();
                            let restored = map
                                .iter()
                                .filter(|(_, x)| x.as_u64().unwrap_or(0) > now0)
                                .count();
                            for (sym, until) in map {
                                if let Some(u) = until.as_u64() {
                                    if u > now0 {
                                        suspend_symbol_until(sym, u);
                                    }
                                }
                            }
                            if restored > 0 {
                                telemetry_server::telemetry_log!(
                                    "🛑 [FEE-BREAKER] {restored} suspensión(es) restauradas del disco"
                                );
                            }
                        }
                        Ok(_) => {
                            telemetry_server::telemetry_log!(
                                "⚠️ [FEE-BREAKER] data/fee_breaker.json con esquema inesperado — se ignora (fail-safe: sin suspensiones)"
                            );
                        }
                        Err(e) => {
                            // B3.6b (auditoría): la corrupción ya no es
                            // silenciosa — se reporta; la próxima suspensión
                            // reescribe el archivo saneado.
                            telemetry_server::telemetry_log!(
                                "⚠️ [FEE-BREAKER] data/fee_breaker.json corrupto ({}) — se ignora y se reescribirá al próximo disparo",
                                e
                            );
                        }
                    }
                }
                let mut tick = tokio::time::interval(std::time::Duration::from_secs(300));
                tick.set_missed_tick_behavior(tokio::time::MissedTickBehavior::Skip);
                tick.tick().await; // primer tick inmediato: alinear al ciclo de 5 min
                loop {
                    tick.tick().await;
                    let executor = exec_brk.load_full();
                    if executor.is_kill_switch_active() {
                        continue;
                    }
                    // B3.6b — ventana RODANTE: la evidencia de fees no expira
                    // con el arranque del motor. S-5 (ESPECTRALIZACIÓN): la
                    // ventana escala con la τ dominante del SÍMBOLO — un
                    // símbolo de τ 5min es juzgado en su propia escala de
                    // evidencia (24×τ, clamp [1h, 48h]; τ~1h ⇒ 24h como
                    // antes), no en un día de reloj arbitrario. Paginado: el
                    // endpoint trae máx 1000 entradas.
                    let now_breaker = std::time::SystemTime::now()
                        .duration_since(std::time::UNIX_EPOCH)
                        .unwrap_or_default()
                        .as_millis() as u64;
                    // S-5: τ de la escala que el genoma opera (temporal_scale
                    // s∈[0,1] → τ log-lineal) — el breaker corre en su propio
                    // daemon sin acceso al espectro vivo del core; el genoma
                    // es la escala estable de referencia para la ventana.
                    let l_fast = quantum_arena::temporal_spectrum::TAU_ANCHOR_FAST_MS.ln();
                    let l_slow = quantum_arena::temporal_spectrum::TAU_ANCHOR_SLOW_MS.ln();
                    let s_gen = arena_brk
                        .config
                        .temporal_scale
                        .load(std::sync::atomic::Ordering::Relaxed)
                        .clamp(0.0, 1.0);
                    let tau_scale = (l_fast + s_gen * (l_slow - l_fast))
                        .exp()
                        .clamp(30_000.0, 43_200_000.0);
                    let window_ms = (24.0 * tau_scale).clamp(3_600_000.0, 48.0 * 3_600_000.0) as u64;
                    let window_start = now_breaker.saturating_sub(window_ms);
                    let Ok(entries) = executor
                        .fetch_income_paged(&[], window_start, 4)
                        .await
                    else {
                        continue;
                    };
                    #[derive(Default)]
                    struct Agg {
                        realized: f64,   // REALIZED_PNL + FUNDING_FEE (neto de dirección)
                        gross_pos: f64,  // suma de cierres GANADORES
                        fees: f64,       // |COMMISSION|
                        trades: usize,   // cierres con PnL ≠ 0
                    }
                    let mut per: std::collections::HashMap<String, Agg> =
                        std::collections::HashMap::new();
                    for e in &entries {
                        if e.symbol.is_empty() {
                            continue;
                        }
                        let a = per.entry(e.symbol.clone()).or_default();
                        match e.income_type.as_str() {
                            "REALIZED_PNL" => {
                                if e.income != 0.0 {
                                    a.trades += 1;
                                }
                                a.realized += e.income;
                                if e.income > 0.0 {
                                    a.gross_pos += e.income;
                                }
                            }
                            "FUNDING_FEE" => a.realized += e.income,
                            "COMMISSION" => a.fees += e.income.abs(),
                            _ => {}
                        }
                    }
                    for (sym, a) in per {
                        let net = a.realized - a.fees;
                        if a.trades >= 3 && net < 0.0 && a.fees > a.gross_pos {
                            // B3.6b (auditoría): la condición se re-evalúa cada
                            // 5min y extiende until_ms en +4h por diseño (una
                            // mañana mala cuesta el día). ANTES se logueaba y
                            // re-escribía el archivo EN CADA ciclo — spam cada
                            // 5min por símbolo suspendido. Ahora: el LOG sólo
                            // en el PRIMER evento (suspensión nueva o
                            // re-activación tras expirar); las extensiones
                            // silenciosas persisten igual.
                            let was_suspended = SYMBOL_SUSPENDED_UNTIL
                                .lock()
                                .map(|g| g.get(&sym).copied().unwrap_or(0))
                                .unwrap_or(0)
                                > now_breaker;
                            // S-5: suspensión proporcional a la ofensa —
                            // fees/gross mide cuánta fricción paga el símbolo
                            // por unidad de ganancia bruta: ratio 1.0 ⇒ 4h
                            // (histórico), ratio 3.0 ⇒ 12h, clamp [1h, 12h].
                            let offense = (a.fees / a.gross_pos.max(1e-9)).clamp(0.25, 3.0);
                            let suspend_hours = (offense * 4.0).clamp(1.0, 12.0) as u64;
                            let until_ms = now_breaker + suspend_hours * 3_600_000;
                            suspend_symbol_until(&sym, until_ms);
                            // B3.6b — persistir (atómico, purga expirados): la
                            // suspensión sobrevive reinicios del motor.
                            persist_fee_breaker(now_breaker);
                            if !was_suspended {
                                telemetry_server::telemetry_log!(
                                    "🛑 [FEE-BREAKER] {} suspendido {}h para nuevas entradas (persistido): {} cierres, bruto_ganador ${:.2} < fees ${:.2}, neto ${:.2} — el símbolo no paga su fricción",
                                    sym, suspend_hours, a.trades, a.gross_pos, a.fees, net
                                );
                            }
                        }
                    }
                }
            });
        }

        // P-3b — POLLER DE OPEN INTEREST PER-SÍMBOLO: el dinero apalancado
        // DENTRO de cada moneda (/fapi/v1/openInterest, 120s, fuera del hot
        // path). Normalización log contra $100M de contratos abiertos por
        // símbolo (BTC global vive en omni[12] como proxy de régimen).
        // Alimenta el asiento Ente del Mercado vía coin.open_interest_norm.
        {
            let arena_oi = Arc::clone(&arena_real);
            let omni_for_funding = Arc::clone(&omni_state_hot);
            let syms_oi: Vec<(String, usize)> = symbols_clone
                .iter()
                .enumerate()
                .map(|(i, s)| (s.to_uppercase(), i))
                .collect();
            rt_handle.spawn(async move {
                let is_testnet = EnvManager::is_demo_env();
                let base_url = if is_testnet {
                    "https://testnet.binancefuture.com"
                } else {
                    "https://fapi.binance.com"
                };
                let client = reqwest::Client::builder()
                    .timeout(std::time::Duration::from_secs(8))
                    .build()
                    .unwrap_or_else(|_| reqwest::Client::new());
                let mut ticker = tokio::time::interval(std::time::Duration::from_secs(120));
                ticker.set_missed_tick_behavior(tokio::time::MissedTickBehavior::Skip);
                telemetry_server::telemetry_log!(
                    "👁️ [OI-POLLER] open interest per-símbolo activo ({} símbolos, 120s)",
                    syms_oi.len()
                );
                loop {
                    ticker.tick().await;
                    // QO-U1c — funding per-símbolo al registry (el poller de
                    // omni lo llena en funding_by_symbol cada 120s; aquí se
                    // publica por símbolo, fuera del hot path). El core lee
                    // `funding_rate` scoped; sin valor cae al global omni[11].
                    if let Ok(fmap) = omni_for_funding.funding_by_symbol.read() {
                        for (sym, _cid) in syms_oi.iter() {
                            if let Some(f) = fmap.get(sym) {
                                arena_oi.registry.set_scoped(sym, "funding_rate", *f);
                            }
                        }
                    }
                    // QO-U2 — sentimiento de masas al registry (contrarian):
                    // multitud muy long (L/S alto) o takers comprando
                    /// desesperadamente (ratio alto) = riesgo de squeeze.
                    if let Ok(lsmap) = omni_for_funding.ls_account_by_symbol.read() {
                        for (sym, _cid) in syms_oi.iter() {
                            if let Some(r) = lsmap.get(sym) {
                                arena_oi.registry.set_scoped(sym, "ls_account_ratio", *r);
                            }
                        }
                    }
                    if let Ok(tkmap) = omni_for_funding.taker_ratio_by_symbol.read() {
                        for (sym, _cid) in syms_oi.iter() {
                            if let Some(r) = tkmap.get(sym) {
                                arena_oi.registry.set_scoped(sym, "taker_ratio", *r);
                            }
                        }
                    }
                    for (sym, coin_id) in syms_oi.iter() {
                        if *coin_id >= arena_oi.coins.len() {
                            continue;
                        }
                        let url = format!("{}/fapi/v1/openInterest?symbol={}", base_url, sym);
                        if let Ok(res) = client.get(&url).send().await {
                            if let Ok(json) = res.json::<serde_json::Value>().await {
                                if let Some(oi_str) =
                                    json.get("openInterest").and_then(|v| v.as_str())
                                {
                                    if let Ok(oi) = oi_str.parse::<f64>() {
                                        if oi.is_finite() && oi > 0.0 {
                                            // CERT-M1-M02 — OI EN NOTIONAL USD,
                                            // cross-comparable. El endpoint da
                                            // unidades de la BASE: con la vieja
                                            // normalización ln(oi)/ln(1e8), DOGE
                                            // (billones de unidades) saturaba a 1.0
                                            // permanente y BTC (~100k) leía ~0.63 —
                                            // una tabla de lookup por símbolo, no
                                            // una señal. Ahora: notional = oi ×
                                            // precio vivo del coin, escala log
                                            // [$100M … $10B] (ln 11.51→23.03) que
                                            // cubre el rango real de futuros perp.
                                            // El asiento Ente lee apalancamiento
                                            // REAL comparado entre símbolos.
                                            let px = arena_oi.coins[*coin_id]
                                                .current_price
                                                .load(std::sync::atomic::Ordering::Relaxed);
                                            let oi_usd = oi * px.max(0.0);
                                            let norm = if oi_usd > 0.0 {
                                                ((oi_usd.ln() - 11.5129_f64)
                                                    / (23.0259_f64 - 11.5129_f64))
                                                    .clamp(0.0, 1.0)
                                            } else {
                                                0.0
                                            };
                                            arena_oi.coins[*coin_id]
                                                .open_interest_norm
                                                .store(norm, std::sync::atomic::Ordering::Relaxed);
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
            });
        }

        // B1.3 — WATCHDOG DE POSICIÓN DESNUDA.
        // "Toda posición tiene TP/SL" pasa de intención a INVARIANTE auditable:
        // cada 60s (o a los 5s si un ALGO_UPDATE terminal marcó
        // protection_dirty) cruza positionRisk contra openAlgoOrders y hace
        // top-up quirúrgico del lado con shortfall. Cubre: pierna cancelada
        // manualmente, EXPIRED por GTE_GTC, REJECTED por margin check, y el
        // remanente de fills parciales posteriores al bracket.
        {
            let exec_wd = Arc::clone(&exec);
            let arena_wd = Arc::clone(&arena_real);            rt_handle.spawn(async move {
                telemetry_server::telemetry_log!(
                    "🐕 [PROTECTION-WATCHDOG] Vigilante de posición desnuda activo (60s nominal, 5s reactivo)"
                );
                let mut fast = tokio::time::interval(std::time::Duration::from_secs(5));
                fast.set_missed_tick_behavior(tokio::time::MissedTickBehavior::Skip);
                let mut ticks: u32 = 0;
                // B3.5 — ESCALADO ANTI-DESNUDEZ PERSISTENTE (caso peligroso
                // del roadmap: SL-REJECTED por margin-check en el disparo
                // solo generaba telemetría). Si el gap de protección de un
                // símbolo SOBREVIVE a 3 auditorías consecutivas, el
                // re-bracket falló 3 veces (margin-check persistente:
                // reintentar la MISMA orden vuelve a fallar) y el invariante
                // "toda posición con TP/SL" ya no es exigible vía brackets:
                // se cierra la posición a mercado — pérdida realizada y
                // ACOTADA en vez de riesgo desnudo ilimitado. Umbral 3
                // absorbe los estados transitorios (bracket en vuelo tras
                // una entrada, 1 auditoría).
                let mut naked_streak: std::collections::HashMap<String, u32> =
                    std::collections::HashMap::new();
                // B3.5b — snapshot de rechazos por símbolo entre auditorías.
                let mut naked_rejections: std::collections::HashMap<String, u64> =
                    std::collections::HashMap::new();
                const NAKED_ESCALATION_LIMIT: u32 = 3;
                loop {
                    fast.tick().await;
                    ticks = ticks.wrapping_add(1);
                    let dirty = quantum_arena::protection_health::is_dirty();
                    // B3.5b (auditoría) — generación de eventos: un ALGO_UPDATE
                    // terminal que llegue EN MEDIO de una auditoría se perdía —
                    // clear_dirty() al final borraba su bandera y la reacción
                    // caía al ciclo nominal de 60s (hasta 60s desnuda). Con el
                    // contador de eventos, sólo se limpia la bandera si NO
                    // llegó evento nuevo durante la auditoría.
                    let events_at_start = quantum_arena::protection_health::terminal_events_seen();
                    // Cadencia: reactiva si hay señal terminal; nominal cada
                    // 12 ticks de 5s (=60s, alineado con la reconciliación).
                    if !dirty && ticks % 12 != 0 {
                        continue;
                    }
                    let executor = exec_wd.load_full();
                    if executor.is_kill_switch_active() {
                        quantum_arena::protection_health::clear_dirty();
                        continue; // kill-switch: no colocar protecciones nuevas
                    }
                    let Ok(positions) = executor.fetch_position_risk().await else {
                        continue;
                    };
                    // B3.5b (auditoría) — símbolos con posición ABIERTA en esta
                    // auditoría: al salir del loop, el streak y el snapshot de
                    // rechazos de símbolos YA CERRADOS se purgan. Sin esto, un
                    // streak=2 sobrevivía al cierre de la posición (TP/manual)
                    // y una RE-ENTRADA horas después heredaba el conteo: 1
                    // rechazo nuevo → streak 3 → CIERRE DE EMERGENCIA de una
                    // posición sana cuyo bracket apenas estaba en vuelo.
                    let mut open_syms: std::collections::HashSet<String> =
                        std::collections::HashSet::new();
                    // B2.6 — PURGA DE PIERNAS HUÉRFANAS: piernas TP/SL cuya
                    // posición ya cerró disparan al vacío (REJECTED benigno
                    // que quema slots algo y ensucia el stream). Cancelarlas.
                    if let Ok(all_legs) = executor.fetch_all_open_algo_orders().await {
                        for leg in &all_legs {
                            // D-698 (DÉCIMA OLA · auditoría integral): `positionSide`
                            // vacío es DESCONOCIDO, no «no coincide». Con el
                            // predicado anterior, una pierna sin ese campo —el mismo
                            // que ya llegó vacío con `algoStatus` (B1.3-fix)— daba
                            // `side_open = false` para toda posición LARGA viva, y el
                            // watchdog purgaba su TP y su SL dejándola desnuda.
                            let side_unknown = leg.position_side.is_empty();
                            let side_open = positions.iter().any(|p| {
                                p.symbol == leg.symbol
                                    && p.position_amt.abs() > 0.0
                                    && (leg.position_side == "BOTH"
                                        || side_unknown
                                        || (p.position_amt > 0.0) == (leg.position_side == "LONG"))
                            });
                            if !side_open {
                                // D-698: se cancela por `algoId` y el fallo se
                                // reporta: una pierna que sobrevive a la purga
                                // dispara sobre la posición siguiente.
                                match executor
                                    .cancel_algo_order_ids(
                                        &leg.symbol,
                                        leg.algo_id,
                                        &leg.client_algo_id,
                                    )
                                    .await
                                {
                                    Ok(()) => telemetry_server::telemetry_log!(
                                        "🧹 [PROTECTION-WATCHDOG] Pierna huérfana purgada: {} {} algoId {} (posición ya cerrada)",
                                        leg.symbol, leg.order_type, leg.algo_id
                                    ),
                                    Err(e) => telemetry_engine::telemetry_err!(
                                        "🚨 [PROTECTION-WATCHDOG] Pierna huérfana VIVA: {} {} algoId {} no se pudo cancelar: {}",
                                        leg.symbol, leg.order_type, leg.algo_id, e
                                    ),
                                }
                            }
                        }
                    }
                    let mut naked_total = 0usize;
                    for p in positions
                        .iter()
                        .filter(|p| p.is_open() && p.position_amt.abs() > 0.0)
                    {
                        open_syms.insert(p.symbol.clone());
                        let is_long = p.position_amt > 0.0;
                        let qty = p.position_amt.abs();
                        // τ de entrada: la que recuerda el arena local si el
                        // símbolo está mapeado; si no, ancla rápida.
                        let tau = quantum_arena::symbol_registry::try_index(&p.symbol)
                            .and_then(|ci| {
                                arena_wd
                                    .coins
                                    .get(ci)
                                    .map(|c| c.positions.position.entry_tau_ms.load(Ordering::Relaxed))
                            })
                            .unwrap_or(0);
                        let prot = ensure_position_protected(
                            &executor,
                            &arena_wd,
                            &p.symbol,
                            is_long,
                            qty,
                            p.entry_price,
                            tau,
                            "WATCHDOG",
                        )
                        .await;
                        let (g_tp, g_sl) = (prot.tp_gap, prot.sl_gap);
                        if g_tp > 0.0 || g_sl > 0.0 {
                            naked_total += 1;
                            // B3.5b — el streak SOLO sube con evidencia
                            // positiva: rechazos NUEVOS del exchange desde la
                            // auditoría anterior. Un gap sin rechazos nuevos
                            // = fallo de red o estado transitorio: se loguea,
                            // no se cuenta (un blip de 3 ciclos ya no cierra
                            // posiciones sanas a mercado).
                            // HOST-024 — EXCEPCIÓN: gaps bajo minNotional sí
                            // cuentan SIN rechazos: el exchange jamás aceptará
                            // esa orden (evidencia determinística, no red) —
                            // la posición queda parcialmente protegida para
                            // siempre y el riesgo es real.
                            let rejections_now =
                                quantum_arena::protection_health::rejections_of(&p.symbol);
                            let rejections_prev =
                                naked_rejections.get(&p.symbol).copied().unwrap_or(0);
                            naked_rejections.insert(p.symbol.clone(), rejections_now);
                            if rejections_now <= rejections_prev && prot.min_notional_skips == 0 {
                                telemetry_server::telemetry_log!(
                                    "🐕 [WATCHDOG] {} con gap pero sin rechazos nuevos — no escala (¿red?)",
                                    p.symbol
                                );
                                continue;
                            }
                            let streak = naked_streak
                                .entry(p.symbol.clone())
                                .and_modify(|c| *c += 1)
                                .or_insert(1);
                            if *streak >= NAKED_ESCALATION_LIMIT {
                                // B3.5 — el bracket es inaplicable; el cierre
                                // es la única protección que queda. Protocolo
                                // X-009: cerrar PRIMERO con la qty REAL del
                                // exchange (reduce-only), purgar brackets
                                // después. Fallo NO resetea el streak: el
                                // siguiente ciclo reintenta el cierre.
                                telemetry_server::telemetry_log!(
                                    "🚨 [NAKED-ESCALATION] {} desnuda tras {} auditorías (re-bracket inaplicable) — CIERRE DE EMERGENCIA a mercado",
                                    p.symbol, streak
                                );
                                let close_sym = p.symbol.clone();
                                let close_long = is_long;
                                let exec_esc = exec_wd.load_full();
                                // Cantidad REAL re-fetcheada tras la decisión de
                                // cerrar (protocolo X-009). B3.5 (auditoría):
                                // (a) match de LADO — en hedge hay DOS registros
                                //     por símbolo; el find anterior podía tomar
                                //     el lado OPUESTO y cerrar con la qty
                                //     equivocada (-2022 eterno);
                                // (b) Err de red ≠ posición inexistente: el
                                //     `.ok()` anterior confundía un fallo de
                                //     fetch con "ya no existe" y RESETEABA el
                                //     streak — el ciclo siguiente empezaba de
                                //     cero. Ahora Err conserva el streak y
                                //     reintenta el cierre.
                                match exec_esc.fetch_position_risk().await {
                                    Err(fe) => {
                                        telemetry_server::telemetry_log!(
                                            "⚠️ [NAKED-ESCALATION] re-fetch de {} falló ({}) — streak conservado, reintento en el próximo ciclo",
                                            close_sym, fe
                                        );
                                    }
                                    Ok(ps) => {
                                        let real_qty = ps
                                            .iter()
                                            .find(|r| {
                                                r.symbol == close_sym
                                                    && r.position_amt.abs() > 0.0
                                                    && (r.position_amt > 0.0) == close_long
                                            })
                                            .map(|r| r.position_amt.abs());
                                        if let Some(rq) = real_qty {
                                            let close_res = match exec_esc
                                                .get_symbol_filter(&close_sym)
                                                .await
                                            {
                                                Ok(f) => {
                                                    exec_esc
                                                        .execute_reduce_only_market(
                                                            &close_sym, close_long, rq, f.step_size,
                                                        )
                                                        .await
                                                }
                                                Err(e) => Err(format!("filtro: {e}")),
                                            };
                                            match close_res {
                                                Ok(()) => {
                                                    let _ = exec_esc
                                                        .cancel_position_oco_orders(
                                                            &close_sym,
                                                            close_long,
                                                        )
                                                        .await;
                                                    naked_streak.remove(&close_sym);
                                                    // B3.19 — la emergencia también
                                                    // alimenta la contabilidad y el
                                                    // posterior de Kelly (antes sólo
                                                    // el income del fee-breaker la
                                                    // veía: cierres de emergencia
                                                    // invisibles al aprendizaje).
                                                    record_emergency_close(
                                                        &close_sym,
                                                        close_long,
                                                        rq,
                                                        &arena_wd,
                                                        "NAKED-ESC",
                                                    );
                                                    telemetry_server::telemetry_log!(
                                                        "🛑 [NAKED-ESCALATION] {} cerrada por emergencia — brackets purgados",
                                                        close_sym
                                                    );
                                                }
                                                Err(e) => {
                                                    telemetry_server::telemetry_log!(
                                                        "⚠️ [NAKED-ESCALATION] cierre de {} falló ({}) — streak conservado, reintento en el próximo ciclo",
                                                        close_sym, e
                                                    );
                                                }
                                            }
                                        } else {
                                            // La posición ya no existe (TP disparó o
                                            // cierre externo): purga de huérfanas la
                                            // limpiará en esta misma auditoría.
                                            naked_streak.remove(&close_sym);
                                        }
                                    }
                                }
                            }
                        } else {
                            naked_streak.remove(&p.symbol);
                            naked_rejections.remove(&p.symbol);
                        }
                    }
                    // B3.5b (auditoría) — purga de estado de símbolos SIN
                    // posición: streak/snapshot heredados por una re-entrada
                    // futura eran la receta del falso positivo de escalado.
                    naked_streak.retain(|k, _| open_syms.contains(k));
                    naked_rejections.retain(|k, _| open_syms.contains(k));
                    if dirty {
                        if naked_total == 0 {
                            telemetry_server::telemetry_log!(
                                "🐕 [PROTECTION-WATCHDOG] Auditoría reactiva: todas las posiciones cubiertas"
                            );
                        } else {
                            telemetry_server::telemetry_log!(
                                "⚠️ [PROTECTION-WATCHDOG] {} posición(es) con gap de protección residual",
                                naked_total
                            );
                        }
                        // B3.5b (auditoría) — sólo limpiar la bandera si NO
                        // llegó un evento terminal NUEVO durante esta
                        // auditoría (generación estable); si llegó, dejarla
                        // sucia para re-auditar en 5s, no en 60s.
                        if quantum_arena::protection_health::terminal_events_seen()
                            == events_at_start
                        {
                            quantum_arena::protection_health::clear_dirty();
                        }
                    }
                }
            });
        }

        // The PhaseOrchestrator is injected into the execution context
        let _msg_count: u64 = 0;
        let mut has_transitioned = false;
        let mut engine_real = god_engine_core::GodEngineCore::new(Arc::clone(&arena_real));

        // ── F5.1: ENVOLVENTE KELLY BAYESIANA ───────────────────────────────────
        // El leverage YA NO sale de hardcodes (10/5): emerge del posterior del
        // edge con guard de ruina. Sin evidencia ⇒ f=0 ⇒ NO se opera hasta
        // acumular historial (directriz: la matemática decide, no constantes).
        let mut risk_envelope = risk_engine::kelly_envelope::RiskEnvelope::new();
        let mut avg_win_abs: f64 = 0.0;
        let mut avg_loss_abs: f64 = 0.0;
        // HOST-005 — RESTAURAR la envolvente persistida: sin esto cada
        // reinicio borraba el posterior del edge (bootstrap perpetuo). Las
        // EMAs locales de payoff arrancan desde la memoria restaurada, no
        // desde 0 — continuidad de aprendizaje entre sesiones.
        if restore_kelly_envelope(&mut risk_envelope) {
            avg_win_abs = risk_envelope.avg_win;
            avg_loss_abs = risk_envelope.avg_loss;
            telemetry_server::telemetry_log!(
                "🧬 [KELLY] Envolvente restaurada de disco: n={:.0} trades (α={:.1}/β={:.1}), avg_win=${:.4}, avg_loss=${:.4} — sin bootstrap (HOST-005)",
                risk_envelope.posterior.n(),
                risk_envelope.posterior.alpha,
                risk_envelope.posterior.beta,
                risk_envelope.avg_win,
                risk_envelope.avg_loss
            );
        } else {
            telemetry_server::telemetry_log!(
                "🧬 [KELLY] Sin envolvente previa — posterior fresco de Jeffreys (HOST-005)"
            );
        }
        // B3.7 (auditoría) — DEDUP cierre-bracket vs cierre-core: el MISMO
        // trade económico puede llegar por dos caminos (fill de la pierna en
        // el user-stream + condición de salida del core en el evento
        // siguiente). Sin esta ventana simétrica por símbolo, Kelly,
        // totales y WR lo contaban DOS veces.
        // HOST-003 (informe decimocuarto) — la ventana de 120s TRAGABA
        // TRADES LEGÍTIMOS: la carrera real core-vs-bracket es de
        // MILISEGUNDOS (logs B3.7), no 2 minutos — un scalp legítimo del
        // mismo símbolo hasta 2 min después era silenciado como duplicado.
        // 5s cubre la carrera de la pierna vs el cierre del core; los
        // trades legítimos subsiguientes ya no se tragan.
        let mut last_bracket_close_ms: std::collections::HashMap<String, u64> =
            std::collections::HashMap::new();
        let mut last_real_core_close_ms: std::collections::HashMap<String, u64> =
            std::collections::HashMap::new();
        const CLOSE_DEDUP_WINDOW_MS: u64 = 5_000;

        engine_real.reality.mode = god_engine_core::reality_physics::EngineMode::HyperRealistic;
        engine_real.set_model_rx(rx_real);

        // --- PHASE 3 WARMUP INJECTION ---
        telemetry_server::telemetry_log!("📥 [PHASE 3] Inyectando historial REST K-lines para calentar SwingState...");
        for (_i, sym) in symbols_clone.iter().enumerate() {
            // F5.4: índice acotado — universo dinámico mayor que los slots del
            // motor ya no pánico (abort) sino skip documentado.
            if _i >= engine_real.feature_engines.len() {
                telemetry_server::telemetry_log!(
                    "⚠️ [WARMUP] {} fuera de rango de feature_engines ({} slots) — omitido",
                    sym,
                    engine_real.feature_engines.len()
                );
                continue;
            }
            if let Some(klines) = historical_klines.get(sym) {
                // X-017 (REHAB-6): OHLCV REAL del REST — antes: closes con
                // banda sintética ±0.05% y volumen fijo 10 ⇒ v_t/ATR plano,
                // stops des-calibrados por horas. Ahora el rango y volumen
                // históricos verdaderos calibran hurst/spectral/omni Y v_t.
                for k in klines.iter() {
                    let [o, h, l, c, v] = *k;
                    if c > 0.0 {
                        engine_real.feature_engines[_i].process_kline(o, h, l, c, v);
                    }
                }
                telemetry_server::telemetry_log!(
                    "   ✅ Inyectadas {} K-lines 1m OHLCV real para {}",
                    klines.len(),
                    sym
                );
            }
        }
        // --------------------------------

        let mut shadow_forest = evolution_engine::random_forest::ShadowForest::new(initial_capital, initial_genome.clone(), 10);

        // E-03 — DriftAuditor CONECTADO: mide divergencia real-vs-shadow
        // en cada cierre. Si el drift acumulado excede el umbral, el
        // circuit breaker debe dispararse (exactamente el modo de fallo
        // backtest→live que este auditor existe para detectar).
        let drift_auditor = audit_engine::drift_auditor::DriftAuditor::new(0.05);
        // CERT-M4-C02: contadores para el AUTO-REARME del kill-switch por
        // drift — el drift es heurístico, no una condición permanente.
        let drift_kill_armed_at: std::sync::atomic::AtomicU32 = std::sync::atomic::AtomicU32::new(0);
        let drift_clean_closes: std::sync::atomic::AtomicU32 = std::sync::atomic::AtomicU32::new(0);

        // F4.8 — TrajectoryAuditor CONECTADO (era fantasma: solo sus tests lo
        // usaban): track por posición viva de la coherencia entre la
        // trayectoria esperada (magnitud/volumen/duración del genoma) y la
        // REALIDAD tick a tick. Detecta VolumeStarvation, MomentumReversal y
        // TimeExhaustion mientras la posición está abierta — no después, cuando
        // el PnL ya se perdió.
        let mut trajectory_auditor =
            audit_engine::trajectory_auditor::TrajectoryAuditor::new(30);

        let instant_baseline = std::time::Instant::now();
        let local_ms_now = std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).unwrap_or_default().as_millis() as i64;
        let epoch_baseline_ms = local_ms_now + ntp_offset_ms;

        let mut local_orderbooks: Vec<quantum_engine::orderbook::OrderBook> = symbols_clone
            .iter()
            .map(|s| quantum_engine::orderbook::OrderBook::new(s.clone()))
            .collect();
        // P-5 (PREDICTORES / ENTES): tracker de volumen institucional por
        // símbolo — z-score de burst sobre el flujo @trade REAL (la clase
        // estaba exportada con CERO call sites). `whale_burst_z` al registry.
        let mut whale_trackers: Vec<feature_engine::InstitutionalVolumeTracker> = symbols_clone
            .iter()
            .map(|_| feature_engine::InstitutionalVolumeTracker::default())
            .collect();
        // P-6 — detector de spoofing por símbolo: evaporación de muros L2
        // (los 5 niveles del stream alimentan evaluate_wall_decay).
        let mut spoof_detectors: Vec<feature_engine::SpoofingDetector> = symbols_clone
            .iter()
            .map(|_| feature_engine::SpoofingDetector::default())
            .collect();
        let mut msg_count: u64 = 0;
        // D-610: guardia de secuencia del libro, un estado por símbolo del
        // universo (misma indexación que `local_orderbooks`).
        let mut book_seq_guard = parsers::BookSequenceGuard::new(local_orderbooks.len());
        let mut consecutive_slow_ticks = 0;

        // FASE 6 (EVENT LOOP)
        god_engine_core::bootloader::SystemDiagnostics::execute_phase_6_hft();

        while let Ok(mut msg_bytes) = rx_events.recv() {
            // CERT-M4-H02: el centinela de shutdown despierta el recv; el
            // flag ordena el drenaje (el mensaje mismo se descarta).
            if shutdown_flag.load(std::sync::atomic::Ordering::SeqCst) {
                break;
            }
            let start = Instant::now();

            let is_trade = memchr::memmem::find(&msg_bytes, b"\"e\":\"trade\"").is_some();
            let is_kline = memchr::memmem::find(&msg_bytes, b"\"e\":\"kline\"").is_some();
            let is_depth = memchr::memmem::find(&msg_bytes, b"\"e\":\"depthUpdate\"").is_some();
            // P-4: liquidaciones del mercado completo (stream de arreglo:
            // [{e:forceOrder, o:{s,S,q,p,...}}, ...]). Severidad por evento
            // = |precio×qty| log-normalizado; el tensor dark_alpha decae.
            // Estos mensajes son RAROS (docenas/hora) — un parse JSON
            // ligero por evento no toca el hot path de trade/depth.
            let is_force_order = memchr::memmem::find(&msg_bytes, b"forceOrder").is_some();
            if is_force_order && !is_trade && !is_kline && !is_depth {
                if let Ok(v) = serde_json::from_slice::<serde_json::Value>(&msg_bytes) {
                    let items: Vec<&serde_json::Value> = v
                        .as_array()
                        .map(|a| a.iter().collect())
                        .unwrap_or_else(|| vec![&v]);
                    for it in items {
                        let o = it.get("o").unwrap_or(it);
                        let p = o.get("p").and_then(|x| x.as_str()).and_then(|x| x.parse::<f64>().ok());
                        let q = o.get("q").and_then(|x| x.as_str()).and_then(|x| x.parse::<f64>().ok());
                        if let (Some(p), Some(q)) = (p, q) {
                            let notional = (p * q).abs();
                            god_engine_core::liquidation_feed::bump(
                                god_engine_core::liquidation_feed::severity_from_notional(notional),
                            );
                        }
                    }
                }
                msg_count += 1;
                continue;
            }
            let is_reconnect = msg_bytes == b"[SYSTEM:RECONNECT]";

            if is_reconnect {
                telemetry_server::telemetry_log!("🧹 [AUTO-HEALING] Reconnect signal received. Purging Quantum Engine state to prevent time-glitches...");
                engine_real.reset_engines();
                for ob in local_orderbooks.iter_mut() {
                    ob.clear();
                }
                // CERT-M1-H02: resetear el guard de secuencia del libro —
                // sin esto, cada símbolo descartaba hasta 50 mensajes depth
                // consecutivos tras la reconexión (>1300 actualizaciones de
                // libro perdidas por reconnect con 26+ símbolos).
                book_seq_guard = parsers::BookSequenceGuard::new(local_orderbooks.len());
                telemetry_server::telemetry_log!("✅ [AUTO-HEALING] All AI Engines flushed + book sequence guard reset. Entering Warmup Phase (50 ticks).");

                let rx_rest = Arc::clone(&exec);
                rt_handle.spawn(async move {
                    telemetry_server::telemetry_log!("🔄 [REST-SYNC] Fetching truth from Binance API...");
                    if let Ok(positions) = rx_rest.load().fetch_open_positions().await {
                        telemetry_server::telemetry_log!("✅ [REST-SYNC] Binance reports {} active open positions.", positions.len());
                    }
                });
                continue;
            }

            let mut parsed_sym_opt = None;
            let mut current_price = 0.0;
            let mut qty = 0.0;
            let mut is_kline_closed = false;
            let mut event_time = 0i64;
            let mut depth_obi = 0.0;
            let mut depth_micro_div = 0.0;
            let mut dbp = 0.0;
            let mut dap = 0.0;
            let mut dbq = 0.0;
            let mut daq = 0.0;

            // F5.4 — FIX UB: from_utf8_unchecked sobre bytes de RED era UB
            // instantáneo ante payload corrupto. Los parsers son zero-copy
            // in-place (necesitan &mut str): VALIDAR UTF-8 primero y solo
            // entonces mutar — la mutación de los parsers preserva ASCII
            // (invariante documentada), la validez se mantiene. Inválido: fuera.
            if std::str::from_utf8(&msg_bytes).is_err() {
                msg_count += 1;
                continue;
            }
            let msg_str = unsafe { std::str::from_utf8_unchecked_mut(&mut msg_bytes) };

            let mut is_buyer_maker = false;
            if is_trade {
                if let Some((e, _, p, q, m, sym)) = parsers::parse_binance_trade(msg_str) {
                    event_time = e;
                    parsed_sym_opt = Some(sym);
                    current_price = p;
                    qty = q;
                    is_buyer_maker = m;
                }
            } else if is_kline {
                if let Some((e, sym, _, _, _, p, v, c)) = parsers::parse_binance_kline(msg_str) {
                    event_time = e;
                    parsed_sym_opt = Some(sym);
                    current_price = p;
                    qty = v;
                    is_kline_closed = c;
                }
            } else if is_depth {
                // P-6 — UN solo parse con TODOS los niveles (@depth5): el
                // best se deriva del nivel 0 y los muros máximos alimentan
                // el SpoofingDetector por símbolo (mismo costo: un parse
                // simd_json, arrays del caller, zero-alloc).
                let mut lvl_bids = [(0.0f64, 0.0f64); 5];
                let mut lvl_asks = [(0.0f64, 0.0f64); 5];
                if let Some((e, sym, update_id, n_lv)) =
                    parsers::parse_binance_depth5_levels(msg_str, &mut lvl_bids, &mut lvl_asks)
                {
                    let (bp, bq) = lvl_bids[0];
                    let (ap, aq) = lvl_asks[0];
                    // D-610 (DÉCIMA OLA): el `u` del libro se extraía y se descartaba.
                    // Un mensaje no posterior al último aceptado —rancio tras una
                    // reconexión, reordenado o duplicado— ya no actualiza el libro
                    // local ni alimenta OBI, OFI y microprecio.
                    if let Some(sym_id) = symbol_to_id.get(sym).copied() {
                        match book_seq_guard.check(sym_id, update_id) {
                            parsers::SeqVerdict::Accept => {}
                            parsers::SeqVerdict::Stale => {
                                if book_seq_guard.dropped.is_power_of_two() {
                                    telemetry_engine::telemetry_err!(
                                        "⚠️ [D-610 LIBRO] {} update_id {} no posterior al último aceptado: mensaje rancio, duplicado o fuera de orden descartado ({} descartes).",
                                        sym, update_id, book_seq_guard.dropped
                                    );
                                }
                                msg_count += 1;
                                continue;
                            }
                            parsers::SeqVerdict::Resync => {
                                telemetry_engine::telemetry_err!(
                                    "🔄 [D-610 LIBRO] {} secuencia resincronizada en update_id {} tras {} descartes consecutivos (reinicio de secuencia del exchange).",
                                    sym, update_id, parsers::BookSequenceGuard::RESYNC_AFTER
                                );
                            }
                        }
                    }
                    event_time = e;
                    parsed_sym_opt = Some(sym);
                    dbp = bp; dap = ap; dbq = bq; daq = aq;
                    current_price = (bp + ap) * 0.5;
                    qty = (bq + aq) * 0.5;
                    if let Some(sym_id) = symbol_to_id.get(sym).copied() {
                        if let Some(ob) = local_orderbooks.get_mut(sym_id) {
                            ob.update_bid(bp, bq);
                            ob.update_ask(ap, aq);
                        }
                    }
                    let total_q = bq + aq;
                    if total_q > 0.0 {
                        depth_obi = (bq - aq) / total_q;
                        let microprice = (bp * aq + ap * bq) / total_q;
                        let midprice = (bp + ap) / 2.0;
                        depth_micro_div = if midprice > 0.0 { (microprice - midprice) / midprice } else { 0.0 };
                    }
                    // P-6 — ENTE SPOOFING: muros máximos por lado (notional
                    // del peor nivel lleno) → detector de evaporación por
                    // símbolo; score publicado al registry. Fuera del camino
                    // de decisión: sólo computa y publica.
                    if n_lv >= 2 {
                        if let Some(sym_id) = symbol_to_id.get(sym).copied() {
                            if sym_id < spoof_detectors.len() {
                                let mut max_bid_wall = 0.0f64;
                                let mut max_ask_wall = 0.0f64;
                                for &(p, q) in lvl_bids.iter().take(n_lv) {
                                    max_bid_wall = max_bid_wall.max(p * q);
                                }
                                for &(p, q) in lvl_asks.iter().take(n_lv) {
                                    max_ask_wall = max_ask_wall.max(p * q);
                                }
                                let score = spoof_detectors[sym_id].evaluate_wall_decay(
                                    max_bid_wall,
                                    max_ask_wall,
                                    e as u64,
                                );
                                if score.is_finite() && score > 0.05 {
                                    engine_real.arena.registry.set_scoped(
                                        &sym.to_uppercase(),
                                        "spoof_score",
                                        score.clamp(0.0, 1.0),
                                    );
                                }
                            }
                        }
                    }
                }
            }

            // FASE 23: QUANTUM LATENCY KILL-SWITCH (Optimized via TSC)
            let now_ms = epoch_baseline_ms + instant_baseline.elapsed().as_millis() as i64;
            let latency_ms = now_ms - event_time;
            let mut latency_panic = false;

            // X-010 (REHAB-4): WRITER de latencia en el feed real. La métrica
            // arena.last_ws_latency_ms estaba SIEMPRE en 0 (sus writers vivían
            // solo en código muerto) ⇒ el interlock del core y la rama de
            // latencia del sistema inmune jamás disparaban — ni ante muerte
            // silenciosa del feed. Ahora: cada evento vivo escribe la latencia
            // real y LIMPIA el flag de stall del watchdog (puente global).
            if event_time > 0 {
                engine_real
                    .arena
                    .last_ws_latency_ms
                    .store(latency_ms.max(0) as u64, Ordering::Relaxed);
                quantum_arena::feed_health::clear();
            }

            let panic_threshold = engine_real.arena.config.latency_ms_panic_threshold.load(std::sync::atomic::Ordering::Relaxed) as i64;

            // Phase 4: Synthetic Volatility Kill Switch
            if event_time > 0 && latency_ms > panic_threshold {
                consecutive_slow_ticks += 1;
            } else if event_time > 0 {
                consecutive_slow_ticks = 0;
            }

            if consecutive_slow_ticks >= 10 {
                latency_panic = true;
                if consecutive_slow_ticks == 10 {
                    telemetry_server::telemetry_log!("🚨 [KILL_SWITCH] Latencia > {}ms detectada por 10 ticks. Volatilidad Sintética activada. SCALP DESACTIVADO.", panic_threshold);
                }
            }

            if event_time > 0 && latency_ms > panic_threshold {
                latency_panic = true;
                telemetry_server::telemetry_log!("⚠️ [LATENCY_PANIC] Delta = {}ms (>{panic_threshold}ms limit). Skiping O(1) Scalp execution.", latency_ms);
            }

            if let Some(parsed_sym) = parsed_sym_opt {
                // CERT-M1-C01: symbol_to_id está CONGELADO al arranque —
                // cuando el universe manager rota símbolos, los NUEVOS
                // llegan por el WS pero el HashMap no los conoce y son
                // silenciosamente descartados (universo fantasma en el
                // CONSUMIDOR). Ahora: primero el HashMap (rápido), y si
                // no está, el REGISTRY DINÁMICO (que SÍ se actualiza en
                // vivo por update_registry). Si el registry lo conoce,
                // procesarlo; si no, descartar como antes.
                let coin_id = match symbol_to_id.get(parsed_sym).copied() {
                    Some(id) => id,
                    None => {
                        // Fallback dinámico: el registry se actualiza al rotar
                        match quantum_arena::symbol_registry::try_index(parsed_sym) {
                            Some(id) if id < engine_real.arena.coins.len() => {
                                id
                            }
                            _ => {
                                msg_count += 1;
                                continue;
                            }
                        }
                    }
                };

                // F5.4 — GUARD DE ORIGEN: acota TODOS los índices aguas abajo
                // (coins[coin_id], feature_engines[coin_id] en core y aquí).
                // Antes: símbolo fuera del universo ⇒ índice pánico ⇒ abort
                // del proceso con posiciones abiertas.
                if coin_id >= engine_real.arena.coins.len() {
                    msg_count += 1;
                    continue;
                }

                // L-1: Ingesta de Microestructura Física en la Arena Viva (CVD y Muros L2)
                // Corrige la ceguera de volumen agresivo y profundidad del libro en vivo (Causa Forense #D117).
                // D-708: el flujo agregado (CVD) lo actualiza el núcleo dentro de
                // `process_event` —misma fuente para vivo, forense y replay—;
                // llamarlo también aquí lo contaría DOS veces. Al host le queda
                // lo que es sólo del host.
                if is_trade {
                    // P-5 — ENTE BALLENA: z-score de burst sobre el volumen
                    // del trade real; publicado al registry por símbolo.
                    if coin_id < whale_trackers.len() {
                        let notional = qty * current_price;
                        if notional.is_finite() && notional > 0.0 {
                            let (z, _accel, is_burst) = whale_trackers[coin_id].update(notional);
                            if is_burst {
                                let sym_scoped = symbol_to_id
                                    .iter()
                                    .find(|(_, &v)| v == coin_id)
                                    .map(|(k, _)| k.to_uppercase())
                                    .unwrap_or_default();
                                engine_real.arena.registry.set_scoped(
                                    &sym_scoped,
                                    "whale_burst_z",
                                    z.clamp(0.0, 10.0),
                                );
                            }
                        }
                    }
                } else if is_depth {
                    engine_real.arena.update_l2_depth(coin_id, dbq, daq);
                }

                // --- SANITY CHECKS (DATA INTEGRITY & NORMALIZATION) ---
                if current_price <= 0.0 || qty < 0.0 || current_price.is_nan() || qty.is_nan() {
                    continue; // Drop corrupt data
                }

                // --- OUTLIER REJECTION (> 15% FLASH CRASH FILTER) ---
                // X-039 (REHAB-6): snapshot a buffer ESTÁTICO — el Vec por
                // tick era heap alloc/free en el hilo TIME_CRITICAL.
                let mut tick_buf = [quantum_arena::state::CompactTick::default(); 1];
                let n_recent = engine_real.arena.coins[coin_id]
                    .tick_ring
                    .snapshot_recent_into(1, &mut tick_buf);
                if n_recent > 0 {
                    let prev_price = tick_buf[0].bid_price;
                    if prev_price > 0.0 {
                        let jump_pct = (current_price - prev_price).abs() / prev_price;
                        let max_jump = engine_real.arena.config.flash_crash_jump_pct.load(std::sync::atomic::Ordering::Relaxed);
                        if jump_pct > max_jump {
                            telemetry_server::telemetry_log!("⚠️ [DATA INTEGRITY] Dropping anomalous tick for {}! Jump: {:.2}%", parsed_sym, jump_pct * 100.0);
                            continue;
                        }
                    }
                }
                // --- 1. DELEGATE TO UNIFIED GOD ENGINE CORE ---
                // F4.1: features omni REALES (macro FRED/PAXG + sentiment vivos).
                // Antes: &[0.0; 54] — la NN swing evaluaba ceros en producción.
                let omni_features_hot = omni_state_hot.get_features();
                // D-707 (DÉCIMA OLA · auditoría integral): EL LIBRO NO DESAPARECE
                // ENTRE EVENTOS DE DEPTH.
                //
                // `dbq`/`daq` sólo se rellenan en un evento @depth5; en un trade o
                // en una vela llegaban en 0, y el núcleo fabricaba cantidades
                // SIMÉTRICAS a partir del volumen del trade, de modo que
                // `obi_val = 0` exactamente y `book_absent` era cierto SIEMPRE en
                // esos eventos: el bot vivo corría el generador de señales «sin
                // libro» —el que se escribió para el backtest trade-only, con
                // confianzas literales— mientras el forense, que sí recibe
                // cantidades por tick, corría el otro. Era una divergencia
                // backtest↔producción en la rama de decisión, no en un parámetro.
                //
                // El último libro conocido vive en el arena (`update_l2_depth` lo
                // escribe en cada @depth5). En un evento sin libro propio se usa
                // ése: el estado del libro es del mercado, no del tipo de evento.
                let (eff_dbq, eff_daq) = if is_depth || (dbq > 0.0 && daq > 0.0) {
                    (dbq, daq)
                } else {
                    let c = &engine_real.arena.coins[coin_id];
                    (
                        c.l2_bid_wall.load(std::sync::atomic::Ordering::Relaxed),
                        c.l2_ask_wall.load(std::sync::atomic::Ordering::Relaxed),
                    )
                };
                let (mut new_order, closed_order) = engine_real.process_event(
                    coin_id, is_trade, is_kline_closed, is_depth,
                    current_price, qty, dbp, dap, eff_dbq, eff_daq,
                    depth_obi, depth_micro_div, event_time as u64, latency_panic, &omni_features_hot,
                    is_buyer_maker,
                );

                // L-0 + X-018 (REHAB-6): los 10 universos sombra con omni real
                // (D100) pero MUESTREADOS 1-in-10 — antes: 11 evaluaciones
                // completas por evento en el hilo TIME_CRITICAL (la sombra
                // costaba 10× el motor). El shadow-forest aprende umbrales
                // estadísticos: muestreo uniforme preserva la señal, divide
                // el costo por 10.
                if msg_count % 10 == 0 {
                    shadow_forest.broadcast_tick(
                        // D-707: los universos sombra ven el mismo libro que el
                        // motor real, o compararían dos mundos distintos.
                        coin_id, is_trade, is_kline_closed, is_depth,
                        current_price, qty, dbp, dap, eff_dbq, eff_daq,
                        depth_obi, depth_micro_div, event_time as u64,
                        &engine_real.arena,
                        &omni_features_hot,
                        is_buyer_maker,
                        latency_panic, // CERT-M8-H04: paridad de física con el motor real
                    );
                }

                // F4.8 — TRAYECTORIA VIVA: mientras la posición está abierta,
                // cada tick evalúa coherencia esperado-vs-real. Divergencia =
                // alerta ANTES de que el SL la cobre (VolumeStarvation /
                // MomentumReversal / TimeExhaustion).
                {
                    let ts_traj = event_time as u64;
                    for is_scalp in [true, false] {
                        let status = trajectory_auditor.evaluate_tick(
                            coin_id,
                            is_scalp,
                            current_price,
                            qty * current_price,
                            ts_traj,
                        );
                        if let audit_engine::trajectory_auditor::TrajectoryStatus::Divergent { reason, score } = status {
                            telemetry_engine::telemetry!(
                                "📉 [TRAYECTORIA] {} {} divergente ({:?}, score {:.2}) — la tesis se está rompiendo EN VIVO",
                                parsed_sym,
                                if is_scalp { "scalp" } else { "swing" },
                                reason,
                                score
                            );
                        }
                    }
                }

                // --- 2. EXECUTE ORDERS ---
                let current_phase: god_engine_core::orchestrator::SystemPhase;
                let is_trading_allowed: bool;
                let is_paper_trading: bool;
                {
                    let mut orch = orchestrator.write().unwrap_or_else(|e| e.into_inner());
                    current_phase = orch.on_tick();
                    is_trading_allowed = orch.is_trading_allowed();
                    is_paper_trading = orch.is_paper_trading();
                }

                let newly_transitioned = is_trading_allowed && !has_transitioned;
                if newly_transitioned {
                    has_transitioned = true;
                }

                if !is_trading_allowed {
                    if msg_count > 0 && msg_count.is_multiple_of(5000) {
                        telemetry_server::telemetry_log!("🔥 [ORCHESTRATOR] Syncing buffers... {} ticks (Fase: {:?}).", msg_count, current_phase);
                    }
                    // CERT-M6-H02: drenar cierres de bracket INCONDICIONALMENTE.
                    // Antes sólo se drenaban dentro del else (trading permitido) —
                    // durante warmup/vetos, los fills del exchange se acumulaban
                    // (cap 1024, overflow silencioso) y Kelly nunca los veía.
                    // FIX (auditoría 2026-09-19): este drenaje era un fetch_add
                    // sobre to_bits() — suma de representaciones de bits
                    // (bits(a)+bits(b) ≠ bits(a+b)) que corrompía el plano
                    // exchange de X-013 (valores astronómicos/NaN ⇒ el
                    // kill-switch quedaba ciego al plano real). Además era
                    // redundante: el mismo fill dispara ACCOUNT_UPDATE y
                    // on_capital ya guarda la verdad (wallet+unrealized).
                    // El plano exchange queda con ESCRITOR ÚNICO (on_capital),
                    // igual que el drenaje vivo B3.7 que nunca toca capital.
                    for bc in execution_engine::trade_accounting::drain_bracket_closes() {
                        let _ = bc;
                    }
                } else {
                    if newly_transitioned {
                        telemetry_server::telemetry_log!("✅ [WARMUP COMPLETE] System state synchronized & Darwin Approved. Transitioning to {:?}", current_phase);

                        let is_env_testnet = std::env::var("USE_TESTNET").unwrap_or_default().trim().to_lowercase() == "true";
                        let (target_key, target_secret, target_is_testnet) = if is_env_testnet {
                            (testnet_key.clone(), testnet_secret.clone(), true)
                        } else {
                            (mainnet_key.clone(), mainnet_secret.clone(), false)
                        };

                        // R3.2 / K-06: Hot-swap preservando el registry y la arena viva
                        let old_registry = exec.load().registry();
                        let mut mainnet_executor = execution_engine::executor::OrderExecutor::new_with_shared(
                            target_key,
                            target_secret,
                            target_is_testnet,
                            old_registry,
                            Some(Arc::clone(&arena_real)),
                        );
                        if is_env_testnet {
                            mainnet_executor.set_paper_trading(false);
                        } else {
                            mainnet_executor.set_paper_trading(is_paper_trading);
                        }
                        let new_exec_arc = Arc::new(mainnet_executor);

                        // D-177: Asegurar modo hedge en Mainnet antes de almacenar executor
                        // (spawn async — el closure del thread no es async)
                        if !target_is_testnet && !is_paper_trading {
                            let exec_for_hedge = Arc::clone(&new_exec_arc);
                            rt_handle.spawn(async move {
                                match exec_for_hedge.ensure_hedge_mode().await {
                                    Ok(true) => telemetry_server::telemetry_log!("🔀 [TRANSITION] Cuenta Mainnet migrada a modo HEDGE exitosamente"),
                                    Ok(false) => telemetry_server::telemetry_log!("🔀 [TRANSITION] Cuenta Mainnet ya se encuentra en modo HEDGE"),
                                    Err(e) => telemetry_server::telemetry_log!("🚨 [TRANSITION] No se pudo garantizar modo HEDGE en Mainnet: {}", e),
                                }
                            });
                        }

                        exec.store(Arc::clone(&new_exec_arc));

                        // D-178: Re-spawn UserDataStreamer con las credenciales y cliente de Mainnet
                        let mainnet_streamer = execution_engine::user_data_stream::UserDataStreamer::new(
                            new_exec_arc.client().clone(),
                            new_exec_arc.registry(),
                        )
                        .with_sink(std::sync::Arc::new(CapitalBridgeSink {
                            unified_capital: Arc::clone(&unified_capital),
                        }))
                        // D-702: el streamer de mainnet también necesita el arena
                        // para atribuir el PnL de los cierres por bracket.
                        .with_arena(Arc::clone(&arena_real))
                        .with_api_secret(new_exec_arc.api_secret());
                        // X-011: matar el streamer de la era demo ANTES de
                        // spawnear el de mainnet — su ACCOUNT_UPDATE de testnet
                        // pisaría el capital real en el plano compartido.
                        demo_streamer_abort.store(true, std::sync::atomic::Ordering::Release);
                        telemetry_server::telemetry_log!(
                            "🔌 [USER-DATA] Streamer de era demo marcado para apagado"
                        );
                        rt_handle.spawn(async move {
                            mainnet_streamer.start().await;
                        });
                        telemetry_server::telemetry_log!(
                            "🔌 [USER-DATA] Stream privado Mainnet spawned (fills en tiempo real + capital vivo)"
                        );

                        // Re-spawn NTP synchronizer con el nuevo cliente para mantener sincronizado el reloj en mainnet
                        // HOST-016 — el guard de generación aborta el loop de la era demo: sin esto quedaban DOS loops infinitos compitiendo por server_time_offset_ms.
                        spawn_ntp_synchronizer(
                            &rt_handle,
                            Arc::new(new_exec_arc.client().clone()),
                            Arc::clone(&arena_real),
                        );

                        let base_ws_url = if is_env_testnet { "wss://stream.binancefuture.com/stream" } else { "wss://fstream.binance.com/stream" };
                        loop_ws_url.store(Arc::new(format!("{}?streams={}", base_ws_url, loop_streams_str)));
                        let _ = tx_ws_control.try_send(());
                        let exec_clone = Arc::clone(&exec);
                        let db_tx_clone = db_tx.clone();
                        let arena_real_clone = Arc::clone(&engine_real.arena);
                        rt_handle.spawn(async move {
                            if let Ok(bal) = exec_clone.load().fetch_account_balance().await {
                                telemetry_server::telemetry_log!("🌍 [TRANSITION] Mainnet API Real Balance Extracted: ${:.4}", bal);
                                arena_real_clone.unified_capital.store(bal, Ordering::Relaxed);
                                arena_real_clone.config.base_capital.store(bal, Ordering::Relaxed);
                                let _ = db_tx_clone.send((bal, 0.0)).await;
                            }

                            if let Ok(real_positions) = exec_clone.load().fetch_open_positions().await {
                                telemetry_server::telemetry_log!("🔍 [TRUTH-SYNC] Reconciling Mainnet API Positions...");
                                for pos in real_positions {
                                    if pos.qty.abs() > 0.0 {
                                        telemetry_server::telemetry_log!("🚨 [GHOST-DETECTED] Binance has an open position on {}: {:.4} (Long: {}). Alerting for manual or auto reconciliation.", pos.symbol, pos.qty, pos.is_long);
                                    }
                                }
                            }
                        });
                    }

                    let unified_cap = f64::from_bits(unified_capital.load(Ordering::Relaxed));

                    // B3.7 — CIERRES POR BRACKET: contabilidad + aprendizaje.
                    // Los disparos TP/SL del exchange (la mayoría de los
                    // cierres) llegan por la cola de trade_accounting; aquí
                    // alimentan los mismos acumuladores y el posterior de
                    // Kelly que antes sólo veía los cierres del core.
                    for bc in execution_engine::trade_accounting::drain_bracket_closes() {
                        let net_bc = bc.pnl_gross - bc.fees;
                        // D-702: la condición es «se conoce el contexto de
                        // entrada», no «el PnL no es cero». Con la guarda
                        // anterior, un cierre exactamente en el precio de entrada
                        // quedaba fuera de la estadística sin dejar rastro.
                        if bc.entry_price > 0.0 {
                            // B3.7 (auditoría) — ¿el core ya contabilizó ESTE
                            // mismo cierre (condición de salida disparada casi
                            // simultáneamente al fill de la pierna)? Ventana
                            // simétrica por símbolo: sin esto, doble cuenta.
                            let core_ts = last_real_core_close_ms
                                .get(&bc.symbol)
                                .copied()
                                .unwrap_or(0);
                            if core_ts > 0
                                && bc.ts_ms.abs_diff(core_ts) <= CLOSE_DEDUP_WINDOW_MS
                            {
                                telemetry!(
                                    "♻️ [BRACKET CLOSE] {} ya contabilizado por cierre del core (Δ{}ms) — sin doble cuenta",
                                    bc.symbol,
                                    bc.ts_ms.abs_diff(core_ts)
                                );
                            } else {
                                if net_bc >= 0.0 {
                                    avg_win_abs = if avg_win_abs == 0.0 { net_bc.abs() } else { avg_win_abs * 0.95 + net_bc.abs() * 0.05 };
                                } else {
                                    avg_loss_abs = if avg_loss_abs == 0.0 { net_bc.abs() } else { avg_loss_abs * 0.95 + net_bc.abs() * 0.05 };
                                }
                                risk_envelope.record_trade(net_bc > 0.0, avg_win_abs.max(1e-9), -avg_loss_abs.max(1e-9));
                                // HOST-005 — persistir el posterior tras cada
                                // cierre limpio contabilizado: el próximo
                                // reinicio arranca con esta evidencia.
                                persist_kelly_envelope(&risk_envelope);
                                total_gross_pnl += bc.pnl_gross;
                                total_net_pnl += net_bc;
                                total_fees += bc.fees;
                                total_trades += 1;
                                if net_bc > 0.0 { total_wins += 1; }
                                last_bracket_close_ms.insert(bc.symbol.clone(), bc.ts_ms);
                            }
                        }
                        telemetry!(
                            "🎯 [BRACKET CLOSE] {} {} qty {:.6}: entry {:.6} → fill {:.6} (trigger {:.6}, slip {:.1}bps adversos) | bruto {:.4} | fees {:.4} | neto {:.4}{}",
                            bc.symbol,
                            bc.trigger,
                            bc.qty,
                            bc.entry_price,
                            bc.exit_price,
                            bc.stop_price,
                            bc.slippage_bps,
                            bc.pnl_gross,
                            bc.fees,
                            net_bc,
                            if bc.pnl_gross == 0.0 { " — sin contexto de entrada local (adoptada)" } else { "" }
                        );
                    }

                    if let Some((is_long, pnl, qty)) = closed_order {
                        // B3.14 — sólo contabiliza lo que EXISTIÓ en el
                        // exchange. Cierres de entradas vetadas/rechazadas
                        // (round-trips locales de papel, caso KOMA) no
                        // contaminan totales, WR ni el posterior de Kelly.
                        let close_was_real = engine_real
                            .arena
                            .coins
                            .get(coin_id)
                            .map(|c| {
                                c.positions
                                    .position
                                    .last_close_confirmed
                                    .load(Ordering::Relaxed)
                            })
                            .unwrap_or(false);
                        if !close_was_real {
                            telemetry_engine::telemetry!(
                                "📝 [PAPER CLOSE] {} {:+.4} — entrada nunca ejecutada en exchange; NO contabiliza",
                                parsed_sym, pnl
                            );
                            // HOST-004 (informe decimocuarto) — el reduce-only
                            // de respaldo usa la qty de este cierre de PAPEL:
                            // si el core re-abrió y la posición ACTUAL del
                            // arena está confirmada por el exchange (fill real
                            // o adopción), dispararlo cerraría la posición
                            // NUEVA legítima. El "papel" ya fue resuelto —
                            // el respaldo sólo corre contra una posición que
                            // NADIE confirmó.
                            let current_pos_confirmed = engine_real
                                .arena
                                .coins
                                .get(coin_id)
                                .map(|c| {
                                    c.positions
                                        .position
                                        .exchange_confirmed
                                        .load(Ordering::Relaxed)
                                })
                                .unwrap_or(false);
                            if current_pos_confirmed {
                                telemetry_engine::telemetry!(
                                    "📝 [PAPER CLOSE] {} respaldo OMITIDO: posición actual confirmada por exchange (re-entrada legítima vive) — HOST-004",
                                    parsed_sym
                                );
                            } else {
                                // El despacho de cierre igual corre: si algo
                                // coló en el exchange, X-008 lo aterriza.
                                let parsed_sym_str_paper = parsed_sym.to_string();
                                let exec_paper = Arc::clone(&exec);
                                let long_paper = is_long;
                                rt_handle.spawn(async move {
                                    if let Ok(f) = exec_paper
                                        .load()
                                        .get_symbol_filter(&parsed_sym_str_paper)
                                        .await
                                    {
                                        let _ = exec_paper
                                            .load()
                                            .execute_reduce_only_market(
                                                &parsed_sym_str_paper,
                                                long_paper,
                                                qty.abs(),
                                                f.step_size,
                                            )
                                            .await;
                                    }
                                });
                            }
                        }
                        // B3.14 — TODO el bloque de contabilidad/aprendizaje
                        // es exclusivo de cierres con entrada REAL.
                        if close_was_real {
                        // E-03: alimentar el DriftAuditor con cada cierre
                        // real (shadow aprox = pnl real; cuando el shadow
                        // forest esté plenamente vivo, comparará predicción
                        // vs resultado aquí).
                        // G-01 — DriftAuditor CONECTADO de verdad: compara
                        // el PnL real contra el esperado por el modelo (el
                        // ml_prob al entry predice victoria). Si el drift
                        // acumulado supera el umbral, alerta — es el detector
                        // del modo de fallo backtest→live.
                        let real_pnl_pct = if current_price > 0.0 && qty > 0.0 {
                            pnl / (qty * current_price).max(1e-8)
                        } else { 0.0 };
                        {
                            let ts_now = std::time::SystemTime::now()
                                .duration_since(std::time::UNIX_EPOCH)
                                .unwrap_or_default()
                                .as_millis() as u64;
                            let real_tr = audit_engine::drift_auditor::TradeResult {
                                symbol_id: coin_id,
                                is_long,
                                entry_price: 0.0, // no disponible en este scope
                                exit_price: current_price,
                                pnl_pct: real_pnl_pct,
                                timestamp_ms: ts_now,
                            };
                            // D-704 + C-11 (UNIÓN): UN SHADOW CONSTANTE NO ES
                            // UNA COMPARACIÓN. El shadow original fijado en 0
                            // hacía drift = −real: un cierre GANADOR grande
                            // armaba el kill-switch por haber ganado dinero.
                            // C-11 lo sustituyó por shadow = 0,95·real —
                            // sólo cruza el umbral si |real| > 100% del
                            // nocional (contabilidad podrida, p. ej. adoptada
                            // con entry roto): un CENTINELA útil, pero NO una
                            // expectativa — el modo de fallo que este auditor
                            // existe para cazar (bt predice +0,4% y el vivo
                            // entrega −0,4%) sigue invisible hasta que exista
                            // contraparte real (PnL del universo de control
                            // del ShadowForest, o ml_prediction con la
                            // geometría TP/SL comprometida — siguiente paso).
                            // Decisión de unión: el centinela SÍ corta (la
                            // contabilidad podrida no debe seguir operando),
                            // pero con AUTO-REARME tras 10 cierres limpios
                            // (CERT-M4-C02) — no el latch eterno original.
                            let shadow_tr = audit_engine::drift_auditor::TradeResult {
                                symbol_id: coin_id,
                                is_long,
                                entry_price: 0.0,
                                exit_price: current_price,
                                pnl_pct: real_pnl_pct * 0.95,
                                timestamp_ms: ts_now,
                            };
                            match drift_auditor.audit_execution(&real_tr, &shadow_tr) {
                                Err(_) => {                                    telemetry_engine::telemetry!(
                                        "🚨 [DRIFT] Divergencia excede umbral — kill-switch ARMADO (auto-rearme en 10 cierres limpios)"
                                    );
                                    engine_real
                                        .arena
                                        .kill_switch_active
                                        .store(true, Ordering::SeqCst);
                                    // CERT-M4-C02: contar el trigger para auto-rearme
                                    drift_kill_armed_at.fetch_add(1, Ordering::SeqCst);
                                }
                                Ok(_) => {
                                    // CERT-M4-C02: AUTO-REARME — si el drift auditor
                                    // reporta sano DESPUÉS de un kill por drift, y ya
                                    // pasaron ≥10 cierres limpios consecutivos, liberar.
                                    // El drift es HEURÍSTICO (real*0.95 vs real), no una
                                    // condición permanente como el immune latch.
                                    if drift_kill_armed_at.load(Ordering::SeqCst) > 0 {
                                        let clean = drift_clean_closes.fetch_add(1, Ordering::SeqCst);
                                        if clean >= 10 {
                                            drift_kill_armed_at.store(0, Ordering::SeqCst);
                                            drift_clean_closes.store(0, Ordering::SeqCst);
                                            engine_real
                                                .arena
                                                .kill_switch_active
                                                .store(false, Ordering::SeqCst);
                                            telemetry_engine::telemetry!(
                                                "✅ [DRIFT-REARM] 10 cierres limpios consecutivos — kill-switch LIBERADO"
                                            );
                                        }
                                    }
                                }
                            }
                        }

                        // F4.8: cierra el track de trayectoria — el score final
                        // resume cuán fiel fue la realidad a la tesis del entry.
                        {
                            let ts_exit = event_time as u64;
                            let s_scalp = trajectory_auditor.record_exit(coin_id, true, current_price, ts_exit);
                            let s_swing = trajectory_auditor.record_exit(coin_id, false, current_price, ts_exit);
                            if let Some(s) = s_scalp.or(s_swing) {
                                telemetry_engine::telemetry!(
                                    "🎯 [TRAYECTORIA CIERRE] {} score fidelidad {:.2} (1.0 = la realidad siguió la tesis)",
                                    parsed_sym, s
                                );
                            }
                        }
                        // B3.7 (auditoría) — DEDUP dirección inversa: si la
                        // pierna del bracket ya contabilizó este cierre (fill
                        // en el user-stream casi simultáneo a la condición de
                        // salida local), NO re-contabilizar: Kelly, totales y
                        // WR verían el mismo trade dos veces.
                        let ev_close_ms = event_time as u64;
                        let bracket_ts_dedup = last_bracket_close_ms
                            .get(parsed_sym)
                            .copied()
                            .unwrap_or(0);
                        if bracket_ts_dedup > 0
                            && ev_close_ms.abs_diff(bracket_ts_dedup) <= CLOSE_DEDUP_WINDOW_MS
                        {
                            telemetry_engine::telemetry!(
                                "♻️ [CONTINUOUS CORE] CLOSE de {} ya contabilizado por BRACKET CLOSE (Δ{}ms) — sin doble cuenta",
                                parsed_sym,
                                ev_close_ms.abs_diff(bracket_ts_dedup)
                            );
                        } else {
                        let live_maker_fee = engine_real.arena.config.live_maker_fee.load(Ordering::Relaxed);
                        let live_taker_fee = engine_real.arena.config.live_taker_fee.load(Ordering::Relaxed);
                        let fee = (qty * current_price) * (live_maker_fee + live_taker_fee);
                        let net = pnl; // pnl devuelto por el core ya es neto de comisiones
                        let gross = pnl + fee;
                        // F5.1: alimentar el posterior del edge con CADA cierre real.
                        if net >= 0.0 {
                            avg_win_abs = if avg_win_abs == 0.0 { net.abs() } else { avg_win_abs * 0.95 + net.abs() * 0.05 };
                        } else {
                            avg_loss_abs = if avg_loss_abs == 0.0 { net.abs() } else { avg_loss_abs * 0.95 + net.abs() * 0.05 };
                        }
                        risk_envelope.record_trade(net > 0.0, avg_win_abs.max(1e-9), -avg_loss_abs.max(1e-9));
                        // HOST-005 — persistir el posterior tras cada cierre
                        // limpio contabilizado (sin doble cuenta del dedup).
                        persist_kelly_envelope(&risk_envelope);
                        total_gross_pnl += gross;
                        total_net_pnl += net;
                        total_fees += fee;
                        total_trades += 1;
                        if net > 0.0 { total_wins += 1; }
                        last_real_core_close_ms.insert(parsed_sym.to_string(), ev_close_ms);

                        let roi_post_fees = if (qty * current_price) > 0.0 { (net / (qty * current_price)) * 100.0 } else { 0.0 };
                        telemetry!("🛑 [CONTINUOUS CORE] CLOSE HIT! Gross PnL: {:.4} | Net PnL: {:.4} | ROI Post-Fees: {:.4}%", gross, net, roi_post_fees);
                        let _ = db_tx.try_send((unified_cap, 0.0));
                        let _ = loop_telemetry_tx.send(telemetry_server::TelemetryEvent::CapitalUpdate(unified_cap));
                        let ml_prob = engine_real.arena.coins[coin_id].ml_prob.load(Ordering::Relaxed);
                        let _ = loop_telemetry_tx.send(telemetry_server::TelemetryEvent::TradeClosed {
                            coin_id,
                            trade_type: "CONTINUOUS".to_string(),
                            pnl: net,
                            roi_pct: roi_post_fees,
                            duration_ms: 0,
                            ml_prob,
                        });
                        } // fin dedup B3.7

                        // FIX #590 + X-008 (REHAB-3): despacho de cierre con
                        // protocolo CORRECTO — CERRAR PRIMERO, purgar brackets
                        // DESPUÉS. Antes se cancelaba el OCO primero: si el close
                        // fallaba, la posición quedaba naked (sin TP/SL) con el
                        // core ya cerrado localmente. Ahora la protección vive
                        // hasta que la posición está plana en el exchange.
                        let parsed_sym_str = parsed_sym.to_string();
                        let exec_clone = Arc::clone(&exec);
                        let is_long_close = is_long;
                        let arena_emerg = Arc::clone(&engine_real.arena);
                        rt_handle.spawn(async move {
                            let sym_filter = match exec_clone.load().get_symbol_filter(&parsed_sym_str).await {
                                Ok(f) => f,
                                Err(e) => {
                                    // D-631: sin filtro real no se envía un cierre con la
                                    // precisión de otro activo. Mismo protocolo que un
                                    // cierre fallido: los brackets siguen protegiendo y la
                                    // reconciliación toma el mando.
                                    telemetry_engine::telemetry_err!(
                                        "🚨 [X-008 CLOSE FALLIDO] {} sin filtro de precisión ({}): BRACKETS CONSERVADOS como protección. Reconciliación tomará el mando.",
                                        parsed_sym_str, e
                                    );
                                    return;
                                }
                            };
                            let close_res = exec_clone
                                .load()
                                .execute_reduce_only_market(&parsed_sym_str, is_long_close, qty, sym_filter.step_size)
                                .await;
                            let close_res = match close_res {
                                Ok(()) => Ok(()),
                                Err(e1) => {
                                    // Retry único: glitches transitorios de red.
                                    tokio::time::sleep(std::time::Duration::from_millis(150)).await;
                                    exec_clone
                                        .load()
                                        .execute_reduce_only_market(&parsed_sym_str, is_long_close, qty, sym_filter.step_size)
                                        .await
                                        .map_err(|e2| format!("{} / retry: {}", e1, e2))
                                }
                            };
                            match close_res {
                                Ok(()) => {
                                    // Posición plana en el exchange: purgar los
                                    // brackets huérfanos (ya sin subyacente).
                                    if let Err(e) = exec_clone.load().cancel_position_oco_orders(&parsed_sym_str, is_long_close).await {
                                        telemetry_engine::telemetry_err!("⚠️ [PURGE OCO] {} cerrada pero brackets sin purgar: {}", parsed_sym_str, e);
                                    } else {
                                        telemetry_engine::telemetry!("🛡️ [CLOSE SUCCESS] {} cerrada y brackets purgados.", parsed_sym_str);
                                    }
                                }
                                Err(es) => {
                                    // Close fallido: brackets NO se tocan — siguen
                                    // protegiendo la posición viva. El core ya cerró
                                    // local: el ciclo de reconciliación/inmune adoptará
                                    // o aplanará. Alerta crítica (no silencio).
                                    telemetry_engine::telemetry_err!(
                                        "🚨 [X-008 CLOSE FALLIDO] {} no cerró ({}): BRACKETS CONSERVADOS como protección. Reconciliación tomará el mando.",
                                        parsed_sym_str, es
                                    );
                                }
                            }
                        });
                        } // B3.14 — fin del bloque close_was_real
                    }

                    // BUG-617: Veto de nuevas entradas por pánico de memoria RAM (>85% en OS-Guardian)
                    let is_mem_panic = engine_real.arena.panic_memory_dump.load(Ordering::Relaxed);
                    if is_mem_panic && new_order.is_some() {
                        telemetry_engine::telemetry_err!("🚨 [OS-GUARDIAN] Asfixia de RAM (>85%). Nuevas aperturas bloqueadas temporalmente para evitar OOM.");
                        engine_real.rollback_position(coin_id);
                        new_order = None;
                    }

                    let mut order_tp_price = 0.0;
                    let mut order_sl_price = 0.0;
                    let mut is_high_confidence = false;
                    let mut exec_leverage = 1;

                    // F5.1 & BUG-598: Margen libre real = equity - margen total retenido en posiciones vivas
                    let total_margin_used: f64 = engine_real.arena.coins.iter().map(|c| {
                        if c.positions.position.is_open() {
                            c.positions.position.margin_used.load(Ordering::Relaxed)
                        } else { 0.0 }
                    }).sum();
                    let cap_now = (engine_real.arena.unified_capital.load(Ordering::Relaxed) - total_margin_used).max(0.0);

                    if let Some((is_long, entry_price, _qty, core_tp, core_sl)) = new_order {
                        // D-442: Usar la distancia de stop real de la orden (core_sl) para dimensionar la envolvente de riesgo
                        let stop_pct = if core_sl > 0.0 && entry_price > 0.0 {
                            ((entry_price - core_sl).abs() / entry_price).max(0.0015)
                        } else {
                            engine_real
                                .arena
                                .config
                                .scalp_sl_base
                                .load(Ordering::Relaxed)
                                .max(engine_real.feature_engines.get(coin_id).map(|fe| fe.get_atr_pct()).unwrap_or(0.0) * 1.5)
                                .max(0.0015)
                        };
                        // D-641 (completo): la envolvente deja de cambiar de golpe su
                        // nivel de confianza (z 0,85 → 1,64) y su contracción
                        // (k 10 → 50) en $50. Micro pleno a ≤3 operaciones mínimas,
                        // estándar a ≥10, transición continua entre ambos. A $13
                        // el resultado es idéntico al anterior (peso micro = 1).
                        let env_min_notional = 5.0;
                        let env_w =
                            risk_engine::capital_regime::micro_weight(cap_now, env_min_notional);
                        let env_z = risk_engine::capital_regime::lerp(1.64, 0.85, env_w);
                        let env_k = risk_engine::capital_regime::log_lerp(50.0, 10.0, env_w);
                        let (env_lev, operable) = risk_envelope
                            .max_leverage(cap_now, stop_pct, env_min_notional, env_z, env_k);
                        // D-116 (evolucionado por X-022/REHAB-4): la ENVOLVENTE
                        // es AUTORITATIVA — PERO con BOOTSTRAP EXPLORATORIO:
                        // sin evidencia (posterior.n() < 30), el sistema DEBE
                        // poder operar con riesgo mínimo (leverage 1) para
                        // GENERAR la evidencia que la envolvente necesita.
                        // Sin esto: deadlock (0 trades → 0 evidencia → 0 trades).
                        // Es epsilon-greedy estándar: exploración forzada inicial.
                        let envelope_n = risk_envelope.posterior.n();
                        let notional_ord = _qty.abs() * entry_price;
                        let pos_margin = engine_real.arena.coins[coin_id].positions.position.margin_used.load(Ordering::Relaxed);
                        let core_leverage = if pos_margin > 0.0 && notional_ord > 0.0 {
                            (notional_ord / pos_margin).round().clamp(1.0, 50.0) as u32
                        } else {
                            (5.05 / cap_now.max(1.0)).ceil().clamp(1.0, 20.0) as u32
                        };
                        if envelope_n < 30.0 {
                            // BOOTSTRAP: riesgo mínimo para acumular evidencia.
                            // La envolvente told "no" porque no sabe — dejamos
                            // que el sistema APRENDA con skin in the game mínimo.
                            exec_leverage = 1;
                        } else if operable {
                            // C-08 (INFORME 14): la envolvente bayesiana
                            // (LCB + shrinkage + guard de ruina) es ahora
                            // AUTORITATIVA en vivo — paridad C-08 con
                            // booktick_replay. Antes era un veto binario: el
                            // sizing real salía del core_leverage clampeado y
                            // la envolvente sólo lo tapaba con el techo env_lev
                            // — los tamaños que la evolución midió no eran los
                            // que producción ejecutaba. Ahora la fracción de
                            // Kelly bayesiana ESCALA el sizing del core
                            // (×20·f: f=5% de riesgo por trade lo deja intacto;
                            // evidencia débil lo contrae) y la envolvente sigue
                            // siendo el TECHO (min con env_lev). (z, k) son los
                            // MISMOS del régimen de capital que alimentaron
                            // max_leverage arriba — una sola cadena de decisión.
                            let kelly_frac = risk_envelope.risk_fraction(env_z, env_k);
                            let cap = env_lev.floor().clamp(1.0, 20.0) as u32;
                            // S-4 (ESPECTRALIZACIÓN): el ×20 codificaba "5% del
                            // margen en riesgo" INDEPENDIENTE del stop — un
                            // trade de banda lenta con SL de 1.5% arriesgaba
                            // 30% del margen. Ahora el presupuesto de riesgo
                            // (5% · kelly_frac) se divide por la DISTANCIA REAL
                            // del stop a τ de entrada: leverage = riesgo/SL.
                            // SL 0.25% (banda rápida) ⇒ ×20 como antes; SL
                            // 1.5% (banda lenta) ⇒ ×6.7 — misma $ en riesgo.
                            let tau_entry = engine_real.arena.coins[coin_id]
                                .positions
                                .position
                                .entry_tau_ms
                                .load(Ordering::Relaxed)
                                as f64;
                            let sl_frac = engine_real.arena.config.sl_at_tau(
                                if tau_entry > 0.0 { tau_entry } else { 30_000.0 },
                            );
                            let risk_budget = 0.05 * kelly_frac; // fracción del margen por trade
                            // P-1b — VOL-BRAKE: el predictor {SYM}_VOL encoge el
                            // presupuesto cuando la σ pronosticada supera ×1.25
                            // la del régimen en que el modelo se entrenó (base =
                            // init del modelo, unidades exactas). UNILATERAL:
                            // sólo encoge (piso ×0.4), nunca amplifica — primer
                            // despliegue del predictor, la calle es una sola.
                            let vol_fc = engine_real
                                .arena
                                .registry
                                .get_for_coin_or(coin_id, "vol_forecast_pct", 0.0);
                            let vol_base = engine_real
                                .arena
                                .registry
                                .get_for_coin_or(coin_id, "vol_forecast_base", 0.0);
                            let mut vol_brake = 1.0;
                            if vol_base > 1e-9 && vol_fc > 1e-9 {
                                let ratio = vol_fc / vol_base;
                                if ratio > 1.25 {
                                    vol_brake = ((1.25 / ratio).max(0.4)).min(1.0);
                                    telemetry_server::telemetry_log!(
                                        "🛑 [VOL-BRAKE] coin {} σ_pronosticada {:.4} = ×{:.2} la base {:.4} — riesgo ×{:.2}",
                                        coin_id, vol_fc, ratio, vol_base, vol_brake
                                    );
                                }
                            }
                            let lev_from_risk =
                                (risk_budget * vol_brake / sl_frac.max(1e-4)).clamp(1.0, 20.0);
                            exec_leverage =
                                (lev_from_risk as u32).min(cap).max(1);
                        } else {
                            exec_leverage = 0; // SIN ORDEN: la matemática dijo NO
                        }
                        let _ = tx_log_worker.try_send((true, is_long, coin_id));

                        let ml_prob = engine_real.arena.coins[coin_id].ml_prob.load(Ordering::Relaxed);
                        if ml_prob > 0.80 || ml_prob < 0.20 {
                            is_high_confidence = true;
                        }

                        // D-108: Sincronización 1:1 OCO — Consumir directamente los targets calculados por RiskEngine.
                        // X-030/X-016 (REHAB-1): el fallback YA NO es ±40/±30 bps
                        // fijos (arbitrariedad que ignoraba τ y ATR): reconstruye
                        // la CURVA de horizonte desde las vistas vivas del arena
                        // (ambas son derivados de la misma curva post X-003) y la
                        // evalúa en la τ DOMINANTE del espectro de este símbolo.
                        // Sin espectro aún ⇒ ancla rápida (comportamiento scalp).
                        if core_tp > 0.0 || core_sl > 0.0 {
                            order_tp_price = if core_tp > 0.0 { core_tp } else { order_tp_price };
                            order_sl_price = if core_sl > 0.0 { core_sl } else { order_sl_price };
                        } else {
                            let tau_eff = engine_real
                                .temporal_spectrum
                                .get(coin_id)
                                .filter(|s| s.dominant_tau_ms > 0.0)
                                .map(|s| s.dominant_tau_ms)
                                .unwrap_or(quantum_arena::temporal_spectrum::TAU_ANCHOR_FAST_MS);
                            let tp_frac = quantum_arena::temporal_spectrum::HorizonCurve::through_two_points(
                                quantum_arena::temporal_spectrum::TAU_ANCHOR_FAST_MS,
                                engine_real.arena.config.scalp_tp_base.load(Ordering::Relaxed),
                                quantum_arena::temporal_spectrum::TAU_ANCHOR_SLOW_MS,
                                engine_real.arena.config.swing_tp_base.load(Ordering::Relaxed),
                            ).eval(tau_eff);
                            let sl_frac = quantum_arena::temporal_spectrum::HorizonCurve::through_two_points(
                                quantum_arena::temporal_spectrum::TAU_ANCHOR_FAST_MS,
                                engine_real.arena.config.scalp_sl_base.load(Ordering::Relaxed),
                                quantum_arena::temporal_spectrum::TAU_ANCHOR_SLOW_MS,
                                engine_real.arena.config.swing_sl_base.load(Ordering::Relaxed),
                            ).eval(tau_eff);
                            // B3.2: el fallback de entrada también nace viable —
                            // pisos de fricción idénticos a genome_protection_prices.
                            // B3.19: + latency_slip (atr·lat/ref), como el gate.
                            let atr_pct_entry = engine_real
                                .arena
                                .coins
                                .get(coin_id)
                                .map(|c| {
                                    let px = c.current_price.load(Ordering::Relaxed).max(1e-12);
                                    c.current_atr.load(Ordering::Relaxed) / px
                                })
                                .filter(|a| a.is_finite() && *a > 0.0)
                                .unwrap_or(0.0);
                            let lat_ref = engine_real
                                .arena
                                .config
                                .latency_ms_panic_threshold
                                .load(Ordering::Relaxed)
                                .clamp(10.0, 5_000.0);
                            let lat_slip_entry = (atr_pct_entry
                                * (engine_real
                                    .arena
                                    .config
                                    .latency_penalty_ms
                                    .load(Ordering::Relaxed)
                                    .max(0.0)
                                    / lat_ref))
                            .clamp(0.0, 0.05);
                            let fee_rt_entry = 2.0 * engine_real.arena.config.live_taker_fee.load(Ordering::Relaxed)
                                + 2.0 * (engine_real.arena.config.base_slippage_floor.load(Ordering::Relaxed).max(0.00001) + lat_slip_entry);
                            let (sl_frac, tp_frac) =
                                quantum_arena::genome::SuperGenotype::friction_floors(fee_rt_entry, sl_frac, tp_frac);
                            order_tp_price = if is_long { entry_price * (1.0 + tp_frac) } else { entry_price * (1.0 - tp_frac) };
                            order_sl_price = if is_long { entry_price * (1.0 - sl_frac) } else { entry_price * (1.0 + sl_frac) };
                        }

                        // F4.8: nace el track de trayectoria — expectativas
                        // REALES del genoma (magnitud=distancia al TP del core,
                        // volumen=notional, duración=base_duration_ms evolucionable).
                        {
                            let expected_mag = ((core_tp - entry_price) / entry_price).abs();
                            let dur_ms = engine_real
                                .arena
                                .config
                                .base_duration_ms
                                .load(Ordering::Relaxed)
                                .max(1.0) as u64;
                            trajectory_auditor.record_entry(
                                coin_id,
                                is_long,
                                true,
                                entry_price,
                                event_time as u64,
                                expected_mag,
                                _qty.abs() * entry_price,
                                dur_ms,
                            );
                        }
                    }

                    if let Some((final_is_long, _entry_price, final_qty, _, _)) = new_order {
                        let parsed_sym_str = parsed_sym.to_string();
                        let exec_clone = Arc::clone(&exec);
                        let arena_clone = Arc::clone(&engine_real.arena);

                        // B3.29 — MAKER DESACTIVADO PARA MOMENTUM (adverse
                        // selection estructural). Medido en 39 entradas: 0%
                        // fills pasivos con ventanas de 15ms Y 400ms, con
                        // mid Y con bid/ask. Causa: las entradas por momentum
                        // compran cuando el precio YA se mueve a favor — una
                        // orden pasiva al bid sólo llena si el precio RETRO-
                        // CEDE (contradiciendo la señal). Es adverse selection
                        // clásico: maker fills = señal equivocada, taker fills
                        // = señal correcta. La ruta maker añade 400ms de
                        // latencia al 100% de las entradas para capturar ~0%
                        // de ahorro. DESACTIVADA hasta que existan señales
                        // mean-reversion que la justifiquen.
                        let force_maker = false;
                        // B3.28 — PRECIO PASIVO AL LIBRO VIVO, no al mid
                        // congelado. Con maker_price = mid del tick
                        // desencadenante, el post-only a 400ms después o
                        // cruza (→taker con RTT extra) o queda lejos del
                        // libro movido (0% pasivo medido en 38 entradas).
                        // Pasivo al BID (para long) o ASK (para short): si
                        // el libro no se movió, descansa en top of book y
                        // llena con el próximo agresor opuesto.
                        let maker_price = if dbp > 0.0 && dap > 0.0 && dbp <= dap {
                            if final_is_long { dbp } else { dap }
                        } else {
                            current_price
                        };
                        let iceberg_threshold = engine_real.arena.config.iceberg_volume_threshold.load(Ordering::Relaxed);
                        let iceberg_slices = engine_real.arena.config.iceberg_slice_count.load(Ordering::Relaxed).max(2.0);
                        let notional_volume = final_qty.abs() * current_price;
                        let final_qty = final_qty.abs();

                        let rollback_positions = move |arena: &Arc<quantum_arena::GlobalArena>| {
                            if coin_id < arena.coins.len() {
                                let coin = &arena.coins[coin_id];
                                if coin.positions.position.is_open() {
                                    let (_, _, _, margin_used, entry_fee) = coin.positions.position.close_with_fee();
                                    if margin_used > 0.0 {
                                        // MOD6/8-010 (sustituye X-027): resta
                                        // atómica fetch_sub — el RMW load→store
                                        // NO era atómico y perdía liberaciones
                                        // concurrentes (cierre core, reconciliación).
                                        // Si queda levemente negativo por drift, el
                                        // lector satura a 0 (never inflar free_margin).
                                        arena
                                            .used_margin
                                            .fetch_sub(margin_used, Ordering::Relaxed);
                                    }
                                    if entry_fee > 0.0 {
                                        arena.unified_capital.fetch_add(entry_fee, Ordering::Relaxed);
                                        // D-180: No sumar entry_fee a pnl_realized en rollback (nunca fue ganancia)
                                    }
                                }
                            }
                        };

                        rt_handle.spawn(async move {
                            if exec_leverage == 0 {
                                telemetry_engine::telemetry!(
                                    "🛡️ [ENVOLVENTE] Entrada bloqueada: evidencia insuficiente o capital no sostiene el riesgo mínimo (Kelly bayesiano). Ejecutando Rollback de estado."
                                );
                                rollback_positions(&arena_clone);
                                return;
                            }

                            // B3.6 — veto del breaker de fees: el símbolo
                            // demostró (contabilidad del exchange) que no
                            // paga su propia fricción. Posiciones vivas y
                            // sus brackets NO se tocan; sólo nuevas entradas.
                            if is_symbol_suspended(&parsed_sym_str) {
                                telemetry_engine::telemetry!(
                                    "🛑 [FEE-BREAKER] Entrada de {} vetada (símbolo suspendido — fees > bruto ganador). Rollback de estado.",
                                    parsed_sym_str
                                );
                                rollback_positions(&arena_clone);
                                return;
                            }

                            // Sincronización Binance Leverage con Core Sizing y protección -2019 (D-382)
                            // MOD6/8-010: lector saturado — ver GlobalArena::used_margin_saturated.
                            let used_margin = arena_clone.used_margin.load(Ordering::Relaxed).max(0.0);
                            let free_margin = (arena_clone.unified_capital.load(Ordering::Relaxed) - used_margin).max(0.0);
                            let required_margin = notional_volume / exec_leverage as f64;
                            let mut effective_leverage = exec_leverage;

                            if required_margin > free_margin * 0.85 && free_margin > 0.0 {
                                let needed_leverage = (notional_volume / (free_margin * 0.80)).ceil().clamp(1.0, 20.0) as u32;
                                if needed_leverage > effective_leverage {
                                    telemetry_engine::telemetry!(
                                        "⚡ [LEVERAGE-ADAPT] Ajustando apalancamiento para {} de {}x a {}x para evitar rechazo -2019 (notional: {:.2} USDT, margen libre: {:.2} USDT)",
                                        parsed_sym_str, effective_leverage, needed_leverage, notional_volume, free_margin
                                    );
                                    effective_leverage = needed_leverage;
                                }
                            }

                            let final_required_margin = notional_volume / effective_leverage as f64;
                            if final_required_margin > free_margin * 0.95 {
                                telemetry_engine::telemetry!(
                                    "🚨 [MARGIN-GUARD] Orden abortada para {}: margen requerido {:.2} USDT excede 95% del margen libre ({:.2} USDT). Ejecutando Rollback.",
                                    parsed_sym_str, final_required_margin, free_margin
                                );
                                rollback_positions(&arena_clone);
                                return;
                            }

                            if effective_leverage > 1 {
                                let _ = exec_clone.load().set_leverage(&parsed_sym_str, effective_leverage).await;
                            }
                            let sym_filter = match exec_clone.load().get_symbol_filter(&parsed_sym_str).await {
                                Ok(f) => f,
                                Err(e) => {
                                    // D-631: una entrada sin filtro real se aborta antes de
                                    // tocar el exchange y se revierte el estado local.
                                    telemetry_engine::telemetry_err!(
                                        "❌ [ENTRY] {} abortada: {} — Rollback de posición local.",
                                        parsed_sym_str, e
                                    );
                                    rollback_positions(&arena_clone);
                                    return;
                                }
                            };
                            let dyn_step_size = sym_filter.step_size;
                            let dyn_tick_size = sym_filter.tick_size;

                            let entry_result: Result<(), String>;
                            if force_maker {
                                let mut id_buf = [0u8; 32];
                                id_buf[0..3].copy_from_slice(b"mc_");
                                let micros = std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).unwrap().as_micros();
                                let mut itoa_buf = itoa::Buffer::new();
                                let micros_str = itoa_buf.format(micros);
                                id_buf[3..3 + micros_str.len()].copy_from_slice(micros_str.as_bytes());
                                let order_id = std::str::from_utf8(&id_buf[..3 + micros_str.len()]).unwrap_or("mc_0");
                                entry_result = exec_clone.load().execute_maker_chase(&parsed_sym_str, final_is_long, final_qty, maker_price, dyn_step_size, dyn_tick_size, order_id).await;
                            } else if notional_volume > iceberg_threshold {
                                let iceberg_qty = final_qty / iceberg_slices;
                                telemetry_engine::telemetry!("🧊 [ICEBERG ROUTER] Fragmentando orden institucional ({:.2} USDT) en pedazos de {:.2}...", notional_volume, iceberg_qty);
                                entry_result = exec_clone.load().execute_iceberg_limit(&parsed_sym_str, final_is_long, final_qty, maker_price, iceberg_qty, dyn_step_size, dyn_tick_size, "iceberg_01").await;
                            } else {
                                let side_tag = if final_is_long { "L" } else { "S" };
                                let mut client_id = String::with_capacity(36);
                                use std::fmt::Write as _;
                                // FIX -4015: Binance exige newClientOrderId < 36 chars.
                                // "CONT_L_" (7) + UUIDv7 simple (32) = 39 → RECHAZADO.
                                // Formato corto: "cL" + 32 chars UUID = 34 ✓
                                let _ = write!(&mut client_id, "c{}_{}", side_tag, uuid::Uuid::now_v7().simple());
                                entry_result = exec_clone.load().execute_raw_qty_with_client_id(&parsed_sym_str, final_is_long, final_qty, dyn_step_size, &client_id).await;
                            }

                            match entry_result {
                                Err(e) if e.starts_with("AMBIGUOUS")
                                    || e.starts_with("MAKER_CHASE_UNVERIFIED") =>
                                {
                                    // X-007 (REHAB-3): RECONCILE-THEN-ROLLBACK. Un
                                    // timeout de transporte NO es un rechazo: la orden
                                    // PUEDE existir en el exchange. Antes se hacía
                                    // rollback local a ciegas — posición viva en el
                                    // exchange con estado local "limpio" (fantasma).
                                    // Ahora: consultar la verdad del exchange PRIMERO.
                                    // B3.7-repro (auditoría): MAKER_CHASE_UNVERIFIED
                                    // es IGUAL de ambiguo — el post-only pudo llenar
                                    // (parcial o total) antes de que fallara la
                                    // consulta de estado; el rollback a ciegas deja
                                    // la posición real des-confirmada para siempre.
                                    telemetry_engine::telemetry!(
                                        "⏳ [ENTRY AMBIGUA] {} timeout/fill no verificable ({}) — reconciliando con el exchange ANTES de tocar estado local…",
                                        parsed_sym_str, e
                                    );
                                    let adopted = match exec_clone.load().fetch_open_positions().await {
                                        Ok(positions) => {
                                            let ghost = positions.iter().find(|p| p.symbol == parsed_sym_str);
                                            if let Some(pos) = ghost {
                                                telemetry_engine::telemetry!(
                                                    "👻 [GHOST DETECTADO] {} TIENE posición real en el exchange (qty {}) — ADOPTANDO en estado local (sin rollback).",
                                                    parsed_sym_str, pos.qty
                                                );
                                                true
                                            } else {
                                                telemetry_engine::telemetry!(
                                                    "✅ [RECONCILE] {} sin posición en el exchange — la orden jamás aterrizó. Rollback local seguro.",
                                                    parsed_sym_str
                                                );
                                                false
                                            }
                                        }
                                        Err(qe) => {
                                            // Ni siquiera podemos reconciliar: NO
                                            // rollback (podría ser fantasma). El
                                            // ciclo de reconciliación del arranque
                                            // y el inmune lo resolverán; alertamos.
                                            telemetry_engine::telemetry_err!(
                                                "🚨 [X-007] Reconciliación imposible ({}): estado local CONSERVADO por seguridad — verificar {} manualmente.",
                                                qe, parsed_sym_str
                                            );
                                            true
                                        }
                                    };
                                    if !adopted {
                                        rollback_positions(&arena_clone);
                                    } else {
                                        // B3.14 — AMBIGUOUS adoptada: la posición
                                        // es real en el exchange ⇒ confirmada.
                                        if let Some(c) = arena_clone.coins.get(coin_id) {
                                            c.positions
                                                .position
                                                .exchange_confirmed
                                                .store(true, Ordering::Relaxed);
                                        }
                                        // La rama Ok coloca el OCO aquí; la
                                        // adoptada NO lo hace — sin esto la
                                        // posición queda hasta 60s sin bracket
                                        // (ciclo nominal del watchdog). El
                                        // dirty fuerza auditoría en 5s.
                                        quantum_arena::protection_health::mark_dirty();
                                    }
                                }
                                Err(e) => {
                                    telemetry_engine::telemetry!(
                                        "❌ [ENTRY] {} rechazada: {} — Ejecutando Rollback de posición local.",
                                        parsed_sym_str, e
                                    );
                                    rollback_positions(&arena_clone);
                                }
                                Ok(()) => {
                                    // B3.14 — la entrada EXISTE en el exchange:
                                    // los cierres de esta posición contabilizan
                                    // (los vetados/rechazados son papel).
                                    if let Some(c) = arena_clone.coins.get(coin_id) {
                                        c.positions
                                            .position
                                            .exchange_confirmed
                                            .store(true, Ordering::Relaxed);
                                    }
                                    if order_tp_price > 0.0 && order_sl_price > 0.0 {
                                        let tag = if is_high_confidence { "🎯 [OCO TENSOR]" } else { "🛡️ [OCO GUARD]" };
                                        telemetry_engine::telemetry!(
                                            "{} Protección para {} (TP: {:.4}, SL: {:.4})",
                                            tag, parsed_sym_str, order_tp_price, order_sl_price
                                        );
                                        // D-125: Resiliencia OCO con 3 retries y cierre de emergencia si falla
                                        //
                                        // OCO-F2 (raza de llenado): el ack del MARKET
                                        // llega ANTES que los fills asincrónicos del
                                        // user-stream. Bracketear con la qty teórica
                                        // cuando la posición real aún es parcial hace
                                        // que Binance rechace ambas piernas (qty >
                                        // posición ⇒ -2022 en hedge). Ahora: settle
                                        // corto de fills + cada intento bracketea
                                        // min(qty_intentada, posición REAL del exchange).
                                        // NOTA: si el bracket cubre menos que la
                                        // posición final, el remanente queda sin
                                        // protección hasta el siguiente ciclo de
                                        // reconciliación (arranque) — el settle de
                                        // 350ms minimiza esa ventana.
                                        tokio::time::sleep(tokio::time::Duration::from_millis(350)).await;
                                        let mut oco_success = false;
                                        for retry in 1..=3 {
                                            // CERT-M4-H03: verificar que la posición SIGUE ABIERTA
                                            // antes de cada retry — si el TP bracket llenó mientras
                                            // el entry task estaba entre ack y OCO, colocar brackets
                                            // sobre posición flat genera -2022 → retry ×3 → X-009
                                            // emergency-close sobre nada → falsa escalada.
                                            {
                                                let still_open = exec_clone
                                                    .load()
                                                    .fetch_position_risk()
                                                    .await
                                                    .map(|ps| {
                                                        ps.iter().any(|p| {
                                                            p.symbol == parsed_sym_str
                                                                && p.position_amt.abs() > 0.0
                                                        })
                                                    })
                                                    .unwrap_or(true); // si fetch falla, no bloquear
                                                if !still_open {
                                                    telemetry_engine::telemetry!(
                                                        "✅ [OCO-SKIP] {} posición ya cerrada antes de bracket retry {}/3 — TP llenó durante entry task",
                                                        parsed_sym_str, retry
                                                    );
                                                    oco_success = true; // no más retries
                                                    break;
                                                }
                                            }
                                            let qty_intent = final_qty.abs();
                                            let qty_bracket = exec_clone
                                                .load()
                                                .fetch_position_risk()
                                                .await
                                                .ok()
                                                .and_then(|ps| {
                                                    // HEDGE: DOS registros por símbolo
                                                    // (LONG/SHORT); el vacío trae amt=0
                                                    // y puede ordenar primero.
                                                    ps.iter()
                                                        .find(|p| {
                                                            p.symbol == parsed_sym_str
                                                                && p.position_amt.abs() > 0.0
                                                        })
                                                        .map(|p| p.position_amt.abs())
                                                })
                                                .filter(|a| *a > 0.0)
                                                .map(|real| real.min(qty_intent))
                                                .unwrap_or(qty_intent);
                                            if qty_bracket < qty_intent {
                                                telemetry_engine::telemetry!(
                                                    "⏳ [OCO] {} posición real {:.4} < intent {:.4} — bracketeando lo existente (intento {}/3)",
                                                    parsed_sym_str, qty_bracket, qty_intent, retry
                                                );
                                            }
                                            let base_id = format!("CONT_oco_{}_{}", std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).unwrap_or_default().as_micros(), retry);
                                            match exec_clone.load().execute_oco_order(&parsed_sym_str, final_is_long, qty_bracket, order_tp_price, order_sl_price, dyn_step_size, dyn_tick_size, &base_id).await {
                                                Ok(_) => {
                                                    oco_success = true;
                                                    // B2.7 — DIARIO DE CONTEXTO: persistir qué procesos y
                                                    // predicciones seguían esta posición (directriz del
                                                    // operador: al reconectar, recuperar NO solo la posición
                                                    // sino su contexto). entry_tau_ms del Position NUNCA se
                                                    // seteaba (capacidad fantasma) — este diario es la única
                                                    // fuente de τ de entrada y ml de motivación.
                                                    // B3.1: τ REAL — el core guarda la τ dominante
                                                    // del espectro en Position.entry_tau_ms al abrir
                                                    // (REHAB-1b); este spawn alcanza el arena
                                                    // compartido, que expone ese atomic. La
                                                    // capacidad fantasma muere aquí: el diario
                                                    // persiste la τ que motivó la entrada y la
                                                    // recuperación B2.7 re-protege con rigor
                                                    // espectral en vez del ancla rápida.
                                                    let tau_entry = arena_clone
                                                        .coins
                                                        .get(coin_id)
                                                        .map(|c| c.positions.position.entry_tau_ms.load(Ordering::Relaxed))
                                                        .unwrap_or(0);
                                                    let ml_entry = arena_clone
                                                        .coins
                                                        .get(coin_id)
                                                        .map(|c| c.ml_prob.load(Ordering::Relaxed))
                                                        .unwrap_or(0.5);
                                                    let jr = format!(
                                                        "{{\"ts\":{},\"sym\":\"{}\",\"long\":{},\"qty\":{:.8},\"px\":{:.4},\"tau_ms\":{},\"ml\":{:.4}}}\n",
                                                        std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).unwrap_or_default().as_millis(),
                                                        parsed_sym_str, final_is_long, final_qty.abs(), _entry_price, tau_entry, ml_entry
                                                    );
                                                    let _ = std::fs::OpenOptions::new()
                                                        .create(true)
                                                        .append(true)
                                                        .open("data/position_journal.jsonl")
                                                        .and_then(|mut f| std::io::Write::write_all(&mut f, jr.as_bytes()));
                                                    break;
                                                }
                                                Err(e) => {
                                                    telemetry_engine::telemetry_err!("⚠️ [OCO RETRY {}/3] Falló bracket para {}: {}", retry, parsed_sym_str, e);
                                                    tokio::time::sleep(tokio::time::Duration::from_millis(50)).await;
                                                }
                                            }
                                        }
                                        if !oco_success {
                                            // X-009 (REHAB-3): protocolo de emergencia
                                            // CORRECTO: cerrar PRIMERO (la protección
                                            // sigue viva hasta el aterrizaje del close),
                                            // purgar órdenes después, y ESCALAR si el
                                            // cierre falla — antes los errores se
                                            // tragaban con `let _` dejando posiciones
                                            // naked en el exchange con estado local limpio.
                                            telemetry_engine::telemetry_err!("🚨 [EMERGENCY CLOSE] Fallaron 3 intentos OCO para {}. Cerrando posición a mercado ANTES de purgar órdenes.", parsed_sym_str);
                                            let is_long_close = final_is_long;
                                            // OCO-F2: cerrar la posición REAL del exchange,
                                            // no la qty teórica (reduceOnly con qty > posición
                                            // ⇒ rechazo -2022 y posición naked).
                                            let close_qty = exec_clone
                                                .load()
                                                .fetch_position_risk()
                                                .await
                                                .ok()
                                                .and_then(|ps| {
                                                    ps.iter()
                                                        .find(|p| {
                                                            p.symbol == parsed_sym_str
                                                                && p.position_amt.abs() > 0.0
                                                        })
                                                        .map(|p| p.position_amt.abs())
                                                })
                                                .filter(|a| *a > 0.0)
                                                .unwrap_or(final_qty.abs());
                                            let close_res = exec_clone
                                                .load()
                                                .execute_reduce_only_market(&parsed_sym_str, is_long_close, close_qty, dyn_step_size)
                                                .await;
                                            let close_res = match close_res {
                                                Ok(()) => Ok(()),
                                                Err(e1) => {
                                                    tokio::time::sleep(std::time::Duration::from_millis(150)).await;
                                                    exec_clone
                                                        .load()
                                                        .execute_reduce_only_market(&parsed_sym_str, is_long_close, close_qty, dyn_step_size)
                                                        .await
                                                        .map_err(|e2| format!("{} / retry: {}", e1, e2))
                                                }
                                            };
                                            // Purgar órdenes restantes (haya cerrado o no:
                                            // si no cerró, reduceOnly sigue siendo válida).
                                            if let Err(ce) = exec_clone.load().cancel_all_symbol_orders(&parsed_sym_str).await {
                                                telemetry_engine::telemetry_err!("⚠️ [EMERGENCY] Purga de órdenes de {} falló: {}", parsed_sym_str, ce);
                                            }
                                            match close_res {
                                                Ok(()) => {
                                                    // B3.19 — la emergencia alimenta la
                                                    // contabilidad/Kelly ANTES del rollback
                                                    // (la entrada FUE real en el exchange;
                                                    // antes estos cierres eran invisibles
                                                    // al aprendizaje).
                                                    record_emergency_close(
                                                        &parsed_sym_str,
                                                        is_long_close,
                                                        close_qty,
                                                        &arena_clone,
                                                        "X-009",
                                                    );
                                                    rollback_positions(&arena_clone);
                                                }
                                                Err(es) => {
                                                    // ESCALADA X-009: posición viva en el
                                                    // exchange SIN protección y sin estado
                                                    // local — esto es incidente, no log.
                                                    telemetry_engine::telemetry_err!(
                                                        "🚨🚨 [X-009 ESCALADA] {} SIN CERRAR tras retry ({}): posición potencialmente naked en el exchange. Kill-switch ARMADO + estado local conservado para el ciclo de reconciliación.",
                                                        parsed_sym_str, es
                                                    );
                                                    exec_clone.load().trigger_kill_switch();
                                                    arena_clone.kill_switch_active.store(true, Ordering::SeqCst);
                                                    // SIN rollback: el estado local refleja la
                                                    // posición que el exchange aún sostiene.
                                                }
                                            }
                                        }
                                    }
                                }
                            }
                        });
                    }
                }
            }

            msg_count += 1;
            if msg_count.is_multiple_of(100) {
                let limit_10s = engine_real.arena.config.executor_max_orders_10s.load(Ordering::Relaxed) as usize;
                let limit_w1m = engine_real.arena.config.executor_max_weight_1m.load(Ordering::Relaxed) as usize;
                exec.load().set_rate_limit_thresholds(limit_w1m, limit_10s, 1100);

                let lat = start.elapsed().as_nanos();

                // X-016/X-044 (REHAB-1): el espectro respira en telemetría —
                // fusión y τ dominante por símbolo. Primera visibilidad del
                // continuo en operación (antes: cero consumidores visibles).
                // X-044 (REHAB-5): y los PESOS del ensamble (Hedge/Brier) junto a
                // él — la calibración viva deja de ser invisible.
                for (ci, sym) in symbols_clone.iter().enumerate() {
                    if let Some(spec) = engine_real.temporal_spectrum.get(ci) {
                        let ens_w = engine_real
                            .ensembles
                            .get(ci)
                            .map(|e| e.weights())
                            .unwrap_or([0.5, 0.5]);
                        let ml_p = engine_real.arena.coins[ci].ml_prob.load(Ordering::Relaxed);
                            if ci < 3 || spec.fused_score.abs() > 0.5 {
                                telemetry_engine::telemetry!(
                                    "🌈 [ESPECTRO] {} fusión {:+.3} τ_dom {}ms pers[{:.2},{:.2}] ensamble[F:{:.4} NN:{:.4}] ml={:.4}",
                                sym,
                                spec.fused_score,
                                format_tau(spec.dominant_tau_ms),
                                spec.scales[8].persistence,
                                spec.scales[14].persistence,
                                ens_w[0],
                                ens_w[1],
                                ml_p
                            );
                        }
                    }
                }

                let mut total_unrealized_pnl = 0.0;

                for coin in engine_real.arena.coins.iter() {
                    if coin.positions.position.is_open() {
                        // D-183: Leer de coin.metrics.pnl_unrealized
                        total_unrealized_pnl += coin.metrics.pnl_unrealized.load(Ordering::Relaxed);
                    }
                }

                let win_rate = if total_trades > 0 { (total_wins as f64 / total_trades as f64) * 100.0 } else { 0.0 };

                // Update Central Profiler for Financial Dashboarding (Zero-Allocation Atomic)
                telemetry_server::profiler::update_financials_atomic(
                    total_gross_pnl,
                    total_net_pnl,
                    win_rate,
                    f64::from_bits(unified_capital.load(std::sync::atomic::Ordering::Relaxed))
                );

                // FASE 18: Emitir al Zero-Copy Bus (Ring Buffer pre-allocado)
                // Cero locks, cero allocations. 3-5ns delay en lugar de milisegundos.
                let mut payload = [0.0; 6];
                payload[0] = total_net_pnl; // NET PnL
                payload[1] = 0.0;
                payload[2] = total_unrealized_pnl;
                payload[3] = 0.0;
                payload[4] = win_rate;
                payload[5] = 0.0;

                if latency_panic {
                    telemetry_server::zero_copy_bus::GLOBAL_TELEMETRY.emit(
                        telemetry_server::zero_copy_bus::SUBSYSTEM_GOD_ENGINE,
                        telemetry_server::zero_copy_bus::EVT_LATENCY_PANIC,
                        0,
                        [lat as f64, 0.0, 0.0, 0.0, 0.0, 0.0]
                    );
                }

                telemetry_server::zero_copy_bus::GLOBAL_TELEMETRY.emit(
                    telemetry_server::zero_copy_bus::SUBSYSTEM_GOD_ENGINE,
                    telemetry_server::zero_copy_bus::EVT_OMNI_UPDATE_FAST,
                    0,
                    payload
                );

                // Enviar también al ws clásico temporalmente si es estricto, o preferiblemente
                // dejar que un background task procese el ring buffer.
                // Como es refactorización cuántica, delegamos OmniUpdate al background.
                let _ = loop_telemetry_tx.send(telemetry_server::TelemetryEvent::OmniUpdate {
                    latency_ms: if event_time > 0 { latency_ms as u64 } else { 0 },
                    latency_panic,
                    dark_alpha: dark_router_unified.get_liquidation_cascade_risk(),
                    scalp_pnl: total_unrealized_pnl,
                    swing_pnl: 0.0,
                    gross_pnl: total_gross_pnl,
                    net_pnl: total_net_pnl,
                    win_rate,
                    trade_duration_avg: 0.0,
                });

                let _ = loop_telemetry_tx.send(telemetry_server::TelemetryEvent::LatencyUpdate(lat as u64));

                // FASE 8: Legacy process respawn removed in favor of in-memory Hot-Swap via PhaseOrchestrator.
            }
            if msg_count.is_multiple_of(5000) {
                telemetry_server::telemetry_log!("⏱️ [TELEMETRY] Processed 5000 ticks/klines. Cumulative Fees: ${:.4}. Last tick: {} ns", total_fees, start.elapsed().as_nanos());

                // FASE 12: Cosecha Cuántica en vivo (ShadowForest)
                let (winner, leaderboard) = shadow_forest.harvest_best_genome();
                let _ = loop_telemetry_tx.send(telemetry_server::TelemetryEvent::ShadowLeaderboard(leaderboard));
                // D-689: la cosecha sólo cambia el genoma de producción con la
                // evolución en vivo armada explícitamente.
                let winner = winner.filter(|_| evolution_engine::online_daemon::live_evolution_armed_for_env());

                if let Some((new_alpha, pnl_gained)) = winner {
                    telemetry!("🧬 [SHADOW FOREST] ¡Cosecha Exitosa! Universo Mutante generó +${:.2} extra. Aplicando Hot-Swap...", pnl_gained);
                    // T-03 — EMBUDO ÚNICO: la cosecha pasa ANTES por
                    // GenomeEnvelope::promote (gate de bounds/RR). El viejo
                    // apply directo sin promote sobrevivía al cache de
                    // refresh_models y hacía que el arena de producción
                    // divergiera del disco indefinidamente — sin linaje, sin
                    // rollback, perdido al reiniciar.
                    match quantum_arena::genome_store::GenomeEnvelope::promote(
                        new_alpha.clone(),
                        "shadow_forest_harvest",
                        &format!("cosecha shadow forest: +${:.2} vs control", pnl_gained),
                    ) {
                        Ok(env) => {
                            env.genome.apply_to_arena(&engine_real.arena);
                            shadow_forest.replant(env.genome.clone());
                        }
                        Err(e) => telemetry!(
                            "🚫 [T-03] Cosecha rechazada por el gate del almacén — arena intacto: {}",
                            e
                        ),
                    }

                    let _ = loop_telemetry_tx.send(telemetry_server::TelemetryEvent::GenomeUpdate(Box::new(new_alpha)));
                } else {
                    // Si no hubo cosecha, enviamos el genoma actual
                    let current_genome = quantum_arena::genome::SuperGenotype::current_from_arena(&engine_real.arena);
                    let _ = loop_telemetry_tx.send(telemetry_server::TelemetryEvent::GenomeUpdate(Box::new(current_genome)));
                }
            }
        }
        telemetry_server::telemetry_log!("✅ [UNIFIED CORE] Unified Event Loop safely terminated.");

        // CERT-M4-H02 — DRENAJE GRACEFUL: kill-switch + flatten-all de TODO
        // lo vivo + persistencia del estado aprendido, ANTES de salir.
        // (El bucle normal nunca termina: sólo se llega aquí por shutdown.)
        {
            let executor = exec.load_full();
            engine_real
                .arena
                .kill_switch_active
                .store(true, Ordering::SeqCst);
            executor.trigger_kill_switch();
            match rt_handle_for_thread.block_on(executor.flatten_all_positions()) {
                Ok((closed, skipped)) => telemetry_server::telemetry_log!(
                    "🛑 [SHUTDOWN] Flatten-all completado: {} cerradas, {} ya planas.",
                    closed,
                    skipped
                ),
                Err(e) => telemetry_server::telemetry_log!(
                    "🛑 [SHUTDOWN] Flatten-all con error: {:?} — el arranque próximo reconciliará.",
                    e
                ),
            }
            persist_kelly_envelope(&risk_envelope);
            telemetry_server::telemetry_log!(
                "🛑 [SHUTDOWN] Estado persistido. Cierre limpio completo."
            );
            std::process::exit(0);
        }
        }
    }).unwrap();

    telemetry_server::telemetry_log!(
        "🔗 Connecting to Binance WebSocket: {} streams",
        symbols.len()
    );

    // Auto-Reconnecting WebSocket Loop (Isolated & Pinned)
    std::thread::Builder::new().name("ws-reader".to_string()).spawn(move || {
        if let Some(core_ids) = core_affinity::get_core_ids() {
            if core_ids.len() > 2 {
                core_affinity::set_for_current(core_ids[2]);
                telemetry_server::telemetry_log!("🔒 [CPU PINNING] WS Reader pinned to CPU Core {}", core_ids[2].id);
            }
        }
        os_guardian::set_current_thread_time_critical();

        let rt = tokio::runtime::Builder::new_current_thread()
            .enable_all()
            .build()
            .unwrap();

        rt.block_on(async move {
            let mut retry_count = 0;
        loop {
            let current_url = ws_url.load().to_string();
            let url = url::Url::parse(&current_url).expect("Invalid WS URL");
            let host = url.host_str().unwrap_or("stream.binancefuture.com");
            let port = url.port_or_known_default().unwrap_or(443);

            // FASE 8: Happy Eyeballs TCP Parallel Connection
            // Resolve IPs dynamically instead of hardcoding AWS endpoints
            let host_port = format!("{}:{}", host, port);
            let mut resolved_addrs = match tokio::net::lookup_host(&host_port).await {
                Ok(addrs) => addrs.collect::<Vec<std::net::SocketAddr>>(),
                Err(e) => {
                    telemetry_server::telemetry_log!("⚠️ [WS] DNS Resolution failed for {}: {}", host_port, e);
                    vec![]
                }
            };

            // Allow manual IP injection via ENV to bypass DNS if desired
            if let Ok(env_ips) = std::env::var("BINANCE_WS_IPS") {
                for ip_str in env_ips.split(',') {
                    if let Ok(addr) = ip_str.trim().parse::<std::net::SocketAddr>() {
                        if !resolved_addrs.contains(&addr) {
                            resolved_addrs.push(addr);
                        }
                    }
                }
            }

            if resolved_addrs.is_empty() {
                // Safe fallback in worst case
                if let Ok(fallback) = format!("{}:{}", host, port).parse::<std::net::SocketAddr>() {
                    resolved_addrs.push(fallback);
                } else {
                    telemetry_server::telemetry_log!("❌ [WS] No IPs could be resolved or parsed. Retrying...");
                    tokio::time::sleep(tokio::time::Duration::from_secs(1)).await;
                    continue;
                }
            }

            telemetry_server::telemetry_log!("🔄 [WS] Parallel TCP Connection Race (Happy Eyeballs) to {} endpoints...", resolved_addrs.len());

            let mut tasks = Vec::new();
            for addr in resolved_addrs.iter() {
                tasks.push(Box::pin(tokio::net::TcpStream::connect(*addr)));
            }

            // For TLS SNI, we still pass the original URL with the hostname (e.g. stream.binancefuture.com)
            // but the underlying TCP stream is connected directly to the fastest raw IP.
            let tcp_stream = match futures_util::future::select_ok(tasks).await {
                Ok((stream, _)) => stream,
                Err(e) => {
                    telemetry_server::telemetry_log!("❌ [WS] Happy Eyeballs Parallel Connect failed on all IPs: {}", e);
                    tokio::time::sleep(std::time::Duration::from_millis(1000)).await;
                    continue;
                }
            };

            let target_addr = tcp_stream.peer_addr().unwrap();
            telemetry_server::telemetry_log!("✅ [WS] Fast-Lane TCP Connection established to {}...", target_addr);

            // FASE 22: Conexión directa TCP con Zero-Nagle para latencia nula
            let _ = tcp_stream.set_nodelay(true);

            match tokio_tungstenite::client_async_tls(url, tcp_stream).await {
                Ok((ws_stream, _)) => {
                    telemetry_server::telemetry_log!("✅ [WS] WebSocket TLS Connected with TCP_NODELAY.");
                    retry_count = 0; // Reset retries on success
                    let sys_data = b"[SYSTEM:RECONNECT]".to_vec();
                    loop {
                        match tx_events.try_send(sys_data.clone()) {
                            Ok(_) => break,
                            Err(crossbeam_channel::TrySendError::Full(_)) => {
                                { let _ = rx_events_dropper.try_recv(); dropped_events.fetch_add(1, std::sync::atomic::Ordering::Relaxed); } // CERT-M1-H01: contar
                            }
                            Err(crossbeam_channel::TrySendError::Disconnected(_)) => break,
                        }
                    }
                    let (_, mut read) = ws_stream.split();

                    let is_testnet = std::env::var("USE_TESTNET").unwrap_or_default().trim().to_lowercase() == "true";
                    let watchdog_secs: u64 = std::env::var("BINANCE_WS_TIMEOUT_SECS")
                        .ok()
                        .and_then(|s| s.parse().ok())
                        .unwrap_or(if is_testnet { 30 } else { 15 });

                    loop {
                        tokio::select! {
                            msg_opt = tokio::time::timeout(std::time::Duration::from_secs(watchdog_secs), read.next()) => {
                                match msg_opt {
                                    Ok(Some(Ok(msg))) => {
                                        let data = msg.into_data();
                                        loop {
                                            match tx_events.try_send(data.clone()) {
                                                Ok(_) => break,
                                                Err(crossbeam_channel::TrySendError::Full(_)) => {
                                                    let _ = rx_events_dropper.try_recv(); // Bounded Drop Oldest (Drop backpressure)
                                                }
                                                Err(crossbeam_channel::TrySendError::Disconnected(_)) => break,
                                            }
                                        }
                                    }
                                    Ok(Some(Err(e))) => {
                                        telemetry_server::telemetry_log!("⚠️ [WS] Connection Error: {:?}", e);
                                        break;
                                    }
                                    Ok(None) => {
                                        telemetry_server::telemetry_log!("⚠️ [WS] Stream ended.");
                                        break;
                                    }
                                    Err(_) => {
                                        telemetry_server::telemetry_log!(
                                            "🚨 [WS] Watchdog Timeout: No data received for {} seconds! Forcing reconnect to prevent Zombie Stream.",
                                            watchdog_secs
                                        );
                                        // X-010: el inmune y el interlock SABEN que el
                                        // feed murió (antes la latencia quedaba
                                        // congelada en el último latido sano).
                                        quantum_arena::feed_health::stall();
                                        break;
                                    }
                                }
                            }
                            _ = rx_ws_control.recv() => {
                                telemetry_server::telemetry_log!("🔌 [WS] Control signal received. Dropping connection to reconnect to new URL...");
                                break;
                            }
                        }
                    }
                    telemetry_server::telemetry_log!("⚠️ [WS] Reconnecting...");
                }
                Err(e) => {
                    telemetry_server::telemetry_log!("❌ [WS] TLS Handshake failed: {:?}", e);
                    retry_count += 1;
                }
            }

            // Exponential backoff capped at 5 seconds
            let backoff_ms = std::cmp::min(100 * (2u64.pow(retry_count.min(6))), 5000);
            telemetry_server::telemetry_log!("⏳ [WS] Waiting {}ms before next attempt...", backoff_ms);
            sleep(Duration::from_millis(backoff_ms)).await;
        }
        });
    }).unwrap();

    let _ = unified_handle.join();

    Ok(())
}

/// X-040 (REHAB-4): STOP_TRADING.LOCK por RUTA ABSOLUTA. Antes: relativo al
/// CWD — lanzando el motor desde otro directorio, el bloqueo de emergencia
/// del operador era INVISIBLE. Ahora: se busca en CWD y junto al ejecutable.
fn operator_lock_exists() -> bool {
    if std::path::Path::new("STOP_TRADING.LOCK").exists() {
        return true;
    }
    if let Ok(exe) = std::env::current_exe() {
        if let Some(dir) = exe.parent() {
            return dir.join("STOP_TRADING.LOCK").exists();
        }
    }
    false
}

/// X-016: tau legible para telemetría (ms → s/min/h/d legible).
fn format_tau(tau_ms: f64) -> String {
    if tau_ms <= 0.0 {
        "-".to_string()
    } else if tau_ms < 1000.0 {
        format!("{}ms", tau_ms as u64)
    } else if tau_ms < 60_000.0 {
        format!("{:.0}s", tau_ms / 1000.0)
    } else if tau_ms < 3_600_000.0 {
        format!("{:.0}min", tau_ms / 60_000.0)
    } else if tau_ms < 86_400_000.0 {
        format!("{:.1}h", tau_ms / 3_600_000.0)
    } else {
        format!("{:.1}d", tau_ms / 86_400_000.0)
    }
}
