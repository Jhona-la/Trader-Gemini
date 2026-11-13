//! CAPITAL-UNICO V4I (INV-F, puntos 3, 5c y 9c): la frontera de una entrada es la ENTREGA al
//! transporte. El OMS real (`OmsSession::dispatch_entry_reservada`, con su guardia) contra un
//! transporte sintetico que espera donde espera el ejecutor real: dentro de la confirmacion del
//! apalancamiento, antes del POST y, en la ruta maker, entre el GTX y el MARKET del remanente.
//! Cada corte se intercala por separado; el resultado es 0 entregas de la propuesta vieja, o 1 de
//! una propuesta recalculada con la revision nueva; el control sin cambio entrega 1.
//! Sin red, sin credenciales; se sondea a mano con un waker nulo (sin hilos ni plazos).
use arc_swap::ArcSwap;
use execution_engine::entry_dispatch::{EntryRequest, EntryRoute, EntryTransport};
use execution_engine::executor::OrderExecutor;
use execution_engine::reconciliation::PositionRiskEntry;
use quantum_arena::dimensionable::{cruzar_frontera_de_envio, ClaveReserva};
use quantum_arena::{FeeSchedule, GlobalArena, SymbolFees};
use quantum_engine::host_oms::{OmsSession, SessionToken};
use std::future::Future;
use std::pin::Pin;
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::sync::{Arc, Mutex, MutexGuard};
use std::task::{Context, Poll, Waker};

static UNIVERSO: Mutex<()> = Mutex::new(());

#[derive(Default)]
struct Transporte {
    apalancamientos: AtomicUsize,
    posts: AtomicUsize,
    remanentes: AtomicUsize,
    /// Espera dentro de la confirmacion del apalancamiento (despues del permiso).
    espera_apalancamiento: bool,
    /// Espera antes del POST (despues de la ultima revalidacion del guardia).
    espera_post: bool,
    /// Ruta maker: POST del GTX, espera, POST del MARKET del remanente.
    remanente: bool,
    sigue: AtomicBool,
}

async fn esperar(sigue: &AtomicBool) {
    std::future::poll_fn(|_| if sigue.load(Ordering::Acquire) { Poll::Ready(()) } else { Poll::Pending }).await
}

impl EntryTransport for Transporte {
    fn supports_native_iceberg(&self) -> bool {
        false
    }

    async fn configure_leverage(&self, _symbol: &str, _leverage: u32) -> Result<(), String> {
        self.apalancamientos.fetch_add(1, Ordering::Relaxed);
        if self.espera_apalancamiento {
            esperar(&self.sigue).await;
        }
        Ok(())
    }

    async fn submit_entry(&self, _request: &EntryRequest<'_>) -> Result<(), String> {
        if self.espera_post {
            esperar(&self.sigue).await;
        }
        // Como el ejecutor real: la frontera justo antes de cada POST de apertura.
        cruzar_frontera_de_envio()?;
        self.posts.fetch_add(1, Ordering::Relaxed);
        if self.remanente {
            self.sigue.store(false, Ordering::Release);
            esperar(&self.sigue).await; // sondeo, cancelacion y consulta del GTX
            cruzar_frontera_de_envio()?;
            self.remanentes.fetch_add(1, Ordering::Relaxed);
        }
        Ok(())
    }
}

struct Preparado {
    _universo: MutexGuard<'static, ()>,
    arena: Arc<GlobalArena>,
    sesion: Arc<OmsSession>,
    token: SessionToken,
    clave: ClaveReserva,
}

fn request() -> EntryRequest<'static> {
    EntryRequest {
        symbol: "BTCUSDT",
        is_long: true,
        quantity: 0.01,
        step_size: 0.001,
        tick_size: 0.1,
        leverage: 1,
        client_order_id: "cL_v4i_frontera",
        route: EntryRoute::Market,
    }
}

fn preparado() -> Preparado {
    let universo = UNIVERSO.lock().unwrap_or_else(|e| e.into_inner());
    quantum_arena::symbols::update_dynamic_universe(vec!["BTCUSDT".into(), "ETHUSDT".into()]);
    let arena = GlobalArena::build_in_own_stack(100.0);
    let mut ejecutor = OrderExecutor::new(String::new(), "offline-contract".into(), true);
    ejecutor.set_paper_trading(false);
    let sesion = Arc::new(OmsSession::new(Arc::new(ArcSwap::from(Arc::new(ejecutor))), Arc::new(AtomicBool::new(false))));
    let token = sesion.snapshot_token();
    let posiciones = quantum_arena::symbols::get_active_universe_ref()
        .iter()
        .map(|s| PositionRiskEntry { symbol: s.to_uppercase(), position_side: "BOTH".into(), ..Default::default() })
        .collect();
    let filas = quantum_arena::symbols::get_active_universe_ref()
        .iter()
        .map(|s| SymbolFees { symbol: s.to_string(), maker: 0.0, taker: 0.0003 })
        .collect();
    let fees = FeeSchedule::new(token.generation(), filas).map_err(|e| format!("{e:?}"));
    sesion.apply_snapshot(&token, Ok(posiciones), Ok(Vec::new()), fees, &arena, 1_000);
    let clave = ClaveReserva { coin_id: 0, slot: 0, generation: 1 };
    {
        let mut libro = arena.config.entry_publication_gate.lock().unwrap_or_else(|e| e.into_inner());
        libro.activar(&arena);
        libro.saldo_aceptado(&arena, 100.0, Some(100));
        arena.capital_incompleto.store(false, Ordering::Release);
        let capital = Some(arena.unified_capital.load(Ordering::Relaxed));
        assert!(libro.admite(&arena, capital));
        libro.reservar_con_base(&arena, clave, 0.04, capital);
    }
    Preparado { _universo: universo, arena, sesion, token, clave }
}

fn sondear<F: Future>(f: Pin<&mut F>) -> Poll<F::Output> {
    f.poll(&mut Context::from_waker(Waker::noop()))
}

/// El saldo cambia y la admision se reabre (el publicador suelta el veto en el mismo paso).
fn baja_y_reabre(arena: &GlobalArena) {
    let mut libro = arena.config.entry_publication_gate.lock().unwrap_or_else(|e| e.into_inner());
    arena.capital_incompleto.store(true, Ordering::Release);
    libro.saldo_aceptado(arena, 90.0, Some(200));
    arena.capital_incompleto.store(false, Ordering::Release);
}

#[test]
fn control_sin_cambio_entrega_una_vez() {
    let p = preparado();
    let t = Transporte::default();
    let r = request();
    let mut envio = std::pin::pin!(p.sesion.dispatch_entry_reservada(&p.token, &t, &r, &p.arena, Some(p.clave)));
    assert!(matches!(sondear(envio.as_mut()), Poll::Ready(Ok(()))));
    assert_eq!((t.apalancamientos.load(Ordering::Relaxed), t.posts.load(Ordering::Relaxed)), (1, 1));
}

#[test]
fn barrera_antes_del_permiso_ni_apalancamiento_ni_post() {
    let p = preparado();
    baja_y_reabre(&p.arena);
    let t = Transporte::default();
    let r = request();
    let mut envio = std::pin::pin!(p.sesion.dispatch_entry_reservada(&p.token, &t, &r, &p.arena, Some(p.clave)));
    assert!(matches!(sondear(envio.as_mut()), Poll::Ready(Err(_))));
    assert_eq!((t.apalancamientos.load(Ordering::Relaxed), t.posts.load(Ordering::Relaxed)), (0, 0), "una propuesta revocada configuro apalancamiento o abrio");
}

#[test]
fn barrera_despues_del_permiso_sin_post() {
    let p = preparado();
    let t = Transporte { espera_apalancamiento: true, ..Default::default() };
    let r = request();
    let mut envio = std::pin::pin!(p.sesion.dispatch_entry_reservada(&p.token, &t, &r, &p.arena, Some(p.clave)));
    assert!(sondear(envio.as_mut()).is_pending());
    baja_y_reabre(&p.arena);
    t.sigue.store(true, Ordering::Release);
    assert!(matches!(sondear(envio.as_mut()), Poll::Ready(Err(_))));
    assert_eq!(t.posts.load(Ordering::Relaxed), 0);
}

/// Tras la ultima revalidacion y antes de que el transporte entregue: cambio de saldo, veto del
/// PUENTE y veto del LIBRO, por separado. 0 entregas; despues, la propuesta recalculada con la
/// revision nueva entrega 1.
#[test]
fn barrera_despues_de_la_ultima_revalidacion_cero_entregas_y_recalculada_una() {
    for corte in ["saldo", "veto del puente", "veto del libro"] {
        let p = preparado();
        let t = Transporte { espera_post: true, ..Default::default() };
        let r = request();
        {
            let mut envio = std::pin::pin!(p.sesion.dispatch_entry_reservada(&p.token, &t, &r, &p.arena, Some(p.clave)));
            assert!(sondear(envio.as_mut()).is_pending(), "{corte}");
            match corte {
                "saldo" => baja_y_reabre(&p.arena),
                "veto del puente" => p.arena.capital_incompleto.store(true, Ordering::Release),
                _ => p.arena.config.entry_publication_gate.lock().unwrap().cierre_externo(&p.arena),
            }
            t.sigue.store(true, Ordering::Release);
            assert!(matches!(sondear(envio.as_mut()), Poll::Ready(Err(_))), "{corte}: la propuesta vieja llego al transporte");
            assert_eq!(t.posts.load(Ordering::Relaxed), 0, "{corte}");
        }
        if corte == "saldo" {
            // Propuesta recalculada con la revision nueva: se cancela la vieja y se reserva otra.
            let nueva = ClaveReserva { generation: 2, ..p.clave };
            {
                let mut libro = p.arena.config.entry_publication_gate.lock().unwrap();
                libro.cancelar(&p.arena, p.clave, 0.04);
                let capital = Some(p.arena.unified_capital.load(Ordering::Relaxed));
                assert!(libro.admite(&p.arena, capital));
                libro.reservar_con_base(&p.arena, nueva, 0.04, capital);
            }
            let t2 = Transporte::default();
            let mut envio = std::pin::pin!(p.sesion.dispatch_entry_reservada(&p.token, &t2, &r, &p.arena, Some(nueva)));
            assert!(matches!(sondear(envio.as_mut()), Poll::Ready(Ok(()))));
            assert_eq!(t2.posts.load(Ordering::Relaxed), 1, "la recalculada no entrego");
        }
    }
}

/// Ruta maker: el GTX ya se entrego; entre el GTX y el MARKET del remanente baja el saldo. El
/// remanente no sale con la revision vieja (0 POST de apertura); el GTX entregado no se abandona.
#[test]
fn remanente_tras_una_espera_no_sale_con_la_revision_vieja() {
    let p = preparado();
    let t = Transporte { remanente: true, ..Default::default() };
    let r = request();
    let mut envio = std::pin::pin!(p.sesion.dispatch_entry_reservada(&p.token, &t, &r, &p.arena, Some(p.clave)));
    assert!(sondear(envio.as_mut()).is_pending());
    assert_eq!(t.posts.load(Ordering::Relaxed), 1, "fixture: el GTX salio");
    baja_y_reabre(&p.arena);
    assert!(sondear(envio.as_mut()).is_pending(), "una entrega ya cruzada no se abandona");
    t.sigue.store(true, Ordering::Release);
    assert!(matches!(sondear(envio.as_mut()), Poll::Ready(Err(_))));
    assert_eq!(t.remanentes.load(Ordering::Relaxed), 0, "el remanente salio con la revision vieja");
}
