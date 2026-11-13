//! CAPITAL-UNICO V4D: las RED de Codex QRA, TAL COMO ESTAN ESCRITAS en
//! `.agent_tx/diagnostics/auditoria_perdidas/`: `qra_capital_v4c_lote_rechazado.rs`
//! (13:42, P1-d), `qra_capital_v4d_primer_socket.rs` (13:04, P1-a),
//! `qra_capital_v4d_publicacion.rs` (13:07, P1-b) y `qra_capital_v4d_sesion.rs`
//! (13:03, P1-c). Cambios: esta cabecera, un modulo por archivo, `super::`
//! en el `use` del puente, rutas de `include_str!` al arbol del repo, `//!` a
//! `//` dentro del modulo, `#[ignore]` con motivo en las seis que quedan para
//! el coordinador (rediseno) y, en P1-a, el ayudante `con_veto_del_host` pasa el
//! puente por el arranque del streamer real (las aserciones no cambian; ese modulo
//! usa el puente del crate, el mismo tipo que recibe el streamer).
//! V4E: los seis `#[ignore]` se quitaron. Con el contrato nuevo (publicador del
//! puente y testigo de sesion) se adaptan, sin tocar ninguna asercion: el ayudante
//! `con_veto_del_host` (publicador), las anclas de fuente del host, del helper y de
//! la transicion, y en P1-c la sesion de cada ejecutor y la llamada del helper
//! (`observar_rest_de_sesion`). Tabla en integrador_capital_unico.md, seccion V4E.

#[path = "../src/puente_cuenta.rs"]
#[allow(dead_code)]
mod puente_cuenta;
mod qra_capital_v4c_lote_rechazado {
    use super::puente_cuenta::{DatoPosicion, EventoStream, LecturaCuenta, PosicionRest, PuenteCuenta, Rechazo};
    use std::collections::HashMap;

    fn lectura(marca_saldo: u64) -> LecturaCuenta {
        LecturaCuenta {
            saldo: 100.0,
            tiempo_saldo_ms: Some(marca_saldo),
            posiciones: Some(HashMap::from([(("BTCUSDT".to_string(), "BOTH".to_string()), PosicionRest {
                dato: DatoPosicion::Abierta { cantidad: 1.0, upnl: -10.0 },
                tiempo_ms: Some(100),
            })])),
            rechazo_arranque: None,
            upnl_del_activo: Some(-10.0),
        }
    }

    #[test]
    fn lote_ws_rechazado_por_posicion_no_impide_confirmar_saldo_vigente() -> Result<(), Rechazo> {
        let p = PuenteCuenta::new();
        p.observar_rest(p.ticket(), &lectura(99))?;
        let s = p.socket();
        let sube_marca = EventoStream { tiempo_ms: Some(100), saldo: Some(100.0), posiciones: vec![] };
        p.observar_stream_del_socket(p.era(), s, Some(200), &sube_marca, |_| {})?;
        let conflicto = EventoStream {
            tiempo_ms: Some(100),
            saldo: Some(90.0),
            posiciones: vec![(("BTCUSDT".to_string(), "BOTH".to_string()), DatoPosicion::Abierta { cantidad: 0.5, upnl: -5.0 })],
        };
        let rechazo = p.observar_stream_del_socket(p.era(), s, Some(201), &conflicto, |_| {});
        assert!(matches!(rechazo, Err(Rechazo::Contradiccion { componente, .. }) if componente.contains("BTCUSDT")));
        assert_eq!(p.observacion().map(|o| o.saldo), Some(100.0));
        assert_eq!(p.equity_del_freno(), Some(80.0));
        assert!(p.veta_entradas());
        let posterior = p.observar_rest(p.ticket(), &lectura(100));
        assert!(posterior.is_ok(), "foto completa que confirma saldo y posición vigentes: {posterior:?}");
        assert!(!p.veta_entradas());
        assert_eq!(p.equity_del_freno(), Some(90.0));
        Ok(())
    }
}

mod qra_capital_v4d_primer_socket {
    use execution_engine::puente_cuenta::{LecturaCuenta, Observacion, PuenteCuenta, Rechazo};
    use std::collections::HashMap;
    use std::sync::Arc;
    use std::sync::atomic::{AtomicBool, Ordering};

    const HOST: &str = include_str!("../../../src/bin/god_engine.rs");
    const STREAM: &str = include_str!("../src/user_data_stream.rs");
    const OMS: &str = include_str!("../../../src/host_oms.rs");

    fn cuenta_quieta() -> LecturaCuenta {
        LecturaCuenta {
            saldo: 100.0,
            tiempo_saldo_ms: Some(100),
            posiciones: Some(HashMap::new()),
            rechazo_arranque: None,
            upnl_del_activo: Some(0.0),
        }
    }

    // V4D (adaptacion del integrador, unica en este modulo): la barrera vive en el
    // arranque del streamer real (`UserDataStreamer::start` llama primero a
    // `sin_socket_todavia`), no en `PuenteCuenta::new`. El puente pasa por ese paso.
    fn con_veto_del_host() -> (Arc<PuenteCuenta>, Arc<AtomicBool>) {
        let puente = Arc::new(PuenteCuenta::new());
        let streamer = execution_engine::user_data_stream::UserDataStreamer::new(
            execution_engine::client::BinanceClient::new("k".into(), true),
            Arc::new(execution_engine::order_registry::OrderRegistry::new()),
        )
        .with_puente(Arc::clone(&puente));
        streamer.sin_socket_todavia();
        let publicado = Arc::new(AtomicBool::new(false));
        let salida = Arc::clone(&publicado);
        // V4E (adaptacion del integrador): `vincular_veto` ya no recibe `false`;
        // el veto lo suelta el publicador, que publica antes el saldo (como el host).
        let saldo = Arc::new(std::sync::atomic::AtomicU64::new(f64::NAN.to_bits()));
        puente.vincular_publicador(move |p| {
            if let Some(s) = p.saldo {
                saldo.store(s.to_bits(), Ordering::Relaxed);
            }
            salida.store(p.veta, Ordering::Release);
            !p.veta
        });
        (puente, publicado)
    }

    #[test]
    fn red_p1a_rest_de_arranque_no_autoriza_sin_primer_socket_privado() -> Result<(), Rechazo> {
        // El host inicia el GET antes de spawnear el lector privado. Mantener el
        // handshake pendiente significa ejecutar el GET y no entregar una apertura.
        // Mercado, fase y reconciliación REST del OMS están listos por hipótesis.
        let (puente, publicado) = con_veto_del_host();
        assert!(publicado.load(Ordering::Acquire), "la era sin saldo veta");
        let lectura = puente.observar_rest(puente.ticket(), &cuenta_quieta())?;
        assert_eq!(lectura, Observacion { saldo: 100.0, equity: 100.0 });
        assert_eq!(puente.equity_del_freno(), Some(100.0), "REST disponible al freno");
        assert!(
            publicado.load(Ordering::Acquire),
            "P1-a: la REST de arranque levantó la barrera de entradas sin evidencia del primer socket"
        );
        Ok(())
    }

    #[test]
    fn control_reconexion_exige_foto_posterior_y_confirma_cuenta_quieta() -> Result<(), Rechazo> {
        let (puente, publicado) = con_veto_del_host();
        puente.observar_rest(puente.ticket(), &cuenta_quieta())?;
        let anterior = puente.ticket();
        let socket = puente.abrir_socket(puente.era())?;
        assert_ne!(anterior.socket(), socket);
        assert!(matches!(
            puente.observar_rest(anterior, &cuenta_quieta()),
            Err(Rechazo::OtroSocket { .. })
        ));
        assert!(publicado.load(Ordering::Acquire));
        puente.observar_rest(puente.ticket(), &cuenta_quieta())?;
        assert!(!publicado.load(Ordering::Acquire));
        Ok(())
    }

    #[test]
    fn control_el_corte_minimo_ejercita_la_ruta_real_del_host() {
        // V4E (anclas adaptadas): la REST de arranque lleva el testigo de sesion y
        // el veto lo escribe el publicador del puente.
        let rest = HOST.find(".observar_rest_de_sesion(ticket, token_arranque.generation(), &l)");
        let lector = HOST.find("streamer.start().await");
        assert!(matches!((rest, lector), (Some(r), Some(s)) if r < s));
        assert!(HOST.contains("puente_cuenta.vincular_publicador"));
        assert!(HOST.contains("arena.capital_incompleto.store(!suelto, Ordering::Release)"));
        assert!(OMS.contains("if arena.capital_incompleto.load(Ordering::Acquire)"));
        assert!(STREAM.contains("let ws_stream = match connect_async(url.as_str()).await"));
    }
}

mod qra_capital_v4d_publicacion {
    // Contrato RED P1-b del candidato 3 sobre 1421dd28e885a969f438fcb1e8ec94d7657d853a.
    //
    // Copiar este archivo a una carpeta propia con el código del SHA bajo prueba:
    // `crates/execution-engine/src/puente_cuenta.rs` como `puente_cuenta.rs`,
    // `src/bin/god_engine.rs` como `god_engine.rs`, `src/host_oms.rs` como
    // `host_oms.rs` y `crates/risk-engine/src/lib.rs` como `risk_engine_lib.rs`.
    // `rustc --edition 2021 --test qra_capital_v4d_publicacion.rs -o qra_capital_v4d_publicacion.exe`
    // `.\qra_capital_v4d_publicacion.exe --test-threads=1 qra_publicacion_`
    //
    // El puente real ejecuta su observador de veto. El saldo y el veto de arena
    // se representan con los dos atómicos que interesan al lector. El lector se
    // intercala después del GET y antes de la siembra: no hay reloj ni red.
    // Se presupone que el resto de las comprobaciones de admisión pasó.
    // Los ejemplos son lecturas tipadas construidas y el caso 3a ya viene
    // clasificado. No se ejecutan parser JSON, host asíncrono, Risk ni OMS.
    // Sus pruebas de fuente anclan el camino de producción. Si cambia la API
    // del publicador, vincular al arnés el publicador real y ajustar los anclajes.


    use super::puente_cuenta::{LecturaCuenta, Observacion, PuenteCuenta, Rechazo};
    use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
    use std::sync::Arc;

    const HOST: &str = include_str!("../../../src/bin/god_engine.rs");
    const OMS: &str = include_str!("../../../src/host_oms.rs");
    const RISK: &str = include_str!("../../risk-engine/src/lib.rs");

    struct LectorArena {
        saldo: AtomicU64,
        veto: AtomicBool,
    }

    impl LectorArena {
        fn vinculado(puente: &PuenteCuenta, saldo_anterior: f64) -> Arc<Self> {
            let arena = Arc::new(Self {
                saldo: AtomicU64::new(saldo_anterior.to_bits()),
                veto: AtomicBool::new(true),
            });
            let destino = Arc::clone(&arena);
            puente.vincular_veto(move |veta| destino.veto.store(veta, Ordering::Release));
            arena
        }

        fn saldo_local(&self) -> f64 {
            f64::from_bits(self.saldo.load(Ordering::Relaxed))
        }

        fn saldo_si_admite(&self) -> Option<f64> {
            if self.veto.load(Ordering::Acquire) {
                return None;
            }
            let saldo = self.saldo_local();
            (saldo.is_finite() && saldo > 0.0).then_some(saldo)
        }
    }

    fn lectura(saldo: f64, rechazo_3a: Option<&str>) -> LecturaCuenta {
        LecturaCuenta {
            saldo,
            tiempo_saldo_ms: Some(100),
            posiciones: Some(Default::default()),
            rechazo_arranque: rechazo_3a.map(str::to_owned),
            upnl_del_activo: Some(0.0),
        }
    }

    fn transicion() -> Result<(PuenteCuenta, Arc<LectorArena>), Rechazo> {
        let puente = PuenteCuenta::new();
        let arena = LectorArena::vinculado(&puente, 100.0);
        let era = puente.nueva_era();
        // Exigir socket abierto aquí aísla P1-b del primer handshake de P1-a.
        puente.reconexion(era)?;
        assert!(arena.veto.load(Ordering::Acquire));
        Ok((puente, arena))
    }

    #[test]
    fn qra_publicacion_get_positivo_retiene_admision_hasta_publicar_saldo() -> Result<(), Rechazo> {
        let (puente, arena) = transicion()?;
        let obs = puente.observar_rest(puente.ticket(), &lectura(10.0, None))?;
        assert_eq!(obs, Observacion { saldo: 10.0, equity: 10.0 });
        assert_eq!(arena.saldo_local(), 100.0);
        assert_eq!(
            arena.saldo_si_admite(),
            None,
            "el GET nuevo dejó dimensionar con saldo de la sesión anterior",
        );
        Ok(())
    }

    #[test]
    fn qra_publicacion_cero_3a_llega_al_freno_y_retiene_veto_sin_siembra() -> Result<(), Rechazo> {
        let (puente, arena) = transicion()?;
        let rechazado = lectura(0.0, Some("USDT cero con otros fondos o Multi-Assets"));
        puente.observar_rest(puente.ticket(), &rechazado)?;
        assert_eq!(puente.observacion(), Some(Observacion { saldo: 0.0, equity: 0.0 }));
        assert_eq!(arena.saldo_local(), 100.0);
        assert!(arena.veto.load(Ordering::Acquire), "3a soltó el veto pese a que no se siembra el arena");
        Ok(())
    }

    #[test]
    fn qra_publicacion_cuenta_vacia_no_dimensiona_con_saldo_anterior() -> Result<(), Rechazo> {
        let (puente, arena) = transicion()?;
        puente.observar_rest(puente.ticket(), &lectura(0.0, None))?;
        assert_eq!(puente.observacion(), Some(Observacion { saldo: 0.0, equity: 0.0 }));
        assert_eq!(arena.saldo_si_admite(), None, "la cuenta vacía heredó capacidad de entrada");
        Ok(())
    }

    fn bloque<'a>(src: &'a str, inicio: &str, fin: &str) -> Result<&'a str, String> {
        let (_, despues) = src.split_once(inicio).ok_or_else(|| format!("Falta inicio del ancla {inicio}"))?;
        let (tramo, _) = despues.split_once(fin).ok_or_else(|| format!("Falta fin del ancla {fin}"))?;
        Ok(tramo)
    }

    fn punto(src: &str, codigo: &str) -> Result<usize, String> {
        src.find(codigo).ok_or_else(|| format!("Falta el ancla de producción {codigo}"))
    }

    #[test]
    fn qra_publicacion_control_host_conecta_veto_al_oms() -> Result<(), String> {
        // V4E (anclas adaptadas): el veto lo escribe el publicador del puente.
        let vinculo = bloque(HOST, "puente_cuenta.vincular_publicador(move |p| {", "});")?;
        assert!(vinculo.contains("publicar_capital_de_sesion("));
        let publicador = bloque(HOST, "fn publicar_capital_de_sesion(", "fn suspend_symbol_until(")?;
        assert!(publicador.contains("arena.capital_incompleto.store(!suelto, Ordering::Release)"));
        let admision = bloque(OMS, "fn check_entry_locked(", "pub fn begin_refresh(")?;
        let sesion = punto(admision, "self.token_is_current(state.generation, token)")?;
        let veto = punto(admision, "arena.capital_incompleto.load(Ordering::Acquire)")?;
        let preparado = punto(admision, "self.ready.load(Ordering::Acquire)")?;
        assert!(sesion < veto && veto < preparado);
        Ok(())
    }

    #[test]
    fn qra_publicacion_control_host_3a_sin_siembra_y_cero_vacio_admitido_como_saldo() -> Result<(), String> {
        let transicion = bloque(
            HOST,
            "let mut intento: u32 = 0;",
            "if let Ok(real_positions) = balance_executor.fetch_open_positions().await",
        )?;
        // V4E (ancla adaptada): la lectura lleva el testigo de la sesion.
        let observacion = punto(transicion, "puente_transicion.observar_rest_de_sesion(ticket, balance_token.generation(), l)")?;
        let caso_3a = bloque(transicion, "|l| l.rechazo_arranque.clone()) {", "let publicado =")?;
        let rama_3a = punto(transicion, "|l| l.rechazo_arranque.clone()) {")?;
        let publica = punto(transicion, ".publish_current_balance(&balance_token, bal,")?;
        assert!(observacion < rama_3a && rama_3a < publica);
        assert!(caso_3a.contains("break;") && !caso_3a.contains("unified_capital.store("));
        // V4G (ancla adaptada): el saldo dimensionable de la transicion lo entrega el publicador
        // del puente al aceptarse la REST (un solo camino); la transicion ya no escribe el campo
        // y su `base_capital` sigue detras de `publish_current_balance` con el testigo.
        assert!(!caso_3a.contains("saldo_aceptado("));
        assert!(!transicion.contains("unified_capital.store("), "la transicion no es un segundo camino del dimensionable");
        assert!(publica < punto(transicion, "arena_real_clone.config.base_capital.store(balance,")?);

        let publicador = bloque(OMS, "pub fn publish_current_balance", "pub async fn await_quiescence")?;
        let token_y_dominio = punto(publicador, "self.token_is_current(state.generation, token) && balance.is_finite() && balance >= 0.0")?;
        assert!(token_y_dominio < punto(publicador, "publish(balance);")?);
        Ok(())
    }

    #[test]
    fn qra_publicacion_background_rest_usa_publicador_de_saldo_de_sesion() -> Result<(), String> {
        let lectura = bloque(HOST, "let puente_rs = Arc::clone(&puente_cuenta);", "match &r {")?;
        let llamada = punto(lectura, "resincronizar_una_vez(")?;
        let ruta = &lectura[llamada..];
        assert!(ruta.contains("executor.fetch_account_lectura()"));
        // El publicador compartido debe validar sesión y alcanzar el saldo de tamaño.
        // Adaptar este anclaje si la corrección encapsula esa operación en un helper.
        // V4E (anclaje adaptado, como pide la nota): la operacion vive en el
        // publicador del puente (`publicar_capital_de_sesion`), que es el UNICO que
        // suelta el veto; el GET periodico lleva la sesion de su ejecutor.
        assert!(ruta.contains("token.generation(),"), "la lectura periodica no lleva la sesion de su ejecutor");
        assert!(HOST.contains("puente_cuenta.vincular_publicador(move |p| {"));
        let publicador = bloque(HOST, "fn publicar_capital_de_sesion(", "fn suspend_symbol_until(")?;
        // V4G (ancla adaptada): el publicador entrega el saldo al libro del dimensionable, que es
        // quien escribe `unified_capital` (saldo aceptado menos reservas).
        let libro = include_str!("../../quantum-arena/src/dimensionable.rs");
        assert!(
            publicador.contains("publish_current_balance(")
                && publicador.contains(".saldo_aceptado(arena, balance, marca);")
                && libro.contains("arena.unified_capital.store(d, Ordering::Relaxed);"),
            "la resincronización REST puede levantar el veto y sólo actualiza el espejo de saldo",
        );
        Ok(())
    }

    #[test]
    fn qra_publicacion_risk_adquiere_veto_antes_de_leer_saldo_y_equity() -> Result<(), String> {
        let (_, evaluacion) = RISK.split_once("fn evaluate_quantum_order_impl(")
            .ok_or_else(|| "Falta el evaluador de producción".to_owned())?;
        let veto = punto(evaluacion, "arena.capital_incompleto.load(Ordering::Acquire)")?;
        let saldo = punto(evaluacion, "let current_capital = arena.unified_capital.load(")?;
        let equity = punto(evaluacion, "let capital_freno = crate::drawdown::capital_del_freno(arena);")?;
        assert!(veto < saldo && veto < equity, "Risk captura saldo o equity antes de adquirir el veto publicado");
        assert!(evaluacion.contains("current_capital <= 0.0"));
        Ok(())
    }
}

mod qra_capital_v4d_sesion {
    // P1-c · RED de acoplamiento entre ejecutor y ticket de cuenta.
    //
    // Base 1421dd28e885a969f438fcb1e8ec94d7657d853a. Copiar junto a este
    // archivo puente_cuenta.rs, god_engine.rs, host_oms.rs y
    // resincronizar_cuenta.rs de ese SHA. Ejecutar rustc --edition 2021 --test.
    // Se ejercita el puente real. Los puntos de corte se fijan en las llamadas
    // del host y del helper. El transporte se representa con lecturas completas
    // ya disponibles para hacer determinista el orden sin traer el runtime async.
    // Los dos ejecutores representan la misma cuenta y venue. Las dos lecturas
    // representan instantes distintos de esa cuenta.


    use super::puente_cuenta::{LecturaCuenta, Observacion, PuenteCuenta, Rechazo, Ticket};
    use std::collections::HashMap;
    use std::sync::Arc;

    const HOST: &str = include_str!("../../../src/bin/god_engine.rs");
    const OMS: &str = include_str!("../../../src/host_oms.rs");
    const RESINCRONIZADOR: &str = include_str!("../src/resincronizar_cuenta.rs");

    fn tramo<'a>(fuente: &'a str, desde: &str, hasta: &str) -> Result<&'a str, String> {
        fuente
            .split_once(desde)
            .and_then(|(_, resto)| resto.split_once(hasta))
            .map(|(cuerpo, _)| cuerpo)
            .ok_or_else(|| format!("tramo ausente: {desde} / {hasta}"))
    }

    fn en_orden(mut fuente: &str, pasos: &[&str]) -> Result<(), String> {
        for paso in pasos {
            let (_, resto) = fuente
                .split_once(paso)
                .ok_or_else(|| format!("ancla ausente o fuera de orden: {paso}"))?;
            fuente = resto;
        }
        Ok(())
    }

    #[test]
    fn control_fuente_situa_los_dos_intercalados_de_produccion() -> Result<(), String> {
        let trabajador = tramo(HOST, "let puente_rs =", "let mut latched = false;")?;
        // V4E (anclas adaptadas): los dos intercalados siguen existiendo en el
        // texto (el ejecutor se toma antes que el ticket; el OMS publica el
        // ejecutor antes de la era), pero el ejecutor sale del testigo del OMS
        // con su sesion y el puente rechaza una sesion que no es la de su era.
        en_orden(
            trabajador,
            &[
                "if !puente_rs.necesita_resincronizar()",
                "let token = oms_rs.snapshot_token();",
                "let executor = Arc::clone(token.executor());",
                "resincronizar_cuenta::resincronizar_una_vez(",
                "&puente_rs,",
                "token.generation(),",
                "executor.fetch_account_lectura(),",
                ".await;",
            ],
        )?;
        let helper = tramo(RESINCRONIZADOR, "pub async fn resincronizar_una_vez", "pub struct Pausa")?;
        en_orden(
            helper,
            &[
                "let ticket = puente.ticket();",
                "tokio::time::timeout(tope, lectura).await",
                "puente.observar_rest_de_sesion(ticket, sesion, &l)",
                ".con_era_vigente(ticket.era()",
            ],
        )?;
        let publica = tramo(OMS, "pub fn replace_executor(", "fn ocupada_locked(")?;
        en_orden(publica, &["self.executors.store(executor);"])?;
        let transicion = tramo(OMS, "pub fn paso_transicion", "self.replace_executor(Arc::clone(&ejecutor), arena)")?;
        en_orden(transicion, &["let ejecutor ="])?;
        en_orden(
            HOST,
            // V4F (ancla adaptada por el integrador): el relevo del OMS corre
            // dentro de `relevar_sesion`, que abre la era en el mismo paso.
            &[
                "let (paso, era_del_relevo) = puente_cuenta.relevar_sesion(|| {",
                "let paso = oms_session.paso_transicion(",
                "let (sesion_mainnet, new_exec_arc) = match paso.resultado",
                "let Some(era_mainnet) = era_del_relevo else {",
            ],
        )
    }

    fn lectura(saldo: f64, marca: u64) -> LecturaCuenta {
        LecturaCuenta {
            saldo,
            tiempo_saldo_ms: Some(marca),
            posiciones: Some(HashMap::new()),
            rechazo_arranque: None,
            upnl_del_activo: Some(0.0),
        }
    }

    // V4E (adaptacion del integrador): cada ejecutor lleva la sesion OMS de su
    // `SessionToken` (generacion). `PuenteCuenta::new` es la sesion 0 y
    // `nueva_era` la siguiente, como `replace_executor`.
    struct VistaEjecutor {
        cuenta: &'static str,
        testnet: bool,
        sesion: u64,
        lectura: LecturaCuenta,
    }

    fn ejecutores() -> (Arc<VistaEjecutor>, Arc<VistaEjecutor>) {
        let anterior = Arc::new(VistaEjecutor {
            cuenta: "cuenta_fixture",
            testnet: true,
            sesion: 0,
            lectura: lectura(97.0, 110),
        });
        let sucesor = Arc::new(VistaEjecutor {
            cuenta: anterior.cuenta,
            testnet: anterior.testnet,
            sesion: anterior.sesion + 1,
            lectura: lectura(94.0, 120),
        });
        assert_eq!(
            (anterior.cuenta, anterior.testnet),
            (sucesor.cuenta, sucesor.testnet),
        );
        assert!(!Arc::ptr_eq(&anterior, &sucesor));
        (anterior, sucesor)
    }

    fn puente_pendiente() -> Result<PuenteCuenta, Rechazo> {
        let puente = PuenteCuenta::new();
        puente.observar_rest(puente.ticket(), &lectura(100.0, 100))?;
        // Evidencia ilegible dentro del socket abierto. El trabajador cumple
        // needs-resync y la era anterior aún tiene un camino para soltar el veto.
        puente.evidencia_perdida(puente.era())?;
        assert!(puente.necesita_resincronizar());
        Ok(puente)
    }

    fn completar_lectura(
        puente: &PuenteCuenta,
        ticket: Ticket,
        ejecutor: &VistaEjecutor,
        publicado: &mut Option<Observacion>,
    ) -> Result<Observacion, Rechazo> {
        // Mismas llamadas del helper a las APIs reales del puente. La respuesta
        // determinista ocupa el lugar de la espera del GET entre ambos pasos.
        // V4E (adaptacion): el helper de produccion entrega la sesion del ejecutor.
        let observacion = puente.observar_rest_de_sesion(ticket, ejecutor.sesion, &ejecutor.lectura)?;
        puente
            .con_era_vigente(ticket.era(), |vigente| {
                *publicado = Some(vigente.unwrap_or(observacion));
            })
            .ok_or(Rechazo::OtraEra)?;
        Ok(observacion)
    }

    fn verificar_sin_autoridad(puente: &PuenteCuenta, ticket: Ticket, ejecutor: &VistaEjecutor) {
        let antes = (puente.observacion(), puente.equity_del_freno());
        let mut publicado = None;
        let recibo = completar_lectura(puente, ticket, ejecutor, &mut publicado);
        assert_eq!(
            (
                recibo.ok(),
                publicado,
                puente.observacion(),
                puente.equity_del_freno(),
                puente.veta_entradas(),
            ),
            (None, None, antes.0, antes.1, true),
            "una lectura de un ejecutor de otra generación no modifica la vista, la equity ni la admisión",
        );
    }

    #[test]
    fn red_ejecutor_anterior_retenido_hasta_tomar_ticket_de_la_era_nueva() -> Result<(), Rechazo> {
        let puente = puente_pendiente()?;
        let (mut publicado_en_oms, sucesor) = ejecutores();
        let era_anterior = puente.era();
        // El trabajador ya hizo load_full. El otro hilo publica el sucesor y
        // cambia la era antes de que se sondee por primera vez el helper.
        let retenido = Arc::clone(&publicado_en_oms);
        publicado_en_oms = sucesor;
        let era_nueva = puente.nueva_era();
        let ticket = puente.ticket();

        assert!(!Arc::ptr_eq(&retenido, &publicado_en_oms));
        assert_ne!(ticket.era(), era_anterior);
        assert_eq!(ticket.era(), era_nueva);
        assert_eq!(puente.observacion(), None);
        verificar_sin_autoridad(&puente, ticket, &retenido);
        Ok(())
    }

    #[test]
    fn red_ejecutor_nuevo_visible_antes_de_tomar_ticket_de_la_era_anterior() -> Result<(), Rechazo> {
        let puente = puente_pendiente()?;
        let (anterior, sucesor) = ejecutores();
        let era_anterior = puente.era();
        // La transición hizo store del sucesor pero queda detenida antes de
        // nueva_era. El trabajador hace load_full y toma un ticket todavía viejo.
        let publicado_en_oms = sucesor;
        let retenido = Arc::clone(&publicado_en_oms);
        let ticket = puente.ticket();

        assert!(!Arc::ptr_eq(&retenido, &anterior));
        assert_eq!(ticket.era(), era_anterior);
        verificar_sin_autoridad(&puente, ticket, &retenido);
        Ok(())
    }

    #[test]
    fn control_lectura_con_ejecutor_y_era_correspondientes_libera_el_veto() -> Result<(), Rechazo> {
        let puente = puente_pendiente()?;
        let (_, sucesor) = ejecutores();
        let era_nueva = puente.nueva_era();
        let ticket = puente.ticket();
        let mut publicado = None;
        let observacion = completar_lectura(&puente, ticket, &sucesor, &mut publicado)?;

        assert_eq!(ticket.era(), era_nueva);
        assert_eq!(observacion, Observacion { saldo: 94.0, equity: 94.0 });
        assert_eq!(publicado, Some(observacion));
        assert!(!puente.veta_entradas());
        Ok(())
    }

    #[test]
    fn control_ticket_tomado_en_era_anterior_no_modifica_la_nueva() -> Result<(), Rechazo> {
        let puente = puente_pendiente()?;
        let (anterior, _) = ejecutores();
        let ticket_anterior = puente.ticket();
        puente.nueva_era();
        let mut publicado = None;

        let recibo = completar_lectura(&puente, ticket_anterior, &anterior, &mut publicado);
        assert_eq!(recibo, Err(Rechazo::OtraEra));
        assert_eq!(publicado, None);
        assert_eq!(puente.observacion(), None);
        assert!(puente.veta_entradas());
        Ok(())
    }
}
