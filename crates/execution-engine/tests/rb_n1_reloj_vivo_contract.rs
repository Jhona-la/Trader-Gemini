//! CONTRATO RB-N1 (Codex RB, 2026-11-10): el reloj del host VUELVE A VIVIR.
//!
//! QUÉ: desde 97996eba0 (2026-09-17) el sincronizador NTP/Kalman no tomó una
//! sola muestra en producción. La guardia HOST-016 del host guardaba lo que
//! devuelve `fetch_add(1)` —el valor PREVIO— y salía cuando
//! `contador > previo`: verdadero desde el primer chequeo, así que toda
//! generación, también la única, se daba por reemplazada y el `select!`
//! cancelaba el bucle antes de su primera muestra. `server_time_offset_ms` y
//! `server_time_correction_ms` quedaban congelados en el arranque.
//! POR QUÉ IMPORTA: la edad del dato que bloquea entradas (LATENCY PANIC,
//! KILL_SWITCH) es `epoch_baseline + corrección + monótono − E`. Sin
//! corrección, la deriva del oscilador entra entera en ella y crece con la
//! sesión. En esta máquina w32time no sincroniza («Local CMOS Clock») y el
//! reloj local adelanta ≈24 ppm frente a Binance (medido el 2026-11-10);
//! la red al testnet, en cambio, entrega los eventos con p50 ≈ 160 ms. La
//! corrida demo a3d registró 97 812 LATENCY PANIC con «red+reloj» ≈ 7 s.
//! CÓMO: (1) la guardia nueva y el contrafactual del predicado viejo;
//! (2) conducta real con tokio: el bucle corre y sólo lo detiene un relevo;
//! (3) 72 h simuladas con la deriva medida: el error del arranque se absorbe
//! en la primera muestra, la edad queda acotada por el retardo del filtro
//! (sin bucle crece sin límite) y el relevo de generación no la reinicia,
//! al revés que el ancla propia por generación del diseño anterior;
//! (4) la política de avisos del log; (5) guardia de fuente del host;
//! (6) sonda viva contra el testnet de Binance (`#[ignore]`, con red).

use execution_engine::ntp::{
    correr_mientras_vigente, merece_aviso, EstadoReloj, FinSincronizador, MuestraNtp,
    SincronizadorUnico, MUESTRAS_POR_HORA, PERIODO_SINCRONIZACION,
};
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;
use std::time::{Duration, Instant};

#[test]
fn rb_n1_una_generacion_sigue_vigente_hasta_que_otra_la_releva() {
    let guardia = SincronizadorUnico::new();
    let primera = guardia.reclamar();
    assert_eq!(primera, 1, "la primera generación es la 1");
    assert!(guardia.vigente(primera), "la única generación está vigente");

    let segunda = guardia.reclamar();
    assert_eq!(segunda, 2);
    assert!(!guardia.vigente(primera), "la segunda releva a la primera");
    assert!(guardia.vigente(segunda), "y queda vigente ella sola");

    // Contrafactual: el predicado de HOST-016 (97996eba0). Con el valor
    // previo de `fetch_add`, «hay una generación posterior» es verdadero en
    // el primer chequeo de la ÚNICA generación: se reemplazaba a sí misma.
    let contador = AtomicU64::new(0);
    let previo = contador.fetch_add(1, Ordering::AcqRel);
    assert!(
        contador.load(Ordering::Acquire) > previo,
        "el predicado viejo se cumplía sin que nadie reclamara otra generación"
    );
}

#[tokio::test]
async fn rb_n1_el_bucle_corre_y_solo_lo_detiene_el_relevo() {
    let guardia = Arc::new(SincronizadorUnico::new());
    let vueltas = Arc::new(AtomicU64::new(0));
    let primera = guardia.reclamar();

    let tarea = {
        let guardia = Arc::clone(&guardia);
        let vueltas = Arc::clone(&vueltas);
        tokio::spawn(async move {
            // Sin temporizador en el bucle: en Windows `sleep(1 ms)` dura lo
            // que la granularidad global del reloj del SO (~15,6 ms salvo que
            // otro proceso pida 1 ms) y el contrato no puede depender de eso.
            let bucle = async {
                loop {
                    vueltas.fetch_add(1, Ordering::Relaxed);
                    tokio::task::yield_now().await;
                }
            };
            correr_mientras_vigente(&guardia, primera, Duration::from_millis(5), bucle).await
        })
    };

    // Mientras la generación esté vigente el bucle tiene que correr: se espera
    // a que dé 1 000 vueltas (con un plazo de 2 s) y la tarea sigue viva.
    const VUELTAS: u64 = 1_000;
    let plazo = Instant::now() + Duration::from_secs(2);
    while vueltas.load(Ordering::Relaxed) < VUELTAS && Instant::now() < plazo {
        tokio::time::sleep(Duration::from_millis(10)).await;
    }
    assert!(
        !tarea.is_finished(),
        "una generación vigente no puede darse por relevada (el defecto HOST-016)"
    );
    let antes_del_relevo = vueltas.load(Ordering::Relaxed);
    assert!(
        antes_del_relevo >= VUELTAS,
        "el bucle tiene que correr mientras está vigente: {antes_del_relevo} vueltas"
    );

    let segunda = guardia.reclamar();
    let fin = match tokio::time::timeout(Duration::from_secs(2), tarea).await {
        Ok(Ok(fin)) => fin,
        Ok(Err(e)) => panic!("la tarea del sincronizador falló: {e}"),
        Err(_) => panic!("el relevo debe detener a la generación anterior en un sondeo"),
    };
    assert_eq!(fin, FinSincronizador::Relevado);
    assert!(guardia.vigente(segunda));

    // Detenido de verdad: el contador ya no avanza.
    let tras_el_relevo = vueltas.load(Ordering::Relaxed);
    tokio::time::sleep(Duration::from_millis(50)).await;
    assert_eq!(vueltas.load(Ordering::Relaxed), tras_el_relevo);
}

/// Mundo sintético exacto: el servidor avanza 15 000 ms por ciclo y estampa
/// su hora en el punto medio del viaje; la pared local va 270 ms por detrás
/// del servidor; el oscilador monótono ADELANTA `deriva_us` µs por ciclo.
struct MundoSintetico {
    base: Instant,
    servidor_0: u64,
    deriva_us: u64,
}

impl MundoSintetico {
    const PERIODO_MS: u64 = 15_000;
    const RETRASO_PARED_MS: u64 = 270;
    const RTT_MS: u64 = 240;

    fn servidor(&self, k: u64) -> u64 {
        self.servidor_0 + Self::PERIODO_MS * k
    }

    /// Instante monótono en que el servidor estampa la hora del ciclo `k`.
    fn medio(&self, k: u64) -> Instant {
        self.base + Duration::from_micros(Self::PERIODO_MS * 1_000 * k + self.deriva_us * k)
    }

    fn medir(&self, estado: &mut EstadoReloj, k: u64, rtt: u64) -> MuestraNtp {
        let pared_medio = self.servidor(k) - Self::RETRASO_PARED_MS;
        let mitad = Duration::from_millis(rtt / 2);
        let enviado = match self.medio(k).checked_sub(mitad) {
            Some(i) => i,
            None => panic!("el reloj monótono de la prueba no admite restar {mitad:?}"),
        };
        let recibido = self.medio(k) + mitad;
        match estado.procesar(
            pared_medio - rtt / 2,
            pared_medio + rtt / 2,
            self.servidor(k),
            enviado,
            recibido,
        ) {
            Some(m) => m,
            None => panic!("una medición representable tiene que procesarse (ciclo {k})"),
        }
    }

    /// Error de la hora del host en el ciclo `k` (proyección − servidor): el
    /// host ancla `servidor(0) + error_arranque` al instante `medio(0)` y le
    /// suma la corrección publicada.
    fn error_del_host(&self, error_arranque: i64, k: u64, correccion_ms: i64) -> i64 {
        let transcurrido = self.medio(k).saturating_duration_since(self.medio(0));
        self.servidor(0) as i64 + error_arranque + transcurrido.as_millis() as i64 + correccion_ms
            - self.servidor(k) as i64
    }
}

#[test]
fn rb_n1_la_correccion_absorbe_el_arranque_acota_la_edad_y_sobrevive_al_relevo() {
    // 24 ppm medidos en esta máquina: 360 µs por ciclo de 15 s.
    let mundo = MundoSintetico {
        base: Instant::now() + Duration::from_secs(1),
        servidor_0: 1_794_000_000_000,
        deriva_us: 360,
    };
    // El offset de arranque viejo restaba la pared de la llegada: el ancla del
    // host nacía RTT/2 por detrás del servidor.
    let error_arranque = -((MundoSintetico::RTT_MS / 2) as i64);
    const UN_DIA: u64 = 5_760; // ciclos de 15 s
    const RELEVO: u64 = UN_DIA; // transición demo→mainnet tras un día de Demo
    const FIN: u64 = 3 * UN_DIA; // 72 h

    let mut compartido = EstadoReloj::new();
    compartido.fijar_ancla(mundo.servidor(0) as i64 + error_arranque, mundo.medio(0));
    let mut correccion = 0_i64;
    let mut peor_error_vivo = 0_i64;
    let mut correccion_en_el_relevo = 0_i64;
    // Contrafactual del diseño anterior: la generación del relevo ancla la
    // corrección a SU primera muestra (pared + offset recién medido).
    let mut ancla_propia = EstadoReloj::new();
    let mut correccion_propia = 0_i64;
    let mut peor_error_ancla_propia = 0_i64;

    for k in 0..=FIN {
        // Picos de congestión: la puerta de outliers los descarta y lo
        // publicado no cambia.
        if k > 0 && k % 97 == 0 {
            let pico = mundo.medir(&mut compartido, k, 600);
            assert!(!pico.aceptada, "un RTT de 600 ms no entra al filtro");
            assert_eq!(
                pico.correccion_ms,
                Some(correccion),
                "el outlier no mueve la corrección"
            );
            continue;
        }
        let m = mundo.medir(&mut compartido, k, MundoSintetico::RTT_MS);
        assert!(m.aceptada);
        assert_eq!(m.offset_pared_ms, MundoSintetico::RETRASO_PARED_MS as i64);
        correccion = match m.correccion_ms {
            Some(c) => c,
            None => panic!("con el ancla del host cada medición publica corrección (ciclo {k})"),
        };
        let error = mundo.error_del_host(error_arranque, k, correccion);
        if k == 0 {
            assert_eq!(
                error, 0,
                "el error del offset de arranque se absorbe en la primera muestra"
            );
        }
        peor_error_vivo = peor_error_vivo.max(error.abs());
        if k == RELEVO {
            correccion_en_el_relevo = correccion;
            ancla_propia.fijar_ancla(mundo.servidor(k) as i64, mundo.medio(k));
        }
        if k == RELEVO + 1 {
            assert!(
                (correccion - correccion_en_el_relevo).abs() <= 2,
                "el relevo no reinicia la corrección: {correccion_en_el_relevo} → {correccion} ms"
            );
        }
        if k > RELEVO {
            let mp = mundo.medir(&mut ancla_propia, k, MundoSintetico::RTT_MS);
            if let Some(c) = mp.correccion_ms {
                correccion_propia = c;
            }
            peor_error_ancla_propia = peor_error_ancla_propia.max(mundo.error_del_host(
                error_arranque,
                k,
                correccion_propia,
            ));
        }
    }

    let sin_bucle = mundo.error_del_host(error_arranque, FIN, 0);
    assert!(
        sin_bucle > 6_000,
        "sin corrección, 72 h a 24 ppm meten {sin_bucle} ms de reloj en la edad del dato"
    );
    assert!(
        peor_error_vivo <= 100,
        "con el bucle vivo el error queda acotado por el retardo del filtro: peor {peor_error_vivo} ms"
    );
    let acumulado_en_el_relevo = mundo.error_del_host(error_arranque, RELEVO, 0);
    assert!(
        peor_error_ancla_propia >= acumulado_en_el_relevo * 9 / 10,
        "un ancla propia por generación devuelve la deriva acumulada ({acumulado_en_el_relevo} ms) \
         a la edad del dato: peor {peor_error_ancla_propia} ms"
    );
}

#[test]
fn rb_n1_reanclar_reinicia_el_filtro_y_absorbe_el_ancla_nueva_en_una_muestra() {
    let mundo = MundoSintetico {
        base: Instant::now() + Duration::from_secs(1),
        servidor_0: 1_794_000_000_000,
        deriva_us: 360,
    };
    let mut estado = EstadoReloj::new();
    estado.fijar_ancla(mundo.servidor(0) as i64, mundo.medio(0));
    for k in 0..200 {
        mundo.medir(&mut estado, k, MundoSintetico::RTT_MS);
    }
    // El host vuelve a anclar con una estimación 500 ms adelantada.
    const SALTO_MS: i64 = 500;
    estado.fijar_ancla(mundo.servidor(200) as i64 + SALTO_MS, mundo.medio(200));
    let m = mundo.medir(&mut estado, 200, MundoSintetico::RTT_MS);
    let correccion = match m.correccion_ms {
        Some(c) => c,
        None => panic!("con ancla, la medición publica corrección"),
    };
    // Sin reiniciar, el Kalman tardaría ~1/K ≈ 170 ciclos (42 min) en
    // absorber el salto; reiniciado, la primera medición lo inicializa.
    assert_eq!(
        correccion, -SALTO_MS,
        "la corrección contra el ancla nueva se absorbe en la primera medición"
    );
}

#[test]
fn rb_n1_el_log_avisa_al_nacer_cada_hora_y_si_el_reloj_deja_de_corregirse() {
    assert_eq!(PERIODO_SINCRONIZACION, Duration::from_secs(15));
    assert_eq!(MUESTRAS_POR_HORA, 240);
    let base = MuestraNtp {
        aceptadas: 1,
        aceptada: true,
        rechazadas_seguidas: 0,
        rtt_ms: 240,
        offset_pared_ms: 270,
        correccion_ms: Some(120),
    };
    assert!(merece_aviso(&base), "la primera muestra");
    assert!(merece_aviso(&MuestraNtp {
        aceptadas: 2,
        ..base
    }));
    assert!(
        !merece_aviso(&MuestraNtp {
            aceptadas: 3,
            ..base
        }),
        "sin ruido después"
    );
    assert!(
        merece_aviso(&MuestraNtp {
            aceptadas: 240,
            ..base
        }),
        "una por hora"
    );
    assert!(!merece_aviso(&MuestraNtp {
        aceptadas: 241,
        ..base
    }));
    let rechazada = MuestraNtp {
        aceptada: false,
        rechazadas_seguidas: 59,
        rtt_ms: 600,
        ..base
    };
    assert!(!merece_aviso(&rechazada));
    assert!(
        merece_aviso(&MuestraNtp {
            rechazadas_seguidas: 60,
            ..rechazada
        }),
        "un cuarto de hora sin corregir el reloj se avisa"
    );
}

#[test]
fn rb_n1_el_host_usa_la_guardia_viva_y_publica_su_ancla() {
    let fuente = include_str!("../../../src/bin/god_engine.rs");
    assert!(
        !fuente.contains("NTP_SYNC_GENERATION"),
        "RB-N1: la guardia que se reemplazaba a sí misma no puede volver al host"
    );
    assert!(fuente.contains("execution_engine::ntp::SincronizadorUnico::new()"));
    let inicio = fuente
        .find("fn spawn_ntp_synchronizer(")
        .expect("el host conserva su lanzador único");
    let cuerpo = &fuente[inicio..];
    let cuerpo = &cuerpo[..cuerpo.find("\n}\n").unwrap_or(cuerpo.len())];
    for pieza in [
        "SINCRONIZADOR_NTP.reclamar()",
        "SINCRONIZADOR_NTP.estado()",
        "execution_engine::ntp::correr_mientras_vigente(",
        "execution_engine::ntp::merece_aviso(",
    ] {
        assert!(
            cuerpo.contains(pieza),
            "spawn_ntp_synchronizer debe usar {pieza}"
        );
    }
    assert!(
        fuente.matches("spawn_ntp_synchronizer(").count() >= 3,
        "arranque y transición demo→mainnet lanzan por el mismo camino"
    );
    // El ancla que publica el host es la MISMA pareja con la que mide la edad.
    let ancla = fuente
        .find("let epoch_baseline_ms = local_ms_now + ntp_offset_ms;")
        .expect("el host conserva su proyección monótona");
    let tras_el_ancla = &fuente[ancla..];
    let publicada = tras_el_ancla
        .find("SINCRONIZADOR_NTP.fijar_ancla(epoch_baseline_ms, instant_baseline);")
        .expect("RB-N1: el host publica su ancla al sincronizador");
    assert!(
        publicada < tras_el_ancla.find('\n').unwrap_or(0) + 400,
        "la publicación del ancla sigue a su cálculo, antes de medir ninguna edad"
    );
    assert!(
        fuente.contains("servidor - (antes + (local_ms - antes) / 2)"),
        "el offset del arranque se refiere al punto medio del viaje"
    );
}

/// SONDA VIVA (red real): el bucle de producción, con la guardia de
/// producción, contra el testnet de Binance, partiendo de un ancla con el
/// error de un arranque que cayó a la hora local. `cargo test -p
/// execution-engine --test rb_n1_reloj_vivo_contract -- --ignored --nocapture`.
#[tokio::test]
#[ignore = "red: consulta /fapi/v1/time del testnet de Binance durante ~50 s"]
async fn rb_n1_sonda_viva_contra_el_testnet() {
    use execution_engine::client::BinanceClient;
    use quantum_arena::GlobalArena;

    let cliente = Arc::new(BinanceClient::new(String::new(), true));
    let arena = GlobalArena::build_in_own_stack(100.0);
    let guardia = Arc::new(SincronizadorUnico::new());
    let muestras = Arc::new(std::sync::Mutex::new(Vec::<MuestraNtp>::new()));

    // Ancla con el error de un arranque sin respuesta: pared local tal cual.
    let pared_ahora = match std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH) {
        Ok(d) => d.as_millis() as i64,
        Err(e) => panic!("reloj de pared anterior a 1970: {e}"),
    };
    guardia.fijar_ancla(pared_ahora, Instant::now());
    let generacion = guardia.reclamar();

    let tarea = {
        let guardia = Arc::clone(&guardia);
        let muestras = Arc::clone(&muestras);
        let arena = Arc::clone(&arena);
        let cliente = Arc::clone(&cliente);
        tokio::spawn(async move {
            let observador = move |m: MuestraNtp| {
                println!("RB-N1 sonda: {m:?}");
                if let Ok(mut v) = muestras.lock() {
                    v.push(m);
                }
            };
            let sync = execution_engine::ntp::start_ntp_synchronizer(
                cliente,
                arena,
                guardia.estado(),
                observador,
            );
            correr_mientras_vigente(&guardia, generacion, Duration::from_millis(500), sync).await
        })
    };

    tokio::time::sleep(Duration::from_secs(50)).await;
    assert!(
        !tarea.is_finished(),
        "la generación vigente sigue corriendo"
    );
    let vistas = match muestras.lock() {
        Ok(v) => v.clone(),
        Err(e) => e.into_inner().clone(),
    };
    let aceptadas = vistas.iter().filter(|m| m.aceptada).count();
    assert!(
        aceptadas >= 3,
        "≥3 muestras aceptadas en 50 s (una cada 15 s): {vistas:?}"
    );
    let offset = arena.server_time_offset_ms.load(Ordering::Relaxed);
    assert_eq!(
        Some(offset),
        vistas.last().map(|m| m.offset_pared_ms),
        "la arena publica el último offset filtrado"
    );
    // Con el ancla en la pared, la corrección tiene que converger al offset
    // servidor − pared medido de forma independiente.
    let correccion = arena.server_time_correction_ms.load(Ordering::Relaxed);
    let t0 = std::time::SystemTime::now();
    let servidor = match cliente.fetch_server_time().await {
        Ok(s) => s as i64,
        Err(e) => panic!("el testnet no respondió la hora: {e}"),
    };
    let viaje = t0.elapsed().unwrap_or_default();
    let pared_medio = match (t0 + viaje / 2).duration_since(std::time::UNIX_EPOCH) {
        Ok(d) => d.as_millis() as i64,
        Err(e) => panic!("reloj de pared anterior a 1970: {e}"),
    };
    let independiente = servidor - pared_medio;
    println!(
        "RB-N1 sonda: offset de pared {offset} ms, corrección {correccion} ms, \
         offset independiente {independiente} ms (RTT {} ms)",
        viaje.as_millis()
    );
    assert!(
        (correccion - independiente).abs() <= 60,
        "la corrección absorbe el error del ancla: {correccion} ms frente a {independiente} ms"
    );

    guardia.reclamar();
    let fin = tokio::time::timeout(Duration::from_secs(3), tarea).await;
    assert!(
        matches!(fin, Ok(Ok(FinSincronizador::Relevado))),
        "el relevo detiene la sonda: {fin:?}"
    );
}
