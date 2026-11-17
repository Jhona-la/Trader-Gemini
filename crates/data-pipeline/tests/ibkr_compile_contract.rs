// IBKR-I1..I8 (integrador, 2026-11-16): contrato de compilacion del conector
// IBKR de SOLO LECTURA (pedido junie_cli_pedido_conector_ibkr_solo_lectura_v2.md,
// revisado por QODER-CLI: APTO). Ninguna prueba de este archivo usa red ni el
// IB Gateway.

/// I1: los tipos de lectura de ibapi 2.7.0 son accesibles (pin exacto =2.7.0).
#[test]
fn i1_ibapi_dependency_compiles() {
    use ibapi::contracts::Contract;
    use ibapi::market_data::MarketDataType;
    use ibapi::Client;
    let _ = std::any::type_name::<Client>();
    let _ = std::any::type_name::<Contract>();
    let _ = std::any::type_name::<MarketDataType>();
}

/// I2: el modulo ibkr publica sus tipos de solo lectura y las constantes.
#[test]
fn i2_ibkr_module_exposes_read_only_types() {
    use data_pipeline::ibkr::{IbkrBar, IbkrGateway, IBKR_HISTORICAL_MAX_PER_10MIN};
    let _ = std::any::type_name::<IbkrGateway>();
    let _ = std::any::type_name::<IbkrBar>();
    assert_eq!(IBKR_HISTORICAL_MAX_PER_10MIN, 60);
}

/// I2: el modulo importa los tipos exactos de ibapi, no su prelude.
#[test]
fn i2_ibkr_module_no_usa_prelude() {
    let fuente = include_str!("../src/ibkr/mod.rs");
    // Armado por partes: el escaner de I3 tambien recorre tests/.
    let glob = concat!("ibapi::", "prel", "ude");
    assert!(!fuente.contains(glob), "src/ibkr/mod.rs no debe importar el prelude de ibapi");
    for tipo in ["use ibapi::Client;", "use ibapi::contracts::Contract;", "use ibapi::market_data::MarketDataType;"] {
        assert!(fuente.contains(tipo), "falta el import exacto {tipo}");
    }
}

/// Reposicion de una ficha de la cubeta de I4: 10 s. DERIVADO: 600 s / 60.
/// IBKR-I11: retirada de produccion (la admision historica es `RegistroVentana`
/// desde I10); vive aqui solo para el control de 119 y las pruebas de I4.
const IBKR_HISTORICAL_REFILL_SECS: u64 =
    data_pipeline::ibkr::pacing::IBKR_HISTORICAL_WINDOW_SECS / data_pipeline::ibkr::IBKR_HISTORICAL_MAX_PER_10MIN as u64;

/// Cubeta de fichas de I4: `capacidad` fichas, una ficha nueva cada `reposicion`.
///
/// IBKR-I11: copia exacta de la que estaba en `src/ibkr/pacing.rs` (I4..I10),
/// retirada de produccion. NO es la admision de IBKR (su peor caso es
/// C + ceil(W/r) - 1 = 119 por ventana de 600 s); se conserva solo como control
/// de las pruebas i4/i9/i10.
#[derive(Debug, Clone)]
struct TokenBucket {
    capacidad: usize,
    reposicion: std::time::Duration,
    fichas: usize,
    /// Instante desde el que se cuenta la proxima reposicion.
    ultima: Option<std::time::Instant>,
}

impl TokenBucket {
    /// Cubeta llena con `capacidad` fichas. Con `reposicion` cero no repone
    /// nunca (falla cerrado, sin panico).
    fn new(capacidad: usize, reposicion: std::time::Duration) -> Self {
        Self { capacidad, reposicion, fichas: capacidad, ultima: None }
    }

    fn reponer(&mut self, ahora: std::time::Instant) {
        let Some(ultima) = self.ultima else {
            self.ultima = Some(ahora);
            return;
        };
        if ahora <= ultima {
            return;
        }
        let paso = self.reposicion.as_nanos();
        if paso == 0 {
            return;
        }
        let transcurrido = ahora - ultima;
        let nuevas = usize::try_from(transcurrido.as_nanos() / paso).unwrap_or(usize::MAX);
        if nuevas == 0 {
            return;
        }
        self.fichas = self.fichas.saturating_add(nuevas).min(self.capacidad);
        let avance = u32::try_from(nuevas).ok().and_then(|n| self.reposicion.checked_mul(n));
        let siguiente = avance.and_then(|a| ultima.checked_add(a));
        if self.fichas == self.capacidad || siguiente.is_none() {
            self.ultima = Some(ahora);
        } else {
            self.ultima = siguiente;
        }
    }

    /// Toma una ficha en `ahora` si hay. Devuelve `false` sin tomar nada si la
    /// cubeta esta vacia.
    fn try_acquire_at(&mut self, ahora: std::time::Instant) -> bool {
        self.reponer(ahora);
        if self.fichas == 0 {
            return false;
        }
        self.fichas -= 1;
        true
    }

    /// Fichas disponibles en `ahora`.
    fn disponibles_at(&mut self, ahora: std::time::Instant) -> usize {
        self.reponer(ahora);
        self.fichas
    }
}

/// I11: la cubeta de I4 y su reposicion ya no existen en produccion. Falla si
/// alguno de los dos nombres vuelve a aparecer en `src/ibkr/` (codigo o
/// comentario). Los nombres se arman por partes para no depender de este archivo.
#[test]
fn i11_produccion_sin_cubeta_de_i4() {
    let fuentes = [
        ("src/ibkr/mod.rs", include_str!("../src/ibkr/mod.rs")),
        ("src/ibkr/pacing.rs", include_str!("../src/ibkr/pacing.rs")),
        ("src/ibkr/error_handler.rs", include_str!("../src/ibkr/error_handler.rs")),
        ("src/ibkr/cache.rs", include_str!("../src/ibkr/cache.rs")),
        ("src/ibkr/tape_writer.rs", include_str!("../src/ibkr/tape_writer.rs")),
    ];
    let prohibidos = [concat!("Token", "Bucket"), concat!("IBKR_HISTORICAL_", "REFILL_SECS")];
    let mut hallados = Vec::new();
    for (ruta, texto) in fuentes {
        for (i, linea) in texto.lines().enumerate() {
            for p in prohibidos {
                if linea.contains(p) {
                    hallados.push(format!("{ruta}:{} contiene {p}", i + 1));
                }
            }
        }
    }
    assert!(hallados.is_empty(), "la cubeta de I4 sigue en produccion: {hallados:?}");
}

/// I4: la cubeta admite 60, rechaza la 61 y repone 1 ficha cada 10 s.
#[test]
fn i4_token_bucket_respects_capacity_and_refill() {
    use std::time::{Duration, Instant};

    let t0 = Instant::now();
    let mut bucket = TokenBucket::new(60, Duration::from_secs(10));

    for _ in 0..60 {
        assert!(bucket.try_acquire_at(t0), "Deberia haber ficha disponible");
    }
    assert!(!bucket.try_acquire_at(t0), "La 61 debe fallar");
    assert!(!bucket.try_acquire_at(t0 + Duration::from_millis(9_999)), "Aun no repone a 9,999 s");
    assert!(bucket.try_acquire_at(t0 + Duration::from_secs(10)), "Repone 1 ficha tras 10 s");
    assert!(!bucket.try_acquire_at(t0 + Duration::from_secs(10)), "Solo se repuso 1");
    // A los 600 s de la ultima toma la cubeta vuelve a estar llena, y no pasa de 60.
    assert_eq!(bucket.disponibles_at(t0 + Duration::from_secs(10 + 600)), 60);
}

/// I4 (ajustada en I10): la admision historica de IBKR sale de las constantes
/// (C = 60 en W = 600 s; r = 600 s / 60 = 10 s sigue derivada). Desde I10 la
/// admision es `RegistroVentana`, no `TokenBucket`. Desde I11, r vive en este
/// archivo (retirada de produccion).
#[test]
fn i4_token_bucket_ibkr_derivado_de_constantes() {
    use data_pipeline::ibkr::pacing::{RegistroVentana, IBKR_HISTORICAL_WINDOW_SECS};
    use data_pipeline::ibkr::IBKR_HISTORICAL_MAX_PER_10MIN;
    use std::time::Duration;
    assert_eq!(IBKR_HISTORICAL_WINDOW_SECS, 600);
    assert_eq!(IBKR_HISTORICAL_REFILL_SECS * IBKR_HISTORICAL_MAX_PER_10MIN as u64, IBKR_HISTORICAL_WINDOW_SECS);
    let r = RegistroVentana::ibkr_historico();
    assert_eq!(r.capacidad(), IBKR_HISTORICAL_MAX_PER_10MIN);
    assert_eq!(r.ventana(), Duration::from_secs(IBKR_HISTORICAL_WINDOW_SECS));
}

#[test]
fn i4_pacing_enforces_min_interval() {
    use data_pipeline::ibkr::pacing::PacingGate;
    use std::time::{Duration, Instant};
    let gate = PacingGate::new(2500);
    assert_eq!(gate.min_interval_ms(), 2500);

    let mut gate = PacingGate::ibkr();
    let t0 = Instant::now();
    assert!(gate.try_send_at(t0));
    assert!(!gate.try_send_at(t0 + Duration::from_millis(2_499)), "antes de 2500 ms no envia");
    assert!(gate.try_send_at(t0 + Duration::from_millis(2_500)));
    // Con el espaciador nunca caben 6 envios en 2 s (regla oficial de rafaga).
    let mut gate = PacingGate::ibkr();
    let enviados = (0..2_000u64).filter(|ms| gate.try_send_at(t0 + Duration::from_millis(*ms))).count();
    assert!(enviados < 6, "{enviados} envios en 2 s");
}

/// I4: como mucho 50 peticiones en vuelo.
#[test]
fn i4_in_flight_maximo_50() {
    use data_pipeline::ibkr::pacing::{InFlightCounter, IBKR_MAX_IN_FLIGHT_HISTORICAL};
    assert_eq!(IBKR_MAX_IN_FLIGHT_HISTORICAL, 50);
    let mut c = InFlightCounter::ibkr();
    for _ in 0..50 {
        assert!(c.try_begin());
    }
    assert!(!c.try_begin(), "la 51 en vuelo debe esperar");
    c.finish();
    assert!(c.try_begin(), "al cerrar una queda hueco");
    assert_eq!(c.en_vuelo(), 50);
}

/// I4: al abrir la linea 81 se desaloja la menos usada.
#[test]
fn i4_lineas_streaming_lru_80() {
    use data_pipeline::ibkr::pacing::StreamingLines;
    use std::time::{Duration, Instant};
    let t0 = Instant::now();
    let mut l = StreamingLines::ibkr();
    for i in 0..80u64 {
        assert_eq!(l.open_at(&format!("L{i}"), t0 + Duration::from_secs(i)), None);
    }
    // L0 se vuelve a usar: la menos usada pasa a ser L1.
    assert_eq!(l.open_at("L0", t0 + Duration::from_secs(100)), None);
    assert_eq!(l.open_at("L80", t0 + Duration::from_secs(101)), Some("L1".to_string()));
    assert_eq!(l.abiertas(), 80);
    assert!(l.contiene("L0") && !l.contiene("L1"));
}
/// I5: hit a 29 s y miss a 31 s con la clave de 6 campos.
#[test]
fn i5_cache_deduplicates_identical_requests() {
    use data_pipeline::ibkr::cache::IbkrCache;
    use std::time::{Duration, Instant};

    let t0 = Instant::now();
    let mut cache = IbkrCache::new(Duration::from_secs(30));
    let clave = ("ES", "FUT", "CME", "1 min", "", "TRADES");
    let barras = vec![1.0, 2.0, 3.0];

    cache.insert_at(clave, barras.clone(), t0);
    assert_eq!(cache.get_at(clave, t0 + Duration::from_secs(29)), Some(&barras), "Hit a 29 s");
    assert!(cache.get_at(clave, t0 + Duration::from_secs(31)).is_none(), "Miss a 31 s");
}

/// I5: cada uno de los 6 campos distingue la clave; el TTL de IBKR es 30 s.
#[test]
fn i5_cache_clave_de_seis_campos_y_ttl_ibkr() {
    use data_pipeline::ibkr::cache::IbkrCache;
    use data_pipeline::ibkr::IBKR_DEDUP_CACHE_TTL_SECS;
    use std::time::{Duration, Instant};

    let t0 = Instant::now();
    let mut cache: IbkrCache<u8> = IbkrCache::ibkr();
    assert_eq!(cache.ttl(), Duration::from_secs(IBKR_DEDUP_CACHE_TTL_SECS));
    let base = ("ES", "FUT", "CME", "1 min", "20261116 16:00:00 US/Eastern", "TRADES");
    cache.insert_at(base, 1, t0);
    let variantes = [
        ("NQ", base.1, base.2, base.3, base.4, base.5),
        (base.0, "STK", base.2, base.3, base.4, base.5),
        (base.0, base.1, "GLOBEX", base.3, base.4, base.5),
        (base.0, base.1, base.2, "5 mins", base.4, base.5),
        (base.0, base.1, base.2, base.3, "", base.5),
        (base.0, base.1, base.2, base.3, base.4, "MIDPOINT"),
    ];
    for v in variantes {
        assert!(cache.get_at(v, t0).is_none(), "{v:?} no debe chocar con la clave base");
    }
    assert_eq!(cache.get_at(base, t0), Some(&1));
    // Al insertar despues del TTL, lo vencido sale de la memoria.
    cache.insert_at(variantes[0], 2, t0 + Duration::from_secs(IBKR_DEDUP_CACHE_TTL_SECS + 1));
    assert_eq!(cache.len(), 1);
}
fn dir_temporal(nombre: &str) -> std::path::PathBuf {
    let d = std::env::temp_dir().join(format!("ibkr_tape_{nombre}_{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&d);
    std::fs::create_dir_all(&d).expect("crear carpeta temporal");
    d
}

fn barra(segundos_unix: u64, simbolo: &str) -> data_pipeline::ibkr::IbkrBar {
    data_pipeline::ibkr::IbkrBar {
        timestamp: std::time::UNIX_EPOCH + std::time::Duration::from_secs(segundos_unix),
        symbol: simbolo.to_string(),
        open: 5000.0,
        high: 5010.0,
        low: 4995.0,
        close: 5005.0,
        volume: 100,
        wap: 5003.0,
        count: 50,
    }
}

/// I6: escribe ibkr_YYYYMMDD.csv.gz y su companero .sha256.
#[test]
fn i6_tape_writer_produces_valid_csv_with_sha256() {
    use data_pipeline::ibkr::tape_writer::TapeWriter;
    use data_pipeline::ibkr::IbkrBar;
    use std::time::SystemTime;

    let tmp_dir = dir_temporal("i6");
    let mut writer = TapeWriter::new(&tmp_dir);

    let bars = vec![IbkrBar {
        timestamp: SystemTime::now(),
        symbol: "ES".to_string(),
        open: 5000.0,
        high: 5010.0,
        low: 4995.0,
        close: 5005.0,
        volume: 100,
        wap: 5003.0,
        count: 50,
    }];
    writer.write_bars(&bars).expect("Escritura exitosa");

    let csv_gz = std::fs::read_dir(&tmp_dir)
        .unwrap()
        .flatten()
        .map(|e| e.path())
        .find(|p| p.extension().is_some_and(|x| x == "gz"))
        .expect("debe existir ibkr_YYYYMMDD.csv.gz");
    assert!(csv_gz.file_name().unwrap().to_string_lossy().starts_with("ibkr_"));

    // El companero es <archivo>.csv.gz.sha256. (El pedido usaba
    // with_extension("csv.gz.sha256"), que da ibkr_X.csv.csv.gz.sha256.)
    let sha_file = csv_gz.with_extension("gz.sha256");
    assert!(sha_file.exists(), "Debe existir el archivo .sha256 companion");

    // Observacion 1 de QODER: el hash se recalcula, no basta con que exista.
    let csv = TapeWriter::leer_csv(&csv_gz).expect("descomprimir");
    let firma = std::fs::read_to_string(&sha_file).expect("leer firma");
    assert_eq!(firma.trim(), data_pipeline::ibkr::tape_writer::sha256_hex(csv.as_bytes()));

    std::fs::remove_dir_all(&tmp_dir).ok();
}

/// I6: esquema, dia UTC, filas que se agregan y firma que se rehace.
#[test]
fn i6_tape_writer_esquema_dia_utc_y_agregado() {
    use data_pipeline::ibkr::tape_writer::{dia_utc, sha256_hex, timestamp_utc, TapeWriter, IBKR_CINTA_CABECERA};

    assert_eq!(IBKR_CINTA_CABECERA, "timestamp_utc,symbol,open,high,low,close,volume,wap,count");
    // 2026-11-16T23:59:59Z y 2024-02-29T00:00:00Z (bisiesto).
    let fin_dia = std::time::UNIX_EPOCH + std::time::Duration::from_secs(1_794_873_599);
    assert_eq!(dia_utc(fin_dia).unwrap(), "20261116");
    assert_eq!(timestamp_utc(fin_dia).unwrap(), "2026-11-16T23:59:59Z");
    let bisiesto = std::time::UNIX_EPOCH + std::time::Duration::from_secs(1_709_164_800);
    assert_eq!(timestamp_utc(bisiesto).unwrap(), "2024-02-29T00:00:00Z");

    let dir = dir_temporal("i6b");
    let mut w = TapeWriter::new(&dir);
    // Dos barras de dias distintos: dos archivos.
    let tocados = w.write_bars(&[barra(1_794_873_599, "ES"), barra(1_794_873_600, "NQ")]).unwrap();
    assert_eq!(tocados.len(), 2);
    let dia = w.ruta_dia("20261116");
    assert_eq!(
        TapeWriter::leer_csv(&dia).unwrap(),
        format!("{IBKR_CINTA_CABECERA}\n2026-11-16T23:59:59Z,ES,5000,5010,4995,5005,100,5003,50\n")
    );
    // Otra escritura del mismo dia agrega la fila y rehace la firma.
    w.write_bars(&[barra(1_794_873_598, "CL")]).unwrap();
    let csv = TapeWriter::leer_csv(&dia).unwrap();
    assert_eq!(csv.lines().count(), 3, "cabecera + 2 filas: {csv}");
    let firma = std::fs::read_to_string(TapeWriter::ruta_firma(&dia)).unwrap();
    assert_eq!(firma.trim(), sha256_hex(csv.as_bytes()));
    // Un simbolo con coma romperia el esquema: se rechaza.
    assert!(w.write_bars(&[barra(1_794_873_598, "E,S")]).is_err());

    std::fs::remove_dir_all(&dir).ok();
}
/// I7: un 162 de pacing activa el freno de 65 s.
#[test]
fn i7_error_162_triggers_circuit_breaker() {
    use data_pipeline::ibkr::error_handler::ErrorHandler;
    use std::time::Duration;

    let mut handler = ErrorHandler::new();
    handler.receive_error(162, "Pacing violation");

    assert!(handler.is_cooling_down(), "Deberia estar en enfriamiento");
    assert_eq!(handler.cooldown_duration(), Duration::from_secs(65));
}

/// I7: 65 s parado, luego una ventana (I9) a media tasa, luego normal; lee el texto del 162.
#[test]
fn i7_freno_fases_y_causas() {
    use data_pipeline::ibkr::error_handler::{
        clasificar, CausaIbkr, ErrorHandler, EstadoFreno, IBKR_PACING_RECOVERY_HALF_RATE_SECS,
    };
    use data_pipeline::ibkr::IBKR_PACING_BREAKER_COOLDOWN_SECS;
    use std::time::{Duration, Instant};

    let t0 = Instant::now();
    let s = Duration::from_secs;
    let mut h = ErrorHandler::new();
    assert_eq!(h.estado_at(t0), EstadoFreno::Normal);
    assert_eq!(
        h.receive_error_at(162, "Historical Market Data Service error message:API historical data query cancelled: Pacing violation", t0),
        CausaIbkr::PacingHistorico
    );
    let fin = IBKR_PACING_BREAKER_COOLDOWN_SECS;
    let media = IBKR_PACING_RECOVERY_HALF_RATE_SECS;
    assert_eq!(h.estado_at(t0 + s(fin) - Duration::from_millis(1)), EstadoFreno::Cooldown);
    assert_eq!(h.factor_tasa_at(t0 + s(1)), 0.0);
    assert_eq!(h.estado_at(t0 + s(fin)), EstadoFreno::MediaTasa);
    assert_eq!(h.factor_tasa_at(t0 + s(fin)), 0.5);
    assert_eq!(h.estado_at(t0 + s(fin + media)), EstadoFreno::Normal);
    assert_eq!(h.half_rate_duration(), s(media));
    assert_eq!(h.eventos().len(), 1);
    assert_eq!(h.eventos()[0].enfriamiento, s(65));

    // Un pacing nuevo durante el enfriamiento lo alarga.
    h.receive_error_at(420, "streaming pacing", t0 + s(30));
    assert!(h.is_cooling_down_at(t0 + s(fin + 1)));

    // El texto del 162 decide: permisos y sin datos no frenan; lo desconocido si.
    assert_eq!(clasificar(162, "No market data permissions for NYSE STK"), CausaIbkr::SinPermisos);
    assert_eq!(clasificar(162, "HMDS query returned no data: ESZ6@CME Trades"), CausaIbkr::SinDatos);
    assert_eq!(clasificar(162, "algo nuevo"), CausaIbkr::HistoricoDesconocido);
    assert_eq!(clasificar(101, "Max number of tickers has been reached"), CausaIbkr::ExcesoLineas);
    assert_eq!(clasificar(100, "Max rate of messages per second has been exceeded"), CausaIbkr::Otro);
    let mut h2 = ErrorHandler::new();
    h2.receive_error_at(162, "No market data permissions for NYSE STK", t0);
    h2.receive_error_at(101, "Max number of tickers has been reached", t0);
    assert_eq!(h2.estado_at(t0), EstadoFreno::Normal);
    assert_eq!(h2.lineas_excedidas(), 1);
    h2.receive_error_at(162, "algo nuevo", t0);
    assert_eq!(h2.estado_at(t0), EstadoFreno::Cooldown, "162 desconocido falla cerrado");
}
/// I8: todo el modulo es accesible desde data_pipeline::ibkr, con reexportes.
#[test]
fn i8_full_ibkr_module_integration() {
    use data_pipeline::ibkr::{
        cache::IbkrCache, error_handler::ErrorHandler, tape_writer::TapeWriter, IbkrBar,
        IbkrGateway, IBKR_DEDUP_CACHE_TTL_SECS, IBKR_HISTORICAL_MAX_PER_10MIN, IBKR_MAX_STREAMING_LINES_BUDGET,
        IBKR_MIN_PACING_INTERVAL_MS, IBKR_PACING_BREAKER_COOLDOWN_SECS,
    };
    assert_eq!(IBKR_HISTORICAL_MAX_PER_10MIN, 60);
    assert_eq!(IBKR_MIN_PACING_INTERVAL_MS, 2500);
    assert_eq!(IBKR_DEDUP_CACHE_TTL_SECS, 30);
    assert_eq!(IBKR_MAX_STREAMING_LINES_BUDGET, 80);
    assert_eq!(IBKR_PACING_BREAKER_COOLDOWN_SECS, 65);
    let _ = std::any::type_name::<(IbkrGateway, IbkrBar, IbkrCache<u8>, TapeWriter, ErrorHandler)>();

    // Reexportes en la raiz del modulo: los mismos tipos. I11: la cubeta de I4
    // ya no es API publica (lo vigila i11_produccion_sin_cubeta_de_i4).
    use data_pipeline::ibkr as m;
    assert_eq!(std::any::type_name::<m::IbkrCache<u8>>(), std::any::type_name::<IbkrCache<u8>>());
    assert_eq!(std::any::type_name::<m::TapeWriter>(), std::any::type_name::<TapeWriter>());
    assert_eq!(std::any::type_name::<m::ErrorHandler>(), std::any::type_name::<ErrorHandler>());
    let _ = std::any::type_name::<(m::PacingGate, m::InFlightCounter, m::StreamingLines, m::ClaveCache, m::CausaIbkr, m::EstadoFreno)>();
    // I10: la admision historica tambien se reexporta.
    assert_eq!(
        std::any::type_name::<m::RegistroVentana>(),
        std::any::type_name::<data_pipeline::ibkr::pacing::RegistroVentana>()
    );
    assert_eq!(m::tipos_ibapi_usados().len(), 3);
}
/// I4 (RB-P1): los casos limite fallan cerrados sin panico (release = panic = abort).
#[test]
fn i4_casos_limite_sin_panico() {
    use data_pipeline::ibkr::pacing::StreamingLines;
    use std::time::{Duration, Instant};
    let t0 = Instant::now();
    let mut b = TokenBucket::new(1, Duration::ZERO);
    assert!(b.try_acquire_at(t0));
    assert!(!b.try_acquire_at(t0 + Duration::from_secs(3600)), "reposicion cero no repone nunca");
    let mut l = StreamingLines::new(0);
    assert_eq!(l.open_at("ES", t0), Some("ES".to_string()), "techo cero: no se abre");
    assert_eq!(l.abiertas(), 0);
}

/// I10: admision historica que mide `i9_peor_ventana` (registro de I10 o la
/// cubeta vieja de I4, como control).
trait I9Admision {
    fn disponibles_at(&mut self, t: std::time::Instant) -> usize;
    fn try_acquire_at(&mut self, t: std::time::Instant) -> bool;
}
impl I9Admision for TokenBucket {
    fn disponibles_at(&mut self, t: std::time::Instant) -> usize {
        TokenBucket::disponibles_at(self, t)
    }
    fn try_acquire_at(&mut self, t: std::time::Instant) -> bool {
        TokenBucket::try_acquire_at(self, t)
    }
}
impl I9Admision for data_pipeline::ibkr::pacing::RegistroVentana {
    fn disponibles_at(&mut self, t: std::time::Instant) -> usize {
        data_pipeline::ibkr::pacing::RegistroVentana::disponibles_at(self, t)
    }
    fn try_acquire_at(&mut self, t: std::time::Instant) -> bool {
        data_pipeline::ibkr::pacing::RegistroVentana::try_acquire_at(self, t)
    }
}

/// Cubeta vieja de I4 con los limites de IBKR (60 fichas, una cada 10 s).
fn i9_cubeta_vieja() -> TokenBucket {
    use data_pipeline::ibkr::IBKR_HISTORICAL_MAX_PER_10MIN;
    TokenBucket::new(IBKR_HISTORICAL_MAX_PER_10MIN, std::time::Duration::from_secs(IBKR_HISTORICAL_REFILL_SECS))
}

/// I9: emisor voraz con reloj inyectado (paso de 500 ms) sobre la admision y el
/// espaciador reales. `factor(t)` es la fase del freno: 0 no envia; < 1 envia
/// con el doble de espaciado (acople mas favorable posible, que el codigo no
/// hace); 1 normal. Devuelve el maximo de envios en una ventana movil
/// semiabierta [t, t + W).
fn i9_peor_ventana(
    mut cubeta: impl I9Admision,
    factor: impl Fn(std::time::Duration) -> f64,
    horizonte: std::time::Duration,
) -> usize {
    use data_pipeline::ibkr::pacing::{PacingGate, IBKR_HISTORICAL_WINDOW_SECS};
    use data_pipeline::ibkr::IBKR_MIN_PACING_INTERVAL_MS;
    use std::time::{Duration, Instant};
    let t0 = Instant::now();
    let paso = Duration::from_millis(500);
    let doble = Duration::from_millis(2 * IBKR_MIN_PACING_INTERVAL_MS);
    let mut gate = PacingGate::ibkr();
    let mut envios: Vec<Duration> = Vec::new();
    let mut t = Duration::ZERO;
    while t < horizonte {
        let f = factor(t);
        let espera_media = f < 1.0 && envios.last().is_some_and(|u| t - *u < doble);
        if f > 0.0 && !espera_media && cubeta.disponibles_at(t0 + t) > 0 && gate.try_send_at(t0 + t) {
            assert!(cubeta.try_acquire_at(t0 + t));
            envios.push(t);
        }
        t += paso;
    }
    let w = Duration::from_secs(IBKR_HISTORICAL_WINDOW_SECS);
    let (mut j, mut max) = (0usize, 0usize);
    for i in 0..envios.len() {
        while j < envios.len() && envios[j] < envios[i] + w {
            j += 1;
        }
        max = max.max(j - i);
    }
    max
}

/// I9: la media tasa dura una ventana historica completa (T_m = W), no 60 s.
#[test]
fn i9_media_tasa_dura_una_ventana_historica() {
    use data_pipeline::ibkr::error_handler::{ErrorHandler, EstadoFreno, IBKR_PACING_RECOVERY_HALF_RATE_SECS};
    use data_pipeline::ibkr::pacing::IBKR_HISTORICAL_WINDOW_SECS;
    use data_pipeline::ibkr::IBKR_PACING_BREAKER_COOLDOWN_SECS;
    use std::time::{Duration, Instant};
    assert_eq!(IBKR_PACING_RECOVERY_HALF_RATE_SECS, IBKR_HISTORICAL_WINDOW_SECS, "T_m debe ser W");
    let t0 = Instant::now();
    let mut h = ErrorHandler::new();
    h.receive_error_at(162, "Pacing violation", t0);
    let fin_media = Duration::from_secs(IBKR_PACING_BREAKER_COOLDOWN_SECS + IBKR_HISTORICAL_WINDOW_SECS);
    assert_eq!(h.estado_at(t0 + fin_media - Duration::from_millis(1)), EstadoFreno::MediaTasa);
    assert_eq!(h.estado_at(t0 + fin_media), EstadoFreno::Normal);
}

/// I9, invertida en I10: la invariante "nunca mas de C por ventana movil W" la
/// cumple ahora la admision (`RegistroVentana`): el peor caso es exactamente
/// C = 60, con o sin freno y para todo T_m. La cubeta vieja de I4 daba
/// C + ceil(W/r) - 1 = 119 (control: `i10_control_cubeta_vieja_119`).
#[test]
fn i9_invariante_ventana_movil_depende_de_la_cubeta_no_de_t_m() {
    use data_pipeline::ibkr::error_handler::ErrorHandler;
    use data_pipeline::ibkr::pacing::RegistroVentana;
    use data_pipeline::ibkr::{IBKR_HISTORICAL_MAX_PER_10MIN, IBKR_PACING_BREAKER_COOLDOWN_SECS};
    use std::time::{Duration, Instant};
    let c = IBKR_HISTORICAL_MAX_PER_10MIN;
    let n_max = c;
    assert_eq!(n_max, 60);
    let s = Duration::from_secs;
    let tf = IBKR_PACING_BREAKER_COOLDOWN_SECS;

    // Sin freno.
    assert_eq!(i9_peor_ventana(RegistroVentana::ibkr_historico(), |_| 1.0, s(1800)), n_max);

    // Freno en t = 0 con la cubeta llena (peor transicion), para varios T_m.
    for tm in [0u64, 60, 600, 1200] {
        let fase = move |t: Duration| {
            if t < s(tf) {
                0.0
            } else if t < s(tf + tm) {
                0.5
            } else {
                1.0
            }
        };
        let peor = i9_peor_ventana(RegistroVentana::ibkr_historico(), fase, s(tf + tm + 1800));
        assert_eq!(peor, n_max, "T_m = {tm} s cambio el peor caso");
    }

    // El freno real, con su T_m.
    let t0 = Instant::now();
    let mut h = ErrorHandler::new();
    h.receive_error_at(162, "Pacing violation", t0);
    let real = i9_peor_ventana(RegistroVentana::ibkr_historico(), |t| h.factor_tasa_at(t0 + t), s(3600));
    assert_eq!(real, n_max);
}

/// I10 (control del medidor): con la cubeta vieja de I4 el mismo emisor voraz
/// mide C + ceil(W/r) - 1 = 119 por ventana. Si el medidor no viera excesos,
/// la prueba anterior no probaria nada.
#[test]
fn i10_control_cubeta_vieja_119() {
    use data_pipeline::ibkr::pacing::IBKR_HISTORICAL_WINDOW_SECS;
    use data_pipeline::ibkr::IBKR_HISTORICAL_MAX_PER_10MIN;
    use std::time::Duration;
    let (c, w, r) = (IBKR_HISTORICAL_MAX_PER_10MIN, IBKR_HISTORICAL_WINDOW_SECS, IBKR_HISTORICAL_REFILL_SECS);
    let n_vieja = c + (w.div_ceil(r) as usize) - 1;
    assert_eq!(n_vieja, 119);
    assert_eq!(i9_peor_ventana(i9_cubeta_vieja(), |_| 1.0, Duration::from_secs(1800)), n_vieja);
}

/// I11: la admision historica de IBKR es RegistroVentana. Si alguien vuelve a
/// implementar `ibkr_historico()` sobre la cubeta de I4, el peor caso mide 119
/// y esta prueba falla (debe ser 60 = C).
#[test]
fn i11_admision_ibkr_no_es_la_cubeta_de_i4() {
    use data_pipeline::ibkr::pacing::RegistroVentana;
    use std::time::Duration;
    assert_eq!(
        i9_peor_ventana(RegistroVentana::ibkr_historico(), |_| 1.0, Duration::from_secs(1800)),
        60
    );
}

/// I10: el registro admite C en (t - W, t], rechaza la C+1 y la primera marca
/// sale justo en t + W (borde abierto); el reloj hacia atras no admite.
#[test]
fn i10_registro_ventana_borde_exacto() {
    use data_pipeline::ibkr::pacing::RegistroVentana;
    use std::time::{Duration, Instant};
    let t0 = Instant::now();
    let ms = Duration::from_millis;
    let mut r = RegistroVentana::new(3, Duration::from_secs(10));
    assert!(r.try_acquire_at(t0));
    assert!(r.try_acquire_at(t0 + ms(1_000)));
    assert!(r.try_acquire_at(t0 + ms(2_000)));
    assert!(!r.try_acquire_at(t0 + ms(2_000)), "la C+1 en la ventana no entra");
    assert!(!r.try_acquire_at(t0 + ms(9_999)), "t0 sigue en (t - W, t]");
    assert_eq!(r.disponibles_at(t0 + ms(9_999)), 0);
    assert_eq!(r.disponibles_at(t0 + ms(10_000)), 1, "t0 sale en t0 + W");
    assert!(r.try_acquire_at(t0 + ms(10_000)));
    assert!(!r.try_acquire_at(t0 + ms(10_500)));
    assert!(!r.try_acquire_at(t0 + ms(5_000)), "reloj hacia atras: falla cerrado");
    assert_eq!(r.disponibles_at(t0 + ms(5_000)), 0);
    assert_eq!(r.disponibles_at(t0 + ms(30_000)), 3);
}

/// I10 (RB-P1): C = 0 o W = 0 fallan cerrados, sin panico.
#[test]
fn i10_registro_casos_limite_sin_panico() {
    use data_pipeline::ibkr::pacing::RegistroVentana;
    use std::time::{Duration, Instant};
    let t0 = Instant::now();
    let mut sin_capacidad = RegistroVentana::new(0, Duration::from_secs(600));
    assert!(!sin_capacidad.try_acquire_at(t0), "C = 0 no admite");
    assert_eq!(sin_capacidad.disponibles_at(t0 + Duration::from_secs(3600)), 0);
    let mut sin_ventana = RegistroVentana::new(60, Duration::ZERO);
    assert!(!sin_ventana.try_acquire_at(t0), "W = 0 no admite (la ventana vacia no limitaria)");
    assert!(!sin_ventana.try_acquire_at(t0 + Duration::from_secs(3600)));
    assert_eq!(sin_ventana.disponibles_at(t0), 0);
    let mut enorme = RegistroVentana::new(usize::MAX, Duration::MAX);
    assert!(enorme.try_acquire_at(t0), "C y W enormes: sin desborde ni reserva de memoria");
    assert_eq!(enorme.disponibles_at(t0), usize::MAX - 1);
}