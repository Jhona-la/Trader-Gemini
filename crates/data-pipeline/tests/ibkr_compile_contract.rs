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

/// I4: la cubeta admite 60, rechaza la 61 y repone 1 ficha cada 10 s.
#[test]
fn i4_token_bucket_respects_capacity_and_refill() {
    use data_pipeline::ibkr::pacing::TokenBucket;
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

/// I4: la cubeta de IBKR sale de las constantes (60 fichas, 600 s / 60 = 10 s).
#[test]
fn i4_token_bucket_ibkr_derivado_de_constantes() {
    use data_pipeline::ibkr::pacing::{TokenBucket, IBKR_HISTORICAL_REFILL_SECS, IBKR_HISTORICAL_WINDOW_SECS};
    use data_pipeline::ibkr::IBKR_HISTORICAL_MAX_PER_10MIN;
    assert_eq!(IBKR_HISTORICAL_WINDOW_SECS, 600);
    assert_eq!(IBKR_HISTORICAL_REFILL_SECS * IBKR_HISTORICAL_MAX_PER_10MIN as u64, IBKR_HISTORICAL_WINDOW_SECS);
    assert_eq!(TokenBucket::ibkr_historico().capacidad(), IBKR_HISTORICAL_MAX_PER_10MIN);
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