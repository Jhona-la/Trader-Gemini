//! EXPORTADOR DEL VECTOR DE SERVICIO (D-753).
//!
//! PARA QUÉ SIRVE: volcar a CSV, tick a tick, EL MISMO vector de 48
//! dimensiones que `GodEngineCore` entrega al bosque en producción. Es la
//! herramienta con la que se INSPECCIONA la paridad: si un modelo satura en
//! servicio, aquí se ve qué dimensión llega con un soporte que el
//! entrenamiento no tenía.
//!
//! ── QUÉ ESTABA MAL ──────────────────────────────────────────────────────
//!
//! 1. PARIDAD ROTA, LA MISMA QUE EN `train_forest`. Este binario alimentaba
//!    un `StatefulEngine` SUELTO con tres llamadas por tick
//!    (`process_tick` / `update_trade_flow` / `update_ofi`) mientras el motor
//!    vivo pasa cada tick por `GodEngineCore::process_event` DOS veces —depth
//!    y trade—, y dentro de cada una vuelve a llamar a `process_tick` con un
//!    volumen que NO es el del trade (`clamp((bq+aq)·0,005; 0,01; 10)`,
//!    D-119), a `update_ofi` y a `update_macro_features`. Las mismas
//!    dimensiones salían con distribuciones distintas: las EMAs de precio al
//!    doble de velocidad, el CVD sobre otro nocional, la FFT y la entropía
//!    intercalando ceros. Un CSV así no describe lo que el motor sirve.
//!
//! 2. MACRO INVENTADO. Las columnas 40..54 eran literales plausibles
//!    —`0.50` de Fear & Greed, `1.04` de DXY, `0.75` de VIX, `1.05` de
//!    US10Y…— presentados como datos. El propio motor prohíbe exactamente
//!    eso (C-02: «toda feature SIN PRODUCTOR en vivo se sirve como 0.0
//!    (centinela "sin dato"), NUNCA como un literal plausible»), porque un
//!    número que parece real contamina en silencio. Y el bloque macro que el
//!    bosque consume no son esas 14 columnas: son CUATRO, las de
//!    `macro_ml_features` (VIX, SP500, DXY, NASDAQ) sobre los niveles
//!    as-of t−1. El exportador no tenía ninguna de ellas.
//!
//! 3. ETIQUETA DE OTRA GEOMETRÍA. La barrera triple de aquí comprobaba el SL
//!    del corto (+0,18 %) ANTES del TP del largo (+0,36 %), de modo que el TP
//!    era código inalcanzable y todo toque de +0,18 % contaba como victoria:
//!    una barrera SIMÉTRICA ±0,18 % —moneda al aire tras comisiones— en vez
//!    del trade asimétrico RR 2:1 que el motor ejecuta. Es el defecto que
//!    HOST-010 corrigió en `train_forest` y que aquí seguía vivo. Además el
//!    horizonte era de 500 TICKS, no de reloj (con ~75 ms/tick son ~2 s de
//!    mercado: las barreras no se tocan casi nunca), y el relleno por
//!    «retorno terminal > 0,10 %» era una tercera geometría más.
//!
//! ── QUÉ GARANTIZA EL ARREGLO ────────────────────────────────────────────
//!
//! El exportador reproduce el tape POR EL CAMINO DEL MOTOR —`process_event`,
//! depth y trade, libro simulado alrededor del punto medio con suelo de medio
//! tick medido en el propio tape, lado agresor `aq > bq` y cantidad
//! `|bq − aq|`, igual que `audit_forensic_backtest`— y escribe las 48
//! dimensiones del contrato, ni una más. No inventa macro: los cuatro
//! niveles salen de `data/macro/*.csv` con join as-of t−1 estricto, o la
//! fila se marca como sin macro. Y NO ETIQUETA: etiquetar es trabajo de
//! `train_forest`, que es donde vive la geometría del trade real; un segundo
//! etiquetador es un segundo contrato, y dos contratos es como se llegó aquí.
//!
//! Y la paridad se DEMUESTRA fila a fila, no se anuncia: antes de escribir
//! cada línea el vector se compara dimensión a dimensión contra los
//! productores del propio motor (`build_54d_tensor` para [0..34),
//! `macro_ml_features` para [44..48)) con tolerancia CERO, y el binario
//! aborta si alguna difiere. Los canales que un tape de aggTrades no puede
//! producir —liquidaciones (dim [9]) y libro L2 (dims [4] [5] [10])— están
//! declarados en `FEATURES_DEAD_IN_SERVE` y es el MOTOR quien los deja en 0
//! a ambos lados; aquí sólo se verifica que sigue haciéndolo.
//!
//! Uso: feature_exporter <SÍMBOLO> [--in data/SIMBOLO_ticks.bin]
//!                       [--out data/SIMBOLO_FEATURES.csv]
//!                       [--stride-ms 0] [--calentamiento-ms 43200000]

use god_engine_core::ml_inference::{macro_ml_features, NanoForest};
use god_engine_core::stateful_engine::{StatefulEngine, FEATURES_DEAD_IN_SERVE};
use god_engine_core::GodEngineCore;
use quantum_arena::symbol_registry::SymbolSpec;
use std::fs::File;
use std::io::{BufWriter, Write};
use std::sync::atomic::Ordering;

/// Contrato de anchura del vector ML: el mismo que valida el cargador de
/// modelos. Exportar otra anchura sería describir otro motor.
const DIM_VECTOR: usize = NanoForest::ML_VECTOR_DIM;

#[derive(Debug, Clone, Copy)]
#[repr(C)]
struct BinTick {
    pub timestamp: u64,
    pub bid_price: f64,
    pub ask_price: f64,
    pub bid_qty: f64,
    pub ask_qty: f64,
}

/// Ensamblado del vector ML **tal y como lo hace el servicio**: 34
/// universales ⊕ 10 espectrales ⊕ 4 macro, con el saneo de no finitos a 0.0
/// que el motor aplica para que un feed ausente no mate la predicción.
/// Copia literal del contrato de `god-engine-core` (`process_tick_dual`).
fn vector_de_servicio(fe: &StatefulEngine, omni: &[f64; 54]) -> [f32; DIM_VECTOR] {
    let mut v = [0f32; DIM_VECTOR];
    v[..34].copy_from_slice(&fe.get_universal_features());
    v[34..44].copy_from_slice(&fe.get_spectral_ml_features());
    v[44..].copy_from_slice(&macro_ml_features(omni));
    for x in v.iter_mut() {
        if !x.is_finite() {
            *x = 0.0;
        }
    }
    v
}

/// COMPROBACIÓN DE PARIDAD: primera dimensión en la que la fila que se va a
/// escribir NO coincide con lo que el motor produce en ese mismo instante.
/// `None` ⇒ paridad demostrada en esta fila.
///
/// Es la MISMA comprobación que `train_forest`, y por la misma razón: un CSV
/// que dice describir el vector de servicio y describe otra cosa es peor que
/// no tener CSV — se usa para decidir qué dimensión culpar cuando un modelo
/// satura, y culparía a la equivocada.
///
/// · dims [0..34) se contrastan contra `GodEngineCore::build_54d_tensor`, un
///   productor INDEPENDIENTE del motor cuyas 34 primeras entradas son, por
///   construcción, `feature_engines[coin].get_universal_features()`. Si este
///   binario volviera a leer un `StatefulEngine` suelto —el defecto que se
///   corrige— o leyera antes de alimentar el evento, dejarían de coincidir.
/// · dims [44..48) contra `macro_ml_features` sobre el MISMO `omni` que se
///   pasó a `process_event`: el bloque macro no tiene otra fuente.
/// · dims [34..44) no tienen un segundo productor en el motor, pero al quedar
///   certificado por las 34 primeras que el objeto leído ES
///   `core.feature_engines[coin]`, el bloque espectral sale forzosamente de
///   ese mismo objeto: la certificación es transitiva.
///
/// TOLERANCIA CERO: entre los dos lados no media aritmética alguna —sólo un
/// ensanchado f32→f64, que es exacto—, así que cualquier diferencia es
/// ESTRUCTURAL, nunca numérica. Un épsilon sólo taparía el fallo buscado.
fn primera_divergencia(
    v: &[f32; DIM_VECTOR],
    tensor_motor: &[f64; 54],
    macro_motor: &[f32; 4],
) -> Option<(usize, f64, f64)> {
    // El saneo del servicio convierte no finitos en 0.0; el tensor de la red
    // no lo hace. Se compara contra el valor SANEADO para no acusar de
    // divergencia a una diferencia que el propio contrato introduce.
    let saneado = |x: f64| if x.is_finite() { x } else { 0.0 };
    for d in 0..34 {
        let mio = v[d] as f64;
        let suyo = saneado(tensor_motor[d]);
        if mio != suyo {
            return Some((d, mio, suyo));
        }
    }
    for k in 0..4 {
        let mio = v[44 + k] as f64;
        let suyo = saneado(macro_motor[k] as f64);
        if mio != suyo {
            return Some((44 + k, mio, suyo));
        }
    }
    None
}

/// Dimensión declarada MUERTA EN SERVICIO que llega distinta de cero.
///
/// `FEATURES_DEAD_IN_SERVE` es la fuente única de verdad del mapa vivo/muerto
/// y hoy `get_universal_features()` la aplica en AMBOS lados. Este exportador
/// no zerifica nada por su cuenta —hacerlo sería volver a tener dos
/// contratos—: sólo comprueba que el motor cumple el suyo.
fn dim_muerta_no_nula(v: &[f32; DIM_VECTOR]) -> Option<usize> {
    FEATURES_DEAD_IN_SERVE
        .iter()
        .copied()
        .find(|&d| d < DIM_VECTOR && v[d] != 0.0)
}

/// REJILLA DE PRECIOS DEL INSTRUMENTO, MEDIDA EN EL PROPIO TAPE.
///
/// El motor necesita un `tick_size` para poner suelo al semi-spread simulado
/// (D-718/D-752) y ese suelo entra en `update_ofi` ⇒ dim [2] del vector, así
/// que no puede ser un literal. La menor diferencia POSITIVA entre precios
/// medios consecutivos ES la rejilla. Mismo método que `tape_spread_fix`
/// (D-751b). El umbral `mid · 1e-12` descarta el ruido de coma flotante del
/// round-trip cálculo→fichero→mmap, cuatro órdenes por encima del épsilon de
/// f64 y órdenes por debajo de cualquier rejilla real.
fn tick_del_tape(ticks: &[BinTick]) -> f64 {
    let mut menor = f64::INFINITY;
    let mut previo = 0.0f64;
    for t in ticks {
        if t.bid_price <= 0.0 || t.ask_price <= 0.0 || t.bid_price > t.ask_price || t.timestamp == 0
        {
            continue;
        }
        let mid = (t.bid_price + t.ask_price) * 0.5;
        if previo > 0.0 {
            let d = (mid - previo).abs();
            if d > mid * 1e-12 && d < menor {
                menor = d;
            }
        }
        previo = mid;
    }
    if menor.is_finite() {
        menor
    } else {
        0.0
    }
}

/// Niveles macro as-of t−1 ESTRICTO (último cierre de un día ANTERIOR al
/// tick: el del propio día aún no existe intradía) de `data/macro/*.csv`.
/// Devuelve `(dxy, spx, ndx, vix)` — las ranuras 21, 22, 23, 24 del omni,
/// que es donde `macro_ml_features` las busca. Serie ausente ⇒ 0.0 neutro,
/// la misma semántica de «sin dato» que el servicio.
fn serie_macro(tag: &str) -> Vec<(u64, f64)> {
    let path = format!("data/macro/{tag}.csv");
    match std::fs::read_to_string(&path) {
        Ok(c) => c
            .lines()
            .skip(1)
            .filter_map(|l| {
                let mut p = l.split(',');
                let ms = p.next()?.trim().parse::<u64>().ok()?;
                let v = p.next()?.trim().parse::<f64>().ok()?;
                (v.is_finite() && v > 0.0).then_some((ms, v))
            })
            .collect(),
        Err(e) => {
            println!("⚠️ macro {tag}: sin data/macro/{tag}.csv ({e}) — dim en 0 neutro");
            Vec::new()
        }
    }
}

fn main() {
    println!("============================================================");
    println!("🌌 EXPORTADOR DEL VECTOR DE SERVICIO — {} dims", DIM_VECTOR);
    println!("============================================================");

    let args: Vec<String> = std::env::args().collect();
    let symbol = match args.get(1) {
        Some(s) if !s.starts_with('-') => s.trim().to_uppercase(),
        _ => {
            eprintln!(
                "Uso: feature_exporter <SÍMBOLO> [--in FICHERO] [--out FICHERO] \
                 [--stride-ms N] [--calentamiento-ms N]"
            );
            std::process::exit(1);
        }
    };
    let arg = |name: &str, dflt: &str| -> String {
        args.iter()
            .position(|a| a == name)
            .and_then(|p| args.get(p + 1))
            .map(|s| s.to_string())
            .unwrap_or_else(|| dflt.to_string())
    };
    let input_path = arg("--in", &format!("data/{}_ticks.bin", symbol));
    let out_path = arg("--out", &format!("data/{}_FEATURES.csv", symbol));
    // Sin stride se exporta CADA tick: es un inspector, no un muestreador.
    let stride_ms: u64 = arg("--stride-ms", "0").parse().unwrap_or(0);
    // CALENTAMIENTO — en RELOJ del tape, no en ticks. El estado más lento que
    // entra en el vector es `kline_ema_macro` (dim [43]): una EMA de 720
    // velas de 1 minuto declarada en `stateful_engine::process_tick`. Su
    // memoria es 720 minutos de reloj; el 720 no es una constante de este
    // fichero, es el período del motor. Antes del fin del calentamiento las
    // filas se exportan igualmente pero marcadas: el transitorio de la
    // semilla existe también en producción tras cada arranque en frío, y
    // ocultarlo sería describir un motor que no existe.
    const PERIODO_EMA_KLINE_MAS_LENTA: u64 = 720; // velas de 1 minuto
    const MS_POR_VELA: u64 = 60_000;
    let calentamiento_ms: u64 = arg(
        "--calentamiento-ms",
        &(PERIODO_EMA_KLINE_MAS_LENTA * MS_POR_VELA).to_string(),
    )
    .parse()
    .unwrap_or(PERIODO_EMA_KLINE_MAS_LENTA * MS_POR_VELA);

    let file = match File::open(&input_path) {
        Ok(f) => f,
        Err(e) => {
            eprintln!("❌ no pude abrir {}: {}", input_path, e);
            std::process::exit(1);
        }
    };
    let mmap = match unsafe { memmap2::MmapOptions::new().map(&file) } {
        Ok(m) => m,
        Err(e) => {
            eprintln!("❌ no pude mapear {}: {}", input_path, e);
            std::process::exit(1);
        }
    };
    // El formato versionado lleva la cabecera `TGMTICK1` (8 bytes) antes de
    // los registros; el legado no. Mapear siempre desde el byte 0 hacía
    // ILEGIBLE el tape de aggTrades reales (D-691), que es justo el que
    // interesa inspeccionar.
    let header_off = if mmap.len() >= 8 && &mmap[..8] == b"TGMTICK1" { 8usize } else { 0 };
    let tick_size_bytes = std::mem::size_of::<BinTick>();
    let payload = mmap.len() - header_off;
    if payload % tick_size_bytes != 0 || payload == 0 {
        eprintln!(
            "❌ {}: {} bytes de registros no es múltiplo de {} — formato desconocido",
            input_path, payload, tick_size_bytes
        );
        std::process::exit(1);
    }
    let num_ticks = payload / tick_size_bytes;
    let ticks = unsafe {
        std::slice::from_raw_parts(mmap.as_ptr().add(header_off) as *const BinTick, num_ticks)
    };
    println!("✅ {} ticks de {} ({})", num_ticks, symbol, input_path);

    // ── LA FICHA DEL SÍMBOLO SE MIDE, NO SE CABLEA ──────────────────────
    let tick_size = tick_del_tape(ticks);
    if !(tick_size > 0.0) {
        eprintln!(
            "❌ no se pudo medir la rejilla de precios del tape: sin tick no hay suelo de \
             semi-spread y la dim [2] (OFI) no sería la del servicio."
        );
        std::process::exit(1);
    }
    println!("📏 rejilla medida en el tape: tick {:.10}", tick_size);
    quantum_arena::symbol_registry::update_registry(vec![SymbolSpec {
        symbol: symbol.clone(),
        step_size: tick_size,
        tick_size,
        min_qty: tick_size,
        // Límites PUBLICADOS por Binance para USDⓈ-M; ninguno entra en el
        // vector: existen para que el motor sepa sobre qué instrumento corre.
        min_notional: 5.0,
        max_leverage: 125,
        maker_fee: 0.0002,
        taker_fee: 0.0005,
        is_shadow: false,
    }]);

    // El capital no entra en ninguna dimensión y, con las entradas
    // bloqueadas, no se mueve: existe sólo porque el arena exige uno. Se toma
    // del mínimo nocional publicado, el menor con el que una orden sería
    // representable.
    let capital = 5.0f64;
    let arena = quantum_arena::GlobalArena::build_in_own_stack(capital);
    arena.config.live_maker_fee.store(0.0002, Ordering::Relaxed);
    arena.config.live_taker_fee.store(0.0005, Ordering::Relaxed);
    let mut core = GodEngineCore::new(arena.clone());
    // El vector no depende del genoma, pero `refresh_models` relee el almacén
    // cada 1000 ticks: con la generación aplicada al máximo la exportación
    // deja de depender de qué genoma hubiera en disco mientras corría.
    core.applied_generation.store(u64::MAX, Ordering::Relaxed);
    // ENTRADAS BLOQUEADAS. Si el motor operase y arruinase el capital,
    // `process_event` activaría el kill-switch y retornaría ANTES de tocar
    // los rasgos: el CSV seguiría creciendo con el MISMO vector congelado.
    // El propio motor documenta que el bloqueo actúa DESPUÉS de la analítica
    // ML/espectral, así que el vector no cambia por bloquear.
    quantum_arena::feed_health::stall();

    let macro_dxy = serie_macro("DXY");
    let macro_spx = serie_macro("SP500");
    let macro_ndx = serie_macro("NASDAQ");
    let macro_vix = serie_macro("VIX");
    let (mut c_dxy, mut c_spx, mut c_ndx, mut c_vix) = (0usize, 0usize, 0usize, 0usize);
    let mut macro_asof = |ts: u64| -> Option<[f64; 4]> {
        let day_start = ts - (ts % 86_400_000);
        fn adv(series: &[(u64, f64)], cur: &mut usize, day_start: u64) -> Option<f64> {
            // Serie vacía ⇒ 0.0 neutro (centinela «sin dato»), no descarte.
            if series.is_empty() {
                return Some(0.0);
            }
            while *cur + 1 < series.len() && series[*cur + 1].0 < day_start {
                *cur += 1;
            }
            match series.get(*cur) {
                Some(&(ms, v)) if ms < day_start => Some(v),
                _ => None,
            }
        }
        Some([
            adv(&macro_dxy, &mut c_dxy, day_start)?,
            adv(&macro_spx, &mut c_spx, day_start)?,
            adv(&macro_ndx, &mut c_ndx, day_start)?,
            adv(&macro_vix, &mut c_vix, day_start)?,
        ])
    };

    let out_file = match File::create(&out_path) {
        Ok(f) => f,
        Err(e) => {
            eprintln!("❌ no pude crear {}: {}", out_path, e);
            std::process::exit(1);
        }
    };
    let mut out = BufWriter::new(out_file);
    // Cabecera HONESTA: ts, precio medio, si el calentamiento terminó, si el
    // macro del día existe, y las DIM_VECTOR dimensiones del contrato.
    let mut header = String::from("ts_ms,mid,calentado,macro_vigente");
    for d in 0..DIM_VECTOR {
        header.push_str(&format!(",dim_{}", d));
    }
    if let Err(e) = writeln!(out, "{}", header) {
        eprintln!("❌ no pude escribir la cabecera en {}: {}", out_path, e);
        std::process::exit(1);
    }

    let mut omni = [0.0f64; 54];
    let mut dia_macro: i64 = i64::MIN;
    let mut macro_vigente = false;
    let mut first_ts: u64 = 0;
    let mut fin_calentamiento: u64 = 0;
    let mut next_ts: u64 = 0;
    let mut prev_ts: u64 = 0;
    let mut written = 0u64;

    for (i, t) in ticks.iter().enumerate() {
        if t.bid_price <= 0.0 || t.ask_price <= 0.0 || t.bid_price > t.ask_price || t.timestamp == 0
        {
            continue;
        }
        if first_ts == 0 {
            first_ts = t.timestamp;
            fin_calentamiento = first_ts.saturating_add(calentamiento_ms);
        }
        let mid = (t.bid_price + t.ask_price) / 2.0;
        // D-747: el lado del AGRESOR lleva `qty + base` y el pasivo sólo
        // `base`, con `base = max(0,25·qty; 0,1)`. Luego la cantidad real es
        // `|bq − aq|` y `is_buyer_maker` (el comprador era el pasivo) es
        // `aq > bq`. Antes se pasaba `bq + aq` como volumen (≈1,5·qty con un
        // suelo absoluto que deforma los trades pequeños) y `bq > aq` como
        // `is_buyer_maker`, que es exactamente su NEGACIÓN: el flujo agregado
        // salía ESPEJADO respecto al del motor vivo.
        let vol = (t.bid_qty - t.ask_qty).abs();
        let is_buyer_maker = t.ask_qty > t.bid_qty;

        let dia = (t.timestamp / 86_400_000) as i64;
        if dia != dia_macro {
            dia_macro = dia;
            match macro_asof(t.timestamp) {
                Some([dxy, spx, ndx, vix]) => {
                    omni[21] = dxy;
                    omni[22] = spx;
                    omni[23] = ndx;
                    omni[24] = vix;
                    macro_vigente = true;
                }
                None => {
                    omni[21] = 0.0;
                    omni[22] = 0.0;
                    omni[23] = 0.0;
                    omni[24] = 0.0;
                    macro_vigente = false;
                }
            }
        }

        // ── REPRODUCCIÓN POR EL CAMINO DEL MOTOR ────────────────────────
        // Libro simulado SIMÉTRICO alrededor del punto medio con suelo de
        // medio tick. D-752: restar el semi-spread al bid Y sumárselo al ask
        // DUPLICA la horquilla que el tape ya trae; el suelo va sobre el
        // semi-spread, alrededor del medio.
        let half_spread = ((t.ask_price - t.bid_price) / 2.0).max(tick_size * 0.5);
        let sim_bid = mid - half_spread;
        let sim_ask = mid + half_spread;
        let real_obi = if (t.bid_qty + t.ask_qty) > 0.0 {
            (t.bid_qty - t.ask_qty) / (t.bid_qty + t.ask_qty)
        } else {
            0.0
        };
        let is_minute_kline = prev_ts == 0 || (t.timestamp / 60_000) != (prev_ts / 60_000);
        prev_ts = t.timestamp;
        core.arena.update_l2_depth(0, t.bid_qty, t.ask_qty);
        let _ = core.process_event(
            0, false, is_minute_kline, true, mid, vol, sim_bid, sim_ask, t.bid_qty, t.ask_qty,
            real_obi, 0.0, t.timestamp, false, &omni, false,
        );
        let _ = core.process_event(
            0, true, false, false, mid, vol, sim_bid, sim_ask, t.bid_qty, t.ask_qty, real_obi,
            0.0, t.timestamp, false, &omni, is_buyer_maker,
        );
        if core.arena.kill_switch_active.load(Ordering::Relaxed) {
            eprintln!(
                "❌ kill-switch del motor activo en el tick {} — desde aquí `process_event` no \
                 actualiza los rasgos y toda fila posterior sería el MISMO vector congelado.",
                i
            );
            std::process::exit(1);
        }

        if t.timestamp < next_ts {
            continue;
        }
        next_ts = t.timestamp.saturating_add(stride_ms);

        // EL VECTOR SALE DEL MOTOR: `core.feature_engines[0]` es el MISMO
        // objeto del que el servicio lee en `process_tick_dual`, leído justo
        // tras el evento de trade, que es el último del tick.
        let v = vector_de_servicio(&core.feature_engines[0], &omni);
        // ── COMPROBACIÓN DE PARIDAD (aborta) ────────────────────────────
        // Un exportador que no puede demostrar su paridad no debe escribir
        // un CSV: sus columnas se leen para decidir qué dimensión culpar
        // cuando un modelo satura, y culparían a la equivocada.
        if let Some(d) = dim_muerta_no_nula(&v) {
            eprintln!(
                "❌ PARIDAD ROTA en el tick {}: la dim [{}] está declarada MUERTA EN SERVICIO \
                 (FEATURES_DEAD_IN_SERVE) y el motor la entrega en {}. El CSV describiría un \
                 soporte que el vivo no sirve.",
                i, d, v[d]
            );
            std::process::exit(1);
        }
        let tensor_motor = core.build_54d_tensor(0, t.bid_qty, t.ask_qty, mid, &omni);
        let macro_motor = macro_ml_features(&omni);
        if let Some((d, mio, suyo)) = primera_divergencia(&v, &tensor_motor, &macro_motor) {
            eprintln!(
                "❌ PARIDAD ROTA en el tick {}: la dim [{}] vale {:e} en la fila que este \
                 exportador escribiría y {:e} en la que el motor produce en el MISMO instante. \
                 Entre ambos lados no media aritmética alguna, así que la diferencia es \
                 ESTRUCTURAL: el exportador no está leyendo el motor.",
                i, d, mio, suyo
            );
            std::process::exit(1);
        }
        let mut row = format!(
            "{},{:.10},{},{}",
            t.timestamp,
            mid,
            u8::from(t.timestamp >= fin_calentamiento),
            u8::from(macro_vigente)
        );
        for x in v.iter() {
            row.push_str(&format!(",{:.8}", x));
        }
        // Un error de escritura NO se puede tragar: `is_ok()` dejaba que el
        // CSV perdiera filas en silencio (disco lleno, tubería cerrada) y el
        // fichero resultante tendría huecos con aspecto de tape sin ticks.
        // Un inspector que miente por omisión es peor que ninguno.
        if let Err(e) = writeln!(out, "{}", row) {
            eprintln!(
                "❌ no pude escribir la fila del tick {} en {}: {} — el CSV quedaría incompleto \
                 sin avisar.",
                i, out_path, e
            );
            std::process::exit(1);
        }
        written += 1;
    }

    if let Err(e) = out.flush() {
        eprintln!("❌ no pude volcar {}: {}", out_path, e);
        std::process::exit(1);
    }
    // El capital intacto es la PRUEBA de que ninguna rama de operativa corrió
    // y, por tanto, de que el vector exportado es función sólo del tape y del
    // macro — como en servicio.
    let capital_final = arena.unified_capital.load(Ordering::Relaxed);
    if (capital_final - capital).abs() > 0.0 {
        eprintln!(
            "❌ el capital se movió durante la exportación ({} → {}): el motor operó pese al \
             bloqueo de entradas y el vector podría depender de la simulación de PnL.",
            capital, capital_final
        );
        std::process::exit(1);
    }
    println!(
        "🚀 {} filas de {} dims escritas en {} (capital intacto, kill-switch apagado)",
        written, DIM_VECTOR, out_path
    );
}
