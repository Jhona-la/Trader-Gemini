use execution_engine::executor::ExecutionProvider;
use std::sync::atomic::Ordering;
/// 🛡️ AUDIT FORENSIC BACKTEST — Paridad 1:1 con Producción (Zero Mutation)
///
/// Este binario ejecuta un backtest usando EXACTAMENTE los mismos parámetros
/// que `live_trader.rs` cargaría al arrancar (`Genome::load_and_unify_genome(true)`).
/// NO aplica mutaciones evolutivas (UnifiedConfig) ni optimización CMA-ES.
///
/// Objetivo: Responder a la pregunta forense:
///   "Si hubiera ejecutado el bot con la configuración actual sobre datos históricos,
///    ¿qué habría pasado con mi capital real?"
///
/// Flujo:
///   1. Carga el Genoma de Producción (quantum_champion.json → moe_champion.bin → bootstrap_seed)
///   2. Construye QuantumConfig y GlobalArena idénticos a producción
///   3. Instancia GodEngineCore con los mismos modelos ML (NanoForest, DarkAlpha)
///   4. Inyecta ticks históricos de `data/BTCUSDT_ticks.bin` al motor
///   5. Reporta métricas: PnL, WinRate, Drawdown, Sharpe, Capital Final
///
/// PROHIBIDO: Alterar parámetros del Genoma dentro de este binario.
use std::sync::Arc;
use std::time::Instant;

#[global_allocator]
static GLOBAL: mimalloc::MiMalloc = mimalloc::MiMalloc;

#[tokio::main]
async fn main() {
    println!("🛡️ ═══════════════════════════════════════════════════════════════");
    println!("🛡️  AUDIT FORENSIC BACKTEST — Paridad 1:1 con LAUNCH_GOD_MODE.bat");
    println!("🛡️ ═══════════════════════════════════════════════════════════════");
    println!();

    let t0 = Instant::now();

    // ═══════════════════════════════════════════════════════════════════════
    // PASO 1: Cargar Genoma de Producción (IDÉNTICO a live_trader.rs línea 26)
    // ═══════════════════════════════════════════════════════════════════════
    dotenvy::dotenv().ok();
    let is_testnet = std::env::var("USE_TESTNET")
        .unwrap_or("true".to_string())
        .trim()
        .to_lowercase()
        == "true";
    let api_key = if is_testnet {
        std::env::var("BINANCE_DEMO_API_KEY")
            .or_else(|_| std::env::var("BINANCE_TESTNET_API_KEY"))
            .unwrap_or_default()
    } else {
        std::env::var("BINANCE_API_KEY").unwrap_or_default()
    }
    .trim()
    .to_string();
    let api_secret = if is_testnet {
        std::env::var("BINANCE_DEMO_SECRET_KEY")
            .or_else(|_| std::env::var("BINANCE_TESTNET_SECRET_KEY"))
            .unwrap_or_default()
    } else {
        std::env::var("BINANCE_SECRET_KEY").unwrap_or_default()
    }
    .trim()
    .to_string();

    let executor = Arc::new(execution_engine::executor::OrderExecutor::new(
        api_key.clone(),
        api_secret.clone(),
        is_testnet,
    ));
    if std::env::var("TG_GENOME_ENV").is_err() {
        std::env::set_var("TG_GENOME_ENV", "prod");
    }
    // D-686 (DÉCIMA OLA): el evolucionador walk-forward evalúa cada candidato
    // pasando su genoma en un fichero. Sin la variable se carga el genoma
    // activo del entorno, como hasta ahora.
    let genome = match std::env::var("FORENSIC_GENOME_PATH") {
        Ok(path) => {
            let parsed = std::fs::read_to_string(&path)
                .map_err(|e| e.to_string())
                .and_then(|data| {
                    serde_json::from_str::<quantum_arena::genome::SuperGenotype>(&data)
                        .map_err(|e| e.to_string())
                });
            match parsed {
                Ok(g) => g,
                Err(e) => {
                    println!("❌ FORENSIC_GENOME_PATH={} no contiene un genoma válido: {}", path, e);
                    return;
                }
            }
        }
        Err(_) => quantum_arena::genome::SuperGenotype::load_or_default(),
    };

    // Extracción y control del capital base ($13.00 USD) para validación forense
    let initial_capital = std::env::var("INITIAL_CAPITAL")
        .ok()
        .and_then(|v| v.parse::<f64>().ok())
        .unwrap_or(13.0);

    println!("🧬 [GENOMA] Capital Base: ${:.4}", initial_capital);
    println!("🧬 [GENOMA] Scalp TP Base: {:.6}", genome.scalp_tp_base);
    println!("🧬 [GENOMA] Scalp SL Base: {:.6}", genome.scalp_sl_base);
    println!("🧬 [GENOMA] Swing TP Base: {:.6}", genome.swing_tp_base);
    println!("🧬 [GENOMA] Swing SL Base: {:.6}", genome.swing_sl_base);
    println!();

    // ═══════════════════════════════════════════════════════════════════════
    // PASO 2: Construir Arena y Core (Paridad Producción)
    // ═══════════════════════════════════════════════════════════════════════
    // D-684 (DÉCIMA OLA): el arena se construye en un hilo con la MISMA pila que
    // producción (`god_engine`, 32 MiB). `CoinArena` lleva el búfer del ring de
    // ticks en línea y `GlobalArena::build` materializa temporales de ese tamaño:
    // su necesidad de pila depende de cómo el optimizador inlinee la construcción,
    // y con 1 MiB (hilo principal en Windows) un cambio de una línea en `build`
    // bastó para desbordarla. El evolver y el simulador ya seguían este patrón.
    let arena = quantum_arena::GlobalArena::build_in_own_stack(initial_capital);
    genome.apply_to_arena(&arena);

    // Desactivar NanoForest obsoleto para activar DarkAlphaEngine 54D unificado
    println!("🧠 [UNIFIED ML] DarkAlphaEngine 54D configurado como motor neuronal primario.");

    let mut core = god_engine_core::GodEngineCore::new(arena.clone());
    // D-686: con un genoma externo, `refresh_models` no debe sustituirlo por el
    // activo del almacén a mitad de la corrida (lo relee cada 1000 ticks). Una
    // generación aplicada máxima fija el candidato evaluado.
    if std::env::var("FORENSIC_GENOME_PATH").is_ok() {
        core.applied_generation
            .store(u64::MAX, std::sync::atomic::Ordering::Relaxed);
    }

    // F3.1 — FIX PARIDAD REAL: antes se instanciaba una NN ALEATORIA
    // (DarkAlphaEngine::new) y el "backtest forense" evaluaba ruido.
    let model_res = dark_alpha_engine::DarkAlphaEngine::load_json("models/DarkAlpha_BTCUSDT.json");
    let nn = match model_res {
        Ok(mut m) => {
            // Verificar si los pesos contienen NaN o Infinito
            let is_corrupt = m
                .layer1
                .weights
                .iter()
                .any(|&w| w.is_nan() || w.is_infinite());
            if is_corrupt {
                println!("⚠️ [DARK ALPHA] Pesos no finitos detectados en models/DarkAlpha_BTCUSDT.json. Regenerando modelo Xavier 54D.");
                let clean = dark_alpha_engine::DarkAlphaEngine::default_model();
                let _ = clean.save_json("models/DarkAlpha_BTCUSDT.json");
                clean
            } else {
                m.init_buffers();
                m.sanitize_denormals();
                m.freeze(); // T-04: inferencia siempre congelada para paridad 1:1 con producción
                println!("🧠 DarkAlpha (Swing NN) REAL cargado y congelado: models/DarkAlpha_BTCUSDT.json");
                m
            }
        }
        Err(e) => {
            println!(
                "⚠️ DarkAlpha_BTCUSDT.json no disponible ({}). Creando modelo 54D inicializado.",
                e
            );
            let clean = dark_alpha_engine::DarkAlphaEngine::default_model();
            let _ = clean.save_json("models/DarkAlpha_BTCUSDT.json");
            clean
        }
    };
    core.swing_nn = Some(nn);

    // D-699 (DÉCIMA OLA · auditoría integral): EL FORENSE CARGA LOS MISMOS
    // MODELOS QUE EL BOT VIVO.
    //
    // `god_engine` registra al arrancar TODOS los ficheros de `models/` en el
    // registro global de `NanoForest`, por su nombre de fichero, y el núcleo
    // busca el bosque del símbolo con la clave `{SÍMBOLO}_SCALP`. El forense no
    // registraba ninguno: `NanoForest::get_global("BTCUSDT_SCALP")` devolvía
    // `None` y el ensamble corría sin bosque —sólo red y espectro—. Desde que
    // «la predicción decide» (gate de ensamble en toda entrada), medir sin el
    // bosque es medir otro motor. `backtest_windows` y `evolution` ya lo
    // cargaban; el forense, que es el que valida, no.
    let mut forests_cargados = 0usize;
    match std::fs::read_dir("models") {
        Ok(entries) => {
            for entry in entries.filter_map(|e| e.ok()) {
                let path = entry.path();
                let ext = path.extension().and_then(|s| s.to_str());
                if ext != Some("json") && ext != Some("bin") {
                    continue;
                }
                let (Some(stem), Some(path_str)) =
                    (path.file_stem().and_then(|s| s.to_str()), path.to_str())
                else {
                    continue;
                };
                if god_engine_core::ml_inference::NanoForest::load_global(stem, path_str).is_ok() {
                    forests_cargados += 1;
                }
            }
        }
        Err(e) => println!("⚠️ [MODELOS] No se pudo leer models/: {}", e),
    }
    println!(
        "🌲 [MODELOS] {} modelos registrados como en producción · bosque BTCUSDT_SCALP: {}",
        forests_cargados,
        if god_engine_core::ml_inference::NanoForest::get_global("BTCUSDT_SCALP").is_some() {
            "presente"
        } else {
            "AUSENTE (el ensamble corre sin bosque)"
        }
    );

    // INICIALIZAR EL SYMBOL REGISTRY PARA BTCUSDT (coin_id = 0)
    quantum_arena::symbol_registry::update_registry(vec![
        quantum_arena::symbol_registry::SymbolSpec {
            symbol: "BTCUSDT".to_string(),
            step_size: 0.001,
            tick_size: 0.10,
            min_qty: 0.001,
            min_notional: 5.0,
            max_leverage: 125,
            maker_fee: 0.0002,
            taker_fee: 0.0005,
            is_shadow: false,
        },
    ]);

    println!("⚛️  [ARENA] QuantumConfig y GodEngineCore instanciados (Paridad Producción)");

    // Configurar fees dinámicamente desde API real o baseline VIP0
    // D-694 (DÉCIMA OLA): con claves en `.env` el forense consultaba siempre las
    // comisiones reales de la cuenta (taker 0,04 %) y sin ellas usaba VIP0
    // (0,05 %): el mismo binario daba resultados distintos según el directorio,
    // y hacía una llamada autenticada a Binance en cada corrida. Un backtest de
    // validación fija sus costes de forma explícita: la consulta en vivo sólo
    // ocurre si se pide con FORENSIC_LIVE_FEES=1.
    let live_fees_requested = std::env::var("FORENSIC_LIVE_FEES")
        .map(|v| v.trim() == "1")
        .unwrap_or(false);
    let (maker_fee, taker_fee) = if live_fees_requested && !api_key.is_empty() && !api_secret.is_empty() {
        match executor.fetch_commission_rate("BTCUSDT").await {
            Ok((m, t)) if m > 0.0 && t > 0.0 => {
                println!(
                    "🌍 [API] Comisiones Reales de Binance Extraídas: Maker {:.4}%, Taker {:.4}%",
                    m * 100.0,
                    t * 100.0
                );
                (m, t)
            }
            _ => {
                println!("🌍 [API] Usando comisiones estándar VIP0 de Binance (Maker 0.02%, Taker 0.05%)");
                (0.0002, 0.0005)
            }
        }
    } else {
        println!("🌍 [API] Modo Offline / Sin claves API: Usando comisiones estándar VIP0 (Maker 0.02%, Taker 0.05%)");
        (0.0002, 0.0005)
    };
    arena
        .config
        .live_taker_fee
        .store(taker_fee, Ordering::Relaxed);
    arena
        .config
        .live_maker_fee
        .store(maker_fee, Ordering::Relaxed);

    // ═══════════════════════════════════════════════════════════════════════
    // PASO 3: Cargar datos históricos de ticks
    // ═══════════════════════════════════════════════════════════════════════
    let cli_sym = std::env::args().nth(1);
    let data_path = if let Some(sym) = cli_sym {
        if sym.ends_with(".bin") {
            sym
        } else {
            format!("data/{}_ticks.bin", sym.to_uppercase())
        }
    } else {
        std::env::var("FORENSIC_DATA_PATH").unwrap_or_else(|_| "data/BTCUSDT_ticks.bin".to_string())
    };

    let file = match std::fs::File::open(&data_path) {
        Ok(f) => f,
        Err(e) => {
            println!("❌ No se pudo abrir {}: {}", data_path, e);
            println!("   Asegúrese de que exista el archivo de ticks. Puede generarlo con:");
            println!("   cargo run --release --bin parquet_to_bin");
            return;
        }
    };

    // FIX #1475: Mapeo MMAP seguro sin expect
    let mmap = match unsafe { memmap2::MmapOptions::new().map(&file) } {
        Ok(m) => m,
        Err(e) => {
            println!("❌ Fallo al mapear en memoria {}: {}", data_path, e);
            return;
        }
    };
    let bytes_len = mmap.len();
    #[derive(Debug, Clone, Copy)]
    #[repr(C)]
    struct BinTick {
        pub timestamp: u64,
        pub bid_price: f64,
        pub ask_price: f64,
        pub bid_qty: f64,
        pub ask_qty: f64,
    }

    let tick_size = std::mem::size_of::<BinTick>();
    // D-691 (DÉCIMA OLA): el formato versionado lleva la cabecera `TGMTICK1`
    // (8 bytes) antes de los registros; el legado no. Se mapeaba desde el byte 0
    // en ambos casos, así que el fichero de aggTrades reales era ilegible y todo
    // el forense corría sobre velas expandidas. Mismo criterio que
    // `backtest_engine::tick_replayer`.
    // D-721: la cabecera declara el ORIGEN del dato, no sólo la versión.
    let (origen, header_len) = backtest_engine::tick_replayer::TickOrigin::from_header(
        &mmap[..bytes_len.min(8)],
    );
    let payload_len = bytes_len - header_len;
    if payload_len % tick_size != 0 {
        println!(
            "❌ {}: {} bytes de registros no es múltiplo de {} — formato desconocido",
            data_path, payload_len, tick_size
        );
        return;
    }
    println!("📦 Origen de los datos: {}", origen.descripcion());
    if !origen.es_real() {
        println!(
            "⚠️  [VALIDEZ] Este veredicto NO puede sostener conclusiones de microestructura ni de coste: los datos no son ticks del exchange (D-691)."
        );
    }
    let total_file_ticks = payload_len / tick_size;
    let max_ticks_env = std::env::var("MAX_TICKS")
        .ok()
        .and_then(|v| v.parse::<usize>().ok())
        .unwrap_or(total_file_ticks);
    // D-686: FORENSIC_START_TICK desplaza el inicio de la ventana, de modo que
    // las particiones de entrenamiento y validación se evalúan sin copiar datos.
    let start_tick = std::env::var("FORENSIC_START_TICK")
        .ok()
        .and_then(|v| v.parse::<usize>().ok())
        .unwrap_or(0)
        .min(total_file_ticks);
    let num_ticks = (total_file_ticks - start_tick).min(max_ticks_env);

    if num_ticks < 1000 {
        println!("❌ Datos insuficientes: {} ticks (mínimo 1000)", num_ticks);
        return;
    }

    let ticks_slice = unsafe {
        std::slice::from_raw_parts(
            mmap.as_ptr().add(header_len + start_tick * tick_size) as *const BinTick,
            num_ticks,
        )
    };

    println!(
        "📊 [DATOS] {} ticks cargados desde {} ({:.1} MB)",
        num_ticks,
        data_path,
        bytes_len as f64 / 1_048_576.0
    );
    println!(
        "📊 [DATOS] Primer precio: ${:.2} | Último precio: ${:.2}",
        ticks_slice[0].bid_price,
        ticks_slice[num_ticks - 1].bid_price
    );
    println!();

    // ═══════════════════════════════════════════════════════════════════════
    // PASO 4: Calentamiento (Warm-Up) — Primeros 200 ticks sin contar trades
    // ═══════════════════════════════════════════════════════════════════════
    let warmup_ticks = 15000.min(num_ticks / 10);
    println!(
        "🔥 [WARM-UP] Alimentando {} ticks de calentamiento profundo al motor...",
        warmup_ticks
    );

    for i in 0..warmup_ticks {
        let t = &ticks_slice[i];
        let price = (t.bid_price + t.ask_price) / 2.0;
        let vol = t.bid_qty + t.ask_qty;
        let is_buyer_maker = t.ask_qty > t.bid_qty;
        core.arena.update_market_data(
            0,
            t.bid_price,
            t.ask_price,
            t.bid_qty,
            t.ask_qty,
            t.timestamp,
        );
        // D-708: el flujo agregado lo actualiza el núcleo en `process_event`.
        core.arena.update_l2_depth(0, t.bid_qty, t.ask_qty);
        core.feature_engines[0].process_tick(price, vol, t.timestamp);
        core.feature_engines[0].update_trade_flow(vol, is_buyer_maker);
        core.feature_engines[0].update_ofi(t.bid_price, t.ask_price, t.bid_qty, t.ask_qty);
    }
    arena
        .unified_capital
        .store(initial_capital, Ordering::Relaxed);

    println!("✅ [WARM-UP] Completado. Los tensores y EMAs están 100% calibrados.");
    println!();

    // ═══════════════════════════════════════════════════════════════════════
    // F3.1 — MACRO HISTÓRICO REAL (era &[0.0; 54]: la NN swing evaluaba ceros)
    // FRED entrega la serie COMPLETA con fechas: alineamos VIX/S&P/Nasdaq/
    // US10Y/DXY/WTI por fecha a cada barra. Las features crypto cross-exchange
    // no tienen serie histórica accesible — quedan en default y CONSTAN;
    // el macro (la mitad informativa del vector omni) es el REAL de cada día.
    // ═══════════════════════════════════════════════════════════════════════
    println!("🌐 [MACRO-HIST] Descargando series históricas FRED...");
    let ts_first_ms = ticks_slice[warmup_ticks].timestamp;
    let cosd = chrono::DateTime::<chrono::Utc>::from_timestamp((ts_first_ms / 1000) as i64, 0)
        .map(|d| d.format("%Y-%m-%d").to_string())
        .unwrap_or_else(|| "2024-01-01".to_string());

    let fred_series: [(&str, u8); 6] = [
        ("SP500", 0),
        ("NASDAQCOM", 1),
        ("VIXCLS", 2),
        ("DGS10", 3),
        ("DTWEXBGS", 4),
        ("DCOILWTICO", 5),
    ];
    let http = reqwest::Client::builder()
        .timeout(std::time::Duration::from_millis(1500))
        .build()
        .unwrap_or_default();
    let mut macro_hist: Vec<Vec<(i64, f64)>> = Vec::new(); // (días desde epoch, valor)
    for (series, _) in &fred_series {
        let url = format!(
            "https://fred.stlouisfed.org/graph/fredgraph.csv?id={}&cosd={}",
            series, cosd
        );
        let parsed: Vec<(i64, f64)> = match http.get(&url).send().await {
            Ok(res) if res.status().is_success() => match res.text().await {
                Ok(csv) => csv
                    .lines()
                    .skip(1)
                    .filter_map(|l| {
                        let mut p = l.split(',');
                        let d = p.next()?.trim();
                        let v = p.next()?.trim().parse::<f64>().ok()?;
                        let date = chrono::NaiveDate::parse_from_str(d, "%Y-%m-%d").ok()?;
                        Some((date.and_hms_opt(0, 0, 0)?.and_utc().timestamp() / 86_400, v))
                    })
                    .collect(),
                Err(_) => Vec::new(),
            },
            _ => Vec::new(),
        };
        println!("   {} → {} puntos", series, parsed.len());
        macro_hist.push(parsed);
    }
    let macro_lookup = |idx: usize, days: i64| -> f64 {
        // Step function: último valor conocido ≤ fecha de la barra.
        let s = &macro_hist[idx];
        match s.binary_search_by_key(&days, |&(d, _)| d) {
            Ok(i) => s[i].1,
            Err(0) => s.first().map(|v| v.1).unwrap_or(0.0),
            Err(i) => s[i - 1].1,
        }
    };

    let omni_state = Arc::new(data_pipeline::omni_multiplexer::OmniState::new());
    let mut last_macro_day = i64::MIN;
    println!("🌐 [MACRO-HIST] Series alineadas por fecha a cada barra.\n");

    // ═══════════════════════════════════════════════════════════════════════
    // PASO 5: Simulación Forense (tick-by-tick, idéntico a producción)
    // ═══════════════════════════════════════════════════════════════════════
    println!(
        "⚡ [SIMULACIÓN] Iniciando procesamiento forense de {} ticks...",
        num_ticks - warmup_ticks
    );

    let sim_start = Instant::now();
    let mut total_trades = 0u64;
    let mut total_gross_wins = 0u64;
    let mut total_net_wins = 0u64;
    let mut total_opens = 0u64;
    let mut total_closes = 0u64;
    let mut total_pnl_dollars = 0.0f64;
    let mut total_gross_pnl_dollars = 0.0f64;
    let mut peak_capital = initial_capital;
    let mut max_drawdown = 0.0f64;
    let mut pnl_series: Vec<f64> = Vec::with_capacity((num_ticks - warmup_ticks) / 100);
    let mut long_trades = 0u64;
    let mut long_net_wins = 0u64;
    let mut long_pnl = 0.0f64;
    let mut short_trades = 0u64;
    let mut short_net_wins = 0u64;
    let mut short_pnl = 0.0f64;
    let mut reason_tp = 0u64;
    let mut reason_sl = 0u64;
    let mut reason_trail = 0u64;
    let mut reason_zombie = 0u64;
    let mut reason_toxic = 0u64;

    // D-718: piso del medio spread = medio tick del símbolo registrado (el
    // mínimo físicamente representable en ese instrumento), no un importe en
    // dólares. Para BTCUSDT (tick 0,10) vale 0,05, el literal que había; para
    // cualquier otro símbolo deja de ser una fricción inventada.
    let min_half_spread = quantum_arena::symbol_registry::try_spec(0)
        .map(|spec| spec.tick_size * 0.5)
        .unwrap_or(0.0)
        .max(0.0);
    println!(
        "📐 [FRICCIÓN] Piso de medio spread: {:.8} (medio tick del símbolo registrado)",
        min_half_spread
    );

    // Precompute ATR for delta-normalization (same logic as backtest-engine/lib.rs)
    let alpha = 2.0 / (14.0 + 1.0);
    let mut running_atr = 0.001 * ticks_slice[warmup_ticks].bid_price;
    let mut prev_ts: u64 = 0;

    for i in warmup_ticks..num_ticks {
        let t = &ticks_slice[i];
        let price = (t.bid_price + t.ask_price) / 2.0;
        let vol = t.bid_qty + t.ask_qty;
        let prev_price = if i > 0 {
            (ticks_slice[i - 1].bid_price + ticks_slice[i - 1].ask_price) / 2.0
        } else {
            price
        };
        // D-717 (DÉCIMA OLA · auditoría integral): EL LADO AGRESOR VIAJA EN EL
        // DATO, NO SE DEDUCE DEL PRECIO SIMULADO.
        //
        // `binance_vision_sync` codifica el `isBuyerMaker` oficial de cada
        // aggTrade en las cantidades: maker ⇒ (bid = base, ask = qty + base), es
        // decir `bid_qty < ask_qty`. El calentamiento de este mismo binario ya
        // lee ese convenio (línea 387); el bucle principal, en cambio, pasaba al
        // núcleo `price <= sim_bid`, con `price` el punto medio y
        // `sim_bid = bid − medio spread`: una condición FALSA en todos los ticks
        // del histórico. Con el flag constante en false, `agg_sell_vol` nunca
        // crecía y `rolling_cvd` se quedaba en +1,0 desde el primer tick: presión
        // compradora máxima permanente en el binario que dicta el veredicto y que
        // alimenta al evolucionador walk-forward, con las ramas Short de
        // price-action inalcanzables por construcción y un sesgo aditivo de +0,30
        // en el micro-score.
        let is_buyer_maker = t.bid_qty < t.ask_qty;

        let tr = (price - prev_price).abs();
        running_atr = alpha * tr + (1.0 - alpha) * running_atr;

        // Emulación de Slippage Microestructural Realista (Binance L2 Top-of-Book)
        // D-718 (DÉCIMA OLA · auditoría integral): el piso del medio spread es
        // una propiedad del INSTRUMENTO, no un importe en dólares. `0,05` es
        // exactamente medio tick de BTCUSDT: en XRP (~0,55 $) ese mismo piso
        // añade un 9 % por lado y toda operación nace con una pérdida
        // instantánea de ese orden, de modo que el forense era inutilizable
        // fuera de BTC —y las validaciones cruzadas en XRP/SOL/BNB se midieron
        // con él—. Ahora sale del `tick_size` del símbolo registrado.
        let half_spread = ((t.ask_price - t.bid_price) / 2.0).max(min_half_spread);
        let sim_bid = t.bid_price - half_spread;
        let sim_ask = t.ask_price + half_spread;
        let bid_qty = t.bid_qty;
        let ask_qty = t.ask_qty;

        // Update AggTrade and L2 (same as backtest-engine/lib.rs lines 163-165)
        // D-708: el flujo agregado lo actualiza el núcleo en `process_event`.
        core.arena.update_l2_depth(0, bid_qty, ask_qty);

        let ts = t.timestamp;
        let is_minute_kline = prev_ts == 0 || (ts / 60_000) != (prev_ts / 60_000);
        prev_ts = ts;

        let ts_sec = (t.timestamp / 1000) as i64;
        let _dt = chrono::DateTime::<chrono::Utc>::from_timestamp(ts_sec, 0).unwrap_or_default();
        // Compute real OBI from bid/ask quantities for depth_obi parameter
        let real_obi = if (bid_qty + ask_qty) > 0.0 {
            (bid_qty - ask_qty) / (bid_qty + ask_qty)
        } else {
            0.0
        };

        // F3.1: features omni con MACRO REAL del día de esta barra.
        // Solo re-consultamos cuando cambia el día (macro es diaria).
        // D-719 (DÉCIMA OLA · auditoría integral): EL MACRO DISPONIBLE EN t ES
        // EL CIERRE DE t−1. `macro_lookup` devuelve el valor con fecha ≤ día, es
        // decir el CIERRE DEL PROPIO DÍA: a las 00:01 UTC el motor ya conocía el
        // S&P, el Nasdaq, el VIX, el DGS10, el DXY y el WTI de la jornada que aún
        // no ha ocurrido, y cuatro de esos seis son features directas del bosque.
        // El evaluador del GA (`booktick_replay`) ya usa t−1 y lo documenta; el
        // binario que dicta el veredicto y alimenta al walk-forward usaba t, de
        // modo que medía con información del futuro y además divergía de la
        // selección y del contrato con el que se entrenó el bosque.
        let day = ((ts / 86_400_000) as i64 - 1).max(0);
        if day != last_macro_day {
            last_macro_day = day;
            use std::sync::atomic::Ordering as Ord2;
            omni_state
                .sp500
                .store(macro_lookup(0, day).to_bits(), Ord2::Relaxed);
            omni_state
                .nasdaq
                .store(macro_lookup(1, day).to_bits(), Ord2::Relaxed);
            omni_state
                .vix
                .store(macro_lookup(2, day).to_bits(), Ord2::Relaxed);
            omni_state
                .us10y
                .store(macro_lookup(3, day).to_bits(), Ord2::Relaxed);
            omni_state
                .dxy
                .store(macro_lookup(4, day).to_bits(), Ord2::Relaxed);
            omni_state
                .oil_wti
                .store(macro_lookup(5, day).to_bits(), Ord2::Relaxed);
        }
        let omni_features = omni_state.get_features();

        let (new_ord_1, closed_ord_1) = core.process_event(
            0,
            false, // is_trade = false for Depth event
            is_minute_kline,
            true, // is_depth = true
            price,
            vol,
            sim_bid,
            sim_ask,
            bid_qty,
            ask_qty,
            real_obi,
            0.0,
            ts,
            false,
            &omni_features,
            false,
        );
        // D-717: el lado agresor es el del dato, no una comparación con el
        // libro simulado (que era falsa siempre).
        let sim_trade_buyer_maker = is_buyer_maker;
        let (new_ord_2, closed_ord_2) = core.process_event(
            0,
            true,  // is_trade = true for Trade event
            false, // is_kline_closed = false (already processed)
            false, // is_depth = false
            price,
            vol,
            sim_bid,
            sim_ask,
            bid_qty,
            ask_qty,
            real_obi,
            0.0,
            ts,
            false,
            &omni_features,
            sim_trade_buyer_maker,
        );

        let new_ord = new_ord_1.or(new_ord_2);
        let closed_ord = closed_ord_1.or(closed_ord_2);

        // DEEP DIAGNOSTIC: Log ML predictions, features, and signal flow every 100k ticks
        if i < warmup_ticks + 5 || (i % 100_000 == 0) {
            let atr_pct = core.feature_engines[0].get_atr_pct();
            let features = core.feature_engines[0].get_features();
            let ml_prob = core.last_ml_prob;
            let regime = core.feature_engines[0].get_market_regime();
            let scalp_intent = core.last_scalp_intent[0];
            let current_cap = arena.unified_capital.load(Ordering::Relaxed);
            let ml_threshold = arena.config.ml_threshold_long.load(Ordering::Relaxed);

            println!("🔍 [DEEP TRACE] i={}: atr={:.8} ml_prob={:.4} ml_thresh={:.4} regime={:?} scalp_sig={:?} cap={:.2} obi={:.4} hurst={:.4}", 
                i, atr_pct, ml_prob, ml_threshold, regime, scalp_intent.signal, current_cap, real_obi, features[1]);
        }

        // Count opens
        if new_ord.is_some() {
            if total_opens < 5 {
                let atr_pct = core.feature_engines[0].get_atr_pct();
                let dyn_atr = arena.config.dynamic_atr_min.load(Ordering::Relaxed);
                let live_maker = arena.config.live_maker_fee.load(Ordering::Relaxed);
                let live_taker = arena.config.live_taker_fee.load(Ordering::Relaxed);
                println!(
                    "🔍 [OPEN TRACE] Trade {}: ATR={:.6}, dyn_atr={:.6}, maker={:.6}, taker={:.6}",
                    total_opens, atr_pct, dyn_atr, live_maker, live_taker
                );
            }
            total_opens += 1;
        }

        // Count closes and PnL
        if let Some((is_long, net_close_pnl, qty)) = closed_ord {
            total_closes += 1;
            total_trades += 1;

            let live_taker = arena.config.live_taker_fee.load(Ordering::Relaxed);
            let exit_fee_est = qty * price * live_taker;
            let gross_pnl = net_close_pnl + exit_fee_est;
            let true_net_pnl = net_close_pnl;

            total_pnl_dollars += true_net_pnl;
            total_gross_pnl_dollars += gross_pnl;
            if gross_pnl > 0.0 {
                total_gross_wins += 1;
            }
            if true_net_pnl > 0.0 {
                total_net_wins += 1;
            }

            let r_code = arena.coins[0].last_close_reason.load(Ordering::Relaxed);
            let r_name = match r_code {
                1 => "TP",
                2 => "SL",
                3 | 4 => "TRAIL",
                5 => "ZOMBIE",
                _ => "TOXIC",
            };
            println!(
                "🚪 [TRADE #{:02}] dir={:5} reason={:6} net=${:+.4} gross=${:+.4} cap=${:.4}",
                total_trades,
                if is_long { "LONG" } else { "SHORT" },
                r_name,
                true_net_pnl,
                gross_pnl,
                arena.unified_capital.load(Ordering::Relaxed)
            );
            match r_code {
                1 => reason_tp += 1,
                2 => reason_sl += 1,
                3 | 4 => reason_trail += 1,
                5 => reason_zombie += 1,
                _ => reason_toxic += 1,
            }
            if is_long {
                long_trades += 1;
                long_pnl += true_net_pnl;
                if true_net_pnl > 0.0 {
                    long_net_wins += 1;
                }
            } else {
                short_trades += 1;
                short_pnl += true_net_pnl;
                if true_net_pnl > 0.0 {
                    short_net_wins += 1;
                }
            }
        }

        // Drawdown tracking
        let current_cap = arena.unified_capital.load(Ordering::Relaxed);
        if current_cap > peak_capital {
            peak_capital = current_cap;
        }
        let dd = if peak_capital > 0.0 {
            (peak_capital - current_cap) / peak_capital
        } else {
            0.0
        };
        if dd > max_drawdown {
            max_drawdown = dd;
        }

        // Sample equity curve every 1000 ticks
        if (i - warmup_ticks) % 1000 == 0 {
            pnl_series.push(current_cap);
        }

        // Margin Call
        if current_cap <= 0.0 {
            println!(
                "💀 [MARGIN CALL] Capital agotado en tick {}. Abortando simulación.",
                i
            );
            break;
        }

        // Progress report every 10% of ticks
        if (i - warmup_ticks) % ((num_ticks - warmup_ticks) / 10).max(1) == 0 && i > warmup_ticks {
            let pct = ((i - warmup_ticks) as f64 / (num_ticks - warmup_ticks) as f64) * 100.0;
            println!(
                "   📈 {:.0}% — Capital: ${:.4} | Trades: {} | DD: {:.2}%",
                pct,
                current_cap,
                total_trades,
                max_drawdown * 100.0
            );
        }
    }

    let start_timestamp = ticks_slice[warmup_ticks].timestamp;
    let end_timestamp = ticks_slice[num_ticks - 1].timestamp;

    use chrono::{TimeZone, Utc};
    let start_date = Utc.timestamp_millis_opt(start_timestamp as i64).unwrap();
    let end_date = Utc.timestamp_millis_opt(end_timestamp as i64).unwrap();

    let sim_elapsed = sim_start.elapsed();
    let gross_roi = if initial_capital > 0.0 {
        total_gross_pnl_dollars / initial_capital
    } else {
        0.0
    };
    let final_capital = arena.unified_capital.load(Ordering::Relaxed);
    let roi = if initial_capital > 0.0 {
        (final_capital - initial_capital) / initial_capital
    } else {
        0.0
    };
    let gross_win_rate = if total_trades > 0 {
        total_gross_wins as f64 / total_trades as f64
    } else {
        0.0
    };
    let net_win_rate = if total_trades > 0 {
        total_net_wins as f64 / total_trades as f64
    } else {
        0.0
    };

    // Calcular Sharpe simplificado sobre la curva de equity
    let sharpe = if pnl_series.len() > 2 {
        let returns: Vec<f64> = pnl_series
            .windows(2)
            .map(|w| (w[1] - w[0]) / w[0].max(1e-10))
            .collect();
        let mean_ret = returns.iter().sum::<f64>() / returns.len() as f64;
        let variance = returns.iter().map(|r| (r - mean_ret).powi(2)).sum::<f64>()
            / (returns.len() - 1) as f64;
        let std_dev = variance.sqrt();
        if std_dev > 0.0 {
            mean_ret / std_dev
        } else {
            0.0
        }
    } else {
        0.0
    };

    // ═══════════════════════════════════════════════════════════════════════
    // PASO 6: Reporte Final
    // ═══════════════════════════════════════════════════════════════════════
    println!();
    println!("🛡️ ═══════════════════════════════════════════════════════════════");
    println!("🛡️  VEREDICTO FORENSE — PARIDAD 1:1 CON PRODUCCIÓN");
    println!("🛡️ ═══════════════════════════════════════════════════════════════");
    println!();
    println!(
        "  📅 Periodo:           {} a {}",
        start_date.format("%Y-%m-%d %H:%M:%S"),
        end_date.format("%Y-%m-%d %H:%M:%S")
    );
    println!("  💰 Capital Inicial:   ${:.4}", initial_capital);
    println!("  💰 Capital Final:     ${:.4}", final_capital);
    println!(
        "  💸 GROSS PnL:         ${:.4} (Antes de fees)",
        total_gross_pnl_dollars
    );
    println!(
        "  📉 Total Fees Paid:   ${:.4}",
        total_gross_pnl_dollars - total_pnl_dollars
    );
    println!(
        "  💵 NET PnL:           ${:.4} (Despues de fees)",
        total_pnl_dollars
    );
    println!("  📈 GROSS ROI:         {:.2}%", gross_roi * 100.0);
    println!("  📈 NET ROI:           {:.2}%", roi * 100.0);
    println!("  📊 Trades Totales:    {}", total_trades);
    println!(
        "  ✅ GROSS Wins:        {} ({:.1}%) [Sin comisiones]",
        total_gross_wins,
        gross_win_rate * 100.0
    );
    println!(
        "  💎 NET Wins:          {} ({:.1}%) [Post-Comisiones]",
        total_net_wins,
        net_win_rate * 100.0
    );
    println!("  ❌ NET Losses:        {}", total_trades - total_net_wins);
    println!(
        "  📈 LONG Trades:       {} | NET Wins: {} ({:.1}%) | Net PnL: ${:.4}",
        long_trades,
        long_net_wins,
        if long_trades > 0 {
            long_net_wins as f64 / long_trades as f64 * 100.0
        } else {
            0.0
        },
        long_pnl
    );
    println!(
        "  📉 SHORT Trades:      {} | NET Wins: {} ({:.1}%) | Net PnL: ${:.4}",
        short_trades,
        short_net_wins,
        if short_trades > 0 {
            short_net_wins as f64 / short_trades as f64 * 100.0
        } else {
            0.0
        },
        short_pnl
    );
    println!(
        "  🎯 Exit Reasons:      TP: {} | SL: {} | TRAIL: {} | ZOMBIE: {} | TOXIC: {}",
        reason_tp, reason_sl, reason_trail, reason_zombie, reason_toxic
    );
    println!("  📉 Max Drawdown:      {:.2}%", max_drawdown * 100.0);
    println!("  📐 Sharpe Ratio:      {:.4}", sharpe);
    println!("  🚀 Continuous Opens:  {}", total_opens);
    println!("  🔴 Continuous Closes: {}", total_closes);
    println!("  ⏱️  Tiempo Simulación: {:?}", sim_elapsed);
    println!("  ⏱️  Tiempo Total:      {:?}", t0.elapsed());
    println!("  🔢 Ticks Procesados:  {}", num_ticks - warmup_ticks);
    println!(
        "  ⚡ Velocidad:         {:.0} ticks/seg",
        (num_ticks - warmup_ticks) as f64 / sim_elapsed.as_secs_f64()
    );
    println!();

    // Diagnóstico de divergencia
    if total_trades == 0 {
        println!("⚠️ [DIAGNÓSTICO] CERO TRADES generados. Posibles causas:");
        println!("   1. ML thresholds demasiado altos (Scalp OBI: {:.4}, Swing L: {:.4}, Swing S: {:.4})",
            genome.scalp_obi_threshold, genome.ml_threshold_long, genome.ml_threshold_short);
        println!(
            "   2. ATR thresholds muy restrictivos (ATR min: {:.6})",
            genome.dynamic_atr_min
        );
        println!("   3. Capital insuficiente para notional mínimo con el leverage actual");
        println!("   4. Kill switch activado (verificar macro features)");
    } else if roi < -0.5 {
        println!(
            "🚨 [DIAGNÓSTICO] PÉRDIDA SEVERA (>{:.0}%). Este genoma NO es seguro para producción.",
            roi.abs() * 100.0
        );
        println!("   Recomendación: Ejecutar el Evolver antes de pasar a mainnet.");
    } else if roi > 0.0 {
        println!(
            "✅ [DIAGNÓSTICO] Rendimiento POSITIVO. Genoma potencialmente apto para producción."
        );
        if roi > 1.0 {
            println!("   ⚠️ PRECAUCIÓN: ROI > 100% puede indicar overfitting o falta de penalización de slippage.");
        }
    }

    // D-686: resumen legible por máquina, una sola línea.
    println!(
        "FORENSIC_JSON {{\"start_tick\":{},\"ticks\":{},\"start_ts\":{},\"end_ts\":{},\"initial_capital\":{:.6},\"final_capital\":{:.6},\"net_roi\":{:.6},\"gross_roi\":{:.6},\"trades\":{},\"net_wins\":{},\"max_drawdown\":{:.6},\"sharpe\":{:.6},\"tp\":{},\"sl\":{},\"trail\":{},\"zombie\":{},\"toxic\":{}}}",
        start_tick,
        num_ticks,
        start_timestamp,
        end_timestamp,
        initial_capital,
        final_capital,
        roi,
        gross_roi,
        total_trades,
        total_net_wins,
        max_drawdown,
        if sharpe.is_finite() { sharpe } else { 0.0 },
        reason_tp,
        reason_sl,
        reason_trail,
        reason_zombie,
        reason_toxic
    );
    // Diagnóstico por dirección del embudo de entrada (sólo telemetría).
    println!("{}", core.diag_dir.report());
    // Rechazos del risk-engine por compuerta y dirección (sólo telemetría).
    println!("RISK_REJECT {}", risk_engine::reject_report());

    println!();
    println!("🛡️ ═══════════════════════════════════════════════════════════════");
    println!("🛡️  FIN DE AUDITORÍA FORENSE");
    println!("🛡️ ═══════════════════════════════════════════════════════════════");
}
