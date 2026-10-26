//! OCO-F3: sonda forense del bracket OCO en TESTNET.
//!
//! Reproduce EXACTAMENTE el ciclo del motor god_engine: entrada MARKET short
//! → settle de fills → lectura de posición REAL (positionRisk) → bracket
//! TP/SL con execute_oco_order → verificación → limpieza total.
//!
//! Motivo: las 3 falencias del incidente SOLUSDT (2026-09-15) quedaron
//! invisibles porque execute_oco_order descartaba el cuerpo del rechazo.
//! Esta sonda imprime cada respuesta/error crudo de Binance.
//!
//! Uso: export BINANCE_DEMO_API_KEY/SECRET (o BINANCE_TESTNET_*) y ejecutar.
//! Sólo testnet: el ejecutor se crea con `is_testnet = true` (no hay otra
//! guarda).
//!
//! ⚠️ Cancela las órdenes de su símbolo y aplana TODAS las posiciones de la
//! cuenta (`flatten_all_positions`), también las del motor: nunca con el motor
//! vivo en la misma cuenta. Sale por la misma IP con su propio freno 429/418,
//! que no ve el del motor: no lanzarla durante un freno del motor
//! (docs/BINANCE_API_MAP.md §5, fila N32 del censo).

use execution_engine::executor::{ExecutionProvider, OrderExecutor};

/// N32 (CL-205): por caracteres. Cortar por bytes entraba en
/// pánico con un error que llevaba una letra acentuada o un emoji en ese
/// punto.
fn truncate_str(s: &str, max: usize) -> String {
    if s.chars().count() <= max {
        s.to_string()
    } else {
        format!("{}…", execution_engine::order_types::truncate(s, max))
    }
}

/// Microsegundos desde la época para los ids de la sonda; 0 con un reloj
/// anterior a 1970, sin pánico (CL-205).
fn micros_ahora() -> u128 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map_or(0, |d| d.as_micros())
}

#[tokio::main]
async fn main() {
    let (key, secret) = (
        std::env::var("BINANCE_DEMO_API_KEY")
            .or_else(|_| std::env::var("BINANCE_TESTNET_API_KEY"))
            .unwrap_or_default(),
        std::env::var("BINANCE_DEMO_SECRET_KEY")
            .or_else(|_| std::env::var("BINANCE_TESTNET_SECRET_KEY"))
            .unwrap_or_default(),
    );
    if key.is_empty() || secret.is_empty() {
        eprintln!("❌ Faltan BINANCE_DEMO_API_KEY / BINANCE_DEMO_SECRET_KEY");
        std::process::exit(1);
    }

    let symbol = std::env::var("OCO_PROBE_SYMBOL").unwrap_or_else(|_| "SOLUSDT".to_string());
    let qty: f64 = std::env::var("OCO_PROBE_QTY")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(0.2);

    println!("═══ SONDA OCO (testnet) — {} qty={} ═══", symbol, qty);

    let mut exec = OrderExecutor::new(key, secret, true);
    exec.set_paper_trading(false);

    // 0. Modo de posición: el motor asume HEDGE; verificar contra el exchange.
    match exec.ensure_hedge_mode().await {
        Ok(dual) => println!("[0] positionSide/dual = {}", dual),
        Err(e) => println!("[0] ❌ ensure_hedge_mode: {}", truncate_str(&e, 200)),
    }

    // 1. Filtros reales del símbolo (fuente del tickSize que usa el bracket).
    match exec.fetch_all_symbol_filters().await {
        Ok(map) => println!(
            "[1a] exchangeInfo: {} símbolos; {} presente: {}",
            map.len(),
            symbol,
            map.contains_key(&symbol)
        ),
        Err(e) => println!("[1a] ❌ exchangeInfo crudo: {}", truncate_str(&e, 300)),
    }
    let (step, tick, min_notional) = match exec.get_symbol_filter(&symbol).await {
        Ok(f) => {
            println!("[1] filtros: step={} tick={} minNotional={}", f.step_size, f.tick_size, f.min_notional);
            (f.step_size, f.tick_size, f.min_notional)
        }
        Err(e) => {
            println!("[1] ❌ get_symbol_filter: {}", truncate_str(&e, 200));
            return;
        }
    };

    // 2. Aplastar estado previo (sonda idempotente).
    let _ = exec.cancel_all_symbol_orders(&symbol).await;
    let _ = exec.flatten_all_positions().await;
    println!("[2] estado previo purgado");

    // N32 (CL-205): desde la entrada la cuenta puede tener una posición (un
    // error AMBIGUOUS de la entrada también pudo llenar). Un fallo de los
    // pasos 3 a 6 sale del bloque y pasa por la limpieza del paso 7; antes
    // volvía de main y dejaba la posición abierta y sin TP/SL.
    let bracket: Option<Result<(), String>> = 'sonda: {
        // 3. Entrada MARKET SHORT — mismo camino que el motor
        //    (execute_raw_qty_with_client_id).
        let coid = format!("probe_e_{}", micros_ahora());
        println!("[3] entrada MARKET SHORT {} {}…", symbol, qty);
        if let Err(e) = exec
            .execute_raw_qty_with_client_id(&symbol, false, qty, step, &coid)
            .await
        {
            println!("[3] ❌ entrada rechazada: {}", truncate_str(&e, 300));
            break 'sonda None;
        }

        // 4. Settle de fills (OCO-F2) + posición REAL.
        tokio::time::sleep(std::time::Duration::from_millis(600)).await;
        let (entry_price, real_qty) = match exec.fetch_position_risk().await {
            Ok(ps) => {
                // HEDGE: positionRisk devuelve DOS registros por símbolo (LONG y
                // SHORT); el lado vacío trae positionAmt=0 y ordena primero.
                // Seleccionar el registro CON magnitud, no el primero por nombre.
                let Some(p) = ps
                    .iter()
                    .find(|p| p.symbol == symbol && p.position_amt.abs() > 0.0)
                else {
                    println!("[4] ❌ sin posición con magnitud en positionRisk tras entrada");
                    break 'sonda None;
                };
                println!(
                    "[4] posición REAL: amt={:.6} entry={:.4} (hedge side: {})",
                    p.position_amt,
                    p.entry_price,
                    if p.position_amt < 0.0 { "SHORT" } else { "LONG" }
                );
                (p.entry_price, p.position_amt.abs())
            }
            Err(e) => {
                println!("[4] ❌ positionRisk: {}", truncate_str(&e, 200));
                break 'sonda None;
            }
        };
        if real_qty <= 0.0 {
            println!("[4] ❌ posición 0 — la entrada no llenó");
            break 'sonda None;
        }
        if entry_price * real_qty < min_notional {
            println!("[4] ⚠️ notional {:.2} < minNotional {} — el bracket puede caer por -4164", entry_price * real_qty, min_notional);
        }

        // 5. Bracket OCO — precios con artefactos float deliberados (el redondeo
        //    interno al tick debe absorberlos; así también probamos esas funcs).
        let tp = entry_price * 0.9800000001; // short: TP abajo
        let sl = entry_price * 1.0200000001; // short: SL arriba
        let base_id = format!("probe_oco_{}", micros_ahora());
        println!("[5] bracket: qty={:.6} TP={:.6} SL={:.6} base_id={} (len={})", real_qty, tp, sl, base_id, base_id.len());
        let bracket = exec
            .execute_oco_order(&symbol, false, real_qty, tp, sl, step, tick, &base_id)
            .await;
        match &bracket {
            Ok(()) => println!("[5] ✅ BRACKET COLOCADO — ambas piernas aceptadas"),
            Err(e) => println!("[5] ❌ bracket rechazado:\n{}", e),
        }

        // 6. Verificación final: órdenes abiertas + posición.
        if let Ok(ps) = exec.fetch_position_risk().await {
            if let Some(p) = ps.iter().find(|p| p.symbol == symbol) {
                println!("[6] posición final: {:.6} @ {:.4}", p.position_amt, p.entry_price);
            }
        }
        Some(bracket)
    };

    // 7. Limpieza total. N32 (CL-205): un aplanado que falla se dice, con su
    // error; antes se leía como cero cerradas y «órdenes purgadas», y se
    // contaban los símbolos cancelados como posiciones cerradas.
    let _ = exec.cancel_all_symbol_orders(&symbol).await;
    match exec.flatten_all_positions().await {
        Ok((_, cerradas)) => println!("[7] limpieza: {} posiciones cerradas, órdenes purgadas", cerradas),
        Err(e) => println!(
            "[7] ❌ limpieza incompleta: {} — revisar la cuenta a mano (posición sin TP/SL)",
            truncate_str(&e, 300)
        ),
    }

    match bracket {
        Some(Ok(())) => {
            println!("\n═══ VEREDICTO: OCO funcional — el incidente fue la carrera de llenado (OCO-F2) ═══")
        }
        Some(Err(_)) => {
            println!("\n═══ VEREDICTO: OCO aún roto — el error crudo arriba es el diagnóstico ═══")
        }
        None => println!("\n═══ VEREDICTO: sin bracket — la sonda no llegó a colocarlo ═══"),
    }
}
