//! OPEN F325 (censo C-F7b, lote 34, fila N8; RET-34c): una entrada IOC de
//! LARGO en la orden mínima del lote puede salir con nocional bajo el mínimo
//! del símbolo y Binance la rechaza (-4164). Una OPEN que pasa NO es remedio:
//! cuando se arregle, esta prueba falla y se invierte.
//!
//! Cadena viva (por ancla; líneas tras la tanda 7 del censo):
//! 1. risk-engine lib.rs (`nocional_minimo_ejecutable` al `current_price`,
//!    :1467-1473) dimensiona la orden mínima del lote al precio MEDIO con
//!    +0,10 USD; `required_margin_for_min_notional * 1.0005` (:1832) acota
//!    la sonda.
//! 2. god-engine-core lib.rs (`cantidad_de_la_orden(nominal_size,
//!    real_entry_price, …)`, :10110-10139) divide por la entrada MODELADA
//!    de `RealityPhysics::calculate_market_entry`, ask·(1+s) con s ≥ el
//!    suelo de slippage, y sólo garantiza el mínimo a ESE precio.
//! 3. god_engine.rs (`ref_price`, `dyn_slip`, `raw_ioc_price` y
//!    `round_price_to_tick`, :5111-5127) pone el límite IOC de compra en
//!    ask·(1+dyn_slip) redondeado HACIA ABAJO al tick: queda bajo la entrada.
//! 4. executor.rs baja la cantidad al paso; nadie revisa el nocional.
//! 5. Binance juzga el nocional de una LIMIT/IOC al precio de la orden.
//! Los cortos no fallan: al dividir por un precio menor ganan lotes y
//! conservan el margen de 0,10 USD (2 % > dyn_slip ≤ 0,35 %).
//!
//! La prueba copia la aritmética del host, del núcleo y del riesgo. Las
//! guardas de texto del final fijan esas líneas (riesgo: el nocional mínimo
//! al precio actual y el tope de la sonda; núcleo: la cantidad a la entrada
//! modelada y su suelo de slippage; host: precio de referencia, deslizamiento,
//! límite y redondeo) y la ponen roja si cambian, para que no siga verde
//! sobre una cadena que ya no existe. Un arreglo que no toque ninguna de esas
//! líneas (por ejemplo, volver a comprobar el nocional en el ejecutor tras
//! bajar al paso) no la pone roja: quien lo haga debe invertirla. Si
//! `dispatch_entry` empieza a rechazar la orden, la prueba también falla.
use execution_engine::entry_dispatch::{
    dispatch_entry, EntryRequest, EntryRoute, EntryTransport,
};
use execution_engine::executor::OrderExecutor;
use execution_engine::fixed_filter::{BinanceFixedFilter, FilterRejectReason};
use risk_engine::capital_regime::{
    cantidad_de_la_orden, effective_min_notional, nocional_minimo_ejecutable,
};
use std::sync::Mutex;

#[derive(Default)]
struct Captura {
    configuraciones: Mutex<u32>,
    enviadas: Mutex<Vec<(f64, EntryRoute)>>,
}

impl EntryTransport for Captura {
    fn supports_native_iceberg(&self) -> bool {
        false
    }

    async fn configure_leverage(&self, _symbol: &str, _leverage: u32) -> Result<(), String> {
        *self.configuraciones.lock().map_err(|e| e.to_string())? += 1;
        Ok(())
    }

    async fn submit_entry(&self, request: &EntryRequest<'_>) -> Result<(), String> {
        self.enviadas
            .lock()
            .map_err(|e| e.to_string())?
            .push((request.quantity, request.route));
        Ok(())
    }
}

/// Fuente sin comentarios de línea ni espacios (patrón de entrada_host_contract.rs).
fn sin_espacios(fuente: &str) -> String {
    fuente
        .lines()
        .filter(|l| !l.trim_start().starts_with("//"))
        .collect::<String>()
        .split_whitespace()
        .collect()
}

#[tokio::test]
async fn open_f325_ioc_largo_en_el_minimo_sale_bajo_min_notional() {
    let paso: f64 = 0.1;
    let cantidad_minima: f64 = 0.1;
    let tick: f64 = 0.001;
    let minimo = effective_min_notional(5.0);
    let ask: f64 = 1.219;
    let medio: f64 = 1.2185;
    // base_slippage_floor; impacto + latencia por debajo del suelo.
    let suelo: f64 = 0.000_45;
    let atr: f64 = 0.000_7;

    // 1. Riesgo: orden mínima del lote al medio (+0,10 USD), sonda ×1,0005.
    let orden_minima = nocional_minimo_ejecutable(minimo + 0.1, paso, cantidad_minima, medio)
        .unwrap_or(f64::NAN);
    assert!((orden_minima - 4.2 * medio).abs() < 1e-9, "42 lotes: {orden_minima}");
    let nominal = orden_minima * 1.0005;

    // 2. Núcleo: entrada modelada ask·(1+s) y cantidad de la orden.
    let entrada = ask * (1.0 + suelo);
    let qty = cantidad_de_la_orden(nominal, entrada, minimo, paso, cantidad_minima);
    assert!(qty > 4.1 && qty < 4.2, "qty={qty}");
    let lotes = (qty / paso).floor() * paso;
    assert!(lotes * entrada >= minimo, "el núcleo cree cumplir: {}", lotes * entrada);

    // 3. Host: límite IOC de compra, redondeado hacia abajo al tick.
    let dyn_slip = (suelo.max(0.0001) + atr * 0.20).clamp(0.0005, 0.0035);
    let limite_crudo = ask * (1.0 + dyn_slip);
    let limite = OrderExecutor::round_price_to_tick(limite_crudo, tick, false);
    assert_eq!(limite, 1.219);
    assert!(limite < entrada);

    // 4-5. Lo que juzga el exchange: 41 lotes a 1,219 = 4,9979 USD.
    assert!(lotes * limite < minimo, "nocional al límite: {}", lotes * limite);

    // El cuantizador canónico SÍ lo detecta.
    let canonico = BinanceFixedFilter::with_limits(
        tick,
        paso,
        tick,
        f64::MAX,
        cantidad_minima,
        f64::MAX,
        minimo,
    );
    assert!(matches!(
        canonico.validate_and_quantize(limite_crudo, qty, false),
        Err(FilterRejectReason::NotionalBelowMin { current, min }) if current < min
    ));

    // dispatch_entry lo deja pasar: configura el apalancamiento y lo envía.
    let captura = Captura::default();
    let pedido = EntryRequest {
        symbol: "RET325USDT",
        is_long: true,
        quantity: qty,
        step_size: paso,
        tick_size: tick,
        leverage: 1,
        client_order_id: "ret325_minimo_ioc",
        route: EntryRoute::Ioc { price: limite },
    };
    assert!(
        dispatch_entry(&captura, &pedido).await.is_ok(),
        "OPEN cerrada: invierte esta prueba"
    );
    assert_eq!(
        *captura
            .configuraciones
            .lock()
            .unwrap_or_else(|e| e.into_inner()),
        1
    );
    let enviadas = captura
        .enviadas
        .lock()
        .unwrap_or_else(|e| e.into_inner());
    assert_eq!(enviadas.len(), 1, "OPEN cerrada: invierte esta prueba");
    assert_eq!(enviadas[0], (qty, EntryRoute::Ioc { price: limite }));
}

/// Los literales que la OPEN reproduce siguen en el host y en el riesgo.
#[test]
fn open_f325_la_cadena_reproducida_sigue_en_el_codigo() {
    let host = sin_espacios(include_str!("../../../src/bin/god_engine.rs"));
    for ancla in [
        "letbase_slip=arena_clone.config.base_slippage_floor.load(Ordering::Relaxed).max(0.0001);",
        "letdyn_slip=(base_slip+entry_atr_pct*0.20).clamp(0.0005,0.0035);",
        "letref_price=iffinal_is_long{ifdap>0.0{dap}else{current_price}}else{ifdbp>0.0{dbp}else{current_price}};",
        "letraw_ioc_price=iffinal_is_long{ref_price*(1.0+dyn_slip)}else{ref_price*(1.0-dyn_slip)};",
        "round_price_to_tick(raw_ioc_price,dyn_tick_size,!final_is_long,);",
    ] {
        assert_eq!(host.matches(ancla).count(), 1, "host: cambió «{ancla}»; revisar la OPEN");
    }
    let riesgo = sin_espacios(include_str!("../../risk-engine/src/lib.rs"));
    for ancla in [
        "nocional_minimo_ejecutable(dynamic_min_notional+0.1,spec.step_size,spec.min_qty,current_price,)",
        "final_margin=final_margin.min(required_margin_for_min_notional*1.0005);",
    ] {
        assert_eq!(riesgo.matches(ancla).count(), 1, "riesgo: cambió «{ancla}»; revisar la OPEN");
    }
    let nucleo = sin_espacios(include_str!("../../god-engine-core/src/lib.rs"));
    let ancla = "risk_engine::capital_regime::cantidad_de_la_orden(nominal_size,real_entry_price,";
    assert_eq!(nucleo.matches(ancla).count(), 1, "núcleo: cambió «{ancla}»; revisar la OPEN");
    let fisica = sin_espacios(include_str!("../../god-engine-core/src/reality_physics.rs"));
    let entrada = fisica
        .split("pubfncalculate_market_entry(")
        .nth(1)
        .and_then(|resto| resto.split("pubfncalculate_maker_entry(").next())
        .unwrap_or("");
    for ancla in [
        "lettotal_slippage_pct=(slippage_impact_pct+latency_slippage).max(base_slippage_floor).clamp(0.0,0.05);",
        "base_price*(1.0+total_slippage_pct)",
    ] {
        assert_eq!(entrada.matches(ancla).count(), 1, "entrada modelada: cambió «{ancla}»; revisar la OPEN");
    }
}
