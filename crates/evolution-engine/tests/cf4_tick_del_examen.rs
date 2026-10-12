//! CL-88 (atr-2, revisión de los PR #43–#55): el examen del demonio usa su
//! propio tick como piso de resolución del ATR.
//!
//! Teoría: el ATR de las decisiones nunca baja de un tick (CL-69), porque la
//! rejilla de precios es la resolución del instrumento. El examen walk-forward
//! reproduce retornos del mercado en un instrumento sintético de 100 USD
//! (WFD{i}), cuya rejilla es la relativa mediana del universo vivo
//! (`WfMarketEnv::tick_pct`) llevada a 100 USD. Pero sus monedas ocupan los
//! slots 0..n del núcleo, que son los del universo vivo, y `tick_size_de(i)`
//! devuelve el tick de `universe[i]`. Con BTCUSDT (tick 0,1) en el slot 0, la
//! serie de 100 USD tenía un piso de 10 pb.
//!
//! La prueba fija el universo vivo, así que vive en su propio proceso.

use evolution_engine::online_daemon::{atr_pct_del_examen, WfMarketEnv};
use quantum_arena::symbol_registry::{update_registry, SymbolSpec};

struct Lcg(u64);
impl Lcg {
    fn u(&mut self) -> f64 {
        self.0 = self
            .0
            .wrapping_mul(6_364_136_223_846_793_005)
            .wrapping_add(1_442_695_040_888_963_407);
        ((self.0 >> 11) as f64 + 0.5) / (1u64 << 53) as f64
    }
    fn n(&mut self) -> f64 {
        let (u1, u2) = (self.u(), self.u());
        (-2.0 * u1.ln()).sqrt() * (2.0 * std::f64::consts::PI * u2).cos()
    }
}

/// Barra del examen: 8 micro-ticks de 2 s.
const BARRA_S: f64 = 16.0;
/// σ del log-retorno de 60 s del mercado de la prueba.
const SIGMA60: f64 = 2e-4;

/// Verdad: el ATR del examen mide la volatilidad de la serie, como en vivo,
/// y su piso es el tick del examen (1,7·10⁻⁴ USD sobre 100 USD con la
/// rejilla relativa de BTCUSDT a 60 000 USD), no el de BTCUSDT (0,1 USD).
/// Con σ60 = 2 pb, el ATR esperado es √(8/π)·σ60 = 3,19 pb. Esperado: el ATR
/// tras 400 barras está en [0,5; 2] veces ese valor, en tres caminos. Antes
/// salía en el piso de 10 pb (3,1 veces).
#[test]
fn cl88_el_atr_del_examen_no_usa_el_tick_de_un_simbolo_vivo() {
    quantum_arena::symbols::update_dynamic_universe(vec!["BTCUSDT".into()]);
    update_registry(vec![SymbolSpec {
        symbol: "BTCUSDT".into(),
        step_size: 0.001,
        tick_size: 0.1,
        min_qty: 0.001,
        min_notional: 100.0,
        max_leverage: 125,
        maker_fee: 0.0002,
        taker_fee: 0.0005,
        is_shadow: false,
    }]);
    let market = WfMarketEnv {
        observed_half_spread_pct: 1e-4,
        tick_pct: 0.1 / 60_000.0,
        max_leverage: 20.0,
    };
    let esperado = (8.0 / std::f64::consts::PI).sqrt() * SIGMA60;
    let sigma_barra = SIGMA60 * (BARRA_S / 60.0).sqrt();
    for semilla in 0..3u64 {
        let mut rng = Lcg(0xC188_0000 ^ semilla);
        let serie: Vec<f64> = (0..400).map(|_| sigma_barra * rng.n()).collect();
        let atr = atr_pct_del_examen(&serie, &market);
        let razon = atr / esperado;
        println!("camino {semilla}: ATR = {:.2} pb, ATR/esperado = {razon:.3}", atr * 1e4);
        assert!(
            (0.5..=2.0).contains(&razon),
            "camino {semilla}: ATR del examen {:.2} pb, {razon:.2} veces √(8/π)·σ60",
            atr * 1e4
        );
    }
}
