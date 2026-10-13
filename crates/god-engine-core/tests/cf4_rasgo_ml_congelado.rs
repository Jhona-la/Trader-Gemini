//! CL-68 (C-F4, regla S-8): el rasgo del modelo sigue el ATR con el que se
//! entrenaron los modelos servidos, aunque cambie el ATR de las decisiones.
//!
//! `StatefulEngine::vt_rasgo_ml` es el rasgo que leen 12D[3], 54D[34], [35],
//! [38] y [39] y el exportador. Verdad: sigue el algoritmo legado (semilla de
//! 50 pb, trinquete dentro de la vela, EMA 0,85/0,15 con suelo de 10 pb al
//! cierre de la vela interna de 60 s y kline 0,8/0,2 con suelo de 5 pb),
//! replicado aquí de forma independiente, bit a bit, sobre una cinta con
//! calentamiento por klines, eventos de precio y un reinicio.
//!
//! Hasta CL-71 además `v_t` coincidía con el rasgo bit a bit; CL-71 cambia
//! `v_t` (σ de 60 s) y esa comprobación se retiró. La réplica del algoritmo
//! legado sigue.

use god_engine_core::stateful_engine::StatefulEngine;

/// LCG de 64 bits (Knuth/MMIX), uniforme en (0, 1).
struct Lcg(u64);
impl Lcg {
    fn u(&mut self) -> f64 {
        self.0 = self
            .0
            .wrapping_mul(6_364_136_223_846_793_005)
            .wrapping_add(1_442_695_040_888_963_407);
        ((self.0 >> 11) as f64 + 0.5) / 9_007_199_254_740_992.0
    }
}

/// Réplica independiente del ATR legado (el de los modelos entrenados).
#[derive(Default)]
struct Legado {
    v: f64,
    ticks: u64,
    inicio_ms: u64,
    alto: f64,
    bajo: f64,
}

impl Legado {
    fn kline(&mut self, alto: f64, bajo: f64, cierre: f64) {
        let tr = (alto - bajo).max(cierre * 0.0005);
        self.v = if self.v == 0.0 || !self.v.is_finite() {
            tr
        } else {
            (self.v * 0.8 + tr * 0.2).max(cierre * 0.0005)
        };
    }

    fn tick(&mut self, precio: f64, ts: u64) {
        if self.ticks == 0 {
            self.inicio_ms = ts;
            self.alto = precio;
            self.bajo = precio;
            if self.v == 0.0 && precio > 0.0 {
                self.v = precio * 0.005;
            }
        } else {
            self.alto = self.alto.max(precio);
            self.bajo = self.bajo.min(precio);
            let intra = (self.alto - self.bajo).max(precio * 0.0010);
            if intra > self.v || self.v == 0.0 || !self.v.is_finite() {
                self.v = intra;
            }
            if ts.saturating_sub(self.inicio_ms) >= 60_000 {
                let tr = (self.alto - self.bajo).max(precio * 0.0010);
                self.v = if self.v == 0.0 || !self.v.is_finite() {
                    tr
                } else {
                    (self.v * 0.85 + tr * 0.15).max(precio * 0.0010)
                };
                self.inicio_ms = ts;
                self.alto = precio;
                self.bajo = precio;
            }
        }
        self.ticks += 1;
    }
}

#[test]
fn cl68_rasgo_ml_sigue_el_algoritmo_legado() {
    let mut rng = Lcg(0xC168);
    let mut motor = StatefulEngine::new();
    let mut legado = Legado::default();
    let mut precio = 100.0f64;
    // Calentamiento: 200 klines de 1 min.
    for _ in 0..200 {
        let apertura = precio;
        let cierre = precio * (1.0 + 0.001 * (rng.u() - 0.5));
        let alto = apertura.max(cierre) * (1.0 + 0.002 * rng.u());
        let bajo = apertura.min(cierre) * (1.0 - 0.002 * rng.u());
        motor.process_kline(apertura, alto, bajo, cierre, 10.0);
        legado.kline(alto, bajo, cierre);
        precio = cierre;
        assert_eq!(
            motor.vt_rasgo_ml.to_bits(),
            legado.v.to_bits(),
            "calentamiento"
        );
    }
    // 2 h de eventos con intervalos irregulares, un reinicio a la mitad.
    let mut ts = 1_000_000u64;
    for i in 0..14_400u64 {
        if i == 7_200 {
            motor.reset();
            legado = Legado::default();
        }
        ts += 200 + (rng.u() * 900.0) as u64;
        precio *= 1.0 + 0.0004 * (rng.u() - 0.5);
        motor.process_tick(precio, 1.0, ts);
        legado.tick(precio, ts);
        assert_eq!(
            motor.vt_rasgo_ml.to_bits(),
            legado.v.to_bits(),
            "evento {i}: rasgo {} frente al legado {}",
            motor.vt_rasgo_ml,
            legado.v
        );
    }
    assert_eq!(
        motor.rasgo_atr_pct_ml().to_bits(),
        (motor.vt_rasgo_ml / motor.last_price).to_bits()
    );
}

/// Guarda de lectores: los constructores de rasgos del modelo leen el rasgo
/// congelado y no el ATR de las decisiones.
#[test]
fn cl68_los_rasgos_del_modelo_leen_el_rasgo_congelado() {
    let nucleo = include_str!("../src/lib.rs");
    assert!(
        nucleo.contains("let delta = self.feature_engines[coin_id].vt_rasgo_ml;")
            && nucleo.contains("let cur_atr = self.feature_engines[coin_id].rasgo_atr_pct_ml();"),
        "build_54d_tensor no lee el rasgo congelado"
    );
    let motor = include_str!("../src/stateful_engine.rs");
    assert!(
        motor.contains("(self.vt_rasgo_ml / self.last_price).clamp(0.0, 1.0)"),
        "get_features()[3] no lee el rasgo congelado"
    );
    let exportador = include_str!("../../../src/bin/feature_exporter.rs");
    assert!(
        exportador.contains("engine.vt_rasgo_ml / t.mid()")
            && exportador.contains("engine.rasgo_atr_pct_ml()"),
        "el exportador de rasgos no lee el rasgo congelado"
    );
    // El maker es un consumidor de decisión: ya no lee el rasgo del modelo.
    assert!(
        !nucleo.contains("get_features()[3] as f64"),
        "el maker sigue leyendo el rasgo del modelo como volatilidad"
    );
}
