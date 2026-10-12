//! CL-107 (LC-F4-10, decisión del dueño «Ambos», 2026-10-12) — FRENO DE LA
//! ESTRATEGIA: UNA PRUEBA CON NIVEL PARA TODO TIEMPO DE QUE LA VENTAJA EXISTE.
//!
//! H0 es «la estrategia tiene la ventaja que pide la meta»:
//! [`VENTAJA_H0_ESTRATEGIA_R`] = +0,30 R por operación, con R el riesgo de la
//! orden al stop con su fricción (`SL + f`). Cada cierre aporta su resultado
//! en R; un proceso e de Ville ([`VilleEProcess`]) apuesta contra H0 con
//! `X = (μ0 − R)/escala` (H0 ⇔ E[X] ≤ 0). Por la desigualdad maximal de
//! Ville, P_H0(sup_t M_t ≥ 1/α) ≤ α: se puede mirar en cada cierre, sin
//! horizonte y sin inflar el error. Cuando `M ≥ 1/α` (20 con α = 0,05), H0 se
//! rechaza y el freno queda echado.
//!
//! Qué hace al rechazar: el riesgo deja de admitir entradas nuevas
//! (`estrategia_falsada`). Las posiciones abiertas se siguen gestionando; no
//! aplana ni dispara el kill-switch global, que son del freno duro de la
//! cuenta ([`crate::drawdown::umbral_freno_duro`]). Se suelta sólo
//! restaurando un estado nuevo (otra estrategia es otra hipótesis).
//!
//! Escala: el proceso exige `X ∈ [−1, 1]`. Con la geometría medida de las
//! órdenes, una pérdida al stop es −1 R y un TP es `w/l` R; la escala cubre
//! ese rango con un margen de 1,5 para deslizamientos y fricciones mayores
//! que las medidas. La escala de cada cierre se fija con lo medido ANTES de
//! ese resultado, así que la apuesta sigue siendo predecible.
//!
//! Con 13 USD, RR 2,25 y la orden mínima, el Monte Carlo del dueño (mezcla
//! de apuestas fijas) daba: falsa alarma ≤ 5 % para todo t; detección de
//! E[R] = 0 en una mediana de 117-141 operaciones y de −0,10 R en 61-75. La
//! apuesta adaptativa de [`VilleEProcess`] tiene su propio contrato
//! (tests/cf4_freno_estrategia.rs).

use crate::ville_e_process::VilleEProcess;

/// Ventaja de H0, en R: la que pide la meta (decisión del dueño).
pub const VENTAJA_H0_ESTRATEGIA_R: f64 = 0.30;
/// Nivel de la prueba: el α = 0,05 del sistema (`drawdown::ALFA_FALSACION`).
pub const ALFA_ESTRATEGIA: f64 = crate::drawdown::ALFA_FALSACION;
/// Margen de la escala sobre el rango de R medido.
const MARGEN_ESCALA: f64 = 1.5;

/// Estado del freno: el proceso e y si ya rechazó.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct FrenoEstrategia {
    pub ville: VilleEProcess,
    /// H0 rechazada: no se admiten entradas nuevas. Queda echado.
    pub rechazada: bool,
}

impl Default for FrenoEstrategia {
    fn default() -> Self {
        Self::new()
    }
}

impl FrenoEstrategia {
    pub fn new() -> Self {
        Self {
            ville: VilleEProcess::new(ALFA_ESTRATEGIA).expect("α en (0, 1)"),
            rechazada: false,
        }
    }

    /// Escala de `X` con la geometría medida de las órdenes (en stops): una
    /// pérdida es −1 R y un TP `w/l` R. Sin geometría medida, la de RR 2,25
    /// sin fricción.
    pub fn escala(ganancia_en_stops: f64, perdida_en_stops: f64) -> f64 {
        let tp_r = if ganancia_en_stops.is_finite()
            && ganancia_en_stops > 0.0
            && perdida_en_stops.is_finite()
            && perdida_en_stops > 0.0
        {
            ganancia_en_stops / perdida_en_stops
        } else {
            2.25
        };
        MARGEN_ESCALA * (1.0 + VENTAJA_H0_ESTRATEGIA_R).max(tp_r - VENTAJA_H0_ESTRATEGIA_R)
    }

    /// Registra un cierre: `resultado_en_stops` es el PnL neto entre el
    /// riesgo al stop (`resultado_r` del núcleo) y `perdida_en_stops` = l
    /// pasa a R. Devuelve el valor del proceso e.
    pub fn registrar(
        &mut self,
        resultado_en_stops: f64,
        ganancia_en_stops: f64,
        perdida_en_stops: f64,
    ) -> f64 {
        if !resultado_en_stops.is_finite() {
            return self.ville.e_value;
        }
        let l = if perdida_en_stops.is_finite() && perdida_en_stops >= 1.0 {
            perdida_en_stops
        } else {
            1.0
        };
        let r = resultado_en_stops / l;
        let x = (VENTAJA_H0_ESTRATEGIA_R - r) / Self::escala(ganancia_en_stops, perdida_en_stops);
        let e = self.ville.update(x);
        if e >= self.ville.threshold {
            self.rechazada = true;
        }
        e
    }

    /// ¿Está echado el freno?
    #[inline]
    pub fn rechazada(&self) -> bool {
        self.rechazada
    }
}
