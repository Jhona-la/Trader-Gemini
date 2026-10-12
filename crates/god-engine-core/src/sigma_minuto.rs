//! CL-70 (C-F4, LC-F6-01): σ de 60 s invariante a la densidad de eventos.
//!
//! El ATR vivo (`StatefulEngine::v_t`) es una EMA del rango alto-bajo de una
//! vela interna de 60 s, con trinquete dentro de la vela y suelo de 10 pb. El
//! rango de m muestras de un browniano depende de m: E[R_m]/E[R_∞] vale 0,50
//! con m = 1, ≈ 0,70 con m ≈ 4 y 0,875 con m = 30. El mismo mercado da stops
//! distintos en vivo, en el forense y en los replays (el defecto que D-615b
//! corrigió para el Hurst).
//!
//! Este estimador mide directamente la varianza del log-retorno de 60 s:
//! - Rejilla de 1 s con el último precio: el valor del segundo g es el
//!   log-precio del último evento con marca < (g + 1)·1000 ms.
//! - Por cada segundo cerrado, r60 = x(g) − x(g − 60), solapados.
//! - Varianza realizada con ponderación exponencial: dos sumas con el mismo
//!   decaimiento (1 − α) por minuto de datos, S = Σ r60² y N = número de
//!   r60, y σ̂² = S/N. Cada r60 pesa lo mismo esté en la vela que esté.
//! - CL-86 (atr-1): un hueco sin eventos más largo que la ventana (más de 61
//!   s) es DATO AUSENTE. No fabrica r60 nulos ni decae las sumas, y reinicia
//!   la rejilla: ningún r60 cruza el hueco, y σ̂ conserva su valor hasta
//!   tener 61 s nuevos. Antes el hueco entraba como r60 = 0 (G − 61 en el
//!   contador más los de la rejilla rellena con el último precio). La media
//!   TEMPORAL salía sin sesgo, porque el salto de después lo compensaba,
//!   pero la σ̂ que se usaba justo tras el hueco caía (0,44 de la real a
//!   600 s y 0,14 a 3 600 s), y sin salto (precio quieto, como entre
//!   pasadas del examen del demonio) se hundía al ~4 %. Los huecos de hasta
//!   61 s siguen el muestreo por último precio, que es insesgado
//!   (`cl70_browniano_misma_varianza_con_cualquier_densidad`).
//! - E[r60²] = σ²·60 s con cualquier densidad de eventos estacionaria: el
//!   intervalo entre los dos últimos eventos de los segundos g y g − 60 mide
//!   60 s en esperanza. Es el estimador de varianza realizada no centrado:
//!   una deriva μ suma μ²·60² (despreciable frente a σ²·60 a esta escala).
//!
//! En el calentamiento REST (klines de 1 min), `observar_cierre_kline` usa
//! el log-retorno entre cierres, que es un r60 de la misma cantidad.
//!
//! CL-71 lo conecta a `StatefulEngine::v_t` con la conversión explícita a
//! rango, √(8/π)·σ̂60·precio, más una envolvente rápida con el rango de la
//! vela abierta (ver `stateful_engine.rs`).

/// Peso de la EWMA por vela de 60 s (el 0,15 del ATR vivo).
pub const ALFA_POR_MINUTO: f64 = 0.15;

/// Segundos del retorno que se mide.
pub const SEGUNDOS: usize = 60;

/// σ de 60 s por varianza realizada de log-retornos solapados.
#[derive(Debug, Clone)]
pub struct SigmaMinuto {
    /// Log-precios de los últimos 61 segundos cerrados (anillo).
    rejilla: [f64; SEGUNDOS + 1],
    /// Segundos cerrados escritos en la rejilla (satura en 61).
    llenos: usize,
    /// Segundo abierto (marca / 1000) y su log-precio vigente.
    segundo_abierto: Option<u64>,
    x_abierto: f64,
    /// Acumuladores de la vela en curso.
    suma_r2: f64,
    n_r2: u64,
    /// Sumas exponenciales de r60² y del número de r60 (mismo decaimiento).
    suma_ewma: f64,
    n_ewma: f64,
    /// Último cierre de kline del calentamiento.
    cierre_kline_previo: Option<f64>,
}

impl Default for SigmaMinuto {
    fn default() -> Self {
        Self::new()
    }
}

impl SigmaMinuto {
    pub fn new() -> Self {
        Self {
            rejilla: [0.0; SEGUNDOS + 1],
            llenos: 0,
            segundo_abierto: None,
            x_abierto: 0.0,
            suma_r2: 0.0,
            n_r2: 0,
            suma_ewma: 0.0,
            n_ewma: 0.0,
            cierre_kline_previo: None,
        }
    }

    /// Vuelve al estado inicial.
    pub fn reset(&mut self) {
        *self = Self::new();
    }

    /// Cierra el segundo `g` con log-precio `x`: lo escribe en la rejilla y,
    /// si hay 60 segundos previos, acumula r60².
    fn cerrar_segundo(&mut self, g: u64, x: f64) {
        let i = (g % (SEGUNDOS as u64 + 1)) as usize;
        self.rejilla[i] = x;
        if self.llenos < SEGUNDOS + 1 {
            self.llenos += 1;
        }
        if self.llenos == SEGUNDOS + 1 {
            let j = ((g + 1) % (SEGUNDOS as u64 + 1)) as usize; // g − 60
            let r = x - self.rejilla[j];
            self.suma_r2 += r * r;
            self.n_r2 += 1;
        }
    }

    /// Registra un evento con precio positivo y finito en `ts_ms`. Ignora
    /// los precios no válidos y los eventos con marca anterior al segundo
    /// abierto (la rejilla sólo avanza).
    pub fn observar(&mut self, ts_ms: u64, precio: f64) {
        if !(precio.is_finite() && precio > 0.0) {
            return;
        }
        let s = ts_ms / 1000;
        let x = precio.ln();
        match self.segundo_abierto {
            None => {
                self.segundo_abierto = Some(s);
                self.x_abierto = x;
            }
            Some(abierto) if s < abierto => {}
            Some(abierto) if s == abierto => self.x_abierto = x,
            Some(abierto) => {
                let x_previo = self.x_abierto;
                let total = s - abierto;
                if total > SEGUNDOS as u64 + 1 {
                    // CL-86 (atr-1): hueco más largo que la ventana, dato
                    // ausente. Cierra el último segundo con datos y reinicia
                    // la rejilla, sin r60 sintéticos.
                    self.cerrar_segundo(abierto, x_previo);
                    self.llenos = 0;
                } else {
                    // Cierra el segundo abierto y rellena los segundos sin
                    // eventos con el último precio (muestreo por último
                    // precio, insesgado dentro de la ventana).
                    for g in abierto..s {
                        self.cerrar_segundo(g, x_previo);
                    }
                }
                self.segundo_abierto = Some(s);
                self.x_abierto = x;
            }
        }
    }

    /// Incorpora un bloque de `n` r60 con suma de cuadrados `suma`: las dos
    /// sumas decaen (1 − α)^(n/60) y suman el bloque.
    fn acumular(&mut self, suma: f64, n: f64) {
        let decaimiento = (1.0 - ALFA_POR_MINUTO).powf(n / SEGUNDOS as f64);
        self.suma_ewma = self.suma_ewma * decaimiento + suma;
        self.n_ewma = self.n_ewma * decaimiento + n;
    }

    /// Cierra la vela en curso: sus r60 entran en las sumas exponenciales.
    /// Sin r60 no hace nada.
    pub fn cerrar_vela(&mut self) {
        if self.n_r2 == 0 {
            return;
        }
        let (suma, n) = (self.suma_r2, self.n_r2 as f64);
        self.acumular(suma, n);
        self.suma_r2 = 0.0;
        self.n_r2 = 0;
    }

    /// Calentamiento con klines REST de 1 min: el log-retorno entre cierres
    /// consecutivos es un r60 y entra como un minuto de datos (60 muestras de
    /// ese r60², la misma masa por minuto que el camino vivo).
    pub fn observar_cierre_kline(&mut self, cierre: f64) {
        if !(cierre.is_finite() && cierre > 0.0) {
            return;
        }
        let x = cierre.ln();
        if let Some(previo) = self.cierre_kline_previo {
            let r = x - previo;
            let n = SEGUNDOS as f64;
            self.acumular(r * r * n, n);
        }
        self.cierre_kline_previo = Some(x);
    }

    /// Varianza del log-retorno de 60 s; None sin ningún r60.
    pub fn varianza_60s(&self) -> Option<f64> {
        (self.n_ewma > 0.0).then(|| self.suma_ewma / self.n_ewma)
    }

    /// σ del log-retorno de 60 s (fracción del precio).
    pub fn sigma_60s(&self) -> Option<f64> {
        self.varianza_60s().map(f64::sqrt)
    }

    /// Varianza de 60 s con la vela abierta incluida: (S + Σ r60² de la vela)
    /// / (N + n de la vela). Con datos estacionarios es la misma cantidad que
    /// `varianza_60s`, pero ve cada segundo cerrado sin esperar al cierre de
    /// la vela (CL-71: el ATR no se queda congelado un minuto ante un choque).
    pub fn varianza_60s_viva(&self) -> Option<f64> {
        let n = self.n_ewma + self.n_r2 as f64;
        (n > 0.0).then(|| (self.suma_ewma + self.suma_r2) / n)
    }

    /// σ de 60 s con la vela abierta incluida (fracción del precio).
    pub fn sigma_60s_viva(&self) -> Option<f64> {
        self.varianza_60s_viva().map(f64::sqrt)
    }
}
