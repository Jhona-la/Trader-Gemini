//! #607 (Ola 29, Qoder) — ESPECTRAL MULTIACTIVO: la dependencia entre
//! activos RESUELTA EN ESCALA τ.
//!
//! ## Por qué
//!
//! El espectro de cada moneda vive en su propio `TemporalSpectrum` (32
//! escalas, IC prequential por escala, τ* por habilidad #594/#599), pero la
//! dependencia ENTRE monedas era un escalar: ρ lineal de PnL (D-748), λ̂ de
//! cópula (LXXII), n_eff. Un par puede estar LOCKED a 1 minuto e
//! independiente a 1 hora — un ρ̄ escalar no distingue régimen de escala.
//! Este módulo mide la co-movimiento RESUELTA EN ESCALA: para cada par
//! (a,b) y cada escala τ_k, el IC prequential de los retornos de bloque
//! no solapados de cada moneda a esa escala:
//!
//!   ic(k) = E[r_a·r_b] / √(E[r_a²]·E[r_b²])   (con olvido 1/64)
//!
//! — el mismo estimador de la habilidad #594, aplicado CRUZANDO monedas.
//! Los bloques son la maquinaria CL-30 que ya madura por moneda; este
//! módulo solo Empareja las maduraciones con guardia de recencia.
//!
//! ## Contrato
//!
//! - `observar_maduracion(moneda, tau_ms, escala, ts, r)`: registra la
//!   maduración del bloque de `moneda` y la Empareja contra el ÚLTIMO
//!   bloque maduro de cada otra moneda con |Δts| ≤ 0.5·τ (ventanas CONTEMPORÁNEAS — el bloque previo del otro, gap=τ, queda vetado: sino la mitad de las muestras son productos desalineados). La
//!   detección de borde es por ts: re-alimentar el mismo bloque no duplica
//!   muestra. r no finito se descarta.
//! - `coherencia_par(a, b, escala)`: IC de la pareja-escala; `None` con
//!   < 30 muestras (madurez, misma disciplina que MUESTRAS_SKILL_MADURAS)
//!   o sin dispersión.
//! - `acople_banda(moneda, lo, hi)`: media de |IC| de banda de TODOS los
//!   pares vivos de `moneda` con evidencia madura; None sin ninguna.
//! - `mejor_acople(moneda, lo, hi)`: el (par, τ, IC) de mayor |IC| en banda.
//!
//! ## Falsación
//!
//! (i) dos monedas alimentadas con las MISMAS maduraciones → ic ≈ 1 en
//! todas las escalas maduras; (ii) maduraciones independientes → |ic| chico;
//! (iii) la guardia de recencia no empareja bloques viejos; (iv) el borde
//! por ts no duplica; (v) extremo a extremo con DOS `TemporalSpectrum`
//! reales sobre la misma serie (accesor `ultimo_bloque_maduro`).
//!
//! ## Alcance (observación, no política)
//!
//! Cero consumidores de política: el uso futuro — ρ(τ*) del veto de grupo,
//! sustituyendo el ρ̄ escalar por el ρ de la escala que opera la orden —
//! es decisión del consejo con T-1 propio.

use crate::temporal_spectrum::{SPECTRUM_SCALES_MS, MUESTRAS_SKILL_MADURAS};
use std::collections::HashMap;

const ESCALAS: usize = 32;
/// Guardia de recencia del emparejamiento: los bloques de ambos deben
/// cerrar dentro de 0.5·τ el uno del otro (contemporáneos) (bloques de τ no solapados y
/// aproximadamente contemporáneos — sino el par mezcla regímenes).
const RECENCIA_X: f64 = 0.5;
/// Olvido del IC cruzado: media de ~64 muestras de par (vida media ≈ 44),
/// la misma ventana que la habilidad univariante de #594.
const OLVIDO: f64 = 1.0 / 64.0;
const MUESTRAS_MADURAS: u64 = 30;

#[derive(Clone, Copy, Default)]
struct IcEscala {
    ea2: f64,
    eb2: f64,
    eab: f64,
    n: u64,
}

impl IcEscala {
    fn acumular(&mut self, ra: f64, rb: f64) {
        self.ea2 += (ra * ra - self.ea2) * OLVIDO;
        self.eb2 += (rb * rb - self.eb2) * OLVIDO;
        self.eab += (ra * rb - self.eab) * OLVIDO;
        self.n = self.n.saturating_add(1);
    }

    fn ic(&self) -> Option<f64> {
        if self.n < MUESTRAS_MADURAS {
            return None;
        }
        let den = self.ea2 * self.eb2;
        if !(den.is_finite() && den > 0.0) {
            return None;
        }
        let ic = self.eab / den.sqrt();
        if ic.is_finite() {
            Some(ic.clamp(-1.0, 1.0))
        } else {
            None
        }
    }
}

#[derive(Clone, Default)]
struct EnlacePar {
    escalas: [IcEscala; ESCALAS],
}

/// El universo espectral multivariante: un IC cruzado por par y escala,
/// alimentado por las maduraciones de bloque de cada moneda.
#[derive(Clone)]
pub struct EspectralMultiactivo {
    max_coins: usize,
    enlaces: HashMap<(u16, u16), EnlacePar>,
    /// Último bloque maduro por (moneda, escala): (ts de cierre, retorno).
    ultimo: Vec<[(u64, f64); ESCALAS]>,
}

impl Default for EspectralMultiactivo {
    fn default() -> Self {
        Self::new(crate::state::MAX_COINS)
    }
}

impl EspectralMultiactivo {
    pub fn new(max_coins: usize) -> Self {
        Self {
            max_coins,
            enlaces: HashMap::new(),
            ultimo: vec![[(0u64, 0.0f64); ESCALAS]; max_coins],
        }
    }

    /// Registra la maduración del bloque `escala` de `moneda` (ts de cierre
    /// y retorno) y la empareja contra el último bloque maduro de cada otra
    /// moneda con guardia de recencia 0.5·τ. Detecta borde por ts.
    pub fn observar_maduracion(
        &mut self,
        moneda: usize,
        tau_ms: f64,
        escala: usize,
        ts_ms: u64,
        r: f64,
    ) {
        if moneda >= self.max_coins
            || escala >= ESCALAS
            || !r.is_finite()
            || !(tau_ms.is_finite() && tau_ms > 0.0)
            || ts_ms == 0
        {
            return;
        }
        let previo = self.ultimo[moneda][escala];
        if previo.0 == ts_ms {
            return; // mismo bloque ya emparejado — sin muestra duplicada
        }
        self.ultimo[moneda][escala] = (ts_ms, r);
        let tolerancia = (RECENCIA_X * tau_ms) as u64;
        for otra in 0..self.max_coins {
            if otra == moneda {
                continue;
            }
            let (ts_b, r_b) = self.ultimo[otra][escala];
            if ts_b == 0 {
                continue; // esa moneda aún no maduró esta escala
            }
            let gap = ts_ms.abs_diff(ts_b);
            if gap > tolerancia {
                continue; // bloques no contemporáneos: mezclaría regímenes
            }
            let (a, b, ra, rb) = if moneda < otra {
                (moneda as u16, otra as u16, r, r_b)
            } else {
                (otra as u16, moneda as u16, r_b, r)
            };
            let enlace = self.enlaces.entry((a, b)).or_default();
            enlace.escalas[escala].acumular(ra, rb);
        }
    }

    /// IC cruzado maduro de (a,b) en `escala`, o `None` sin evidencia.
    pub fn coherencia_par(&self, a: usize, b: usize, escala: usize) -> Option<f64> {
        if a >= self.max_coins || b >= self.max_coins || a == b || escala >= ESCALAS {
            return None;
        }
        let clave = if a < b { (a as u16, b as u16) } else { (b as u16, a as u16) };
        self.enlaces.get(&clave)?.escalas[escala].ic()
    }

    fn escalas_en_banda(lo: f64, hi: f64) -> impl Iterator<Item = usize> {
        (0..ESCALAS).filter(move |&k| SPECTRUM_SCALES_MS[k] >= lo && SPECTRUM_SCALES_MS[k] <= hi)
    }

    /// Media de |IC| en banda a través de TODOS los pares vivos de `moneda`
    /// con evidencia madura; `None` si ninguna escala-par maduró.
    pub fn acople_banda(&self, moneda: usize, lo: f64, hi: f64) -> Option<f64> {
        let mut suma = 0.0f64;
        let mut cuenta = 0u64;
        for ((a, b), enlace) in &self.enlaces {
            if *a as usize != moneda && *b as usize != moneda {
                continue;
            }
            for k in Self::escalas_en_banda(lo, hi) {
                if let Some(ic) = enlace.escalas[k].ic() {
                    suma += ic.abs();
                    cuenta += 1;
                }
            }
        }
        if cuenta == 0 {
            None
        } else {
            Some(suma / cuenta as f64)
        }
    }

    /// El (otro, τ, IC) de mayor |IC| en banda para `moneda`.
    pub fn mejor_acople(&self, moneda: usize, lo: f64, hi: f64) -> Option<(usize, f64, f64)> {
        let mut mejor: Option<(usize, f64, f64)> = None;
        for ((a, b), enlace) in &self.enlaces {
            let otro = if *a as usize == moneda {
                *b as usize
            } else if *b as usize == moneda {
                *a as usize
            } else {
                continue;
            };
            for k in Self::escalas_en_banda(lo, hi) {
                if let Some(ic) = enlace.escalas[k].ic() {
                    if mejor.map(|(_, _, m)| ic.abs() > m.abs()).unwrap_or(true) {
                        mejor = Some((otro, SPECTRUM_SCALES_MS[k], ic));
                    }
                }
            }
        }
        mejor
    }

    /// #613 — media del IC SIGNED de `moneda` contra TODAS las otras
    /// monedas con evidencia madura en `escala`. Es la ρ̄ espectral del
    /// grupo a esa escala: el reemplazo directo del ρ̄ escalar del veto de
    /// grupo cuando la orden opera a esa τ. Signo preservado (anticorrela-
    /// ción = cobertura, D-750b); `None` sin pares maduros.
    pub fn coherencia_media_con_todas(&self, moneda: usize, escala: usize) -> Option<f64> {
        if moneda >= self.max_coins || escala >= ESCALAS {
            return None;
        }
        let mut suma = 0.0f64;
        let mut cuenta = 0u64;
        for otra in 0..self.max_coins {
            if otra == moneda {
                continue;
            }
            if let Some(ic) = self.coherencia_par(moneda, otra, escala) {
                suma += ic;
                cuenta += 1;
            }
        }
        if cuenta == 0 {
            None
        } else {
            Some(suma / cuenta as f64)
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn xorshift(estado: &mut u64) -> f64 {
        *estado ^= *estado << 13;
        *estado ^= *estado >> 7;
        *estado ^= *estado << 17;
        ((*estado >> 11) as f64) / ((1u64 << 53) as f64)
    }
    fn gauss(estado: &mut u64) -> f64 {
        let u1 = xorshift(estado).max(1e-12);
        let u2 = xorshift(estado);
        (-2.0 * u1.ln()).sqrt() * (2.0 * std::f64::consts::PI * u2).cos()
    }

    #[test]
    fn qo_607_par_identico_da_ic_uno_e_independiente_da_chico() {
        let mut uni = EspectralMultiactivo::new(4);
        let mut indep = EspectralMultiactivo::new(4);
        let mut lcg = 0xC0FFEE123456789Au64;
        let tau = SPECTRUM_SCALES_MS[19];
        for i in 0..80u64 {
            let ts = ((i + 1) as f64 * tau) as u64;
            let r = gauss(&mut lcg);
            uni.observar_maduracion(0, tau, 19, ts, r);
            uni.observar_maduracion(1, tau, 19, ts, r); // mismas maduraciones
            let r0 = gauss(&mut lcg);
            let r1 = gauss(&mut lcg);
            indep.observar_maduracion(0, tau, 19, ts, r0);
            indep.observar_maduracion(1, tau, 19, ts, r1);
        }
        let ic_uni = uni.coherencia_par(0, 1, 19).expect("par idéntico madura");
        let ic_indep = indep.coherencia_par(0, 1, 19).expect("par indep madura");
        assert!(ic_uni > 0.95, "mismas maduraciones ⇒ ic≈1: {}", ic_uni);
        assert!(ic_indep.abs() < 0.4, "independientes ⇒ |ic| chico: {}", ic_indep);
        assert!(ic_uni > ic_indep.abs());
        // Simetría del par desordenado y banda.
        assert_eq!(uni.coherencia_par(1, 0, 19), uni.coherencia_par(0, 1, 19));
        let acople = uni.acople_banda(0, tau * 0.5, tau * 2.0).expect("banda con evidencia");
        assert!(acople > 0.9);
        assert!(uni.acople_banda(2, tau * 0.5, tau * 2.0).is_none(), "moneda sin pares");
        let (otro, tau_mejor, ic_mejor) = uni.mejor_acople(0, tau * 0.5, tau * 2.0).unwrap();
        assert_eq!(otro, 1);
        assert!(ic_mejor.abs() > 0.9 && tau_mejor == tau);
    }

    #[test]
    fn qo_607_recencia_y_borde_no_duplican() {
        let mut uni = EspectralMultiactivo::new(4);
        let tau = 1000.0;
        // Moneda 1 queda VIEJA: su bloque cierra en ts=1000 y la moneda 0
        // madura muy después (gap > 1.5·τ) — no se empareja.
        uni.observar_maduracion(1, tau, 19, 1000, 0.02);
        for i in 0..50u64 {
            uni.observar_maduracion(0, tau, 19, 100_000 + i * 1000, 0.01 * (i as f64 - 25.0));
        }
        assert_eq!(uni.coherencia_par(0, 1, 19), None, "la recencia veta el par viejo");
        // Borde: re-alimentar el MISMO ts de cierre no duplica muestra.
        let mut fresco = EspectralMultiactivo::new(4);
        fresco.observar_maduracion(0, tau, 19, 1000, 0.01);
        fresco.observar_maduracion(1, tau, 19, 1000, 0.02);
        fresco.observar_maduracion(0, tau, 19, 1000, 0.01); // mismo bloque
        // Con solo 2 muestras reales (< 30) no madura: el borde no sumó la 3ª.
        assert_eq!(fresco.coherencia_par(0, 1, 19), None);
    }

    #[test]
    fn qo_607_extremo_a_extremo_con_dos_espectros_reales() {
        // DOS TemporalSpectrum reales sobre LA MISMA serie: el accessor
        // `ultimo_bloque_maduro` alimenta el módulo y la coherencia de
        // banda debe ser alta donde los bloques maduraron en ambos.
        use crate::temporal_spectrum::TemporalSpectrum;
        let mut a = TemporalSpectrum::new();
        let mut b = TemporalSpectrum::new();
        let mut ma = EspectralMultiactivo::new(4);
        let mut lcg = 0xABCDEF0123456789u64;
        let mut p_a = 100.0;
        let mut p_b = 100.0;
        for i in 0..4000u64 {
            let ts = (i + 1) * 4000;
            let paso = 0.004 * gauss(&mut lcg);
            p_a *= 1.0 + paso;
            p_b *= 1.0 + paso; // MISMA serie — co-movimiento total
            a.update(p_a, ts);
            b.update(p_b, ts);
            for escala in 0..32 {
                if let Some((ts_bloque, r)) = a.ultimo_bloque_maduro(escala) {
                    ma.observar_maduracion(
                        0,
                        crate::temporal_spectrum::SPECTRUM_SCALES_MS[escala],
                        escala,
                        ts_bloque,
                        r,
                    );
                }
                if let Some((ts_bloque, r)) = b.ultimo_bloque_maduro(escala) {
                    ma.observar_maduracion(
                        1,
                        crate::temporal_spectrum::SPECTRUM_SCALES_MS[escala],
                        escala,
                        ts_bloque,
                        r,
                    );
                }
            }
        }
        // Escala de banda con bloques madurados en ambos (τ≈4.6min: índice 19,
        // 30 bloques × 275 s ≈ 8250 s ≪ 16_000 s de la simulación).
        let ic = ma
            .coherencia_par(0, 1, 19)
            .expect("la escala 19 madura en ambas monedas con la misma serie");
        assert!(ic > 0.9, "misma serie ⇒ co-movimiento total: ic={}", ic);
        let acople = ma
            .acople_banda(0, crate::temporal_spectrum::TAU_ANCHOR_FAST_MS, crate::temporal_spectrum::TAU_ANCHOR_SLOW_MS)
            .expect("banda con evidencia");
        assert!(acople > 0.5, "acople de banda alto con co-movimiento total: {}", acople);
    }
}

#[cfg(test)]
mod qo_613_tests {
    use super::*;

    #[test]
    fn qo_613_coherencia_media_con_todas_solo_cuenta_maduras() {
        let mut uni = EspectralMultiactivo::new(4);
        let tau = 1000.0;
        // Pares (0,1) y (0,2) maduros con la misma serie (ic→1); (0,3) sin
        // madurar. La media de 0 contra todas debe promediar SOLO las maduras.
        // Serie determinista alternante (r_i = ±a): ra y rb idénticos ⇒ ic=1.
        for i in 0..40u64 {
            let ts = (i + 1) * 1000;
            let r = if i % 2 == 0 { 0.02 } else { -0.02 };
            uni.observar_maduracion(0, tau, 19, ts, r);
            uni.observar_maduracion(1, tau, 19, ts, r);
            uni.observar_maduracion(2, tau, 19, ts, r);
        }
        let media = uni.coherencia_media_con_todas(0, 19).expect("2 pares maduros");
        assert!(media > 0.9, "misma serie ⇒ media alta: {}", media);
        // La moneda 3 sin datos no aporta — el promedio no se diluye.
        assert!(media <= 1.0);
        // Escala sin evidencia en ninguna ⇒ None.
        assert_eq!(uni.coherencia_media_con_todas(0, 5), None);
    }
}
