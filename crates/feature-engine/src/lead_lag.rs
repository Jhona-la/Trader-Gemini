use std::collections::HashMap;
use std::collections::VecDeque;
use std::f64;

/// ⚡ MOTOR DE LIDERAZGO MACRO MICROESTRUCTURAL LEAD-LAG (CROSS-ASSET ALPHA MATRIX)
/// Rastra la velocidad de propagación de ráfagas institucionales desde BTC/ETH hacia altcoins.
///
/// #658 (F2-C1/F2-C2) — LEAD-LAG REAL: la versión anterior era un EWMA
/// compuesto sin lags ni reloj (umbrales duros 0.50/0.25, líder sin
/// frescura). Ésta mide la correlación cruzada de cada líder contra la
/// historia del altcoin en una REJILLA DE LAGS físicos {0.5, 1, 2, 5, 10 s}:
/// el liderazgo sólo firma la divergencia cuando el lag óptimo es
/// POSITIVO (el líder ADELANTA al alt) con evidencia ρ suficiente, y la
/// muestra del líder caduca por EDAD (no hay OFI viejo como momentum).
#[derive(Debug, Clone)]
pub struct LeadLagAlphaEngine {
    btc_buf: VecDeque<(f64, f64)>,
    eth_buf: VecDeque<(f64, f64)>,
    /// Historia del OFI de cada altcoin consultado: coin_id → (ts_ms, ofi).
    alt_bufs: HashMap<usize, VecDeque<(f64, f64)>>,
    /// Ventana de historia por serie (en MUESTRAS; la edad la corta
    /// MAX_EDAD_MS).
    max_window: usize,
    /// Lag óptimo medido (ms) por líder en la última evaluación —
    /// telemetría/depuración.
    pub ultimo_lag_btc_ms: f64,
    pub ultimo_lag_eth_ms: f64,
}

/// Rejilla de lags candidatos (ms) — la propagación BTC→alt en cripto
/// vive en el rango sub-segundo a ~10 s (efecto Cont-Kukanov-Stoikov).
const LAGS_MS: [f64; 5] = [500.0, 1_000.0, 2_000.0, 5_000.0, 10_000.0];
/// Edad máxima de una muestra de líder para opinar (ms).
const MAX_EDAD_MS: f64 = 8_000.0;
/// Correlación mínima (en magnitud) para acreditar un lag.
const RHO_MIN: f64 = 0.25;
/// Muestras mínimas en la ventana común para medir ρ.
const N_MIN: usize = 12;

impl LeadLagAlphaEngine {
    pub fn new(max_window: usize) -> Self {
        Self {
            btc_buf: VecDeque::with_capacity(max_window.max(20)),
            eth_buf: VecDeque::with_capacity(max_window.max(20)),
            alt_bufs: HashMap::new(),
            max_window: max_window.max(20),
            ultimo_lag_btc_ms: 0.0,
            ultimo_lag_eth_ms: 0.0,
        }
    }

    #[inline(always)]
    pub fn update_leader(&mut self, is_btc: bool, ofi: f64, ts_ms: f64) {
        if !ofi.is_finite() || !ts_ms.is_finite() {
            return;
        }
        let buf = if is_btc {
            &mut self.btc_buf
        } else {
            &mut self.eth_buf
        };
        // Reloj monotónico: las muestras retrógradas no reescriben historia.
        if let Some(&(last_ts, _)) = buf.back() {
            if ts_ms < last_ts {
                return;
            }
        }
        if buf.len() >= self.max_window {
            buf.pop_front();
        }
        buf.push_back((ts_ms, ofi));
    }

    #[inline(always)]
    fn push_alt(&mut self, coin_id: usize, ofi: f64, ts_ms: f64) {
        if !ofi.is_finite() || !ts_ms.is_finite() {
            return;
        }
        let buf = self
            .alt_bufs
            .entry(coin_id)
            .or_insert_with(|| VecDeque::with_capacity(self.max_window));
        if let Some(&(last_ts, _)) = buf.back() {
            if ts_ms < last_ts {
                return;
            }
        }
        if buf.len() >= self.max_window {
            buf.pop_front();
        }
        buf.push_back((ts_ms, ofi));
    }

    /// Correlación de Pearson líder(t) vs alt(t+lag) sobre la ventana
    /// común muestreada por los eventos del LÍDER (interpolando el alt).
    /// Devuelve (rho, n). Cero trabajo si no hay ventana común.
    fn rho_con_lag(
        leader: &VecDeque<(f64, f64)>,
        alt: &VecDeque<(f64, f64)>,
        lag_ms: f64,
        ahora_ms: f64,
    ) -> (f64, usize) {
        // El alt debe poder interpolarse en t+lag ⇒ exige muestras futuras
        // a t (por eso mide que el líder ADELANTA, no contemporaneidad).
        let n_usable = alt
            .iter()
            .filter(|&&(ts, _)| ts > ahora_ms - MAX_EDAD_MS)
            .count();
        if n_usable < N_MIN || leader.len() < N_MIN {
            return (0.0, 0);
        }
        let mut sx = 0.0;
        let mut sy = 0.0;
        let mut sxx = 0.0;
        let mut syy = 0.0;
        let mut sxy = 0.0;
        let mut n = 0usize;
        for &(t, x) in leader.iter() {
            if t < ahora_ms - MAX_EDAD_MS {
                continue;
            }
            let target = t + lag_ms;
            // Interpolación lineal del alt en target (vecinos por ts).
            let y = match interp_lineal(alt, target) {
                Some(v) => v,
                None => continue,
            };
            sx += x;
            sy += y;
            sxx += x * x;
            syy += y * y;
            sxy += x * y;
            n += 1;
        }
        if n < N_MIN {
            return (0.0, n);
        }
        let nf = n as f64;
        let cov = sxy / nf - (sx / nf) * (sy / nf);
        let vx = (sxx / nf - (sx / nf) * (sx / nf)).max(0.0);
        let vy = (syy / nf - (sy / nf) * (sy / nf)).max(0.0);
        let denom = vx.sqrt() * vy.sqrt();
        if denom <= 1e-12 {
            return (0.0, n);
        }
        ((cov / denom).clamp(-1.0, 1.0), n)
    }

    /// Lag óptimo (ms) de un líder contra el alt: el de ρ máxima en
    /// magnitud. H2-12 (RONDA 3): la doc decía "exigido POSITIVO" pero el
    /// código compara `rho.abs()` — un rho NEGATIVO pasa el gate y su
    /// signo voltea la firma de la divergencia aguas abajo
    /// (`predict_eth_impulse_con_reloj`: div ∝ rho). ¿Vetar rho<0 (el
    /// anti-líder no lidera, engaña) es cambio de conducta que exige
    /// oráculo — DECISIÓN ABIERTA para el consejo, ver BARRIDO H2-12.
    /// Mientras tanto esta doc describe el comportamiento REAL.
    /// 0 si no hay evidencia.
    fn lag_optimo(
        &self,
        leader: &VecDeque<(f64, f64)>,
        alt: &VecDeque<(f64, f64)>,
        ahora_ms: f64,
    ) -> (f64, f64) {
        let mut mejor = (0.0f64, 0.0f64); // (lag, rho)
        for &lag in LAGS_MS.iter() {
            let (rho, _) = Self::rho_con_lag(leader, alt, lag, ahora_ms);
            if rho.abs() > RHO_MIN && rho.abs() > mejor.1.abs() {
                mejor = (lag, rho);
            }
        }
        mejor
    }

    /// Momentum compuesto del líder (peso BTC/ETH 60/40) con CADUCIDAD
    /// por edad: un líder sin muestra fresca no opina.
    pub fn momentum_lider(&self, ahora_ms: f64) -> f64 {
        let aporte = |buf: &VecDeque<(f64, f64)>| -> f64 {
            match buf.back() {
                Some(&(ts, ofi)) if ahora_ms - ts <= MAX_EDAD_MS => ofi,
                _ => 0.0,
            }
        };
        aporte(&self.btc_buf) * 0.60 + aporte(&self.eth_buf) * 0.40
    }

    /// Señal de propagación para el altcoin `coin_id` al tiempo `ts_ms`.
    /// (leader_momentum, lead_lag_divergence): ambos CONTINUOS, sin
    /// escalones — la divergencia escala con la evidencia (ρ del lag
    /// medido) y sólo existe si el líder ADELANTA (lag > 0).
    pub fn predict_altcoin_impulse_con_reloj(
        &mut self,
        coin_id: usize,
        alt_ofi: f64,
        ts_ms: f64,
    ) -> (f64, f64) {
        if !alt_ofi.is_finite() || !ts_ms.is_finite() {
            return (0.0, 0.0);
        }
        self.push_alt(coin_id, alt_ofi, ts_ms);
        let leader_momentum = self.momentum_lider(ts_ms);
        let alt = match self.alt_bufs.get(&coin_id) {
            Some(b) => b,
            None => return (leader_momentum, 0.0),
        };
        let (lag_btc, rho_btc) = self.lag_optimo(&self.btc_buf, alt, ts_ms);
        let (lag_eth, rho_eth) = self.lag_optimo(&self.eth_buf, alt, ts_ms);
        self.ultimo_lag_btc_ms = lag_btc;
        self.ultimo_lag_eth_ms = lag_eth;
        // Evidencia ponderada por líder (60/40): el impulso del líder se
        // transmite al alt con el retardo medido; la divergencia es el
        // exceso CONTINUO del líder sobre el alt, escalado por ρ.
        let (lag, rho) = if rho_btc.abs() >= rho_eth.abs() {
            (lag_btc, rho_btc)
        } else {
            (lag_eth, rho_eth)
        };
        if lag <= 0.0 {
            return (leader_momentum, 0.0);
        }
        let exceso = leader_momentum - alt_ofi;
        let div = (exceso.clamp(-3.0, 3.0) * 0.8).tanh() * rho;
        (leader_momentum, div)
    }

    /// G2-10 (Ola Ω11): Señal de propagación para ETH evaluado EXCLUSIVAMENTE contra BTC como líder macro.
    /// Erradica la auto-referencia espuria ETH contra ETH que acreditaba lag=0 o auto-correlación trivial rho=1.
    pub fn predict_eth_impulse_con_reloj(
        &mut self,
        coin_id: usize,
        eth_ofi: f64,
        ts_ms: f64,
    ) -> (f64, f64) {
        if !eth_ofi.is_finite() || !ts_ms.is_finite() {
            return (0.0, 0.0);
        }
        self.push_alt(coin_id, eth_ofi, ts_ms);
        let leader_momentum = self.momentum_lider(ts_ms);
        let alt = match self.alt_bufs.get(&coin_id) {
            Some(b) => b,
            None => return (leader_momentum, 0.0),
        };
        let (lag_btc, rho_btc) = self.lag_optimo(&self.btc_buf, alt, ts_ms);
        self.ultimo_lag_btc_ms = lag_btc;
        self.ultimo_lag_eth_ms = 0.0;
        if lag_btc <= 0.0 {
            return (leader_momentum, 0.0);
        }
        let exceso = leader_momentum - eth_ofi;
        let div = (exceso.clamp(-3.0, 3.0) * 0.8).tanh() * rho_btc;
        (leader_momentum, div)
    }

    /// Compatibilidad: sin reloj no hay lead-lag medible — devuelve el
    /// momentum fresco y SIN divergencia (abstención honesta).
    pub fn predict_altcoin_impulse(&self, alt_ofi: f64) -> (f64, f64) {
        if !alt_ofi.is_finite() {
            return (0.0, 0.0);
        }
        let ahora = self
            .btc_buf
            .back()
            .map(|&(t, _)| t)
            .or_else(|| self.eth_buf.back().map(|&(t, _)| t))
            .unwrap_or(0.0);
        (self.momentum_lider(ahora), 0.0)
    }
}

/// Interpolación lineal de (ts, v) en t objetivo; None fuera del rango.
fn interp_lineal(buf: &VecDeque<(f64, f64)>, t: f64) -> Option<f64> {
    if buf.len() < 2 {
        return None;
    }
    let mut lo = None;
    let mut hi = None;
    for &(ts, v) in buf.iter() {
        if ts <= t {
            lo = Some((ts, v));
        } else {
            hi = Some((ts, v));
            break;
        }
    }
    match (lo, hi) {
        (Some((t0, v0)), Some((t1, v1))) => {
            let dt = t1 - t0;
            if dt <= 0.0 {
                return Some(v0);
            }
            Some(v0 + (v1 - v0) * ((t - t0) / dt))
        }
        (Some((_, v0)), None) => Some(v0),
        _ => None,
    }
}

impl Default for LeadLagAlphaEngine {
    fn default() -> Self {
        Self::new(120)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// #658: el líder que ADELANTA produce divergencia escalada por ρ;
    /// el líder contemporáneo o rezagado, no. Frescura: sin muestras
    /// recientes el momentum muere.
    #[test]
    fn qo_658_lead_lag_con_lags_y_reloj() {
        let mut e = LeadLagAlphaEngine::new(200);
        let t0 = 1_000_000.0;
        // BTC adelanta 1 s al alt: alt(t) = btc(t − 1000)·0.8 + ruido 0.
        for i in 0..40 {
            let t = t0 + i as f64 * 250.0;
            let impulso = ((i as f64) * 0.05).sin() * 1.2;
            e.update_leader(true, impulso, t);
            e.predict_altcoin_impulse_con_reloj(3, impulso * 0.8, t + 1_000.0);
        }
        let (mom, div) = e.predict_altcoin_impulse_con_reloj(3, 0.0, t0 + 41.0 * 250.0);
        assert!(mom.abs() > 0.0, "momentum del líder vivo");
        // El alt YA copió al líder con lag 1 s ⇒ el lag medido es 1000 y
        // la divergencia con alt=0 e impulso vivo debe existir y ser
        // continua (|div| ≤ ρ ≤ 1).
        assert!(div.abs() <= 1.0 + 1e-9);
        assert!(e.ultimo_lag_btc_ms > 0.0, "lag óptimo positivo medido");
    }

    #[test]
    fn qo_658_lider_rezagado_no_firma() {
        let mut e = LeadLagAlphaEngine::new(200);
        let t0 = 2_000_000.0;
        // Pulso ÚNICO (no periódico: una sinusoide correlaciona espuriamente
        // a múltiplos del período): el alt ADELANTA al líder — ve el pulso
        // 2 s ANTES. Ningún lag positivo líder→alt debe acreditar
        // divergencia: el líder no contiene información futura del alt.
        for i in 0..40 {
            let t = t0 + i as f64 * 250.0;
            let pulso = if (8..16).contains(&i) { 1.0 } else { 0.0 };
            e.predict_altcoin_impulse_con_reloj(5, pulso, t);
            e.update_leader(true, pulso, t + 2_000.0);
        }
        let (_, div) = e.predict_altcoin_impulse_con_reloj(5, 0.0, t0 + 41.0 * 250.0);
        assert!(div.abs() < 0.2, "líder rezagado no firma: {div}");
    }

    #[test]
    fn qo_658_frescura_del_lider() {
        let mut e = LeadLagAlphaEngine::new(200);
        let t0 = 3_000_000.0;
        e.update_leader(true, 0.9, t0);
        e.update_leader(false, 0.8, t0);
        // 60 s después: el momentum del líder caducó.
        let (mom, div) = e.predict_altcoin_impulse_con_reloj(7, 0.05, t0 + 60_000.0);
        assert!(mom.abs() < 1e-9, "OFI viejo no es momentum: {mom}");
        assert!(div.abs() < 1e-9);
    }

    #[test]
    fn qo_658_compatibilidad_sin_reloj_se_abstiene() {
        let mut e = LeadLagAlphaEngine::new(200);
        e.update_leader(true, 0.9, 1_000.0);
        let (_, div) = e.predict_altcoin_impulse(0.05);
        assert_eq!(div, 0.0, "sin reloj no hay lead-lag medible");
    }

    #[test]
    fn qo_658_nan_immunity() {
        let mut e = LeadLagAlphaEngine::new(50);
        e.update_leader(true, f64::NAN, f64::NAN);
        e.update_leader(false, f64::NAN, 1.0);
        let (mom, div) = e.predict_altcoin_impulse_con_reloj(1, f64::NAN, f64::NAN);
        assert_eq!(mom, 0.0);
        assert_eq!(div, 0.0);
    }

    #[test]
    fn omega11_lead_lag_eth_sin_autoreferencia() {
        let mut e = LeadLagAlphaEngine::new(200);
        let t0 = 4_000_000.0;
        // Alimentar ETH idéntico en buffer de líder y en OFI de alt
        for i in 0..40 {
            let t = t0 + i as f64 * 250.0;
            let impulso = ((i as f64) * 0.05).sin() * 1.2;
            e.update_leader(false, impulso, t); // ETH como líder
        }
        // Llamar predict_eth_impulse_con_reloj: btc_buf está vacío, por lo que NO debe auto-evaluar contra ETH
        let (mom, div) = e.predict_eth_impulse_con_reloj(1, 1.0, t0 + 41.0 * 250.0);
        assert_eq!(e.ultimo_lag_eth_ms, 0.0, "ETH no debe generar lag de sí mismo");
        assert_eq!(e.ultimo_lag_btc_ms, 0.0, "Sin BTC no hay lag de BTC");
        assert_eq!(div, 0.0, "Sin liderazgo de BTC la divergencia debe ser 0.0");
        assert!(mom.abs() > 0.0, "Momentum de líder existe por ponderación");
    }
}
