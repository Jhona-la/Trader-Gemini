//! BANCO DEL ESPECTRO — mide, sobre un tape REAL de aggTrades, dos cosas:
//!
//! 1. La salud del `TemporalSpectrum` que gobierna el motor continuo: qué
//!    parte del peso de su fusión recae en escalas que los datos todavía no
//!    han llenado, cuánto se parece `fused_score` a «el precio está por
//!    encima o por debajo del de arranque», y cuánto predice el retorno
//!    siguiente (coeficiente de información).
//! 2. La habilidad fuera de muestra del pronóstico espectral
//!    (`quantum_arena::spectral_tape`) de volatilidad, volumen, intensidad,
//!    flujo de dinero y concentración de entes, frente a la persistencia y
//!    a la climatología, a varios horizontes.
//!
//! Sólo lee el fichero de ticks: no toca credenciales, red ni estado.
//!
//! Uso: spectral_bench <fichero TGMTICK1> [max_ticks]

use quantum_arena::spectral_tape::{ForecastTarget, HorizonForecaster, SpectralTape};
use quantum_arena::temporal_spectrum::{TemporalSpectrum, SPECTRUM_SCALES_MS};
use std::fs::File;

#[repr(C)]
#[derive(Clone, Copy)]
struct BinTick {
    timestamp: u64,
    bid_price: f64,
    ask_price: f64,
    bid_qty: f64,
    ask_qty: f64,
}

/// Réplica literal de la fusión del espectro ANTERIOR a D-742 (semilla 1e-7
/// en la vol de desviación y peso 1/vol sin masa ni resolución), para medir
/// en la misma pasada el antes y el después.
struct LegacyFusion {
    ewma: [f64; 32],
    dev_vol: [f64; 32],
    last_ts: u64,
    fused: f64,
}

impl LegacyFusion {
    fn new() -> Self {
        Self { ewma: [0.0; 32], dev_vol: [0.0; 32], last_ts: 0, fused: 0.0 }
    }
    fn update(&mut self, price: f64, ts: u64) {
        if self.last_ts == 0 {
            self.ewma = [price; 32];
            self.dev_vol = [1e-7; 32];
            self.last_ts = ts;
            return;
        }
        if ts <= self.last_ts {
            return;
        }
        let dt = (ts - self.last_ts) as f64;
        self.last_ts = ts;
        let (mut ws, mut wss) = (0.0, 0.0);
        for i in 0..32 {
            let alpha = 1.0 - (-dt / SPECTRUM_SCALES_MS[i]).exp();
            let prev = self.ewma[i];
            let dev = (price - prev) / prev;
            self.ewma[i] += alpha * (price - prev);
            self.dev_vol[i] += alpha * (dev.abs() - self.dev_vol[i]);
            let z = if self.dev_vol[i] > 1e-12 { dev / self.dev_vol[i] } else { 0.0 };
            let sig = z.clamp(-5.0, 5.0).tanh();
            let w = if self.dev_vol[i] > 1e-12 { 1.0 / self.dev_vol[i] } else { 0.0 };
            ws += w;
            wss += w * sig;
        }
        self.fused = if ws > 0.0 { (wss / ws).clamp(-1.0, 1.0) } else { 0.0 };
    }
}

/// Réplica literal de la fusión CERT-M3-H01 (peso por contenido informativo
/// `max((persistence − 0,5)·2, 0,05)` sobre la vol de desviación sembrada),
/// que es la que vivía en origin/main antes de esta rama.
struct CertFusion {
    ewma: [f64; 32],
    dev_vol: [f64; 32],
    persistence: [f64; 32],
    prev_dev: [f64; 32],
    last_ts: u64,
    fused: f64,
}

impl CertFusion {
    fn new() -> Self {
        Self {
            ewma: [0.0; 32],
            dev_vol: [0.0; 32],
            persistence: [0.0; 32],
            prev_dev: [0.0; 32],
            last_ts: 0,
            fused: 0.0,
        }
    }
    fn update(&mut self, price: f64, ts: u64) {
        if self.last_ts == 0 {
            self.ewma = [price; 32];
            self.dev_vol = [1e-7; 32];
            self.last_ts = ts;
            return;
        }
        if ts <= self.last_ts {
            return;
        }
        let dt = (ts - self.last_ts) as f64;
        self.last_ts = ts;
        let (mut ws, mut wss) = (0.0, 0.0);
        for i in 0..32 {
            let alpha = 1.0 - (-dt / SPECTRUM_SCALES_MS[i]).exp();
            let prev = self.ewma[i];
            let dev = (price - prev) / prev;
            self.ewma[i] += alpha * (price - prev);
            self.dev_vol[i] += alpha * (dev.abs() - self.dev_vol[i]);
            let z = if self.dev_vol[i] > 1e-12 { dev / self.dev_vol[i] } else { 0.0 };
            let sig = z.clamp(-5.0, 5.0).tanh();
            let agree = (dev * self.prev_dev[i]).signum()
                * (if dev.abs() > 1e-12 && self.prev_dev[i].abs() > 1e-12 { 1.0 } else { 0.0 });
            self.persistence[i] += alpha * (agree - self.persistence[i]);
            self.prev_dev[i] = dev;
            let w = ((self.persistence[i] - 0.5) * 2.0).max(0.05);
            ws += w;
            wss += w * sig;
        }
        self.fused = if ws > 1e-12 {
            (wss / ws).clamp(-1.0, 1.0)
        } else {
            (self.fused * 0.0) + 0.0
        };
    }
}

/// Correlación de Pearson acumulada.
#[derive(Default)]
struct Corr {
    n: f64,
    sx: f64,
    sy: f64,
    sxx: f64,
    syy: f64,
    sxy: f64,
}

impl Corr {
    fn add(&mut self, x: f64, y: f64) {
        if !x.is_finite() || !y.is_finite() {
            return;
        }
        self.n += 1.0;
        self.sx += x;
        self.sy += y;
        self.sxx += x * x;
        self.syy += y * y;
        self.sxy += x * y;
    }
    fn r(&self) -> f64 {
        if self.n < 3.0 {
            return f64::NAN;
        }
        let cov = self.sxy / self.n - (self.sx / self.n) * (self.sy / self.n);
        let vx = self.sxx / self.n - (self.sx / self.n).powi(2);
        let vy = self.syy / self.n - (self.sy / self.n).powi(2);
        if vx <= 0.0 || vy <= 0.0 {
            return f64::NAN;
        }
        cov / (vx * vy).sqrt()
    }
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    if args.len() < 2 {
        eprintln!("uso: spectral_bench <fichero TGMTICK1> [max_ticks]");
        std::process::exit(1);
    }
    let path = &args[1];
    let max_ticks: usize = args
        .get(2)
        .and_then(|s| s.parse().ok())
        .unwrap_or(usize::MAX);

    let file = File::open(path).expect("no se pudo abrir el fichero de ticks");
    let mmap = unsafe { memmap2::MmapOptions::new().map(&file).expect("mmap") };
    if mmap.len() < 8 || &mmap[..8] != b"TGMTICK1" {
        eprintln!("❌ {} no es un tape REAL (cabecera TGMTICK1): este banco sólo mide datos reales", path);
        std::process::exit(2);
    }
    let rec = std::mem::size_of::<BinTick>();
    let n_total = (mmap.len() - 8) / rec;
    let n = n_total.min(max_ticks);
    let ticks: &[BinTick] =
        unsafe { std::slice::from_raw_parts(mmap.as_ptr().add(8) as *const BinTick, n_total) };
    println!("📼 {} — {} trades reales (se leen {})", path, n_total, n);

    // ── 1. Salud del espectro temporal del motor ───────────────────────────
    let mut spec = TemporalSpectrum::new();
    let mut first_price = 0.0f64;
    let mut first_ts = 0u64;
    let mut next_probe = 0u64;
    let probe_ms = 60_000u64;
    let mut corr_boot = Corr::default();
    let mut cold_share_sum = 0.0;
    let mut cold_share_n = 0.0;
    // Coeficiente de información a 5 min: fused_score(t) frente a ln(P(t+5m)/P(t)).
    let ic_h = 300_000u64;
    let mut ic_pending: std::collections::VecDeque<(u64, f64, f64)> = Default::default();
    let mut ic = Corr::default();
    let mut legacy = LegacyFusion::new();
    let mut corr_boot_legacy = Corr::default();
    let mut ic_legacy = Corr::default();
    let mut ic_pending_legacy: std::collections::VecDeque<(u64, f64, f64)> = Default::default();
    let mut cert = CertFusion::new();
    let mut corr_boot_cert = Corr::default();
    let mut ic_cert = Corr::default();
    let mut ic_pending_cert: std::collections::VecDeque<(u64, f64, f64)> = Default::default();

    // ── 2. Pronóstico espectral ────────────────────────────────────────────
    let mut tape = SpectralTape::with_band(16.0, 1.2e9);
    let horizons_idx = [16usize, 18, 19, 20, 21, 22];
    let mut forecasters: Vec<HorizonForecaster> = Vec::new();
    for &hi in &horizons_idx {
        for t in ForecastTarget::ALL {
            forecasters.push(HorizonForecaster::new(&tape, t, SPECTRUM_SCALES_MS[hi]));
        }
    }

    let t_start = std::time::Instant::now();
    for (i, t) in ticks.iter().take(n).enumerate() {
        if !(t.bid_price > 0.0 && t.ask_price > 0.0) {
            continue;
        }
        let price = (t.bid_price + t.ask_price) * 0.5;
        let qty = (t.bid_qty - t.ask_qty).abs();
        // binance_vision_sync: el lado del agresor lleva qty + base.
        let buyer = t.bid_qty > t.ask_qty;
        let ts = t.timestamp;
        if first_ts == 0 {
            first_ts = ts;
            first_price = price;
        }

        spec.update(price, ts);
        legacy.update(price, ts);
        cert.update(price, ts);
        while let Some(&(t0, s0, p0)) = ic_pending_cert.front() {
            if t0 + ic_h > ts {
                break;
            }
            ic_pending_cert.pop_front();
            ic_cert.add(s0, (price / p0).ln());
        }
        while let Some(&(t0, s0, p0)) = ic_pending_legacy.front() {
            if t0 + ic_h > ts {
                break;
            }
            ic_pending_legacy.pop_front();
            ic_legacy.add(s0, (price / p0).ln());
        }

        while let Some(&(t0, s0, p0)) = ic_pending.front() {
            if t0 + ic_h > ts {
                break;
            }
            ic_pending.pop_front();
            ic.add(s0, (price / p0).ln());
        }

        if ts >= next_probe {
            next_probe = ts + probe_ms;
            let elapsed = (ts - first_ts) as f64;
            // Peso que la fusión ANTERIOR daba a cada escala (1/vol con semilla).
            let mut w_all = 0.0;
            let mut w_cold = 0.0;
            for (i, dv) in legacy.dev_vol.iter().enumerate() {
                let s_tau = SPECTRUM_SCALES_MS[i];
                if *dv > 1e-12 {
                    let w = 1.0 / dv;
                    w_all += w;
                    if s_tau > elapsed {
                        w_cold += w;
                    }
                }
            }
            if w_all > 0.0 && elapsed > 0.0 {
                cold_share_sum += w_cold / w_all;
                cold_share_n += 1.0;
            }
            corr_boot.add(spec.fused_score, (price / first_price).ln().signum());
            ic_pending.push_back((ts, spec.fused_score, price));
            corr_boot_legacy.add(legacy.fused, (price / first_price).ln().signum());
            ic_pending_legacy.push_back((ts, legacy.fused, price));
            corr_boot_cert.add(cert.fused, (price / first_price).ln().signum());
            ic_pending_cert.push_back((ts, cert.fused, price));
        }

        for f in forecasters.iter_mut() {
            f.before_trade(&tape, ts);
        }
        tape.on_trade(ts, price, qty, buyer);
        for f in forecasters.iter_mut() {
            f.after_trade(&tape, ts);
        }

        if i > 0 && i % 5_000_000 == 0 {
            println!("   … {} trades ({:.0}s)", i, t_start.elapsed().as_secs_f64());
        }
    }
    let days = (tape.last_ts().saturating_sub(first_ts)) as f64 / 86_400_000.0;

    println!();
    println!("═══ 1. ESPECTRO TEMPORAL DEL MOTOR ({:.1} días) ═══", days);
    println!(
        "   peso medio que la fusión anterior a D-742 daba a escalas MÁS LARGAS que lo observado: {:.1} %",
        100.0 * cold_share_sum / cold_share_n.max(1.0)
    );
    println!("   {:<44} {:>12} {:>14}", "fusión", "corr(arranque)", "IC 5 min");
    println!(
        "   {:<44} {:>+12.3} {:>+14.4}",
        "1/vol original (paridad de riesgo)",
        corr_boot_legacy.r(),
        ic_legacy.r()
    );
    println!(
        "   {:<44} {:>+12.3} {:>+14.4}",
        "CERT-M3-H01 (contenido informativo)",
        corr_boot_cert.r(),
        ic_cert.r()
    );
    println!(
        "   {:<44} {:>+12.3} {:>+14.4}",
        "D-742 (informativo × observable)",
        corr_boot.r(),
        ic.r()
    );
    println!(
        "   corr(arranque) = correlación con signo(precio − precio de arranque); IC = correlación con el retorno de los 5 min siguientes (n = {})",
        ic.n
    );

    println!();
    println!("═══ 2. PRONÓSTICO ESPECTRAL fuera de muestra (prequential) ═══");
    println!(
        "   {:<34} {:>9} {:>9} {:>12} {:>12} {:>12}",
        "objetivo", "horizonte", "n", "vs persist.", "persist/clim", "modelo/clim"
    );
    for f in &forecasters {
        let h = f.horizon_ms;
        let hs = if h < 60_000.0 {
            format!("{:.1} s", h / 1000.0)
        } else if h < 3_600_000.0 {
            format!("{:.1} min", h / 60_000.0)
        } else {
            format!("{:.2} h", h / 3_600_000.0)
        };
        println!(
            "   {:<34} {:>9} {:>9} {:>+11.3}  {:>+11.3}  {:>+11.3}",
            f.target.nombre(),
            hs,
            f.score.n,
            f.score.skill_vs_persistence(),
            f.score.persistence_vs_climatology(),
            f.score.skill_vs_climatology()
        );
    }
    println!();
    println!(
        "   (vs persist. > 0 ⇒ el modelo bate al pronóstico ingenuo; persist/clim = cuánto explica ya la persistencia; tiempo {:.0}s)",
        t_start.elapsed().as_secs_f64()
    );
}
