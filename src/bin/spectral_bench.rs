//! BANCO DEL ESPECTRO — mide, sobre un tape REAL de aggTrades, dos cosas:
//!
//! 1. Las cinco formas de fusionar el espectro temporal que han existido en
//!    el motor: cuánto peso recae en escalas que los datos todavía no han
//!    llenado, cuánto se parece cada `fused_score` a «el precio está por
//!    encima o por debajo del de arranque», y sobre todo cuánto PREDICE el
//!    retorno siguiente (coeficiente de información a tres horizontes).
//! 2. La habilidad fuera de muestra del pronóstico espectral
//!    (`quantum_arena::spectral_tape`) de volatilidad, volumen, intensidad,
//!    flujo de dinero y concentración de entes, frente a la persistencia y a
//!    la climatología, a varios horizontes.
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

/// Réplicas de las fusiones que han gobernado el motor, calculadas en la
/// MISMA pasada para poder decidir por evidencia y no por teoría.
struct Fusiones {
    ewma: [f64; 32],
    /// EWMA de |desviación| con la semilla histórica de 1e-7.
    dev_vol_semilla: [f64; 32],
    /// Suma del núcleo de |desviación| SIN semilla (D-742).
    raw_dev: [f64; 32],
    persistence: [f64; 32],
    prev_dev: [f64; 32],
    last_ts: u64,
    first_ts: u64,
    /// [0] 1/vol original · [1] CERT-M3-H01 informativo · [2] informativo ×
    /// observable · [3] 1/vol × observable · [4] uniforme sobre lo observable.
    fused: [f64; 5],
}

impl Fusiones {
    fn new() -> Self {
        Self {
            ewma: [0.0; 32],
            dev_vol_semilla: [0.0; 32],
            raw_dev: [0.0; 32],
            persistence: [0.0; 32],
            prev_dev: [0.0; 32],
            last_ts: 0,
            first_ts: 0,
            fused: [0.0; 5],
        }
    }

    fn update(&mut self, price: f64, ts: u64, updates: u64) {
        if self.last_ts == 0 {
            self.ewma = [price; 32];
            self.dev_vol_semilla = [1e-7; 32];
            self.last_ts = ts;
            self.first_ts = ts;
            return;
        }
        if ts <= self.last_ts {
            return;
        }
        let dt = (ts - self.last_ts) as f64;
        self.last_ts = ts;
        let elapsed = (ts - self.first_ts) as f64;
        let updates_f = updates.max(1) as f64;
        let mean_dt = (elapsed / updates_f).max(1e-9);

        let mut acc = [(0.0f64, 0.0f64); 5]; // (Σw, Σw·señal)
        for i in 0..32 {
            let tau = SPECTRUM_SCALES_MS[i];
            let alpha = 1.0 - (-dt / tau).exp();
            let prev = self.ewma[i];
            let dev = (price - prev) / prev;
            self.ewma[i] += alpha * (price - prev);

            self.dev_vol_semilla[i] += alpha * (dev.abs() - self.dev_vol_semilla[i]);
            self.raw_dev[i] = self.raw_dev[i] * (1.0 - alpha) + alpha * dev.abs();
            let mass = 1.0 - (-elapsed / tau).exp();
            let dev_vol_obs = if mass > 0.0 { self.raw_dev[i] / mass } else { 0.0 };

            let agree = (dev * self.prev_dev[i]).signum()
                * (if dev.abs() > 1e-12 && self.prev_dev[i].abs() > 1e-12 {
                    1.0
                } else {
                    0.0
                });
            self.persistence[i] += alpha * (agree - self.persistence[i]);
            self.prev_dev[i] = dev;

            // Señal con la vol sembrada (fusiones 0 y 1) y con la vol
            // observada sin semilla (fusiones 2, 3 y 4).
            let sig_semilla = if self.dev_vol_semilla[i] > 1e-12 {
                (dev / self.dev_vol_semilla[i]).clamp(-5.0, 5.0).tanh()
            } else {
                0.0
            };
            let sig_obs = if dev_vol_obs > 1e-12 {
                (dev / dev_vol_obs).clamp(-5.0, 5.0).tanh()
            } else {
                0.0
            };

            let resolution = 1.0 - (-tau / 1.0f64).exp();
            let observable = mass * resolution;
            let n_eff = (tau / mean_dt).min(updates_f).max(1.0);
            let info = (self.persistence[i].abs() - 1.0 / n_eff.sqrt()).max(0.0);

            let pesos = [
                if self.dev_vol_semilla[i] > 1e-12 {
                    1.0 / self.dev_vol_semilla[i]
                } else {
                    0.0
                },
                ((self.persistence[i] - 0.5) * 2.0).max(0.05),
                observable * info,
                if dev_vol_obs > 1e-12 {
                    observable / dev_vol_obs
                } else {
                    0.0
                },
                observable,
            ];
            let senales = [sig_semilla, sig_semilla, sig_obs, sig_obs, sig_obs];
            for f in 0..5 {
                acc[f].0 += pesos[f];
                acc[f].1 += pesos[f] * senales[f];
            }
        }
        for f in 0..5 {
            self.fused[f] = if acc[f].0 > 1e-12 {
                (acc[f].1 / acc[f].0).clamp(-1.0, 1.0)
            } else {
                0.0
            };
        }
    }

    /// Peso que la fusión 1/vol original da a las escalas más largas que el
    /// tiempo observado.
    fn peso_escalas_frias(&self, now: u64) -> f64 {
        let elapsed = (now.saturating_sub(self.first_ts)) as f64;
        let (mut all, mut cold) = (0.0, 0.0);
        for i in 0..32 {
            if self.dev_vol_semilla[i] > 1e-12 {
                let w = 1.0 / self.dev_vol_semilla[i];
                all += w;
                if SPECTRUM_SCALES_MS[i] > elapsed {
                    cold += w;
                }
            }
        }
        if all > 0.0 {
            cold / all
        } else {
            0.0
        }
    }
}

/// Correlación de Pearson acumulada.
#[derive(Default, Clone, Copy)]
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
        eprintln!(
            "❌ {} no es un tape REAL (cabecera TGMTICK1): este banco sólo mide datos reales",
            path
        );
        std::process::exit(2);
    }
    let rec = std::mem::size_of::<BinTick>();
    let n_total = (mmap.len() - 8) / rec;
    let n = n_total.min(max_ticks);
    let ticks: &[BinTick] =
        unsafe { std::slice::from_raw_parts(mmap.as_ptr().add(8) as *const BinTick, n_total) };
    println!("📼 {} — {} trades reales (se leen {})", path, n_total, n);

    // ── 1. Fusiones del espectro ───────────────────────────────────────────
    let mut spec = TemporalSpectrum::new();
    let mut fus = Fusiones::new();
    let mut updates = 0u64;
    let mut first_price = 0.0f64;
    let mut first_ts = 0u64;
    let mut next_probe = 0u64;
    let probe_ms = 60_000u64;
    let mut corr_boot: [Corr; 5] = Default::default();
    let mut cold_share_sum: f64 = 0.0;
    let mut cold_share_n: f64 = 0.0;
    // El IC de una sola ventana no decide nada: tres horizontes.
    let ic_hs: [u64; 3] = [60_000, 300_000, 1_800_000];
    let mut ic: [[Corr; 3]; 5] = Default::default();
    let mut pend: std::collections::VecDeque<(u64, [f64; 5], f64)> = Default::default();
    let mut pend_head = [0usize; 3];

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
        // binance_vision_sync: el lado del AGRESOR lleva qty + profundidad base.
        let buyer = t.bid_qty > t.ask_qty;
        let ts = t.timestamp;
        if first_ts == 0 {
            first_ts = ts;
            first_price = price;
        }

        updates += 1;
        spec.update(price, ts);
        fus.update(price, ts, updates);

        // Puntuación por horizonte: cada muestra se evalúa en los tres.
        for (hi, &h) in ic_hs.iter().enumerate() {
            while let Some(&(t0, scores, p0)) = pend.get(pend_head[hi]) {
                if t0 + h > ts {
                    break;
                }
                let fwd = (price / p0).ln();
                for (fi, sc) in scores.iter().enumerate() {
                    ic[fi][hi].add(*sc, fwd);
                }
                pend_head[hi] += 1;
            }
        }
        let done = pend_head.iter().copied().min().unwrap_or(0);
        if done > 0 {
            for _ in 0..done {
                pend.pop_front();
            }
            for h in pend_head.iter_mut() {
                *h -= done;
            }
        }

        if ts >= next_probe {
            next_probe = ts + probe_ms;
            if ts > first_ts {
                cold_share_sum += fus.peso_escalas_frias(ts);
                cold_share_n += 1.0;
            }
            let boot = (price / first_price).ln().signum();
            for f in 0..5 {
                corr_boot[f].add(fus.fused[f], boot);
            }
            pend.push_back((ts, fus.fused, price));
        }

        for f in forecasters.iter_mut() {
            f.before_trade(&tape, ts);
        }
        tape.on_trade(ts, price, qty, buyer);
        for f in forecasters.iter_mut() {
            f.after_trade(&tape, ts);
        }

        if i > 0 && i % 10_000_000 == 0 {
            println!("   … {} trades ({:.0}s)", i, t_start.elapsed().as_secs_f64());
        }
    }
    let days = (tape.last_ts().saturating_sub(first_ts)) as f64 / 86_400_000.0;
    // La fusión VIVA del motor corre en la misma pasada (misma entrada que la
    // réplica [2]); se lee para que el compilador no la elimine.
    let _ = spec.fused_score;

    println!();
    println!("═══ 1. FUSIONES DEL ESPECTRO ({:.1} días) ═══", days);
    println!(
        "   peso que la fusión 1/vol da a escalas MÁS LARGAS que lo observado: {:.1} %",
        100.0 * cold_share_sum / cold_share_n.max(1.0)
    );
    let nombres = [
        "1/vol original (paridad de riesgo)",
        "CERT-M3-H01 (contenido informativo)",
        "informativo × observable",
        "1/vol × observable (D-742)",
        "uniforme sobre lo observable",
    ];
    println!(
        "   {:<38} {:>9} {:>9} {:>9} {:>10}",
        "IC con el retorno siguiente a", "1 min", "5 min", "30 min", "corr(ini)"
    );
    for (i, nombre) in nombres.iter().enumerate() {
        println!(
            "   {:<38} {:>+9.4} {:>+9.4} {:>+9.4} {:>+10.3}",
            nombre,
            ic[i][0].r(),
            ic[i][1].r(),
            ic[i][2].r(),
            corr_boot[i].r()
        );
    }
    println!(
        "   n = {} muestras · error típico del IC ≈ ±{:.4} · corr(ini) = correlación con signo(precio − precio de arranque)",
        ic[0][1].n as u64,
        1.0 / ic[0][1].n.max(1.0).sqrt()
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
