//! CL-86 (atr-1, revisión de los PR #43–#55): un hueco sin eventos más largo
//! que la ventana de 61 s es dato ausente para σ̂60
//! (`god_engine_core::sigma_minuto::SigmaMinuto`).
//!
//! Teoría: el estimador mide la varianza del log-retorno de 60 s a partir de
//! los eventos. Si durante G segundos no llega ningún evento (caída del
//! feed, o el salto entre pasadas del examen del demonio), no hay
//! observaciones de esos segundos. Un hueco no es información sobre la
//! varianza: la σ̂ que se usa justo después debe ser la de antes, y
//! acercarse a la verdad al llegar datos nuevos.
//!
//! Antes de CL-86 el hueco entraba como r60 = 0. Por eso:
//! - con el precio quieto (examen del demonio), σ̂ se hundía y no se
//!   recuperaba en decenas de minutos;
//! - tras una caída del feed (el precio sigue moviéndose), σ̂ caía justo
//!   después del hueco y sólo la compensaba más tarde el salto.
//!
//! El núcleo cierra la vela de σ̂60 en el primer evento con ≥ 60 s desde la
//! anterior, tras `observar`, y el ATR lee `sigma_60s_viva`. Las pruebas
//! reproducen ese orden.

use god_engine_core::sigma_minuto::SigmaMinuto;

/// LCG de 64 bits con Box-Muller.
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

/// El estimador con el cierre de vela del núcleo.
struct Nucleo {
    est: SigmaMinuto,
    inicio_vela_ms: Option<u64>,
}
impl Nucleo {
    fn new() -> Self {
        Self { est: SigmaMinuto::new(), inicio_vela_ms: None }
    }
    /// Un evento: observa y, si pasaron ≥ 60 s desde el inicio de la vela,
    /// la cierra. Devuelve σ̂² viva tras el evento.
    fn evento(&mut self, t_ms: u64, precio: f64) -> Option<f64> {
        self.est.observar(t_ms, precio);
        match self.inicio_vela_ms {
            None => self.inicio_vela_ms = Some(t_ms),
            Some(t0) if t_ms.saturating_sub(t0) >= 60_000 => {
                self.est.cerrar_vela();
                self.inicio_vela_ms = Some(t_ms);
            }
            _ => {}
        }
        self.est.varianza_60s_viva()
    }
}

const SIGMA1: f64 = 2e-4;
const SEMILLAS: u64 = 200;
const PREVIO_S: u64 = 120 * 60;

/// Un camino: 2 h de browniano a un evento por segundo, un hueco de `hueco`
/// segundos sin eventos y 20 min más. Con `quieto`, el precio no se mueve
/// durante el hueco; si no, el browniano sigue corriendo. Devuelve σ̂² viva
/// en el último evento antes del hueco y en los instantes `tras` (segundos
/// desde el primer evento posterior).
fn camino(semilla: u64, hueco: u64, quieto: bool, tras: &[u64]) -> (f64, Vec<f64>) {
    let mut rng = Lcg(0xC186_0000 ^ semilla);
    let mut n = Nucleo::new();
    let mut x = 100f64.ln();
    let mut antes = f64::NAN;
    for s in 0..PREVIO_S {
        x += SIGMA1 * rng.n();
        if let Some(v) = n.evento(s * 1000 + 500, x.exp()) {
            antes = v;
        }
    }
    for _ in 0..hueco {
        let paso = SIGMA1 * rng.n();
        if !quieto {
            x += paso;
        }
    }
    let inicio = PREVIO_S + hueco;
    let mut despues = vec![f64::NAN; tras.len()];
    for k in 0..(20 * 60) {
        x += SIGMA1 * rng.n();
        let v = n.evento((inicio + k) * 1000 + 500, x.exp());
        for (j, &t) in tras.iter().enumerate() {
            if t == k {
                despues[j] = v.unwrap_or(f64::NAN);
            }
        }
    }
    (antes, despues)
}

/// Verdad: con el precio quieto en el hueco (sin información nueva), σ̂²
/// justo después del hueco es la de antes. Esperado: media sobre 200
/// caminos de σ̂²(después)/σ̂²(antes) en [0,95; 1,05], para huecos de 600 s
/// y de 1 h.
#[test]
fn cl86_un_hueco_con_el_precio_quieto_no_hunde_sigma() {
    for hueco in [600u64, 3_600] {
        let mut suma = 0.0;
        for s in 0..SEMILLAS {
            let (antes, despues) = camino(s, hueco, true, &[0]);
            suma += despues[0] / antes;
        }
        let media = suma / SEMILLAS as f64;
        println!("hueco {hueco} s, precio quieto: σ̂² después / antes = {media:.4}");
        assert!(
            (0.95..=1.05).contains(&media),
            "hueco {hueco} s con el precio quieto: σ̂² cae a {media:.4} de la de antes"
        );
    }
}

/// Verdad: tras una caída del feed (el precio sigue moviéndose, sin
/// eventos), E[σ̂²] = 60·σ1² en todo instante posterior: antes del hueco el
/// estimador es insesgado, y el hueco no aporta datos. Esperado: media
/// sobre 200 caminos de σ̂²/(60·σ1²) en [0,85; 1,15] a 0, 30, 60, 120, 300 y
/// 600 s del primer evento posterior (error típico de la media ≈ 0,035),
/// para huecos de 600 s y de 1 h.
#[test]
fn cl86_tras_una_caida_del_feed_sigma_no_se_hunde() {
    let verdad = 60.0 * SIGMA1 * SIGMA1;
    let tras = [0u64, 30, 60, 120, 300, 600];
    for hueco in [600u64, 3_600] {
        let mut sumas = vec![0.0; tras.len()];
        for s in 0..SEMILLAS {
            let (_, despues) = camino(1_000 + s, hueco, false, &tras);
            for (j, v) in despues.iter().enumerate() {
                sumas[j] += v / verdad;
            }
        }
        let medias: Vec<f64> = sumas.iter().map(|x| x / SEMILLAS as f64).collect();
        println!("hueco {hueco} s, caída del feed: σ̂²/verdad a {tras:?} s = {medias:.3?}");
        for (j, m) in medias.iter().enumerate() {
            assert!(
                (0.85..=1.15).contains(m),
                "hueco {hueco} s: a {} s del regreso σ̂²/verdad = {m:.3}",
                tras[j]
            );
        }
    }
}
