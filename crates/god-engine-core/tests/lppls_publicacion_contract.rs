//! Contrato de publicación del detector LPPLS de Sornette en el
//! `StatefulEngine` (GLM 190, mandato §11.7). Contabilidad de ticks del
//! engine (medida en el 188): el bloque de telemetría corre desde el 2º
//! tick con `tick_count` PREVIO al incremento ⇒ los pushes del anillo
//! comienzan en el tick 2 y el anillo de 256 se LLENA en el tick 257,
//! que es exactamente la primera frontera %64 elegible (tick_count=256).
//! La calibración del detector es la del 189 (n=256, LCG determinista
//! idéntico): recuperación exacta del bin, evidencia monótona, nulo
//! caminante 0/16 en estable.
use god_engine_core::stateful_engine::StatefulEngine;

/// LCG determinista (idéntico al del contrato del 189): Knuth 64-bit,
/// uniforme por los 53 bits altos, gauss por Box–Muller con par fijo.
struct Lcg(u64);

impl Lcg {
    fn unif(&mut self) -> f64 {
        self.0 = self
            .0
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        (self.0 >> 11) as f64 / (1u64 << 53) as f64
    }
    fn gauss(&mut self) -> f64 {
        let u1 = self.unif();
        let u2 = self.unif().max(1e-300);
        (2.0 * std::f64::consts::PI * u1).cos() * (-2.0 * u2.ln()).sqrt()
    }
}

/// Serie LPPLS con t_c = 384.5 (unidades de tick global): el tick k
/// (k=1..=n_total) ve u = (t_c − k)/255. La VENTANA del engine (ticks
/// 2..257 con t_end=257) queda con u_end = 0.5 EXACTO — la misma
/// geometría calibrada del 189.
fn alimentar_lppls(e: &mut StatefulEngine, amplitud_c: f64, sigma: f64, n_total: usize) {
    let mut rng = Lcg(7);
    let t_c = 384.5f64;
    let mut ts = 1_000u64;
    for k in 1..=n_total {
        let u = (t_c - k as f64) / 255.0;
        let um = u.powf(0.4);
        let price = 100.0 - 20.0 * um + amplitud_c * um * (8.0 * u.ln()).cos();
        let price = if sigma > 0.0 {
            price + rng.gauss() * sigma
        } else {
            price
        };
        e.process_tick(price, 1.0, ts);
        ts += 100;
    }
}

/// Caminante aleatorio determinista (el null real de un precio).
fn alimentar_caminante(e: &mut StatefulEngine, semilla: u64, n_total: usize) {
    let mut rng = Lcg(semilla);
    let mut precio = 100.0f64;
    let mut ts = 1_000u64;
    for _ in 0..n_total {
        precio += rng.gauss() * 0.5;
        e.process_tick(precio, 1.0, ts);
        ts += 100;
    }
}

#[test]
fn sintetica_de_mercado_publica_tc_omega_y_evidencia_en_la_primera_frontera_elegible() {
    let mut e = StatefulEngine::new();
    // 257 ticks: anillo lleno (pushes en ticks 2..=257) y tick_count=256
    // durante el tick 257 ⇒ PRIMERA frontera %64 con análisis.
    alimentar_lppls(&mut e, 6.0, 0.5, 257);
    assert!(
        e.lppls_evidencia.is_finite(),
        "la primera frontera elegible publica (evidencia finita)"
    );
    assert!(
        e.lppls_evidencia > 0.30,
        "evidencia de la sintética de mercado: {} (calibrada > 0.30 en el 189)",
        e.lppls_evidencia
    );
    // t_c verdadero de la ventana: 0.5·255 = 127.5 ticks adelante.
    assert!(
        (e.lppls_tc_ticks_adelante - 127.5).abs() < 1e-3,
        "t_c recuperado: {} ticks (verdadero 127.5)",
        e.lppls_tc_ticks_adelante
    );
    assert_eq!(e.lppls_omega, 8.0, "ω recuperada exacta");
}

#[test]
fn warmup_sin_analisis_hasta_el_anillo_lleno() {
    let mut e = StatefulEngine::new();
    // 256 ticks: el anillo llega a 255 precios (el tick 1 es semilla sin
    // push) ⇒ NINGUNA frontera publica aunque crucen %64 (64, 128, 192,
    // 256 durante-block).
    alimentar_lppls(&mut e, 6.0, 0.5, 256);
    assert!(
        e.lppls_evidencia.is_nan(),
        "sin anillo lleno no hay análisis: NaN, jamás un ajuste parcial"
    );
    assert!(!e.lppls_estable);
    assert!(e.lppls_tc_ticks_adelante.is_nan());
    assert!(e.lppls_omega.is_nan());
}

#[test]
fn el_nulo_caminante_no_encadena_nunca_en_el_engine() {
    // Caminante determinista (semillas del 189): en CADA frontera
    // elegible la señal compuesta (evidencia >= 0.10 && estable) NO
    // dispara — estable 0/16 en la calibración del núcleo.
    for semilla in [42u64, 1, 9] {
        let mut e = StatefulEngine::new();
        alimentar_caminante(&mut e, semilla, 385);
        // Dos fronteras elegibles (ticks 257 y 321); el estado final
        // refleja la última. La señal compuesta no dispara.
        let dispara = e.lppls_evidencia >= 0.10 && e.lppls_estable;
        assert!(
            !dispara,
            "semilla {semilla}: la señal compuesta no dispara en ruido (0/16 calibrado)"
        );
        assert!(
            !e.lppls_estable,
            "semilla {semilla}: el caminante no encadena"
        );
    }
}

#[test]
fn conducta_invariante_el_observable_no_alimenta_ninguna_puerta() {
    // Dos engines alimentados IDÉNTICOS; los campos LPPLS del B se
    // envenenan post-feed: si cualquier puerta de decisión los leyera,
    // la superficie de salida divergiría (patrón σz/λ_KS/β_RG/caputo).
    fn alimentar(e: &mut StatefulEngine) {
        let mut ts = 1_000u64;
        for i in 0..300u32 {
            let price = 100.0 + (f64::from(i) * 0.31).sin() * 3.0;
            e.process_tick(price, 1.0, ts);
            ts += 100;
        }
    }
    let mut a = StatefulEngine::new();
    let mut b = StatefulEngine::new();
    alimentar(&mut a);
    alimentar(&mut b);
    // El observable está VIVO y determinista en ambas (análisis en las
    // fronteras 257 y... 300 ticks ⇒ fronteras elegibles 257; 288 no es
    // frontera con lleno... tick 257 única elegible ≤300).
    assert_eq!(a.lppls_evidencia, b.lppls_evidencia);
    // Veneno post-feed: la decisión no los lee.
    b.lppls_tc_ticks_adelante = 999.0;
    b.lppls_omega = -777.0;
    b.lppls_evidencia = 0.99;
    b.lppls_estable = true;
    assert_eq!(
        a.get_universal_features(),
        b.get_universal_features(),
        "el vector universal 34D no lee el observable LPPLS"
    );
    assert_eq!(
        a.get_features(),
        b.get_features(),
        "las features 12D no leen el observable LPPLS"
    );
    assert_eq!(
        a.get_spectral_ml_features(),
        b.get_spectral_ml_features(),
        "las features espectrales ML no leen el observable LPPLS"
    );
}
