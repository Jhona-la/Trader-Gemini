//! Contrato de publicación de la deformación Fisher-Rao en el
//! `StatefulEngine` (GLM 192, mandato §3.4). Contabilidad de ticks
//! (identidad aritmética con el 190): los pushes de retornos corren en
//! ticks 2..N (tick 1 = semilla); WARM=512 ⇒ push #512 = tick 513, y
//! durante ese tick tick_count=512 ≡ 0 (mod 64) — la PRIMERA frontera
//! %64 publica con la escala lenta recién convergida.
//!
//! Toda la numerología fue medida en un ESPEJO python del engine (mismo
//! LCG, feed multiplicativo, muestreo por fronteras) ANTES de escribir
//! este contrato. Ese espejo descartó una aserción y produjo un campo:
//! el muestreo INSTANTÁNEO de frontera diluye los picos transitorios
//! (σ×2: instantánea 0.41 vs pico real 0.71) y el nulo del feed
//! multiplicativo del motor es más ancho que la calibración aditiva del
//! 191 (pico-época máx 0.52 en 10 caminantes vs 0.372 aditivo) — por eso
//! existe `fr_deformacion_pico_epoca` y por eso la separación certificada
//! es σ×4 (0.98 medido) con σ×2 documentado en el medio.
use god_engine_core::stateful_engine::StatefulEngine;

/// LCG determinista (idéntico al de los contratos 189/190/191).
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

/// Corre el feed multiplicativo (precio *= 1 + ruido·σ, como el motor:
/// norm_return = p/p_last − 1) y devuelve (máx pico-época pre-shift,
/// máx pico-época post-shift, primera frontera con publicación).
fn correr(semilla: u64, n: usize, shift: Option<usize>, factor: f64) -> (f32, f32, i64) {
    let mut rng = Lcg(semilla);
    let mut e = StatefulEngine::new();
    let mut precio = 100.0f64;
    let mut ts = 1_000u64;
    let (mut max_pre, mut max_post) = (0.0f32, 0.0f32);
    let mut primera_frontera = -1i64;
    for i in 0..n {
        let sigma = match shift {
            Some(d) if i >= d => 0.004 * factor,
            _ => 0.004,
        };
        precio *= 1.0 + rng.gauss() * sigma;
        e.process_tick(precio, 1.0, ts);
        ts += 100;
        // El pico de época se lee JUSTO tras la frontera (el motor lo
        // reinicia ahí): el valor completo vive en el instante frontera.
        let es_frontera = i >= 1 && (i as u64) % 64 == 0 && e.fr_deformacion_pico_epoca > 0.0;
        if es_frontera {
            if primera_frontera < 0 {
                primera_frontera = i as i64;
            }
            let pico = e.fr_deformacion_pico_epoca;
            match shift {
                Some(d) if i >= d => max_post = max_post.max(pico),
                _ => max_pre = max_pre.max(pico),
            }
        }
    }
    (max_pre, max_post, primera_frontera)
}

#[test]
fn el_nulo_caminante_publica_picos_de_epoca_bajo_la_cota_medida() {
    // 10 caminantes LCG (las semillas del espejo): el PICO de época del
    // nulo del feed multiplicativo del motor queda bajo 0.60 (medido:
    // máx 0.5226 en la semilla 9 — MÁS ANCHO que la calibración aditiva
    // del 191, 0.372: el contrato certifica lo que el ENGINE hace, no
    // lo que el núcleo hacía en banca de pruebas).
    let mut maximo = 0.0f32;
    for semilla in [42u64, 1, 2, 3, 4, 5, 6, 7, 8, 9] {
        let (pre, _, primera) = correr(semilla, 1024, None, 1.0);
        assert!(primera > 0, "semilla {semilla}: hay publicaciones");
        assert!(
            pre < 0.60,
            "semilla {semilla}: pico-época del nulo {pre} < 0.60 (medido 0.5226)"
        );
        maximo = maximo.max(pre);
    }
    // Calibración viva: el nulo NO es cero (cola de estimación).
    assert!(maximo > 0.25, "máx nulo {maximo} > 0.25 (medido 0.52)");
}

#[test]
fn el_cambio_de_regimen_x4_publica_un_pico_sobre_la_cota() {
    // σ×4 desde el tick 768 (semilla 42, espejo): pico-época post
    // 0.978 — separación certificada contra el nulo (<0.60) con margen
    // real. σ×2 (pico 0.71 medido) queda documentado en el medio: la
    // separación honesta a esta cadencia es ×4.
    let (pre, post, _) = correr(42, 1024, Some(768), 4.0);
    assert!(
        pre < 0.60,
        "tramo estacionario bajo la cota del nulo: {pre} < 0.60"
    );
    assert!(
        post > 0.90,
        "σ×4 deforma: pico-época publicado {post} > 0.90 (medido 0.978)"
    );
}

#[test]
fn warmup_sin_publicacion_hasta_la_escala_lenta_convergida() {
    // 512 ticks: el push #511 cae en el tick 512 — NINGUNA frontera
    // publica (la escala lenta aún no cubre su período; la lección
    // medida del 191: publicar antes es medir la convergencia propia).
    let (pre, post, primera) = correr(42, 512, None, 1.0);
    assert_eq!(primera, -1, "512 ticks: sin publicaciones ({pre}/{post})");
    assert!(pre == 0.0 && post == 0.0);
    // El tick 513 completa el warmup Y es frontera %64: primera
    // publicación inmediata (espejo: tick 513 exacto).
    let (_, _, primera513) = correr(4242, 513, None, 1.0);
    assert_eq!(primera513, 512, "primera frontera publicada: tick 513");
}

#[test]
fn conducta_invariante_el_observable_no_alimenta_ninguna_puerta() {
    // Dos engines idénticos (señal sinusoidal); los TRES campos FR
    // envenenados post-feed: la superficie de decisión es bit a bit
    // idéntica (patrón σz/λ_KS/β_RG/caputo/LPPLS — sexta aplicación).
    fn alimentar_seno(e: &mut StatefulEngine) {
        let mut ts = 1_000u64;
        for i in 0..300u32 {
            let price = 100.0 + (f64::from(i) * 0.33).sin() * 2.5;
            e.process_tick(price, 1.0, ts);
            ts += 100;
        }
    }
    let mut a = StatefulEngine::new();
    let mut b = StatefulEngine::new();
    alimentar_seno(&mut a);
    alimentar_seno(&mut b);
    // 300 ticks < warmup 512: ambos sin análisis aún — NaN explícito
    // (IEEE: NaN != NaN, jamás assert_eq sobre NaN — cazado por CI).
    assert!(a.fr_deformacion.is_nan() && b.fr_deformacion.is_nan());
    // Veneno post-feed: la decisión no los lee.
    b.fr_deformacion_ultimo = -777.0;
    b.fr_deformacion = 999.0;
    b.fr_deformacion_pico_epoca = 0.99;
    assert_eq!(
        a.get_universal_features(),
        b.get_universal_features(),
        "el vector universal 34D no lee la deformación FR"
    );
    assert_eq!(
        a.get_features(),
        b.get_features(),
        "las features 12D no leen la deformación FR"
    );
    assert_eq!(
        a.get_spectral_ml_features(),
        b.get_spectral_ml_features(),
        "las features espectrales ML no leen la deformación FR"
    );
}
