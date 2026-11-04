//! Contrato de publicación del operador D^α de Caputo del precio en el
//! `StatefulEngine` (GLM 188, mandato §11.7). Cuatro invariantes:
//! (1) CADENCIA — el crudo (`caputo_ultimo`) vive tick a tick, pero la
//!     instantánea (`caputo_d_alfa`) SOLO se publica en la frontera %64
//!     (primera frontera: tick 65, con 64 pushes acumulados) y REZAGA al
//!     crudo entre fronteras;
//! (2) SOLUCIÓN CONOCIDA — plano perfecto ⇒ D^α = Σw_k·0 = 0.0 EXACTO
//!     (bit a bit); rampa ↑ ⇒ > 0; rampa ↓ ⇒ < 0;
//! (3) ADUANA — un tick inválido (precio NaN) lo rechaza la validación de
//!     entrada ANTES de llegar al operador y no disturba su estado;
//! (4) CONDUCTA INVARIANTE — el observable no alimenta ninguna puerta:
//!     con los campos envenenados, la superficie de decisión (vector
//!     universal 34D, features 12D y features espectrales ML) es bit a
//!     bit idéntica. Patrón σz (175) / λ_KS (181) / β_RG (183-184).
use god_engine_core::stateful_engine::{FeatureInputError, StatefulEngine};

#[test]
fn cadencia_warmup_y_rezago_de_la_publicacion() {
    let mut e = StatefulEngine::new();
    let mut ts = 1_000u64;
    // 64 ticks de rampa ascendente: la primera frontera %64 cae en el
    // tick 65 (el bloque de telemetría corre con tick_count PREVIO — el
    // contador incrementa al final del tick).
    for i in 0..64u32 {
        e.process_tick(100.0 + f64::from(i) * 0.25, 1.0, ts);
        ts += 10;
    }
    assert!(
        e.caputo_d_alfa.is_nan(),
        "sin frontera %64 aún (tick 64): la publicación espera al tick 65"
    );
    assert!(
        e.caputo_ultimo.is_finite() && e.caputo_ultimo > 0.0,
        "el crudo vive tick a tick (rampa ↑ ⇒ D^α > 0): {}",
        e.caputo_ultimo
    );
    // Tick 65: primera frontera — publica la instantánea del instante.
    e.process_tick(100.0 + 64.0 * 0.25, 1.0, ts);
    assert!(e.caputo_d_alfa.is_finite(), "primera frontera publica");
    assert_eq!(
        e.caputo_d_alfa, e.caputo_ultimo as f32,
        "la instantánea ES el crudo de la frontera (sin promedio)"
    );
    // REZAGO + MEMORIA: plano perfecto hasta el tick 100 (35 planicies).
    // El crudo NO cae a 0: los diffs de la rampa más viejos siguen DENTRO
    // del ring (W=64) pesando w_k·0.25 — la cola que funde plazos del
    // mandato §11.7. Sólo queda memoria RESIDUAL (<< la rampa plena),
    // mientras la instantánea conserva el valor de frontera completo.
    for _ in 65..100u32 {
        e.process_tick(100.0 + 64.0 * 0.25, 1.0, ts);
        ts += 10;
    }
    assert!(
        e.caputo_ultimo > 0.0 && e.caputo_ultimo < 0.1,
        "35 planicies dejan sólo la memoria residual de la rampa: {}",
        e.caputo_ultimo
    );
    assert!(
        e.caputo_d_alfa > 0.1 && e.caputo_d_alfa > e.caputo_ultimo as f32,
        "la instantánea REZAGA: conserva la frontera del tick 65 ({}) sobre el crudo residual",
        e.caputo_d_alfa
    );
    // 64 planicies EXACTAS (ticks 66..129) expulsan la rampa COMPLETA del
    // ring: en la MISMA frontera %64 del tick 129, crudo e instantánea
    // caen a 0.0 EXACTO — la memoria se apaga sólo cuando TODO el ring
    // es plano (identidad D^α = Σ w_k · 0).
    for _ in 100..129u32 {
        e.process_tick(100.0 + 64.0 * 0.25, 1.0, ts);
        ts += 10;
    }
    assert_eq!(e.caputo_ultimo, 0.0, "ring 100% plano: crudo = 0.0 exacto");
    assert_eq!(
        e.caputo_d_alfa, 0.0,
        "la frontera del tick 129 publica el 0.0 exacto"
    );
}

#[test]
fn plano_es_cero_exacto_y_las_rampas_tienen_signo() {
    // Plano perfecto: TODAS las diferencias son 0.0 ⇒ D^α = 0.0 EXACTO —
    // el observable no fabrica ruido donde no hay movimiento.
    let mut plano = StatefulEngine::new();
    for i in 0..65u32 {
        plano.process_tick(100.0, 1.0, 1_000 + u64::from(i) * 10);
    }
    assert_eq!(
        plano.caputo_d_alfa, 0.0,
        "plano: D^α = Σ w_k · 0 = 0.0 bit a bit"
    );
    // Rampa descendente: memoria con signo — la derivada fraccional de una
    // caída constante es negativa (impulsoUnitario de 187 generalizado).
    let mut bajista = StatefulEngine::new();
    for i in 0..65u32 {
        bajista.process_tick(100.0 - f64::from(i) * 0.5, 1.0, 1_000 + u64::from(i) * 10);
    }
    assert!(
        bajista.caputo_d_alfa.is_finite() && bajista.caputo_d_alfa < 0.0,
        "rampa ↓ ⇒ D^α < 0 finito: {}",
        bajista.caputo_d_alfa
    );
}

#[test]
fn tick_invalido_lo_rechaza_la_aduana_y_el_operador_sobrevive() {
    let mut e = StatefulEngine::new();
    let mut ts = 1_000u64;
    for i in 0..10u32 {
        e.process_tick(100.0 + f64::from(i) * 0.25, 1.0, ts);
        ts += 10;
    }
    let ultimo_antes = e.caputo_ultimo;
    assert!(
        ultimo_antes.is_finite(),
        "rampa viva antes del tick inválido"
    );
    // Precio NaN: la ADUANA del engine (try_process_tick) lo rechaza ANTES
    // de que llegue al operador — la segunda capa es el None interno del
    // núcleo (descarta sin inyectar cero), certificada en el 187.
    assert_eq!(
        e.try_process_tick(f64::NAN, 1.0, ts),
        Err(FeatureInputError::InvalidPrice)
    );
    assert_eq!(
        e.caputo_ultimo, ultimo_antes,
        "el rechazo no disturba el estado del operador"
    );
    assert!(e.caputo_d_alfa.is_nan(), "sin frontera aún");
    // La rampa continúa y el operador sigue vivo.
    e.process_tick(100.0 + 10.0 * 0.25, 1.0, ts);
    assert!(
        e.caputo_ultimo.is_finite() && e.caputo_ultimo > 0.0,
        "el operador sobrevive al tick inválido"
    );
}

#[test]
fn conducta_invariante_el_observable_no_alimenta_ninguna_puerta() {
    // Dos engines alimentados IDÉNTICOS (señal sinusoidal con estructura
    // multiescala); después del feed se ENVENEN los campos del observable
    // del engine B: si cualquier puerta de decisión los leyera, la
    // superficie de salida divergiría. Es la prueba de regresión cero del
    // patrón σz/λ_KS/β_RG — observables listos, consumo = consejo.
    fn alimentar(e: &mut StatefulEngine) {
        let mut ts = 1_000u64;
        for i in 0..70u32 {
            let price = 100.0 + (f64::from(i) * 0.37).sin() * 2.5;
            e.process_tick(price, 1.0, ts);
            ts += 100;
        }
    }
    let mut a = StatefulEngine::new();
    let mut b = StatefulEngine::new();
    alimentar(&mut a);
    alimentar(&mut b);
    // El observable está VIVO y determinista: mismo feed ⇒ mismo valor.
    assert_eq!(a.caputo_d_alfa, b.caputo_d_alfa);
    assert_eq!(a.caputo_ultimo, b.caputo_ultimo);
    // Veneno post-feed (los campos son pub): la decisión no los lee.
    b.caputo_d_alfa = 999.0;
    b.caputo_ultimo = -777.0;
    assert_eq!(
        a.get_universal_features(),
        b.get_universal_features(),
        "el vector universal 34D no lee el observable D^α"
    );
    assert_eq!(
        a.get_features(),
        b.get_features(),
        "las features 12D no leen el observable D^α"
    );
    assert_eq!(
        a.get_spectral_ml_features(),
        b.get_spectral_ml_features(),
        "las features espectrales ML no leen el observable D^α"
    );
}
