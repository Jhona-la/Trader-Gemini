//! C-F5 · W-1 de la revisión del PR #60 (Línea C, CL-109): una reconexión
//! del WS no rehace la prueba de reversión de StatArb.
//!
//! `reset_engines` (cada conexión WS de futuros salvo la primera) recreaba la
//! prueba. Madurar exige media memoria de pares seguidos (8·τ_k): 2,4 min en
//! el nodo 17 y 41 h en el 22. Con el corte diario de Binance, el nodo 22 no
//! maduraba nunca y el 21 sólo el 58 % del tiempo.
//!
//! Ahora la reconexión corta sólo el EMPAREJAMIENTO (mid del futuro y marca
//! del spot previos), y la prueba conserva sus sumas. Los huecos ya decaen
//! las sumas y cortan la cadena de retardos.
//!
//! Fixture. Base OU con reversión rápida (t½ = 3 s, mucho menor que
//! 2τ_17 = 34 s), un par por segundo durante 10 min: el nodo 17 madura y
//! demuestra la reversión.

use god_engine_core::GodEngineCore;
use quantum_arena::temporal_spectrum::SPECTRUM_SCALES_MS;
use quantum_arena::GlobalArena;

struct Lcg(u64);
impl Lcg {
    fn u(&mut self) -> f64 {
        self.0 = self.0.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        ((self.0 >> 11) as f64 + 0.5) / (1u64 << 53) as f64
    }
    fn normal(&mut self) -> f64 {
        (-2.0 * self.u().ln()).sqrt() * (2.0 * std::f64::consts::PI * self.u()).cos()
    }
}

/// Núcleo con la prueba del nodo 17 de la moneda 1 madura y demostrada.
fn madura() -> GodEngineCore {
    let mut c = GodEngineCore::new(GlobalArena::build_in_own_stack(1_000.0));
    let mut azar = Lcg(9);
    let rho = (-std::f64::consts::LN_2 / 3.0).exp(); // t½ = 3 s con pasos de 1 s
    let mut x = 0.0f64;
    for s in 1..=600u64 {
        x = rho * x + (1.0 - rho * rho).sqrt() * 1e-4 * azar.normal();
        let futuro = 100.0 * (1.0 + x).max(0.5);
        c.statarb_ou_engines[1].observar_par(futuro, 100.0, s * 1_000);
    }
    c
}

/// (demostrada, masa del nodo 17, pares) de la moneda 1.
fn huella(c: &GodEngineCore) -> (Option<bool>, Option<f64>, u64) {
    let e = &c.statarb_ou_engines[1];
    let tau = SPECTRUM_SCALES_MS[17];
    (
        // Piso de la base de un tick de 0,01 a 100 USD (CL-111: sin piso no hay voto).
        e.veredicto(tau, 1e-4).map(|v| v.demostrada),
        e.prueba.as_ref().and_then(|p| p.estado_nodo(17)).map(|n| n.masa),
        e.prueba.as_ref().map_or(0, |p| p.pares()),
    )
}

/// Esperado: tras la reconexión la prueba sigue madura y demostrada, con la
/// misma masa y los mismos pares. Antes volvía a no tener ningún par.
#[test]
fn cl109_la_reconexion_no_rehace_la_prueba_de_reversion() {
    let mut c = madura();
    let antes = huella(&c);
    c.reset_engines();
    let despues = huella(&c);
    println!("observado: antes {antes:?}; tras la reconexión {despues:?}");
    assert_eq!(antes.0, Some(true), "precondición: el nodo 17 demuestra la reversión");
    assert_eq!(despues, antes);
}

/// Esperado: la reconexión corta el emparejamiento. El primer dato de spot
/// que llega después no se empareja con el mid del futuro de antes.
#[test]
fn cl109_la_reconexion_corta_el_emparejamiento() {
    let mut c = madura();
    let e = &mut c.statarb_ou_engines[1];
    // Dos eventos del futuro con el mismo dato de spot: deja mid y marca previos.
    e.observar_evento(101.0, 100.0, 700_000, 700_000);
    e.observar_evento(101.0, 100.0, 700_000, 700_500);
    let pares = huella(&c).2;
    c.reset_engines();
    // Llega un dato de spot nuevo con el primer evento tras la reconexión.
    c.statarb_ou_engines[1].observar_evento(102.0, 100.5, 800_000, 800_000);
    println!("observado: pares antes {pares}, después {}", huella(&c).2);
    assert_eq!(huella(&c).2, pares, "sin mid vigente tras la reconexión no hay par");
}
