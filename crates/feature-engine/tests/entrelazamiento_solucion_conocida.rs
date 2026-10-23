//! Contratos del entrelazamiento temporal neto (GLM 155) con SOLUCIÓN
//! CONOCIDA sintética — certifican la lógica en CI sin tape. El vehículo
//! de tape real es `backtest-engine/tests/mps_entrelazamiento_medicion.rs`
//! (#[ignore], investigación local).
use feature_engine::entrelazamiento::{
    EntrelazamientoError, ModoBinarizacion, entrelazamiento_neto, estado_empirico,
};

/// xorshift64* determinista para las series sintéticas.
struct Rng(u64);
impl Rng {
    fn next(&mut self) -> u64 {
        self.0 ^= self.0 >> 12;
        self.0 ^= self.0 << 25;
        self.0 ^= self.0 >> 27;
        self.0.wrapping_mul(0x2545_F491_4F6C_DD1D)
    }
    fn uniforme(&mut self) -> f64 {
        (self.next() >> 11) as f64 / (1u64 << 53) as f64
    }
}

const L: usize = 8;

#[test]
fn iid_por_asimetrica_que_sea_delta_cero() {
    // Bernoulli p=0.3 iid: los bits iid dan estado PRODUCTO (S = 0 exacto
    // en población) sin importar p — el nulo cancela el ruido de muestreo.
    let mut rng = Rng(0xDEAD_BEEF_CAFE_0001);
    let rets: Vec<f64> = (0..200_000)
        .map(|_| if rng.uniforme() < 0.3 { 1.0 } else { -1.0 })
        .collect();
    for modo in [ModoBinarizacion::Signo, ModoBinarizacion::Mediana] {
        let m = entrelazamiento_neto(&rets, L, modo).expect("iid medible");
        assert!(
            m.delta.abs() < 5e-3,
            "iid: |ΔS| {d} < 5e-3 ({modo:?})",
            d = m.delta
        );
        assert!(m.s < 0.05 && m.s_nulo < 0.05, "iid: S y S_nulo ~ ruido");
        assert_eq!(m.ventanas, rets.len() + 1 - L);
        assert_eq!(m.descartados, 0);
    }
}

#[test]
fn markov_persistente_tiene_entrelazamiento_positivo() {
    // P(↑|↑) = 0.9, P(↑|↓) = 0.1: memoria fuerte ⇒ la distribución de
    // ventanas NO factoriza ⇒ ΔS > 0 claramente distinto del nulo.
    let mut rng = Rng(0x1234_5678_9ABC_DEF0);
    let mut arriba = true;
    let mut rets = Vec::with_capacity(200_000);
    for _ in 0..200_000 {
        let p_arriba = if arriba { 0.9 } else { 0.1 };
        arriba = rng.uniforme() < p_arriba;
        rets.push(if arriba { 1.0 } else { -1.0 });
    }
    let m = entrelazamiento_neto(&rets, L, ModoBinarizacion::Signo).expect("markov medible");
    assert!(
        m.delta > 0.02,
        "markov persistente: ΔS {d} > 0.02 (S={s}, S_nulo={n})",
        d = m.delta,
        s = m.s,
        n = m.s_nulo
    );
}

#[test]
fn serie_alternada_recupera_ln2_exacto() {
    // Ancla GHZ a través del pipeline COMPLETO: la serie +1/−1 alternante
    // produce exactamente dos configuraciones complementarias (0101… y
    // 1010…) con cuenta igual ⇒ ψ es GHZ ⇒ S(corte medio) = ln 2.
    let rets: Vec<f64> = (0..100_000)
        .map(|i| if i % 2 == 0 { 1.0 } else { -1.0 })
        .collect();
    let (psi, _, _) = estado_empirico(&rets, L, ModoBinarizacion::Signo).expect("estado");
    let s = feature_engine::mps::entrelazamiento_corte(&psi, L, L / 2).expect("S");
    assert!(
        (s - std::f64::consts::LN_2).abs() < 1e-9,
        "GHZ: S {s} = ln2",
        s = s
    );
    let m = entrelazamiento_neto(&rets, L, ModoBinarizacion::Signo).expect("neto");
    assert!(
        m.delta > 0.5,
        "GHZ: ΔS {d} domina al nulo (S_nulo {n})",
        d = m.delta,
        n = m.s_nulo
    );
}

#[test]
fn acotado_por_el_maximo_y_no_negativo() {
    // 0 ≤ S ≤ (L/2)·ln2 para el corte medio (anclas del 151); vale para
    // cualquier serie, incluida la patológica.
    let mut rng = Rng(0x0F0F_0F0F_0F0F_0F0F);
    let rets: Vec<f64> = (0..50_000)
        .map(|_| (rng.uniforme() * 2.0 - 1.0).powi(3) * 10.0)
        .collect();
    let m = entrelazamiento_neto(&rets, L, ModoBinarizacion::Mediana).expect("cola pesada");
    assert!(m.s >= 0.0 && m.s <= (L as f64 / 2.0) * std::f64::consts::LN_2 + 1e-9);
}

#[test]
fn no_finitos_se_excluyen_con_recuento_honesto() {
    // NaN e ±Inf se filtran ANTES de ventanear; la serie se comprime y el
    // descarte se reporta. Con datos suficientes la medición procede.
    let mut rng = Rng(0xAAAA_BBBB_CCCC_DDDD);
    let mut rets: Vec<f64> = Vec::with_capacity(10_000);
    for _ in 0..10_000 {
        let u = rng.uniforme();
        rets.push(if u < 0.02 {
            f64::NAN
        } else if u < 0.04 {
            f64::INFINITY
        } else {
            rng.uniforme() * 2.0 - 1.0
        });
    }
    let m = entrelazamiento_neto(&rets, L, ModoBinarizacion::Signo).expect("con basura fina");
    assert!(m.descartados >= 200 && m.descartados <= 500, "descartes ~4%: {}", m.descartados);
    assert_eq!(m.ventanas, 10_000 - m.descartados + 1 - L);
    // Todos no finitos ⇒ ausencia honesta, jamás inventado.
    let todo_nan = vec![f64::NAN; 100];
    assert_eq!(
        entrelazamiento_neto(&todo_nan, L, ModoBinarizacion::Signo),
        Err(EntrelazamientoError::RetornosInsuficientes { recibidos: 0, necesarios: L })
    );
}

#[test]
fn serie_constante_modo_mediana_es_ausencia_no_invention() {
    // Constante: la mediana degenera — error honesto; en modo signo la
    // medición procede (todo ↓, estado producto, S = 0).
    let rets = vec![0.001; 5_000];
    assert_eq!(
        entrelazamiento_neto(&rets, L, ModoBinarizacion::Mediana),
        Err(EntrelazamientoError::MedianaDegenerada)
    );
    let m = entrelazamiento_neto(&rets, L, ModoBinarizacion::Signo).expect("signo con constante");
    assert!(m.s.abs() < 1e-9 && m.delta.abs() < 1e-9, "constante ⇒ producto");
}

#[test]
fn longitud_invalida_y_serie_corta_son_error() {
    // Válidos: 1..=MPS_MAX_L (12). El vehículo usa L=8 por estimabilidad
    // (256 configs), no porque 9+ sea imposible.
    for l in [0usize, 13, 100] {
        assert_eq!(
            entrelazamiento_neto(&[1.0, -1.0], l, ModoBinarizacion::Signo),
            Err(EntrelazamientoError::LongitudInvalida { l })
        );
    }
    assert_eq!(
        entrelazamiento_neto(&[1.0, -1.0, 1.0], L, ModoBinarizacion::Signo),
        Err(EntrelazamientoError::RetornosInsuficientes { recibidos: 3, necesarios: L })
    );
    // L=9 VÁLIDO pero la serie corta sigue siendo insuficiente.
    assert_eq!(
        entrelazamiento_neto(&[1.0, -1.0], 9, ModoBinarizacion::Signo),
        Err(EntrelazamientoError::RetornosInsuficientes { recibidos: 2, necesarios: 9 })
    );
}
