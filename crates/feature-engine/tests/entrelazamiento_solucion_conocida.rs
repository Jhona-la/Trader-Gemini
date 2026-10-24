//! Contratos del entrelazamiento temporal neto (GLM 155) con SOLUCIÓN
//! CONOCIDA sintética — certifican la lógica en CI sin tape. El vehículo
//! de tape real es `backtest-engine/tests/mps_entrelazamiento_medicion.rs`
//! (#[ignore], investigación local). GLM 156 añadió la curva de fases y
//! GLM 157 el primitivo vivo `EntrelazamientoStreaming`.
use feature_engine::entrelazamiento::{
    EntrelazamientoError, EntrelazamientoStreaming, ModoBinarizacion, entrelazamiento_neto,
    estado_empirico,
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

// ══════════════════════════════════════════════════════════════════════
// GLM 156 — CURVA DE FASES S↔g DEL ISING (calibración física del ΔS).
// Grid grueso (12 puntos, D=4, 4 sweeps) para que CI sea rápido; el
// vehículo de tape usa el grid fino D=8. Solución conocida: campana con
// máximo en la región crítica g≈1, producto (S≈0) en ambos extremos.
// ══════════════════════════════════════════════════════════════════════

fn curva_prueba() -> feature_engine::entrelazamiento::CurvaFases {
    // 8 sweeps: con menos, el flanco desordenado (ground state |→…⟩ lejano
    // del init |↑…↑⟩) queda subconvergido y desplaza el máximo de g_max≈1
    // hacia la rama ordenada (hallazgo del propio contrato, 156).
    feature_engine::entrelazamiento::CurvaFases::construir(L, 4, 8, 0.05, 20.0, 10)
        .expect("curva construible")
}

/// Los contratos de la curva comparten UNA construcción (el DMRG es el
/// costo). Física de cadena FINITA: g→0 es el estado GATO (S→ln2, NO el
/// producto — el propio contrato corrigió la campana asumida), g→∞ es el
/// producto paramagnético (S→0); la curva es decreciente con el crítico
/// g_c=1 como hombro. Esto lo enseñó el 156.
#[test]
fn curva_fases_gato_critico_y_paramagnetico() {
    let c = curva_prueba();
    let s_en = |g0: f64| -> f64 {
        c.puntos
            .iter()
            .min_by(|a, b| {
                (a.g - g0).abs().partial_cmp(&(b.g - g0).abs()).unwrap_or(std::cmp::Ordering::Equal)
            })
            .map(|p| p.s)
            .unwrap_or(f64::NAN)
    };
    // — Extremo ordenado: GATO coherente, S ≈ ln2 (NO producto).
    assert!(
        (s_en(0.05) - std::f64::consts::LN_2).abs() < 0.02,
        "g→0 es el gato: S={s0} ≈ ln2",
        s0 = s_en(0.05)
    );
    // — Extremo desordenado: producto paramagnético, S ≈ 0.
    assert!(s_en(20.0) < 0.02, "g→∞ producto: {s1}", s1 = s_en(20.0));
    // — Decreciente con hombro crítico: gato > crítico > paramagnético.
    let s_critico = s_en(1.0);
    assert!(s_en(0.05) > s_critico, "gato > crítico");
    assert!(s_critico > s_en(5.0), "crítico > paramagnético");
    assert!(s_critico > 0.3 && s_critico < 0.69, "crítico L=8 D=4 ≈ 0.42: {s_critico}");
    assert!((c.g_max - 0.05).abs() < 1e-9 || c.s_max < 0.71, "s_max en el borde ordenado");

    // — Roundtrip: S(g de cada lado del crítico) → invertir → algúm cruce
    // cerca de ese g (cruces = TODOS los que haya; basta el más cercano).
    for g_prueba in [0.4f64, 2.5] {
        let s_prueba = c
            .puntos
            .iter()
            .min_by(|a, b| {
                (a.g - g_prueba)
                    .abs()
                    .partial_cmp(&(b.g - g_prueba).abs())
                    .unwrap_or(std::cmp::Ordering::Equal)
            })
            .map(|p| p.s)
            .expect("punto cercano");
        let inv = c.invertir(s_prueba).expect("invertible dentro de la curva");
        let mejor = inv
            .cruces
            .iter()
            .fold(f64::INFINITY, |m, g| m.min((g.ln() - g_prueba.ln()).abs()));
        assert!(
            mejor < 0.5,
            "g={g_prueba}: S={s_prueba:.4} ⇒ cruces {:?} (mejor log-dist {mejor})",
            inv.cruces
        );
    }

    // — Sobre el techo (más coherente que el gato del Ising L=8) ⇒
    // ausencia honesta con el límite real; jamás cruce inventado.
    let s_obj = c.s_max + 0.05;
    match c.invertir(s_obj) {
        Err(EntrelazamientoError::FueraDeCurva { s, s_max, .. }) => {
            assert_eq!(s, s_obj);
            assert_eq!(s_max, c.s_max);
        }
        otro => panic!("sobre el techo ⇒ FueraDeCurva, llegó {otro:?}"),
    }
    // Sub-borde (serie casi-iid, por debajo del S mínimo del grid): igual.
    match c.invertir(1e-6) {
        Err(EntrelazamientoError::FueraDeCurva { .. }) => {}
        otro => panic!("sub-borde ⇒ FueraDeCurva, llegó {otro:?}"),
    }
}

#[test]
fn curva_fases_es_determinista_y_valida_parametros() {
    // Dos construcciones ⇒ idénticas bit a bit (init fijo, Jacobi propio,
    // sin azar en ninguna etapa). Grid L=6 para que sea barato.
    let a = feature_engine::entrelazamiento::CurvaFases::construir(6, 4, 3, 0.05, 20.0, 8)
        .expect("curva 6 construible");
    let b = feature_engine::entrelazamiento::CurvaFases::construir(6, 4, 3, 0.05, 20.0, 8)
        .expect("curva 6 construible");
    assert_eq!(a, b, "determinismo bit a bit");
    // Parámetros degenerados ⇒ error honesto.
    assert_eq!(
        feature_engine::entrelazamiento::CurvaFases::construir(L, 4, 3, -1.0, 2.0, 10),
        Err(EntrelazamientoError::CurvaDegenerada)
    );
    assert_eq!(
        feature_engine::entrelazamiento::CurvaFases::construir(1, 4, 3, 0.05, 2.0, 10),
        Err(EntrelazamientoError::LongitudInvalida { l: 1 })
    );
    assert_eq!(
        feature_engine::entrelazamiento::CurvaFases::construir(L, 4, 3, 0.05, 2.0, 3),
        Err(EntrelazamientoError::CurvaDegenerada)
    );
}

// ══════════════════════════════════════════════════════════════════════
// GLM 157 — ENTRELAZAMIENTO STREAMING (el primitivo vivo del Átomo 3.3
// de AGY): paridad bit con el batch por construcción, ventana acotada al
// ÚLTIMO tramo, exclusión honesta de no finitos.
// ══════════════════════════════════════════════════════════════════════

fn serie_markov(n: usize, p_misma: f64, semilla: u64) -> Vec<f64> {
    let mut rng = Rng(semilla);
    let mut arriba = true;
    (0..n)
        .map(|_| {
            arriba = rng.uniforme() < if arriba { p_misma } else { 1.0 - p_misma };
            if arriba { 1.0 } else { -1.0 }
        })
        .collect()
}

#[test]
fn streaming_es_bit_paridad_con_batch() {
    // Markov persistente (señal real) + no finitos intercalados: el
    // streaming (push uno a uno) y el batch sobre la MISMA serie con NaN
    // deben coincidir EXACTO (misma binarización tras filtrar, misma
    // permutación determinista del mismo vector filtrado).
    let limpia = serie_markov(30_000, 0.8, 0x157_C0FFEE);
    let mut con_nan = limpia.clone();
    let mut rng = Rng(0x157_DEAD_BEEF);
    for k in (0..con_nan.len()).step_by(97) {
        if rng.uniforme() < 0.5 {
            con_nan[k] = f64::NAN;
        }
    }
    let batch = entrelazamiento_neto(&con_nan, L, ModoBinarizacion::Signo).expect("batch");
    let mut st = EntrelazamientoStreaming::new(L, ModoBinarizacion::Signo, 100_000)
        .expect("streaming válido");
    for &r in &con_nan {
        st.push(r);
    }
    let m = st.medir().expect("streaming mide");
    assert_eq!(m.ventanas, batch.ventanas, "ventanas idénticas");
    assert_eq!(m.descartados, batch.descartados, "descartes idénticos");
    assert!(
        (m.s - batch.s).abs() < 1e-12 && (m.s_nulo - batch.s_nulo).abs() < 1e-12,
        "S y S_nulo bit-paridad: {s1} vs {s2} / {n1} vs {n2}",
        s1 = m.s,
        s2 = batch.s,
        n1 = m.s_nulo,
        n2 = batch.s_nulo
    );
    assert!((m.delta - batch.delta).abs() < 1e-12);
    assert_eq!(st.activos(), batch.ventanas + L - 1, "ring retiene los finitos");
    assert_eq!(st.descartados_totales(), batch.descartados);
}

#[test]
fn streaming_expira_al_ultimo_tramo() {
    // Capacidad W < N: medir() tras N pushes == batch sobre los ÚLTIMOS W
    // retornos (el ring descarta los viejos, no los mezcla).
    let serie = serie_markov(5_000, 0.8, 0x157_ABCDEF);
    let w = 2_000;
    let mut st = EntrelazamientoStreaming::new(L, ModoBinarizacion::Signo, w)
        .expect("streaming válido");
    for &r in &serie {
        st.push(r);
    }
    let m = st.medir().expect("mide el tramo");
    let cola = &serie[serie.len() - w..];
    let batch_cola = entrelazamiento_neto(cola, L, ModoBinarizacion::Signo).expect("batch cola");
    assert_eq!(m.ventanas, batch_cola.ventanas);
    assert!((m.s - batch_cola.s).abs() < 1e-12, "mide el ÚLTIMO tramo");
    assert_eq!(st.activos(), w, "ring lleno exacto en W");
    // El batch del tramo distinto del inicial demuestra que expiró de verdad.
    let batch_inicio =
        entrelazamiento_neto(&serie[..w], L, ModoBinarizacion::Signo).expect("batch inicio");
    assert!((m.s - batch_inicio.s).abs() > 1e-9, "no es el tramo inicial");
}

#[test]
fn streaming_iid_no_inventa_y_bordes_son_honestos() {
    // iid ⇒ |ΔS| pequeño (el primitivo vivo no fabrica señal).
    let mut st = EntrelazamientoStreaming::new(L, ModoBinarizacion::Signo, 50_000).unwrap();
    let mut rng = Rng(0x157_0123);
    for _ in 0..40_000 {
        st.push(if rng.uniforme() < 0.42 { 1.0 } else { -1.0 });
    }
    let m = st.medir().expect("iid mide");
    assert!(m.delta.abs() < 5e-3, "iid streaming: |ΔS| {d}", d = m.delta);
    // Serie corta (< L en el ring): ausencia honesta.
    let mut corto = EntrelazamientoStreaming::new(L, ModoBinarizacion::Signo, 100).unwrap();
    for k in 0..3 {
        corto.push(k as f64);
    }
    assert_eq!(
        corto.medir(),
        Err(EntrelazamientoError::RetornosInsuficientes { recibidos: 3, necesarios: L })
    );
    // Constructor con capacidad < L o L inválido: error temprano
    // (L=8 ⇒ capacidad 7 insuficiente).
    assert!(matches!(
        EntrelazamientoStreaming::new(L, ModoBinarizacion::Signo, 7),
        Err(EntrelazamientoError::RetornosInsuficientes { recibidos: 7, necesarios: 8 })
    ));
    assert!(matches!(
        EntrelazamientoStreaming::new(0, ModoBinarizacion::Signo, 100),
        Err(EntrelazamientoError::LongitudInvalida { l: 0 })
    ));
    // El recuento de descartes se REINICIA por medición (semántica tramo).
    let mut st2 = EntrelazamientoStreaming::new(L, ModoBinarizacion::Signo, 100).unwrap();
    st2.push(f64::NAN);
    st2.push(f64::INFINITY);
    for k in 0..100 {
        st2.push(if k % 2 == 0 { 1.0 } else { -1.0 });
    }
    let m2 = st2.medir().expect("mide tras descartes");
    assert_eq!(m2.descartados, 2, "dos descartes en el tramo");
    let m3 = st2.medir().expect("segunda medición");
    assert_eq!(m3.descartados, 0, "recuento por tramo, reiniciado");
    assert_eq!(st2.descartados_totales(), 2, "total acumulado");
}

// ══════════════════════════════════════════════════════════════════════
// GLM 158 — ENTRELAZAMIENTO DE TSALLIS S_q (familia no extensiva, mandato
// §10). Límite q→1 = von Neumann exacto; ancla GHZ = ln_q(2) cerrado;
// iid sin señal; monotonía en q; guardas honestas.
// ══════════════════════════════════════════════════════════════════════

#[test]
fn tsallis_limite_q1_es_von_neumann_exacto() {
    // Sobre el psi empírico de un Markov: q=1.0 EXACTO (mismo camino) y
    // q=1.0001 converge al von Neumann a <1e-6 (límite analítico real).
    let serie = serie_markov(30_000, 0.8, 0x158_CAFE);
    let (psi, _, _) = estado_empirico(&serie, L, ModoBinarizacion::Signo).expect("estado");
    let s_vn = feature_engine::mps::entrelazamiento_corte(&psi, L, L / 2).expect("VN");
    let s_q1 = feature_engine::entrelazamiento::entrelazamiento_tsallis(&psi, L, L / 2, 1.0)
        .expect("q=1");
    assert_eq!(s_q1, s_vn, "q=1.0: EXACTO (camino von Neumann)");
    let s_q1e4 =
        feature_engine::entrelazamiento::entrelazamiento_tsallis(&psi, L, L / 2, 1.0001)
            .expect("q→1");
    // El límite es de PRIMER orden: dS_q/dq|_{q=1} ≠ 0 (relacionado con la
    // varianza de ln p del espectro) ⇒ |S_{1+ε} − S| ≤ C·ε con C acotado.
    // Con ε=1e-4, exigir <1e-4 (derivada de orden unidad) es lo honesto;
    // exigir 1e-6 confundiría orden de convergencia con exactitud.
    assert!(
        (s_q1e4 - s_vn).abs() < 1e-4,
        "límite de primer orden: |S_{{1+ε}} − S| = {d} ≤ ε·C",
        d = (s_q1e4 - s_vn).abs()
    );
    // El pipeline neto también: q=1 ⇒ idéntico al neto clásico.
    let m_q1 = feature_engine::entrelazamiento::entrelazamiento_neto_q(
        &serie, L, ModoBinarizacion::Signo, 1.0,
    )
    .expect("neto q=1");
    let m_clasico = entrelazamiento_neto(&serie, L, ModoBinarizacion::Signo).expect("neto");
    assert_eq!(m_q1.s, m_clasico.s);
    assert_eq!(m_q1.delta, m_clasico.delta);
}

#[test]
fn tsallis_ghz_recupera_ln_q_de_2_cerrado() {
    // La serie alternante produce el gato (Schmidt ½,½): S_q = ln_q(2) =
    // (2^{1−q} − 1)/(1−q) forma cerrada para TODO q > 0.
    let rets: Vec<f64> = (0..100_000).map(|i| if i % 2 == 0 { 1.0 } else { -1.0 }).collect();
    let (psi, _, _) = estado_empirico(&rets, L, ModoBinarizacion::Signo).expect("estado");
    for q in [0.5f64, 1.0, 1.5, 2.0, 3.0] {
        let s_q = feature_engine::entrelazamiento::entrelazamiento_tsallis(&psi, L, L / 2, q)
            .expect("S_q del gato");
        let ln_q2 = if (q - 1.0).abs() < 1e-12 {
            std::f64::consts::LN_2
        } else {
            (2.0f64.powf(1.0 - q) - 1.0) / (1.0 - q)
        };
        assert!(
            (s_q - ln_q2).abs() < 1e-9,
            "q={q}: S_q {s_q} = ln_q(2) {ln_q2}"
        );
    }
    // Producto ⇒ S_q = 0 para toda q.
    let plano: Vec<f64> = vec![1.0; 20_000];
    let (psi_p, _, _) = estado_empirico(&plano, L, ModoBinarizacion::Signo).expect("producto");
    for q in [0.5f64, 1.5] {
        let s_q = feature_engine::entrelazamiento::entrelazamiento_tsallis(&psi_p, L, L / 2, q)
            .expect("S_q del producto");
        assert!(s_q.abs() < 1e-9, "producto ⇒ S_q=0 (q={q})");
    }
}

#[test]
fn tsallis_neto_iid_no_inventa_y_monotonia_en_q() {
    // iid: |ΔS_q| pequeño — con tolerancia DEPENDIENTE de q: q<1 eleva
    // p pequeños (p^q > p), amplificando el ruido de muestreo de los bins
    // raros ⇒ el nulo cancela menos bien (medido: ~8e-3 con 200k ventanas
    // a q=0.5). Propiedad del estimador, no bug.
    let mut rng = Rng(0x158_777);
    let iid: Vec<f64> = (0..200_000)
        .map(|_| if rng.uniforme() < 0.35 { 1.0 } else { -1.0 })
        .collect();
    for (q, tol) in [(0.5f64, 2e-2), (1.5, 5e-3)] {
        let m = feature_engine::entrelazamiento::entrelazamiento_neto_q(
            &iid, L, ModoBinarizacion::Signo, q,
        )
        .expect("iid medible");
        assert!(
            m.delta.abs() < tol,
            "iid q={q}: |ΔS_q| {d} < {tol}",
            d = m.delta
        );
    }
    // Monotonía: S_q NO creciente en q para un espectro fijo con p_max>1/2
    // (Tsallis penaliza la cola ancha al crecer q). Markov persistente.
    let serie = serie_markov(30_000, 0.85, 0x158_ABCD);
    let (psi, _, _) = estado_empirico(&serie, L, ModoBinarizacion::Signo).expect("estado");
    let mut anterior = f64::INFINITY;
    for q in [0.5f64, 0.8, 1.0, 1.2, 1.5, 2.0] {
        let s_q = feature_engine::entrelazamiento::entrelazamiento_tsallis(&psi, L, L / 2, q)
            .expect("S_q");
        assert!(
            s_q <= anterior + 1e-12,
            "monotonía en q: S_{q} {s_q} ≤ anterior {anterior}"
        );
        anterior = s_q;
    }
    // Guardas: q inválido ⇒ error honesto, jamás NaN (el caso q=NaN se
    // verifica aparte porque NaN ≠ NaN en assert_eq de campos).
    for q in [0.0f64, -1.0, f64::INFINITY] {
        assert_eq!(
            feature_engine::entrelazamiento::entrelazamiento_tsallis(&psi, L, L / 2, q),
            Err(feature_engine::entrelazamiento::EntrelazamientoError::QInvalido { q })
        );
    }
    match feature_engine::entrelazamiento::entrelazamiento_tsallis(&psi, L, L / 2, f64::NAN) {
        Err(feature_engine::entrelazamiento::EntrelazamientoError::QInvalido { q }) => {
            assert!(q.is_nan(), "el error preserva el q=NaN");
        }
        otro => panic!("q=NaN ⇒ QInvalido, llegó {otro:?}"),
    }
}
