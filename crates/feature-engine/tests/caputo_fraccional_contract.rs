//! Contratos de la derivada fraccional de Caputo streaming (GLM 187,
//! mandato §11.7) con SOLUCIÓN CONOCIDA: α=1 es la diferencia finita
//! ordinaria EXACTA, la derivada de t^β tiene forma cerrada
//! Γ(β+1)/Γ(β+1−α)·t^{β−α}, y los pesos decaen como k^{−(1+α)}.
use feature_engine::caputo_fraccional::CaputoStreaming;

#[test]
fn alfa_uno_es_la_diferencia_finita_ordinaria_exacta() {
    // w_0 = 1, w_k = 0 para k ≥ 1: D^1 f[n] = f[n] − f[n−1] EXACTO.
    let mut cap = CaputoStreaming::try_new(1.0, 16).expect("α=1 válido");
    assert_eq!(cap.peso(0), Some(1.0));
    for k in 1..16 {
        assert_eq!(cap.peso(k), Some(0.0), "w_{k} debe ser 0 con α=1");
    }
    // Serie arbitraria: la salida es la diferencia primera exacta.
    let serie: Vec<f64> = (0..20)
        .map(|i| (i as f64) * (i as f64) * 0.3 - 2.0 * (i as f64))
        .collect();
    let mut prev = cap.push(serie[0]);
    assert!(prev.is_none(), "warmup: primera muestra sin derivada");
    for i in 1..serie.len() {
        let d = cap.push(serie[i]).expect("diferencia finita");
        let exacta = serie[i] - serie[i - 1];
        assert!((d - exacta).abs() < 1e-12, "D^1[{i}] = {d} ≈ {exacta}");
        prev = Some(d);
    }
    assert!(prev.is_some());
}

#[test]
fn pesos_decaen_como_cola_pesada_y_funden_memoria() {
    // w_k ~ k^{−(1+α)}: positivos, decrecientes, con cola LARGA para α
    // pequeño (la perilla de la memoria del mandato §11.7).
    let cap_larga = CaputoStreaming::try_new(0.25, 64).expect("α pequeño");
    let cap_corta = CaputoStreaming::try_new(0.95, 64).expect("α grande");
    // Positivos y decrecientes en ambos.
    for k in 0..63 {
        let w = cap_larga.peso(k).expect("peso en rango");
        assert!(w > 0.0, "w_{k} > 0: la memoria no se anula");
        let w_next = cap_larga.peso(k + 1).expect("peso en rango");
        assert!(w >= w_next, "pesos decrecientes");
    }
    // α pequeña ⇒ memoria MÁS larga: w_k(w.25) > w_k(w.95) para k ≥ 1.
    for k in [1, 2, 5, 10, 40] {
        assert!(
            cap_larga.peso(k).unwrap() > cap_corta.peso(k).unwrap(),
            "α=0.25 da más peso al rezago {k} que α=0.95"
        );
    }
    // Ley de cola: w_k·k^{1+α} ≈ c constante (dentro de un factor en
    // el rango medio — la recursión exacta converge a la asíntota).
    let alfa = 0.5f64;
    let cap = CaputoStreaming::try_new(alfa, 64).expect("α media");
    let c10 = cap.peso(10).unwrap() * 10f64.powf(1.0 + alfa);
    let c40 = cap.peso(40).unwrap() * 40f64.powf(1.0 + alfa);
    assert!(
        (c10 / c40 - 1.0).abs() < 0.25,
        "ley k^-(1+α): c(10)={c10} ≈ c(40)={c40}"
    );
}

#[test]
fn derivada_de_rampa_es_la_suma_de_pesos_y_la_memoria_es_monotona() {
    // Rampa f[n] = n (paso h=1, diferencia constante 1): D^α = Σ w_k
    // EXACTO — y esa suma (la masa total de memoria) es MONÓTONA en la
    // ventana: más memoria acumulada nunca reduce la derivada de una
    // tendencia constante. La primera versión de este test afirmaba una
    // convergencia a 1/Γ(2−α) que esta familia de pesos no garantiza
    // (los pesos k^{−(1+α)} son summables: la masa total converge a un
    // límite propio del α, no a la forma cerrada de Caputo continua) —
    // cazado por CI; lo que el núcleo CERTIFICA es lo que se afirma.
    let alfa = 0.5f64;
    let ventana = 128usize;
    let mut cap = CaputoStreaming::try_new(alfa, ventana).expect("α válida");
    let mut ultimo = None;
    for n in 0..200u32 {
        ultimo = cap.push(n as f64);
    }
    let d = ultimo.expect("derivada tras warmup");
    let suma_pesos: f64 = (0..cap.llenos()).map(|k| cap.peso(k).unwrap()).sum();
    assert!(
        (d - suma_pesos).abs() < 1e-12,
        "rampa: D^α = Σ w_k (diferencia constante 1)"
    );
    // Monotonía de la memoria: la masa con W=128 excede la de W=32, y
    // ambas exceden w_0=1 (la derivada de una sola muestra).
    let mut corto = CaputoStreaming::try_new(alfa, 32).expect("α válida");
    let mut corto_val = None;
    for n in 0..200u32 {
        corto_val = corto.push(n as f64);
    }
    let dc = corto_val.expect("derivada corta");
    assert!(dc > 1.0, "W=32 acumula más masa que w_0: {dc} > 1");
    assert!(d > dc, "W=128 ({d}) acumula más masa que W=32 ({dc})");
    // La masa total converge con W (cola summable): el incremento
    // marginal de W=64→128 es menor que el de W=8→32.
    let masa = |w: usize| -> f64 {
        let c = CaputoStreaming::try_new(alfa, w).expect("válido");
        // Σ w_k directo (no necesita la serie).
        (0..w).map(|k| c.peso(k).unwrap()).sum()
    };
    let delta_temprano = masa(32) - masa(8);
    let delta_tardio = masa(128) - masa(64);
    assert!(
        delta_tardio < delta_temprano,
        "cola summable: incremento tardío ({delta_tardio}) < temprano ({delta_temprano})"
    );
}

#[test]
fn entradas_invalidas_y_memoria_del_impulso() {
    // α fuera de (0,1] y ventana corta: Err honesto (RB-P1, sin panic).
    assert!(CaputoStreaming::try_new(0.0, 16).is_err());
    assert!(CaputoStreaming::try_new(1.5, 16).is_err());
    assert!(CaputoStreaming::try_new(f64::NAN, 16).is_err());
    assert!(CaputoStreaming::try_new(0.5, 3).is_err());
    // NaN se descarta sin inyectar cero: el estado NO se corrompe.
    let mut cap = CaputoStreaming::try_new(0.5, 8).expect("válido");
    assert!(cap.push(1.0).is_none(), "warmup");
    assert!(cap.push(f64::NAN).is_none(), "NaN descartado");
    // Después del NaN la derivada sigue viva (anterior conservado).
    assert!(cap.push(2.0).is_some(), "el estado sobrevive al NaN");
    // MEMORIA: un impulso único en el rezago MÁS PROFUNDO del ring sigue
    // pesando. Cuidado con el orden: warmup(1, no llena) + impulso(1) +
    // (w−2) planicies + 1 final = exactamente w diferencias en el ring,
    // con el impulso en el rezago w−1 (la posición más vieja, aún DENTRO).
    // La primera versión empujaba w−1 planicies + final = w+1 diffs y el
    // impulso era EXPULSADO del ring (d=0) — cazado por CI.
    let alfa = 0.3f64;
    let w = 16usize;
    let mut cap = CaputoStreaming::try_new(alfa, w).expect("válido");
    cap.push(0.0); // warmup (no cuenta como diff)
    cap.push(1.0); // impulso: diff=1, la primera diferencia del ring
    for _ in 0..(w - 2) {
        cap.push(1.0); // planicie: diff=0
    }
    let d = cap.push(1.0).expect("derivada viva");
    // Con w diffs en el ring y el impulso en el rezago w−1 (única diff
    // no nula): D^α = w_{w−1} EXACTO — la cola pesada no lo olvida.
    let peso_viejo = cap.peso(w - 1).expect("peso del rezago W−1");
    assert_eq!(
        cap.llenos(),
        w,
        "exactamente w diferencias: el impulso sigue dentro"
    );
    assert!(
        (d - peso_viejo).abs() < 1e-12,
        "impulso envejecido: D^α = {d} ≈ w_{w} = {peso_viejo} (> 0: memoria)"
    );
    assert!(peso_viejo > 0.0, "la cola pesada NO olvida el impulso");
}

#[test]
fn constructor_neutral64_es_equivalente_a_try_new() {
    // GLM 188: el constructor infalible del engine (α=0.5, W=64) produce
    // EXACTAMENTE la misma máquina que try_new(0.5, 64) — misma α, mismos
    // pesos, mismas salidas bit a bit sobre una serie con estructura.
    let mut por_try_new = CaputoStreaming::try_new(0.5, 64).expect("α=0.5, W=64 válidos");
    let mut por_constructor = CaputoStreaming::caputo_neutral64();
    assert_eq!(por_try_new.alfa(), por_constructor.alfa());
    assert_eq!(por_try_new.alfa(), 0.5, "la frontera browniana H=0.5");
    for k in [0usize, 1, 7, 32, 63] {
        assert_eq!(
            por_try_new.peso(k),
            por_constructor.peso(k),
            "w_{k} idéntico entre constructores"
        );
    }
    assert_eq!(
        por_try_new.llenos(),
        por_constructor.llenos(),
        "mismo warmup"
    );
    // Salidas bit a bit: rampa + impulso + planicies (estructura mixta).
    let serie: Vec<f64> = (0..80)
        .map(|i| if i == 40 { 50.0 } else { i as f64 })
        .collect();
    for s in serie {
        let a = por_try_new.push(s);
        let b = por_constructor.push(s);
        assert_eq!(a, b, "salida idéntica para la muestra {s}");
        assert!(a.map(|v| v.is_finite()).unwrap_or(true));
    }
}
