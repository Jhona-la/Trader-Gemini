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
fn derivada_de_rampa_converge_a_la_forma_cerrada() {
    // D^α t = t^{1−α}/Γ(2−α). La GL truncada converge desde arriba (el
    // truncamiento quita masa positiva); la tolerancia es honesta.
    let alfa = 0.5f64;
    let ventana = 128usize;
    let mut cap = CaputoStreaming::try_new(alfa, ventana).expect("α válida");
    // Serie rampa f[n] = n (paso h=1): diferencia constante 1.
    let mut ultimo = None;
    for n in 0..200u32 {
        ultimo = cap.push(n as f64);
    }
    let d = ultimo.expect("derivada tras warmup");
    // GL truncada sobre rampa: para paso h=1 la suma de pesos ES la
    // derivada (diferencia constante 1).
    let suma_pesos: f64 = (0..cap.llenos()).map(|k| cap.peso(k).unwrap()).sum();
    assert!(
        (d - suma_pesos).abs() < 1e-12,
        "rampa: D^α = Σ w_k (diferencia constante 1)"
    );
    // Convergencia a 1/Γ(2−α) con ventana creciente: el error de
    // truncamiento decae — la ventana 128 está más cerca que la 32.
    // Γ(2−α) con α=0.5 ⇒ Γ(1.5) = √π/2 (forma cerrada explícita: el
    // nightly del proyecto no estabiliza f64::gamma — E0658 en CI).
    let mut corto = CaputoStreaming::try_new(alfa, 32).expect("α válida");
    let mut corto_val = None;
    for n in 0..200u32 {
        corto_val = corto.push(n as f64);
    }
    let objetivo = 1.0 / (std::f64::consts::PI.sqrt() * 0.5);
    let err_largo = (d - objetivo).abs();
    let err_corto = (corto_val.expect("derivada") - objetivo).abs();
    assert!(
        err_largo <= err_corto,
        "ventana 128 (err {err_largo}) converge mejor que 32 (err {err_corto})"
    );
    assert!(
        err_largo < 0.35 * objetivo,
        "GL(128) cerca de 1/Γ(2−α): err {err_largo} vs objetivo {objetivo}"
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
    // MEMORIA: un impulso único hace W muestras atrás sigue pesando.
    let alfa = 0.3f64;
    let w = 16usize;
    let mut cap = CaputoStreaming::try_new(alfa, w).expect("válido");
    cap.push(0.0); // warmup
    cap.push(1.0); // impulso: diff=1
    for _ in 0..(w - 1) {
        cap.push(1.0); // planicie: diff=0 — el impulso envejece en el ring
    }
    let d = cap.push(1.0).expect("derivada viva");
    // El impulso está ahora en el rezago w−1: su peso es ~w^{−(1+α)} > 0
    // — y como es la ÚNICA diferencia no nula, D^α ≈ w_{w−1} exacto.
    let peso_viejo = cap.peso(w - 1).expect("peso del rezago W−1");
    assert!(
        (d - peso_viejo).abs() < 1e-12,
        "impulso envejecido: D^α = w_{w} = {d} ≈ {peso_viejo} (> 0: memoria)"
    );
    assert!(peso_viejo > 0.0, "la cola pesada NO olvida el impulso");
}
