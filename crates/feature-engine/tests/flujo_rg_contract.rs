//! Contratos del flujo de renormalización de Wilson (GLM-183, mandato
//! §12.2) con SOLUCIÓN CONOCIDA: para una ley de potencias pura el
//! exponente espectral es un PUNTO FIJO EXACTO del coarse-graining
//! diádico (identidad algebraica), el ruido blanco fija β=0, y un
//! espectro de dos pendientes rompe el flujo en la escala crítica.
use feature_engine::flujo_rg::flujo_wilson;

/// Espectro 1/f^β muestreado por bin (bin 0 = DC, ignorado por el módulo).
fn espectro_potencia(n: usize, beta: f64) -> Vec<f64> {
    (0..n)
        .map(|i| if i == 0 { 1.0 } else { (i as f64).powf(-beta) })
        .collect()
}

#[test]
fn ley_de_potencias_pura_es_punto_fijo_exacto() {
    for beta in [0.5_f64, 1.0, 1.7, 2.5] {
        let flujo = flujo_wilson(&espectro_potencia(257, beta)).expect("filtrable");
        assert!(flujo.n_octavas >= 7, "257 bins ⇒ ≥7 octavas");
        for (k, &b) in flujo.betas.iter().enumerate() {
            assert!(
                (b - beta).abs() < 1e-9,
                "β_{k} = {b} debe ser el punto fijo {beta} (identidad algebraica)"
            );
        }
        assert!(
            flujo.desvio_punto_fijo < 1e-9,
            "sin desviación del punto fijo, dio {}",
            flujo.desvio_punto_fijo
        );
        // Con desviación nula la ruptura es None o salto ~0: no hay escala
        // crítica que reportar — el régimen es invariante.
        if let Some(salto) = flujo.salto_rompimiento {
            assert!(salto < 1e-9, "salto residual {salto}");
        }
    }
}

#[test]
fn ruido_blanco_fija_beta_cero() {
    let flujo = flujo_wilson(&espectro_potencia(129, 0.0)).expect("filtrable");
    assert!(flujo.betas.iter().all(|b| b.abs() < 1e-9));
    assert!(flujo.desvio_punto_fijo < 1e-9);
}

#[test]
fn dos_pendientes_rompen_el_flujo_en_la_escala_critica() {
    // β1=0.2 para f ≤ 16, β2=2.2 arriba, CONTINUO en el quiebre (la rama
    // alta anclada a 16^−β1 — un espectro físico con ruptura espectral no
    // salta de nivel; la primera versión del test era discontinua y medía
    // el salto de NIVEL (9.5), no el de PENDIENTE).
    let quiebre = 16.0f64;
    let (beta1, beta2) = (0.2_f64, 2.2_f64);
    let mut espectro = Vec::with_capacity(257);
    espectro.push(1.0); // DC
    for i in 1..257 {
        let f = i as f64;
        let p = if f <= quiebre {
            f.powf(-beta1)
        } else {
            quiebre.powf(-beta1) * (f / quiebre).powf(-beta2)
        };
        espectro.push(p);
    }
    let flujo = flujo_wilson(&espectro).expect("filtrable");
    assert!(
        flujo.desvio_punto_fijo > 1.0,
        "la ruptura debe ser visible, desviación {}",
        flujo.desvio_punto_fijo
    );
    let salto = flujo.salto_rompimiento.expect("ruptura presente");
    assert!(
        (1.0..2.5).contains(&salto),
        "salto {salto} ≈ β2−β1 = 2.0 (mezcla de la octava del quiebre)"
    );
    let escala = flujo.escala_rompimiento.expect("escala presente");
    // El quiebre está en f=16 = 2^4 ⇒ la ruptura cae entre las escalas
    // 3-5 (octavas alrededor de 2^4); no en los extremos del flujo.
    assert!(
        (2..=6).contains(&escala),
        "escala de ruptura {escala} cerca del quiebre 2^4"
    );
    // Fuera del quiebre el flujo vuelve a casi-fijarse en cada régimen:
    // el primer β ≈ β1 y el último ≈ β2.
    assert!((flujo.betas[0] - beta1).abs() < 0.15);
    let ultimo = *flujo.betas.last().expect("no vacío");
    assert!((ultimo - beta2).abs() < 0.35);
}

#[test]
fn entradas_invalidas_son_error_honesto() {
    assert!(flujo_wilson(&[]).is_err());
    assert!(flujo_wilson(&[1.0, 1.0]).is_err());
    assert!(flujo_wilson(&[1.0, 1.0, 1.0]).is_err());
    // GLM 184: un bin sin energía se EXCLUYE de su octava — pero si la
    // octava queda ENTERA muda no hay media de log: Err honesto. Los tres
    // casos de abajo tienen su bin inválido en la primera octava (1 bin
    // por octava al inicio del eje) ⇒ octava muda ⇒ Err.
    assert!(flujo_wilson(&[1.0, 0.0, 1.0, 1.0, 1.0]).is_err());
    assert!(flujo_wilson(&[1.0, -0.5, 1.0, 1.0, 1.0]).is_err());
    assert!(flujo_wilson(&[1.0, f64::NAN, 1.0, 1.0, 1.0]).is_err());
    // 4 bins (3 tras DC) = mínimo con dos octavas: válido.
    assert!(flujo_wilson(&[1.0, 1.0, 1.0, 1.0]).is_ok());
}

#[test]
fn bins_sin_energia_en_octavas_pobladas_se_excluyen_sin_inventar() {
    // GLM 184: el FFT de 64 retornos deja bins a CERO exacto (frecuencias
    // ausentes). Un espectro 1/f^β con ceros intercalados en octavas
    // POBLADAS conserva el punto fijo EXACTO: los ceros salen de la media
    // — la afinidad log P ↔ log f de los bins vivos es la misma ley.
    let beta = 1.3_f64;
    let mut espectro = espectro_potencia(257, beta);
    for i in [5, 9, 10, 33, 100] {
        espectro[i] = 0.0; // bins mudos en octavas con otros vivos.
    }
    let flujo = flujo_wilson(&espectro).expect("flujo vivo pese a ceros");
    for (k, &b) in flujo.betas.iter().enumerate() {
        assert!(
            (b - beta).abs() < 1e-9,
            "β_{k} = {b} punto fijo {beta} pese a bins excluidos"
        );
    }
}
