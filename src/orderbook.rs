//! Microestructura del mejor nivel del libro (RET-45 / F472 del censo C-F7b;
//! RB-M1a de Codex RB, 2026-10-19).
//!
//! Antes vivía aquí un `OrderBook` con dos `BTreeMap<OrderedFloat, f64>` por
//! símbolo que el host alimentaba con el nivel 0 de cada mensaje `@depth5`.
//! Un snapshot de profundidad nunca trae cantidad 0, así que el `remove` no
//! corría jamás: cada precio que alguna vez fue mejor bid o ask quedaba en
//! memoria hasta la siguiente reconexión (fuga ALTA del censo RB-M1), con un
//! `insert` O(log n) por mensaje, y nadie leía esos mapas. El OBI y el
//! microprecio que sí llegan al núcleo se calculaban en línea en el host con
//! las mismas fórmulas que los métodos del struct, y el censo pidió UNIFICAR
//! hacia el método. Queda una función pura, sin estado y sin asignaciones, que
//! el host llama por mensaje; estos contratos la certifican bit a bit contra
//! la fórmula previa del host y verifican que el host la usa.

/// OBI y divergencia del microprecio en el mejor nivel de una instantánea del
/// libro: precios > 0, cantidades ≥ 0, todo finito y bid < ask (lo que
/// garantiza `parsers::parse_binance_depth5_levels`).
///
/// - `obi = (q_b − q_a) / (q_b + q_a)` ∈ [−1, 1]: desequilibrio del mejor nivel.
/// - `microprecio = (p_b·q_a + p_a·q_b) / (q_b + q_a)`: el precio del mejor
///   nivel ponderado por la cola CONTRARIA (micro-price de Stoikov): con más
///   volumen en el bid, el próximo movimiento probable es hacia el ask.
/// - `divergencia = (microprecio − medio) / medio`, `medio = (p_b + p_a) / 2`.
///
/// Identidad cerrada (álgebra directa): `microprecio − medio = (s/2)·obi` con
/// `s = p_a − p_b`, así que `divergencia = obi · s / (p_a + p_b)`: el
/// microprecio se aparta del medio exactamente medio spread por el OBI.
///
/// Sin volumen en el mejor nivel, con una suma de cantidades no finita o con
/// un medio no positivo devuelve `(0.0, 0.0)` (neutral): exactamente el valor
/// con que el host inicializa ambos campos en cada mensaje. Las salidas nunca
/// son NaN ni infinitas.
#[inline]
pub fn microestructura_mejor_nivel(bp: f64, bq: f64, ap: f64, aq: f64) -> (f64, f64) {
    let total_q = bq + aq;
    if !(total_q > 0.0 && total_q.is_finite()) {
        return (0.0, 0.0);
    }
    // Mismas expresiones y mismo orden que el cálculo previo del host
    // (god_engine.rs): resultado bit a bit idéntico en todo su dominio.
    let obi = (bq - aq) / total_q;
    let microprecio = (bp * aq + ap * bq) / total_q;
    let medio = (bp + ap) / 2.0;
    let divergencia = if medio > 0.0 {
        (microprecio - medio) / medio
    } else {
        0.0
    };
    (
        if obi.is_finite() { obi } else { 0.0 },
        if divergencia.is_finite() {
            divergencia
        } else {
            0.0
        },
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Fórmula previa del host, copiada literalmente de god_engine.rs
    /// (anterior a RET-45): el oráculo de la paridad bit a bit.
    fn formula_previa_del_host(bp: f64, bq: f64, ap: f64, aq: f64) -> (f64, f64) {
        let mut depth_obi = 0.0;
        let mut depth_micro_div = 0.0;
        let total_q = bq + aq;
        if total_q > 0.0 {
            depth_obi = (bq - aq) / total_q;
            let microprice = (bp * aq + ap * bq) / total_q;
            let midprice = (bp + ap) / 2.0;
            depth_micro_div = if midprice > 0.0 {
                (microprice - midprice) / midprice
            } else {
                0.0
            };
        }
        (depth_obi, depth_micro_div)
    }

    fn xorshift(estado: &mut u64) -> f64 {
        *estado ^= *estado << 13;
        *estado ^= *estado >> 7;
        *estado ^= *estado << 17;
        (*estado >> 11) as f64 / (1u64 << 53) as f64
    }

    /// Libro válido aleatorio en el dominio del parser: precios de 1e-8 a
    /// 1e6 (PEPE a BTC), spread relativo de 1e-6 a 1e-2 y cantidades de 0 a
    /// 1e7 (con ceros y cantidades sub-unidad frecuentes).
    fn libro_aleatorio(estado: &mut u64) -> (f64, f64, f64, f64) {
        fn cantidad(e: &mut u64) -> f64 {
            if xorshift(e) < 0.05 {
                0.0
            } else {
                10f64.powf(-6.0 + 13.0 * xorshift(e))
            }
        }
        let bp = 10f64.powf(-8.0 + 14.0 * xorshift(estado));
        let ap = bp * (1.0 + 10f64.powf(-6.0 + 4.0 * xorshift(estado)));
        let bq = cantidad(estado);
        let aq = cantidad(estado);
        (bp, bq, ap, aq)
    }

    /// RET-45 — paridad BIT A BIT con la fórmula previa del host en 200.000
    /// libros válidos: la unificación no cambia ni un ulp de lo que llega al
    /// núcleo (T-1 cero por construcción).
    #[test]
    fn ret45_paridad_bit_a_bit_con_la_formula_previa_del_host() {
        let mut estado = 0x9E37_79B9_7F4A_7C15u64;
        let mut con_volumen = 0usize;
        for _ in 0..200_000 {
            let (bp, bq, ap, aq) = libro_aleatorio(&mut estado);
            let nuevo = microestructura_mejor_nivel(bp, bq, ap, aq);
            let previo = formula_previa_del_host(bp, bq, ap, aq);
            assert_eq!(nuevo.0.to_bits(), previo.0.to_bits(), "obi en {bp} {bq} {ap} {aq}");
            assert_eq!(nuevo.1.to_bits(), previo.1.to_bits(), "divergencia en {bp} {bq} {ap} {aq}");
            if bq + aq > 0.0 {
                con_volumen += 1;
            }
        }
        assert!(con_volumen > 180_000, "la muestra ejercita el camino con volumen");
    }

    /// Valores cerrados: bid 100 × 3, ask 101 × 1 ⇒ obi = 2/4 = 0,5;
    /// microprecio = (100·1 + 101·3)/4 = 100,75; medio = 100,5;
    /// divergencia = 0,25/100,5.
    #[test]
    fn ret45_valores_cerrados_de_obi_y_microprecio() {
        let (obi, div) = microestructura_mejor_nivel(100.0, 3.0, 101.0, 1.0);
        assert_eq!(obi, 0.5);
        assert!((div - 0.25 / 100.5).abs() < 1e-15, "divergencia {div}");
    }

    /// La identidad cerrada del microprecio: divergencia = obi·s/(p_a + p_b).
    /// Si el microprecio pesara por la cola propia en lugar de la contraria,
    /// el signo se invertiría y esta identidad fallaría.
    #[test]
    fn ret45_el_microprecio_se_aparta_del_medio_medio_spread_por_el_obi() {
        let mut estado = 0xD1B5_4A32_D192_ED03u64;
        for _ in 0..50_000 {
            let (bp, bq, ap, aq) = libro_aleatorio(&mut estado);
            if bq + aq <= 0.0 {
                continue;
            }
            let (obi, div) = microestructura_mejor_nivel(bp, bq, ap, aq);
            assert!((-1.0..=1.0).contains(&obi), "obi fuera de [-1, 1]: {obi}");
            let esperado = obi * (ap - bp) / (ap + bp);
            // Cota del error de redondeo: `microprecio − medio` cancela, con
            // error absoluto de unos pocos ε en la divergencia (relativa al
            // medio); `ap − bp` es exacta por el lema de Sterbenz.
            let tolerancia = 1e-12 * esperado.abs() + 8.0 * f64::EPSILON;
            assert!((div - esperado).abs() <= tolerancia, "div {div} vs {esperado}");
            if esperado.abs() > tolerancia {
                assert_eq!(div.signum(), obi.signum(), "más volumen en el bid empuja hacia el ask");
            }
        }
    }

    /// Antisimetría: intercambiar las cantidades de los dos lados invierte el
    /// signo del OBI y el de la divergencia.
    #[test]
    fn ret45_intercambiar_lados_invierte_el_signo() {
        let (o1, d1) = microestructura_mejor_nivel(50.0, 7.0, 50.5, 2.0);
        let (o2, d2) = microestructura_mejor_nivel(50.0, 2.0, 50.5, 7.0);
        assert_eq!(o1, -o2);
        assert!((d1 + d2).abs() < 1e-14, "divergencias opuestas: {d1} {d2}");
        assert_eq!(microestructura_mejor_nivel(50.0, 4.0, 50.5, 4.0).0, 0.0, "libro equilibrado");
    }

    /// Neutral y finito ante lo que el parser no debería dejar pasar.
    #[test]
    fn ret45_neutral_sin_volumen_y_ante_entradas_no_finitas() {
        assert_eq!(microestructura_mejor_nivel(10.0, 0.0, 10.1, 0.0), (0.0, 0.0));
        for (bp, bq, ap, aq) in [
            (10.0, f64::NAN, 10.1, 1.0),
            (10.0, 1.0, 10.1, f64::INFINITY),
            (10.0, f64::MAX, 10.1, f64::MAX),
            (f64::NAN, 1.0, 10.1, 1.0),
            (10.0, 1.0, f64::INFINITY, 1.0),
            (-10.0, 1.0, -9.0, 1.0),
        ] {
            let (o, d) = microestructura_mejor_nivel(bp, bq, ap, aq);
            assert!(o.is_finite() && d.is_finite(), "salida no finita con {bp} {bq} {ap} {aq}");
        }
    }

    /// Conexión real y fuga erradicada: el host calcula el OBI y el
    /// microprecio con ESTA función y ya no guarda ningún libro por símbolo.
    #[test]
    fn ret45_el_host_usa_la_funcion_y_no_acumula_libros() {
        let host: String = include_str!("bin/god_engine.rs")
            .chars()
            .filter(|c| !c.is_whitespace())
            .collect();
        assert!(
            host.contains("quantum_engine::orderbook::microestructura_mejor_nivel(bp,bq,ap,aq)"),
            "el host deriva depth_obi y depth_micro_div de la función unificada"
        );
        for prohibido in ["local_orderbooks", "orderbook::OrderBook", "BTreeMap<OrderedFloat"] {
            assert!(!host.contains(prohibido), "el host no debe contener {prohibido}");
        }
        assert!(
            !host.contains("depth_obi=(bq-aq)/total_q"),
            "una sola fórmula: sin copia en línea en el host"
        );
    }
}
