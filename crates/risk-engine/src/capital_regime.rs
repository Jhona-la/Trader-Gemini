//! RÉGIMEN DE CAPITAL CONTINUO (D-641 completo — DÉCIMA OLA).
//!
//! # El problema
//!
//! El censo de la auditoría encontró ~24 sitios con umbrales literales de
//! capital repartidos por nueve archivos: `<= 15.0`, `<= 20.0`, `< 30.0`,
//! `<= 50.0`, `< 100.0`, `< 300.0`. En cada uno el comportamiento del sistema
//! cambiaba de golpe al cruzar el umbral. Los peores:
//!
//! * `leverage_matrix`: el techo de apalancamiento saltaba de **4× a ~40×**
//!   entre $20,00 y $20,01.
//! * `evaluate_quantum_order`: el cortacircuitos de drawdown usaba 0,85 fijo
//!   por debajo de $15 e ignoraba el gen `global_max_drawdown`.
//! * `guard::check_drawdown_limit`: 0,75 fijo por debajo de $50.
//!
//! Y el efecto sistémico: la evolución corre con capital de backtest y la
//! producción con ~$13, de modo que ambos entornos ejecutaban **ramas de
//! código distintas**. Ningún genoma se evaluaba contra las reglas que lo
//! gobiernan en vivo.
//!
//! La primera corrección de D-641 cubrió 3 de los sitios con una transición
//! `1/(1 + N/3)` que, además, cambiaba el comportamiento a $13 (aplicaba sólo
//! el 54 % del régimen micro que se había diseñado para esa cuenta).
//!
//! # La variable física
//!
//! Los umbrales no eran arbitrarios en dólares: eran **múltiplos del notional
//! mínimo del exchange** ($5). $15 = 3 operaciones mínimas, $50 = 10. El
//! régimen micro existe porque, con poca holgura, el tamaño mínimo obliga a
//! comprometer una fracción grande del capital en cada operación.
//!
//! ```text
//! N = capital / notional_mínimo        (operaciones mínimas que caben)
//! ```
//!
//! Expresar las reglas sobre `N` hace que el régimen se desplace solo si el
//! exchange cambia su mínimo, en lugar de quedar anclado a dólares.
//!
//! # La transición
//!
//! Peso del régimen micro `w ∈ [0, 1]`, C¹ en `ln N`:
//!
//! ```text
//! t = (ln N − ln 3) / (ln 10 − ln 3)
//! w = 1 − smoothstep(t),   smoothstep(t) = 3t² − 2t³  con t acotado a [0,1]
//! ```
//!
//! * `N ≤ 3`  ⇒ `w = 1` exactamente: el régimen micro pleno, **tal como se
//!   diseñó**. Una cuenta de $13 (N = 2,6) conserva su comportamiento.
//! * `N ≥ 10` ⇒ `w = 0` exactamente: el régimen estándar pleno.
//! * Entre ambos, transición suave sin saltos.
//!
//! Los extremos 3 y 10 no son parámetros nuevos: son los puntos en que ya
//! estaban calibradas las reglas existentes ($15 y $50 con notional $5). Lo
//! único que cambia es que la frontera deja de ser un escalón.
//!
//! **Lo que sí cambia de comportamiento** es la franja entre 3 y 10
//! operaciones mínimas ($15–$50): las reglas calibradas en $15 se extienden
//! parcialmente hacia $50, y las calibradas en $50 se atenúan desde $15.
//! Producción a $13 no está en esa franja.

/// Operaciones mínimas por debajo de las cuales rige el régimen micro pleno.
/// Corresponde al antiguo umbral de $15 con notional mínimo de $5.
pub const MICRO_FULL_ROOM: f64 = 3.0;

/// Operaciones mínimas a partir de las cuales rige el régimen estándar pleno.
/// Corresponde al antiguo umbral de $50 con notional mínimo de $5.
pub const STANDARD_FULL_ROOM: f64 = 10.0;

/// Notional mínimo que el resto del sistema asume para Binance USDT-M cuando
/// no se dispone del valor real del símbolo.
pub const DEFAULT_MIN_NOTIONAL: f64 = 5.0;

/// Operaciones de tamaño mínimo que caben en el capital.
///
/// Entradas no finitas o no positivas se tratan como la cuenta más pequeña
/// posible (`N = 0`), coherente con la sanitización existente del sistema,
/// que sustituye capitales corruptos por $13.
#[inline]
pub fn trades_of_room(capital: f64, min_notional: f64) -> f64 {
    let notional = if min_notional.is_finite() && min_notional > 0.0 {
        min_notional
    } else {
        DEFAULT_MIN_NOTIONAL
    };
    if !capital.is_finite() || capital <= 0.0 {
        return 0.0;
    }
    capital / notional
}

/// Peso del régimen micro en `[0, 1]`. 1 = micro pleno, 0 = estándar pleno.
#[inline]
pub fn micro_weight(capital: f64, min_notional: f64) -> f64 {
    let n = trades_of_room(capital, min_notional);
    if n <= MICRO_FULL_ROOM {
        return 1.0;
    }
    if n >= STANDARD_FULL_ROOM {
        return 0.0;
    }
    let t = ((n.ln() - MICRO_FULL_ROOM.ln())
        / (STANDARD_FULL_ROOM.ln() - MICRO_FULL_ROOM.ln()))
    .clamp(0.0, 1.0);
    let smooth = t * t * (3.0 - 2.0 * t);
    (1.0 - smooth).clamp(0.0, 1.0)
}

/// Interpolación lineal entre el valor estándar y el micro según `w`.
/// Adecuada para fracciones, umbrales y probabilidades.
#[inline]
pub fn lerp(standard: f64, micro: f64, w: f64) -> f64 {
    let w = if w.is_finite() { w.clamp(0.0, 1.0) } else { 1.0 };
    standard * (1.0 - w) + micro * w
}

/// Interpolación geométrica entre el valor estándar y el micro según `w`.
///
/// Adecuada para magnitudes multiplicativas —apalancamiento, factores de
/// escala—, donde el punto medio natural entre 4× y 36× es 12×, no 20×. Si
/// alguno de los extremos no es positivo cae a la interpolación lineal.
#[inline]
pub fn log_lerp(standard: f64, micro: f64, w: f64) -> f64 {
    if !(standard > 0.0 && micro > 0.0) {
        return lerp(standard, micro, w);
    }
    let w = if w.is_finite() { w.clamp(0.0, 1.0) } else { 1.0 };
    (standard.ln() * (1.0 - w) + micro.ln() * w).exp()
}

/// Fracción máxima del capital comprometible como margen. Deja el 2 % para
/// comisiones de salida y redondeos del exchange; es el valor que el sistema ya
/// usaba como máximo operativo en régimen micro.
pub const MAX_MARGIN_UTILIZATION: f64 = 0.98;

/// Colchón mínimo que un genoma puede fijar: la mitad del capital.
pub const MIN_MARGIN_UTILIZATION: f64 = 0.50;

/// Colchón que se aplica si el gen no es un número utilizable.
pub const FALLBACK_MARGIN_UTILIZATION: f64 = 0.80;

/// D-634/D-635 (DÉCIMA OLA): FUENTE ÚNICA del colchón de margen. El gen
/// `margin_cushion_pct` fija la fracción del capital comprometible; la escasez
/// de capital la relaja de forma continua hacia el máximo operativo. La usan el
/// risk-engine al validar la orden y el núcleo al comprobar el margen libre.
#[inline]
pub fn margin_cushion(gene: f64, scarcity: f64) -> f64 {
    let genomic = if gene.is_finite() && gene > 0.0 {
        gene.clamp(MIN_MARGIN_UTILIZATION, MAX_MARGIN_UTILIZATION)
    } else {
        FALLBACK_MARGIN_UTILIZATION
    };
    lerp(genomic, MAX_MARGIN_UTILIZATION, scarcity)
        .clamp(MIN_MARGIN_UTILIZATION, MAX_MARGIN_UTILIZATION)
}

/// D-635: notional mínimo efectivo de un símbolo. El del exchange cuando se
/// conoce, nunca por debajo del mínimo universal de Binance USDⓈ-M.
#[inline]
pub fn effective_min_notional(spec_min_notional: f64) -> f64 {
    if spec_min_notional.is_finite() && spec_min_notional > 0.0 {
        spec_min_notional.max(DEFAULT_MIN_NOTIONAL)
    } else {
        DEFAULT_MIN_NOTIONAL
    }
}

/// D-750 — VIABILIDAD DE LA ORDEN: FUENTE ÚNICA.
///
/// # Qué estaba mal
///
/// El dimensionado micro llevaba un «piso de viabilidad» expresado como
/// FRACCIÓN DE APUESTA:
///
/// ```text
///   micro_min_viable = (min_notional · 5 / capital).clamp(0, 0,10)
/// ```
///
/// Con 13 $ de capital y un mínimo de 5 $ el cálculo daba 1,92, se recortaba a
/// 0,10 y ese 0,10 se convertía en el límite INFERIOR del Kelly: el 10 % del
/// capital apostado aunque el edge medido fuese nulo. El `· 5` era además un
/// apalancamiento escrito a mano dentro de una fórmula que hablaba de
/// fracciones.
///
/// # Qué es la viabilidad
///
/// Dos magnitudes distintas, ninguna de ellas una fracción de apuesta:
///
/// * **Margen mínimo**: `min_notional / L`. Es el capital que hay que
///   inmovilizar para que la orden alcance el nocional mínimo del símbolo al
///   apalancamiento que se va a usar. Es una restricción de EJECUCIÓN.
/// * **Riesgo mínimo**: `min_notional · SL / capital`. Es la fracción del
///   capital que se pierde si el stop de la orden MÁS PEQUEÑA que el símbolo
///   acepta se toca. Es una restricción de SUPERVIVENCIA — y nótese que el
///   apalancamiento NO aparece: reparte el mismo nocional entre margen y
///   préstamo, pero no cambia lo que se pierde. Por eso «subir el
///   apalancamiento para que quepa» no hace viable nada.
///
/// Si el riesgo mínimo excede el tope de riesgo por evento del sistema, la
/// orden es inviable y debe RECHAZARSE.
#[inline]
pub fn margen_minimo_viable(min_notional: f64, apalancamiento: f64) -> f64 {
    let mn = effective_min_notional(min_notional);
    let l = if apalancamiento.is_finite() && apalancamiento >= 1.0 {
        apalancamiento
    } else {
        1.0
    };
    mn / l
}

/// Fracción del capital que arriesga la orden más pequeña que el símbolo
/// acepta. `f64::INFINITY` si el capital no es utilizable: sin capital no hay
/// orden viable.
#[inline]
pub fn riesgo_minimo_viable(min_notional: f64, sl_pct: f64, capital: f64) -> f64 {
    if !capital.is_finite() || capital <= 0.0 {
        return f64::INFINITY;
    }
    if !sl_pct.is_finite() || sl_pct <= 0.0 {
        return f64::INFINITY;
    }
    effective_min_notional(min_notional) * sl_pct / capital
}

/// ¿Cabe la orden mínima del símbolo dentro del tope de riesgo por evento?
#[inline]
pub fn orden_viable(min_notional: f64, sl_pct: f64, capital: f64, tope_riesgo: f64) -> bool {
    if !tope_riesgo.is_finite() || tope_riesgo <= 0.0 {
        return false;
    }
    riesgo_minimo_viable(min_notional, sl_pct, capital) <= tope_riesgo
}

#[cfg(test)]
mod tests {
    use super::*;

    /// La cuenta de producción conserva EXACTAMENTE su régimen diseñado.
    #[test]
    fn la_cuenta_de_13_dolares_conserva_el_regimen_micro_pleno() {
        assert_eq!(micro_weight(13.0, 5.0), 1.0);
        assert_eq!(micro_weight(15.0, 5.0), 1.0);
    }

    #[test]
    fn las_cuentas_grandes_estan_en_regimen_estandar_pleno() {
        assert_eq!(micro_weight(50.0, 5.0), 0.0);
        assert_eq!(micro_weight(10_000.0, 5.0), 0.0);
    }

    /// El núcleo de D-641: ningún centavo produce un salto de comportamiento.
    #[test]
    fn no_hay_saltos_entre_centavos_consecutivos() {
        let mut prev = micro_weight(5.0, 5.0);
        let mut worst = 0.0f64;
        let mut c = 5.01;
        while c <= 200.0 {
            let w = micro_weight(c, 5.0);
            worst = worst.max((w - prev).abs());
            prev = w;
            c += 0.01;
        }
        assert!(
            worst < 5e-4,
            "un paso de 1 centavo cambió el peso en {worst}: sigue habiendo un acantilado"
        );
    }

    #[test]
    fn el_peso_decrece_monotonamente_con_el_capital() {
        let mut prev = 1.0;
        for i in 0..2_000 {
            let c = 1.0 + i as f64 * 0.1;
            let w = micro_weight(c, 5.0);
            assert!(w <= prev + 1e-12, "no monótono en ${c}: {prev} -> {w}");
            prev = w;
        }
    }

    /// El régimen se expresa en operaciones mínimas, no en dólares: si el
    /// exchange multiplica su mínimo por 10, la frontera se mueve con él.
    #[test]
    fn el_regimen_escala_con_el_notional_minimo() {
        for &(c, m) in &[(13.0, 5.0), (20.0, 5.0), (30.0, 5.0), (45.0, 5.0)] {
            let a = micro_weight(c, m);
            let b = micro_weight(c * 10.0, m * 10.0);
            assert!((a - b).abs() < 1e-12, "${c}/${m}: {a} vs {b}");
        }
    }

    /// Capital corrupto se trata como la cuenta mínima, igual que la
    /// sanitización existente del sistema.
    #[test]
    fn entradas_no_finitas_se_tratan_como_cuenta_minima() {
        assert_eq!(micro_weight(f64::NAN, 5.0), 1.0);
        assert_eq!(micro_weight(-10.0, 5.0), 1.0);
        assert_eq!(micro_weight(13.0, f64::NAN), 1.0);
    }

    #[test]
    fn el_colchon_sale_del_gen_y_la_escasez_lo_relaja() {
        assert!((margin_cushion(0.70, 0.0) - 0.70).abs() < 1e-12);
        assert!((margin_cushion(0.70, 1.0) - MAX_MARGIN_UTILIZATION).abs() < 1e-12);
        assert!((margin_cushion(1.09, 0.0) - MAX_MARGIN_UTILIZATION).abs() < 1e-12);
        assert!((margin_cushion(f64::NAN, 0.0) - FALLBACK_MARGIN_UTILIZATION).abs() < 1e-12);
    }

    #[test]
    fn el_notional_minimo_respeta_el_del_exchange() {
        assert_eq!(effective_min_notional(100.0), 100.0);
        assert_eq!(effective_min_notional(1.0), DEFAULT_MIN_NOTIONAL);
        assert_eq!(effective_min_notional(f64::NAN), DEFAULT_MIN_NOTIONAL);
    }

    /// EL DEFECTO (D-750): el piso de viabilidad era una fracción de apuesta
    /// que, con la cuenta de producción, valía exactamente 0,10 — el 10 % del
    /// capital apostado sin edge. La viabilidad no es eso: es el margen que
    /// alcanza el nocional mínimo, y el riesgo que ese nocional mínimo toma.
    #[test]
    fn el_margen_minimo_sale_del_nocional_y_del_apalancamiento() {
        // 5 $ de nocional mínimo a 5× necesitan 1 $ de margen; a 1×, 5 $.
        assert!((margen_minimo_viable(5.0, 5.0) - 1.0).abs() < 1e-12);
        assert!((margen_minimo_viable(5.0, 1.0) - 5.0).abs() < 1e-12);
        // El mínimo del símbolo manda sobre el universal del exchange.
        assert!((margen_minimo_viable(20.0, 4.0) - 5.0).abs() < 1e-12);
        // Apalancamiento corrupto ⇒ el caso más exigente (1×).
        assert!((margen_minimo_viable(5.0, f64::NAN) - 5.0).abs() < 1e-12);
    }

    /// El apalancamiento NO aparece en el riesgo: subirlo no hace viable nada.
    #[test]
    fn el_riesgo_minimo_no_depende_del_apalancamiento() {
        let r = riesgo_minimo_viable(5.0, 0.01, 13.0);
        assert!((r - (5.0 * 0.01 / 13.0)).abs() < 1e-12, "{r}");
        assert!(r.is_finite());
        // Sin capital o sin stop no hay orden viable que evaluar.
        assert!(riesgo_minimo_viable(5.0, 0.01, 0.0).is_infinite());
        assert!(riesgo_minimo_viable(5.0, 0.0, 13.0).is_infinite());
    }

    /// Si ni la orden mínima del símbolo cabe en el tope de riesgo, la
    /// respuesta correcta es NO OPERAR — jamás inflar la apuesta.
    #[test]
    fn una_orden_que_no_cabe_en_el_tope_de_riesgo_es_inviable() {
        // Cuenta de 13 $, símbolo con mínimo de 100 $ y stop del 5 %:
        // la orden mínima arriesga 5 $ = 38 % del capital. Tope del 25 %.
        assert!(!orden_viable(100.0, 0.05, 13.0, 0.25));
        // El mismo símbolo con un stop del 0,5 % arriesga el 3,8 %: cabe.
        assert!(orden_viable(100.0, 0.005, 13.0, 0.25));
        // Tope degenerado ⇒ nada es viable.
        assert!(!orden_viable(5.0, 0.01, 13.0, 0.0));
    }

    #[test]
    fn interpolaciones_respetan_los_extremos() {
        assert_eq!(lerp(0.62, 0.66, 1.0), 0.66);
        assert_eq!(lerp(0.62, 0.66, 0.0), 0.62);
        assert!((log_lerp(36.0, 4.0, 1.0) - 4.0).abs() < 1e-12);
        assert!((log_lerp(36.0, 4.0, 0.0) - 36.0).abs() < 1e-12);
        // El punto medio geométrico entre 4× y 36× es 12×.
        assert!((log_lerp(36.0, 4.0, 0.5) - 12.0).abs() < 1e-9);
    }
}
