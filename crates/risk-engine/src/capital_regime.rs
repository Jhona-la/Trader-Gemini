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
    fn interpolaciones_respetan_los_extremos() {
        assert_eq!(lerp(0.62, 0.66, 1.0), 0.66);
        assert_eq!(lerp(0.62, 0.66, 0.0), 0.62);
        assert!((log_lerp(36.0, 4.0, 1.0) - 4.0).abs() < 1e-12);
        assert!((log_lerp(36.0, 4.0, 0.0) - 36.0).abs() < 1e-12);
        // El punto medio geométrico entre 4× y 36× es 12×.
        assert!((log_lerp(36.0, 4.0, 0.5) - 12.0).abs() < 1e-9);
    }
}
