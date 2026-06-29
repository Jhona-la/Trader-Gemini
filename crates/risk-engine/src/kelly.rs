/// Fórmula Dinámica de Kelly para Supervivencia Absoluta
/// Kelly = W - [(1 - W) / R]
/// donde W = Probabilidad de acierto (Win Rate), R = Profit Factor (Beneficio / Riesgo)

/// Retorna el porcentaje del capital que debe arriesgarse. 
/// Utiliza 'Half-Kelly' (dividir entre 2) para proteger contra la varianza matemática (cisnes negros).
#[inline(always)]
pub fn calculate_kelly_fraction(win_rate: f64, profit_factor: f64) -> f64 {
    // Protección contra divisiones por cero o sistemas perdedores
    if profit_factor <= 0.0 || win_rate < 0.01 {
        return 0.0;
    }

    let kelly = win_rate - ((1.0 - win_rate) / profit_factor);

    // Si el Kelly es negativo, la esperanza matemática es perdedora.
    if kelly <= 0.0 {
        return 0.0;
    }

    // Half-Kelly para control de drawdown masivo
    let half_kelly = kelly / 2.0;

    // Límite duro absoluto: Jamás apostar más del 20% del capital real en un solo trade
    if half_kelly > 0.20 {
        0.20
    } else {
        half_kelly
    }
}
