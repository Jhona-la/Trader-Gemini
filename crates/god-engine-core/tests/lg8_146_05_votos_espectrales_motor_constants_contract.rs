//! CONTRATO FORMAL: LG8-146-05 (Línea G Ola 8 / Ola Ω151)
//!
//! Anclaje estricto de constantes MOTOR_* en el ensamble de 15 motores de `votos_espectrales`.
//!
//! Previene que un cambio o reordenamiento silencioso en `lib.rs` desfase los votos
//! respecto a las series temporales de habilidad prequential e-proceso de `SkillMotores`.

use signal_engine::skill_motores::{
    MOTOR_COAXIAL, MOTOR_CONFLUENCIA, MOTOR_CONFORMAL, MOTOR_ENTROPIA, MOTOR_FEYNMAN,
    MOTOR_FLOW, MOTOR_HAWKES, MOTOR_NASH, MOTOR_NAVIER_STOKES, MOTOR_PERCEPTRON,
    MOTOR_RESONANCIA, MOTOR_SHOCK, MOTOR_SOLITON, MOTOR_SOMBRA, MOTOR_TREND_RUNNER,
    MOTORES, NOMBRES_MOTORES,
};
use signal_engine::voto_espectral::VotoEspectral;

#[test]
fn contract_lg8_146_05_motor_constants_bijective_and_ordered() {
    assert_eq!(MOTORES, 15, "El ensamble espectral integral debe constar de 15 motores");

    let indices = [
        MOTOR_SOMBRA,
        MOTOR_SOLITON,
        MOTOR_SHOCK,
        MOTOR_RESONANCIA,
        MOTOR_COAXIAL,
        MOTOR_HAWKES,
        MOTOR_NASH,
        MOTOR_FLOW,
        MOTOR_PERCEPTRON,
        MOTOR_CONFORMAL,
        MOTOR_CONFLUENCIA,
        MOTOR_TREND_RUNNER,
        MOTOR_ENTROPIA,
        MOTOR_NAVIER_STOKES,
        MOTOR_FEYNMAN,
    ];

    // Verificar que los 15 índices son estrictamente 0..15 en orden
    for (i, &idx) in indices.iter().enumerate() {
        assert_eq!(idx, i, "El motor en la posición {i} debe tener índice constante {i}");
    }

    // Verificar nombres canónicos
    assert_eq!(NOMBRES_MOTORES[MOTOR_SOMBRA], "sombra");
    assert_eq!(NOMBRES_MOTORES[MOTOR_SOLITON], "soliton");
    assert_eq!(NOMBRES_MOTORES[MOTOR_SHOCK], "shock");
    assert_eq!(NOMBRES_MOTORES[MOTOR_RESONANCIA], "resonancia");
    assert_eq!(NOMBRES_MOTORES[MOTOR_COAXIAL], "coaxial");
    assert_eq!(NOMBRES_MOTORES[MOTOR_HAWKES], "hawkes");
    assert_eq!(NOMBRES_MOTORES[MOTOR_NASH], "nash");
    assert_eq!(NOMBRES_MOTORES[MOTOR_FLOW], "flow");
    assert_eq!(NOMBRES_MOTORES[MOTOR_PERCEPTRON], "perceptron");
    assert_eq!(NOMBRES_MOTORES[MOTOR_CONFORMAL], "conformal");
    assert_eq!(NOMBRES_MOTORES[MOTOR_CONFLUENCIA], "confluencia");
    assert_eq!(NOMBRES_MOTORES[MOTOR_TREND_RUNNER], "trend_runner");
    assert_eq!(NOMBRES_MOTORES[MOTOR_ENTROPIA], "entropia");
    assert_eq!(NOMBRES_MOTORES[MOTOR_NAVIER_STOKES], "navier_stokes");
    assert_eq!(NOMBRES_MOTORES[MOTOR_FEYNMAN], "feynman");

    // Verificar asignación directa por constante idéntica a la que realiza lib.rs
    let mut votos = [VotoEspectral::default(); MOTORES];
    votos[MOTOR_NAVIER_STOKES] = VotoEspectral::desde_escalar(0.77);
    votos[MOTOR_FEYNMAN] = VotoEspectral::desde_escalar(-0.33);

    assert_eq!(votos[13].dominante().unwrap().1, 0.77);
    assert_eq!(votos[14].dominante().unwrap().1, -0.33);
}
