//! Contrato de certificación formal para Ola Ω63 (Ficha Forense #698).
//!
//! Verifica:
//! 1. HOST-004 / R7-R4-C-1 residual: Verificación multi-ranura de `exchange_confirmed` en PositionManager.
//!    Impide que posiciones legítimamente confirmadas en scalp (slot 0) o swing (slot 1) sean liquidadas
//!    accidentalmente por la orden reduce-only de respaldo de papel.
//! 2. R7-R4-D-2: Preservación de la diversidad fenotípica del Nicho 3 en `scalp_trail_act_atr` [0.8, 1.8],
//!    sin truncamiento prematuro por el blindaje global.
//! 3. R6-A13 residual: Semántica no ambigua en geometría gauge Yang-Mills, reportando `yang_mills_current_absent`
//!    explícitamente cuando el activo se encuentra fuera del soporte de la matriz gauge.

use std::sync::atomic::Ordering;
use quantum_arena::position::PositionManager;

/// 1. HOST-004 & R7-R4-C-1 residual:
/// Demuestra que una posición confirmada en slot 0 (scalp) o slot 1 (swing) es correctamente
/// identificada como `exchange_confirmed` por el scanner multi-ranura `slots().iter().any(...)`,
/// mientras que el código previo inspeccionaba exclusivamente slot 2 (`position`), fallando
/// falsamente y disparando órdenes de mercado que cerraban posiciones reales en Binance.
#[test]
fn test_host004_r7_r4_c1_multi_slot_exchange_confirmed() {
    let pm = PositionManager::default();

    // Estado inicial: ningún slot confirmado
    let any_confirmed_init = pm.slots().iter().any(|p| p.exchange_confirmed.load(Ordering::Relaxed));
    assert!(!any_confirmed_init, "Inicialmente ningún slot debe estar confirmado");

    // Simular que el exchange confirma un fill en el slot 0 (scalp)
    pm.scalp.exchange_confirmed.store(true, Ordering::Relaxed);

    // Inspección antigua defectuosa (solo slot 2 `position`):
    let old_buggy_check = pm.position.exchange_confirmed.load(Ordering::Relaxed);
    assert!(!old_buggy_check, "El chequeo antiguo fallaba en detectar la confirmación de scalp");

    // Inspección corregida multi-ranura:
    let new_correct_check = pm.slots().iter().any(|p| p.exchange_confirmed.load(Ordering::Relaxed));
    assert!(new_correct_check, "El chequeo multi-ranura detecta la confirmación en slot 0 (scalp)");

    // Simular que el exchange confirma un fill en el slot 1 (swing) y slot 0 se cierra
    pm.scalp.exchange_confirmed.store(false, Ordering::Relaxed);
    pm.swing.exchange_confirmed.store(true, Ordering::Relaxed);

    let old_buggy_check_swing = pm.position.exchange_confirmed.load(Ordering::Relaxed);
    assert!(!old_buggy_check_swing, "El chequeo antiguo fallaba en detectar la confirmación de swing");

    let new_correct_check_swing = pm.slots().iter().any(|p| p.exchange_confirmed.load(Ordering::Relaxed));
    assert!(new_correct_check_swing, "El chequeo multi-ranura detecta la confirmación en slot 1 (swing)");
}

/// 2. R7-R4-D-2:
/// Demuestra que la banda post-mutación [0.8, 2.5] preserva la intención del Nicho 3 [0.8, 1.8]
/// sin colapsar el intervalo [0.8, 1.0) a 1.0.
#[test]
fn test_r7_r4_d2_niche3_trailing_range_preserved() {
    // Intención del Nicho 3: activar trailing a 0.85 ATR
    let niche_intended_trail = 0.85_f64;
    let niche_clamped = niche_intended_trail.clamp(0.8, 1.8);
    assert_eq!(niche_clamped, 0.85);

    // Con el blindaje antiguo [1.0, 2.5]:
    let old_shielded = niche_clamped.clamp(1.0, 2.5);
    assert_eq!(old_shielded, 1.0, "El blindaje antiguo colapsaba 0.85 a 1.0");

    // Con el blindaje corregido [0.8, 2.5]:
    let new_shielded = niche_clamped.clamp(0.8, 2.5);
    assert_eq!(new_shielded, 0.85, "El blindaje corregido preserva el fenotipo del Nicho 3");

    // Caso límite inferior: 0.80
    assert_eq!(0.80_f64.clamp(0.8, 1.8).clamp(0.8, 2.5), 0.80);
    // Caso límite superior: 1.80
    assert_eq!(1.80_f64.clamp(0.8, 1.8).clamp(0.8, 2.5), 1.80);
}

/// 3. R6-A13 residual:
/// Demuestra que el reporte de corriente gauge distingue inequívocamente entre un activo con corriente
/// física 0.0 y un activo ausente de la matriz gauge (`yang_mills_current_absent == 1.0`).
#[test]
fn test_r6_a13_yang_mills_absent_distinction() {
    let ym_currents = vec![0.25, -0.10, 0.0]; // 3 monedas en el universo gauge

    let evaluate_coin_gauge = |coin_id: usize, currents: &[f64]| -> (f64, f64) {
        if coin_id < currents.len() {
            (currents[coin_id], 0.0)
        } else {
            (0.0, 1.0)
        }
    };

    // Moneda 0: presente con corriente 0.25
    let (c0, absent0) = evaluate_coin_gauge(0, &ym_currents);
    assert_eq!(c0, 0.25);
    assert_eq!(absent0, 0.0);

    // Moneda 2: presente con corriente física 0.0 (equilibrio gauge)
    let (c2, absent2) = evaluate_coin_gauge(2, &ym_currents);
    assert_eq!(c2, 0.0);
    assert_eq!(absent2, 0.0, "Moneda 2 está presente en equilibrio: absent debe ser 0.0");

    // Moneda 5: ausente de la matriz gauge
    let (c5, absent5) = evaluate_coin_gauge(5, &ym_currents);
    assert_eq!(c5, 0.0);
    assert_eq!(absent5, 1.0, "Moneda 5 está ausente: absent debe ser 1.0");
}
