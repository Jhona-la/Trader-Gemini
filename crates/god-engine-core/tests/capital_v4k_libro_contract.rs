//! CAPITAL-UNICO V4K, A5 (dictamen V4J :37, criterios :16): suelo, relevo y aislamiento del libro
//! por cuenta, sobre el libro de produccion bajo la puerta del arena real. Fixture del contrato
//! V4J `relevo_no_presta_marca_al_debito_retenido` (cuenta 7, T500, relevo T300, callback T400,
//! cargo pendiente sin marca de 0,5 y cargo T400 de 0,25), sin saltar a la foto T600: intercala
//! las fotos T300/T400 del relevo, pasa por otra cuenta y vuelve a la original.
use quantum_arena::dimensionable::{ClaveReserva, LibroDimensionable};
use quantum_arena::GlobalArena;
use std::sync::atomic::Ordering;
use std::sync::MutexGuard;

const A: ClaveReserva = ClaveReserva { coin_id: 0, slot: 0, generation: 1 };
const B: ClaveReserva = ClaveReserva { coin_id: 0, slot: 1, generation: 1 };

fn libro(arena: &GlobalArena) -> MutexGuard<'_, LibroDimensionable> {
    arena.config.entry_publication_gate.lock().unwrap()
}

fn dim(arena: &GlobalArena) -> f64 {
    arena.unified_capital.load(Ordering::Relaxed)
}

#[test]
fn relevo_intercalado_y_vuelta_a_la_cuenta_original() {
    let a = GlobalArena::build_in_own_stack(0.0);
    let mut l = libro(&a);
    l.activar(&a);
    l.fijar_cuenta(&a, 7);
    l.fijar_alcance(1, 1);
    l.saldo_aceptado(&a, 100.0, Some(500));
    l.reservar(&a, A, 0.5);
    l.vincular_orden(A, "cL_a");
    l.fill_de_entrada(&a, 0, "cL_a", 9, Some(0.5), None);
    l.ejecutar(&a, A, 0.0, None);
    assert_eq!(l.reservado(), 0.5, "fixture: cargo sin marca");
    // Relevo de la misma cuenta: foto T300, callback T400, foto T400 y una T350 rezagada.
    l.fijar_alcance(2, 2);
    l.saldo_aceptado(&a, 99.5, Some(300));
    assert_eq!(l.reservado(), 0.5, "T300 presto la marca T500 al cargo sin marca");
    l.fill_de_cierre(&a, 0, 77, 0.25, 400, true);
    assert_eq!(l.reservado(), 0.75, "fixture: cargo T400 retenido");
    l.saldo_aceptado(&a, 99.5, Some(400));
    assert_eq!(l.reservado(), 0.75, "la foto T400 (igual, no posterior) incluyo el cargo T400 o el sin marca");
    l.saldo_aceptado(&a, 99.5, Some(350));
    assert_eq!(l.reservado(), 0.75, "una foto rezagada T350 movio el libro");
    assert_eq!(dim(&a), 99.5 - 0.75, "capacidad de la cuenta 7");

    // Otra cuenta: no soporta los debitos de la 7, ni su foto (marca muy posterior) los libera.
    l.fijar_cuenta(&a, 8);
    assert!(l.saldo().is_none(), "la cuenta 8 heredo el saldo de la 7");
    l.saldo_aceptado(&a, 50.0, Some(10_000));
    assert_eq!(l.reservado(), 0.0, "la cuenta 8 soporta debitos de la 7");
    assert_eq!(dim(&a), 50.0, "capacidad de la cuenta 8 con debitos ajenos");
    assert_eq!(l.reservado_de_otras_cuentas(), 0.75, "la foto de la cuenta 8 libero debitos de la 7");
    // El mismo trade 77 en la cuenta 8 es otro cargo (identidad aislada).
    l.fill_de_cierre(&a, 0, 77, 0.25, 10_100, true);
    assert_eq!(l.reservado(), 0.25, "el trade 77 de la cuenta 8 se dedujo contra el de la 7");

    // Vuelta a la 7: identidad y obligacion conservadas. Foto T390 (anterior al cargo T400).
    l.fijar_cuenta(&a, 7);
    assert!(l.saldo().is_none(), "la vuelta heredo el saldo de la 8");
    l.saldo_aceptado(&a, 99.5, Some(390));
    assert_eq!(l.reservado(), 0.75, "la vuelta a la cuenta 7 perdio su obligacion");
    assert_eq!(dim(&a), 99.5 - 0.75);
    // Replay del cargo T400 en la 7: una sola vez.
    l.fill_de_cierre(&a, 0, 77, 0.25, 400, true);
    assert_eq!(l.reservado(), 0.75, "replay en la cuenta 7");
    // Foto posterior de la 7: incluye el cargo T400 (su marca), nunca el sin marca.
    l.saldo_aceptado(&a, 99.25, Some(800));
    assert_eq!(l.reservado(), 0.5, "la foto T800 de la 7 no incluyo el T400 o acredito el sin marca");
    assert_eq!(l.reservado_de_otras_cuentas(), 0.25, "el cargo de la 8 salio del libro");
    // ABA de revision tras la vuelta: mismo importe no revalida la revision anterior.
    let capital = l.dimensionable();
    l.reservar_con_base(&a, B, 0.0, capital);
    l.saldo_aceptado(&a, 90.0, Some(810));
    l.saldo_aceptado(&a, 99.25, Some(820));
    assert_eq!(l.dimensionable(), capital, "fixture: mismo importe");
    assert!(!l.envio_valido(B), "el mismo importe revalido la revision anterior");
}
