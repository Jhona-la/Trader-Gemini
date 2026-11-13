//! CAPITAL-UNICO V4J, punto 3 (A3) y el control A5 de `qra_v4j_criterios.md`, sobre el libro del
//! dimensionable de produccion bajo la puerta del arena real.
use quantum_arena::dimensionable::{ClaveReserva, ComisionDeFill, LibroDimensionable};
use quantum_arena::GlobalArena;
use std::sync::atomic::Ordering;
use std::sync::{Arc, MutexGuard};

const A: ClaveReserva = ClaveReserva { coin_id: 0, slot: 0, generation: 1 };
const B: ClaveReserva = ClaveReserva { coin_id: 0, slot: 1, generation: 1 };

fn arena() -> Arc<GlobalArena> {
    let arena = GlobalArena::build_in_own_stack(0.0);
    {
        let mut l = arena.config.entry_publication_gate.lock().unwrap();
        l.activar(&arena);
        l.saldo_aceptado(&arena, 100.0, Some(100));
    }
    arena.capital_incompleto.store(false, Ordering::Release);
    arena
}

fn libro(arena: &GlobalArena) -> MutexGuard<'_, LibroDimensionable> {
    arena.config.entry_publication_gate.lock().unwrap()
}

fn dim(arena: &GlobalArena) -> f64 {
    arena.unified_capital.load(Ordering::Relaxed)
}

/// A3: A reservada con la decision de 100; B posterior con un fill ya ejecutado y 0,75 USDT de
/// comision sin incluir y el ACK aun sin cambiar su estado: el envio de A cuenta lo ejecutado de
/// B. Igual durante otro parcial, en el terminal, con cancelacion tardia de B, con el replay del
/// fill y con dos fotos que no lo cubren. Control: B viva SIN ejecucion no invalida a A (politica
/// vigente para lo no ejecutado), y una foto que cubre el cargo de B devuelve el envio de A.
#[test]
fn parcial_de_reserva_posterior_cuenta_antes_del_ack() {
    for variante in ["antes del ACK", "segundo parcial", "terminal", "cancelacion tardia", "replay", "dos fotos"] {
        let a = arena();
        let mut l = libro(&a);
        l.reservar_con_base(&a, A, 0.25, Some(100.0));
        l.reservar(&a, B, 0.25);
        l.vincular_orden_con_cantidad(B, "cL_b", 1.0);
        assert!(l.envio_valido_en(&a, A), "{variante}: control, B viva sin ejecucion no invalida a A");
        l.fill_de_entrada_con(&a, 0, "cL_b", 42, Some(0.5), ComisionDeFill::Activo(0.75), None);
        match variante {
            "segundo parcial" => {
                l.fill_de_entrada_con(&a, 0, "cL_b", 43, Some(0.25), ComisionDeFill::Activo(0.10), Some(150));
            }
            "terminal" => l.marca_de_orden(&a, 0, "cL_b", 160),
            "cancelacion tardia" => l.cancelar(&a, B, 0.25),
            "replay" => {
                l.fill_de_entrada_con(&a, 0, "cL_b", 42, Some(0.5), ComisionDeFill::Activo(0.75), None);
                assert!(l.reservado() <= 0.25 + 0.75 + 0.125 + 1e-12, "{variante}: el replay cargo dos veces");
            }
            "dos fotos" => {
                // Dos fotos que repiten el saldo vigente (misma revision): no cubren un fill sin T.
                l.saldo_aceptado(&a, 100.0, Some(100));
                l.saldo_aceptado(&a, 100.0, Some(100));
            }
            _ => {}
        }
        assert!(!l.envio_valido_en(&a, A), "{variante}: el envio de A excluyo lo ya ejecutado de B (retenido {})", l.reservado());
    }
    // La cobertura del cargo de B (su propia marca) lo saca de lo retenido: solo queda el
    // estimado de la mitad no ejecutada de B (0,125) y la reserva de A (0,25).
    let a = arena();
    let mut l = libro(&a);
    l.reservar_con_base(&a, A, 0.25, Some(99.5));
    l.reservar(&a, B, 0.25);
    l.vincular_orden_con_cantidad(B, "cL_b", 1.0);
    l.fill_de_entrada_con(&a, 0, "cL_b", 42, Some(0.5), ComisionDeFill::Activo(0.75), Some(150));
    assert!(!l.envio_valido_en(&a, A), "100 - 0,75 < 99,5: fixture");
    l.saldo_aceptado(&a, 99.25, Some(151));
    assert_eq!(l.reservado(), 0.25 + 0.125, "la foto que cubre el cargo de B no lo incluyo");
}

/// A3: fixture estimado 10, media ejecucion con cargo 8 y estimado residual 5. Una foto que cubre
/// el 8 conserva el residual 5 en la capacidad global; sin cubrir, 8 + 5. Si la identidad o el
/// orden de la cobertura siguen inciertos (fill sin T), se retiene. Al ACK de la media ejecucion
/// (devuelta 5), el residual sale.
#[test]
fn cobertura_del_parcial_preserva_estimado_del_resto() {
    let a = arena();
    let mut l = libro(&a);
    l.reservar(&a, A, 10.0);
    l.vincular_orden_con_cantidad(A, "cL_a", 2.0);
    l.fill_de_entrada_con(&a, 0, "cL_a", 1, Some(1.0), ComisionDeFill::Activo(8.0), Some(150));
    assert_eq!(l.reservado(), 13.0, "sin cubrir: real 8 mas residual 5");
    l.saldo_aceptado(&a, 92.0, Some(151));
    assert_eq!(l.reservado(), 5.0, "la foto que cubre el 8 no conserva el residual 5");
    assert_eq!(dim(&a), 87.0, "capacidad global");
    l.ejecutar(&a, A, 5.0, None);
    assert_eq!(l.reservado(), 0.0, "el ACK de la media ejecucion no libero el residual");
    // Cobertura incierta: el mismo parcial sin T no se acredita con ninguna foto.
    let a = arena();
    let mut l = libro(&a);
    l.reservar(&a, A, 10.0);
    l.vincular_orden_con_cantidad(A, "cL_a", 2.0);
    l.fill_de_entrada_con(&a, 0, "cL_a", 1, Some(1.0), ComisionDeFill::Activo(8.0), None);
    l.saldo_aceptado(&a, 92.0, Some(151));
    assert_eq!(l.reservado(), 13.0, "una foto sin orden demostrable acredito el cargo sin T");
}

/// A5 (control de INV-D/E y V4I): en la misma cuenta, foto acreditada T500, relevo con T300,
/// callback T400 y cargo pendiente: ninguna de esas observaciones prueba la inclusion del cargo
/// por avanzar sobre T300. ABA de revision aunque vuelva el mismo importe. La identidad y el libro
/// de otra cuenta quedan aislados.
#[test]
fn relevo_no_presta_marca_al_debito_retenido() {
    let a = GlobalArena::build_in_own_stack(0.0);
    let mut l = libro(&a);
    l.activar(&a);
    l.fijar_cuenta(&a, 7);
    l.fijar_alcance(1, 1);
    l.saldo_aceptado(&a, 100.0, Some(500));
    l.reservar(&a, A, 0.5);
    l.vincular_orden(A, "cL_a");
    // Cargo pendiente sin marca (su T no llego).
    l.fill_de_entrada(&a, 0, "cL_a", 9, Some(0.5), None);
    l.ejecutar(&a, A, 0.0, None);
    assert_eq!(l.reservado(), 0.5);
    // Relevo de la misma cuenta: el alcance nuevo empieza con T300.
    l.fijar_alcance(2, 2);
    l.saldo_aceptado(&a, 99.5, Some(300));
    assert_eq!(l.reservado(), 0.5, "el relevo T300 presto la marca T500 al cargo");
    // Callback T400 del alcance vigente (otra operacion) y foto que avanza sobre T300.
    l.fill_de_cierre(&a, 0, 77, 0.25, 400, true);
    assert_eq!(l.reservado(), 0.75, "fixture: el cargo T400 retenido");
    // Foto del alcance vigente que avanza (T600 > T500): incluye el cargo T400, no el sin marca.
    l.saldo_aceptado(&a, 99.25, Some(600));
    assert_eq!(l.reservado(), 0.5, "avanzar sobre T300 acredito el cargo sin marca");
    // ABA de revision con el mismo importe: la reserva de la revision anterior no se envia.
    let capital = l.dimensionable();
    l.reservar_con_base(&a, B, 0.0, capital);
    l.saldo_aceptado(&a, 90.0, Some(610));
    l.saldo_aceptado(&a, 99.25, Some(620));
    assert_eq!(l.dimensionable(), capital, "fixture: mismo importe");
    assert!(!l.envio_valido(B), "el mismo importe revalido la revision anterior");
    // Un cargo retenido de esta cuenta (T700, aun no incluido).
    l.fill_de_cierre(&a, 0, 78, 0.25, 700, true);
    // Otra cuenta: empieza sin saldo, sus cargos no se deduplican contra los de la anterior.
    l.fijar_cuenta(&a, 8);
    assert!(l.saldo().is_none(), "la otra cuenta heredo el saldo");
    assert!(!l.admite(&a, capital), "admitida sin saldo de la cuenta nueva");
    let antes = l.reservado();
    l.fill_de_cierre(&a, 0, 78, 0.25, 480, true);
    assert_eq!(l.reservado(), antes + 0.25, "el trade 78 de la otra cuenta se tomo por el de la anterior");
    // Y en la misma cuenta nueva, el repetido no carga dos veces.
    l.fill_de_cierre(&a, 0, 78, 0.25, 480, true);
    assert_eq!(l.reservado(), antes + 0.25, "repetido en la misma cuenta");
}
