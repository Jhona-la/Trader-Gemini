//! CAPITAL-UNICO V4J, punto 4, salida A4b de `qra_v4j_criterios.md` (corte offline): el corte de
//! cuenta conciliado de V4I (`LibroDimensionable::conciliar` y sus tipos) se RETIRO del API
//! productivo de `quantum-arena`. La prueba negativa de compilacion son los doctests
//! `compile_fail` del modulo `dimensionable` (un consumidor externo no puede nombrarlo; corren con
//! `cargo test -p quantum-arena --doc`, con su control positivo). Aqui: las exportaciones y los
//! llamadores, y las tres secuencias de Codex montadas en el API vivo con sus importes.
use quantum_arena::dimensionable::{ClaveReserva, LibroDimensionable};
use quantum_arena::GlobalArena;
use std::path::{Path, PathBuf};
use std::sync::atomic::Ordering;
use std::sync::MutexGuard;

const DIMENSIONABLE: &str = include_str!("../../quantum-arena/src/dimensionable.rs");
const ARENA_LIB: &str = include_str!("../../quantum-arena/src/lib.rs");

fn fuentes(dir: &Path, out: &mut Vec<PathBuf>) {
    let Ok(entradas) = std::fs::read_dir(dir) else { return };
    for e in entradas.flatten() {
        let p = e.path();
        let nombre = p.file_name().and_then(|n| n.to_str()).unwrap_or("");
        if p.is_dir() {
            if !matches!(nombre, "target" | "tests" | "benches" | ".git") {
                fuentes(&p, out);
            }
        } else if nombre.ends_with(".rs") {
            out.push(p);
        }
    }
}

/// A4b: el metodo y sus tipos no existen en el API productivo (ni re-exportados), el doctest
/// `compile_fail` del consumidor externo esta en el modulo con su control positivo, y ningun
/// codigo de produccion del workspace (crates y host) los nombra.
#[test]
fn corte_aislado_no_se_puede_llamar_desde_host() {
    let codigo = DIMENSIONABLE.replace("\r\n", "\n");
    let sin_docs: String = codigo.lines().filter(|l| !l.trim_start().starts_with("//")).collect::<Vec<_>>().join("\n");
    for simbolo in ["fn conciliar", "struct CorteDeCuenta", "struct MovimientoDeCuenta", "enum TipoMovimiento", "enum CorteRechazado", "fn a_e8"] {
        assert!(!sin_docs.contains(simbolo), "el API productivo vuelve a exponer `{simbolo}`");
    }
    assert!(!ARENA_LIB.contains("CorteDeCuenta") && !ARENA_LIB.contains("conciliar"), "quantum-arena re-exporta el corte");
    // Prueba negativa de compilacion (doctests del modulo) y su control positivo.
    assert_eq!(codigo.matches("//! ```compile_fail").count(), 2, "faltan los doctests compile_fail del corte");
    assert!(codigo.contains("//! use quantum_arena::dimensionable::CorteDeCuenta;"));
    assert!(codigo.contains("//!     let _ = l.conciliar(a, &());"));
    assert!(codigo.contains("//!     l.marca_de_orden(a, 0, \"cL_x\", 1);"), "falta el control positivo del montaje");
    // Ningun llamador de produccion: crates/*/src y src/ del host.
    let raiz = Path::new(env!("CARGO_MANIFEST_DIR")).join("../..");
    let mut archivos = Vec::new();
    fuentes(&raiz.join("crates"), &mut archivos);
    fuentes(&raiz.join("src"), &mut archivos);
    assert!(archivos.len() > 50, "fixture: el recorrido no encontro las fuentes ({})", archivos.len());
    for a in archivos {
        // Solo codigo (los doctests compile_fail del modulo nombran el corte a proposito).
        let texto: String = std::fs::read_to_string(&a)
            .unwrap_or_default()
            .lines()
            .filter(|l| !l.trim_start().starts_with("//"))
            .collect::<Vec<_>>()
            .join("\n");
        for prohibido in ["CorteDeCuenta", "MovimientoDeCuenta", "CorteRechazado", "dimensionable::a_e8"] {
            assert!(!texto.contains(prohibido), "{} nombra `{prohibido}`", a.display());
        }
        // `conciliar` del libro (el auditor de trayectorias tiene otro `conciliar`, sin relacion).
        for llamada in [".conciliar(&arena", ".conciliar(arena", ".conciliar(&self.arena", ".conciliar(&arena_"] {
            assert!(!texto.contains(llamada), "{} concilia el libro (`{llamada}`)", a.display());
        }
    }
}

fn arena() -> std::sync::Arc<GlobalArena> {
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

/// La frontera productiva de admision: lo que Risk lee y lo que `admite` acepta con ello.
fn admision(arena: &GlobalArena) -> (f64, bool) {
    let capital = arena.unified_capital.load(Ordering::Relaxed);
    let mut l = libro(arena);
    let admite = l.admite(arena, Some(capital));
    (capital, admite)
}

/// A4b: las tres secuencias de corte de Codex (`qra_capital_v4i_adversario.rs`), con sus
/// importes, por la frontera productiva (el libro bajo la puerta y la admision de Risk). Sin corte
/// en vivo no hay capacidad nueva ni retiro del libro: el cargo queda retenido hasta su evidencia
/// propia (una foto con marca mayor que la del cargo; nunca para un cargo sin marca) o el veto.
#[test]
fn sin_corte_en_vivo_retiene_hasta_evidencia_o_veta() {
    // 1. Resultado realizado del mismo trade (PnL +1, hora 200) no acredita la comision 0,5 sin T.
    let a = arena();
    libro(&a).fill_de_entrada(&a, 0, "sin-reserva", 42, Some(0.5), None);
    libro(&a).saldo_aceptado(&a, 101.0, Some(201));
    let (capital, _) = admision(&a);
    assert!(libro(&a).vetado() || capital <= 100.5, "el PnL realizado acredito la comision: {capital}");
    assert_eq!({ let l = libro(&a); (l.reservas(), l.reservado()) }, (1, 0.5), "retiro del libro sin evidencia");
    libro(&a).saldo_aceptado(&a, 101.0, Some(10_000));
    assert_eq!(libro(&a).reservado(), 0.5, "un cargo sin marca salio con una marca mayor cualquiera");
    // 2. Corte vacio con frontera igual (fill T200, saldo T200): no prueba el debito del extremo.
    let a = arena();
    libro(&a).saldo_aceptado(&a, 100.0, Some(200));
    libro(&a).fill_de_entrada(&a, 0, "sin-reserva", 42, Some(0.5), Some(200));
    let (capital, _) = admision(&a);
    assert!(libro(&a).vetado() || capital <= 99.5, "frontera igual acredito el cargo: {capital}");
    assert_eq!(libro(&a).reservas(), 1);
    // Evidencia propia: una foto con marca ESTRICTAMENTE mayor que la del cargo lo incluye una vez.
    libro(&a).saldo_aceptado(&a, 99.5, Some(201));
    assert_eq!({ let l = libro(&a); (l.reservas(), l.reservado(), a.unified_capital.load(Ordering::Relaxed)) }, (0, 0.0, 99.5));
    // 3. Movimiento duplicado (comision 42 de -0,5 contada dos veces): ninguna capacidad nueva.
    let a = arena();
    libro(&a).saldo_aceptado(&a, 99.0, Some(201));
    libro(&a).fill_de_entrada(&a, 0, "sin-reserva", 42, Some(0.5), None);
    let antes = admision(&a);
    libro(&a).fill_de_entrada(&a, 0, "sin-reserva", 42, Some(0.5), None);
    let despues = admision(&a);
    assert_eq!(antes.0, 98.5, "fixture");
    assert_eq!(despues.0, antes.0, "el duplicado cambio la capacidad");
    assert_eq!({ let l = libro(&a); (l.reservas(), l.reservado()) }, (1, 0.5), "el duplicado se conto o salio del libro");
    // Una reserva de entrada con ese cargo pendiente no se dimensiona sobre lo no probado.
    let clave = ClaveReserva { coin_id: 0, slot: 0, generation: 1 };
    let mut l = libro(&a);
    l.reservar_con_base(&a, clave, 0.04, Some(despues.0));
    assert!(!l.envio_valido_en(&a, clave) || l.dimensionable().is_some_and(|d| d < 98.5), "dimensionado de mas: el cargo sin probar no resta");
}
