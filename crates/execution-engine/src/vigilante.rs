//! I-14 (Línea I, filas N44, N34 y N42 del censo C-F7b): el vigilante de
//! protección sólo actúa sobre lo que leyó.
//!
//! El vigilante del host (`god_engine.rs`) audita cada 5 s con la bandera
//! de protección sucia y cada 60 s sin ella. Lee las posiciones
//! (`positionRisk`), purga las piernas TP/SL sin posición debajo
//! (`purge_orphan_legs`, sobre `openAlgoOrders` sin símbolo) y completa la
//! cobertura de cada posición abierta (`ensure_position_protected`, sobre
//! `openAlgoOrders` del símbolo). Estas decisiones vivían en el host, sin
//! pruebas. Dos trataban un error de lectura como un dato (N44 y N34) y una
//! contaba rechazos viejos como nuevos (N42):
//!
//! - N44: las piernas de la posición se leían con
//!   `fetch_open_algo_orders(symbol).await.unwrap_or_default()`. Un error
//!   (5xx, red, cuerpo ilegible, cuota) se leía como «ninguna pierna», y el
//!   vigilante colocaba TP y SL encima de las que quizá seguían vivas. Las
//!   duplicadas sobreviven al cierre de la posición y quedan armadas sobre
//!   la siguiente (N15b).
//! - N34: el listado de huérfanas se tragaba su error (`if let Ok`) y la
//!   auditoría limpiaba la bandera sucia. Una huérfana cuya cancelación
//!   fallaba sólo se reportaba: una reentrada en su símbolo abría con ella
//!   armada encima, y el vigilante la contaba como protección de la nueva.
//! - N42: la escalada B3.5b sólo suma un paso de racha hacia el cierre de
//!   emergencia con rechazos NUEVOS del exchange. Comparaba el acumulado de
//!   vida (`protection_health::rejections_of`, que no vuelve a cero) con una
//!   foto que se borraba al quedar cubierto o cerrado el símbolo; en el
//!   siguiente hueco la foto valía 0 y todo rechazo antiguo contaba como
//!   nuevo.
//!
//! Una auditoría con alguna lectura fallida o frenada no está completa: la
//! bandera sucia no se limpia y [`ReintentoLectura`] decide cuándo releer.
//! Tras un fallo de lectura, la espera es de 5, 10, 20, 40 y 60 s: sin
//! ella, un fallo persistente (un cuerpo que no se puede leer cuesta su peso
//! entero, 40 sin símbolo) se repetiría cada 5 s. Con el uso por encima del
//! 80 % no sale el listado de huérfanas, del tramo de lectura
//! (`RATE_LIMIT_HEADROOM_`), pero las lecturas críticas (`positionRisk`, las
//! piernas de cada símbolo) sí salen y gastan peso: también espera así.
//! Cuando una lectura crítica vuelve con el 429 o el 418 que arma el freno,
//! o el ejecutor la rechaza sin enviarla porque dura ese freno o porque la
//! cuota crítica está agotada (`RATE_LIMIT_EXHAUSTED_`), el vigilante relee
//! en el ciclo siguiente: mientras dura, la relectura no sale ni gasta peso,
//! y protege en cuanto se libera. Lo mismo vale para la descarga de los
//! filtros de un símbolo sin spec en el registro: su error conserva el texto
//! del freno (`get_symbol_filter`). El freno de un 429 dura su
//! Retry-After (60 s por defecto), el del tercero seguido (RATE_BRAKE) su
//! Retry-After más 60 s y el de un 418 su Retry-After (300 s por defecto);
//! la cuota crítica se libera al rodar su ventana, de un minuto como mucho.
//! Ni la cuota ni el freno se anuncian como fallo.
//!
//! Con todas las lecturas completas el vigilante decide lo mismo que antes
//! salvo en dos casos: la escalada no cuenta un rechazo viejo como nuevo
//! (N42) y una huérfana que no se pudo cancelar queda como purga pendiente
//! (N34).

use std::collections::{HashMap, HashSet};

use crate::freno_temporal::es_freno_temporal;
use crate::order_types::OpenAlgoOrder;
use crate::reconciliation::PositionRiskEntry;

/// Ciclos del vigilante (de 5 s) entre dos auditorías nominales: 60 s,
/// alineado con la reconciliación. Es también el techo de la espera entre
/// relecturas.
pub const CICLOS_NOMINALES: u32 = 12;

/// Cobertura TP y SL (unidades del activo) que dan a la posición de lado
/// `is_long` las piernas que el exchange devolvió para su símbolo.
///
/// N44: con la lectura fallida no hay cobertura que medir y la función
/// devuelve el error tal cual (el vigilante lo clasifica con
/// `freno_temporal`); el host no coloca piernas sobre lo que no leyó. Cuenta
/// las piernas de su lado, las BOTH (one-way) y las que llegan sin
/// `positionSide`, vivas (NEW, TRIGGERING o TRIGGERED). TAKE_PROFIT cubre el
/// TP; STOP_MARKET y TRAILING_STOP_MARKET, el stop (B1.3).
pub fn cobertura_leida(
    lectura: Result<Vec<OpenAlgoOrder>, String>,
    is_long: bool,
) -> Result<(f64, f64), String> {
    let piernas = lectura?;
    let want_side = if is_long { "LONG" } else { "SHORT" };
    let mut tp = 0.0f64;
    let mut sl = 0.0f64;
    for a in &piernas {
        let side_ok =
            a.position_side == want_side || a.position_side == "BOTH" || a.position_side.is_empty();
        let live = matches!(a.algo_status.as_str(), "NEW" | "TRIGGERING" | "TRIGGERED");
        if !side_ok || !live {
            continue;
        }
        if a.order_type.contains("TAKE_PROFIT") {
            tp += a.quantity;
        } else {
            sl += a.quantity;
        }
    }
    Ok((tp, sl))
}

/// B2.6 / D-698: si la pierna no tiene debajo una posición que pueda estar
/// protegiendo. Una pierna LONG o SHORT (hedge) la tiene si hay una posición
/// abierta de su lado; una BOTH (one-way) o sin `positionSide`, si hay
/// cualquier posición abierta del símbolo: un lado vacío es DESCONOCIDO, no
/// «no coincide», y purgarlo dejaba desnuda una posición larga viva (D-698).
pub fn es_huerfana(pierna: &OpenAlgoOrder, posiciones: &[PositionRiskEntry]) -> bool {
    let side_unknown = pierna.position_side.is_empty();
    let side_open = posiciones.iter().any(|p| {
        p.symbol == pierna.symbol
            && p.position_amt.abs() > 0.0
            && (pierna.position_side == "BOTH"
                || side_unknown
                || (p.position_amt > 0.0) == (pierna.position_side == "LONG"))
    });
    !side_open
}

/// N34: el lado con que se anota la purga pendiente (I-9) de una huérfana
/// cuya cancelación falló. Una pierna SHORT (hedge) protegía un corto; las
/// demás, un largo. En one-way basta cualquiera: la purga pendiente
/// (`cancel_position_oco_orders`) cancela las piernas BOTH y las sin
/// `positionSide` de los dos lados, y `lado_ocupado` mira todo el símbolo.
pub fn lado_de_la_purga(pierna: &OpenAlgoOrder) -> bool {
    pierna.position_side != "SHORT"
}

/// N42: la base de rechazos de cada símbolo para la escalada B3.5b.
///
/// `protection_health::rejections_of` es un acumulado de vida que no vuelve
/// a cero. La escalada sólo cuenta un paso de racha si trae rechazos desde la
/// última vez que el vigilante miró el símbolo. Antes la base se borraba al
/// quedar el símbolo cubierto o cerrado, y el primer hueco siguiente
/// comparaba contra 0. Aquí la base se fija en esos casos al acumulado de ese
/// momento, y un hueco posterior sólo cuenta lo que llegue después.
///
/// Guarda una entrada por símbolo que el vigilante vio abierto en el
/// proceso: está acotada por el universo de contratos del exchange.
#[derive(Debug, Default)]
pub struct BaseDeRechazos {
    base: HashMap<String, u64>,
}

impl BaseDeRechazos {
    pub fn new() -> Self {
        Self::default()
    }

    /// La base del símbolo, 0 si el vigilante no lo ha visto en el proceso
    /// (todo rechazo del proceso es posterior), y deja `ahora` como base.
    pub fn tomar(&mut self, symbol: &str, ahora: u64) -> u64 {
        match self.base.get_mut(symbol) {
            Some(b) => std::mem::replace(b, ahora),
            None => {
                self.base.insert(symbol.to_string(), ahora);
                0
            }
        }
    }

    /// El símbolo quedó cubierto: su base pasa a `ahora`.
    pub fn fijar(&mut self, symbol: &str, ahora: u64) {
        match self.base.get_mut(symbol) {
            Some(b) => *b = ahora,
            None => {
                self.base.insert(symbol.to_string(), ahora);
            }
        }
    }

    /// Los símbolos que el vigilante ya vio y que no tienen posición abierta
    /// en esta auditoría: su base pasa a su acumulado de ahora.
    pub fn fijar_los_cerrados<F: Fn(&str) -> u64>(
        &mut self,
        abiertos: &HashSet<String>,
        rechazos_de: F,
    ) {
        for (symbol, b) in self.base.iter_mut() {
            if !abiertos.contains(symbol) {
                *b = rechazos_de(symbol);
            }
        }
    }

    /// Cuántos símbolos tiene.
    pub fn len(&self) -> usize {
        self.base.len()
    }

    pub fn is_empty(&self) -> bool {
        self.base.is_empty()
    }
}

/// Las lecturas de una auditoría: si alguna falló o la frenó el ejecutor.
#[derive(Debug, Default, Clone, PartialEq, Eq)]
pub struct Lecturas {
    /// El primer error de lectura que no es un freno temporal, con lo que
    /// se estaba leyendo.
    fallo: Option<String>,
    /// Alguna lectura del tramo de lectura no salió por la cuota
    /// (`RATE_LIMIT_HEADROOM_`): las críticas sí salieron y gastaron peso.
    cuota_de_lectura: bool,
    /// Alguna lectura volvió con el 429/418 que arma el freno, o el ejecutor
    /// no la envió: dura ese freno o la cuota crítica está agotada
    /// (`RATE_LIMIT_EXHAUSTED_`).
    bloqueo: bool,
}

/// El rechazo del tramo de lectura, el único que deja salir a las lecturas
/// críticas. Con el resto de `freno_temporal` (el 429/418 que arma el freno,
/// el freno mismo y la cuota crítica agotada) las lecturas críticas
/// siguientes no salen hasta que se libere.
fn es_cuota_de_lectura(e: &str) -> bool {
    e.starts_with("RATE_LIMIT_HEADROOM_")
}

impl Lecturas {
    pub fn new() -> Self {
        Self::default()
    }

    /// Anota el error de la lectura de `que`.
    pub fn error(&mut self, que: &str, e: &str) {
        if es_cuota_de_lectura(e) {
            self.cuota_de_lectura = true;
        } else if es_freno_temporal(e) {
            self.bloqueo = true;
        } else if self.fallo.is_none() {
            self.fallo = Some(format!("{que}: {e}"));
        }
    }

    /// Si todas las lecturas de la auditoría se completaron.
    pub fn completas(&self) -> bool {
        self.fallo.is_none() && !self.cuota_de_lectura && !self.bloqueo
    }
}

/// Cómo cerró una auditoría sus lecturas.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Cierre {
    /// Todas completas. Lleva cuántas auditorías seguidas habían quedado
    /// incompletas antes (0 si ninguna).
    Completa { incompletas_previas: u32 },
    /// Alguna falló: se relee tras `espera_ciclos` ciclos de 5 s como mucho.
    Fallida { error: String, espera_ciclos: u32 },
    /// Ninguna falló, y alguna lectura volvió con el 429/418 que arma el freno
    /// o el ejecutor no la envió (ese freno o la cuota crítica agotada): se
    /// relee en el ciclo siguiente.
    /// `seguidas` cuenta las auditorías frenadas seguidas, ésta incluida.
    Frenada { seguidas: u32 },
    /// Ninguna falló ni la frenó el ejecutor, y la cuota del tramo de
    /// lectura no dejó salir alguna: se relee tras `espera_ciclos` ciclos de
    /// 5 s como mucho.
    PorCuota { espera_ciclos: u32 },
}

impl Cierre {
    /// Si la auditoría completó sus lecturas: sólo entonces se limpia la
    /// bandera de protección.
    pub fn completa(&self) -> bool {
        matches!(self, Cierre::Completa { .. })
    }
}

/// Cuándo releer tras auditorías incompletas (con alguna lectura fallida o
/// frenada).
///
/// Tras un fallo de lectura o un rechazo de la cuota de lectura, la primera
/// relectura va al ciclo siguiente (5 s) y cada auditoría así seguida dobla
/// la espera, hasta los 60 s de la auditoría nominal. Si una lectura volvió
/// con un 429/418 o el ejecutor no la envió (su freno o la cuota crítica
/// agotada) se relee en el ciclo siguiente, sin doblar: mientras dura, la
/// relectura no sale ni gasta peso, y protege en cuanto se libera.
/// Un evento terminal de protección ajeno a la auditoría (un ALGO_UPDATE, una
/// purga pendiente nueva de otra tarea) adelanta la relectura: trae
/// información que la auditoría no tenía. Una auditoría completa la deja a
/// cero.
///
/// El vigilante no vuelve a marcar sucia la bandera de protección cuando una
/// auditoría queda incompleta: `mark_dirty` cuenta además un evento terminal
/// y adelantaría su propia relectura en cada ciclo. Mientras haya una
/// relectura pendiente decide [`ReintentoLectura::toca`], no la bandera.
#[derive(Debug, Default, Clone, PartialEq, Eq)]
pub struct ReintentoLectura {
    /// Auditorías seguidas con un fallo de lectura o un rechazo de la cuota
    /// de lectura.
    incompletas: u32,
    /// Auditorías seguidas frenadas (un 429/418, su freno o la cuota crítica
    /// agotada).
    frenadas: u32,
    /// Ciclos de espera de la relectura pendiente; 0 si no hay ninguna.
    espera: u32,
    /// El ciclo de la última auditoría incompleta.
    desde: u32,
    /// Los eventos terminales que esa auditoría ya tenía en cuenta.
    eventos: u64,
}

impl ReintentoLectura {
    pub fn new() -> Self {
        Self::default()
    }

    /// Ciclos de espera tras `incompletas` auditorías con fallo o cuota
    /// seguidas (`incompletas` >= 1): 1, 2, 4, 8 y luego
    /// [`CICLOS_NOMINALES`].
    pub fn espera_de(incompletas: u32) -> u32 {
        let potencia = incompletas.saturating_sub(1).min(31);
        (1u32 << potencia).min(CICLOS_NOMINALES)
    }

    /// Cuántas auditorías seguidas tuvieron un fallo de lectura o un rechazo
    /// de la cuota de lectura.
    pub fn incompletas(&self) -> u32 {
        self.incompletas
    }

    /// Si toca auditar en el ciclo `ciclo`, fuera de la cadencia nominal, con
    /// `eventos` terminales vistos y la bandera de protección `sucia`. Sin
    /// relectura pendiente decide la bandera, como antes; con una pendiente,
    /// su espera o un evento terminal ajeno.
    pub fn toca(&self, ciclo: u32, eventos: u64, sucia: bool) -> bool {
        if self.espera == 0 {
            return sucia;
        }
        eventos != self.eventos || ciclo.wrapping_sub(self.desde) >= self.espera
    }

    /// Cierra la auditoría del ciclo `ciclo`. `eventos` son los terminales
    /// vistos al empezarla más las marcas de la propia auditoría (sus purgas
    /// pendientes nuevas): un evento por encima de eso, aunque llegue a mitad
    /// de la auditoría, adelanta la relectura.
    pub fn cerrar(&mut self, ciclo: u32, eventos: u64, lecturas: &Lecturas) -> Cierre {
        if lecturas.completas() {
            let incompletas_previas = std::mem::take(&mut self.incompletas);
            self.frenadas = 0;
            self.espera = 0;
            return Cierre::Completa {
                incompletas_previas,
            };
        }
        self.desde = ciclo;
        self.eventos = eventos;
        if lecturas.fallo.is_none() && lecturas.bloqueo {
            self.frenadas = self.frenadas.saturating_add(1);
            self.espera = 1;
            return Cierre::Frenada {
                seguidas: self.frenadas,
            };
        }
        self.frenadas = 0;
        self.incompletas = self.incompletas.saturating_add(1);
        self.espera = Self::espera_de(self.incompletas);
        match &lecturas.fallo {
            Some(error) => Cierre::Fallida {
                error: error.clone(),
                espera_ciclos: self.espera,
            },
            None => Cierre::PorCuota {
                espera_ciclos: self.espera,
            },
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn pierna(symbol: &str, tipo: &str, lado: &str, estado: &str, qty: f64) -> OpenAlgoOrder {
        OpenAlgoOrder {
            symbol: symbol.to_string(),
            order_type: tipo.to_string(),
            position_side: lado.to_string(),
            algo_status: estado.to_string(),
            quantity: qty,
            ..Default::default()
        }
    }

    fn posicion(symbol: &str, amt: f64) -> PositionRiskEntry {
        PositionRiskEntry {
            symbol: symbol.to_string(),
            position_amt: amt,
            ..Default::default()
        }
    }

    /// RED antes de I-14: el host leía un error como `Vec::new()` y la
    /// cobertura salía 0, con el hueco entero que se rellenaba.
    #[test]
    fn i14_n44_sin_lectura_no_hay_cobertura_que_rellenar() {
        let r = cobertura_leida(Err("HTTP 503".to_string()), true);
        assert_eq!(r, Err("HTTP 503".to_string()), "el error, tal cual");
        assert_eq!(
            cobertura_leida(Ok(Vec::new()), true),
            Ok((0.0, 0.0)),
            "leída y vacía: hueco entero"
        );
    }

    #[test]
    fn i14_n44_la_cobertura_cuenta_su_lado_vivo_como_antes() {
        let piernas = vec![
            pierna("BTCUSDT", "TAKE_PROFIT_MARKET", "LONG", "NEW", 0.4),
            pierna("BTCUSDT", "STOP_MARKET", "LONG", "NEW", 0.5),
            pierna("BTCUSDT", "TRAILING_STOP_MARKET", "BOTH", "TRIGGERING", 0.1),
            pierna("BTCUSDT", "TAKE_PROFIT", "", "TRIGGERED", 0.2),
            pierna("BTCUSDT", "STOP_MARKET", "SHORT", "NEW", 9.0),
            pierna("BTCUSDT", "STOP_MARKET", "LONG", "CANCELED", 9.0),
            pierna("BTCUSDT", "TAKE_PROFIT_MARKET", "LONG", "FINISHED", 9.0),
        ];
        let (tp, sl) = cobertura_leida(Ok(piernas.clone()), true).unwrap_or((-1.0, -1.0));
        assert!((tp - 0.6).abs() < 1e-12, "tp {tp}");
        assert!((sl - 0.6).abs() < 1e-12, "sl {sl}");
        let (tp, sl) = cobertura_leida(Ok(piernas), false).unwrap_or((-1.0, -1.0));
        assert!(
            (tp - 0.2).abs() < 1e-12,
            "el corto sólo ve BOTH, sin lado y SHORT: tp {tp}"
        );
        assert!((sl - 9.1).abs() < 1e-12, "sl {sl}");
    }

    /// La regla de D-698 que el host aplicaba en línea, sin cambios.
    #[test]
    fn i14_n34_huerfana_segun_d698() {
        let largo = [posicion("BTCUSDT", 0.5)];
        let corto = [posicion("BTCUSDT", -0.5)];
        let otra = [posicion("ETHUSDT", 1.0)];
        let plano = [posicion("BTCUSDT", 0.0)];
        let both = pierna("BTCUSDT", "STOP_MARKET", "BOTH", "NEW", 0.5);
        let sin_lado = pierna("BTCUSDT", "STOP_MARKET", "", "NEW", 0.5);
        let long = pierna("BTCUSDT", "STOP_MARKET", "LONG", "NEW", 0.5);
        let short = pierna("BTCUSDT", "STOP_MARKET", "SHORT", "NEW", 0.5);
        for p in [&both, &sin_lado] {
            assert!(!es_huerfana(p, &largo));
            assert!(
                !es_huerfana(p, &corto),
                "lado desconocido: cualquier posición del símbolo"
            );
            assert!(es_huerfana(p, &otra));
            assert!(es_huerfana(p, &plano), "una posición a cero no protege");
            assert!(es_huerfana(p, &[]));
        }
        assert!(!es_huerfana(&long, &largo));
        assert!(
            es_huerfana(&long, &corto),
            "hedge: el lado contrario no la sostiene"
        );
        assert!(!es_huerfana(&short, &corto));
        assert!(es_huerfana(&short, &largo));
    }

    #[test]
    fn i14_n34_lado_de_la_purga_pendiente() {
        assert!(lado_de_la_purga(&pierna(
            "X",
            "STOP_MARKET",
            "LONG",
            "NEW",
            1.0
        )));
        assert!(!lado_de_la_purga(&pierna(
            "X",
            "STOP_MARKET",
            "SHORT",
            "NEW",
            1.0
        )));
        assert!(lado_de_la_purga(&pierna(
            "X",
            "STOP_MARKET",
            "BOTH",
            "NEW",
            1.0
        )));
        assert!(lado_de_la_purga(&pierna(
            "X",
            "STOP_MARKET",
            "",
            "NEW",
            1.0
        )));
    }

    /// RED antes de I-14 (N42): con la foto borrada al quedar cubierto, el
    /// primer hueco siguiente contaba los dos rechazos viejos como nuevos.
    #[test]
    fn i14_n42_un_rechazo_viejo_no_cuenta_en_el_hueco_siguiente() {
        let mut b = BaseDeRechazos::new();
        // Primer hueco: dos rechazos del exchange en dos auditorías.
        assert_eq!(b.tomar("BTCUSDT", 1), 0, "primer rechazo: nuevo");
        assert_eq!(b.tomar("BTCUSDT", 2), 1, "segundo: nuevo");
        // La pierna entra: cubierta.
        b.fijar("BTCUSDT", 2);
        // Un hueco después, sin rechazos nuevos: la base es 2, no 0.
        assert_eq!(
            b.tomar("BTCUSDT", 2),
            2,
            "sin rechazos nuevos no hay paso de racha"
        );
        // Un rechazo nuevo en ese hueco sí cuenta.
        assert_eq!(b.tomar("BTCUSDT", 3), 2);
    }

    /// Rechazos mientras el símbolo está cubierto (un TP que se reintenta con
    /// el SL completo): al quedar cubierto la base sube con ellos, y el hueco
    /// de stop siguiente no los cuenta.
    #[test]
    fn i14_n42_fijar_mueve_la_base() {
        let mut b = BaseDeRechazos::new();
        assert_eq!(b.tomar("BTCUSDT", 1), 0);
        b.fijar("BTCUSDT", 4);
        assert_eq!(b.tomar("BTCUSDT", 4), 4, "no 1");
        b.fijar("ETHUSDT", 3);
        assert_eq!(b.tomar("ETHUSDT", 3), 3, "un símbolo nuevo también, no 0");
    }

    #[test]
    fn i14_n42_el_cierre_fija_la_base_y_la_reentrada_solo_cuenta_lo_nuevo() {
        let mut b = BaseDeRechazos::new();
        assert_eq!(b.tomar("BTCUSDT", 5), 0);
        assert_eq!(b.tomar("ETHUSDT", 1), 0);
        // BTC cierra y acumula otro rechazo de su posición anterior antes de
        // la auditoría que lo ve cerrado.
        let abiertos: HashSet<String> = ["ETHUSDT".to_string()].into_iter().collect();
        b.fijar_los_cerrados(&abiertos, |s| if s == "BTCUSDT" { 6 } else { 99 });
        // ETH sigue abierto: su base no se toca.
        assert_eq!(b.tomar("ETHUSDT", 1), 1);
        // Reentrada en BTC: su primer hueco no ve los 6 viejos.
        assert_eq!(b.tomar("BTCUSDT", 6), 6);
        assert_eq!(
            b.tomar("BTCUSDT", 7),
            6,
            "el rechazo de la reentrada sí cuenta"
        );
        assert_eq!(b.len(), 2);
    }

    /// Un 429/418 no deja salir nada: se relee en el ciclo siguiente, sin
    /// doblar, y no se anuncia como fallo.
    #[test]
    fn i14_un_freno_se_relee_en_el_ciclo_siguiente() {
        let mut l = Lecturas::new();
        assert!(l.completas());
        l.error(
            "positionRisk",
            "RATE_LIMIT_COOLDOWN: activo hasta 2 (ahora 1)",
        );
        assert!(!l.completas());
        let mut r = ReintentoLectura::new();
        for (n, ciclo) in (10..20).enumerate() {
            assert_eq!(
                r.cerrar(ciclo, 0, &l),
                Cierre::Frenada {
                    seguidas: n as u32 + 1
                }
            );
            assert!(!r.toca(ciclo, 0, true), "el mismo ciclo no");
            assert!(
                r.toca(ciclo + 1, 0, false),
                "el siguiente sí, aunque esté limpia"
            );
        }
        assert_eq!(r.incompletas(), 0, "un freno no dobla la espera");
        assert!(r.cerrar(20, 0, &Lecturas::new()).completa());
        assert_eq!(
            r.cerrar(21, 0, &l),
            Cierre::Frenada { seguidas: 1 },
            "tras una completa, la racha de frenadas vuelve a empezar"
        );
        for e in [
            "RATE_BRAKE: freno temporal activo hasta 9 (ahora 1)",
            "HTTP_429_RATE_LIMITED retry_after=7",
            "HTTP_418_IP_BANNED retry_after=120",
            "FRENO_ARMADO: HTTP_429_RATE_LIMITED retry_after=7",
            "RATE_LIMIT_EXHAUSTED_WEIGHT: 2396 usado + 0 en vuelo + 5 > 2400",
            "RATE_LIMIT_EXHAUSTED_ORDERS_10S: 300 usado + 0 en vuelo + 1 > 300",
        ] {
            let mut l = Lecturas::new();
            l.error("positionRisk", e);
            let mut r = ReintentoLectura::new();
            assert_eq!(r.cerrar(0, 0, &l), Cierre::Frenada { seguidas: 1 }, "{e}");
        }
    }

    /// La cuota del tramo de lectura deja salir a las críticas, que gastan
    /// peso: espera doblada, como un fallo, sin anunciarse como fallo.
    #[test]
    fn i14_la_cuota_de_lectura_espera_como_un_fallo() {
        let mut l = Lecturas::new();
        l.error("openAlgoOrders", "RATE_LIMIT_HEADROOM_WEIGHT: 2000/2400");
        let mut r = ReintentoLectura::new();
        assert_eq!(r.cerrar(0, 0, &l), Cierre::PorCuota { espera_ciclos: 1 });
        assert_eq!(r.cerrar(1, 0, &l), Cierre::PorCuota { espera_ciclos: 2 });
        assert!(!r.toca(2, 0, true));
        assert!(r.toca(3, 0, true));
        // Con una lectura crítica sin enviar a la vez, manda el bloqueo: lo
        // que no se leyó es la protección.
        let mut ambos = Lecturas::new();
        ambos.error("openAlgoOrders", "RATE_LIMIT_HEADROOM_WEIGHT: 2000/2400");
        ambos.error(
            "BTCUSDT",
            "RATE_LIMIT_EXHAUSTED_WEIGHT: 2396 usado + 0 en vuelo + 5 > 2400",
        );
        assert_eq!(r.cerrar(3, 0, &ambos), Cierre::Frenada { seguidas: 1 });
        assert!(r.toca(4, 0, true));
    }

    /// RED en la tercera versión de I-14: la cuota crítica agotada doblaba
    /// la espera como la de lectura, aunque `positionRisk` rechazado no
    /// envía nada. Una posición desnuda que el vigilante debía proteger
    /// esperaba hasta 40 s más tras liberarse la cuota; en main se releía
    /// cada 5 s.
    #[test]
    fn i14_la_cuota_critica_agotada_se_relee_en_el_ciclo_siguiente() {
        let mut agotada = Lecturas::new();
        agotada.error(
            "positionRisk",
            "RATE_LIMIT_EXHAUSTED_WEIGHT: 2396 usado + 0 en vuelo + 5 > 2400",
        );
        let mut r = ReintentoLectura::new();
        for ciclo in 1..=8u32 {
            assert_eq!(
                r.cerrar(ciclo, 0, &agotada),
                Cierre::Frenada { seguidas: ciclo }
            );
            assert!(r.toca(ciclo + 1, 0, true), "relee a los 5 s");
        }
        assert_eq!(r.incompletas(), 0);
        assert_eq!(
            r.cerrar(9, 0, &Lecturas::new()),
            Cierre::Completa {
                incompletas_previas: 0
            }
        );
    }

    /// Una frenada tras fallos: la cuota sigue doblando; un freno relee en
    /// el ciclo siguiente sin perder la cuenta de los fallos.
    #[test]
    fn i14_una_frenada_tras_fallos_no_reinicia_la_espera() {
        let mut fallo = Lecturas::new();
        fallo.error("openAlgoOrders", "OPEN_ALGO_PARSE: expected value");
        let mut cuota = Lecturas::new();
        cuota.error("openAlgoOrders", "RATE_LIMIT_HEADROOM_WEIGHT: 2000/2400");
        let mut freno = Lecturas::new();
        freno.error(
            "positionRisk",
            "RATE_BRAKE: freno temporal activo hasta 9 (ahora 1)",
        );
        let mut r = ReintentoLectura::new();
        r.cerrar(0, 0, &fallo);
        r.cerrar(1, 0, &fallo);
        assert_eq!(
            r.cerrar(3, 0, &cuota),
            Cierre::PorCuota { espera_ciclos: 4 }
        );
        assert!(!r.toca(6, 0, true));
        assert!(r.toca(7, 0, true));
        assert_eq!(r.cerrar(7, 0, &freno), Cierre::Frenada { seguidas: 1 });
        assert!(r.toca(8, 0, true));
        assert!(matches!(
            r.cerrar(8, 0, &fallo),
            Cierre::Fallida {
                espera_ciclos: 8,
                ..
            }
        ));
        // Tras el fallo, otra frenada vuelve a contar desde 1.
        assert_eq!(r.cerrar(16, 0, &freno), Cierre::Frenada { seguidas: 1 });
    }

    #[test]
    fn i14_solo_una_completa_limpia_la_bandera() {
        assert!(Cierre::Completa {
            incompletas_previas: 3
        }
        .completa());
        assert!(!Cierre::Frenada { seguidas: 1 }.completa());
        assert!(!Cierre::PorCuota { espera_ciclos: 4 }.completa());
        assert!(!Cierre::Fallida {
            error: "x".to_string(),
            espera_ciclos: 1
        }
        .completa());
    }

    #[test]
    fn i14_sin_relectura_pendiente_decide_la_bandera() {
        let r = ReintentoLectura::new();
        assert!(!r.toca(5, 3, false), "limpia: sólo la nominal");
        assert!(r.toca(5, 3, true), "sucia: cada ciclo, como antes");
    }

    #[test]
    fn i14_la_espera_dobla_hasta_la_nominal() {
        let esperas: Vec<u32> = (1..=7).map(ReintentoLectura::espera_de).collect();
        assert_eq!(esperas, vec![1, 2, 4, 8, 12, 12, 12]);
        assert_eq!(ReintentoLectura::espera_de(u32::MAX), CICLOS_NOMINALES);
        let mut r = ReintentoLectura::new();
        let mut l = Lecturas::new();
        l.error("openAlgoOrders", "OPEN_ALGO_PARSE: expected value");
        l.error("positionRisk", "HTTP 503"); // sólo se guarda el primero
        let c = r.cerrar(100, 7, &l);
        assert_eq!(
            c,
            Cierre::Fallida {
                error: "openAlgoOrders: OPEN_ALGO_PARSE: expected value".to_string(),
                espera_ciclos: 1
            }
        );
        assert!(r.toca(101, 7, true));
        assert!(matches!(
            r.cerrar(101, 7, &l),
            Cierre::Fallida {
                espera_ciclos: 2,
                ..
            }
        ));
        assert!(
            !r.toca(102, 7, true),
            "a los 5 s todavía no, aunque esté sucia"
        );
        assert!(r.toca(103, 7, false), "a los 10 s sí, aunque esté limpia");
        assert!(
            r.toca(102, 8, true),
            "un evento terminal posterior al cierre adelanta la relectura"
        );
        assert_eq!(
            r.cerrar(103, 8, &Lecturas::new()),
            Cierre::Completa {
                incompletas_previas: 2
            }
        );
        assert_eq!(r.incompletas(), 0);
        assert!(r.toca(104, 8, true));
        assert!(!r.toca(104, 8, false));
    }

    #[test]
    fn i14_el_ciclo_que_da_la_vuelta_no_bloquea_la_relectura() {
        let mut r = ReintentoLectura::new();
        let mut l = Lecturas::new();
        l.error("positionRisk", "HTTP 500");
        r.cerrar(u32::MAX, 0, &l);
        assert!(
            r.toca(0, 0, false),
            "u32::MAX + 1 ciclo = la espera de un ciclo"
        );
    }

    /// El texto de los rechazos del freno lo fija `freno_temporal`; aquí
    /// basta que un error del exchange o de red sí cuente.
    #[test]
    fn i14_errores_que_si_cuentan() {
        for e in [
            "HTTP 503 Service Unavailable",
            "-1021 Timestamp outside recvWindow",
            "error sending request",
        ] {
            let mut l = Lecturas::new();
            l.error("positionRisk", e);
            let mut r = ReintentoLectura::new();
            assert!(matches!(r.cerrar(0, 0, &l), Cierre::Fallida { .. }), "{e}");
        }
    }
}
