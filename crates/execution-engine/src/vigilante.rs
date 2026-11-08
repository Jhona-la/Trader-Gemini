//! I-14 (Línea I, filas N44, N34 y N42 del censo C-F7b): el vigilante de
//! protección sólo actúa sobre lo que leyó.
//!
//! El vigilante del host (`god_engine.rs`) audita cada 5 s con la bandera
//! de protección sucia y cada 60 s sin ella. Lista las piernas TP/SL
//! (`openAlgoOrders` sin símbolo), lee las posiciones (`positionRisk`),
//! purga las piernas sin posición debajo (`purge_orphan_legs`) y completa la
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
//!   Además el listado se juzgaba con las posiciones leídas ANTES: una
//!   entrada que colocaba su bracket entre las dos lecturas lo veía
//!   cancelado como huérfano. Ahora el listado va primero.
//! - N42: la escalada B3.5b sólo suma un paso de racha hacia el cierre de
//!   emergencia con rechazos NUEVOS del exchange. Comparaba el acumulado de
//!   vida (`protection_health::rejections_of`, que no vuelve a cero) con una
//!   foto que se borraba al quedar cubierto o cerrado el símbolo; en el
//!   siguiente hueco la foto valía 0 y todo rechazo antiguo contaba como
//!   nuevo. Y en hedge la racha, la foto y el acumulado eran por símbolo: el
//!   lado cubierto borraba la racha del lado desnudo, que nunca escalaba, y
//!   los rechazos de un lado contaban para el otro. Ahora son por (símbolo,
//!   lado).
//!
//! Una auditoría con alguna lectura fallida o frenada, o una colocación
//! frenada, no está completa: la bandera sucia no se limpia y
//! [`ReintentoLectura`] decide cuándo releer. Las lecturas que cuentan son
//! las que deciden qué coloca o purga la auditoría ([`Lecturas`]). Tras un
//! fallo de lectura, la relectura sale como mucho 5, 10, 20, 40 y 60 s
//! después de acabar la auditoría: sin esa espera, un fallo persistente (un
//! cuerpo que no se puede leer cuesta su peso entero, 40 sin símbolo) se
//! repetiría cada 5 s. Con el uso por encima del 80 % no sale el listado de
//! huérfanas, del tramo de lectura (`RATE_LIMIT_HEADROOM_`), pero las
//! lecturas críticas (`positionRisk`, las piernas de cada símbolo) sí salen
//! y gastan peso: también espera así. Cuando una lectura crítica vuelve con
//! el 429 o el 418 que arma el freno, o el ejecutor la rechaza sin enviarla
//! porque dura ese freno o porque la cuota crítica está agotada
//! (`RATE_LIMIT_EXHAUSTED_`), el vigilante relee en el ciclo siguiente,
//! aunque otra lectura de la auditoría haya fallado: mientras dura el freno
//! no sale ninguna petición, y protege en cuanto se libera. Lo mismo vale
//! para la colocación de una pierna frenada así, y para la descarga de los
//! filtros de un símbolo sin spec en el registro, cuyo error conserva el
//! texto del freno (`get_symbol_filter`). El freno de un 429 dura su
//! Retry-After (60 s por defecto), el del tercero seguido (RATE_BRAKE) su
//! Retry-After más 60 s y el de un 418 su Retry-After (300 s por defecto);
//! la cuota crítica se libera al rodar su ventana, de un minuto como mucho.
//! Ni la cuota ni el freno se anuncian como fallo.
//!
//! Con todas las lecturas completas y ninguna colocación frenada, el
//! vigilante decide lo mismo que antes salvo en que: la escalada no cuenta un
//! rechazo viejo ni el del otro lado como nuevo, y en hedge el lado desnudo
//! escala al cierre de emergencia aunque el otro esté cubierto (N42); una
//! huérfana que no se pudo cancelar y sigue abierta queda como purga
//! pendiente, y las huérfanas se juzgan con posiciones leídas tras el
//! listado (N34); y una posición cubierta del todo no necesita el filtro de
//! su símbolo para darse por cubierta.

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

/// D-698: si la purga de las piernas de la posición de lado `is_long`
/// (`cancel_position_oco_orders`) cancela una pierna con este
/// `positionSide`. Cancela las de su lado y las BOTH. Las que llegan sin
/// `positionSide`, sólo en one-way: en hedge un vacío es DESCONOCIDO y
/// cancelarlo podía dejar desnuda la posición del otro lado.
pub fn pierna_del_lado(position_side: &str, is_long: bool, hedge: bool) -> bool {
    let lado = if is_long { "LONG" } else { "SHORT" };
    position_side == lado || position_side == "BOTH" || (position_side.is_empty() && !hedge)
}

/// N34: quita del listado de huérfanas las piernas que el reintento de purgas
/// pendientes de esta auditoría (I-9) acaba de cancelar. El listado se leyó
/// antes que el reintento, y volver a cancelarlas fallaría. Quita sólo las
/// que cancela la purga de cada (símbolo, lado) purgado
/// ([`pierna_del_lado`]): en hedge, las del otro lado del símbolo se siguen
/// juzgando en esta auditoría.
pub fn quitar_las_purgadas(
    piernas: &mut Vec<OpenAlgoOrder>,
    purgados: &HashSet<(String, bool)>,
    hedge: bool,
) {
    piernas.retain(|l| {
        !purgados.iter().any(|(symbol, is_long)| {
            *symbol == l.symbol && pierna_del_lado(&l.position_side, *is_long, hedge)
        })
    });
}

/// N34: si una huérfana cuya cancelación falló sigue abierta, según la
/// relectura de las piernas abiertas de su símbolo.
///
/// El listado de huérfanas es anterior a la foto de posiciones. Una pierna
/// que disparó o se canceló entre las dos lecturas ya no existe, su
/// cancelación falla («orden inexistente»), y anotarla como purga pendiente
/// dejaría el símbolo sin entradas sin motivo. Basta con que aparezca en la
/// relectura: el exchange sólo lista piernas abiertas. Se identifica por
/// `algoId`, o por `clientAlgoId` si el `algoId` no se conoce (0). Si la
/// relectura falla no se sabe, y se trata como abierta.
pub fn sigue_abierta(pierna: &OpenAlgoOrder, releida: &Result<Vec<OpenAlgoOrder>, String>) -> bool {
    let Ok(abiertas) = releida else {
        return true;
    };
    abiertas.iter().any(|a| {
        if pierna.algo_id != 0 {
            a.algo_id == pierna.algo_id
        } else {
            a.client_algo_id == pierna.client_algo_id
        }
    })
}

/// N42: la base de rechazos de cada posición, (símbolo, lado), para la
/// escalada B3.5b.
///
/// `protection_health::rejections_of` es un acumulado de vida por (símbolo,
/// lado) que no vuelve a cero. La escalada sólo cuenta un paso de racha si
/// trae rechazos desde la última vez que el vigilante miró la posición.
/// Antes la base se borraba al quedar el símbolo cubierto o cerrado, y el
/// primer hueco siguiente comparaba contra 0. Aquí el vigilante la siembra
/// al mirar la posición por primera vez, antes de colocar nada, y la fija al
/// quedar cubierta o cerrada: un hueco posterior sólo cuenta lo que llegue
/// después. La re-protección del arranque (RESTORE) corre fuera del
/// vigilante y a la vez que su primera auditoría: los rechazos que anotó
/// antes de que el vigilante mire la posición quedan dentro de la base al
/// sembrarla, y los que anote después cuentan como nuevos (N44c).
///
/// En hedge el largo y el corto de un símbolo son dos posiciones: antes la
/// base y la racha eran por símbolo, y el lado cubierto borraba en cada
/// auditoría la racha del lado desnudo, que no llegaba nunca al cierre de
/// emergencia.
///
/// Guarda una entrada por posición que el vigilante vio abierta en el
/// proceso: está acotada por el universo de contratos del exchange.
#[derive(Debug, Default)]
pub struct BaseDeRechazos {
    /// Por lado: `[cortos, largos]`, indexado con `usize::from(is_long)`.
    lados: [HashMap<String, u64>; 2],
}

impl BaseDeRechazos {
    pub fn new() -> Self {
        Self::default()
    }

    fn lado(&mut self, is_long: bool) -> &mut HashMap<String, u64> {
        &mut self.lados[usize::from(is_long)]
    }

    /// El vigilante mira la posición: si no la tenía, su base es `ahora`, y
    /// los rechazos posteriores a esta mirada cuentan como nuevos. Si ya la
    /// tenía, no la toca.
    pub fn sembrar(&mut self, symbol: &str, is_long: bool, ahora: u64) {
        let lado = self.lado(is_long);
        if !lado.contains_key(symbol) {
            lado.insert(symbol.to_string(), ahora);
        }
    }

    /// La base de la posición, y deja `ahora` como base. Sin sembrar antes
    /// devuelve `ahora`: no hay base desde la que contar rechazos nuevos.
    pub fn tomar(&mut self, symbol: &str, is_long: bool, ahora: u64) -> u64 {
        match self.lado(is_long).get_mut(symbol) {
            Some(b) => std::mem::replace(b, ahora),
            None => {
                self.lado(is_long).insert(symbol.to_string(), ahora);
                ahora
            }
        }
    }

    /// La posición quedó cubierta: su base pasa a `ahora`.
    pub fn fijar(&mut self, symbol: &str, is_long: bool, ahora: u64) {
        self.lado(is_long).insert(symbol.to_string(), ahora);
    }

    /// Las posiciones que el vigilante ya vio y que no están abiertas en esta
    /// auditoría: su base pasa a su acumulado de ahora.
    pub fn fijar_los_cerrados<F: Fn(&str, bool) -> u64>(
        &mut self,
        abiertas: &HashSet<(String, bool)>,
        rechazos_de: F,
    ) {
        for (i, lado) in self.lados.iter_mut().enumerate() {
            let is_long = i == 1;
            for (symbol, b) in lado.iter_mut() {
                if !abiertas.contains(&(symbol.clone(), is_long)) {
                    *b = rechazos_de(symbol, is_long);
                }
            }
        }
    }
}

/// Las lecturas de una auditoría: si alguna falló o la frenó el ejecutor.
/// También cuenta como frenada la auditoría en que el freno o la cuota
/// crítica no dejaron salir la colocación de una pierna.
///
/// Son las lecturas que deciden qué coloca o purga la auditoría:
/// `positionRisk`, el listado de huérfanas, las piernas de cada posición y
/// el filtro de su símbolo. Las del reintento de purgas pendientes (I-9) no
/// entran: su fallo deja la purga pendiente y el símbolo sin entradas.
/// Tampoco la relectura de una huérfana que no se pudo cancelar
/// ([`sigue_abierta`]): si falla, la huérfana se anota igual.
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
    /// (`RATE_LIMIT_EXHAUSTED_`). O una colocación, igual.
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

    /// El freno temporal o la cuota crítica agotada no dejaron salir la
    /// colocación de una pierna: la auditoría se repite en el ciclo
    /// siguiente, como con una lectura frenada. Antes cerraba completa,
    /// limpiaba la bandera y la pierna esperaba a la auditoría nominal.
    pub fn colocacion_frenada(&mut self) {
        self.bloqueo = true;
    }

    /// Si todas las lecturas de la auditoría se completaron.
    pub fn completas(&self) -> bool {
        self.fallo.is_none() && !self.cuota_de_lectura && !self.bloqueo
    }
}

/// Cómo cerró una auditoría sus lecturas.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Cierre {
    /// Todas completas. Lleva cuántas auditorías, desde la anterior completa,
    /// cerraron [`Cierre::Fallida`] o [`Cierre::PorCuota`]
    /// (`incompletas_previas`) y cuántas [`Cierre::Frenada`]
    /// (`frenadas_previas`): cada auditoría sin completar cuenta una vez, y
    /// una frenada con un fallo cuenta como frenada.
    Completa {
        incompletas_previas: u32,
        frenadas_previas: u32,
    },
    /// Alguna falló: se relee `espera_ciclos` ciclos después del inicio de la
    /// auditoría, como mucho 5·`espera_ciclos` s tras acabar.
    Fallida { error: String, espera_ciclos: u32 },
    /// Alguna lectura o colocación volvió con el 429/418 que arma el freno o
    /// el ejecutor no la envió (ese freno o la cuota crítica agotada): se
    /// relee en el ciclo siguiente aunque otra lectura de la auditoría haya
    /// fallado (`fallo`). `seguidas` cuenta las auditorías frenadas
    /// seguidas, ésta incluida.
    Frenada {
        seguidas: u32,
        fallo: Option<String>,
    },
    /// Ninguna falló ni la frenó el ejecutor, y la cuota del tramo de
    /// lectura no dejó salir alguna: se relee como en [`Cierre::Fallida`].
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
/// frenada, o una colocación frenada).
///
/// Las esperas se cuentan en ciclos: los disparos del intervalo de 5 s del
/// vigilante (`MissedTickBehavior::Skip`), desde el disparo en que EMPEZÓ la
/// auditoría incompleta. Una espera de k ciclos sale como mucho 5·k s
/// después de ACABAR la auditoría. Si la auditoría dura más de 5 s (plazos
/// agotados), `Skip` pierde los disparos que caen durante ella, y medida
/// desde su inicio la espera dura más.
///
/// Tras un fallo de lectura o un rechazo de la cuota de lectura, la primera
/// relectura va al ciclo siguiente y cada auditoría así seguida dobla la
/// espera, hasta los 60 s de la auditoría nominal. Si una lectura o una
/// colocación volvió con un 429/418 o el ejecutor no la envió (su freno o la
/// cuota crítica agotada), se relee en el ciclo siguiente, sin doblar,
/// aunque otra lectura de la misma auditoría haya fallado: con el freno no
/// sale ninguna petición, y la cuota crítica se libera al rodar su ventana,
/// de un minuto como mucho. Así protege en cuanto se libera. Un fallo o un
/// rechazo de la cuota de lectura de esa auditoría cuentan para la espera de
/// las siguientes.
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
    /// Auditorías con un fallo de lectura o un rechazo de la cuota de lectura
    /// desde la última completa, también las frenadas que tuvieron uno: fija
    /// la espera tras un fallo. Una frenada sin ninguno de los dos no la
    /// cambia.
    incompletas: u32,
    /// Auditorías frenadas seguidas (un 429/418, su freno o la cuota crítica
    /// agotada): una fallida o por cuota la vuelve a 0. Espacia el aviso de
    /// una racha larga (un baneo 418).
    frenadas: u32,
    /// Desde la última completa: cuántas auditorías cerraron fallidas o por
    /// cuota (`[0]`) y cuántas frenadas (`[1]`), para el aviso al volver a
    /// leerlo todo.
    sin_completar: [u32; 2],
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
            let [incompletas_previas, frenadas_previas] = std::mem::take(&mut self.sin_completar);
            self.incompletas = 0;
            self.frenadas = 0;
            self.espera = 0;
            return Cierre::Completa {
                incompletas_previas,
                frenadas_previas,
            };
        }
        self.desde = ciclo;
        self.eventos = eventos;
        if lecturas.bloqueo {
            // Manda el bloqueo: lo que no salió es lo que protege. Un fallo o
            // la cuota de lectura de la misma auditoría cuentan para las
            // esperas siguientes.
            if lecturas.fallo.is_some() || lecturas.cuota_de_lectura {
                self.incompletas = self.incompletas.saturating_add(1);
            }
            self.frenadas = self.frenadas.saturating_add(1);
            self.sin_completar[1] = self.sin_completar[1].saturating_add(1);
            self.espera = 1;
            return Cierre::Frenada {
                seguidas: self.frenadas,
                fallo: lecturas.fallo.clone(),
            };
        }
        self.frenadas = 0;
        self.incompletas = self.incompletas.saturating_add(1);
        self.sin_completar[0] = self.sin_completar[0].saturating_add(1);
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

    /// El predicado de `cancel_position_oco_orders` (D-698), sin cambios.
    #[test]
    fn i14_n34_lo_que_cancela_la_purga_de_un_lado() {
        for hedge in [true, false] {
            assert!(pierna_del_lado("LONG", true, hedge));
            assert!(pierna_del_lado("BOTH", true, hedge));
            assert!(!pierna_del_lado("SHORT", true, hedge));
            assert!(pierna_del_lado("SHORT", false, hedge));
            assert!(pierna_del_lado("BOTH", false, hedge));
            assert!(!pierna_del_lado("LONG", false, hedge));
        }
        assert!(
            pierna_del_lado("", true, false),
            "one-way: la pierna es nuestra"
        );
        assert!(pierna_del_lado("", false, false));
        assert!(
            !pierna_del_lado("", true, true),
            "hedge: un vacío es desconocido"
        );
        assert!(!pierna_del_lado("", false, true));
    }

    /// RED en la sexta versión de I-14: el listado perdía TODAS las piernas
    /// del símbolo purgado. En hedge, la purga pendiente del largo no cancela
    /// la huérfana del corto, que se saltaba justo en la auditoría en que el
    /// símbolo vuelve a admitir entradas.
    #[test]
    fn i14_n34_la_purga_de_un_lado_no_salva_las_huerfanas_del_otro() {
        let piernas = vec![
            pierna("BTCUSDT", "STOP_MARKET", "LONG", "NEW", 1.0),
            pierna("BTCUSDT", "TAKE_PROFIT_MARKET", "SHORT", "NEW", 1.0),
            pierna("BTCUSDT", "STOP_MARKET", "BOTH", "NEW", 1.0),
            pierna("BTCUSDT", "STOP_MARKET", "", "NEW", 1.0),
            pierna("ETHUSDT", "STOP_MARKET", "LONG", "NEW", 1.0),
        ];
        let purgados: HashSet<(String, bool)> =
            [("BTCUSDT".to_string(), true)].into_iter().collect();
        let lados = |v: &[OpenAlgoOrder]| -> Vec<(String, String)> {
            v.iter()
                .map(|l| (l.symbol.clone(), l.position_side.clone()))
                .collect()
        };
        let mut en_hedge = piernas.clone();
        quitar_las_purgadas(&mut en_hedge, &purgados, true);
        assert_eq!(
            lados(&en_hedge),
            vec![
                ("BTCUSDT".to_string(), "SHORT".to_string()),
                ("BTCUSDT".to_string(), String::new()),
                ("ETHUSDT".to_string(), "LONG".to_string()),
            ],
            "hedge: quedan el corto, la sin lado (la purga no la toca) y el otro símbolo"
        );
        let mut en_one_way = piernas;
        quitar_las_purgadas(&mut en_one_way, &purgados, false);
        assert_eq!(
            lados(&en_one_way),
            vec![
                ("BTCUSDT".to_string(), "SHORT".to_string()),
                ("ETHUSDT".to_string(), "LONG".to_string()),
            ],
            "one-way: la sin lado también la canceló la purga"
        );
        let mut sin_purgas = vec![pierna("BTCUSDT", "STOP_MARKET", "LONG", "NEW", 1.0)];
        quitar_las_purgadas(&mut sin_purgas, &HashSet::new(), true);
        assert_eq!(sin_purgas.len(), 1);
    }

    /// RED en la sexta versión de I-14: una huérfana que disparó o se
    /// canceló entre el listado y la cancelación quedaba como purga
    /// pendiente, y su símbolo sin entradas.
    #[test]
    fn i14_n34_solo_se_anota_la_huerfana_que_sigue_abierta() {
        let mut leg = pierna("BTCUSDT", "STOP_MARKET", "LONG", "NEW", 1.0);
        leg.algo_id = 7;
        leg.client_algo_id = "a".to_string();
        let mut otra = leg.clone();
        otra.algo_id = 8;
        otra.client_algo_id = "b".to_string();
        assert!(sigue_abierta(&leg, &Ok(vec![otra.clone(), leg.clone()])));
        assert!(
            !sigue_abierta(&leg, &Ok(vec![otra.clone()])),
            "ya no está: disparó o se canceló"
        );
        assert!(!sigue_abierta(&leg, &Ok(Vec::new())));
        assert!(
            sigue_abierta(&leg, &Err("HTTP 503".to_string())),
            "sin relectura no se sabe: se anota"
        );
        // Por algoId cuando se conoce, aunque cambie el clientAlgoId.
        let mut misma = leg.clone();
        misma.client_algo_id = "otro".to_string();
        assert!(sigue_abierta(&leg, &Ok(vec![misma])));
        // Sin algoId, por clientAlgoId.
        let mut sin_id = leg.clone();
        sin_id.algo_id = 0;
        let mut por_cliente = otra.clone();
        por_cliente.client_algo_id = "a".to_string();
        assert!(sigue_abierta(&sin_id, &Ok(vec![por_cliente])));
        assert!(!sigue_abierta(&sin_id, &Ok(vec![otra])));
        // Con el estado vacío también: el exchange sólo lista abiertas.
        let mut sin_estado = leg.clone();
        sin_estado.algo_status = String::new();
        assert!(sigue_abierta(&leg, &Ok(vec![sin_estado])));
    }

    /// RED antes de I-14 (N42): con la foto borrada al quedar cubierto, el
    /// primer hueco siguiente contaba los dos rechazos viejos como nuevos.
    #[test]
    fn i14_n42_un_rechazo_viejo_no_cuenta_en_el_hueco_siguiente() {
        let mut b = BaseDeRechazos::new();
        // Primer hueco: el vigilante siembra antes de colocar, y el exchange
        // rechaza en dos auditorías.
        b.sembrar("BTCUSDT", true, 0);
        assert_eq!(b.tomar("BTCUSDT", true, 1), 0, "primer rechazo: nuevo");
        b.sembrar("BTCUSDT", true, 1);
        assert_eq!(b.tomar("BTCUSDT", true, 2), 1, "segundo: nuevo");
        // La pierna entra: cubierta.
        b.fijar("BTCUSDT", true, 2);
        // Un hueco después, sin rechazos nuevos: la base es 2, no 0.
        b.sembrar("BTCUSDT", true, 2);
        assert_eq!(
            b.tomar("BTCUSDT", true, 2),
            2,
            "sin rechazos nuevos no hay paso de racha"
        );
        // Un rechazo nuevo en ese hueco sí cuenta.
        assert_eq!(b.tomar("BTCUSDT", true, 3), 2);
    }

    /// Rechazos mientras la posición está cubierta (un TP que se reintenta
    /// con el SL completo): al quedar cubierta la base sube con ellos, y el
    /// hueco de stop siguiente no los cuenta.
    #[test]
    fn i14_n42_fijar_mueve_la_base() {
        let mut b = BaseDeRechazos::new();
        b.sembrar("BTCUSDT", true, 0);
        assert_eq!(b.tomar("BTCUSDT", true, 1), 0);
        b.fijar("BTCUSDT", true, 4);
        assert_eq!(b.tomar("BTCUSDT", true, 4), 4, "no 1");
        b.fijar("ETHUSDT", false, 3);
        assert_eq!(
            b.tomar("ETHUSDT", false, 5),
            3,
            "una posición nueva también: su base es 3, no 0 ni la de tomar"
        );
    }

    /// RED antes de I-14 (N42, quinta revisión): sin sembrar, la primera
    /// mirada del vigilante comparaba contra 0 y contaba como nuevos los
    /// rechazos de RESTORE, que corre fuera de él.
    #[test]
    fn i14_n42_sembrar_absorbe_los_rechazos_anteriores_y_no_pisa() {
        let mut b = BaseDeRechazos::new();
        // RESTORE dejó 3 rechazos antes de que el vigilante mirara; el
        // cuarto llega después de la mirada.
        b.sembrar("BTCUSDT", true, 3);
        assert_eq!(
            b.tomar("BTCUSDT", true, 4),
            3,
            "los de RESTORE no son nuevos; el posterior a la mirada sí"
        );
        // Sembrar otra vez no pisa la base que ya tiene.
        b.fijar("BTCUSDT", true, 5);
        b.sembrar("BTCUSDT", true, 9);
        assert_eq!(
            b.tomar("BTCUSDT", true, 9),
            5,
            "cuatro nuevos desde la base"
        );
        // Sin sembrar no hay base desde la que contar.
        assert_eq!(b.tomar("ETHUSDT", true, 7), 7);
    }

    /// RED antes de I-14 (N42, quinta revisión): en hedge, la base era por
    /// símbolo y el lado cubierto movía la del lado desnudo.
    #[test]
    fn i14_n42_en_hedge_cada_lado_tiene_su_base() {
        let mut b = BaseDeRechazos::new();
        b.sembrar("BTCUSDT", true, 0);
        b.sembrar("BTCUSDT", false, 0);
        // El corto, desnudo: su SL rechazado.
        assert_eq!(b.tomar("BTCUSDT", false, 1), 0);
        // El largo, cubierto, con sus propios rechazos de TP.
        b.fijar("BTCUSDT", true, 4);
        // El corto sigue con su base: el rechazo siguiente es nuevo.
        assert_eq!(b.tomar("BTCUSDT", false, 2), 1);
        assert_eq!(b.tomar("BTCUSDT", true, 4), 4);
    }

    #[test]
    fn i14_n42_el_cierre_fija_la_base_y_la_reentrada_solo_cuenta_lo_nuevo() {
        let mut b = BaseDeRechazos::new();
        b.sembrar("BTCUSDT", true, 0);
        assert_eq!(b.tomar("BTCUSDT", true, 5), 0);
        b.sembrar("ETHUSDT", true, 0);
        assert_eq!(b.tomar("ETHUSDT", true, 1), 0);
        b.sembrar("BTCUSDT", false, 2);
        // El largo de BTC cierra y acumula otro rechazo de su posición
        // anterior antes de la auditoría que lo ve cerrado; el corto de BTC
        // también cierra.
        let abiertas: HashSet<(String, bool)> =
            [("ETHUSDT".to_string(), true)].into_iter().collect();
        b.fijar_los_cerrados(&abiertas, |s, l| match (s, l) {
            ("BTCUSDT", true) => 6,
            ("BTCUSDT", false) => 3,
            _ => 99,
        });
        // ETH sigue abierto: su base no se toca.
        assert_eq!(b.tomar("ETHUSDT", true, 1), 1);
        // Reentrada en BTC: su primer hueco no ve los viejos.
        b.sembrar("BTCUSDT", true, 6);
        assert_eq!(b.tomar("BTCUSDT", true, 6), 6);
        assert_eq!(
            b.tomar("BTCUSDT", true, 7),
            6,
            "el rechazo de la reentrada sí cuenta"
        );
        assert_eq!(b.tomar("BTCUSDT", false, 3), 3, "cada lado, el suyo");
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
                    seguidas: n as u32 + 1,
                    fallo: None
                }
            );
            assert!(!r.toca(ciclo, 0, true), "el mismo ciclo no");
            assert!(
                r.toca(ciclo + 1, 0, false),
                "el siguiente sí, aunque esté limpia"
            );
        }
        assert_eq!(
            r.cerrar(20, 0, &Lecturas::new()),
            Cierre::Completa {
                incompletas_previas: 0,
                frenadas_previas: 10
            },
            "un freno no dobla la espera; la completa dice cuántas frenó"
        );
        assert_eq!(
            r.cerrar(21, 0, &l),
            Cierre::Frenada {
                seguidas: 1,
                fallo: None
            },
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
            assert_eq!(
                r.cerrar(0, 0, &l),
                Cierre::Frenada {
                    seguidas: 1,
                    fallo: None
                },
                "{e}"
            );
        }
    }

    /// RED en la quinta versión de I-14: con un fallo y un freno en la
    /// misma auditoría ganaba el fallo y la espera se doblaba hasta 60 s,
    /// aunque lo frenado (la protección de una posición) no sale ni gasta
    /// peso mientras dura el freno.
    #[test]
    fn i14_con_un_fallo_y_un_freno_manda_el_freno_sin_perder_el_fallo() {
        let mut fallo = Lecturas::new();
        fallo.error("openAlgoOrders", "OPEN_ALGO_PARSE: expected value");
        let mut r = ReintentoLectura::new();
        for ciclo in 0..2u32 {
            r.cerrar(ciclo, 0, &fallo);
        }
        let mut mixta = Lecturas::new();
        mixta.error("openAlgoOrders", "OPEN_ALGO_PARSE: expected value");
        mixta.error("BTCUSDT", "HTTP_429_RATE_LIMITED retry_after=7");
        assert_eq!(
            r.cerrar(20, 0, &mixta),
            Cierre::Frenada {
                seguidas: 1,
                fallo: Some("openAlgoOrders: OPEN_ALGO_PARSE: expected value".to_string())
            }
        );
        assert!(r.toca(21, 0, false), "relee en el ciclo siguiente");
        // El fallo contó: la auditoría fallida siguiente es la cuarta con
        // fallo y espera 8 ciclos, no 4.
        assert_eq!(
            r.cerrar(21, 0, &fallo),
            Cierre::Fallida {
                error: "openAlgoOrders: OPEN_ALGO_PARSE: expected value".to_string(),
                espera_ciclos: 8
            }
        );
        // El orden de las anotaciones no cambia nada.
        let mut al_reves = Lecturas::new();
        al_reves.error("BTCUSDT", "RATE_LIMIT_COOLDOWN: activo hasta 2 (ahora 1)");
        al_reves.error("positionRisk", "HTTP 503");
        let mut r = ReintentoLectura::new();
        assert_eq!(
            r.cerrar(0, 0, &al_reves),
            Cierre::Frenada {
                seguidas: 1,
                fallo: Some("positionRisk: HTTP 503".to_string())
            }
        );
    }

    /// RED en la quinta versión de I-14: una colocación frenada (el TP con
    /// un 429 que arma el freno, el SL rechazado sin enviar) dejaba la
    /// auditoría completa, se limpiaba la bandera y la posición sin SL
    /// esperaba a la auditoría nominal.
    #[test]
    fn i14_una_colocacion_frenada_se_reintenta_en_el_ciclo_siguiente() {
        let mut l = Lecturas::new();
        l.colocacion_frenada();
        assert!(!l.completas());
        let mut r = ReintentoLectura::new();
        assert_eq!(
            r.cerrar(3, 0, &l),
            Cierre::Frenada {
                seguidas: 1,
                fallo: None
            }
        );
        assert!(r.toca(4, 0, false));
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
        assert_eq!(
            r.cerrar(3, 0, &ambos),
            Cierre::Frenada {
                seguidas: 1,
                fallo: None
            }
        );
        assert!(r.toca(4, 0, true));
        // RED en la sexta versión de I-14: la cuota de la auditoría frenada no
        // contaba para la espera, y la siguiente por cuota esperaba 4 ciclos.
        assert_eq!(
            r.cerrar(4, 0, &l),
            Cierre::PorCuota { espera_ciclos: 8 },
            "la cuota cuenta para la espera también cuando manda el bloqueo"
        );
        assert_eq!(
            r.cerrar(12, 0, &Lecturas::new()),
            Cierre::Completa {
                incompletas_previas: 3,
                frenadas_previas: 1
            },
            "cada auditoría sin completar cuenta una vez"
        );
    }

    /// RED en la sexta versión de I-14: una fallida borraba la cuenta de
    /// frenadas, y el aviso al volver a completas decía 0 frenadas; una
    /// frenada con un fallo contaba en las dos.
    #[test]
    fn i14_la_completa_cuenta_todas_las_auditorias_sin_completar() {
        let mut freno = Lecturas::new();
        freno.error(
            "positionRisk",
            "RATE_BRAKE: freno temporal activo hasta 9 (ahora 1)",
        );
        let mut fallo = Lecturas::new();
        fallo.error("openAlgoOrders", "OPEN_ALGO_PARSE: expected value");
        let mut r = ReintentoLectura::new();
        for ciclo in 0..3u32 {
            r.cerrar(ciclo, 0, &freno);
        }
        assert!(matches!(r.cerrar(3, 0, &fallo), Cierre::Fallida { .. }));
        assert_eq!(
            r.cerrar(4, 0, &Lecturas::new()),
            Cierre::Completa {
                incompletas_previas: 1,
                frenadas_previas: 3
            }
        );
        let mut mixta = fallo.clone();
        mixta.error("BTCUSDT", "HTTP_429_RATE_LIMITED retry_after=7");
        assert!(matches!(r.cerrar(5, 0, &mixta), Cierre::Frenada { .. }));
        assert_eq!(
            r.cerrar(6, 0, &Lecturas::new()),
            Cierre::Completa {
                incompletas_previas: 0,
                frenadas_previas: 1
            },
            "la mixta, una sola vez"
        );
    }

    /// RED en la tercera versión de I-14: la cuota crítica agotada doblaba
    /// la espera como la de lectura, aunque `positionRisk` rechazado no
    /// envía nada. Una posición desnuda que el vigilante debía proteger
    /// esperaba hasta 40 s más tras liberarse la cuota; en main, con la
    /// bandera sucia, se releía cada 5 s.
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
                Cierre::Frenada {
                    seguidas: ciclo,
                    fallo: None
                }
            );
            assert!(r.toca(ciclo + 1, 0, true), "relee a los 5 s");
        }
        assert_eq!(
            r.cerrar(9, 0, &Lecturas::new()),
            Cierre::Completa {
                incompletas_previas: 0,
                frenadas_previas: 8
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
        assert_eq!(
            r.cerrar(7, 0, &freno),
            Cierre::Frenada {
                seguidas: 1,
                fallo: None
            }
        );
        assert!(r.toca(8, 0, true));
        assert!(matches!(
            r.cerrar(8, 0, &fallo),
            Cierre::Fallida {
                espera_ciclos: 8,
                ..
            }
        ));
        // Tras el fallo, otra frenada vuelve a contar desde 1.
        assert_eq!(
            r.cerrar(16, 0, &freno),
            Cierre::Frenada {
                seguidas: 1,
                fallo: None
            }
        );
    }

    #[test]
    fn i14_solo_una_completa_limpia_la_bandera() {
        assert!(Cierre::Completa {
            incompletas_previas: 3,
            frenadas_previas: 1
        }
        .completa());
        assert!(!Cierre::Frenada {
            seguidas: 1,
            fallo: None
        }
        .completa());
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
                incompletas_previas: 2,
                frenadas_previas: 0
            }
        );
        assert!(r.toca(104, 8, true));
        assert!(!r.toca(104, 8, false));
        // La cuenta vuelve a cero: el fallo siguiente espera un ciclo.
        assert!(matches!(
            r.cerrar(104, 8, &l),
            Cierre::Fallida {
                espera_ciclos: 1,
                ..
            }
        ));
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
