//! RB-E2 (Codex RB, 2026-11-10) — EMBUDO VIVO: por qué no se abre cada
//! intención, en el log del host de Demo y Producción.
//!
//! QUÉ: el núcleo ya contaba el embudo de entrada (`direction_diag` y los
//! contadores `diag_*`), pero sólo lo imprimía el backtest forense: en vivo era
//! telemetría fantasma. La Demo del 2026-11-10 pasó 6,5 h sin una operación y su
//! log no decía qué puerta la detenía. Este módulo cuenta lo que faltaba —los
//! ticks cuyas entradas se bloquean antes de evaluarlas, por causa; las
//! intenciones que mueren sin ranura; los rechazos del riesgo por motivo, de
//! ESTE núcleo— y resume cada ventana en una línea con su cuello de botella.
//!
//! CÓMO: `BloqueosEntrada::registrar` sustituye la disyunción `entries_blocked`
//! del núcleo con la misma semántica y atribuye cada tick bloqueado a la
//! PRIMERA causa activa, de modo que las causas parten los bloqueos.
//! `Instantanea::de` copia los contadores; `linea` resume una ventana (la resta
//! de dos instantáneas) junto al acumulado. Sólo observa: ninguna decisión lee
//! estos contadores (contrato `rb_e2_embudo_vivo_contract` y huella A/B).
//!
//! Unidades: los bloqueos cuentan ticks que llegan a la frontera de entradas de
//! `process_tick_dual`; los escudos, intenciones unificadas con dirección y sin
//! posición abierta (`DirectionDiag::funnel_begin`); ranura, riesgo, consejo,
//! ML-GATE, margen y aperturas, intenciones con dirección que llegaron a esa
//! etapa. Los rechazos del riesgo se leen con `risk_engine::ultimo_rechazo`
//! justo tras la evaluación de este núcleo: los contadores globales
//! (`REJECT_COUNTERS`) mezclan todas las instancias del proceso, incluidos los
//! universos sombra del host.

use crate::direction_diag::{FUNNEL_STAGES, N_FUNNEL_STAGES};
use crate::GodEngineCore;
use risk_engine::{REJECT_NAMES, REJECT_SLOTS};

/// Causas por las que el núcleo no evalúa entradas en un tick.
pub const CAUSAS_BLOQUEO: usize = 7;
/// Nombre de cada causa, en el orden en que se atribuye el tick.
pub const NOMBRES_BLOQUEO: [&str; CAUSAS_BLOQUEO] = [
    "el llamador no las permite",
    "comisiones sin observar",
    "kill switch",
    "veto de deriva",
    "feed estancado",
    "latencia sobre el umbral",
    "liquidaciones sin marca válida",
];
/// Índice de la causa «el llamador no las permite» (`allow_entries` falso).
pub const BLOQUEO_LLAMADOR: usize = 0;

fn suma(xs: &[u64]) -> u64 {
    xs.iter().fold(0, |a, &x| a.saturating_add(x))
}

fn resta<const N: usize>(a: [u64; N], b: [u64; N]) -> [u64; N] {
    let mut r = [0; N];
    for ((x, a), b) in r.iter_mut().zip(a).zip(b) {
        *x = a.saturating_sub(b);
    }
    r
}

/// Ticks que llegan a la frontera de entradas y, de ellos, los bloqueados por
/// causa.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct BloqueosEntrada {
    pub ticks: u64,
    /// Ticks bloqueados, cada uno atribuido a su primera causa activa.
    pub por_causa: [u64; CAUSAS_BLOQUEO],
}

impl BloqueosEntrada {
    /// Cuenta el tick y, si alguna causa está activa, lo atribuye a la primera
    /// (orden de `NOMBRES_BLOQUEO`). Devuelve la disyunción de las causas: si
    /// las entradas quedan bloqueadas.
    #[inline]
    pub fn registrar(&mut self, causas: [bool; CAUSAS_BLOQUEO]) -> bool {
        self.ticks = self.ticks.saturating_add(1);
        let Some(primera) = causas.iter().position(|&activa| activa) else {
            return false;
        };
        if let Some(n) = self.por_causa.get_mut(primera) {
            *n = n.saturating_add(1);
        }
        true
    }

    pub fn bloqueados(&self) -> u64 {
        suma(&self.por_causa)
    }

    fn menos(&self, previa: &Self) -> Self {
        Self {
            ticks: self.ticks.saturating_sub(previa.ticks),
            por_causa: resta(self.por_causa, previa.por_causa),
        }
    }
}

/// Etapas que siguen al embudo unificado, para una población de intenciones.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct EtapasPostEmbudo {
    /// Intenciones con dirección sin ranura: [sin ranura resonante, misma
    /// dirección sin asegurar].
    pub sin_ranura: [u64; 2],
    /// Evaluaciones del riesgo de este núcleo que admitió.
    pub riesgo_aceptadas: u64,
    /// Rechazos del riesgo de este núcleo, por motivo (`REJ_*`).
    pub riesgo_rechazos: [u64; REJECT_SLOTS],
    /// Rechazos sin motivo contado (capital asignado ≤ 0, ver
    /// `risk_engine::ultimo_rechazo`).
    pub riesgo_sin_motivo: u64,
}

impl EtapasPostEmbudo {
    pub fn rechazos(&self) -> u64 {
        suma(&self.riesgo_rechazos).saturating_add(self.riesgo_sin_motivo)
    }

    fn menos(&self, previa: &Self) -> Self {
        Self {
            sin_ranura: resta(self.sin_ranura, previa.sin_ranura),
            riesgo_aceptadas: self
                .riesgo_aceptadas
                .saturating_sub(previa.riesgo_aceptadas),
            riesgo_rechazos: resta(self.riesgo_rechazos, previa.riesgo_rechazos),
            riesgo_sin_motivo: self
                .riesgo_sin_motivo
                .saturating_sub(previa.riesgo_sin_motivo),
        }
    }
}

/// Lo que el núcleo cuenta para el embudo vivo y `direction_diag` no contaba.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct ContadoresNucleo {
    /// Eventos de `process_event` sin oportunidad de entrada: sólo libro.
    pub solo_libro: u64,
    /// Eventos de `process_event` con el pánico de latencia del host (en un
    /// replay, el calentamiento: `suppress_entries`).
    pub panico_host: u64,
    pub bloqueos: BloqueosEntrada,
    /// Intenciones con el libro de su moneda plano: la población del embudo
    /// unificado (`DirectionDiag::funnel_begin`), la que compite por abrir.
    pub libro_plano: EtapasPostEmbudo,
    /// Intenciones con alguna posición abierta en su moneda: fuera del embudo
    /// (añadirían exposición; no abren la primera posición).
    pub con_posicion: EtapasPostEmbudo,
}

impl ContadoresNucleo {
    /// Un evento de `process_event`: el pánico del host se atribuye antes que
    /// el evento de sólo libro, igual que en `allow_entries`.
    #[inline]
    pub fn registrar_llamada(&mut self, solo_libro: bool, panico_host: bool) {
        if panico_host {
            self.panico_host = self.panico_host.saturating_add(1);
        } else if solo_libro {
            self.solo_libro = self.solo_libro.saturating_add(1);
        }
    }

    fn poblacion(&mut self, libro_plano: bool) -> &mut EtapasPostEmbudo {
        if libro_plano {
            &mut self.libro_plano
        } else {
            &mut self.con_posicion
        }
    }

    /// Una intención con dirección sin ranura.
    #[inline]
    pub fn registrar_sin_ranura(&mut self, libro_plano: bool, misma_direccion_sin_asegurar: bool) {
        let i = usize::from(misma_direccion_sin_asegurar);
        if let Some(n) = self.poblacion(libro_plano).sin_ranura.get_mut(i) {
            *n = n.saturating_add(1);
        }
    }

    /// Una evaluación del riesgo: `motivo` es `risk_engine::ultimo_rechazo()`
    /// leído justo después de ella.
    #[inline]
    pub fn registrar_riesgo(&mut self, libro_plano: bool, aceptada: bool, motivo: Option<usize>) {
        let p = self.poblacion(libro_plano);
        if aceptada {
            p.riesgo_aceptadas = p.riesgo_aceptadas.saturating_add(1);
            return;
        }
        match motivo.and_then(|m| p.riesgo_rechazos.get_mut(m)) {
            Some(n) => *n = n.saturating_add(1),
            None => p.riesgo_sin_motivo = p.riesgo_sin_motivo.saturating_add(1),
        }
    }

    fn menos(&self, previa: &Self) -> Self {
        Self {
            solo_libro: self.solo_libro.saturating_sub(previa.solo_libro),
            panico_host: self.panico_host.saturating_sub(previa.panico_host),
            bloqueos: self.bloqueos.menos(&previa.bloqueos),
            libro_plano: self.libro_plano.menos(&previa.libro_plano),
            con_posicion: self.con_posicion.menos(&previa.con_posicion),
        }
    }
}

/// Copia de los contadores del embudo de un núcleo y del host en un instante.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct Instantanea {
    pub nucleo: ContadoresNucleo,
    /// Fusiones bayesianas evaluadas (tras la frontera de entradas).
    pub evaluaciones: u64,
    /// Intenciones de rama aplanadas por convicción no positiva (LG-1).
    pub sin_conviccion: u64,
    /// Intenciones unificadas con dirección y sin posición: [largo, corto].
    pub intenciones: [u64; 2],
    /// Vetos de cada escudo del embudo unificado (largo + corto).
    pub escudos: [u64; N_FUNNEL_STAGES],
    pub consejo: u64,
    pub ml_gate: u64,
    pub margen: u64,
    pub aperturas: u64,
    /// Host: órdenes del núcleo descartadas porque la fase no permite operar
    /// (orquestador en auditoría o refresco del OMS en curso).
    pub sin_permiso_de_fase: u64,
}

impl Instantanea {
    pub fn de(core: &GodEngineCore, sin_permiso_de_fase: u64) -> Self {
        let d = &core.diag_dir;
        let mut escudos = [0; N_FUNNEL_STAGES];
        for ((e, l), c) in escudos
            .iter_mut()
            .zip(d.funnel_veto[0])
            .zip(d.funnel_veto[1])
        {
            *e = l.saturating_add(c);
        }
        Self {
            nucleo: core.diag_embudo,
            evaluaciones: d.evaluations,
            sin_conviccion: core.diag_sin_conviccion,
            intenciones: d.funnel_entered,
            escudos,
            consejo: core.diag_council_vetoes,
            ml_gate: core.diag_ml_vetoes,
            margen: core.diag_margin_drops,
            aperturas: core.diag_opened,
            sin_permiso_de_fase,
        }
    }

    /// Lo ocurrido desde `previa` (contadores monótonos; resta saturada).
    pub fn menos(&self, previa: &Self) -> Self {
        Self {
            nucleo: self.nucleo.menos(&previa.nucleo),
            evaluaciones: self.evaluaciones.saturating_sub(previa.evaluaciones),
            sin_conviccion: self.sin_conviccion.saturating_sub(previa.sin_conviccion),
            intenciones: resta(self.intenciones, previa.intenciones),
            escudos: resta(self.escudos, previa.escudos),
            consejo: self.consejo.saturating_sub(previa.consejo),
            ml_gate: self.ml_gate.saturating_sub(previa.ml_gate),
            margen: self.margen.saturating_sub(previa.margen),
            aperturas: self.aperturas.saturating_sub(previa.aperturas),
            sin_permiso_de_fase: self
                .sin_permiso_de_fase
                .saturating_sub(previa.sin_permiso_de_fase),
        }
    }

    /// Cada etapa que pierde intenciones, en orden del pipeline, con su cuenta.
    fn perdidas(&self) -> Vec<(String, u64)> {
        let mut v: Vec<(String, u64)> = FUNNEL_STAGES
            .iter()
            .zip(self.escudos)
            .map(|(nombre, n)| (format!("escudo {nombre}"), n))
            .collect();
        // Sólo la población del embudo: con el libro plano.
        let plano = &self.nucleo.libro_plano;
        v.push(("sin ranura".to_string(), suma(&plano.sin_ranura)));
        v.extend(
            REJECT_NAMES
                .iter()
                .zip(plano.riesgo_rechazos)
                .map(|(nombre, n)| (format!("riesgo:{nombre}"), n)),
        );
        v.push(("riesgo:sin motivo".to_string(), plano.riesgo_sin_motivo));
        v.push(("consejo".to_string(), self.consejo));
        v.push(("ML-GATE".to_string(), self.ml_gate));
        v.push(("margen".to_string(), self.margen));
        v.push(("sin permiso de fase".to_string(), self.sin_permiso_de_fase));
        v
    }

    /// La etapa que más intenciones perdió (la primera del pipeline ante un
    /// empate). Sin pérdidas, la causa que más ticks bloqueó. `None` si nada.
    pub fn cuello(&self) -> Option<(String, u64)> {
        let mut mejor: Option<(String, u64)> = None;
        for (nombre, n) in self.perdidas() {
            if n > mejor.as_ref().map_or(0, |m| m.1) {
                mejor = Some((nombre, n));
            }
        }
        if mejor.is_some() {
            return mejor;
        }
        let b = &self.nucleo.bloqueos;
        let mut max: Option<(usize, u64)> = None;
        for (i, &n) in b.por_causa.iter().enumerate() {
            if n > max.map_or(0, |m| m.1) {
                max = Some((i, n));
            }
        }
        max.and_then(|(i, n)| {
            NOMBRES_BLOQUEO
                .get(i)
                .map(|nombre| (format!("ticks bloqueados: {nombre}"), n))
        })
    }
}

fn lista_no_nula(pares: impl IntoIterator<Item = (String, u64)>) -> String {
    let partes: Vec<String> = pares
        .into_iter()
        .filter(|(_, n)| *n > 0)
        .map(|(nombre, n)| format!("{nombre} {n}"))
        .collect();
    if partes.is_empty() {
        "ninguno".to_string()
    } else {
        partes.join(" · ")
    }
}

fn texto_cuello(i: &Instantanea) -> String {
    i.cuello().map_or("ninguno".to_string(), |(nombre, n)| {
        format!("{nombre} ({n})")
    })
}

/// Una línea: la ventana `v` de `ventana_ms` y el acumulado `total`.
pub fn linea(v: &Instantanea, total: &Instantanea, ventana_ms: u64) -> String {
    let n = &v.nucleo;
    let b = &n.bloqueos;
    let causas = lista_no_nula(NOMBRES_BLOQUEO.iter().zip(b.por_causa).enumerate().map(
        |(i, (nombre, cuenta))| {
            if i == BLOQUEO_LLAMADOR && cuenta > 0 {
                (
                    format!(
                        "{nombre} (sólo libro {} · pánico o calentamiento del host {})",
                        n.solo_libro, n.panico_host
                    ),
                    cuenta,
                )
            } else {
                (nombre.to_string(), cuenta)
            }
        },
    ));
    let escudos = lista_no_nula(
        FUNNEL_STAGES
            .iter()
            .zip(v.escudos)
            .map(|(nombre, cuenta)| (nombre.to_string(), cuenta)),
    );
    let motivos = |e: &EtapasPostEmbudo| {
        lista_no_nula(
            REJECT_NAMES
                .iter()
                .zip(e.riesgo_rechazos)
                .map(|(nombre, cuenta)| (nombre.to_string(), cuenta))
                .chain(std::iter::once((
                    "sin motivo".to_string(),
                    e.riesgo_sin_motivo,
                ))),
        )
    };
    let plano = &n.libro_plano;
    let otra = &n.con_posicion;
    let con_posicion = if suma(&otra.sin_ranura) > 0
        || otra.riesgo_aceptadas > 0
        || otra.rechazos() > 0
    {
        format!(
            " · con posición abierta (fuera del embudo): sin ranura {}, riesgo admite {}, rechaza {} ({})",
            suma(&otra.sin_ranura),
            otra.riesgo_aceptadas,
            otra.rechazos(),
            motivos(otra)
        )
    } else {
        String::new()
    };
    let tb = &total.nucleo.bloqueos;
    let pct_bloqueado = if tb.ticks > 0 {
        100.0 * tb.bloqueados() as f64 / tb.ticks as f64
    } else {
        0.0
    };
    format!(
        "🔭 [EMBUDO] {} s · ticks {} ({} bloqueados: {}) · evaluaciones {} · sin convicción {} · intenciones con el libro plano {} (L {} · C {}) → escudos: {} → sin ranura {} → riesgo: admite {}, rechaza {} ({}) → consejo {} → ML-GATE {} → margen {} → aperturas {} · sin permiso de fase {}{} · CUELLO: {} · acumulado: {} ticks ({:.1} % bloqueados), {} intenciones, {} aperturas, cuello {}",
        ventana_ms / 1_000,
        b.ticks,
        b.bloqueados(),
        causas,
        v.evaluaciones,
        v.sin_conviccion,
        suma(&v.intenciones),
        v.intenciones[0],
        v.intenciones[1],
        escudos,
        suma(&plano.sin_ranura),
        plano.riesgo_aceptadas,
        plano.rechazos(),
        motivos(plano),
        v.consejo,
        v.ml_gate,
        v.margen,
        v.aperturas,
        v.sin_permiso_de_fase,
        con_posicion,
        texto_cuello(v),
        tb.ticks,
        pct_bloqueado,
        suma(&total.intenciones),
        total.aperturas,
        texto_cuello(total),
    )
}

/// Cadencia del embudo vivo en el host: una línea por ventana de `periodo_ms`.
#[derive(Debug, Clone)]
pub struct EmbudoVivo {
    periodo_ms: u64,
    inicio_ventana_ms: Option<u64>,
    previa: Instantanea,
}

impl EmbudoVivo {
    pub fn new(periodo_ms: u64) -> Self {
        Self {
            periodo_ms: periodo_ms.max(1),
            inicio_ventana_ms: None,
            previa: Instantanea::default(),
        }
    }

    /// ¿Toca cerrar la ventana en `ahora_ms` (reloj monótono)? La primera
    /// llamada abre la primera ventana y no la cierra. O(1): el llamador sólo
    /// copia los contadores cuando toca.
    #[inline]
    pub fn toca(&mut self, ahora_ms: u64) -> bool {
        match self.inicio_ventana_ms {
            None => {
                self.inicio_ventana_ms = Some(ahora_ms);
                false
            }
            Some(inicio) => ahora_ms.saturating_sub(inicio) >= self.periodo_ms,
        }
    }

    /// Cierra la ventana: la línea de lo ocurrido desde la anterior, con el
    /// acumulado `actual`, y abre la siguiente.
    pub fn cerrar(&mut self, actual: Instantanea, ahora_ms: u64) -> String {
        let inicio = self.inicio_ventana_ms.unwrap_or(ahora_ms);
        let texto = linea(
            &actual.menos(&self.previa),
            &actual,
            ahora_ms.saturating_sub(inicio),
        );
        self.previa = actual;
        self.inicio_ventana_ms = Some(ahora_ms);
        texto
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::direction_diag::STAGE_BAYES;

    #[test]
    fn rb_e2_cada_tick_bloqueado_va_a_su_primera_causa() {
        let mut b = BloqueosEntrada::default();
        assert!(!b.registrar([false; CAUSAS_BLOQUEO]));
        let mut todas = [true; CAUSAS_BLOQUEO];
        assert!(b.registrar(todas), "todas activas: va a la primera");
        todas[BLOQUEO_LLAMADOR] = false;
        assert!(b.registrar(todas), "sin la primera: va a la segunda");
        let mut sola = [false; CAUSAS_BLOQUEO];
        sola[CAUSAS_BLOQUEO - 1] = true;
        assert!(b.registrar(sola));
        assert_eq!(b.ticks, 4);
        assert_eq!(b.por_causa, [1, 1, 0, 0, 0, 0, 1]);
        assert_eq!(b.bloqueados(), 3, "las causas parten los ticks bloqueados");
        // La devolución es exactamente la disyunción, para las 2^7 entradas.
        for mascara in 0u32..(1 << CAUSAS_BLOQUEO) {
            let mut c = [false; CAUSAS_BLOQUEO];
            for (k, x) in c.iter_mut().enumerate() {
                *x = mascara & (1 << k) != 0;
            }
            assert_eq!(
                b.registrar(c),
                c.iter().any(|&x| x),
                "máscara {mascara:07b}"
            );
        }
    }

    #[test]
    fn rb_e2_el_panico_del_host_se_atribuye_antes_que_el_evento_de_libro() {
        let mut c = ContadoresNucleo::default();
        c.registrar_llamada(true, true);
        c.registrar_llamada(true, false);
        c.registrar_llamada(false, false);
        assert_eq!((c.panico_host, c.solo_libro), (1, 1));
        c.registrar_sin_ranura(true, false);
        c.registrar_sin_ranura(true, true);
        c.registrar_sin_ranura(true, true);
        c.registrar_sin_ranura(false, false);
        assert_eq!(c.libro_plano.sin_ranura, [1, 2]);
        assert_eq!(
            c.con_posicion.sin_ranura,
            [1, 0],
            "cada población por su lado"
        );
        c.registrar_riesgo(true, true, Some(risk_engine::REJ_EV));
        c.registrar_riesgo(true, false, Some(risk_engine::REJ_CONFIDENCE));
        c.registrar_riesgo(true, false, None);
        c.registrar_riesgo(true, false, Some(REJECT_SLOTS));
        c.registrar_riesgo(false, false, Some(risk_engine::REJ_SONDA_OCUPADA));
        let p = c.libro_plano;
        assert_eq!(p.riesgo_aceptadas, 1, "admitida: el motivo se ignora");
        assert_eq!(p.riesgo_rechazos[risk_engine::REJ_CONFIDENCE], 1);
        assert_eq!(p.riesgo_sin_motivo, 2, "sin motivo o motivo fuera de rango");
        assert_eq!(p.rechazos(), 3);
        assert_eq!(
            c.con_posicion.riesgo_rechazos[risk_engine::REJ_SONDA_OCUPADA],
            1
        );
    }

    fn instantanea() -> Instantanea {
        let mut i = Instantanea {
            evaluaciones: 50,
            sin_conviccion: 9,
            intenciones: [12, 8],
            consejo: 3,
            ml_gate: 2,
            aperturas: 1,
            ..Instantanea::default()
        };
        i.escudos[STAGE_BAYES] = 4;
        i.nucleo.bloqueos.ticks = 1_000;
        i.nucleo.bloqueos.por_causa[BLOQUEO_LLAMADOR] = 900;
        i.nucleo.solo_libro = 880;
        i.nucleo.panico_host = 20;
        i.nucleo.libro_plano.riesgo_rechazos[risk_engine::REJ_CONFIDENCE] = 7;
        i.nucleo.libro_plano.riesgo_aceptadas = 6;
        i
    }

    #[test]
    fn rb_e2_la_ventana_es_la_resta_y_el_cuello_la_mayor_perdida() {
        let previa = instantanea();
        let mut actual = previa;
        actual.nucleo.bloqueos.ticks += 500;
        actual.nucleo.bloqueos.por_causa[5] += 30;
        actual.intenciones[0] += 6;
        actual.nucleo.libro_plano.riesgo_rechazos[risk_engine::REJ_EV] += 4;
        // Con posición abierta no compite por el cuello, por grande que sea.
        actual.nucleo.con_posicion.sin_ranura[0] += 1_000;
        actual.nucleo.con_posicion.riesgo_rechazos[risk_engine::REJ_SONDA_OCUPADA] += 900;
        actual.consejo += 2;
        let v = actual.menos(&previa);
        assert_eq!(v.nucleo.bloqueos.ticks, 500);
        assert_eq!(v.nucleo.bloqueos.bloqueados(), 30);
        assert_eq!(v.intenciones, [6, 0]);
        assert_eq!(v.cuello(), Some(("riesgo:EV".to_string(), 4)));
        let texto = linea(&v, &actual, 60_000);
        assert!(
            texto.contains(
                "con posición abierta (fuera del embudo): sin ranura 1000, riesgo admite 0, rechaza 900 (sonda_ocupada 900)"
            ),
            "{texto}"
        );
        assert_eq!(
            actual.cuello(),
            Some(("riesgo:confianza".to_string(), 7)),
            "en el acumulado manda la confianza"
        );
        // Resta saturada: un contador que retrocede no fabrica una pérdida.
        assert_eq!(previa.menos(&actual).nucleo.bloqueos.ticks, 0);
        // Empate: gana la etapa anterior en el pipeline.
        let mut empate = Instantanea::default();
        empate.consejo = 5;
        empate.ml_gate = 5;
        assert_eq!(empate.cuello(), Some(("consejo".to_string(), 5)));
    }

    #[test]
    fn rb_e2_sin_intenciones_el_cuello_es_el_bloqueo_de_ticks() {
        let mut i = Instantanea::default();
        assert_eq!(i.cuello(), None);
        i.nucleo.bloqueos.ticks = 100;
        i.nucleo.bloqueos.por_causa[1] = 10;
        i.nucleo.bloqueos.por_causa[5] = 60;
        assert_eq!(
            i.cuello(),
            Some(("ticks bloqueados: latencia sobre el umbral".to_string(), 60))
        );
    }

    #[test]
    fn rb_e2_la_linea_nombra_solo_lo_que_ocurrio() {
        let total = instantanea();
        let texto = linea(&total, &total, 60_000);
        println!("observado: {texto}");
        for parte in [
            "🔭 [EMBUDO] 60 s",
            "ticks 1000 (900 bloqueados: el llamador no las permite (sólo libro 880 · pánico o calentamiento del host 20) 900)",
            "intenciones con el libro plano 20 (L 12 · C 8)",
            "escudos: convicción bayesiana (D-472) 4",
            "riesgo: admite 6, rechaza 7 (confianza 7)",
            "consejo 3 → ML-GATE 2",
            "aperturas 1",
            "CUELLO: riesgo:confianza (7)",
            "acumulado: 1000 ticks (90.0 % bloqueados), 20 intenciones, 1 aperturas",
        ] {
            assert!(texto.contains(parte), "falta «{parte}» en: {texto}");
        }
        assert!(
            !texto.contains("kill switch"),
            "las causas a cero no aparecen"
        );
        let vacia = linea(&Instantanea::default(), &Instantanea::default(), 0);
        assert!(vacia.contains("escudos: ninguno") && vacia.contains("CUELLO: ninguno"));
    }

    #[test]
    fn rb_e2_una_linea_por_ventana() {
        let mut e = EmbudoVivo::new(60_000);
        assert!(!e.toca(5_000), "la primera llamada abre la ventana");
        assert!(!e.toca(64_999));
        assert!(e.toca(65_000));
        let mut actual = instantanea();
        let primera = e.cerrar(actual, 65_000);
        assert!(primera.contains("🔭 [EMBUDO] 60 s") && primera.contains("ticks 1000 "));
        assert!(!e.toca(124_999), "la ventana siguiente empieza al cerrar");
        assert!(e.toca(125_000));
        actual.nucleo.bloqueos.ticks += 10;
        let segunda = e.cerrar(actual, 125_000);
        assert!(segunda.contains("ticks 10 (0 bloqueados"), "{segunda}");
        assert!(segunda.contains("acumulado: 1010 ticks"), "{segunda}");
        assert_eq!(EmbudoVivo::new(0).periodo_ms, 1, "periodo mínimo de 1 ms");
    }
}
