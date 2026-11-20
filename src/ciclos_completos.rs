//! CICLOS-2 (pedido 20261118-0535-cicloscompletos; espec. `.agent_tx/diagnostics/auditoria_perdidas/
//! ciclos_completos_v2.md` §§1-3, decision del coordinador sobre §2.3: opcion C con D de respaldo,
//! cuadratura por igualdad exacta, sin tolerancia; la opcion E queda prohibida). CICLOS-3 (adenda
//! del coordinador, punto 4; QODER-APP C6): los importes son decimales EXACTOS CON SIGNO sobre el
//! texto del venue (`importe_venue::DecimalExacto`), sin escala elegida; las cantidades siguen en
//! la escala de cantidad (1e-8, la de `decimal_e8`), que nunca se aplica a dinero.
//!
//! QUE: reconstruye CICLOS COMPLETOS (de plano acreditado a plano acreditado, por entorno, venue,
//! cuenta y simbolo) a partir de los fills y de los cashflows del venue, y emite el certificado
//! v2 (`unidad = ciclo_completo`) que la puerta DemoVerify -> ProductionMainnet consume.
//! POR QUE: el certificado legado cuenta una fila `REALIZED_PNL` como un cierre: diez ciclos con
//! tres parciales cada uno son 30 "cierres" y abren la puerta (MIN_CIERRES_PARA_PROD = 30).
//! FAIL-CLOSED: cualquier dato critico sin acreditar deja el ciclo o el certificado en N/D con
//! su causa; nunca un cero inventado.
//!
//! CICLOS-4 (adenda 20261118-0547 del coordinador y N1 de QODER-APP `qoder_app_n1_ciclos_v2.md`,
//! Parte 5):
//! - C1 INVARIANTE DE DENOMINADOR: "dato critico" es cualquier partida de cualquier ciclo cerrado
//!   estructuralmente de la ventana. Si alguno tiene neto N/D, el certificado entero es N/D
//!   (`CicloNdEnDenominador`): los ciclos N/D NO se excluyen del denominador, bloquean el recibo.
//!   La media de los completos observables (`neto_total`) es descripcion y no alimenta la puerta.
//! - C2 DECISION (coordinador, 18 nov 03:20): C primaria (importes explicitos del venue para el
//!   tramo de cierre y el de apertura, sin prorrateo local), D de respaldo (N/D si C no alcanza);
//!   A, B y E descartadas y prohibido reintroducirlas. Con un solo fill y una sola fila de
//!   cashflow por tradeId, C no se satisface y el ciclo es N/D. El adaptador conserva la
//!   granularidad POR FILA (identidad `tranId`); no usa el agrupador contable de
//!   `telemetry-server/src/trade_ledger.rs` (884-891, 931-936), que suma las filas de un tradeId.
//! - C3 CUADRATURA por IGUALDAD EXACTA, sin tolerancia: walletBalance(T1) - walletBalance(T0) ==
//!   suma de netos de los ciclos + flujos externos, en decimal exacto. Solo esta definida en la
//!   cohorte USDT unico sin FX (fuera de ella, N/D); si falla con datos reales, N/D con la causa
//!   publicada y la puerta cerrada. NINGUN implementador puede introducir una tolerancia: que el
//!   saldo del venue cuadre exacto es una hipotesis a verificar empiricamente, no un hecho.
//! - C5: no se usan ni para probar plano ni para cuadrar `user_data_stream.rs:1115`
//!   (|c| <= 1e-8 => Cerrada), `puente_cuenta.rs:355-357` (`mismo_saldo`, relativa 1e-9 con piso
//!   1,0) ni `state_validator.rs:62-77` (0,01 y 1,0 absolutos; kill switch comentado en :78). El
//!   plano es cero exacto.
//! - C6/C7: importes en `DecimalExacto` (nunca `decimal_e8`, que es de cantidad y sin signo);
//!   activo ausente nunca es USDT (no se usa el lector legado `parse_income_jsonl`).
//! - C10 COSTE ACEPTADO: C+D con igualdad exacta puede dejar la puerta cerrada de forma
//!   indefinida. Es el comportamiento deseado; la via para ampliar la elegibilidad es mas
//!   evidencia del venue, NO una tolerancia ni el regreso a A/B. Las filas, el freno de Produccion (minimo 10 filas), el
//! profiler, el panel y el freno estadistico por simbolo NO cambian: siguen leyendo filas.
use std::collections::{BTreeMap, BTreeSet, HashMap};
use std::path::{Path, PathBuf};

use execution_engine::importe_venue::{CausaImporteNd, DecimalExacto, ImporteVenue};
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};

pub const VERSION_CERTIFICADO: u32 = 2;
pub const UNIDAD: &str = "ciclo_completo";
/// Destino v2, distinto del legado `evidencia_demo.json` (que el binario anterior lee).
pub const ARCHIVO_CERTIFICADO_V2: &str = "evidencia_ciclos_v2.json";
/// La puerta solo admite certificados de la Demo en Testnet.
pub const ENTORNO_TESTNET: &str = "testnet";
pub const ACTIVO_DE_CUENTA: &str = "USDT";
pub const ESTADO_VALIDO: &str = "VALIDO";
pub const ESTADO_ND: &str = "ND";
pub const ESTADO_REVOCADO: &str = "REVOCADO";
/// Marca de revocacion con hash ilegible: retiene todo certificado hasta que el operador la retire.
pub const HASH_DESCONOCIDO: &str = "DESCONOCIDO";

/// Libro de posicion: la historia de un simbolo en una cuenta, un venue y un entorno.
#[derive(Debug, Clone, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct Libro {
    pub entorno: String,
    pub venue: String,
    pub cuenta: String,
    pub simbolo: String,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Lado {
    Compra,
    Venta,
}

/// `positionSide` crudo del venue.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Pierna {
    Both,
    Long,
    Short,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Canal {
    Ws,
    Rest,
}

/// Etiqueta de causa de salida (por identidad de intento y evidencia del venue). No cambia el
/// limite del ciclo: solo el cero acreditado lo cierra.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CausaSalida {
    Normal,
    KillSwitch,
    Liquidacion,
    Adl,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ModoPosicion {
    OneWay,
    Hedge,
}

/// Un fill canonico, identificado por `(libro, trade_id)`. `qty_e8` es la cantidad DE ESE FILL
/// (`l`), nunca la acumulada. `rp` conserva su presencia: ausente o ilegible es N/D.
#[derive(Debug, Clone, PartialEq)]
pub struct FillVenue {
    pub libro: Libro,
    pub trade_id: u64,
    pub order_id: u64,
    /// `None`: lado desconocido.
    pub lado: Option<Lado>,
    pub pierna: Pierna,
    /// `None`: cantidad ilegible.
    pub qty_e8: Option<i64>,
    pub precio_e8: Option<i64>,
    pub tiempo_ms: u64,
    pub rp: Result<ImporteVenue, CausaImporteNd>,
    pub canal: Canal,
    pub causa: CausaSalida,
}

impl FillVenue {
    /// Misma evidencia salvo el canal (un replay identico WS/REST es idempotente).
    fn misma_evidencia(&self, o: &Self) -> bool {
        self.libro == o.libro
            && self.trade_id == o.trade_id
            && self.order_id == o.order_id
            && self.lado == o.lado
            && self.pierna == o.pierna
            && self.qty_e8 == o.qty_e8
            && self.precio_e8 == o.precio_e8
            && self.tiempo_ms == o.tiempo_ms
            && self.rp == o.rp
            && self.causa == o.causa
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum TipoCaja {
    RealizedPnl,
    Comision,
    Funding,
    /// Cargo identificable de liquidacion o ADL no clasificado en otra partida.
    CargoForzoso,
    /// Flujo externo de la cuenta (TRANSFER y similares): no es de ningun ciclo; entra solo en la
    /// cuadratura con el saldo.
    Externo,
}

/// Tramo de una reversion one-way, cuando el venue entrega el importe de cada tramo (opcion C).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Tramo {
    Cierre,
    Apertura,
}

/// Una fila de cashflow del venue (income), con su importe en punto fijo.
#[derive(Debug, Clone, PartialEq)]
pub struct CajaVenue {
    pub libro: Libro,
    pub tipo: TipoCaja,
    pub tran_id: u64,
    pub trade_id: Option<u64>,
    pub tiempo_ms: u64,
    pub asset: String,
    pub importe: Result<ImporteVenue, CausaImporteNd>,
    pub tramo: Option<Tramo>,
}

/// Foto remota de posicion de un simbolo (cantidades exactas por pierna).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct Foto {
    pub both_e8: i64,
    pub long_e8: i64,
    pub short_e8: i64,
}

impl Foto {
    pub fn plana(&self) -> bool {
        self.both_e8 == 0 && self.long_e8 == 0 && self.short_e8 == 0
    }
}

/// `walletBalance` de la cuenta en un borde, con su activo y el texto del venue.
#[derive(Debug, Clone, PartialEq)]
pub struct SaldoVenue {
    pub asset: String,
    pub importe: Result<ImporteVenue, CausaImporteNd>,
}

/// Lo que fija el recibo: cuenta Testnet, sesion, estrategia y SHA del binario y la configuracion.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct AmbitoCertificado {
    pub entorno: String,
    pub venue: String,
    pub cuenta: String,
    pub sesion: String,
    pub estrategia: String,
    pub sha_binario: String,
    pub sha_config: String,
    pub roster: Vec<String>,
}

/// Entrada del productor: todo inyectado (sin red, disco ni reloj).
#[derive(Debug, Clone)]
pub struct EntradaCiclos {
    pub ambito: AmbitoCertificado,
    pub modo: ModoPosicion,
    pub ventana_desde_ms: u64,
    pub ventana_hasta_ms: u64,
    pub fills: Vec<FillVenue>,
    pub caja: Vec<CajaVenue>,
    /// Foto completa de la cuenta en T0 (`None`: sin prueba de posicion de borde). Un simbolo que
    /// no aparece en una foto presente esta plano.
    pub foto_t0: Option<BTreeMap<String, Foto>>,
    pub foto_t1: Option<BTreeMap<String, Foto>>,
    /// Paginacion, continuidad y reconciliacion de fills e income acreditadas.
    pub cobertura_fills_completa: bool,
    /// Cobertura de funding comprobada: la ausencia de filas FUNDING_FEE vale cero acreditado.
    pub cobertura_funding_completa: bool,
    /// Saldo de la cuenta en T0 y en T1 (`None`: sin saldo de borde, cuadratura N/D).
    pub saldo_t0: Option<SaldoVenue>,
    pub saldo_t1: Option<SaldoVenue>,
}

/// Causas N/D (del ciclo o del certificado).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub enum CausaCiclosNd {
    FotoBordeAusente,
    CoberturaFillsIncompleta,
    DuplicadoContradictorio,
    CajaSinIdentidad,
    CajaSinFill,
    LadoOCantidadDesconocida,
    ModoInconsistente,
    SobreReduccionHedge,
    DescuadreFotoT1,
    BordeNoPlano,
    RpAusenteOIlegible,
    TestigosPnlContradictorios,
    ComisionSinEvidencia,
    ImporteIlegible,
    MonedaNoUsdt,
    FundingSinCobertura,
    FundingAmbiguo,
    FundingSinExposicion,
    CargoForzosoIlegible,
    ReversionSinImportesExplicitos,
    Desbordamiento,
    /// C1: un ciclo cerrado estructuralmente tiene neto N/D: bloquea el recibo entero.
    CicloNdEnDenominador,
    /// Falta el saldo de un borde o es ilegible.
    SaldoBordeAusente,
    /// Fuera de la cohorte USDT unico sin FX la cuadratura no esta definida.
    CuadraturaFueraDeCohorte,
    /// Algun ciclo, arrastre o abierto impide sumar todos los netos.
    CuadraturaNoEvaluable,
    /// Delta de saldo distinto (por exacto) de netos + externos.
    CuadraturaDescuadrada,
}

/// Resumen de un ciclo cerrado dentro de la ventana (importes decimales exactos del venue, firmados
/// como flujo; se serializan como texto decimal canonico).
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ResumenCiclo {
    pub simbolo: String,
    pub apertura_trade_id: u64,
    pub cierre_trade_id: u64,
    pub fills: u32,
    pub bruto: DecimalExacto,
    pub comision: DecimalExacto,
    pub funding: DecimalExacto,
    pub cargos: DecimalExacto,
    /// `None`: N/D (ver `causas_nd`).
    pub neto: Option<DecimalExacto>,
    pub causas_nd: Vec<String>,
}

/// Certificado v2 de ciclos completos.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct CertificadoCiclosV2 {
    pub version: u32,
    pub unidad: String,
    pub ambito: AmbitoCertificado,
    pub modo: String,
    pub ventana_desde_ms: u64,
    pub ventana_hasta_ms: u64,
    /// Diagnostico: filas REALIZED_PNL del ambito y la ventana (la unidad legada).
    pub filas_realized_pnl: u64,
    pub ciclos_estructurales: u64,
    pub n_ciclos_validos: u64,
    pub ciclos_nd: u64,
    pub arrastres: u64,
    pub abiertos: u64,
    pub ganadores: u64,
    /// Suma de los netos acreditados (descriptiva si el estado es ND: no alimenta la puerta).
    pub neto_total: DecimalExacto,
    /// Saldos de borde (texto exacto del venue) y flujos externos de la ventana: la cuadratura.
    pub saldo_t0: Option<DecimalExacto>,
    pub saldo_t1: Option<DecimalExacto>,
    pub externos: DecimalExacto,
    /// EV de certificacion por ciclo: `None` si el estado no es VALIDO o no hay ciclos.
    pub ev_neto_por_ciclo: Option<f64>,
    pub causas_nd: Vec<String>,
    pub ciclos: Vec<ResumenCiclo>,
    /// sha256 del ambito, la ventana y los ciclos: respaldo del recibo.
    pub hash_ciclos: String,
    pub estado: String,
    pub motivo_revocacion: Option<String>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Rol {
    Apertura,
    Aumento,
    Reduccion,
    ReversionCierre,
    ReversionApertura,
}

#[derive(Debug, Clone)]
struct Ciclo {
    libro: Libro,
    arrastre: bool,
    t_apertura: u64,
    t_cierre: Option<u64>,
    fills: Vec<(usize, Rol)>,
    causas: BTreeSet<CausaCiclosNd>,
    comision: DecimalExacto,
    funding: DecimalExacto,
    cargos: DecimalExacto,
}

impl Ciclo {
    fn nuevo(libro: &Libro, arrastre: bool, t: u64) -> Self {
        Self {
            libro: libro.clone(),
            arrastre,
            t_apertura: t,
            t_cierre: None,
            fills: Vec::new(),
            causas: BTreeSet::new(),
            comision: DecimalExacto::CERO,
            funding: DecimalExacto::CERO,
            cargos: DecimalExacto::CERO,
        }
    }
}

fn sumar(acc: &mut DecimalExacto, x: DecimalExacto, causas: &mut BTreeSet<CausaCiclosNd>) {
    match acc.checked_add(x) {
        Some(v) => *acc = v,
        None => {
            causas.insert(CausaCiclosNd::Desbordamiento);
        }
    }
}

fn del_ambito(l: &Libro, a: &AmbitoCertificado) -> bool {
    l.entorno == a.entorno && l.venue == a.venue && l.cuenta == a.cuenta
}

fn en_ventana(t: u64, e: &EntradaCiclos) -> bool {
    t >= e.ventana_desde_ms && t < e.ventana_hasta_ms
}

/// Estado de la posicion de un libro.
#[derive(Debug, Clone, Copy, Default)]
struct Posicion {
    both: i64,
    long: i64,
    short: i64,
}

impl Posicion {
    fn plana(&self) -> bool {
        self.both == 0 && self.long == 0 && self.short == 0
    }
    fn foto(&self) -> Foto {
        Foto { both_e8: self.both, long_e8: self.long, short_e8: self.short }
    }
}

/// Reconstruye los ciclos y emite el certificado v2 (sin escribirlo).
pub fn certificar_ciclos(e: &EntradaCiclos) -> CertificadoCiclosV2 {
    let mut globales: BTreeSet<CausaCiclosNd> = BTreeSet::new();
    if e.foto_t0.is_none() || e.foto_t1.is_none() {
        globales.insert(CausaCiclosNd::FotoBordeAusente);
    }
    if !e.cobertura_fills_completa {
        globales.insert(CausaCiclosNd::CoberturaFillsIncompleta);
    }

    // Fills del ambito y la ventana, deduplicados por (libro, trade_id).
    let mut por_clave: HashMap<(Libro, u64), FillVenue> = HashMap::new();
    for f in e.fills.iter().filter(|f| del_ambito(&f.libro, &e.ambito) && en_ventana(f.tiempo_ms, e)) {
        match por_clave.get(&(f.libro.clone(), f.trade_id)) {
            Some(previo) if !previo.misma_evidencia(f) => {
                globales.insert(CausaCiclosNd::DuplicadoContradictorio);
            }
            Some(_) => {}
            None => {
                por_clave.insert((f.libro.clone(), f.trade_id), f.clone());
            }
        }
    }
    let mut fills: Vec<FillVenue> = por_clave.into_values().collect();
    // Orden por evidencia del venue: tiempo y, en el empate, tradeId.
    fills.sort_by(|a, b| (&a.libro, a.tiempo_ms, a.trade_id).cmp(&(&b.libro, b.tiempo_ms, b.trade_id)));

    // Cashflows del ambito y la ventana, deduplicados por (libro, tipo, tranId).
    let mut caja_clave: HashMap<(Libro, TipoCaja, u64), CajaVenue> = HashMap::new();
    for c in e.caja.iter().filter(|c| del_ambito(&c.libro, &e.ambito) && en_ventana(c.tiempo_ms, e)) {
        if c.tran_id == 0 {
            globales.insert(CausaCiclosNd::CajaSinIdentidad);
            continue;
        }
        match caja_clave.get(&(c.libro.clone(), c.tipo, c.tran_id)) {
            Some(previo) if previo != c => {
                globales.insert(CausaCiclosNd::DuplicadoContradictorio);
            }
            Some(_) => {}
            None => {
                caja_clave.insert((c.libro.clone(), c.tipo, c.tran_id), c.clone());
            }
        }
    }
    let mut caja: Vec<CajaVenue> = caja_clave.into_values().collect();
    caja.sort_by(|a, b| (&a.libro, a.tiempo_ms, a.tran_id).cmp(&(&b.libro, b.tiempo_ms, b.tran_id)));

    // Libros: los de los fills y los no planos de las fotos.
    let mut libros: BTreeSet<Libro> = fills.iter().map(|f| f.libro.clone()).collect();
    let libro_de = |s: &String| Libro {
        entorno: e.ambito.entorno.clone(),
        venue: e.ambito.venue.clone(),
        cuenta: e.ambito.cuenta.clone(),
        simbolo: s.clone(),
    };
    for foto in [&e.foto_t0, &e.foto_t1].into_iter().flatten() {
        for (s, f) in foto {
            if !f.plana() {
                libros.insert(libro_de(s));
            }
        }
    }

    let mut ciclos: Vec<Ciclo> = Vec::new();
    // Ciclo(s) de cada fill (una reversion pertenece a dos).
    let mut ciclos_del_fill: Vec<Vec<(usize, Rol)>> = vec![Vec::new(); fills.len()];
    for libro in &libros {
        let inicial = e.foto_t0.as_ref().and_then(|m| m.get(&libro.simbolo)).copied().unwrap_or_default();
        let mut pos = Posicion { both: inicial.both_e8, long: inicial.long_e8, short: inicial.short_e8 };
        let mut abierto: Option<usize> = None;
        if !pos.plana() {
            ciclos.push(Ciclo::nuevo(libro, true, e.ventana_desde_ms));
            abierto = Some(ciclos.len() - 1);
        }
        for (i, f) in fills.iter().enumerate().filter(|(_, f)| &f.libro == libro) {
            let (Some(lado), Some(q)) = (f.lado, f.qty_e8.filter(|q| *q > 0)) else {
                globales.insert(CausaCiclosNd::LadoOCantidadDesconocida);
                break;
            };
            let previa = pos;
            match (e.modo, f.pierna) {
                (ModoPosicion::OneWay, Pierna::Both) => {
                    let d = if lado == Lado::Compra { q } else { -q };
                    let Some(nueva) = previa.both.checked_add(d) else {
                        globales.insert(CausaCiclosNd::Desbordamiento);
                        break;
                    };
                    pos.both = nueva;
                    if previa.both == 0 {
                        ciclos.push(Ciclo::nuevo(libro, false, f.tiempo_ms));
                        let c = ciclos.len() - 1;
                        abierto = Some(c);
                        ciclos[c].fills.push((i, Rol::Apertura));
                        ciclos_del_fill[i].push((c, Rol::Apertura));
                    } else if nueva != 0 && (previa.both > 0) != (nueva > 0) {
                        // Reversion: cierre del tramo previo y apertura del nuevo con el mismo fill.
                        if let Some(c) = abierto {
                            ciclos[c].fills.push((i, Rol::ReversionCierre));
                            ciclos[c].t_cierre = Some(f.tiempo_ms);
                            ciclos_del_fill[i].push((c, Rol::ReversionCierre));
                        }
                        ciclos.push(Ciclo::nuevo(libro, false, f.tiempo_ms));
                        let n = ciclos.len() - 1;
                        abierto = Some(n);
                        ciclos[n].fills.push((i, Rol::ReversionApertura));
                        ciclos_del_fill[i].push((n, Rol::ReversionApertura));
                    } else {
                        let rol = if nueva.abs() > previa.both.abs() { Rol::Aumento } else { Rol::Reduccion };
                        if let Some(c) = abierto {
                            ciclos[c].fills.push((i, rol));
                            ciclos_del_fill[i].push((c, rol));
                            if nueva == 0 {
                                ciclos[c].t_cierre = Some(f.tiempo_ms);
                                abierto = None;
                            }
                        }
                    }
                }
                (ModoPosicion::Hedge, Pierna::Long | Pierna::Short) => {
                    let sube = (f.pierna == Pierna::Long) == (lado == Lado::Compra);
                    let pierna = if f.pierna == Pierna::Long { &mut pos.long } else { &mut pos.short };
                    let Some(nueva) = (if sube { pierna.checked_add(q) } else { pierna.checked_sub(q) }) else {
                        globales.insert(CausaCiclosNd::Desbordamiento);
                        break;
                    };
                    if nueva < 0 {
                        globales.insert(CausaCiclosNd::SobreReduccionHedge);
                        break;
                    }
                    *pierna = nueva;
                    if previa.plana() {
                        ciclos.push(Ciclo::nuevo(libro, false, f.tiempo_ms));
                        abierto = Some(ciclos.len() - 1);
                    }
                    let rol = if previa.plana() {
                        Rol::Apertura
                    } else if sube {
                        Rol::Aumento
                    } else {
                        Rol::Reduccion
                    };
                    if let Some(c) = abierto {
                        ciclos[c].fills.push((i, rol));
                        ciclos_del_fill[i].push((c, rol));
                        if pos.plana() {
                            ciclos[c].t_cierre = Some(f.tiempo_ms);
                            abierto = None;
                        }
                    }
                }
                _ => {
                    globales.insert(CausaCiclosNd::ModoInconsistente);
                    break;
                }
            }
        }
        let final_t1 = e.foto_t1.as_ref().and_then(|m| m.get(&libro.simbolo)).copied().unwrap_or_default();
        if e.foto_t1.is_some() && final_t1 != pos.foto() {
            globales.insert(CausaCiclosNd::DescuadreFotoT1);
        }
    }

    // Bruto: rp del fill, conciliado con la fila REALIZED_PNL del mismo trade (una sola partida).
    let mut indice_fill: HashMap<(Libro, u64), usize> = HashMap::new();
    for (i, f) in fills.iter().enumerate() {
        indice_fill.insert((f.libro.clone(), f.trade_id), i);
    }
    let mut pnl_de_fila: HashMap<usize, Vec<&CajaVenue>> = HashMap::new();
    let mut comision_de_fila: HashMap<usize, Vec<&CajaVenue>> = HashMap::new();
    let mut filas_realized_pnl: u64 = 0;
    let mut externos = DecimalExacto::CERO;
    for c in &caja {
        if c.asset != ACTIVO_DE_CUENTA {
            globales.insert(CausaCiclosNd::MonedaNoUsdt);
        }
        match c.tipo {
            TipoCaja::Externo => match &c.importe {
                Ok(imp) => sumar(&mut externos, imp.valor, &mut globales),
                Err(_) => {
                    globales.insert(CausaCiclosNd::ImporteIlegible);
                }
            },
            TipoCaja::RealizedPnl | TipoCaja::Comision => {
                if c.tipo == TipoCaja::RealizedPnl {
                    filas_realized_pnl += 1;
                }
                let Some(i) = c.trade_id.and_then(|t| indice_fill.get(&(c.libro.clone(), t))).copied() else {
                    globales.insert(CausaCiclosNd::CajaSinFill);
                    continue;
                };
                let destino = if c.tipo == TipoCaja::RealizedPnl { &mut pnl_de_fila } else { &mut comision_de_fila };
                destino.entry(i).or_default().push(c);
            }
            TipoCaja::Funding | TipoCaja::CargoForzoso => {
                let por_trade = c.trade_id.and_then(|t| indice_fill.get(&(c.libro.clone(), t))).copied();
                let mut ambiguos: Vec<usize> = Vec::new();
                let destino: Vec<usize> = match por_trade {
                    Some(i) => ciclos_del_fill[i].iter().map(|(k, _)| *k).take(1).collect(),
                    None => {
                        // Por exposicion en el instante de liquidacion; en el borde de un ciclo
                        // (mismo instante que su primer o su ultimo fill) no se adjudica: N/D.
                        let mut en: Vec<usize> = Vec::new();
                        for (k, ci) in ciclos.iter().enumerate().filter(|(_, ci)| ci.libro == c.libro) {
                            let tc = ci.t_cierre.unwrap_or(u64::MAX);
                            let borde = (!ci.arrastre && c.tiempo_ms == ci.t_apertura) || c.tiempo_ms == tc;
                            if borde {
                                ambiguos.push(k);
                            } else if c.tiempo_ms > ci.t_apertura && c.tiempo_ms < tc {
                                en.push(k);
                            }
                        }
                        en
                    }
                };
                if !ambiguos.is_empty() {
                    for k in ambiguos {
                        ciclos[k].causas.insert(CausaCiclosNd::FundingAmbiguo);
                    }
                    continue;
                }
                let Some(&k) = destino.first() else {
                    globales.insert(CausaCiclosNd::FundingSinExposicion);
                    continue;
                };
                match &c.importe {
                    Ok(imp) => {
                        let ci = &mut ciclos[k];
                        if c.tipo == TipoCaja::Funding {
                            sumar(&mut ci.funding, imp.valor, &mut ci.causas);
                        } else {
                            sumar(&mut ci.cargos, imp.valor, &mut ci.causas);
                        }
                    }
                    Err(_) => {
                        let causa = if c.tipo == TipoCaja::Funding {
                            CausaCiclosNd::ImporteIlegible
                        } else {
                            CausaCiclosNd::CargoForzosoIlegible
                        };
                        ciclos[k].causas.insert(causa);
                    }
                }
            }
        }
    }

    // Por fill: bruto (rp) y comision (filas explicitas del venue, incluido "0").
    let mut bruto_ciclo: Vec<DecimalExacto> = vec![DecimalExacto::CERO; ciclos.len()];
    for (i, f) in fills.iter().enumerate() {
        let destinos = ciclos_del_fill[i].clone();
        if destinos.is_empty() {
            continue;
        }
        // rp: todo al tramo de cierre (en una reversion, al ciclo que cierra).
        let k_rp = destinos.iter().find(|(_, r)| *r != Rol::ReversionApertura).map(|(k, _)| *k).unwrap_or(destinos[0].0);
        match &f.rp {
            Ok(rp) => {
                for fila in pnl_de_fila.get(&i).into_iter().flatten() {
                    match &fila.importe {
                        Ok(v) if v.valor == rp.valor => {}
                        Ok(_) => {
                            ciclos[k_rp].causas.insert(CausaCiclosNd::TestigosPnlContradictorios);
                        }
                        Err(_) => {
                            ciclos[k_rp].causas.insert(CausaCiclosNd::ImporteIlegible);
                        }
                    }
                }
                let mut causas = std::mem::take(&mut ciclos[k_rp].causas);
                sumar(&mut bruto_ciclo[k_rp], rp.valor, &mut causas);
                ciclos[k_rp].causas = causas;
            }
            Err(_) => {
                for (k, _) in &destinos {
                    ciclos[*k].causas.insert(CausaCiclosNd::RpAusenteOIlegible);
                }
            }
        }
        let filas = comision_de_fila.get(&i).cloned().unwrap_or_default();
        if filas.is_empty() {
            for (k, _) in &destinos {
                ciclos[*k].causas.insert(CausaCiclosNd::ComisionSinEvidencia);
            }
            continue;
        }
        let ilegible = filas.iter().any(|c| c.importe.is_err());
        if ilegible {
            for (k, _) in &destinos {
                ciclos[*k].causas.insert(CausaCiclosNd::ImporteIlegible);
            }
            continue;
        }
        let importe = |c: &CajaVenue| c.importe.as_ref().map(|v| v.valor).unwrap_or(DecimalExacto::CERO);
        if destinos.len() > 1 {
            // Reversion: opcion C (importe explicito del venue por tramo) o D (N/D).
            let todos_con_tramo = filas.iter().all(|c| c.tramo.is_some());
            let hay_cierre = filas.iter().any(|c| c.tramo == Some(Tramo::Cierre));
            let hay_apertura = filas.iter().any(|c| c.tramo == Some(Tramo::Apertura));
            if !(todos_con_tramo && hay_cierre && hay_apertura) {
                for (k, _) in &destinos {
                    ciclos[*k].causas.insert(CausaCiclosNd::ReversionSinImportesExplicitos);
                }
                continue;
            }
            for (k, rol) in &destinos {
                let tramo = if *rol == Rol::ReversionApertura { Tramo::Apertura } else { Tramo::Cierre };
                let mut causas = std::mem::take(&mut ciclos[*k].causas);
                for c in filas.iter().filter(|c| c.tramo == Some(tramo)) {
                    sumar(&mut ciclos[*k].comision, importe(*c), &mut causas);
                }
                ciclos[*k].causas = causas;
            }
        } else {
            let k = destinos[0].0;
            let mut causas = std::mem::take(&mut ciclos[k].causas);
            for c in &filas {
                sumar(&mut ciclos[k].comision, importe(*c), &mut causas);
            }
            ciclos[k].causas = causas;
        }
    }
    if !e.cobertura_funding_completa {
        for c in ciclos.iter_mut() {
            c.causas.insert(CausaCiclosNd::FundingSinCobertura);
        }
    }

    // Resumen.
    let mut resumenes: Vec<ResumenCiclo> = Vec::new();
    let (mut arrastres, mut abiertos) = (0u64, 0u64);
    for (k, c) in ciclos.iter().enumerate() {
        if c.arrastre {
            arrastres += 1;
            continue;
        }
        if c.t_cierre.is_none() {
            abiertos += 1;
            continue;
        }
        let mut causas = c.causas.clone();
        let mut neto = bruto_ciclo[k];
        for x in [c.comision, c.funding, c.cargos] {
            sumar(&mut neto, x, &mut causas);
        }
        let primero = c.fills.first().map(|(i, _)| fills[*i].trade_id).unwrap_or(0);
        let ultimo = c.fills.last().map(|(i, _)| fills[*i].trade_id).unwrap_or(0);
        resumenes.push(ResumenCiclo {
            simbolo: c.libro.simbolo.clone(),
            apertura_trade_id: primero,
            cierre_trade_id: ultimo,
            fills: c.fills.len() as u32,
            bruto: bruto_ciclo[k],
            comision: c.comision,
            funding: c.funding,
            cargos: c.cargos,
            neto: causas.is_empty().then_some(neto),
            causas_nd: causas.iter().map(|x| format!("{x:?}")).collect(),
        });
    }
    if arrastres > 0 || abiertos > 0 {
        globales.insert(CausaCiclosNd::BordeNoPlano);
    }
    let modo = match e.modo {
        ModoPosicion::OneWay => "one_way",
        ModoPosicion::Hedge => "hedge",
    };
    let mut cert = CertificadoCiclosV2 {
        version: VERSION_CERTIFICADO,
        unidad: UNIDAD.to_string(),
        ambito: e.ambito.clone(),
        modo: modo.to_string(),
        ventana_desde_ms: e.ventana_desde_ms,
        ventana_hasta_ms: e.ventana_hasta_ms,
        filas_realized_pnl,
        ciclos_estructurales: 0,
        n_ciclos_validos: 0,
        ciclos_nd: 0,
        arrastres,
        abiertos,
        ganadores: 0,
        neto_total: DecimalExacto::CERO,
        saldo_t0: e.saldo_t0.as_ref().and_then(|s| s.importe.as_ref().ok()).map(|i| i.valor),
        saldo_t1: e.saldo_t1.as_ref().and_then(|s| s.importe.as_ref().ok()).map(|i| i.valor),
        externos,
        ev_neto_por_ciclo: None,
        causas_nd: globales.iter().map(|x| format!("{x:?}")).collect(),
        ciclos: resumenes,
        hash_ciclos: String::new(),
        estado: String::new(),
        motivo_revocacion: None,
    };
    let (n, neto, ganadores, desborde) = contar(&cert.ciclos);
    if desborde {
        cert.causas_nd.push(format!("{:?}", CausaCiclosNd::Desbordamiento));
    }
    cert.ciclos_estructurales = cert.ciclos.len() as u64;
    cert.n_ciclos_validos = n;
    cert.ciclos_nd = cert.ciclos_estructurales - n;
    cert.ganadores = ganadores;
    cert.neto_total = neto;
    // C1: un ciclo estructural con neto N/D bloquea el recibo entero (no sale del denominador).
    if cert.ciclos_nd > 0 {
        cert.causas_nd.push(format!("{:?}", CausaCiclosNd::CicloNdEnDenominador));
    }
    // C3: cuadratura por igualdad exacta, solo en la cohorte USDT unico sin FX.
    if let Some(causa) = causa_cuadratura(e, &caja, &cert) {
        cert.causas_nd.push(format!("{causa:?}"));
    }
    let valido = cert.causas_nd.is_empty() && cert.ciclos_nd == 0;
    cert.estado = if valido { ESTADO_VALIDO } else { ESTADO_ND }.to_string();
    cert.ev_neto_por_ciclo = if valido { ev_por_ciclo(neto, n) } else { None };
    cert.hash_ciclos = hash_de(&cert);
    cert
}

/// C3: `None` si la cuadratura es exacta; si no, la causa N/D. Sin tolerancia de ningun tipo.
fn causa_cuadratura(e: &EntradaCiclos, caja: &[CajaVenue], cert: &CertificadoCiclosV2) -> Option<CausaCiclosNd> {
    let saldos = [&e.saldo_t0, &e.saldo_t1];
    let fuera = caja.iter().any(|c| c.asset != ACTIVO_DE_CUENTA)
        || saldos.iter().any(|s| s.as_ref().is_some_and(|s| s.asset != ACTIVO_DE_CUENTA));
    if fuera {
        return Some(CausaCiclosNd::CuadraturaFueraDeCohorte);
    }
    let (Some(s0), Some(s1)) = (cert.saldo_t0, cert.saldo_t1) else {
        return Some(CausaCiclosNd::SaldoBordeAusente);
    };
    if cert.arrastres > 0 || cert.abiertos > 0 || cert.ciclos.iter().any(|c| c.neto.is_none()) {
        return Some(CausaCiclosNd::CuadraturaNoEvaluable);
    }
    match cuadra(s0, s1, cert.neto_total, cert.externos) {
        Some(true) => None,
        Some(false) => Some(CausaCiclosNd::CuadraturaDescuadrada),
        None => Some(CausaCiclosNd::Desbordamiento),
    }
}

/// Igualdad exacta entre delta de saldo y netos más externos. `None` si desborda una operación.
fn cuadra(s0: DecimalExacto, s1: DecimalExacto, neto: DecimalExacto, externos: DecimalExacto) -> Option<bool> {
    Some(s1.checked_sub(s0)? == neto.checked_add(externos)?)
}

/// (validos, neto total, ganadores por X > 0, desbordamiento).
fn contar(ciclos: &[ResumenCiclo]) -> (u64, DecimalExacto, u64, bool) {
    let (mut n, mut neto, mut g, mut desborde) = (0u64, DecimalExacto::CERO, 0u64, false);
    for c in ciclos {
        if let (Some(x), true) = (c.neto, c.causas_nd.is_empty()) {
            n += 1;
            if x.signo() == std::cmp::Ordering::Greater {
                g += 1;
            }
            match neto.checked_add(x) {
                Some(v) => neto = v,
                None => desborde = true,
            }
        }
    }
    (n, neto, g, desborde)
}

/// EV por ciclo en `f64` SOLO para la puerta vigente (que lo compara con 0). El signo sale del
/// decimal exacto; si la presentacion en `f64` no conserva ese signo, N/D.
fn ev_por_ciclo(neto: DecimalExacto, n: u64) -> Option<f64> {
    if n == 0 {
        return None;
    }
    let ev = neto.como_f64() / n as f64;
    let mismo_signo = match neto.signo() {
        std::cmp::Ordering::Greater => ev > 0.0,
        std::cmp::Ordering::Less => ev < 0.0,
        std::cmp::Ordering::Equal => ev == 0.0,
    };
    (ev.is_finite() && mismo_signo).then_some(ev)
}

fn hash_de(c: &CertificadoCiclosV2) -> String {
    let base = serde_json::to_string(&(
        (&c.ambito, &c.modo, c.ventana_desde_ms, c.ventana_hasta_ms, &c.ciclos),
        (&c.saldo_t0, &c.saldo_t1, &c.externos),
    ))
    .unwrap_or_default();
    let mut h = Sha256::new();
    h.update(base.as_bytes());
    h.finalize().iter().map(|b| format!("{b:02x}")).collect()
}

/// Por que la puerta no recibe evidencia del certificado v2.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MotivoSinEvidencia {
    Ausente,
    Ilegible,
    /// Un JSON legado por filas (`EvidenciaDemo`): solo diagnostico.
    Legado,
    Corrupto,
    VersionDesconocida,
    Revocado,
    /// Estado ND (datos criticos sin acreditar) o no VALIDO.
    Incompleto,
    /// No es la Demo en Testnet (Mainnet, paper u otro).
    OtroEntorno,
    /// Cuenta, venue, sesion, estrategia, binario, configuracion, roster o ventana discordantes.
    AmbitoDiscordante,
    /// El arranque no tiene el ambito firmado con que comparar.
    AmbitoNoAcreditado,
    /// Los recuentos, el EV o el hash no cuadran con los ciclos del propio recibo.
    Inconsistente,
    SinRespaldo,
}

/// El ambito y la ventana con que el arranque de Produccion compara el recibo.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct AmbitoEsperado {
    pub ambito: AmbitoCertificado,
    pub ventana_desde_ms: u64,
    pub ventana_hasta_ms: u64,
}

impl AmbitoEsperado {
    /// CICLOS-2: el arranque de Produccion todavia no tiene una fuente firmada del ambito Testnet
    /// (cuenta, sesion, binario, configuracion, ventana). N/D: sin ella no hay evidencia.
    pub fn del_arranque() -> Option<Self> {
        None
    }
}

/// Ruta de la marca de revocacion, junto al certificado.
pub fn ruta_revocacion(ruta_certificado: &Path) -> PathBuf {
    ruta_certificado.with_extension("revocado")
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct MarcaRevocacion {
    hash_ciclos: String,
    motivo: String,
}

fn escribir_atomico(ruta: &Path, texto: &str) -> Result<(), String> {
    let tmp = ruta.with_extension("tmp");
    std::fs::write(&tmp, texto).map_err(|e| format!("{}: {e}", tmp.display()))?;
    std::fs::rename(&tmp, ruta).map_err(|e| format!("{}: {e}", ruta.display()))
}

/// Escribe el certificado por reemplazo atomico (tmp + rename) tras validarlo.
pub fn escribir_certificado_v2(ruta: &Path, cert: &CertificadoCiclosV2) -> Result<(), String> {
    if cert.version != VERSION_CERTIFICADO || cert.unidad != UNIDAD || cert.hash_ciclos != hash_de(cert) {
        return Err("certificado v2 incoherente: no se escribe".into());
    }
    let texto = serde_json::to_string_pretty(cert).map_err(|e| e.to_string())?;
    escribir_atomico(ruta, &texto)
}

/// Revoca el recibo v2 de forma PERSISTENTE: marca junto al certificado (con su hash, o
/// `DESCONOCIDO` si el recibo es ilegible) y el propio recibo reescrito en estado REVOCADO. Una
/// carga posterior (reinicio) lo rechaza. Un recibo nuevo (re-certificacion en Demo) tiene otro hash.
pub fn revocar_certificado_v2(ruta_certificado: &Path, motivo: &str) -> Result<(), String> {
    let actual = match std::fs::read_to_string(ruta_certificado) {
        Ok(t) => Some(serde_json::from_str::<CertificadoCiclosV2>(&t).ok()),
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => None,
        Err(_) => Some(None),
    };
    let Some(leido) = actual else {
        return Ok(());
    };
    let hash = leido.as_ref().map(|c| c.hash_ciclos.clone()).unwrap_or_else(|| HASH_DESCONOCIDO.to_string());
    let marca = MarcaRevocacion { hash_ciclos: hash, motivo: motivo.to_string() };
    let texto = serde_json::to_string_pretty(&marca).map_err(|e| e.to_string())?;
    escribir_atomico(&ruta_revocacion(ruta_certificado), &texto)?;
    if let Some(mut c) = leido {
        c.estado = ESTADO_REVOCADO.to_string();
        c.motivo_revocacion = Some(motivo.to_string());
        let texto = serde_json::to_string_pretty(&c).map_err(|e| e.to_string())?;
        escribir_atomico(ruta_certificado, &texto)?;
    }
    Ok(())
}

/// Lectura para la puerta de Produccion: solo un v2 VALIDO, no revocado, coherente y del ambito
/// esperado da evidencia (`EvidenciaDemo` en unidades de CICLO: `cierres` = ciclos validos). El
/// legado por filas, la ausencia, la corrupcion y cualquier discordancia => sin evidencia.
pub fn cargar_evidencia_ciclos_v2(
    ruta_certificado: &Path,
    esperado: Option<&AmbitoEsperado>,
) -> Result<god_engine_core::orchestrator::EvidenciaDemo, MotivoSinEvidencia> {
    let texto = match std::fs::read_to_string(ruta_certificado) {
        Ok(t) => t,
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => return Err(MotivoSinEvidencia::Ausente),
        Err(_) => return Err(MotivoSinEvidencia::Ilegible),
    };
    let cert: CertificadoCiclosV2 = match serde_json::from_str(&texto) {
        Ok(c) => c,
        Err(_) => {
            let v: Option<serde_json::Value> = serde_json::from_str(&texto).ok();
            if serde_json::from_str::<god_engine_core::orchestrator::EvidenciaDemo>(&texto).is_ok() {
                return Err(MotivoSinEvidencia::Legado);
            }
            if v.as_ref().and_then(|v| v.get("version")).is_some_and(|x| x.as_u64() != Some(u64::from(VERSION_CERTIFICADO))) {
                return Err(MotivoSinEvidencia::VersionDesconocida);
            }
            return Err(MotivoSinEvidencia::Corrupto);
        }
    };
    if cert.version != VERSION_CERTIFICADO || cert.unidad != UNIDAD {
        return Err(MotivoSinEvidencia::VersionDesconocida);
    }
    let ruta_marca = ruta_revocacion(ruta_certificado);
    match std::fs::read_to_string(&ruta_marca) {
        Ok(t) => match serde_json::from_str::<MarcaRevocacion>(&t) {
            Ok(m) if m.hash_ciclos != HASH_DESCONOCIDO && m.hash_ciclos != cert.hash_ciclos => {}
            _ => return Err(MotivoSinEvidencia::Revocado),
        },
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => {}
        Err(_) => return Err(MotivoSinEvidencia::Revocado),
    }
    if cert.estado == ESTADO_REVOCADO || cert.motivo_revocacion.is_some() {
        return Err(MotivoSinEvidencia::Revocado);
    }
    if cert.estado != ESTADO_VALIDO {
        return Err(MotivoSinEvidencia::Incompleto);
    }
    if cert.hash_ciclos.len() != 64 || !cert.hash_ciclos.bytes().all(|b| b.is_ascii_hexdigit()) {
        return Err(MotivoSinEvidencia::SinRespaldo);
    }
    if cert.ambito.entorno != ENTORNO_TESTNET {
        return Err(MotivoSinEvidencia::OtroEntorno);
    }
    let (n, neto, ganadores, desborde) = contar(&cert.ciclos);
    let coherente = !desborde
        && cert.hash_ciclos == hash_de(&cert)
        && cert.causas_nd.is_empty()
        && cert.arrastres == 0
        && cert.abiertos == 0
        && cert.ciclos_estructurales == cert.ciclos.len() as u64
        && cert.n_ciclos_validos == n
        && cert.ciclos_nd == 0
        && n == cert.ciclos_estructurales
        && cert.neto_total == neto
        && cert.ganadores == ganadores
        && cert.ev_neto_por_ciclo.map(f64::to_bits) == ev_por_ciclo(neto, n).map(f64::to_bits)
        && matches!((cert.saldo_t0, cert.saldo_t1), (Some(s0), Some(s1)) if cuadra(s0, s1, neto, cert.externos) == Some(true));
    if !coherente {
        return Err(MotivoSinEvidencia::Inconsistente);
    }
    let Some(esp) = esperado else {
        return Err(MotivoSinEvidencia::AmbitoNoAcreditado);
    };
    if esp.ambito.entorno != ENTORNO_TESTNET {
        return Err(MotivoSinEvidencia::OtroEntorno);
    }
    if cert.ambito != esp.ambito || cert.ventana_desde_ms != esp.ventana_desde_ms || cert.ventana_hasta_ms != esp.ventana_hasta_ms {
        return Err(MotivoSinEvidencia::AmbitoDiscordante);
    }
    let Some(ev) = cert.ev_neto_por_ciclo else {
        return Err(MotivoSinEvidencia::Incompleto);
    };
    Ok(god_engine_core::orchestrator::EvidenciaDemo { cierres: n, ev_neto_por_trade: ev, wins: ganadores })
}
