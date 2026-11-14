//! CAPITAL-UNICO V4G, V4H y V4I: el saldo dimensionable (`GlobalArena::unified_capital`, lo que
//! lee Risk) deja de ser un acumulador en el arena vivo y se DERIVA:
//!
//! dimensionable = saldo vigente aceptado por el puente (era y sesion vigentes)
//!                 - debitos que el exchange aun no refleja.
//!
//! El libro vive dentro de la puerta que ya serializa la publicacion de entradas
//! (`QuantumConfig::entry_publication_gate`): el nucleo ya la tiene tomada cuando reserva, asi que
//! no hay candado nuevo en el camino de los ticks. Cada operacion vuelve a calcular el valor entero
//! (no por incrementos): sin debitos es exactamente el saldo vigente.
//!
//! Invariantes (secciones "V4G - DISENO", "V4H - DISENO" y "V4I - DISENO" de
//! `integrador_capital_unico.md`):
//!
//! - INV-D: el saldo vigente es el de MAYOR marca del exchange del alcance (era, sesion). Una marca
//!   menor no mueve nada; con marcas iguales, o sin marca, se toma el MENOR de los dos valores, con
//!   SU marca. V4I: un saldo SIN marca que queda vigente es un suelo: desde ahi el saldo solo baja,
//!   con o sin marca, hasta que llegue evidencia positiva (el primer saldo que el puente entrega
//!   despues de vetar, es decir, una REST pedida despues). Un cambio de cuenta empieza de cero.
//! - INV-E: un debito solo sale cuando el saldo vigente tiene marca ESTRICTAMENTE mayor que la
//!   marca propia de ese debito. V4I: el debito de una entrada se lleva por fill (`trade_id`) con
//!   su comision REAL; nunca con el `updateTime` de un ACK acumulado. Lo retenido por una entrada
//!   es el mayor entre lo estimado y lo real, menos lo real ya incluido. V4J: la marca del evento
//!   terminal de la orden solo libera el estimado sobrante (la orden ya no ejecuta); cada cargo
//!   real sale con SU marca cubierta; con cantidades, lo no ejecutado retiene su parte del estimado.
//! - INV-F: la propuesta se revalida bajo la puerta al reservar ([`LibroDimensionable::admite`],
//!   V4I: por revision identificable, no por importe) y en la frontera del transporte
//!   ([`FronteraDeEnvio`] y [`LibroDimensionable::envio_valido_en`]).
//! - INV-G: los debitos locales conocidos y aun no observados (cierre local con perdida, fill de
//!   cierre, funding pagado) restan en el acto; ningun credito local suma; un cierre externo veta
//!   la admision hasta un saldo pedido despues de detectarlo.
//!
//! Sin activar ([`LibroDimensionable::activar`]: backtest, replay, simuladores, pruebas) el arena
//! sigue siendo el acumulador de siempre: las mismas sumas atomicas que antes.
//!
//! V4J (seccion "V4J - DISENO"): la frontera del transporte revalida EN el cruce
//! ([`FronteraDeEnvio::paso`] recibe un validador); un terminal de orden no libera el cargo de un
//! fill sin cobertura propia; lo ejecutado de una reserva posterior cuenta en el envio de una
//! anterior; y el corte de cuenta conciliado sale del API productivo (salida A4b): sin corte, un
//! cargo sin marca queda retenido hasta evidencia propia o veto. Un consumidor no puede nombrarlo:
//!
//! ```compile_fail
//! use quantum_arena::dimensionable::CorteDeCuenta;
//! ```
//!
//! ```compile_fail
//! fn corte(l: &mut quantum_arena::dimensionable::LibroDimensionable, a: &quantum_arena::GlobalArena) {
//!     let _ = l.conciliar(a, &());
//! }
//! ```
//!
//! Control positivo (el mismo montaje compila con el API vivo):
//!
//! ```
//! fn vivo(l: &mut quantum_arena::dimensionable::LibroDimensionable, a: &quantum_arena::GlobalArena) {
//!     let _ = l.reservado();
//!     l.marca_de_orden(a, 0, "cL_x", 1);
//! }
//! ```

use crate::state::GlobalArena;
use std::collections::HashSet;
use std::sync::atomic::Ordering;

/// Identidad de una reserva: la ranura exacta y su generacion (la orden de esa ranura).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ClaveReserva {
    pub coin_id: usize,
    pub slot: usize,
    pub generation: u64,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Origen {
    /// Comision de una entrada del nucleo.
    Entrada(ClaveReserva),
    /// Cierre local del nucleo con resultado neto negativo (estimado, sin marca del exchange).
    CierreLocal(ClaveReserva),
    /// Un fill de cierre del stream, por moneda y `trade_id`.
    FillCierre { coin_id: usize, trade_id: u64 },
    /// V4I: un fill de entrada que no se pudo atribuir a ninguna reserva (por ejemplo el
    /// remanente de otra orden): su comision real queda retenida por su identidad.
    FillEntradaSuelto { coin_id: usize, trade_id: u64 },
    /// Funding pagado visto en el historial de ingresos.
    Funding(u64),
}

#[derive(Debug, Clone, Copy, PartialEq)]
enum Estado {
    /// Reserva de entrada sin ejecucion conocida: ningun saldo la quita.
    Viva,
    /// Estimado local sin marca: ningun saldo lo quita.
    Estimado,
    /// Debito ejecutado. Para una entrada, la marca de su evento terminal en el stream; para el
    /// resto, la de su evento. `None` = desconocida (no se puede probar inclusion por marca).
    Pendiente(Option<u64>),
}

/// V4J (punto 2): la comision de un fill tal como la entrega el stream.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum ComisionDeFill {
    /// Cargada al activo del capital (USDT), cifra valida; cero explicito incluido.
    Activo(f64),
    /// Explicita en otro activo: el fill queda atribuido, sin restar del USDT.
    OtroActivo,
    /// Ausente o ilegible: no se sabe cuanto se cargo (se retiene el estimado o se veta).
    Desconocida,
}

/// V4I (INV-E): un fill de una entrada, por su identidad nativa.
#[derive(Debug, Clone, PartialEq)]
struct FillEntrada {
    trade_id: u64,
    /// Comision real cargada al activo del capital (USDT). Otra moneda o desconocida: cero aqui.
    comision: f64,
    /// V4J: la comision no se conoce (ausente o ilegible en el stream).
    desconocida: bool,
    /// V4J (punto 3): cantidad ejecutada en este fill (`l` del stream), si se conoce.
    cantidad: Option<f64>,
    /// V4J (punto 2): `l` y `z` (acumulado de la orden tras este fill) en unidades de 1e-8,
    /// leidos del decimal del stream: la cadena de fills se comprueba sin aritmetica binaria.
    cantidad_e8: Option<i64>,
    acumulado_e8: Option<i64>,
    marca: Option<u64>,
}

/// V4J (punto 2): un fill de entrada tal como lo entrega el stream privado.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct FillDelStream {
    pub trade_id: u64,
    /// `l` (cantidad de este fill), si es valida.
    pub cantidad: Option<f64>,
    /// `l` y `z` en unidades de 1e-8, si su decimal es exacto en 8 cifras.
    pub cantidad_e8: Option<i64>,
    pub acumulado_e8: Option<i64>,
    pub comision: ComisionDeFill,
    /// `T` del fill (ausente o 0: `None`).
    pub marca: Option<u64>,
}

#[derive(Debug, Clone, PartialEq)]
struct Debito {
    origen: Origen,
    /// Entrada: comision estimada (tras la ejecucion, la de lo ejecutado). Resto: el debito.
    importe: f64,
    estado: Estado,
    /// Alcance (era, sesion) vigente al crearlo.
    alcance: Option<(u64, u64)>,
    /// Orden de creacion: al enviar, las reservas posteriores no invalidan una anterior.
    seq: u64,
    /// INV-F: el dimensionable con el que Risk decidio esta entrada.
    base: Option<f64>,
    /// INV-F: revision del saldo vigente al reservar (cualquier saldo nuevo la cambia).
    rev: u64,
    /// V4I (INV-E): `clientOrderId` de la orden que respalda la entrada.
    orden: Option<String>,
    /// V4I (INV-E): fills conocidos de la entrada, por `trade_id`.
    fills: Vec<FillEntrada>,
    /// V4I (INV-E): marca del evento terminal de la orden en el stream (llega antes o despues
    /// del ACK; al ejecutar se toma la mayor de las dos).
    marca_orden: Option<u64>,
    /// V4J (punto 3): el estimado de la reserva entera (el `importe` baja con la devolucion).
    estimado: f64,
    /// V4J (punto 3): cantidad de la orden que respalda la reserva, si se conoce.
    cantidad: Option<f64>,
    /// V4J (punto 2): la cuenta vigente al crearlo (identidad de un fill suelto o de cierre).
    /// Solo un saldo de ESA cuenta puede probar su inclusion.
    cuenta: Option<u64>,
    /// V4J (punto 2): `z` del evento terminal de la orden (acumulado final), en 1e-8.
    acumulado_final_e8: Option<i64>,
    /// V4K (A2): llegaron terminales de la orden con acumulados distintos (por ejemplo, uno
    /// reciente con ejecucion y otro antiguo con acumulado nulo). La marca y el acumulado ya no
    /// son de la misma observacion: ningun terminal libera el estimado (se retiene).
    terminal_incoherente: bool,
}

fn cubre(marca_saldo: Option<u64>, marca: Option<u64>) -> bool {
    matches!((marca_saldo, marca), (Some(m), Some(t)) if m > t)
}

fn cantidad_valida(q: Option<f64>) -> Option<f64> {
    q.filter(|q| q.is_finite() && *q > 0.0)
}

impl Debito {
    fn nuevo(origen: Origen, importe: f64, estado: Estado, alcance: Option<(u64, u64)>, seq: u64) -> Self {
        Self {
            origen,
            importe,
            estado,
            alcance,
            seq,
            base: None,
            rev: 0,
            orden: None,
            fills: Vec::new(),
            marca_orden: None,
            estimado: importe,
            cantidad: None,
            cuenta: None,
            acumulado_final_e8: None,
            terminal_incoherente: false,
        }
    }

    /// V4J (punto 2): el evento terminal dio el acumulado final y los fills vistos lo cubren sin
    /// huecos (cadena exacta en 1e-8: cada fill empieza donde acaba el anterior y el ultimo acaba
    /// en el acumulado final). Sin eso, un fill puede no haber llegado aun (terminal primero).
    fn ejecucion_completa(&self) -> bool {
        let Some(fin) = self.acumulado_final_e8 else { return false };
        let mut cadena: Vec<(i64, i64)> = Vec::with_capacity(self.fills.len());
        for f in &self.fills {
            match (f.acumulado_e8, f.cantidad_e8) {
                (Some(z), Some(l)) => {
                    if l > 0 {
                        cadena.push((z, l));
                    }
                }
                _ => return false,
            }
        }
        cadena.sort_unstable();
        let mut previo = 0i64;
        for (z, l) in cadena {
            if z.checked_sub(l) != Some(previo) {
                return false;
            }
            previo = z;
        }
        previo == fin
    }

    /// V4J: comisiones reales de los fills que la marca del saldo vigente no cubre (cada cargo
    /// necesita su propia evidencia; el terminal de la orden no la presta).
    fn real_pendiente(&self, marca_saldo: Option<u64>) -> f64 {
        self.fills
            .iter()
            .filter(|f| !cubre(marca_saldo, f.marca))
            .map(|f| f.comision)
            .sum()
    }

    /// Lo que este debito retiene con el saldo vigente de marca `marca_saldo` (INV-E, V4J).
    fn retenido(&self, marca_saldo: Option<u64>) -> f64 {
        // V4K (A2): con terminales contradictorios, la marca terminal no prueba nada del acumulado:
        // se trata como no cubierta (retiene lo ejecutado sin cobertura y el estimado del resto).
        let terminal_cubierto =
            !self.terminal_incoherente && matches!(self.estado, Estado::Pendiente(m) if cubre(marca_saldo, m));
        if self.terminal_incoherente {
            return self.real_pendiente(marca_saldo) + self.estimado.max(self.importe);
        }
        // V4J (punto 2): con la cantidad de la orden conocida, ninguna marca libera lo ejecutado
        // y aun no visto (terminal recibido antes que su fill): rige la cadena o la cantidad.
        if self.fills.is_empty() && cantidad_valida(self.cantidad).is_none() {
            return if terminal_cubierto { 0.0 } else { self.importe };
        }
        if self.fills.is_empty() && terminal_cubierto && self.acumulado_final_e8 == Some(0) {
            return 0.0;
        }
        let pendiente = self.real_pendiente(marca_saldo);
        // V4J (punto 2): un fill con la comision desconocida retiene el estimado entero (y el
        // libro veta la admision hasta la evidencia: ver `fill_de_entrada_del_stream`).
        if self.fills.iter().any(|f| f.desconocida) {
            return pendiente + self.estimado.max(self.importe);
        }
        // V4J (punto 2): el terminal cubierto solo libera el estimado sobrante si TODA la
        // ejecucion de la orden esta vista fill a fill (cadena exacta hasta el acumulado final).
        if terminal_cubierto && self.ejecucion_completa() {
            return pendiente;
        }
        // V4J (punto 3): con la cantidad de la orden y la de cada fill, lo aun no ejecutado
        // retiene su parte del estimado y lo ejecutado su comision real hasta cubrirla.
        if let Some(q) = cantidad_valida(self.cantidad) {
            let ejecutado: Option<f64> = self
                .fills
                .iter()
                .map(|f| f.cantidad.filter(|c| c.is_finite() && *c >= 0.0))
                .sum();
            if let Some(x) = ejecutado {
                return pendiente + (self.importe - self.estimado * (x / q)).max(0.0);
            }
        }
        if terminal_cubierto {
            // V4J (punto 2): el terminal prueba que la orden no ejecuta mas (el estimado sobrante
            // sale), no que cada cargo este incluido: los fills sin cobertura propia se retienen.
            return pendiente;
        }
        let real: f64 = self.fills.iter().map(|f| f.comision).sum();
        let incluido = real - pendiente;
        (self.importe.max(real) - incluido).max(0.0)
    }

    /// V4J (punto 3): lo que retiene la parte YA ejecutada de una reserva viva (sus fills).
    fn retenido_ejecutado(&self, marca_saldo: Option<u64>) -> f64 {
        let pendiente = self.real_pendiente(marca_saldo);
        if self.fills.iter().any(|f| f.desconocida) {
            pendiente + self.estimado.max(self.importe)
        } else {
            pendiente
        }
    }

    fn es_entrada_viva(&self) -> bool {
        matches!(self.origen, Origen::Entrada(_)) && self.estado == Estado::Viva
    }
}

fn mayor(a: Option<u64>, b: Option<u64>) -> Option<u64> {
    match (a, b) {
        (Some(x), Some(y)) => Some(x.max(y)),
        (x, None) => x,
        (None, y) => y,
    }
}

/// V4J (punto 4, salida A4b): el corte de cuenta conciliado de V4I (`conciliar` y sus tipos) se
/// retiro del API productivo: no tenia llamadores en el camino vivo y no era una prueba valida
/// de inclusion (dictamen QRA de V4I). Sin corte, un cargo sin marca solo sale con la cobertura de
/// su propia marca; si no, queda retenido (o la admision vetada). Integrarlo exige A4a
/// (`qra_v4j_criterios.md`).

/// El libro del dimensionable. Se usa siempre bajo `entry_publication_gate`.
#[derive(Debug, Default)]
pub struct LibroDimensionable {
    derivado: bool,
    saldo: Option<f64>,
    /// INV-D: marca de la lectura que quedo vigente (la que prueba inclusion, INV-E).
    marca: Option<u64>,
    /// INV-D: la mayor marca conocida (ordena las lecturas; no prueba inclusion).
    marca_max: Option<u64>,
    /// INV-F: revision del saldo vigente; sube con cada saldo que cambia valor, marca o alcance,
    /// con cada veto del puente y con cada cierre externo.
    rev: u64,
    alcance: Option<(u64, u64)>,
    debitos: Vec<Debito>,
    /// INV-G: veto del libro por un cierre externo. `false`: esperando que el puente vete por la
    /// evidencia perdida que pide el host; `true`: ya veto, y el primer saldo que entregue al
    /// soltar (una REST pedida despues de detectarlo) suelta este veto.
    veto: Option<bool>,
    /// V4I (INV-D): suelo sin marca. `false`: el saldo vigente salio de una lectura sin marca y
    /// solo puede bajar; `true`: el puente ya veto, y el primer saldo que entregue al soltar es
    /// evidencia positiva (una REST pedida despues).
    suelo: Option<bool>,
    /// V4I (INV-D): identidad de la cuenta (venue y clave publica, sin credenciales).
    cuenta: Option<u64>,
    /// Alcance de la publicacion en curso (lo fija el publicador antes del saldo).
    entrante: Option<(u64, u64)>,
    seq: u64,
    /// V4I (INV-F): valores publicados en esta revision y en las anteriores (bits del f64). Una
    /// decision cuyo importe se publico en una revision anterior no identifica su revision.
    publicados_ahora: HashSet<u64>,
    publicados_antes: HashSet<u64>,
    /// V4I (INV-F): un importe ambiguo ya se rechazo en esta revision: desde ahi se publica un
    /// valor que no aparecio en ninguna revision anterior (el double inmediato inferior).
    desambiguar: bool,
    publicado: Option<f64>,
    /// V4L (L-A1): testigos de los compromisos locales (los ultimos `MAX_COMPROMISOS`).
    compromisos: std::collections::VecDeque<CompromisoDeEntrega>,
    seq_compromiso: u64,
}

/// V4L (L-A1): testigos de compromiso que conserva el libro (diagnostico y pruebas).
const MAX_COMPROMISOS: usize = 256;

impl LibroDimensionable {
    /// El host lo activa en el arena vivo antes de vincular el publicador del puente. Desde ahi el
    /// cierre local y la reconciliacion dejan de escribir el campo (lo trae el saldo del exchange).
    pub fn activar(&mut self, arena: &GlobalArena) {
        self.derivado = true;
        arena.config.dimensionable_derivado.store(true, Ordering::Release);
    }

    pub fn derivado(&self) -> bool {
        self.derivado
    }

    /// Saldo vigente y su marca.
    pub fn saldo(&self) -> Option<(f64, Option<u64>)> {
        self.saldo.map(|s| (s, self.marca))
    }

    /// Debitos (reservas y debitos locales) que el exchange aun no refleja, de la cuenta vigente.
    /// V4K (A5): el libro de cada cuenta es suyo: los debitos de otra cuenta no restan del saldo de
    /// esta (otra billetera) ni una foto de esta los libera; se conservan y vuelven a restar al
    /// volver a su cuenta. Un debito sin cuenta (anterior a fijarla) resta en cualquiera.
    pub fn reservado(&self) -> f64 {
        self.debitos.iter().filter(|d| self.de_esta_cuenta(d)).map(|d| d.retenido(self.marca_para(d))).sum()
    }

    /// V4K (A5): reservado de las cuentas que no son la vigente (obligaciones conservadas).
    pub fn reservado_de_otras_cuentas(&self) -> f64 {
        self.debitos.iter().filter(|d| !self.de_esta_cuenta(d)).map(|d| d.retenido(None)).sum()
    }

    fn de_esta_cuenta(&self, d: &Debito) -> bool {
        d.cuenta.is_none() || d.cuenta == self.cuenta
    }

    /// V4J (punto 2): la marca del saldo vigente solo prueba inclusion de los cargos de SU
    /// cuenta; los de otra cuenta se conservan (restan, y nada de esta cuenta los retira).
    fn marca_para(&self, d: &Debito) -> Option<u64> {
        if d.cuenta == self.cuenta {
            self.marca
        } else {
            None
        }
    }

    pub fn reservas(&self) -> usize {
        self.debitos.len()
    }

    /// INV-E: el alcance (era, sesion) con el que se creo la reserva de esa entrada, si sigue en
    /// el libro. La reserva se identifica por sesion, ranura y generacion.
    pub fn alcance_de(&self, clave: ClaveReserva) -> Option<Option<(u64, u64)>> {
        self.debitos.iter().find(|d| d.origen == Origen::Entrada(clave)).map(|d| d.alcance)
    }

    /// INV-G: la admision esta vetada por el libro (cierre externo sin saldo posterior).
    pub fn vetado(&self) -> bool {
        self.veto.is_some()
    }

    /// V4I: el libro espera evidencia positiva del puente (cierre externo o suelo sin marca): el
    /// host debe pedirle evidencia perdida (veta hasta una REST pedida despues).
    pub fn espera_evidencia(&self) -> bool {
        self.derivado && (self.veto == Some(false) || self.suelo == Some(false))
    }

    /// V4I: el saldo vigente es un suelo sin marca (solo baja hasta la evidencia positiva).
    pub fn en_suelo(&self) -> bool {
        self.suelo.is_some()
    }

    /// saldo vigente - reservado, si hay saldo. V4I (INV-F): tras rechazar un importe ambiguo,
    /// el mayor double por debajo que no se publico en una revision anterior (nunca por encima).
    pub fn dimensionable(&self) -> Option<f64> {
        self.saldo.map(|s| {
            let mut v = s - self.reservado();
            if self.desambiguar {
                while self.publicados_antes.contains(&v.to_bits()) && v.is_finite() {
                    v = v.next_down();
                }
            }
            v
        })
    }

    fn publicar(&mut self, arena: &GlobalArena) {
        if !self.derivado {
            return;
        }
        if let Some(d) = self.dimensionable() {
            arena.unified_capital.store(d, Ordering::Relaxed);
        }
        // V4I (INV-F): lo publicado en esta revision identifica la decision que lo use.
        if let Some(d) = self.dimensionable() {
            self.publicados_ahora.insert(d.to_bits());
            self.publicado = Some(d);
        }
    }

    fn nueva_revision(&mut self) {
        self.rev = self.rev.wrapping_add(1);
        let ahora = std::mem::take(&mut self.publicados_ahora);
        self.publicados_antes.extend(ahora);
        self.desambiguar = false;
    }

    fn siguiente(&mut self) -> u64 {
        self.seq = self.seq.wrapping_add(1);
        self.seq
    }

    fn retirar_incluidos(&mut self) {
        let (marca, cuenta) = (self.marca, self.cuenta);
        self.debitos.retain(|d| {
            let m = if d.cuenta == cuenta { marca } else { None };
            !(matches!(d.estado, Estado::Pendiente(_)) && d.retenido(m) <= 0.0)
        });
    }

    /// El publicador del puente: saldo aceptado con su marca del exchange, en el alcance que fijo
    /// [`fijar_alcance`](Self::fijar_alcance) (o el vigente).
    pub fn saldo_aceptado(&mut self, arena: &GlobalArena, saldo: f64, marca: Option<u64>) {
        let alcance = self.entrante.take().or(self.alcance);
        self.aceptar(arena, alcance, saldo, marca);
    }

    /// El alcance (era, sesion) de la publicacion que llega. Un alcance nuevo empieza con su primer
    /// saldo: INV-D ordena por marca dentro de cada alcance.
    pub fn fijar_alcance(&mut self, era: u64, sesion: u64) {
        self.entrante = Some((era, sesion));
    }

    /// V4J (punto 1): el veto del puente sobre la admision de entradas (lo lee el cruce).
    pub fn veto_de_admision(arena: &GlobalArena) -> bool {
        arena.capital_incompleto.load(Ordering::Acquire)
    }

    /// V4J: la cuenta fijada (para el host, al relevar).
    pub fn cuenta(&self) -> Option<u64> {
        self.cuenta
    }

    /// V4I (INV-D): la cuenta del ejecutor (identidad estable de venue y clave publica). Otra
    /// cuenta empieza sin saldo (la admision queda vetada hasta su primera lectura) y nada de lo
    /// leido en la anterior ordena ni prueba inclusion en esta; los debitos se conservan.
    pub fn fijar_cuenta(&mut self, arena: &GlobalArena, cuenta: u64) {
        if self.cuenta == Some(cuenta) {
            return;
        }
        let habia = self.cuenta.is_some();
        self.cuenta = Some(cuenta);
        if !habia {
            // V4K (A5): lo registrado antes de fijar la primera cuenta es de esa cuenta.
            for d in self.debitos.iter_mut().filter(|d| d.cuenta.is_none()) {
                d.cuenta = Some(cuenta);
            }
        }
        if habia {
            self.saldo = None;
            self.marca = None;
            self.marca_max = None;
            self.suelo = None;
            self.nueva_revision();
            self.publicar(arena);
        }
    }

    fn aceptar(&mut self, arena: &GlobalArena, alcance: Option<(u64, u64)>, saldo: f64, marca: Option<u64>) {
        if !saldo.is_finite() {
            return;
        }
        let mismo_alcance = alcance == self.alcance;
        let menor = |actual: f64, marca_actual: Option<u64>| {
            if saldo < actual {
                (saldo, marca)
            } else {
                (actual, marca_actual)
            }
        };
        let evidencia = self.suelo == Some(true);
        let (valor, marca_elegida) = match self.saldo {
            None => (saldo, marca),
            // V4I (INV-D): el primer saldo que el puente entrega tras vetar es una REST pedida
            // despues de la caida sin marca: evidencia positiva, sale del suelo.
            Some(_) if evidencia => (saldo, marca),
            // V4I (INV-D): suelo sin marca: solo baja, con o sin marca (una foto con marca mayor
            // pudo tomarse antes de la caida sin marca).
            Some(actual) if self.suelo.is_some() => menor(actual, self.marca),
            Some(actual) => match (marca, self.marca_max) {
                (Some(n), Some(v)) if n > v => (saldo, marca),
                // INV-D: una marca menor no mueve nada dentro del alcance.
                (Some(n), Some(v)) if n < v && mismo_alcance => return,
                // INV-D: marca igual, sin marca, o una foto mas vieja de otro alcance: no hay
                // orden; el menor de los dos, con SU marca (no se presta la de la otra lectura).
                _ => menor(actual, self.marca),
            },
        };
        if self.saldo != Some(valor) || self.marca != marca_elegida || !mismo_alcance {
            self.nueva_revision();
        }
        self.marca_max = if evidencia { marca } else { mayor(self.marca_max, marca) };
        // V4I (INV-D): un saldo sin marca que queda vigente abre el suelo; la evidencia positiva
        // lo cierra si trae marca.
        if evidencia || self.suelo.is_none() {
            self.suelo = marca_elegida.is_none().then_some(false);
        }
        // INV-G: el veto por cierre externo se suelta con el primer saldo que el puente entrega
        // despues de haber vetado por la evidencia perdida (una REST pedida despues del hecho).
        if self.veto == Some(true) {
            self.veto = None;
        }
        self.saldo = Some(valor);
        self.marca = marca_elegida;
        self.alcance = alcance;
        self.retirar_incluidos();
        self.publicar(arena);
    }

    /// INV-G: el puente veto (el publicador lo ve). Si el libro espera la conciliacion de un
    /// cierre externo o la salida de un suelo sin marca, desde aqui el primer saldo entregado al
    /// soltar es posterior al hecho. V4I (INV-F): el veto abre una revision nueva (una decision
    /// anterior al veto no se reserva ni se envia despues).
    pub fn veto_del_puente(&mut self) {
        if let Some(visto) = self.veto.as_mut() {
            *visto = true;
        }
        if let Some(visto) = self.suelo.as_mut() {
            *visto = true;
        }
        self.nueva_revision();
    }

    /// Reserva de una entrada (el nucleo, al abrir la ranura).
    pub fn reservar(&mut self, arena: &GlobalArena, clave: ClaveReserva, comision: f64) {
        self.reservar_con_base(arena, clave, comision, None);
    }

    /// Reserva de una entrada con el dimensionable con el que Risk la decidio (INV-F).
    pub fn reservar_con_base(&mut self, arena: &GlobalArena, clave: ClaveReserva, comision: f64, base: Option<f64>) {
        if !self.derivado {
            arena.unified_capital.fetch_add(-comision, Ordering::Relaxed);
            return;
        }
        // Tambien con comision cero: la reserva conserva la identidad de la orden (INV-F, envio).
        // Un rebate no suma (INV-G): el importe no baja de cero.
        if comision.is_finite() {
            let seq = self.siguiente();
            let mut d = Debito::nuevo(Origen::Entrada(clave), comision.max(0.0), Estado::Viva, self.alcance, seq);
            d.base = base;
            d.rev = self.rev;
            d.cuenta = self.cuenta;
            self.debitos.push(d);
        }
        self.publicar(arena);
    }

    /// INV-F, al reservar (bajo la puerta, antes de reservar): la entrada que Risk dimensiono con
    /// `capital` sigue solo si el veto del puente esta suelto, el libro no veta y `capital` es un
    /// valor publicado en la revision VIGENTE (y en ninguna anterior: 100, 90 y otra vez 100 no
    /// identifican la decision) que el valor publicado aun cubre. Un importe ambiguo se rechaza y
    /// desde ahi se publica un valor que lo identifique: la decision se recalcula con el.
    /// Sin activar, siempre.
    pub fn admite(&mut self, arena: &GlobalArena, capital: Option<f64>) -> bool {
        if !self.derivado {
            return true;
        }
        let (Some(c), Some(p)) = (capital, self.publicado) else { return false };
        // V4K (F4): cero y deuda observados conservan la insolvencia: sin capacidad positiva no
        // se reserva ninguna entrada (no depende de que Risk lo rechace antes).
        if !(c > 0.0 && p > 0.0) {
            return false;
        }
        if !c.is_finite()
            || self.veto.is_some()
            || arena.capital_incompleto.load(Ordering::Acquire)
            || self.dimensionable().is_none()
        {
            return false;
        }
        let bits = c.to_bits();
        if self.publicados_antes.contains(&bits) {
            if !self.desambiguar {
                self.desambiguar = true;
                self.publicar(arena);
            }
            return false;
        }
        self.publicados_ahora.contains(&bits) && p >= c
    }

    /// INV-F, al enviar: la reserva sigue viva, el libro no veta y el disponible para ella (saldo
    /// menos todos los debitos salvo ella misma y las reservas de entrada posteriores) cubre aun
    /// la base con la que Risk la decidio. Sin activar, siempre.
    pub fn envio_valido(&self, clave: ClaveReserva) -> bool {
        if !self.derivado {
            return true;
        }
        let Some(propia) = self
            .debitos
            .iter()
            .find(|d| d.origen == Origen::Entrada(clave) && d.estado == Estado::Viva)
        else {
            return false;
        };
        let Some(saldo) = self.saldo else { return false };
        if self.veto.is_some() || propia.rev != self.rev {
            // Veto del libro, o el saldo vigente cambio desde la reserva (revision distinta,
            // aunque el importe vuelva a ser el mismo): la propuesta se recalcula.
            return false;
        }
        let Some(base) = propia.base else { return true };
        // Deliberado: la parte SIN ejecutar de una reserva de entrada POSTERIOR y todavia viva no
        // invalida esta (se decidio contando con ella); una posterior ya ejecutada cuenta con lo
        // que retiene (comision real incluida). V4J (punto 3): lo YA ejecutado de una posterior
        // viva (sus fills, aunque el ACK aun no cambie su estado) cuenta siempre. El margen
        // combinado de las dos lo valida el exchange.
        let ajeno: f64 = self
            .debitos
            .iter()
            .filter(|d| d.seq != propia.seq && self.de_esta_cuenta(d))
            .map(|d| {
                let marca = self.marca_para(d);
                if d.es_entrada_viva() && d.seq > propia.seq {
                    d.retenido_ejecutado(marca)
                } else {
                    d.retenido(marca)
                }
            })
            .sum();
        base.is_finite() && saldo - ajeno >= base
    }

    /// INV-F, la ultima comprobacion antes del transporte: el veto del PUENTE (pudo entrar
    /// despues del permiso del OMS) y [`envio_valido`](Self::envio_valido) (veto del libro,
    /// revision e importe).
    pub fn envio_valido_en(&self, arena: &GlobalArena, clave: ClaveReserva) -> bool {
        if self.derivado && arena.capital_incompleto.load(Ordering::Acquire) {
            return false;
        }
        self.envio_valido(clave)
    }

    /// CAPITAL-UNICO V4L (L-A1): registra el compromiso local de una entrega validada. Lo llama el
    /// validador del OMS con el estado de la sesion y esta puerta tomados, en el mismo paso de la
    /// validacion: los escritores (revision, vetos, sesion) quedan ordenados antes o despues. El
    /// testigo lleva la revision vigente: un cambio posterior no lo reetiqueta.
    pub fn comprometer(&mut self, sesion: u64, reserva: Option<ClaveReserva>, fase: FaseDeEntrega) -> CompromisoDeEntrega {
        self.seq_compromiso += 1;
        let c = CompromisoDeEntrega { seq: self.seq_compromiso, sesion, revision: self.rev, reserva, fase };
        if self.compromisos.len() == MAX_COMPROMISOS {
            self.compromisos.pop_front();
        }
        self.compromisos.push_back(c);
        c
    }

    /// V4L (L-A1): los testigos de compromiso conservados, del mas antiguo al mas reciente.
    pub fn compromisos(&self) -> Vec<CompromisoDeEntrega> {
        self.compromisos.iter().copied().collect()
    }

    /// V4L (L-A1): revision vigente del libro (la que llevan los testigos de compromiso).
    pub fn revision(&self) -> u64 {
        self.rev
    }

    /// CAPITAL-UNICO V4L (L-A1H): el stream ya informo ejecucion positiva de la orden de esta
    /// reserva (algun fill con cantidad > 0, o un terminal con acumulado > 0). Un rechazo o un
    /// terminal sin ejecucion que la contradiga no revierte la reserva: se conserva y se reconcilia.
    pub fn ejecucion_observada(&self, clave: ClaveReserva) -> bool {
        self.debitos.iter().any(|d| {
            d.origen == Origen::Entrada(clave)
                && (d.fills.iter().any(|f| f.cantidad.is_some_and(|q| q > 0.0) || f.cantidad_e8.is_some_and(|q| q > 0) || f.cantidad.is_none())
                    || d.acumulado_final_e8.is_some_and(|z| z > 0))
        })
    }

    /// Cancelacion sin ejecucion: quita la reserva viva, no suma dinero. Repetida: nada.
    /// V4I (INV-E): si el stream ya informo fills de esa orden, su comision real se conserva
    /// (solo se libera la parte no ejecutada).
    pub fn cancelar(&mut self, arena: &GlobalArena, clave: ClaveReserva, comision: f64) {
        if !self.derivado {
            if comision.is_finite() && comision != 0.0 {
                arena.unified_capital.fetch_add(comision, Ordering::AcqRel);
            }
            return;
        }
        if let Some(i) = self
            .debitos
            .iter()
            .position(|d| d.origen == Origen::Entrada(clave) && d.estado == Estado::Viva)
        {
            if self.debitos[i].fills.is_empty() {
                self.debitos.remove(i);
            } else {
                let d = &mut self.debitos[i];
                d.importe = 0.0;
                d.estado = Estado::Pendiente(d.marca_orden);
            }
        }
        self.retirar_incluidos();
        self.publicar(arena);
    }

    /// Ejecucion confirmada: `devuelta` es la comision estimada de lo NO ejecutado; `marca_fill`,
    /// la marca que cubre TODA la ejecucion de la orden (V4I: la del evento terminal del stream;
    /// `None` si no se conoce). Repetida: solo puede subir la marca. Si el saldo vigente ya tiene
    /// marca mayor, el debito sale en el acto (sin sumar nada).
    pub fn ejecutar(&mut self, arena: &GlobalArena, clave: ClaveReserva, devuelta: f64, marca_fill: Option<u64>) {
        if !self.derivado {
            if devuelta.is_finite() && devuelta != 0.0 {
                arena.unified_capital.fetch_add(devuelta, Ordering::AcqRel);
            }
            return;
        }
        if let Some(i) = self.debitos.iter().position(|d| d.origen == Origen::Entrada(clave)) {
            let vacia = {
                let d = &mut self.debitos[i];
                match d.estado {
                    Estado::Viva => {
                        if devuelta.is_finite() {
                            d.importe = (d.importe - devuelta).max(0.0);
                        }
                        d.estado = Estado::Pendiente(mayor(marca_fill, d.marca_orden));
                    }
                    Estado::Pendiente(m) => d.estado = Estado::Pendiente(mayor(m, marca_fill)),
                    Estado::Estimado => {}
                }
                !(d.importe.is_finite() && d.importe > 0.0) && d.fills.is_empty()
            };
            if vacia {
                self.debitos.remove(i);
            }
        }
        self.retirar_incluidos();
        self.publicar(arena);
    }

    /// V4I (INV-E): la orden `orden` (su `clientOrderId`) respalda la reserva `clave`. El host lo
    /// fija bajo la puerta antes de enviarla; asi el stream atribuye cada fill a su reserva.
    pub fn vincular_orden(&mut self, clave: ClaveReserva, orden: &str) {
        if let Some(d) = self.debitos.iter_mut().find(|d| d.origen == Origen::Entrada(clave)) {
            d.orden = Some(orden.to_owned());
        }
    }

    /// V4J (punto 3): como [`vincular_orden`](Self::vincular_orden), con la cantidad de la orden
    /// (unidades del activo base, las mismas que la `l` de cada fill del stream). Con ella, lo
    /// aun no ejecutado retiene su parte del estimado y lo ejecutado su comision real.
    pub fn vincular_orden_con_cantidad(&mut self, clave: ClaveReserva, orden: &str, cantidad: f64) {
        if let Some(d) = self.debitos.iter_mut().find(|d| d.origen == Origen::Entrada(clave)) {
            d.orden = Some(orden.to_owned());
            d.cantidad = cantidad_valida(Some(cantidad));
        }
    }

    /// V4I (INV-E): un fill de entrada del stream, por su identidad (`trade_id`) y con su comision
    /// REAL en el activo del capital (`None`: la comision va en otra moneda; no se resta aqui).
    /// Repetido (mismo `trade_id`), nada; `trade_id` 0 (sin identidad) se suma siempre. Si el
    /// saldo vigente ya tiene marca mayor, la comision ya esta incluida. Un fill que no se atribuye
    /// a ninguna reserva queda retenido por su identidad (`coin_id`, `trade_id`).
    pub fn fill_de_entrada(
        &mut self,
        arena: &GlobalArena,
        coin_id: usize,
        orden: &str,
        trade_id: u64,
        comision: Option<f64>,
        marca: Option<u64>,
    ) {
        let comision = match comision {
            None => ComisionDeFill::OtroActivo,
            Some(c) if c.is_finite() => ComisionDeFill::Activo(c),
            Some(_) => ComisionDeFill::Desconocida,
        };
        let _ = self.fill_de_entrada_con(arena, coin_id, orden, trade_id, None, comision, marca);
    }

    /// V4J (puntos 2 y 3): un fill de entrada con su cantidad (`l`) y su comision tal como llego
    /// (sin el decimal exacto del stream: ver [`fill_de_entrada_del_stream`](Self::fill_de_entrada_del_stream)).
    #[allow(clippy::too_many_arguments)]
    pub fn fill_de_entrada_con(
        &mut self,
        arena: &GlobalArena,
        coin_id: usize,
        orden: &str,
        trade_id: u64,
        cantidad: Option<f64>,
        comision: ComisionDeFill,
        marca: Option<u64>,
    ) -> bool {
        let fill = FillDelStream { trade_id, cantidad, cantidad_e8: None, acumulado_e8: None, comision, marca };
        self.fill_de_entrada_del_stream(arena, coin_id, orden, &fill)
    }

    /// V4J (puntos 2 y 3): un fill de entrada del stream privado. Identidad: `trade_id` dentro de
    /// la reserva de ESTA cuenta; un fill suelto, ademas, por moneda y cuenta. Marca ausente o 0:
    /// el fill nunca prueba inclusion por marca. Comision desconocida (ausente, ilegible o sin
    /// activo): no hay cota demostrable del cargo, asi que, ademas de retener el estimado de la
    /// reserva, la admision queda vetada hasta una REST pedida despues (devuelve `true`: el
    /// llamador pide al puente la evidencia perdida, como en un cierre externo).
    pub fn fill_de_entrada_del_stream(&mut self, arena: &GlobalArena, coin_id: usize, orden: &str, fill: &FillDelStream) -> bool {
        if !self.derivado {
            return false;
        }
        let (importe, desconocida) = match fill.comision {
            ComisionDeFill::Activo(c) if c.is_finite() => (c.max(0.0), false),
            ComisionDeFill::OtroActivo => (0.0, false),
            _ => (0.0, true),
        };
        let trade_id = fill.trade_id;
        let marca = fill.marca.filter(|t| *t > 0);
        let marca_saldo = self.marca;
        let cuenta = self.cuenta;
        if let Some(d) = self.debitos.iter_mut().find(|d| {
            matches!(d.origen, Origen::Entrada(c) if c.coin_id == coin_id) && d.orden.as_deref() == Some(orden) && d.cuenta == cuenta
        }) {
            if trade_id != 0 && d.fills.iter().any(|f| f.trade_id == trade_id) {
                return false;
            }
            d.fills.push(FillEntrada {
                trade_id,
                comision: importe,
                desconocida,
                cantidad: fill.cantidad,
                cantidad_e8: fill.cantidad_e8,
                acumulado_e8: fill.acumulado_e8,
                marca,
            });
        } else if !desconocida {
            let ya = trade_id != 0
                && self.debitos.iter().any(|d| {
                    d.origen == Origen::FillEntradaSuelto { coin_id, trade_id } && d.cuenta == cuenta
                });
            if ya || importe <= 0.0 || cubre(marca_saldo, marca) {
                return false;
            }
            let seq = self.siguiente();
            let tid = if trade_id != 0 { trade_id } else { u64::MAX - seq };
            let mut d = Debito::nuevo(
                Origen::FillEntradaSuelto { coin_id, trade_id: tid },
                importe,
                Estado::Pendiente(marca),
                self.alcance,
                seq,
            );
            d.cuenta = cuenta;
            self.debitos.push(d);
        }
        if desconocida {
            // Sin cota del cargo: veto hasta la evidencia (una REST pedida despues).
            self.veto = Some(false);
            self.nueva_revision();
        }
        self.retirar_incluidos();
        self.publicar(arena);
        desconocida
    }
    /// V4I (INV-E): el evento terminal de la orden en el stream (FILLED, CANCELED o EXPIRED) con
    /// su marca: cubre toda la ejecucion de esa orden. Un ACK acumulado no la aporta.
    pub fn marca_de_orden(&mut self, arena: &GlobalArena, coin_id: usize, orden: &str, marca: u64) {
        self.marca_de_orden_con_acumulado(arena, coin_id, orden, marca, None);
    }

    /// V4J (punto 2): el evento terminal con su acumulado final (`z` en 1e-8). La marca cubre el
    /// ESTADO de la orden; el estimado sobrante solo sale si los fills vistos cubren ese acumulado.
    pub fn marca_de_orden_con_acumulado(&mut self, arena: &GlobalArena, coin_id: usize, orden: &str, marca: u64, acumulado_e8: Option<i64>) {
        if !self.derivado || marca == 0 {
            return;
        }
        let cuenta = self.cuenta;
        if let Some(d) = self.debitos.iter_mut().find(|d| {
            matches!(d.origen, Origen::Entrada(c) if c.coin_id == coin_id) && d.orden.as_deref() == Some(orden) && d.cuenta == cuenta
        }) {
            // V4K (A2): el acumulado no se reemplaza a ciegas mientras la marca guarda el maximo.
            // Un segundo terminal con OTRO acumulado contradice al primero (una orden termina una
            // vez): se conserva el mayor (z solo crece) y la orden queda incoherente (retiene).
            if let Some(z) = acumulado_e8 {
                match d.acumulado_final_e8 {
                    None => d.acumulado_final_e8 = Some(z),
                    Some(prev) if prev != z => {
                        d.terminal_incoherente = true;
                        d.acumulado_final_e8 = Some(prev.max(z));
                    }
                    Some(_) => {}
                }
            }
            d.marca_orden = mayor(d.marca_orden, Some(marca));
            if let Estado::Pendiente(m) = d.estado {
                d.estado = Estado::Pendiente(mayor(m, Some(marca)));
            }
        }
        self.retirar_incluidos();
        self.publicar(arena);
    }

    /// INV-G: cierre local del nucleo de una posicion que existio en el exchange con resultado
    /// neto negativo. Debito estimado (sin marca) hasta que la reconciliacion vea la moneda plana.
    pub fn debito_cierre_local(&mut self, arena: &GlobalArena, clave: ClaveReserva, importe: f64) {
        if !self.derivado || !(importe.is_finite() && importe > 0.0) {
            return;
        }
        if self.debitos.iter().any(|d| d.origen == Origen::CierreLocal(clave)) {
            return;
        }
        let seq = self.siguiente();
        let mut d = Debito::nuevo(Origen::CierreLocal(clave), importe, Estado::Estimado, self.alcance, seq);
        d.cuenta = self.cuenta;
        self.debitos.push(d);
        self.publicar(arena);
    }

    /// INV-G: un fill de cierre del stream. `debito` = comision en el activo del capital - PnL
    /// realizado de ese fill (solo cuenta si es positivo: ningun credito suma). `terminal`: la
    /// orden de cierre termino. V4I: `trade_id` 0 (sin identidad) no se deduplica (ruta
    /// conservadora); marca 0 es desconocida.
    pub fn fill_de_cierre(&mut self, arena: &GlobalArena, coin_id: usize, trade_id: u64, debito: f64, marca: u64, terminal: bool) {
        if !self.derivado {
            return;
        }
        let marca = (marca > 0).then_some(marca);
        let ya_incluido = cubre(self.marca, marca);
        let cuenta = self.cuenta;
        let repetido = trade_id != 0
            && self
                .debitos
                .iter()
                .any(|d| d.origen == Origen::FillCierre { coin_id, trade_id } && d.cuenta == cuenta);
        if debito.is_finite() && debito > 0.0 && !ya_incluido && !repetido {
            let seq = self.siguiente();
            let tid = if trade_id != 0 { trade_id } else { u64::MAX - seq };
            let mut d = Debito::nuevo(
                Origen::FillCierre { coin_id, trade_id: tid },
                debito,
                Estado::Pendiente(marca),
                self.alcance,
                seq,
            );
            d.cuenta = cuenta;
            self.debitos.push(d);
        }
        // El fill no dice de que ranura es: un terminal NO retira el estimado de ningun cierre
        // local (el de otra ranura de la misma moneda seguiria debiendose). Los estimados salen
        // cuando la reconciliacion ve la moneda plana (`moneda_plana`).
        let _ = terminal;
        self.publicar(arena);
    }

    /// INV-G: funding pagado (importe positivo) con su marca. Si el saldo vigente ya tiene marca
    /// mayor, ya esta incluido. Identificado por `id`: repetido, nada.
    pub fn debito_funding(&mut self, arena: &GlobalArena, id: u64, importe: f64, marca: u64) {
        if !self.derivado || !(importe.is_finite() && importe > 0.0) {
            return;
        }
        let marca = (marca > 0).then_some(marca);
        let ya_incluido = cubre(self.marca, marca);
        let origen = Origen::Funding(id);
        let cuenta = self.cuenta;
        if ya_incluido || self.debitos.iter().any(|d| d.origen == origen && d.cuenta == cuenta) {
            return;
        }
        let seq = self.siguiente();
        let mut d = Debito::nuevo(origen, importe, Estado::Pendiente(marca), self.alcance, seq);
        d.cuenta = cuenta;
        self.debitos.push(d);
        self.publicar(arena);
    }

    /// INV-G: el exchange aplico algo sin evento local (cierre externo, liquidacion). La admision
    /// queda vetada hasta el primer saldo que el puente entregue despues de vetar por la evidencia
    /// perdida que pide el host (una REST pedida despues de detectarlo): una foto anterior al
    /// hecho, aunque traiga marca mayor, no lo suelta.
    pub fn cierre_externo(&mut self, arena: &GlobalArena) {
        if !self.derivado {
            return;
        }
        // Hace falta un veto del puente posterior a esto (el host pide evidencia perdida al
        // puente: veta hasta una REST pedida despues) antes de que un saldo lo suelte.
        self.veto = Some(false);
        self.nueva_revision();
        self.publicar(arena);
    }

    /// INV-G: la reconciliacion ve la moneda plana en el exchange. Un estimado de cierre local de
    /// esa moneda ya no tiene fill que lo reemplace: sale, y la admision queda vetada hasta el
    /// primer saldo posterior (que trae ese cierre).
    /// Devuelve `true` si vetó (el host debe pedir evidencia perdida al puente).
    pub fn moneda_plana(&mut self, arena: &GlobalArena, coin_id: usize) -> bool {
        if !self.derivado {
            return false;
        }
        let antes = self.debitos.len();
        self.debitos.retain(|d| !matches!(d.origen, Origen::CierreLocal(c) if c.coin_id == coin_id));
        if self.debitos.len() != antes {
            self.cierre_externo(arena);
            return true;
        }
        false
    }
}

/// V4I (INV-F) y V4J (punto 1): la frontera del transporte. El guardia del OMS corre cada paso de
/// la entrega (cada `poll`) con su VALIDADOR activo en el hilo que corre ese paso; el ejecutor llama
/// a [`cruzar_frontera_de_envio`] justo antes de cada POST de apertura (y del apalancamiento que la
/// acompana), sin ninguna espera entre esa llamada y la entrega al cliente HTTP. El cruce EJECUTA
/// el validador en ese instante: un veto, un cambio de revision o de sesion que entro despues de la
/// revalidacion del inicio del paso, dentro del mismo `poll` o desde otro hilo, se ve en el cruce.
/// El validador toma sus candados y los suelta antes de devolver: ninguno se sostiene durante la
/// red, y los escritores publican mientras un envio valido espera la respuesta. Un paquete cruzado
/// queda comprometido con la revision que lo valido. Sin frontera activa (salidas defensivas,
/// cancelaciones, otros caminos), cruza siempre.
pub struct FronteraDeEnvio {
    cruzada: std::sync::atomic::AtomicBool,
}

/// El validador de un paso: `Ok` si la entrega sigue vigente (sesion, vetos, revision e importe).
pub type ValidadorDeEnvio<'a> = dyn Fn() -> Result<(), String> + 'a;

thread_local! {
    /// V4J: el validador del paso en curso. Puntero sin tiempo de vida: solo vive mientras dura
    /// [`FronteraDeEnvio::paso`] (lo restaura un guardia al salir, tambien ante un panico).
    static FRONTERA: std::cell::Cell<Option<*const ValidadorDeEnvio<'static>>> = const { std::cell::Cell::new(None) };
    static CRUCE: std::cell::Cell<bool> = const { std::cell::Cell::new(false) };
}

impl Default for FronteraDeEnvio {
    fn default() -> Self {
        Self::nueva()
    }
}

struct Restaurar {
    previa: Option<*const ValidadorDeEnvio<'static>>,
    cruce_previo: bool,
}

impl Drop for Restaurar {
    fn drop(&mut self) {
        FRONTERA.with(|c| c.set(self.previa));
        CRUCE.with(|c| c.set(self.cruce_previo));
    }
}

impl FronteraDeEnvio {
    pub fn nueva() -> Self {
        Self { cruzada: std::sync::atomic::AtomicBool::new(false) }
    }

    /// El ejecutor ya entrego una orden de apertura de esta entrada: desde ahi no se abandona la
    /// entrega (la respuesta decide), pero cada POST siguiente vuelve a pasar la frontera.
    pub fn cruzada(&self) -> bool {
        self.cruzada.load(Ordering::Relaxed)
    }

    /// Corre un paso de la entrega con `validar` activo: cada cruce de este paso lo ejecuta.
    pub fn paso<R>(&self, validar: &ValidadorDeEnvio<'_>, f: impl FnOnce() -> R) -> R {
        let ptr: *const ValidadorDeEnvio<'_> = validar;
        // SAFETY: solo se borra el tiempo de vida del puntero. `FRONTERA` lo guarda unicamente
        // mientras dura esta llamada (`Restaurar` lo quita al salir, tambien ante un panico) y
        // solo se usa en este hilo, dentro de `f`, mientras `validar` sigue prestado.
        let ptr: *const ValidadorDeEnvio<'static> = unsafe { std::mem::transmute(ptr) };
        let guardia = Restaurar {
            previa: FRONTERA.with(|c| c.replace(Some(ptr))),
            cruce_previo: CRUCE.with(|c| c.replace(false)),
        };
        POR_COMPROMETER.with(|c| c.set(false));
        let r = f();
        // V4K (A1): un cruce sin compromiso no pasa al paso siguiente.
        POR_COMPROMETER.with(|c| c.set(false));
        if CRUCE.with(|c| c.get()) {
            self.cruzada.store(true, Ordering::Relaxed);
        }
        drop(guardia);
        r
    }
}

/// V4I (INV-F) y V4J (punto 1): el ejecutor la llama justo antes de cada POST de apertura y del
/// apalancamiento. Con la frontera activa, ejecuta el validador del paso EN ESTE INSTANTE: si la
/// entrega ya no vale (veto, revision, sesion), la orden NO se entrega; si vale, el cruce queda
/// registrado (la entrega se compromete con esa revision).
pub fn cruzar_frontera_de_envio() -> Result<(), String> {
    let Some(ptr) = FRONTERA.with(|c| c.get()) else { return Ok(()) };
    // SAFETY: el puntero lo puso `FronteraDeEnvio::paso` en este hilo y sigue vivo: `paso` aun no
    // termino (su guardia lo quita al salir).
    let validar = unsafe { &*ptr };
    match validar() {
        Ok(()) => {
            CRUCE.with(|c| c.set(true));
            // V4K (A1): el POST que sigue a este cruce se vuelve a validar en su compromiso
            // local (`comprometer_entrega`, en el cliente HTTP, con la solicitud ya construida).
            POR_COMPROMETER.with(|c| c.set(true));
            Ok(())
        }
        Err(e) => Err(format!("OMS_ENTRY_WITHHELD: at the transport boundary: {e}")),
    }
}

thread_local! {
    /// V4K (A1): hay un cruce de apertura cuyo POST aun no se comprometio en el cliente.
    static POR_COMPROMETER: std::cell::Cell<bool> = const { std::cell::Cell::new(false) };
    /// V4K (A1): barrera de prueba en el intervalo entre el cruce y el compromiso local.
    static BARRERA_DE_COMPROMISO: std::cell::RefCell<Option<Box<dyn Fn()>>> = const { std::cell::RefCell::new(None) };
}

/// V4K (A1, dictamen V4J :32 y :52): el COMPROMISO LOCAL de la entrega. El cliente HTTP la llama
/// con la solicitud ya construida (cabeceras cargadas, URL firmada), inmediatamente antes de
/// entregarla al transporte, sin ninguna espera entre las dos. Solo actua sobre el POST que sigue a
/// un cruce de apertura ([`cruzar_frontera_de_envio`]): vuelve a ejecutar el validador del paso en
/// ESTE instante, asi que una publicacion (revision, veto, sesion) entre el cruce y el compromiso
/// retiene el paquete. Cierres, cancelaciones y piernas protectoras (sin cruce) pasan siempre. Un
/// paquete comprometido pertenece a la revision que lo valido aqui.
pub fn comprometer_entrega() -> Result<(), String> {
    comprometer_entrega_de(FaseDeEntrega::Orden)
}

/// CAPITAL-UNICO V4L (L-A1): la fase de la entrega que se compromete (el POST de `/leverage` o el
/// de la orden de apertura). Solo identifica el paquete en su testigo.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FaseDeEntrega {
    Apalancamiento,
    Orden,
}

/// CAPITAL-UNICO V4L (L-A1): el testigo de un compromiso local. Lo registra el validador del OMS
/// en el MISMO paso en que valida (sesion, vetos y reserva), con el estado de la sesion y la puerta
/// del libro tomados: el orden de los escritores (relevo, refresco, saldo, veto, cierre externo).
/// Desde ese paso el paquete es irrevocable y pertenece a esta revision; una revision posterior no
/// lo reetiqueta ni lo reenvia.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct CompromisoDeEntrega {
    /// Orden del compromiso en el libro (unico; nunca se repite).
    pub seq: u64,
    /// Generacion de la sesion del OMS que lo valido.
    pub sesion: u64,
    /// Revision del libro con la que se valido.
    pub revision: u64,
    /// La reserva que lo respalda (`None`: una entrega sin reserva del libro).
    pub reserva: Option<ClaveReserva>,
    pub fase: FaseDeEntrega,
}

/// CAPITAL-UNICO V4L (L-A1): el compromiso local de la entrega en la fase dada. Como
/// [`comprometer_entrega`]: solo tras un cruce de apertura; corre el validador del paso con la marca
/// de compromiso, y el validador del OMS registra el testigo bajo sus candados en ese mismo paso.
/// Despues de un `Ok` no queda ninguna comprobacion: el cliente entrega la solicitud ya construida.
pub fn comprometer_entrega_de(fase: FaseDeEntrega) -> Result<(), String> {
    if !POR_COMPROMETER.with(|c| c.replace(false)) {
        return Ok(());
    }
    BARRERA_DE_COMPROMISO.with(|b| {
        if let Some(f) = b.borrow().as_ref() {
            f();
        }
    });
    let Some(ptr) = FRONTERA.with(|c| c.get()) else { return Ok(()) };
    // SAFETY: como en `cruzar_frontera_de_envio`: el puntero lo puso `FronteraDeEnvio::paso` en
    // este hilo y `paso` aun no termino (el cliente compromete dentro del mismo `poll` del cruce).
    let validar = unsafe { &*ptr };
    let previa = COMPROMETIENDO.with(|c| c.replace(Some(fase)));
    let r = validar();
    COMPROMETIENDO.with(|c| c.set(previa));
    let r = r.map_err(|e| {
        // El paquete no se entrego: este paso no cuenta como cruzado.
        CRUCE.with(|c| c.set(false));
        format!("OMS_ENTRY_WITHHELD: at the local commit: {e}")
    });
    if r.is_ok() {
        // V4L (L-A1): barrera de prueba DESPUES del compromiso y antes de la entrega: un escritor
        // que gana aqui ya no retiene el paquete (pertenece a la revision que lo acepto).
        BARRERA_TRAS_COMPROMISO.with(|b| {
            if let Some(f) = b.borrow().as_ref() {
                f();
            }
        });
    }
    r
}

/// CAPITAL-UNICO V4L (L-A1): el validador corre como compromiso local de esta fase (`None`: como
/// cruce, sin comprometer). Lo consulta el validador del OMS para registrar el testigo.
pub fn fase_en_compromiso() -> Option<FaseDeEntrega> {
    COMPROMETIENDO.with(|c| c.get())
}

/// V4L (L-A1): solo para pruebas. Barrera determinista entre el compromiso local y la entrega.
#[doc(hidden)]
pub fn fijar_barrera_tras_compromiso(barrera: Option<Box<dyn Fn()>>) {
    BARRERA_TRAS_COMPROMISO.with(|b| *b.borrow_mut() = barrera);
}

thread_local! {
    static COMPROMETIENDO: std::cell::Cell<Option<FaseDeEntrega>> = const { std::cell::Cell::new(None) };
    static BARRERA_TRAS_COMPROMISO: std::cell::RefCell<Option<Box<dyn Fn()>>> = const { std::cell::RefCell::new(None) };
}

/// V4K (A1): solo para pruebas. Fija (o quita, con `None`) en ESTE hilo una barrera determinista
/// que corre entre el cruce de la frontera y el compromiso local de cada entrega (mismo `poll`).
#[doc(hidden)]
pub fn fijar_barrera_de_compromiso(barrera: Option<Box<dyn Fn()>>) {
    BARRERA_DE_COMPROMISO.with(|b| *b.borrow_mut() = barrera);
}