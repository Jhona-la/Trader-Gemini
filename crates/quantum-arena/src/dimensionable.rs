//! CAPITAL-UNICO V4G y V4H: el saldo dimensionable (`GlobalArena::unified_capital`, lo que lee
//! Risk) deja de ser un acumulador en el arena vivo y se DERIVA:
//!
//! dimensionable = saldo vigente aceptado por el puente (era y sesion vigentes)
//!                 - debitos que el exchange aun no refleja.
//!
//! El libro vive dentro de la puerta que ya serializa la publicacion de entradas
//! (`QuantumConfig::entry_publication_gate`): el nucleo ya la tiene tomada cuando reserva, asi que
//! no hay candado nuevo en el camino de los ticks. Cada operacion vuelve a calcular el valor entero
//! (no por incrementos): sin debitos es exactamente el saldo vigente.
//!
//! Invariantes (secciones "V4G - DISENO" y "V4H - DISENO" de `integrador_capital_unico.md`):
//!
//! - INV-D: el saldo vigente es el de MAYOR marca del exchange del alcance (era, sesion). Una marca
//!   menor no mueve nada; con marcas iguales, o sin marca, se toma el MENOR de los dos valores.
//! - INV-E: un debito solo sale cuando un saldo vigente tiene marca ESTRICTAMENTE mayor que la
//!   marca propia de ese debito (la mayor `T` de los fills de su orden). Igualdad o marca
//!   desconocida: se conserva. Una reserva viva (sin ejecucion) no la quita ningun saldo:
//!   cancelar la QUITA (no suma dinero). Confirmar o cancelar dos veces no cambia nada.
//! - INV-F: la propuesta se revalida bajo la puerta al reservar ([`LibroDimensionable::admite`])
//!   y al enviar ([`LibroDimensionable::envio_valido`]).
//! - INV-G: los debitos locales conocidos y aun no observados (cierre local con perdida, fill de
//!   cierre, funding pagado) restan en el acto; ningun credito local suma; un cierre externo veta
//!   la admision hasta un saldo pedido despues de detectarlo.
//!
//! Sin activar ([`LibroDimensionable::activar`]: backtest, replay, simuladores, pruebas) el arena
//! sigue siendo el acumulador de siempre: las mismas sumas atomicas que antes.

use crate::state::GlobalArena;
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
    /// Funding pagado visto en el historial de ingresos.
    Funding(u64),
}

#[derive(Debug, Clone, Copy, PartialEq)]
enum Estado {
    /// Reserva de entrada sin ejecucion conocida: ningun saldo la quita.
    Viva,
    /// Estimado local sin marca: ningun saldo lo quita (lo reemplazan los fills de cierre).
    Estimado,
    /// Debito con la marca maxima de sus fills; `None` = desconocida (no se puede probar inclusion).
    Pendiente(Option<u64>),
}

#[derive(Debug, Clone, Copy, PartialEq)]
struct Debito {
    origen: Origen,
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
}

impl Debito {
    /// INV-E: el saldo con marca `marca_saldo` ya lo incluye.
    fn incluido(&self, marca_saldo: Option<u64>) -> bool {
        match (self.estado, marca_saldo) {
            (Estado::Pendiente(Some(t)), Some(m)) => m > t,
            _ => false,
        }
    }
}

fn mayor(a: Option<u64>, b: Option<u64>) -> Option<u64> {
    match (a, b) {
        (Some(x), Some(y)) => Some(x.max(y)),
        (x, None) => x,
        (None, y) => y,
    }
}

/// El libro del dimensionable. Se usa siempre bajo `entry_publication_gate`.
#[derive(Debug, Default)]
pub struct LibroDimensionable {
    derivado: bool,
    saldo: Option<f64>,
    /// INV-D: marca de la lectura que quedo vigente (la que prueba inclusion, INV-E).
    marca: Option<u64>,
    /// INV-D: la mayor marca conocida (ordena las lecturas; no prueba inclusion).
    marca_max: Option<u64>,
    /// INV-F: revision del saldo vigente; sube con cada saldo que cambia valor, marca o alcance.
    rev: u64,
    alcance: Option<(u64, u64)>,
    debitos: Vec<Debito>,
    /// INV-G: veto del libro por un cierre externo. `false`: esperando que el puente vete por la
    /// evidencia perdida que pide el host; `true`: ya veto, y el primer saldo que entregue al
    /// soltar (una REST pedida despues de detectarlo) suelta este veto.
    veto: Option<bool>,
    /// Alcance de la publicacion en curso (lo fija el publicador antes del saldo).
    entrante: Option<(u64, u64)>,
    seq: u64,
}

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

    /// Debitos (reservas y debitos locales) que el exchange aun no refleja.
    pub fn reservado(&self) -> f64 {
        self.debitos.iter().map(|d| d.importe).sum()
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

    /// saldo vigente - reservado, si hay saldo.
    pub fn dimensionable(&self) -> Option<f64> {
        self.saldo.map(|s| s - self.reservado())
    }

    fn publicar(&self, arena: &GlobalArena) {
        if !self.derivado {
            return;
        }
        if let Some(d) = self.dimensionable() {
            arena.unified_capital.store(d, Ordering::Relaxed);
        }
    }

    fn siguiente(&mut self) -> u64 {
        self.seq = self.seq.wrapping_add(1);
        self.seq
    }

    fn retirar_incluidos(&mut self) {
        let marca = self.marca;
        self.debitos.retain(|d| !d.incluido(marca));
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

    fn aceptar(&mut self, arena: &GlobalArena, alcance: Option<(u64, u64)>, saldo: f64, marca: Option<u64>) {
        if !saldo.is_finite() {
            return;
        }
        let mismo_alcance = alcance == self.alcance;
        let (valor, marca_elegida) = match self.saldo {
            None => (saldo, marca),
            Some(actual) => match (marca, self.marca_max) {
                (Some(n), Some(v)) if n > v => (saldo, marca),
                // INV-D: una marca menor no mueve nada dentro del alcance.
                (Some(n), Some(v)) if n < v && mismo_alcance => return,
                // INV-D: marca igual, sin marca, o una foto mas vieja de otro alcance: no hay
                // orden; el menor de los dos, con SU marca (no se presta la de la otra lectura).
                _ => {
                    if saldo < actual {
                        (saldo, marca)
                    } else {
                        (actual, self.marca)
                    }
                }
            },
        };
        if self.saldo != Some(valor) || self.marca != marca_elegida || !mismo_alcance {
            self.rev = self.rev.wrapping_add(1);
        }
        self.marca_max = mayor(self.marca_max, marca);
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
    /// cierre externo, desde aqui el primer saldo entregado al soltar es posterior al hecho.
    pub fn veto_del_puente(&mut self) {
        if let Some(visto) = self.veto.as_mut() {
            *visto = true;
        }
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
            let rev = self.rev;
            self.debitos.push(Debito {
                origen: Origen::Entrada(clave),
                importe: comision.max(0.0),
                estado: Estado::Viva,
                alcance: self.alcance,
                seq,
                base,
                rev,
            });
        }
        self.publicar(arena);
    }

    /// INV-F, al reservar (bajo la puerta, antes de reservar): la entrada que Risk dimensiono con
    /// `capital` sigue solo si el veto del puente esta suelto, el libro no veta y el dimensionable
    /// actual no es menor. Sin activar, siempre.
    pub fn admite(&self, arena: &GlobalArena, capital: Option<f64>) -> bool {
        if !self.derivado {
            return true;
        }
        match (capital, self.dimensionable()) {
            (Some(c), Some(d)) => {
                c.is_finite()
                    && self.veto.is_none()
                    && !arena.capital_incompleto.load(Ordering::Acquire)
                    && d >= c
            }
            _ => false,
        }
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
        // Deliberado: una reserva de entrada POSTERIOR y todavia viva no invalida esta (se
        // decidio contando con ella); una posterior ya ejecutada si cuenta. El margen combinado
        // de las dos lo valida el exchange.
        let ajeno: f64 = self
            .debitos
            .iter()
            .filter(|d| {
                d.seq != propia.seq
                    && !(matches!(d.origen, Origen::Entrada(_)) && d.estado == Estado::Viva && d.seq > propia.seq)
            })
            .map(|d| d.importe)
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

    /// Cancelacion sin ejecucion: quita la reserva viva, no suma dinero. Repetida: nada.
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
            self.debitos.remove(i);
        }
        self.publicar(arena);
    }

    /// Ejecucion confirmada: `devuelta` es la comision de lo NO ejecutado; `marca_fill`, la mayor
    /// `T` del exchange de los fills de esa orden. Repetida: solo puede subir la marca. Si el saldo
    /// vigente ya tiene marca mayor, el debito sale en el acto (sin sumar nada).
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
                            d.importe -= devuelta;
                        }
                        d.estado = Estado::Pendiente(marca_fill);
                    }
                    Estado::Pendiente(m) => d.estado = Estado::Pendiente(mayor(m, marca_fill)),
                    Estado::Estimado => {}
                }
                !(d.importe.is_finite() && d.importe > 0.0)
            };
            if vacia {
                self.debitos.remove(i);
            }
        }
        self.retirar_incluidos();
        self.publicar(arena);
    }

    /// INV-G: cierre local del nucleo de una posicion que existio en el exchange con resultado
    /// neto negativo. Debito estimado (sin marca) hasta que lo reemplacen los fills de cierre.
    pub fn debito_cierre_local(&mut self, arena: &GlobalArena, clave: ClaveReserva, importe: f64) {
        if !self.derivado || !(importe.is_finite() && importe > 0.0) {
            return;
        }
        if self.debitos.iter().any(|d| d.origen == Origen::CierreLocal(clave)) {
            return;
        }
        let seq = self.siguiente();
        self.debitos.push(Debito {
            origen: Origen::CierreLocal(clave),
            importe,
            estado: Estado::Estimado,
            alcance: self.alcance,
            seq,
            base: None,
            rev: 0,
        });
        self.publicar(arena);
    }

    /// INV-G: un fill de cierre del stream. `debito` = comision - PnL realizado de ese fill (solo
    /// cuenta si es positivo: ningun credito suma). `terminal`: la orden de cierre termino; sus
    /// debitos exactos reemplazan el estimado del cierre local de esa moneda.
    pub fn fill_de_cierre(&mut self, arena: &GlobalArena, coin_id: usize, trade_id: u64, debito: f64, marca: u64, terminal: bool) {
        if !self.derivado {
            return;
        }
        let marca = (marca > 0).then_some(marca);
        let origen = Origen::FillCierre { coin_id, trade_id };
        let ya_incluido = matches!((self.marca, marca), (Some(m), Some(t)) if m > t);
        if debito.is_finite() && debito > 0.0 && !ya_incluido && !self.debitos.iter().any(|d| d.origen == origen) {
            let seq = self.siguiente();
            self.debitos.push(Debito {
                origen,
                importe: debito,
                estado: Estado::Pendiente(marca),
                alcance: self.alcance,
                seq,
                base: None,
                rev: 0,
            });
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
        let ya_incluido = matches!((self.marca, marca), (Some(m), Some(t)) if m > t);
        let origen = Origen::Funding(id);
        if ya_incluido || self.debitos.iter().any(|d| d.origen == origen) {
            return;
        }
        let seq = self.siguiente();
        self.debitos.push(Debito {
            origen,
            importe,
            estado: Estado::Pendiente(marca),
            alcance: self.alcance,
            seq,
            base: None,
            rev: 0,
        });
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
