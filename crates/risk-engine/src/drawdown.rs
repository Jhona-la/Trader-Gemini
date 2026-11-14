//! D-744 — LA CAÍDA MÁXIMA NO ES UNA OPINIÓN: ES UNA PRUEBA DE HIPÓTESIS.
//!
//! # Qué estaba mal
//!
//! El gen `global_max_drawdown` nacía como `1 − taker·100` —con la comisión
//! por defecto, 0,95— y su banda evolutiva era [0,50; 0,99]. Ningún genoma
//! podía cortar por debajo de perder la mitad de la cuenta, y el genoma base
//! esperaba a perder el 95 % (con $13, a $0,65) antes de que el sistema inmune
//! aplanara. Peor: el mismo número se leía con TRES semánticas incompatibles
//! —cortacircuitos del host, veto de entradas del risk-engine (mezclado con un
//! 0,85 literal en régimen micro) y colchón de margen del orquestador—, de modo
//! que evolucionar el gen movía tres cosas distintas en direcciones distintas.
//!
//! # Qué lo sustituyó
//!
//! D-744 derivó el umbral de una prueba de racha: con riesgo `r` por
//! operación y tasa de pérdida `q`, la racha más larga que el azar produce
//! con probabilidad `α` en `H` operaciones, `k(α) = ln(H/α)/ln(1/q)`, y
//! `DD_max = 1 − (1 − r)^{k(α)}`, con el gen leído como confianza (α = 1 − gen).
//! Con 13 USD daba 8-12 R y, con +0,30 R de ventaja, una falsa alarma del
//! 36-49 % en 200 operaciones. CL-106 (ADR-0017) la sustituyó por el freno
//! duro de la cuenta ([`umbral_freno_duro`]): el cuantil 95 % de la caída
//! máxima en 200 operaciones bajo H0 = +0,05 R, con la geometría medida de
//! las órdenes, acotado por d* ([`drawdown_de_falsacion`]). CL-121 (T3-09 de
//! la Línea G) retiró las funciones de la prueba de racha, que se habían
//! quedado sin consumidores.
//!
//! F1-085 (Ola Ω189): El gen `global_max_drawdown` se conserva congelado en el genotipo
//! (slot vec[0]) para preservar la dimensión contractual canónica 144 sin generar ruido
//! evolutivo. La protección de capital y el cortacircuitos de cuenta se ejecutan
//! exclusivamente a través de [`freno_duro`] y [`drawdown_de_falsacion`].

/// Horizonte de operaciones sobre el que se exige supervivencia. Es el MISMO
/// que usa el tope de ruina del dimensionado (`ruin::TRADE_HORIZON`): si el
/// sizing promete sobrevivir a la peor racha de H operaciones, el
/// cortacircuitos debe medir contra esa misma promesa.
pub use crate::ruin::TRADE_HORIZON;

/// Media móvil exponencial del riesgo por operación: el peso de la última
/// orden es 1/n hasta n = `memoria`, y 1/memoria a partir de ahí — así la
/// primera orden no queda diluida ni la última manda sola.
#[inline]
pub fn actualizar_riesgo_ewma(previo: f64, nuevo: f64, memoria: f64) -> f64 {
    if !nuevo.is_finite() || nuevo <= 0.0 {
        return previo;
    }
    let m = memoria.max(2.0);
    if !previo.is_finite() || previo <= 0.0 {
        return nuevo;
    }
    previo + (nuevo - previo) / m
}

/// QS-D3 (ADR-0016) — Fracción de Kelly máxima que admite el sistema
/// (½ Kelly: el 75 % del crecimiento óptimo con 1/8 de probabilidad de caer
/// alguna vez a la mitad, frente a 1/2 con Kelly pleno).
pub const FRACCION_KELLY_MAXIMA: f64 = 0.5;

/// QS-D3 — Nivel de la prueba de falsación (el α = 0,05 del z95 del sistema).
pub const ALFA_FALSACION: f64 = 0.05;

/// QS-D3 — LA CAÍDA QUE FALSIFICA EL MODELO DE CRECIMIENTO.
///
/// Con riqueza browniana geométrica apostando una fracción `c` del Kelly
/// óptimo (`c = f/f*`), la probabilidad de caer ALGUNA VEZ a `x·W₀` es
/// `x^(2/c − 1)` (Kelly pleno: `x`; ½ Kelly: `x³`). Igualándola a `α`:
///
/// `P(DD ≥ d) = (1 − d)^(2/c − 1) = α  ⇒  d* = 1 − α^(c/(2 − c))`
///
/// Una caída mayor que `d*` rechaza al nivel `α` que la estrategia tenga el
/// edge con el que se dimensiona: el veto no protege una «opinión» de riesgo,
/// detiene un modelo refutado. Con `c ≤ ½`, `d*(½, 0,05) = 0,632` es la COTA
/// SUPERIOR: cualquier política más conservadora tiene un `d*` menor. Por eso
/// la tolerancia micro (D-641) nunca puede exceder este valor; antes era el
/// literal 0,85, que con 13 USD permitía caer a 1,95 USD.
///
/// Entradas no finitas o fuera de dominio devuelven el `d*` de ½ Kelly a α =
/// 0,05 (el valor más tolerante que el sistema admite), nunca 1.
#[inline]
pub fn drawdown_de_falsacion(c: f64, alfa: f64) -> f64 {
    let c = if c.is_finite() && c > 0.0 { c.min(FRACCION_KELLY_MAXIMA) } else { FRACCION_KELLY_MAXIMA };
    let alfa = if alfa.is_finite() && alfa > 0.0 && alfa < 1.0 { alfa } else { ALFA_FALSACION };
    1.0 - alfa.powf(c / (2.0 - c))
}

/// CL-99 (LC-F4-10, LC-F0-10) — EL PICO ES UNO Y SOBREVIVE AL REINICIO.
///
/// Los dos cortacircuitos de drawdown medían desde picos distintos: el
/// riesgo, desde `RiskEngine::peak_capital`; el sistema inmune del host,
/// desde una variable local de su tarea que nacía con el capital del
/// arranque. Ninguno se guardaba, así que un reinicio fijaba el pico en el
/// capital ya caído y borraba la caída acumulada: el piso de ADR-0016 (D3)
/// no sobrevivía a reiniciar el proceso.
///
/// Ahora el pico vive en `GlobalArena::pico_capital`, lo actualizan los dos
/// con esta función (monótona: sólo sube) y el host lo guarda por entorno.
/// Los umbrales no cambian. Lo que cambia para el host es desde dónde mide:
/// antes, desde el máximo de su propio plano conservador (mínimo entre
/// arena y exchange); ahora, desde el pico compartido, que también
/// actualiza el riesgo con el capital del arena. Es el más alto de los dos.
///
/// Devuelve el pico tras registrar `capital`. Un capital no finito o ≤ 0
/// no mueve el pico.
#[inline]
pub fn registrar_pico(pico: &quantum_arena::AtomicF64, capital: f64) -> f64 {
    use std::sync::atomic::Ordering;
    if !(capital.is_finite() && capital > 0.0) {
        return pico.load(Ordering::Relaxed);
    }
    match pico.fetch_update(Ordering::AcqRel, Ordering::Acquire, |actual| {
        if !(actual.is_finite() && actual >= capital) {
            Some(capital)
        } else {
            None
        }
    }) {
        Ok(_) => capital,
        Err(actual) => actual,
    }
}

/// CL-99 — guarda el pico en `ruta` (escritura a un temporal y renombrado).
/// El formato es el número en texto: la representación decimal de `f64` en
/// Rust vuelve al mismo valor al leerla.
pub fn guardar_pico(ruta: &std::path::Path, pico: f64) -> std::io::Result<()> {
    if !(pico.is_finite() && pico > 0.0) {
        return Err(std::io::Error::new(
            std::io::ErrorKind::InvalidInput,
            "pico no finito o no positivo",
        ));
    }
    // CL-108f: sincronizado con el disco antes de renombrar, para que un
    // corte de luz no deje el archivo renombrado y vacío.
    use std::io::Write;
    let tmp = ruta.with_extension("tmp");
    {
        let mut f = std::fs::File::create(&tmp)?;
        f.write_all(format!("{pico}\n").as_bytes())?;
        f.sync_all()?;
    }
    std::fs::rename(&tmp, ruta)
}

/// CL-133 (B2 de la revisión del PR #62) — lo que se encontró al leer el
/// pico guardado. Un archivo ilegible (coma decimal, BOM, error de E/S) no es
/// un archivo ausente: tratarlo como ausente perdonaba la caída sin aviso, y
/// el host lo sobrescribía a los 5 s. Mismo criterio que el estado aprendido
/// (CL-108f).
#[derive(Debug, Clone, PartialEq)]
pub enum CargaPico {
    /// No hay archivo: primer arranque, o el operador lo borró para empezar
    /// de cero tras un retiro de fondos.
    Ausente,
    /// El archivo existe pero no se puede leer o no contiene un capital
    /// finito y positivo.
    Ilegible(String),
    Leido(f64),
}

/// CL-99 / CL-133 — lee el pico guardado.
pub fn leer_pico(ruta: &std::path::Path) -> CargaPico {
    let texto = match std::fs::read_to_string(ruta) {
        Ok(t) => t,
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => return CargaPico::Ausente,
        Err(e) => return CargaPico::Ilegible(format!("{}: no se puede leer: {e}", ruta.display())),
    };
    // CL-137 (PR70-INT-08 de la Línea G): el motivo nombra el archivo; el
    // bloqueo es fail-closed y el operador tiene que saber cuál reparar
    // (puede ser el antiguo sin venue que se iba a adoptar).
    match texto.trim().parse::<f64>() {
        Ok(pico) if pico.is_finite() && pico > 0.0 => CargaPico::Leido(pico),
        _ => CargaPico::Ilegible(format!(
            "{}: {:?} no es un capital finito y positivo",
            ruta.display(),
            texto.trim().chars().take(40).collect::<String>()
        )),
    }
}

/// CL-133 (R5 de la revisión del PR #62) — el pico es del VENUE de este
/// proceso. Testnet y papel de mainnet guardan en `data/demo` (CL-61): con un
/// solo archivo, el pico de 5 000 USDT de una sesión de testnet bloqueaba la
/// sesión de papel siguiente con 13 USD reales (caída del 99,7 % ≥ d*).
///
/// CL-136 (PR70-INT-02 de la Línea G) — el venue es el del SALDO con que se
/// mide la caída, y el saldo lo decide el endpoint: testnet sólo con
/// `USE_TESTNET=true`. `SHADOW_MODE` y `BINANCE_USE_DEMO` operan papel contra
/// los endpoints y el saldo de MAINNET. El venue de testnet se llama «prueba».
///
/// CAPITAL-UNICO V3 (opción A pura, decisión 4 del coordinador del
/// 2026-11-12) — el ESPACIO del pico es venue + cuenta + definición:
/// `pico_capital_{venue}_{cuenta}_usdt.txt`, con `cuenta` = [`huella_cuenta`]
/// de la clave API del ejecutor. El pico nuevo nace vacío SIEMPRE; ningún
/// archivo de otra definición o de otro espacio se adopta nunca (ni con
/// unidades en apariencia iguales: una foto actual no prueba la unidad del
/// pico histórico), ni se borra, ni se renombra.
pub fn venue_de(testnet: bool) -> &'static str {
    if testnet {
        "prueba"
    } else {
        "mainnet"
    }
}

fn venue_del_pico() -> &'static str {
    venue_de(std::env::var("USE_TESTNET").is_ok_and(|s| s.trim().eq_ignore_ascii_case("true")))
}

/// Huella de la cuenta para el espacio del pico: FNV-1a de 64 bits de la clave
/// API (la pública, nunca el secreto), en 12 cifras hexadecimales. Otra clave
/// = otra cuenta para el pico (rotar la clave reinicia el pico, con aviso).
pub fn huella_cuenta(clave_api: &[u8]) -> String {
    let mut h: u64 = 0xcbf2_9ce4_8422_2325;
    for b in clave_api {
        h ^= u64::from(*b);
        h = h.wrapping_mul(0x0000_0100_0000_01b3);
    }
    format!("{:012x}", h >> 16)
}

/// Espacio del pico: `{venue}_{cuenta}`.
pub fn espacio_pico(testnet: bool, huella: &str) -> String {
    format!("{}_{huella}", venue_de(testnet))
}

fn espacio_vigente() -> &'static std::sync::Mutex<Option<String>> {
    static ESPACIO: std::sync::OnceLock<std::sync::Mutex<Option<String>>> = std::sync::OnceLock::new();
    ESPACIO.get_or_init(|| std::sync::Mutex::new(None))
}

/// El host fija el espacio del pico del proceso (arranque y transición que
/// cambie de cuenta o de venue). Sólo lo leen [`archivo_pico`] y los mensajes.
pub fn fijar_espacio_pico(espacio: &str) {
    *espacio_vigente().lock().unwrap_or_else(|p| p.into_inner()) = Some(espacio.to_string());
}

/// Nombre del archivo del pico del espacio `espacio`.
pub fn archivo_pico_del_espacio(espacio: &str) -> String {
    format!("pico_capital_{espacio}_usdt.txt")
}

/// Marca de la definición del espacio `espacio` (ver [`archivo_marca_definicion`]).
pub fn archivo_marca_del_espacio(espacio: &str) -> String {
    format!("pico_capital_{espacio}_usdt.definicion")
}

/// Archivo del pico del espacio vigente ([`fijar_espacio_pico`]); sin espacio
/// fijado, el del venue sin cuenta (`pico_capital_{venue}_sin_cuenta_usdt.txt`).
pub fn archivo_pico() -> String {
    let e = espacio_vigente().lock().unwrap_or_else(|p| p.into_inner()).clone();
    archivo_pico_del_espacio(&e.unwrap_or_else(|| format!("{}_sin_cuenta", venue_del_pico())))
}

/// CAPITAL-UNICO — archivo del pico de este venue con la definición anterior
/// (`totalWalletBalance`, todos los activos). Sólo se nombra en el aviso.
pub fn archivo_pico_definicion_anterior() -> String {
    format!("pico_capital_{}.txt", venue_del_pico())
}

/// El archivo del pico de V2 (definición USDT, sin cuenta en el espacio). Sólo
/// se nombra en el aviso.
pub fn archivo_pico_v2_sin_cuenta() -> String {
    format!("pico_capital_{}_usdt.txt", venue_del_pico())
}

/// El archivo del pico de antes de CL-133, sin venue. Sólo se nombra en el aviso.
pub const ARCHIVO_PICO_SIN_VENUE: &str = "pico_capital.txt";

/// CAPITAL-UNICO V2 (revisión independiente, 7a) — marca de la definición del
/// pico del espacio vigente. El host la escribe tras guardar por primera vez
/// el pico del espacio. Con ella se distingue «el pico falta porque el
/// operador lo retiró» (hay marca: rearme de los stops, F1-011) de «el pico
/// falta porque el espacio es nuevo» (no hay marca: los stops se conservan).
pub fn archivo_marca_definicion() -> String {
    archivo_pico().replace("_usdt.txt", "_usdt.definicion")
}

/// Contenido de la marca de la definición.
///
/// CAPITAL-UNICO V4J (punto 5): la marca nombra la MAGNITUD del pico. Desde
/// V4J el pico es el máximo de la equity conservadora del puente; antes de V4J
/// se registraba con el mínimo entre la capacidad del libro y la equity, que
/// queda por debajo. Un pico guardado con otra marca (o sin ella) es de
/// procedencia incierta: no se adopta por coincidencia numérica (ver
/// [`decidir_arranque_pico`]).
pub const MARCA_DEFINICION: &str =
    "capital = equity USDT conservadora del puente; pico de esa magnitud (CAPITAL-UNICO V4J)\n";

/// V4J (punto 5): la marca existe y nombra la magnitud vigente.
pub fn marca_vigente(ruta: &std::path::Path) -> bool {
    std::fs::read_to_string(ruta).is_ok_and(|t| t == MARCA_DEFINICION)
}

/// Escribe la marca de la definición (temporal y renombrado, como el pico).
pub fn guardar_marca_definicion(ruta: &std::path::Path) -> std::io::Result<()> {
    use std::io::Write;
    let tmp = ruta.with_extension("definicion.tmp");
    {
        let mut f = std::fs::File::create(&tmp)?;
        f.write_all(MARCA_DEFINICION.as_bytes())?;
        f.sync_all()?;
    }
    std::fs::rename(&tmp, ruta)
}

/// Rutas que mira el arranque del pico.
#[derive(Debug, Clone, PartialEq)]
pub struct RutasPico {
    /// Pico del espacio: el único que se lee como pico.
    pub pico: std::path::PathBuf,
    /// Marca de la definición del espacio.
    pub marca: std::path::PathBuf,
    /// Archivos de otra definición o de otro espacio de este venue: sólo se
    /// nombran en la línea de aviso. Nunca se adoptan, escriben, borran ni renombran.
    pub ajenos: Vec<std::path::PathBuf>,
}

impl RutasPico {
    /// Las rutas del espacio `espacio`; `ruta` pone el directorio de datos.
    pub fn del_espacio(espacio: &str, ruta: impl Fn(&str) -> std::path::PathBuf) -> Self {
        let venue = espacio.split('_').next().unwrap_or("");
        let mut ajenos = vec![
            ruta(&format!("pico_capital_{venue}.txt")),
            ruta(ARCHIVO_PICO_SIN_VENUE),
            ruta(&format!("pico_capital_{venue}_usdt.txt")),
        ];
        ajenos.dedup();
        Self {
            pico: ruta(&archivo_pico_del_espacio(espacio)),
            marca: ruta(&archivo_marca_del_espacio(espacio)),
            ajenos,
        }
    }
}

/// Lo que el arranque hace con el pico.
#[derive(Debug, Clone, PartialEq)]
pub struct ArranquePico {
    /// El pico a restaurar (o ausente / ilegible), como lo usa el host.
    pub carga: CargaPico,
    /// `true` sólo si el operador retiró el pico (hay marca y no hay pico):
    /// rearme de los contadores en stops (F1-011). Espacio nuevo: `false`.
    pub rearmar_stops: bool,
    /// Como mucho UNA línea para el log del arranque (picos ajenos ignorados).
    pub avisos: Vec<String>,
}

/// CAPITAL-UNICO V3 (opción A pura) — decide el pico del arranque. Sólo lee.
///
/// - El pico del espacio manda si existe (legible o ilegible, CL-133 B2).
/// - Si falta y hay marca, lo retiró el operador: se rearman los stops (F1-011).
/// - Si falta y no hay marca, el espacio es nuevo: el pico nace vacío y los
///   stops se conservan (7a).
/// - Ningún archivo ajeno se adopta, escribe, borra ni renombra; los que
///   existan se nombran en una sola línea de aviso.
pub fn decidir_arranque_pico(rutas: &RutasPico) -> ArranquePico {
    let avisos = aviso_picos_ignorados(&rutas.ajenos).into_iter().collect();
    let carga = match leer_pico(&rutas.pico) {
        // V4J (punto 5): un pico legible sin la marca de la magnitud vigente es
        // de procedencia incierta (un pico legado del mínimo con la capacidad, o
        // sin marca). Se trata como ilegible: no entran posiciones nuevas y el
        // archivo no se sobrescribe hasta que el operador lo migre (marca
        // vigente tras verificarlo) o lo retire. Nunca se adopta por su cifra.
        CargaPico::Leido(p) if !marca_vigente(&rutas.marca) => CargaPico::Ilegible(format!(
            "{}: pico {p} de procedencia incierta (sin la marca de la magnitud vigente en {}); migrarlo o retirarlo",
            rutas.pico.display(),
            rutas.marca.display()
        )),
        otra => otra,
    };
    let rearmar_stops = matches!(carga, CargaPico::Ausente) && rutas.marca.exists();
    ArranquePico {
        carga,
        rearmar_stops,
        avisos,
    }
}

/// CAPITAL-UNICO V3 — UNA línea con los picos de otra definición o de otro
/// espacio que el arranque IGNORA (`None` si no hay ninguno). Sólo lee.
pub fn aviso_picos_ignorados(rutas: &[std::path::PathBuf]) -> Option<String> {
    let partes: Vec<String> = rutas
        .iter()
        .filter_map(|ruta| match leer_pico(ruta) {
            CargaPico::Ausente => None,
            CargaPico::Leido(p) => Some(format!("{p:.4} en {}", ruta.display())),
            CargaPico::Ilegible(m) => Some(format!("ilegible ({m})")),
        })
        .collect();
    (!partes.is_empty()).then(|| {
        format!(
            "picos de otra definición o de otro espacio IGNORADOS (opción A: el pico de este espacio nace vacío): {}; no se adoptan, no se borran y no se renombran",
            partes.join("; ")
        )
    })
}

/// CL-132 (R3 de la revisión del PR #62) / F2 (Ola Omega220): EL CAPITAL DE LOS
/// FRENOS DE DRAWDOWN: el más conservador de los dos planos, el del arena y el
/// saldo del exchange que publica el host (X-013). El host y el riesgo registran
/// el pico y miden la caída con él.
///
/// Si el saldo del exchange es FINITO (incluyendo 0.0 o saldo negativo por deuda),
/// se toma estrictamente `capital.min(exchange)`: un cero observado en exchange
/// apaga inmediatamente el sistema por caída de fondos.
/// Si el exchange no ha sido observado (f64::NAN en backtest o inicio sin conexión),
/// se utiliza el capital del arena.
///
/// CAPITAL-UNICO V4J (punto 5, `qra_plano_del_freno.md`): con el libro del
/// dimensionable activo (`dimensionable_derivado`), `unified_capital` es la
/// CAPACIDAD de entrada (saldo menos reservas y débitos aún no incluidos): una
/// reserva o un débito local no son una caída del patrimonio. El freno y el
/// pico miden entonces la equity conservadora que acepta y conserva el puente
/// (la que el inmune publica en `capital_exchange`); sin ella, el valor no es
/// finito (ausencia, nunca la capacidad del libro): no escribe pico y el
/// freno lo distingue antes de medir. Cero y deuda observados son cifras
/// válidas. Sin activar (acumulador: backtest, replay, simulación), el mínimo
/// de siempre. No cambia ningún umbral: sólo la magnitud que se mide.
pub fn capital_del_freno(arena: &quantum_arena::GlobalArena) -> f64 {
    use std::sync::atomic::Ordering::Relaxed;
    if arena
        .config
        .dimensionable_derivado
        .load(std::sync::atomic::Ordering::Acquire)
    {
        // Acquire: empareja con `publicar_equity_del_freno` (el pico ya incluye esta equity).
        let exchange = arena.capital_exchange.load(std::sync::atomic::Ordering::Acquire);
        return if exchange.is_finite() { exchange } else { f64::NAN };
    }
    let exchange = arena.capital_exchange.load(Relaxed);
    let capital = arena.unified_capital.load(Relaxed);
    if exchange.is_finite() {
        capital.min(exchange)
    } else {
        capital
    }
}


/// CAPITAL-UNICO V4J (punto 5): publica la magnitud del freno con el libro derivado: la equity
/// conservadora del puente (`None` o no finita: ausencia, NaN). El pico de ese ambito se registra
/// ANTES de publicar la magnitud (`Release`): quien la lea (`capital_del_freno`, `Acquire`) ve un
/// pico que ya la incluye. El host la llama bajo el mutex del puente (mismo paso que la
/// observacion aceptada, el relevo y el cambio de espacio del pico).
pub fn publicar_equity_del_freno(arena: &quantum_arena::GlobalArena, equity: Option<f64>) {
    let e = equity.filter(|e| e.is_finite()).unwrap_or(f64::NAN);
    escribir_plano_del_freno(arena, || {
        // V4K (F6): con el pico del ambito pendiente de restaurar, la equity no es evidencia lista:
        // ni escribe pico ni se publica (ausencia) hasta `pico_restaurado`.
        if arena.pico_pendiente.load(std::sync::atomic::Ordering::Acquire) {
            arena.capital_exchange.store(f64::NAN, std::sync::atomic::Ordering::Release);
            return;
        }
        registrar_pico(&arena.pico_capital, e);
        arena.capital_exchange.store(e, std::sync::atomic::Ordering::Release);
    });
}

/// CAPITAL-UNICO V4K (F6): prepara un ambito nuevo del plano del freno (arranque, o relevo a otra
/// cuenta o venue): magnitud en ausencia, pico vacio, pico pendiente de restaurar y ambito nuevo,
/// en una sola revision. Hasta [`pico_restaurado`] ninguna equity es evidencia lista.
pub fn preparar_ambito_del_pico(arena: &quantum_arena::GlobalArena) {
    use std::sync::atomic::Ordering;
    escribir_plano_del_freno(arena, || {
        arena.pico_pendiente.store(true, Ordering::Release);
        arena.ambito_del_plano.fetch_add(1, Ordering::AcqRel);
        arena.capital_exchange.store(f64::NAN, Ordering::Release);
        arena.pico_capital.store(0.0, Ordering::Release);
    });
}

/// CAPITAL-UNICO V4K (F6): el pico persistido del ambito ya se aplico (leido, ausente o
/// ilegible, que veta por `pico_ilegible`): desde aqui la equity vuelve a ser evidencia.
pub fn pico_restaurado(arena: &quantum_arena::GlobalArena) {
    escribir_plano_del_freno(arena, || {
        arena.pico_pendiente.store(false, std::sync::atomic::Ordering::Release);
    });
}

/// CAPITAL-UNICO V4K (F6): el ambito vigente del plano del freno.
pub fn ambito_del_plano(arena: &quantum_arena::GlobalArena) -> u64 {
    arena.ambito_del_plano.load(std::sync::atomic::Ordering::Acquire)
}

/// CAPITAL-UNICO V4K (F2): una observacion del plano del freno (equity del puente y pico de su
/// ambito) con la revision comun que la identifica. Risk y el host deciden con la misma.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct VistaDelFreno {
    /// La equity conservadora del puente (no finita: ausencia).
    pub equity: f64,
    /// El pico del ambito vigente.
    pub pico: f64,
    /// La revision par de `GlobalArena::plano_freno_rev` en la que se leyeron los dos.
    pub revision: u64,
    /// V4K (F6): el ambito del plano en esa revision.
    pub ambito: u64,
    /// V4K (F6): el pico del ambito ya esta restaurado (la equity es evidencia lista).
    /// V4L (L-F6): y la vista es comparable: equity finita y pico finito y positivo del MISMO
    /// ambito y revision. Un saldo que llega antes, una equity ausente (NaN) o un pico vacio no es
    /// una vista lista (no se sustituye la evidencia ausente por caida cero).
    pub lista: bool,
    /// V4L (L-F2): el pico guardado es ilegible (CL-133), leido en la MISMA revision.
    pub pico_ilegible: bool,
}

/// CAPITAL-UNICO V4K (F2): escribe el plano del freno (equity y/o pico) como UNA revision: la sube
/// a impar antes y a par despues. Los escritores del host (observador del puente, lazo del inmune,
/// relevo y arranque del pico) ya se serializan bajo el mutex del puente; esto hace que el lector
/// sin candado (Risk) no combine la equity de una observacion con el pico de otra.
pub fn escribir_plano_del_freno<R>(arena: &quantum_arena::GlobalArena, f: impl FnOnce() -> R) -> R {
    let _g = EscrituraDelPlano::abrir(arena);
    f()
}

thread_local! {
    static PROFUNDIDAD_PLANO: std::cell::Cell<u32> = const { std::cell::Cell::new(0) };
}

/// CAPITAL-UNICO V4K (F2): guard de una escritura del plano del freno. Reentrante en el mismo
/// hilo (un escritor dentro de otro, por ejemplo el observador del puente dentro del relevo):
/// solo el mas externo abre (impar) y cierra (par) la revision.
/// V4L (L-F2/F5): el escritor mas externo toma ademas el candado de escritor unico del plano
/// (`GlobalArena::plano_freno_escritor`) durante toda su revision: dos escritores de hilos distintos
/// (por ejemplo el observador del puente y el arranque del pico) nunca abren revisiones a la vez.
pub struct EscrituraDelPlano<'a> {
    arena: &'a quantum_arena::GlobalArena,
    escritor: Option<std::sync::MutexGuard<'a, ()>>,
}

impl<'a> EscrituraDelPlano<'a> {
    pub fn abrir(arena: &'a quantum_arena::GlobalArena) -> Self {
        let externo = PROFUNDIDAD_PLANO.with(|p| p.get() == 0);
        // V4L (L-F2/F5): el candado se toma ANTES de marcar la profundidad (si otro escritor lo
        // tiene, este hilo aun no figura como escritor).
        let escritor = externo.then(|| arena.plano_freno_escritor.lock().unwrap_or_else(|e| e.into_inner()));
        PROFUNDIDAD_PLANO.with(|p| p.set(p.get() + 1));
        if externo {
            arena.plano_freno_rev.fetch_add(1, std::sync::atomic::Ordering::AcqRel);
        }
        Self { arena, escritor }
    }

    /// V4L (L-F2): la revision par que deja esta escritura al cerrarse (si es la externa): el
    /// testigo con el que el host etiqueta su paso. Dentro de otra escritura, la de la externa.
    pub fn revision_al_cerrar(&self) -> u64 {
        let r = self.arena.plano_freno_rev.load(std::sync::atomic::Ordering::Acquire);
        if r % 2 == 1 { r + 1 } else { r }
    }
}

impl Drop for EscrituraDelPlano<'_> {
    fn drop(&mut self) {
        let externo = PROFUNDIDAD_PLANO.with(|p| {
            let d = p.get().saturating_sub(1);
            p.set(d);
            d == 0
        });
        if externo {
            self.arena.plano_freno_rev.fetch_add(1, std::sync::atomic::Ordering::Release);
        }
        drop(self.escritor.take());
    }
}

/// CAPITAL-UNICO V4K (F2): lectura coherente del plano del freno (seqlock). `None` si no se pudo
/// leer una revision estable (un escritor a mitad en todos los intentos): el llamador veta.
///
/// V4L (L-F2/F5): `SONDEOS_DE_LA_VISTA` es una cota de implementacion de la adquisicion (no un
/// parametro economico ni una politica): con los escritores serializados una revision dura pocas
/// instrucciones, y si en esa cota no hay una revision par estable la adquisicion falla CERRADA
/// (`None`: el llamador veta). No se cambio su valor respecto de K.
pub const SONDEOS_DE_LA_VISTA: usize = 64;

pub fn vista_del_freno(arena: &quantum_arena::GlobalArena) -> Option<VistaDelFreno> {
    use std::sync::atomic::{fence, Ordering};
    for _ in 0..SONDEOS_DE_LA_VISTA {
        let r1 = arena.plano_freno_rev.load(Ordering::Acquire);
        if r1 % 2 == 1 {
            std::hint::spin_loop();
            continue;
        }
        let equity = arena.capital_exchange.load(Ordering::Acquire);
        let pico = arena.pico_capital.load(Ordering::Acquire);
        let ambito = arena.ambito_del_plano.load(Ordering::Acquire);
        let restaurado = !arena.pico_pendiente.load(Ordering::Acquire);
        let pico_ilegible = arena.pico_ilegible.load(Ordering::Acquire);
        fence(Ordering::Acquire);
        let r2 = arena.plano_freno_rev.load(Ordering::Relaxed);
        if r1 == r2 {
            // V4L (L-F6): lista = restaurado + equity finita + pico compatible (finito y > 0) de
            // esta misma revision. Sin eso, ausencia (veto), nunca caida cero.
            let lista = restaurado && equity.is_finite() && pico.is_finite() && pico > 0.0;
            let equity = if equity.is_finite() && restaurado { equity } else { f64::NAN };
            return Some(VistaDelFreno { equity, pico, revision: r1, ambito, lista, pico_ilegible });
        }
    }
    None
}

/// CL-106 (LC-F4-10, decisión del dueño «Ambos», 2026-10-12) — FRENO DURO DE
/// LA CUENTA: UNA PRUEBA DE NIVEL α CON EL TAMAÑO REAL DE LAS ÓRDENES.
///
/// H0 es «el sistema tiene al menos la ventaja mínima que justifica operar»,
/// [`VENTAJA_H0_FRENO_DURO_R`] = +0,05 R por operación, con R el riesgo de la
/// orden al stop con su fricción (`SL + f`). Una operación gana `w = (TP − f)/SL`
/// o pierde `l = (SL + f)/SL`, en unidades del stop; con ventaja μ0 (en R),
/// `p·w − (1 − p)·l = μ0·l`. La caída máxima de la cuenta en
/// [`HORIZONTE_FRENO_DURO`] operaciones es una variable aleatoria bajo H0; su
/// cuantil `1 − α` ([`ALFA_FALSACION`]) en stops, `K(w, l)`, es el umbral: si
/// la caída lo supera, H0 se rechaza con error ≤ α en ese horizonte.
///
/// Con 13 USD, la orden mínima de 5,10 USD, RR 2,25 y 14,5 pb de fricción,
/// `K` son unos 31-33 R (7,8 % de la cuenta con stop del 0,5 %; 15 % con
/// stop del 1 %). El freno de racha de antes (`drawdown_compatible`, 8-12 R)
/// daba un 36-49 % de falsa alarma en 200 operaciones con una ventaja de
/// +0,30 R; éste, ≈ 5 % con +0,05 R y ≈ 0 con +0,30 R.
///
/// Una caída medida desde el pico móvil sólo tiene nivel con horizonte: con
/// 2 000 operaciones la falsa alarma de este umbral sube al 52-57 %. El
/// freno de la estrategia (proceso e de Ville sobre R) es la prueba con nivel
/// para todo tiempo; éste protege la cuenta. d* (ADR-0016 D3) queda como
/// tope de ruina por encima.
pub const VENTAJA_H0_FRENO_DURO_R: f64 = 0.05;
/// Horizonte del freno duro: el del tope de ruina del dimensionado.
pub const HORIZONTE_FRENO_DURO: usize = TRADE_HORIZON as usize;
/// Caminos del Monte Carlo del cuantil: el error de muestreo del cuantil 95 %
/// es ≈ √(α(1−α)/n)/densidad, menos de un 2 % relativo.
const CAMINOS_FRENO_DURO: usize = 20_000;

/// Generador xorshift64* con semilla fija: el umbral es una función
/// determinista de la geometría, no de una semilla del proceso.
struct Xorshift(u64);
impl Xorshift {
    fn uniforme(&mut self) -> f64 {
        self.0 ^= self.0 >> 12;
        self.0 ^= self.0 << 25;
        self.0 ^= self.0 >> 27;
        (self.0.wrapping_mul(0x2545_F491_4F6C_DD1D) >> 11) as f64 / (1u64 << 53) as f64
    }
}

/// Cuantil `1 − α` de la caída máxima (en stops) de `HORIZONTE_FRENO_DURO`
/// operaciones que ganan `w` o pierden `l` stops, con ventaja μ0 (en R).
/// `None` si la geometría no es utilizable o la ventaja hace imposible
/// perder (no hay caída que medir).
pub fn caida_alfa_en_stops(w: f64, l: f64, ventaja_r: f64, alfa: f64) -> Option<f64> {
    if !(w.is_finite() && w > 0.0 && l.is_finite() && l > 0.0 && ventaja_r.is_finite())
        || !(alfa.is_finite() && alfa > 0.0 && alfa < 1.0)
    {
        return None;
    }
    let p = (ventaja_r * l + l) / (w + l);
    if !(p.is_finite() && p > 0.0 && p < 1.0) {
        return None;
    }
    let mut azar = Xorshift(0x9E37_79B9_7F4A_7C15);
    let mut caidas: Vec<f64> = Vec::with_capacity(CAMINOS_FRENO_DURO);
    for _ in 0..CAMINOS_FRENO_DURO {
        let (mut acum, mut pico, mut caida) = (0.0f64, 0.0f64, 0.0f64);
        for _ in 0..HORIZONTE_FRENO_DURO {
            acum += if azar.uniforme() < p { w } else { -l };
            pico = pico.max(acum);
            caida = caida.max(pico - acum);
        }
        caidas.push(caida);
    }
    let k = (((1.0 - alfa) * CAMINOS_FRENO_DURO as f64).ceil() as usize)
        .clamp(1, CAMINOS_FRENO_DURO)
        - 1;
    let (_, q, _) = caidas.select_nth_unstable_by(k, |a, b| a.total_cmp(b));
    Some(*q)
}

/// Paso de la caché del freno duro: la geometría (w̄, l̄) se redondea a 0,02
/// stops (el umbral cambia menos de un 1 % en ese paso).
pub const PASO_CACHE_FRENO_DURO: f64 = 0.02;
/// Radio, en pasos de la caché, dentro del cual un K ya calculado sirve de
/// respaldo conservador mientras llega el exacto (0,10 stops: el umbral de
/// los vecinos difiere del exacto en pocos puntos porcentuales).
pub const RADIO_VECINO_FRENO_DURO: i64 = 5;
/// Capacidad máxima de `exactos` en la caché estática para acotar memoria en daemons largos.
pub const MAX_ENTRADAS_CACHE_FRENO_DURO: usize = 4096;

struct CacheFrenoDuro {
    exactos: std::collections::HashMap<(i64, i64), Option<f64>>,
    pendientes: std::collections::HashSet<(i64, i64)>,
}

fn cache_freno_duro() -> &'static std::sync::Mutex<CacheFrenoDuro> {
    static CACHE: std::sync::OnceLock<std::sync::Mutex<CacheFrenoDuro>> = std::sync::OnceLock::new();
    CACHE.get_or_init(|| {
        std::sync::Mutex::new(CacheFrenoDuro {
            exactos: std::collections::HashMap::new(),
            pendientes: std::collections::HashSet::new(),
        })
    })
}

fn clave_freno_duro(w: f64, l: f64) -> Option<(i64, i64)> {
    if !(w.is_finite() && l.is_finite()) {
        return None;
    }
    Some((
        (w / PASO_CACHE_FRENO_DURO).round() as i64,
        (l / PASO_CACHE_FRENO_DURO).round() as i64,
    ))
}

fn k_exacto_de_clave(clave: (i64, i64)) -> Option<f64> {
    caida_alfa_en_stops(
        clave.0 as f64 * PASO_CACHE_FRENO_DURO,
        clave.1 as f64 * PASO_CACHE_FRENO_DURO,
        VENTAJA_H0_FRENO_DURO_R,
        ALFA_FALSACION,
    )
}

fn insertar_k_exacto(clave: (i64, i64), k: Option<f64>) {
    let mut c = cache_freno_duro()
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner);
    if c.exactos.len() >= MAX_ENTRADAS_CACHE_FRENO_DURO && !c.exactos.contains_key(&clave) {
        if let Some(&lejana) = c
            .exactos
            .keys()
            .max_by_key(|&&(kw, kl)| (kw - clave.0).saturating_abs().saturating_add((kl - clave.1).saturating_abs()))
        {
            c.exactos.remove(&lejana);
        }
    }
    c.exactos.insert(clave, k);
    c.pendientes.remove(&clave);
}

/// Lanza el Monte Carlo exacto de `clave` en un hilo de fondo. Si el sistema
/// no concede el hilo, lo calcula aquí mismo (una vez; nunca queda pendiente).
fn lanzar_calculo_fondo(clave: (i64, i64)) {
    let lanzado = std::thread::Builder::new()
        .name("freno-duro-mc".into())
        .spawn(move || insertar_k_exacto(clave, k_exacto_de_clave(clave)));
    if lanzado.is_err() {
        insertar_k_exacto(clave, k_exacto_de_clave(clave));
    }
}

/// [`caida_alfa_en_stops`] con H0 y α del freno duro, en caché por geometría
/// redondeada a [`PASO_CACHE_FRENO_DURO`].
///
/// F-A4 (Fable, 2026-10-17; hallazgo F1-012). Hasta aquí el Monte Carlo
/// (20 000 caminos × 200 pasos) corría SÍNCRONO en el hilo del evento —
/// dentro de `evaluate_quantum_order`, en cada intención— la primera vez que
/// aparecía cada geometría, y w̄/l̄ derivan con cada orden abierta: paradas de
/// decenas de milisegundos en el camino caliente. Ahora:
/// 1. acierto de caché ⇒ una búsqueda bajo un cerrojo breve;
/// 2. fallo CON vecinos ya calculados a ≤ [`RADIO_VECINO_FRENO_DURO`] pasos ⇒
///    devuelve el MÍNIMO K de esos vecinos (conservador: el freno dispara
///    antes, nunca después) y lanza el exacto en un hilo de fondo; la
///    siguiente consulta ya lo encuentra;
/// 3. fallo SIN vecinos (arranque en frío) ⇒ cálculo síncrono, una vez.
/// El umbral exacto es la MISMA función pura: no cambia ningún umbral, sólo
/// cuándo se calcula. La asignación del hilo ocurre sólo en el caso 2, que es
/// raro (una vez por cada 0,02 stops de deriva de la geometría).
pub fn caida_alfa_freno_duro(w: f64, l: f64) -> Option<f64> {
    let clave = clave_freno_duro(w, l)?;
    let (respaldo, lanzar) = {
        let mut c = cache_freno_duro()
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        if let Some(k) = c.exactos.get(&clave) {
            return *k;
        }
        let mut hay_vecino = false;
        let mut minimo: Option<f64> = None;
        let radio = RADIO_VECINO_FRENO_DURO;
        let max_vecinos = (2 * radio * (radio + 1) + 1) as usize;
        if c.exactos.len() <= max_vecinos {
            for (ck, kv) in c.exactos.iter() {
                if (ck.0 - clave.0).abs() + (ck.1 - clave.1).abs() <= radio {
                    hay_vecino = true;
                    if let Some(k) = kv {
                        minimo = Some(minimo.map_or(*k, |m: f64| m.min(*k)));
                    }
                }
            }
        } else {
            for dw in -radio..=radio {
                let rem = radio - dw.abs();
                for dl in -rem..=rem {
                    if let Some(kv) = c.exactos.get(&(clave.0 + dw, clave.1 + dl)) {
                        hay_vecino = true;
                        if let Some(k) = kv {
                            minimo = Some(minimo.map_or(*k, |m: f64| m.min(*k)));
                        }
                    }
                }
            }
        }
        if hay_vecino {
            (Some(minimo), c.pendientes.insert(clave))
        } else {
            (None, false)
        }
    };
    match respaldo {
        Some(k_vecino) => {
            if lanzar {
                lanzar_calculo_fondo(clave);
            }
            k_vecino
        }
        None => {
            let k = k_exacto_de_clave(clave);
            insertar_k_exacto(clave, k);
            k
        }
    }
}

/// Número de geometrías cuyo K exacto se está calculando en segundo plano.
pub fn freno_duro_calculos_pendientes() -> usize {
    cache_freno_duro()
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner)
        .pendientes
        .len()
}

/// Espera, como mucho `limite`, a que no quede ningún cálculo pendiente.
/// Devuelve `true` si la caché quedó al día. Para diagnósticos y pruebas; el
/// camino caliente nunca espera.
pub fn freno_duro_esperar_pendientes(limite: std::time::Duration) -> bool {
    let inicio = std::time::Instant::now();
    while freno_duro_calculos_pendientes() > 0 {
        if inicio.elapsed() >= limite {
            return false;
        }
        std::thread::sleep(std::time::Duration::from_millis(2));
    }
    true
}

/// CL-130 (R1 de la revisión del PR #62) — un cierre suma su resultado en
/// stops de SU posición (PnL neto / riesgo al stop de esa posición) a la
/// cuenta de la caída del freno duro. Devuelve la caída en stops tras
/// registrarlo. Un resultado no finito (una posición sin SL propio) no
/// cuenta: su caída la cubre d* sobre el capital.
///
/// Antes la caída se medía en USD y se convertía en stops con `s̄`, la EWMA
/// del riesgo al stop de las últimas órdenes (CL-106b). Sólo es la misma
/// cuenta con tamaño fijo: desde el dimensionado en espacio de riesgo
/// (LH-04) y el factor de racha (GLM 118) el tamaño baja en una racha
/// perdedora, `s̄` encoge y la misma caída en USD contaba como muchos más
/// stops. Con L = 10 disparaba tras 13 SL en vez de 31.
pub fn registrar_resultado_en_stops(arena: &quantum_arena::GlobalArena, resultado: f64) -> f64 {
    use std::sync::atomic::Ordering;
    if resultado.is_finite() {
        let suma = arena.resultado_en_stops.fetch_add(resultado, Ordering::AcqRel) + resultado;
        let _ = arena
            .pico_en_stops
            .fetch_update(Ordering::AcqRel, Ordering::Acquire, |pico| {
                (pico.is_nan() || suma > pico).then_some(suma)
            });
        arena.push_resultado_en_stops_ventana(resultado);
    }
    caida_en_stops(arena)
}

/// CL-130 — caída de los cierres desde su máximo, en stops de sus
/// posiciones (≥ 0).
pub fn caida_en_stops(arena: &quantum_arena::GlobalArena) -> f64 {
    use std::sync::atomic::Ordering::Relaxed;
    // F1-011: Medición de caída sobre ventana deslizante de HORIZONTE_FRENO_DURO (200) operaciones.
    // Preserva la validez estadística del cuantil K(alpha=0.05) para cualquier número de operaciones
    // de la cuenta, erradicando falsas alarmas del 57% y estados absorbentes permanentes.
    let caida_win = arena.caida_en_stops_ventana();
    if caida_win > 0.0 {
        caida_win
    } else {
        let caida = arena.pico_en_stops.load(Relaxed) - arena.resultado_en_stops.load(Relaxed);
        if caida.is_finite() {
            caida.max(0.0)
        } else {
            0.0
        }
    }
}

/// CL-134 (R6 de la revisión del PR #62) — la cuenta empieza de cero con
/// `capital`: el pico, lo medido de las órdenes y la caída en stops. Es lo
/// que hace falta para replantar un universo del bosque sombra, que es otra
/// hipótesis.
pub fn reiniciar_la_cuenta(arena: &quantum_arena::GlobalArena, capital: f64) {
    use std::sync::atomic::Ordering::Relaxed;
    arena.pico_capital.store(capital, Relaxed);
    arena.reset_ventana_stops();
    for celda in [
        &arena.riesgo_por_operacion,
        &arena.riesgo_por_operacion_usd,
        &arena.ganancia_en_stops,
        &arena.perdida_en_stops,
        &arena.resultado_en_stops,
        &arena.pico_en_stops,
    ] {
        celda.store(0.0, Relaxed);
    }
}

/// CL-106 → CL-130 — el FRENO DURO de la cuenta, el MISMO para el veto de
/// entradas del riesgo y el sistema inmune del host.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct FrenoDuro {
    /// Caída de los cierres en stops de sus posiciones.
    pub caida_en_stops: f64,
    /// `K(w̄, l̄)`: cuantil 95 % de la caída máxima en 200 operaciones con
    /// H0 = +0,05 R y la geometría medida de las órdenes. `None` sin
    /// geometría medida: no se inventa un umbral.
    pub umbral_en_stops: Option<f64>,
    /// Caída del capital desde el pico, en fracción del pico.
    pub caida_capital: f64,
    /// d* (ADR-0016 D3), el tope de ruina.
    pub tope_capital: f64,
}

impl FrenoDuro {
    /// La caída en stops llegó a `K`.
    pub fn por_stops(&self) -> bool {
        self.umbral_en_stops.is_some_and(|k| self.caida_en_stops >= k)
    }
    /// La caída del capital llegó a d*.
    pub fn por_capital(&self) -> bool {
        self.caida_capital >= self.tope_capital
    }
    pub fn echado(&self) -> bool {
        self.por_stops() || self.por_capital()
    }
}

impl std::fmt::Display for FrenoDuro {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let k = self
            .umbral_en_stops
            .map_or_else(|| "sin medir".to_string(), |k| format!("{k:.1}"));
        write!(
            f,
            "caída de {:.1} stops de sus posiciones (K = {k}) y del {:.1} % del capital (d* = {:.1} %)",
            self.caida_en_stops,
            self.caida_capital * 100.0,
            self.tope_capital * 100.0
        )
    }
}

/// CL-106 → CL-130 — el freno duro con lo medido en el arena, el pico y el
/// capital desde los que se mide la caída ([`capital_del_freno`]).
///
/// Dispara cuando la caída de los cierres, en stops de sus posiciones
/// ([`registrar_resultado_en_stops`]), llega a `K(w̄, l̄)`, o cuando la caída
/// del capital desde el pico llega a d*. La parte en stops sólo ve lo
/// cerrado: las pérdidas abiertas las acotan los stops de las posiciones y
/// d* sobre el capital, que incluye lo que el host mide del exchange.
pub fn freno_duro(arena: &quantum_arena::GlobalArena, pico: f64, capital: f64) -> FrenoDuro {
    use std::sync::atomic::Ordering::Relaxed;
    let w = arena.ganancia_en_stops.load(Relaxed);
    let l = arena.perdida_en_stops.load(Relaxed);
    let umbral_en_stops = if w > 0.0 && l > 0.0 {
        caida_alfa_freno_duro(w, l).filter(|k| k.is_finite() && *k > 0.0)
    } else {
        None
    };
    let caida_capital = if pico.is_finite() && pico > 0.0 && capital.is_finite() {
        (pico - capital) / pico
    } else {
        0.0
    };
    // F1-010: Respetar gen/config global_max_drawdown vivo acotando tope_capital
    let configured_dd = arena.config.global_max_drawdown.load(Relaxed);
    let tope_d3 = drawdown_de_falsacion(FRACCION_KELLY_MAXIMA, ALFA_FALSACION);
    let tope_capital = if configured_dd.is_finite() && configured_dd > 0.0 && configured_dd < 1.0 {
        configured_dd.min(tope_d3)
    } else {
        tope_d3
    };
    FrenoDuro {
        caida_en_stops: caida_en_stops(arena),
        umbral_en_stops,
        caida_capital,
        tope_capital,
    }
}

/// CAPITAL-UNICO V4J (punto 5, F8 de `qra_plano_del_freno.md`): un flujo
/// externo de la cuenta (depósito, retiro o traspaso) visto por su identidad
/// del exchange (`tranId`) y la revisión de la evidencia en que se vio. No
/// existe política de ajuste del pico por flujos: el pico y la protección se
/// conservan (la caída del patrimonio se sigue midiendo y puede disparar), y el
/// flujo queda registrado como INCIERTO para la atribución del disparo: nunca
/// se etiqueta como pérdida de mercado probada ni rearma el freno.
#[derive(Debug, Default, Clone)]
pub struct FlujosExternos {
    /// Cuenta del ambito (la fija el primer flujo; otro ambito se rechaza).
    ambito: Option<u64>,
    vistos: std::collections::HashMap<u64, (f64, u64)>,
    neto_incierto: f64,
    /// Flujos en otro activo: inciertos para el USDT, sin cifra en USDT.
    otros_activos: usize,
    /// V4L (L-F8): flujos sin identidad nativa (sin `tranId`): la atribucion queda incierta, sin
    /// cifra ni deduplicacion inferidas por coincidencia de tipo y hora.
    sin_identidad: usize,
}

/// Lo que pasó al registrar un flujo.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RegistroDeFlujo {
    Nuevo,
    /// Mismo `tranId` (replay o repetición de la consulta): no cuenta otra vez.
    Repetido,
    /// Importe no finito o cero: no es un flujo.
    Invalido,
    /// De otra cuenta (una respuesta retenida de la sesión anterior): no se mezcla.
    OtroAmbito,
    /// En otro activo: queda como incierto, sin sumarse como USDT.
    OtroActivo,
    /// V4L (L-F8): sin identidad nativa: atribucion incierta retenida, sin sumar ni deduplicar.
    SinIdentidad,
}

/// De dónde viene la caída medida, para la línea del disparo.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum OrigenDeLaCaida {
    /// Ningún flujo externo de salida sin acreditar en el ámbito: caída del patrimonio.
    Patrimonio,
    /// Hay salidas externas (o flujos en otro activo) sin política de ajuste: la caída puede
    /// ser un retiro, no una pérdida de mercado probada.
    FlujoIncierto,
}

impl FlujosExternos {
    /// `ambito`: la cuenta vigente del libro; `revision`: la era del puente en que se vio.
    pub fn registrar(&mut self, ambito: u64, id: u64, activo: &str, importe: f64, revision: u64) -> RegistroDeFlujo {
        if self.ambito.is_some_and(|a| a != ambito) {
            return RegistroDeFlujo::OtroAmbito;
        }
        if !(importe.is_finite() && importe != 0.0) {
            return RegistroDeFlujo::Invalido;
        }
        if self.vistos.contains_key(&id) {
            return RegistroDeFlujo::Repetido;
        }
        self.ambito = Some(ambito);
        self.vistos.insert(id, (importe, revision));
        if activo.eq_ignore_ascii_case("USDT") {
            self.neto_incierto += importe;
            RegistroDeFlujo::Nuevo
        } else {
            self.otros_activos += 1;
            RegistroDeFlujo::OtroActivo
        }
    }

    /// CAPITAL-UNICO V4L (L-F8): un flujo externo SIN identidad nativa (`tranId` ausente). No se
    /// infiere identidad por tipo y hora: no se suma ni se deduplica; solo deja la atribucion de la
    /// caida incierta en este ambito (sin ajustar pico ni freno).
    pub fn registrar_sin_identidad(&mut self, ambito: u64, importe: f64) -> RegistroDeFlujo {
        if self.ambito.is_some_and(|a| a != ambito) {
            return RegistroDeFlujo::OtroAmbito;
        }
        if !(importe.is_finite() && importe != 0.0) {
            return RegistroDeFlujo::Invalido;
        }
        self.ambito = Some(ambito);
        self.sin_identidad = self.sin_identidad.saturating_add(1);
        RegistroDeFlujo::SinIdentidad
    }

    /// V4L (L-F8): flujos sin identidad nativa vistos en el ambito.
    pub fn sin_identidad(&self) -> usize {
        self.sin_identidad
    }

    /// Suma en USDT de los flujos registrados (sin política de ajuste: todos inciertos).
    pub fn neto_incierto(&self) -> f64 {
        self.neto_incierto
    }

    pub fn cantidad(&self) -> usize {
        self.vistos.len()
    }

    /// Vínculo con la caída: una salida neta en USDT o un flujo en otro activo la puede
    /// explicar; un depósito no.
    pub fn origen_de_la_caida(&self) -> OrigenDeLaCaida {
        if self.neto_incierto < 0.0 || self.otros_activos > 0 || self.sin_identidad > 0 {
            OrigenDeLaCaida::FlujoIncierto
        } else {
            OrigenDeLaCaida::Patrimonio
        }
    }
}
/// V4J (F8): tipos del historial de ingresos de Binance USD-M que son flujos
/// externos de la cuenta (no resultado de operar).
pub fn es_flujo_externo(tipo: &str) -> bool {
    matches!(tipo, "TRANSFER" | "INTERNAL_TRANSFER" | "CROSS_COLLATERAL_TRANSFER" | "COIN_SWAP_DEPOSIT" | "COIN_SWAP_WITHDRAW")
}

#[cfg(test)]
mod tests {
    use super::*;

    /// QS-D3 — d* = 1 − α^(c/(2−c)): ½ Kelly a α = 0,05 ⇒ 0,6316; Kelly
    /// pleno se recorta a ½; políticas más conservadoras toleran MENOS caída;
    /// y en d* la probabilidad de la caída bajo el modelo es exactamente α.
    #[test]
    fn qs_d3_drawdown_de_falsacion_derivado() {
        let d = drawdown_de_falsacion(FRACCION_KELLY_MAXIMA, ALFA_FALSACION);
        assert!((d - (1.0 - 0.05f64.powf(1.0 / 3.0))).abs() < 1e-15);
        assert!((d - 0.6316).abs() < 1e-4, "{d}");
        assert_eq!(drawdown_de_falsacion(1.0, 0.05), d, "c > ½ se recorta a ½");
        let cuarto = drawdown_de_falsacion(0.25, 0.05);
        assert!(cuarto < d, "¼ Kelly tolera menos caída: {cuarto} vs {d}");
        assert!(drawdown_de_falsacion(0.5, 0.01) > d, "α menor ⇒ umbral mayor");
        // P(DD ≥ d*) = (1 − d*)^(2/c − 1) = α
        for c in [0.1, 0.25, 0.5] {
            let ds = drawdown_de_falsacion(c, 0.05);
            assert!(((1.0 - ds).powf(2.0 / c - 1.0) - 0.05).abs() < 1e-12);
        }
        for malo in [f64::NAN, f64::INFINITY, -1.0, 0.0] {
            assert_eq!(drawdown_de_falsacion(malo, 0.05), d);
            assert_eq!(drawdown_de_falsacion(0.5, malo), d);
        }
    }

    #[test]
    fn la_ewma_del_riesgo_arranca_en_la_primera_orden_y_no_salta_con_la_ultima() {
        let r1 = actualizar_riesgo_ewma(0.0, 0.05, 20.0);
        assert!((r1 - 0.05).abs() < 1e-12);
        let r2 = actualizar_riesgo_ewma(r1, 0.50, 20.0);
        assert!(r2 > 0.05 && r2 < 0.10, "{r2}");
    }
}
