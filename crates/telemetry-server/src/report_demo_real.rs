//! Segundo lector del informe por operación: lo que el motor escribe HOY.
//!
//! El lector original (`report_demo_run`) exige líneas `DEMO SIGNAL/ORDER/
//! FILL/CLOSE` con `trade_id`, que el host todavía NO emite. Este módulo arma
//! la tabla por operación con las líneas REALES de la corrida de humo de
//! Testnet (`logs/corrida_demo_20261112_094131.log`, 12-nov-2026):
//!
//! ```text
//! [OPEN TRACE] #1 dir=SHORT slot=0 ... lev=2.0x margin=$2.55 notional=$5.10 ... ts=1794494603627
//! [OCO GUARD] Protección para VETUSDT (TP: 0.0047, SL: 0.0052)
//! [FILL] VETUSDT SELL LIMIT cS_01a1... @ 0.00512 (acum 996/996 status Filled)
//! [USER-DATA] Posiciones vivas: VETUSDT -996
//! [BRACKET CLOSE] VETUSDT STOP qty 996.000000: entry 0.005120 → fill 0.005100 (...) | bruto 0.0199 | fees -0.0051 | neto 0.0148
//! [CLOSE TRACE] #1 reason=TP pnl_pct=... gross_pnl=$0.0199 exit=... entry=... ts=...
//! ```
//!
//! Reglas de honestidad (encargo del coordinador 12-nov 09:55):
//!  * Las líneas que salen por `println!` SIN `[ts=...]` NO reciben una hora
//!    inventada: se les presta la marca de la línea con hora MÁS CERCANA y se
//!    declara en la salida (`≈`, sección "HORAS PRESTADAS"). Ninguna latencia
//!    se calcula con horas prestadas.
//!  * Los totales del embudo se leen de las líneas `[EMBUDO] 60 s ...` del
//!    log. Desde CABLEADO HOST v7 el `FunnelCounter` del host se vuelca aparte
//!    (`outputs/funnel_demo.json`, para `demo_report --funnel`); este lector
//!    no lo lee. La sección `acumulado:` NO se suma (es acumulado, no ventana).
//!  * Este lector no lee el income (`outputs/income_demo.jsonl` lo exporta el
//!    host y lo lee `demo_report`): las columnas de dinero que no viajan en el
//!    log se reportan `N/D`; nunca se inventan.
//!  * Las líneas `[EMBUDO]` del log están truncadas por el escritor (~482 B):
//!    el parser tolera la cola cortada y sólo cuenta compuertas completas.
//!
//! Lo que haría falta en el host (DECLARADO, no implementado aquí: este módulo
//! no toca el motor) está en `HOST_WIRING_PENDING.md`, junto a
//! `DECLARACION_HOST` más abajo.

use std::collections::BTreeMap;

/// Origen de una marca de tiempo. `Prestada(n)` = se tomó de la línea `n`
/// (1-based) del recorte; NO es la hora del evento.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum OrigenHora {
    /// `[ts=...]` al principio de la propia línea (TG_LOG_TS=1).
    Propia,
    /// Campo `ts=...` embebido en la propia línea (reloj del núcleo).
    CampoInterno,
    /// Línea sin marca: se usa la marca de la línea `n` más cercana.
    Prestada(usize),
    /// No hay ninguna línea con hora en el recorte.
    SinMarca,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Marca {
    pub ms: u128,
    pub origen: OrigenHora,
}

impl Marca {
    fn fmt(&self) -> String {
        match self.origen {
            OrigenHora::Propia | OrigenHora::CampoInterno => self.ms.to_string(),
            OrigenHora::Prestada(n) => format!("{}≈L{}", self.ms, n),
            OrigenHora::SinMarca => "N/D".to_string(),
        }
    }
}

/// Una operación reconstruida del log real. `None` = no viajó en el log.
#[derive(Debug, Clone, PartialEq)]
pub struct OperacionReal {
    pub op_id: String,
    pub symbol: String,
    pub side: String,
    pub client_order_id: String,
    /// Hora de la orden: campo `ts=` del `[OPEN TRACE]` (reloj del núcleo).
    pub orden_ms: Option<Marca>,
    /// Hora del fill (`[FILL]`). Suele ser prestada: esa línea va por println.
    pub fill_ms: Option<Marca>,
    /// Hora del cierre (`[BRACKET CLOSE]` / `[CLOSE TRACE]`).
    pub cierre_ms: Option<Marca>,
    /// Sólo si AMBAS marcas son propias; si no, `None` (no se fabrica).
    pub orden_a_fill_ms: Option<u128>,
    pub entry_price: Option<f64>,
    pub exit_price: Option<f64>,
    pub qty: Option<f64>,
    pub margin: Option<f64>,
    pub notional: Option<f64>,
    pub leverage: Option<f64>,
    pub tp: Option<f64>,
    pub sl: Option<f64>,
    pub bruto: Option<f64>,
    pub fees: Option<f64>,
    pub neto: Option<f64>,
    pub close_reason: String,
    /// Quién ordenó el cierre: el log no lo dice -> "N/D".
    pub who_ordered: String,
    /// El host no emite `trade_id`: siempre "N/D" en esta corrida.
    pub trade_id: String,
    /// Líneas del recorte que formaron la operación (1-based).
    pub lineas: Vec<usize>,
    /// Avisos de procedencia / N/D de esta operación.
    pub notas: Vec<String>,
}

impl OperacionReal {
    fn nueva(op_id: String, linea: usize) -> Self {
        Self {
            op_id,
            symbol: "N/D".to_string(),
            side: "N/D".to_string(),
            client_order_id: "N/D".to_string(),
            orden_ms: None,
            fill_ms: None,
            cierre_ms: None,
            orden_a_fill_ms: None,
            entry_price: None,
            exit_price: None,
            qty: None,
            margin: None,
            notional: None,
            leverage: None,
            tp: None,
            sl: None,
            bruto: None,
            fees: None,
            neto: None,
            close_reason: "N/D".to_string(),
            who_ordered: "N/D".to_string(),
            trade_id: "N/D".to_string(),
            lineas: vec![linea],
            notas: Vec::new(),
        }
    }
}

#[derive(Debug, Clone, Default, PartialEq)]
pub struct InformeReal {
    pub ops: Vec<OperacionReal>,
    /// Totales por compuerta leídos de las líneas `[EMBUDO] 60 s`.
    pub embudo: BTreeMap<String, u64>,
    /// Número de ventanas `[EMBUDO] 60 s` contadas.
    pub ventanas_embudo: usize,
    /// Líneas `[EMBUDO]` truncadas por el escritor del log.
    pub lineas_embudo_truncadas: usize,
    /// "L48 [FILL] -> L49" etc. Se imprimen en la salida.
    pub horas_prestadas: Vec<String>,
    /// Líneas sin `[ts=]` para las que NO hubo ninguna marca cercana.
    pub sin_marca: usize,
}

/// `[ts=MS]` como PREFIJO de línea. Devuelve `None` si no está al principio.
fn ts_prefijo(linea: &str) -> Option<u128> {
    let s = linea.trim_start();
    let rest = s.strip_prefix("[ts=")?;
    let end = rest.find(']')?;
    rest[..end].parse::<u128>().ok()
}

/// Primer entero decimal después de `etiqueta` (busca la primera ocurrencia
/// útil: si tras la etiqueta no hay dígitos, sigue buscando).
fn num_tras(linea: &str, etiqueta: &str) -> Option<u64> {
    let mut desde = 0usize;
    while let Some(rel) = linea[desde..].find(etiqueta) {
        let pos = desde + rel + etiqueta.len();
        let cola = linea[pos..].trim_start();
        let digitos: String = cola.chars().take_while(|c| c.is_ascii_digit()).collect();
        if !digitos.is_empty() {
            return digitos.parse::<u64>().ok();
        }
        desde = pos;
    }
    None
}

/// Entero decimal inmediatamente ANTES de `etiqueta` (p.ej. `2807 bloqueados`).
fn num_antes(linea: &str, etiqueta: &str) -> Option<u64> {
    let pos = linea.find(etiqueta)?;
    let izq = linea[..pos].trim_end();
    let digitos: String = izq
        .chars()
        .rev()
        .take_while(|c| c.is_ascii_digit())
        .collect::<String>()
        .chars()
        .rev()
        .collect();
    if digitos.is_empty() {
        None
    } else {
        digitos.parse::<u64>().ok()
    }
}

/// Valor de un campo real del log, que puede venir como `clave=valor`,
/// `clave=$valor` o `clave: valor` / `clave valor` (los emisores usan ambas).
/// Se devuelve una sub-slice de la línea original, sin asignar.
fn campo<'a>(linea: &'a str, clave: &str) -> Option<&'a str> {
    let pat_eq = format!("{}=", clave);
    let mut pos = linea.find(&pat_eq).map(|p| p + pat_eq.len());
    if pos.is_none() {
        let pat_esp = format!("{} ", clave);
        pos = linea.find(&pat_esp).map(|p| p + pat_esp.len());
    }
    let cola = &linea[pos?..];
    let valor: String = cola.chars().take_while(|c| !c.is_whitespace()).collect();
    if valor.is_empty() {
        None
    } else {
        Some(&cola[..valor.len()])
    }
}

/// Índice de operación `#k` del `[OPEN TRACE]` / `[CLOSE TRACE]`.
fn indice_op(linea: &str) -> String {
    for tok in linea.split_whitespace() {
        if let Some(resto) = tok.strip_prefix('#') {
            let digitos: String = resto.chars().take_while(|c| c.is_ascii_digit()).collect();
            if !digitos.is_empty() {
                return digitos;
            }
        }
    }
    "N/D".to_string()
}

fn campo_f64(linea: &str, clave: &str) -> Option<f64> {
    let v = campo(linea, clave)?
        .trim_start_matches('$')
        .trim_end_matches(|c| matches!(c, ':' | ',' | ')' | '('));
    v.parse::<f64>().ok().filter(|f| f.is_finite())
}

fn campo_u128(linea: &str, clave: &str) -> Option<u128> {
    campo(linea, clave)?.parse::<u128>().ok()
}

/// Totales del embudo leídos de las líneas `[EMBUDO] 60 s ...` del log.
/// Devuelve `(ventanas, totales_por_compuerta, lineas_truncadas)`.
///
/// No usa `FunnelCounter` (no cableado en el host). La parte `· CUELLO:`
/// / `· acumulado:` / `· con posición abierta` NO se suma: el acumulado
/// duplicaría las ventanas y la población "con posición abierta" es otra.
pub fn funnel_totals_from_log(log: &str) -> (usize, BTreeMap<String, u64>, usize) {
    let mut totales: BTreeMap<String, u64> = BTreeMap::new();
    let mut ventanas = 0usize;
    let mut truncadas = 0usize;

    for linea in log.lines() {
        let Some(pos) = linea.find("[EMBUDO]") else {
            continue;
        };
        let cuerpo = &linea[pos + "[EMBUDO]".len()..];
        if !cuerpo.trim_start().starts_with("60 s") {
            continue;
        }
        ventanas += 1;

        // Cortamos antes del cuello / acumulado / población con posición.
        let fin = ["· CUELLO", "· acumulado", "· con posición abierta"]
            .iter()
            .filter_map(|m| cuerpo.find(*m))
            .min()
            .unwrap_or(cuerpo.len());
        let ventana = &cuerpo[..fin];
        // Si no apareció ningún terminador conocido, el escritor cortó la línea.
        if fin == cuerpo.len() && !ventana.ends_with(' ') && !ventana.ends_with('0') {
            truncadas += 1;
        }

        let pares: [(&str, Option<u64>); 12] = [
            ("ticks", num_tras(ventana, "ticks ")),
            ("ticks_bloqueados", num_antes(ventana, " bloqueados")),
            ("evaluaciones", num_tras(ventana, "evaluaciones ")),
            ("sin_conviccion", num_tras(ventana, "sin convicción ")),
            (
                "intenciones_libro_plano",
                num_tras(ventana, "intenciones con el libro plano "),
            ),
            ("sin_ranura", num_tras(ventana, "sin ranura ")),
            ("riesgo_admite", num_tras(ventana, "admite ")),
            ("riesgo_rechaza", num_tras(ventana, "rechaza ")),
            ("consejo", num_tras(ventana, "consejo ")),
            ("ML_GATE", num_tras(ventana, "ML-GATE ")),
            ("margen", num_tras(ventana, "margen ")),
            ("aperturas", num_tras(ventana, "aperturas ")),
        ];
        for (gate, valor) in pares {
            if let Some(v) = valor {
                *totales.entry(gate.to_string()).or_insert(0) += v;
            }
        }
        if let Some(v) = num_tras(ventana, "sin permiso de fase ") {
            *totales
                .entry("sin_permiso_de_fase".to_string())
                .or_insert(0) += v;
        }
        if let Some(v) = num_tras(ventana, "publicación descartada ") {
            *totales
                .entry("publicacion_descartada".to_string())
                .or_insert(0) += v;
        }
        // Escudos con nombre dinámico: "escudos: nombre N · nombre2 N2".
        if let Some(p) = ventana.find("escudos:") {
            let seg = &ventana[p + "escudos:".len()..];
            let seg = seg.split('→').next().unwrap_or(seg);
            for item in seg.split('·') {
                let item = item.trim();
                if item.is_empty() {
                    continue;
                }
                let mut partes = item.split_whitespace().collect::<Vec<_>>();
                let Some(ultimo) = partes.pop() else { continue };
                let Ok(n) = ultimo.parse::<u64>() else {
                    continue;
                };
                if partes.is_empty() {
                    continue;
                }
                let nombre = partes.join(" ");
                let clave = format!("escudo:{}", nombre);
                *totales.entry(clave).or_insert(0) += n;
            }
        }
        // Motivo del rechazo de riesgo: "(confianza 41)", "(sonda_ocupada 187)".
        if let Some(p) = ventana.find("rechaza ") {
            if let Some(abre) = ventana[p..].find('(') {
                let dentro = &ventana[p + abre + 1..];
                if let Some(cierra) = dentro.find(')') {
                    let texto = dentro[..cierra].trim();
                    let mut partes = texto.split_whitespace().collect::<Vec<_>>();
                    let Some(ultimo) = partes.pop() else { continue };
                    if let Ok(n) = ultimo.parse::<u64>() {
                        if !partes.is_empty() {
                            let clave = format!("riesgo_motivo:{}", partes.join(" "));
                            *totales.entry(clave).or_insert(0) += n;
                        }
                    }
                }
            }
        }
    }
    (ventanas, totales, truncadas)
}

/// Construye el informe por operación desde el log real de la corrida.
pub fn build_report_real(log: &str) -> InformeReal {
    let lineas: Vec<&str> = log.lines().collect();
    // Marcas de hora propias: (índice 0-based, ms).
    let marcas: Vec<(usize, u128)> = lineas
        .iter()
        .enumerate()
        .filter_map(|(i, l)| ts_prefijo(l).map(|ms| (i, ms)))
        .collect();

    let mut informe = InformeReal::default();
    let (ventanas, totales, truncadas) = funnel_totals_from_log(log);
    informe.ventanas_embudo = ventanas;
    informe.embudo = totales;
    informe.lineas_embudo_truncadas = truncadas;

    // Marca de hora de una línea: propia si la tiene; si no, la más cercana.
    let marca_de = |i: usize| -> Option<Marca> {
        if let Some(ms) = ts_prefijo(lineas[i]) {
            return Some(Marca {
                ms,
                origen: OrigenHora::Propia,
            });
        }
        let mut mejor: Option<(usize, usize, u128)> = None; // (distancia, linea, ms)
        for &(j, ms) in &marcas {
            let d = i.abs_diff(j);
            let poner = match mejor {
                None => true,
                Some((dm, jm, _)) => d < dm || (d == dm && j < jm),
            };
            if poner {
                mejor = Some((d, j, ms));
            }
        }
        match mejor {
            Some((_, j, ms)) => Some(Marca {
                ms,
                origen: OrigenHora::Prestada(j + 1),
            }),
            None => None,
        }
    };

    for (i, linea) in lineas.iter().enumerate() {
        let n = i + 1;
        let etiqueta = if linea.contains("[OPEN TRACE]") {
            "OPEN TRACE"
        } else if linea.contains("[OCO GUARD]") {
            "OCO GUARD"
        } else if linea.contains("[FILL]") {
            "FILL"
        } else if linea.contains("Posiciones vivas:") {
            "USER-DATA"
        } else if linea.contains("[BRACKET CLOSE]") {
            "BRACKET CLOSE"
        } else if linea.contains("[CLOSE TRACE]") {
            "CLOSE TRACE"
        } else if linea.contains("[FLATTEN]") {
            // Cierre real observado en el apagado con Ctrl+C de la corrida.
            "FLATTEN"
        } else {
            continue;
        };

        // Sólo las líneas de operación sin [ts=] generan aviso de hora prestada.
        let marca = marca_de(i);
        if ts_prefijo(linea).is_none() {
            match marca {
                Some(Marca {
                    origen: OrigenHora::Prestada(j),
                    ..
                }) => {
                    informe.horas_prestadas.push(format!(
                        "L{} [{}] usa la marca de L{} (sin [ts=] propia)",
                        n, etiqueta, j
                    ));
                }
                None => informe.sin_marca += 1,
                _ => {}
            }
        }

        // Unión: la última operación abierta (por símbolo cuando se conoce).
        // El log no trae trade_id, así que NO se puede unir por (symbol, trade_id).
        let idx_actual = informe.ops.len().checked_sub(1);

        match etiqueta {
            "OPEN TRACE" => {
                let op_id = indice_op(linea);
                let mut op = OperacionReal::nueva(format!("#{}", op_id), n);
                op.side = campo(linea, "dir").unwrap_or("N/D").to_string();
                op.margin = campo_f64(linea, "margin");
                op.notional = campo_f64(linea, "notional");
                op.leverage =
                    campo(linea, "lev").and_then(|v| v.trim_end_matches('x').parse::<f64>().ok());
                if let Some(ts) = campo_u128(linea, " ts") {
                    op.orden_ms = Some(Marca {
                        ms: ts,
                        origen: OrigenHora::CampoInterno,
                    });
                } else {
                    op.orden_ms = marca.map(|m| Marca {
                        origen: OrigenHora::Propia,
                        ..m
                    });
                }
                op.notas.push(
                    "sin DEMO SIGNAL: la hora de la orden viene del campo ts= del OPEN TRACE"
                        .to_string(),
                );
                informe.ops.push(op);
            }
            "OCO GUARD" => {
                let symbol = campo(linea, "para").unwrap_or("N/D").to_string();
                let tp =
                    campo(linea, "TP:").and_then(|v| v.trim_end_matches(',').parse::<f64>().ok());
                let sl = campo(linea, "SL:").and_then(|v| {
                    v.trim_end_matches(')')
                        .trim_end_matches(',')
                        .parse::<f64>()
                        .ok()
                });
                if let Some(k) = idx_actual {
                    let op = &mut informe.ops[k];
                    op.symbol = symbol;
                    op.tp = tp;
                    op.sl = sl;
                    op.lineas.push(n);
                }
            }
            "FILL" => {
                let partes: Vec<&str> = linea.split_whitespace().collect();
                let p = partes.iter().position(|t| *t == "[FILL]").unwrap_or(0);
                let symbol = partes.get(p + 1).copied().unwrap_or("N/D").to_string();
                let side = partes.get(p + 2).copied().unwrap_or("N/D").to_string();
                let client = partes
                    .iter()
                    .find(|t| t.starts_with("cL_") || t.starts_with("cS_"))
                    .copied()
                    .unwrap_or("N/D")
                    .to_string();
                let price = partes
                    .iter()
                    .position(|t| *t == "@")
                    .and_then(|q| partes.get(q + 1))
                    .and_then(|v| v.parse::<f64>().ok());
                let qty = campo(linea, "acum")
                    .and_then(|v| v.split('/').next())
                    .and_then(|v| v.parse::<f64>().ok());
                if let Some(k) = idx_actual {
                    let op = &mut informe.ops[k];
                    let es_cierre = op.entry_price.is_some() && op.fill_ms.is_some();
                    if es_cierre {
                        // Segundo FILL de la operación: es el fill de SALIDA
                        // (p. ej. BUY MARKET al aplanar un SHORT con Ctrl+C).
                        // No pisa la entrada, el lado ni el client id de apertura.
                        if op.exit_price.is_none() {
                            op.exit_price = price;
                        }
                        if op.cierre_ms.is_none() {
                            op.cierre_ms = marca;
                        }
                        if op.close_reason == "N/D" {
                            op.close_reason = "FLATTEN".to_string();
                        }
                        op.lineas.push(n);
                        op.notas.push(format!(
                            "L{} es el FILL de salida ({} {}); no se une a la entrada",
                            n, side, client
                        ));
                        continue;
                    }
                    if op.symbol == "N/D" {
                        op.symbol = symbol;
                    }
                    if op.side == "N/D" {
                        op.side = side;
                    }
                    op.client_order_id = client;
                    if op.entry_price.is_none() {
                        op.entry_price = price;
                    }
                    if op.qty.is_none() {
                        op.qty = qty;
                    }
                    op.fill_ms = marca;
                    op.lineas.push(n);
                    op.notas.push(
                        "fill por println sin [ts=]: hora prestada, latencia N/D".to_string(),
                    );
                }
            }
            "USER-DATA" => {
                // "Posiciones vivas: VETUSDT -996"
                let cola = linea.split("Posiciones vivas:").nth(1).unwrap_or("");
                let mut it = cola.split_whitespace();
                let symbol = it.next().unwrap_or("N/D").to_string();
                let qty = it
                    .next()
                    .and_then(|v| v.parse::<f64>().ok())
                    .map(|v: f64| v.abs());
                if let Some(k) = idx_actual {
                    let op = &mut informe.ops[k];
                    if op.symbol == "N/D" {
                        op.symbol = symbol;
                    }
                    if op.qty.is_none() {
                        op.qty = qty;
                    }
                    op.lineas.push(n);
                }
            }
            "BRACKET CLOSE" => {
                let cola = linea.split("[BRACKET CLOSE]").nth(1).unwrap_or("");
                let mut it = cola.split_whitespace();
                let symbol = it.next().unwrap_or("N/D").to_string();
                let trigger = it.next().unwrap_or("N/D").to_string();
                let qty = campo_f64(linea, "qty");
                let entry = campo_f64(linea, "entry");
                let exit = campo_f64(linea, "fill");
                let bruto = campo_f64(linea, "bruto");
                let fees = campo_f64(linea, "fees");
                let neto = campo_f64(linea, "neto");
                if let Some(k) = idx_actual {
                    let op = &mut informe.ops[k];
                    if op.symbol == "N/D" {
                        op.symbol = symbol;
                    }
                    if op.qty.is_none() {
                        op.qty = qty;
                    }
                    if op.entry_price.is_none() {
                        op.entry_price = entry;
                    }
                    op.exit_price = exit;
                    op.bruto = bruto;
                    op.fees = fees;
                    op.neto = neto;
                    op.close_reason = trigger;
                    op.cierre_ms = marca;
                    op.lineas.push(n);
                    op.notas.push(
                        "dinero del log del host, NO conciliado con income del exchange"
                            .to_string(),
                    );
                }
            }
            "FLATTEN" => {
                // "🧹 [FLATTEN] VETUSDT cerrada 996 @ ~0 (status FILLED)"
                let cola = linea.split("[FLATTEN]").nth(1).unwrap_or("");
                let mut it = cola.split_whitespace();
                let symbol = it.next().unwrap_or("N/D").to_string();
                if let Some(k) = idx_actual {
                    let op = &mut informe.ops[k];
                    if op.symbol == "N/D" {
                        op.symbol = symbol;
                    }
                    if op.close_reason == "N/D" {
                        op.close_reason = "FLATTEN".to_string();
                    }
                    if op.cierre_ms.is_none() {
                        op.cierre_ms = marca;
                    }
                    op.lineas.push(n);
                    op.notas.push(
                        "cierre por FLATTEN en el apagado; el precio viene del FILL de salida"
                            .to_string(),
                    );
                }
            }
            "CLOSE TRACE" => {
                let op_id = indice_op(linea);
                // Preferimos la operación con el mismo #k; si no, la última.
                let k = informe
                    .ops
                    .iter()
                    .rposition(|o| o.op_id == format!("#{}", op_id))
                    .or(idx_actual);
                if let Some(k) = k {
                    let op = &mut informe.ops[k];
                    op.close_reason = campo(linea, "reason").unwrap_or("N/D").to_string();
                    if op.bruto.is_none() {
                        // "gross_pnl=$0.0199"
                        op.bruto = campo_f64(linea, "gross_pnl");
                    }
                    // El CLOSE TRACE imprime exit/entry con 2 decimales; el
                    // BRACKET CLOSE trae 6. No se pisa una medida más precisa
                    // por una menos precisa.
                    if op.exit_price.is_none() {
                        op.exit_price = campo_f64(linea, "exit");
                    }
                    if op.entry_price.is_none() {
                        op.entry_price = campo_f64(linea, "entry");
                    }
                    if let Some(ts) = campo_u128(linea, " ts") {
                        op.cierre_ms = Some(Marca {
                            ms: ts,
                            origen: OrigenHora::CampoInterno,
                        });
                    } else {
                        op.cierre_ms = marca;
                    }
                    op.lineas.push(n);
                }
            }
            _ => {}
        }
    }

    // Latencia orden→fill SÓLO con horas propias; con prestada se queda en N/D.
    for op in &mut informe.ops {
        if let (Some(o), Some(f)) = (op.orden_ms, op.fill_ms) {
            let propia = matches!(o.origen, OrigenHora::Propia | OrigenHora::CampoInterno)
                && matches!(f.origen, OrigenHora::Propia | OrigenHora::CampoInterno);
            if propia && f.ms >= o.ms {
                op.orden_a_fill_ms = Some(f.ms - o.ms);
            }
        }
    }
    informe
}

fn fmt_opt(v: Option<f64>) -> String {
    match v {
        Some(x) if x.is_finite() => format!("{:.6}", x),
        _ => "N/D".to_string(),
    }
}
fn fmt_marca(v: Option<Marca>) -> String {
    v.map(|m| m.fmt()).unwrap_or_else(|| "N/D".to_string())
}

/// Tabla de texto del informe con el log real.
pub fn render_report_real(r: &InformeReal) -> String {
    let mut s = String::new();
    s.push_str("INFORME POR OPERACIÓN — líneas reales del motor (sin DEMO, sin income.jsonl)\n");
    s.push_str("op | símbolo | lado | trade_id | orden_ms | fill_ms | cierre_ms | ord→fill | entrada | salida | qty | margen | nocional | lev | TP | SL | bruto | fees | neto | cierre | quien | líneas\n");
    if r.ops.is_empty() {
        s.push_str("(sin operaciones en el recorte)\n");
    }
    for o in &r.ops {
        s.push_str(&format!(
            "{} | {} | {} | {} | {} | {} | {} | {} | {} | {} | {} | {} | {} | {} | {} | {} | {} | {} | {} | {} | {} | {}\n",
            o.op_id,
            o.symbol,
            o.side,
            o.trade_id,
            fmt_marca(o.orden_ms),
            fmt_marca(o.fill_ms),
            fmt_marca(o.cierre_ms),
            o.orden_a_fill_ms
                .map(|v| v.to_string())
                .unwrap_or_else(|| "N/D".to_string()),
            fmt_opt(o.entry_price),
            fmt_opt(o.exit_price),
            fmt_opt(o.qty),
            fmt_opt(o.margin),
            fmt_opt(o.notional),
            fmt_opt(o.leverage),
            fmt_opt(o.tp),
            fmt_opt(o.sl),
            fmt_opt(o.bruto),
            fmt_opt(o.fees),
            fmt_opt(o.neto),
            o.close_reason,
            o.who_ordered,
            o.lineas
                .iter()
                .map(|n| n.to_string())
                .collect::<Vec<_>>()
                .join(","),
        ));
        for nota in &o.notas {
            s.push_str(&format!("     nota: {}\n", nota));
        }
    }
    s.push_str(
        "\nHORAS PRESTADAS (no son la hora del evento; ≈L<n> = línea que prestó la marca)\n",
    );
    if r.horas_prestadas.is_empty() {
        s.push_str("  (ninguna)\n");
    }
    for h in &r.horas_prestadas {
        s.push_str(&format!("  {}\n", h));
    }
    s.push_str(&format!(
        "  líneas sin ninguna marca disponible: {}\n",
        r.sin_marca
    ));
    s.push_str(&render_funnel_real(r));
    s.push_str("\nAVISOS\n");
    s.push_str("  * trade_id: N/D — este lector no lee DEMO_TEL; los FILL con trade_id nativo los une demo_report (ver HOST_WIRING_PENDING.md).\n");
    s.push_str("  * comisión/funding/neto del exchange: N/D aquí — este lector no lee el income; el host lo exporta a outputs/income_demo.jsonl (pasada del FEE-BREAKER, sin cobertura acreditada) y lo concilia demo_report --income. El dinero mostrado viene SÓLO del log del host.\n");
    s.push_str("  * who_ordered: N/D — el log no dice quién ordenó el cierre.\n");
    s.push_str("  * embudo: leído de las líneas [EMBUDO] 60 s del log; el contador del host (IMMUNE_SYSTEM y FLATTEN) se vuelca al apagar a outputs/funnel_demo.json, que lee demo_report --funnel.\n");
    s
}

/// Render de los totales del embudo leídos del log.
pub fn render_funnel_real(r: &InformeReal) -> String {
    let mut s = String::new();
    s.push_str(&format!(
        "\nEMBUDO del log ({} ventanas [EMBUDO] 60 s; {} líneas truncadas por el escritor)\n",
        r.ventanas_embudo, r.lineas_embudo_truncadas
    ));
    if r.embudo.is_empty() {
        s.push_str("  (sin líneas [EMBUDO] 60 s)\n");
    }
    for (g, v) in &r.embudo {
        s.push_str(&format!("  {:52} {}\n", g, v));
    }
    s
}

/// Declaración de lo que haría falta en el HOST. NO se implementa aquí:
/// este módulo pertenece a `telemetry-server` y el encargo prohíbe tocar el motor.
pub const DECLARACION_HOST: &str = "\
PENDIENTE EN EL HOST (el punto 1 sigue declarado, NO implementado; 2 y 3 los implementó CABLEADO HOST v7; no se lee .env ni se llama al exchange):
 1. Líneas DEMO con trade_id: generar un trade_id estable por intención en el punto de
    creación de la orden (src/bin/god_engine.rs, junto a new_entry_client_id / EntryRequest),
    conservarlo en la reserva y en el reintento, y emitir con telemetry_log! (para que
    TG_LOG_TS=1 anteponga [ts=...]) CUATRO líneas con DEMO como PRIMER token tras el prefijo y
    sin emoji delante: 'DEMO SIGNAL|ORDER|FILL|CLOSE symbol=... side=... trade_id=... price=...
    reason=... by=...'. El fill lo puede emitir user_data_stream.rs:506 (ya tiene client id y
    acumulado); el cierre, god_engine.rs:4985 ([BRACKET CLOSE]) y god-engine-core/lib.rs:5462
    ([CLOSE TRACE]). Sin trade_id, demo_report no puede unir log con income.
 2. Income (CABLEADO HOST v7, implementado): el host exporta todas las filas de cada pasada
    del FEE-BREAKER a outputs/income_demo.jsonl (symbol, income_type, income, time_ms,
    trade_id, tran_id, asset), con rotación diaria y publicación atómica. No acredita
    cobertura del exchange. Este lector no lo lee; demo_report --income sí.
 3. FunnelCounter (CABLEADO HOST v7, implementado): IMMUNE_SYSTEM (cierres del aplanado,
    clave agregada) y FLATTEN (cierres agrupados del FEE-BREAKER) se vuelcan al apagar a
    outputs/funnel_demo.json para demo_report --funnel. record_latency_blocked sigue sin
    llamador. Este lector suma las líneas [EMBUDO] 60 s del log.
";

/// CLI: `demo_report_real --log <run.log>`. Sólo LEE el log.
pub fn cli_main_real() {
    let args: Vec<String> = std::env::args().collect();
    let mut log_path: Option<String> = None;
    let mut i = 1;
    while i < args.len() {
        if args[i] == "--log" {
            i += 1;
            log_path = args.get(i).cloned();
        }
        i += 1;
    }
    let Some(path) = log_path else {
        eprintln!("uso: demo_report_real --log <run.log>");
        std::process::exit(2);
    };
    let texto = match std::fs::read_to_string(&path) {
        Ok(t) => t,
        Err(e) => {
            eprintln!("no lee {}: {}", path, e);
            std::process::exit(2);
        }
    };
    let informe = build_report_real(&texto);
    println!("{}", render_report_real(&informe));
    println!("{}", DECLARACION_HOST);
}

#[cfg(test)]
mod tests_report_demo_real {
    use super::*;

    /// Recorte VERBATIM del log real de la corrida de humo (sólo lectura).
    const RECORTE: &str =
        include_str!("../tests/fixtures/corrida_demo_20261112_094131_recorte.log");

    #[test]
    fn real_el_banner_no_es_operacion_y_no_aborta() {
        // El banner real contiene "DEMO" pero no es una línea de evento.
        let informe = build_report_real(RECORTE);
        assert!(
            informe.ops.iter().all(|o| !o.op_id.contains("WARMUP")),
            "el banner no debe fabricar una operación"
        );
    }

    #[test]
    fn real_open_trace_oco_fill_y_user_data_forman_una_operacion() {
        let r = build_report_real(RECORTE);
        assert_eq!(r.ops.len(), 1, "el recorte tiene una sola apertura");
        let o = &r.ops[0];
        assert_eq!(o.op_id, "#1");
        assert_eq!(o.side, "SHORT");
        assert_eq!(o.symbol, "VETUSDT");
        assert!(
            (o.margin.unwrap_or(0.0) - 2.55).abs() < 1e-9,
            "margin=$2.55"
        );
        assert!(
            (o.notional.unwrap_or(0.0) - 5.10).abs() < 1e-9,
            "notional=$5.10"
        );
        assert!((o.leverage.unwrap_or(0.0) - 2.0).abs() < 1e-9, "lev=2.0x");
        assert!((o.tp.unwrap_or(0.0) - 0.0047).abs() < 1e-9, "TP: 0.0047");
        assert!((o.sl.unwrap_or(0.0) - 0.0052).abs() < 1e-9, "SL: 0.0052");
        assert!(
            (o.entry_price.unwrap_or(0.0) - 0.00512).abs() < 1e-9,
            "@ 0.00512"
        );
        assert!((o.qty.unwrap_or(0.0) - 996.0).abs() < 1e-9, "acum 996/996");
        assert!(
            o.client_order_id.starts_with("cS_"),
            "{}",
            o.client_order_id
        );
        // El ts= del OPEN TRACE es el reloj del núcleo, no una hora inventada.
        assert_eq!(o.orden_ms.map(|m| m.ms), Some(1794494603627));
    }

    #[test]
    fn real_fill_sin_ts_presta_la_marca_mas_cercana_y_lo_dice() {
        let r = build_report_real(RECORTE);
        let o = &r.ops[0];
        let f = o.fill_ms.expect("el fill debe tener marca");
        assert!(
            matches!(f.origen, OrigenHora::Prestada(_)),
            "el [FILL] va por println sin [ts=]: {:?}",
            f.origen
        );
        // La línea con hora más cercana al FILL (L48) es L49 ([USER-DATA]).
        assert_eq!(f.ms, 1794494606152, "marca de L49");
        assert_eq!(
            o.orden_a_fill_ms, None,
            "NO se fabrica latencia con hora prestada"
        );
        assert!(
            r.horas_prestadas
                .iter()
                .any(|h| h.contains("[FILL]") && h.contains("L49")),
            "la salida debe declarar la hora prestada: {:?}",
            r.horas_prestadas
        );
        let texto = render_report_real(&r);
        assert!(texto.contains("HORAS PRESTADAS"));
        assert!(texto.contains("≈L49"));
        assert!(texto.contains("N/D"), "trade_id/who_ordered en N/D");
    }

    #[test]
    fn real_totales_del_embudo_vienen_de_las_lineas_embudo() {
        let (ventanas, totales, _) = funnel_totals_from_log(RECORTE);
        assert_eq!(ventanas, 3, "hay 3 líneas [EMBUDO] 60 s en el recorte");
        // ticks: 7748 + 6546 + 8324
        assert_eq!(totales.get("ticks").copied(), Some(22618), "{:?}", totales);
        // bloqueados: 2807 + 1920 + 1963
        assert_eq!(totales.get("ticks_bloqueados").copied(), Some(6690));
        // evaluaciones: 4941 + 4626 + 6361
        assert_eq!(totales.get("evaluaciones").copied(), Some(15928));
        // consejo: 33 + 0 + 0
        assert_eq!(totales.get("consejo").copied(), Some(33));
        // ML-GATE: 8 + 0 + 0
        assert_eq!(totales.get("ML_GATE").copied(), Some(8));
        // aperturas: 0 + 1 + 0
        assert_eq!(totales.get("aperturas").copied(), Some(1));
        // escudo espectral: 55 + 31 + 82
        assert_eq!(
            totales
                .get("escudo:acondicionamiento espectral (X-016)")
                .copied(),
            Some(168)
        );
        assert_eq!(
            totales.get("riesgo_motivo:sonda_ocupada").copied(),
            Some(543)
        );
        let r = build_report_real(RECORTE);
        let texto = render_funnel_real(&r);
        assert!(texto.contains("EMBUDO del log (3 ventanas"));
        assert!(texto.contains("ticks"));
    }

    #[test]
    fn real_bracket_close_y_close_trace_rellenan_salida_y_dinero() {
        // Formato EXACTO de los emisores del host (god_engine.rs:4985 y
        // god-engine-core/lib.rs:5462); la corrida de humo aún no cerró.
        let log = "\
[ts=1794494603846] 🚀 [OPEN TRACE] #7 dir=SHORT slot=0 lev=2.0x margin=$2.55 notional=$5.10 ts=1794494603627
[ts=1794494606103] 🛡️ [OCO GUARD] Protección para VETUSDT (TP: 0.0047, SL: 0.0052)
[ts=1794494606152] 💧 [FILL] VETUSDT SELL LIMIT cS_abc @ 0.00512 (acum 996/996 status Filled)
[ts=1794494700000] 🎯 [BRACKET CLOSE] VETUSDT STOP qty 996.000000: entry 0.005120 → fill 0.005100 (trigger 0.005100, slip 3.9bps adversos) | bruto 0.0199 | fees -0.0051 | neto 0.0148
[ts=1794494700001] 🚪 [CLOSE TRACE] #7 reason=TP pnl_pct=0.3900% peak=0.5000% age=94s gross_pnl=$0.0199 exit=0.01 entry=0.01 trail_stop=0.00 mid=0.01 h=Continuous ts=1794494700000
";
        let r = build_report_real(log);
        assert_eq!(r.ops.len(), 1);
        let o = &r.ops[0];
        assert_eq!(o.symbol, "VETUSDT");
        assert_eq!(o.close_reason, "TP", "CLOSE TRACE manda sobre el trigger");
        assert!((o.exit_price.unwrap_or(0.0) - 0.0051).abs() < 1e-9);
        assert!((o.bruto.unwrap_or(0.0) - 0.0199).abs() < 1e-9);
        assert!((o.fees.unwrap_or(0.0) + 0.0051).abs() < 1e-9);
        assert!((o.neto.unwrap_or(0.0) - 0.0148).abs() < 1e-9);
        assert_eq!(o.trade_id, "N/D");
        assert_eq!(o.who_ordered, "N/D");
        assert!(o.notas.iter().any(|n| n.contains("NO conciliado")));
        // Ambas marcas son propias -> latencia SÍ se calcula.
        assert_eq!(o.orden_a_fill_ms, Some(1794494606152 - 1794494603627));
        assert!(r.horas_prestadas.is_empty());
    }

    #[test]
    fn real_sin_operaciones_el_informe_no_inventa_nada() {
        let r = build_report_real("[ts=1] 🔭 [EMBUDO] 60 s · ticks 0 · aperturas 0\n");
        assert!(r.ops.is_empty());
        let texto = render_report_real(&r);
        assert!(texto.contains("sin operaciones"));
        assert!(texto.contains("HORAS PRESTADAS"));
    }

    /// El apagado con Ctrl+C de la corrida real deja un `[FLATTEN]` y un
    /// SEGUNDO `[FILL]` (BUY MARKET) ~2900 líneas después. Es el fill de
    /// SALIDA: no debe pisar la entrada, el lado ni el client id de apertura.
    #[test]
    fn real_el_fill_del_apagado_es_salida_no_una_segunda_entrada() {
        let r = build_report_real(RECORTE);
        assert_eq!(r.ops.len(), 1, "una sola operación: #1 SHORT y su cierre");
        let o = &r.ops[0];
        assert_eq!(o.side, "SHORT", "el BUY de salida no cambia el lado");
        assert!(
            o.client_order_id.starts_with("cS_"),
            "el id de salida (01a1...) no pisa el de apertura: {}",
            o.client_order_id
        );
        assert!(
            (o.entry_price.unwrap_or(0.0) - 0.00512).abs() < 1e-9,
            "entrada sigue siendo el primer fill"
        );
        assert_eq!(o.close_reason, "FLATTEN");
        assert!(
            (o.exit_price.unwrap_or(0.0) - 0.005072).abs() < 1e-9,
            "salida = precio del BUY MARKET de cierre: {:?}",
            o.exit_price
        );
        assert_eq!(o.orden_a_fill_ms, None, "sin latencia fabricada");
        assert!(
            o.notas
                .iter()
                .any(|n| n.contains("FILL de salida") && !n.contains("entrada,")),
            "se declara el fill de salida: {:?}",
            o.notas
        );
        assert!(
            r.horas_prestadas
                .iter()
                .any(|h| h.contains("L87 [FILL]") && h.contains("L88")),
            "el fill de salida también declara su hora prestada: {:?}",
            r.horas_prestadas
        );
    }

    #[test]
    fn real_declaracion_host_no_implementa_motor() {
        assert!(DECLARACION_HOST.contains("DEMO SIGNAL"));
        // CABLEADOV7-P5: el productor de income ya existe; la declaración dice dónde.
        assert!(DECLARACION_HOST.contains("outputs/income_demo.jsonl"));
        assert!(DECLARACION_HOST.contains("NO implementado"), "el punto 1 sigue pendiente");
    }
}
