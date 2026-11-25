//! ENTRENADOR RUST POR INSTRUMENTO Y OBJETIVO — Gradient Boosting.
//!
//! POR QUÉ: los models/*_SCALP fueron entrenados por el pipeline Python
//! archivado (directriz: cero Python) y llevan semanas congelados — las
//! predicciones cuantizadas (0.337/0.476 constantes) congelan al SA.
//! `NanoForest::predict` es ESPACIO-LOGIT (init_score + Σárboles →
//! sigmoid): es gradient boosting, no promedio de forest. Este entrenador
//! conserva esa forma aditiva en la exportación con shrinkage e índices globales.
//!
//! ── PARIDAD ENTRENAMIENTO↔SERVICIO (D-753) ──────────────────────────────
//!
//! QUÉ ESTABA MAL: este entrenador construía su vector de rasgos alimentando
//! un `StatefulEngine` SUELTO con un SUBCONJUNTO de las actualizaciones que
//! el motor vivo recibe — `process_tick(mid, |bq−aq|, ts)`,
//! `update_trade_flow` y `update_ofi`, una vez por tick— mientras el servicio
//! extrae el MISMO vector de 48 dimensiones desde `GodEngineCore::
//! process_event`, que por cada tick del mercado ejecuta DOS eventos (depth y
//! trade) y en cada uno vuelve a llamar a `process_tick` con un volumen que
//! NO es el del trade (`clamp((bid_qty+ask_qty)·0,005; 0,01; 10)`, D-119),
//! más `update_ofi`, `update_macro_features`, el libro L2, el régimen y el
//! espectro temporal.
//!
//! POR QUÉ IMPORTABA: con dos alimentaciones distintas, las MISMAS 48 dims
//! tienen distribuciones distintas a cada lado. Las EMAs de precio avanzan al
//! doble de velocidad en servicio (dos llamadas por tick con el mismo
//! precio), el CVD de `ContinuousVPIN` se alimenta de nocionales de otra
//! escala, el retorno que entra en la FFT y en la entropía se intercala con
//! ceros, y el EMA del OFI decae a otro ritmo. Un bosque entrenado así sirve
//! un vector que NUNCA vio: SATURA. Medido ayer sobre ATOMUSDT, un modelo con
//! p10=0,092 / p50=0,142 / p90=0,218 en validación devolvía 3e-13 CONSTANTE
//! en servicio; el anterior, 0,99997 constante. Eso no es «poco edge»: es un
//! modelo evaluado fuera de su dominio.
//!
//! QUÉ GARANTIZA EL ARREGLO: el entrenador reproduce el tape POR EL MISMO
//! CAMINO que el motor — instancia `GodEngineCore` sobre un `GlobalArena` y
//! llama a `process_event` (depth + trade) exactamente como
//! `src/bin/audit_forensic_backtest.rs`— y toma el vector de
//! `core.feature_engines[0]`, que es el MISMO objeto del que el servicio lee
//! en `process_tick_dual`. Además la paridad se DEMUESTRA en cada muestra
//! (`primera_divergencia`): el vector que se guarda se compara dimensión a
//! dimensión contra los productores del propio motor y el binario ABORTA si
//! alguna difiere. Un entrenador que no puede demostrar su paridad no debe
//! escribir un modelo.
//!
//! ── INVENTARIO DE LO QUE EL TAPE NO PUEDE PRODUCIR ──────────────────────
//!
//! La paridad se DEMUESTRA, no se supone, y eso incluye enumerar los canales
//! que en vivo tienen productor y al reproducir un fichero de aggTrades no.
//! De las 48 dimensiones del contrato, éstas son TODAS, y ninguna queda
//! «distinta a cada lado»:
//!
//! · dims [4] [5] [10] — `obi_accel` (nivel, velocidad y aceleración del OBI)
//!   y dim [9] — `dark_alpha`, la severidad de liquidaciones, cuyo productor
//!   vivo es `liquidation_feed` y NO existe en un fichero de trades. Las
//!   cuatro figuran en `FEATURES_DEAD_IN_SERVE` y es
//!   `get_universal_features()` —el MOTOR, no este binario— quien las pone a
//!   0.0 EN AMBOS LADOS. Por eso el entrenador ya no las borra por su cuenta:
//!   sólo VERIFICA (`dim_muerta_no_nula`) y aborta el día en que el contrato
//!   deje de aplicarse.
//! · FUNDING: hasta F-A1 (Fable, 2026-10-17) entraba en
//!   `update_macro_features` y actualizaba una `fr_elasticity` que NINGUNA
//!   de las 48 dimensiones leía (arista muerta). F-A1 la retiró: el funding
//!   ya no toca las features del bosque en ningún lado (paridad trivial) y
//!   el per-símbolo vive en `CoinArena::funding_rate_next`, consumido por el
//!   risk-engine como coste de carry esperado de la orden.
//! · El resto del bloque `omni` que el tape no trae (fear&greed [14], oro
//!   [26], us10y…) alimenta `build_54d_tensor[34..54]`, que es el tensor de
//!   la RED, no el vector del BOSQUE. El bloque macro del bosque son cuatro
//!   dimensiones y sólo cuatro ranuras: `omni[21..25]`, que sí se cargan.
//! · dim [46] (DXY) cuando `data/macro/DXY.csv` falta: queda en 0 en TODO el
//!   entrenamiento, luego es una columna CONSTANTE y ningún árbol puede
//!   partir por ella. Un modelo que no lee esa dimensión es indiferente al
//!   valor que el vivo le sirva: no hay ruptura de paridad, hay una feature
//!   sin usar. El recorrido por dimensión la declara al final de la corrida.
//!
//! Etiquetas: triple-barrera (López de Prado) con los pisos institucionales
//! TP 0.36% / SL 0.18%; los neutros se DESCARTAN (promediarlos hacia 0.5 es
//! una causa documentada de forests '~0.5'). NO se tocan en esta revisión.
//!
//! VALIDACIÓN: selección temporal 80/20 (o --val-in), con purga de ventanas
//! de etiqueta. Early stopping usa sólo selección. --promote exige --test-in
//! posterior a TODA la evidencia de ajuste/selección; su gate evalúa el
//! artefacto congelado. Sin test sólo se permite candidato de investigación.
//! QS-2: cada modelo se escribe con `{KEY}.manifest` (linaje: hashes de los
//! tapes, cobertura, argumentos, commit y evidencia del gate) y cada test que
//! juzga un artefacto queda en el libro `--holdout-ledger`; promover con un
//! test que ya juzgó OTRO artefacto del mismo símbolo y objetivo exige
//! `--test-reuse "<motivo>"` (sale con código 3 si no). No prueba PnL.
//!
//! ── UNIDADES DE LAS ETIQUETAS DE REGRESIÓN (revisión D-731/D-732) ───────
//!
//! `--label vol` YA NO es «RMS de retornos por tick». Era
//! `sqrt(Σ r² / n_ticks) · 100`, es decir la volatilidad DIVIDIDA por la
//! intensidad de ticks: un tramo con muchos trades pequeños salía MENOS
//! volátil que uno tranquilo con pocos saltos, justo al revés de lo que
//! mide el mercado. Ahora la etiqueta es la VOLATILIDAD REALIZADA DEL
//! HORIZONTE, `sqrt(Σ r²) · 100` sobre (t, t+τ] — es decir σ(τ) en % de
//! precio, la misma magnitud que el VOL-BRAKE compara contra la base del
//! modelo. CONSECUENCIA: los `models/{SYM}_VOL.json` entrenados antes de
//! este cambio quedan OBSOLETOS — su `init_score` (la media del label) y
//! sus hojas están en la escala vieja (÷ sqrt(n_ticks)); hay que
//! reentrenarlos. El serving NO cambia: el freno usa el COCIENTE
//! pronóstico/base y ese cociente es invariante de escala, pero mezclar un
//! modelo viejo con uno nuevo sí compara manzanas con peras.
//!
//! `--label volu` YA NO es «profundidad media del libro». Sumaba
//! `(bid_qty + ask_qty)` y lo dividía por el número de ticks, pero en el
//! tape de aggTrades ese par NO es un libro: `binance_vision_sync` lo
//! fabrica como `qty + base` en el lado del agresor y `base` en el otro,
//! con `base = max(0,25·qty; 0,1)`. La suma valía ≈1,5·qty con un suelo
//! artificial y la media por tick volvía a dividir por la intensidad: la
//! etiqueta era «tamaño medio de trade inflado un 50%», no volumen. Ahora
//! es el NOCIONAL NEGOCIADO del horizonte, `Σ |qty| · precio` sobre
//! (t, t+τ], con `|qty| = |bid_qty − ask_qty|` (la diferencia CANCELA la
//! base fabricada y recupera el `qty` exacto del aggTrade). Si el tape no
//! procede de aggTrades el binario aborta en vez de etiquetar ruido.
//!
//! ── BASELINE DEL GATE EN REGRESIÓN ──────────────────────────────────────
//!
//! El baseline honesto de una magnitud persistente NO es su media: es la
//! PERSISTENCIA (σ futura ≈ σ reciente, volumen futuro ≈ volumen reciente).
//! Contra la media, un R² de 0,1 no demuestra nada porque la persistencia
//! ya explica bastante más. El gate exige ahora batir el MSE de la
//! persistencia medida sobre la MISMA partición de validación; el R² contra
//! la media sigue imprimiéndose, etiquetado como lo que es (informativo).
//!
//! Uso: train_forest BTCUSDT [--in data/BTCUSDT_SEP26.bin] [--max-samples 400000]
//!      [--horizon-ms 300000] [--trees 300] [--lr 0.1] [--depth 5]
//!      [--early-stop-in early.bin] [--val-in selection.bin] [--test-in later-test.bin] [--promote]
//!      [--calentamiento-ms 43200000] [--cache columnas] [--cache-dir cache/columnas]

use god_engine_core::ml_inference::{macro_ml_features, NanoForest, NanoForestData};
use god_engine_core::stateful_engine::{StatefulEngine, FEATURES_DEAD_IN_SERVE};
use god_engine_core::GodEngineCore;
use quantum_arena::symbol_registry::SymbolSpec;
use rand::rngs::StdRng;
use rand::{RngExt, SeedableRng};
use quantum_engine::model_manifest;
use std::sync::atomic::Ordering;

/// Dimensión del vector ML: el contrato del binario de inferencia. Un modelo
/// que parta por una dim ≥ este valor se rechaza al cargar, así que el
/// entrenador NO puede construir un vector de otra anchura.
const DIM_VECTOR: usize = NanoForest::ML_VECTOR_DIM;

#[derive(Debug, Clone, Copy, Default)]
#[repr(C)]
struct BinTick {
    ts: u64,
    bid: f64,
    ask: f64,
    bq: f64,
    aq: f64,
}

fn valid_tick(t: &BinTick) -> bool {
    t.ts > 0
        && t.bid.is_finite()
        && t.ask.is_finite()
        && t.bid > 0.0
        && t.ask >= t.bid
        && t.bq.is_finite()
        && t.aq.is_finite()
        && t.bq >= 0.0
        && t.aq >= 0.0
        && (t.bq + t.aq).is_finite()
}

fn validate_tick_order(raw: &[BinTick]) -> Result<(), String> {
    if raw.is_empty() || raw.iter().any(|t| t.ts == 0) || raw.windows(2).any(|w| w[1].ts < w[0].ts)
    {
        return Err("tick timestamps must be positive and nondecreasing; no silent sorting".into());
    }
    Ok(())
}

// ── Magnitudes físicas del tape y ventanas CAUSALES ──────────────────────
//
// Todo lo que sigue es aritmética pura sobre ticks: se aísla aquí para que
// las pruebas puedan fijar la geometría de las ventanas (rasgos con
// información ≤ t, etiqueta en (t, t+τ], persistencia en (t−τ, t]) sin
// levantar el motor ni leer un fichero.

/// Muestras construidas desde un tape: rasgos (información ≤ t), etiqueta
/// (ventana (t, t+τ]), baseline de PERSISTENCIA (misma magnitud en (t−τ, t],
/// información ≤ t) y la marca temporal t de cada muestra, que la purga del
/// solape train/val necesita.
struct Samples {
    feats: Vec<Vec<f32>>,
    labels: Vec<f64>,
    persist: Vec<f64>,
    ts: Vec<u64>,
}

/// Particion de muestras en dos lados (temporal 80/20 o cruzada por archivo):
/// features y etiquetas de train y validacion, persistencia de validacion y
/// las marcas temporales de cada lado.
type SplitSamples = (
    Vec<Vec<f32>>,
    Vec<Vec<f32>>,
    Vec<f64>,
    Vec<f64>,
    Vec<f64>,
    Vec<u64>,
    Vec<u64>,
);

/// Acumuladores de UNA ventana temporal de ticks.
#[derive(Default, Clone, Copy, PartialEq, Debug)]
struct WindowStats {
    /// Σ r² de los retornos de mid dentro de la ventana — varianza realizada.
    sum_sq: f64,
    /// Σ |qty| · precio — nocional negociado dentro de la ventana.
    notional: f64,
    /// Nº de retornos observados (0 ⇒ ventana sin información de precio).
    n_ret: usize,
}

/// Cantidad NEGOCIADA de un tick del tape de aggTrades.
///
/// POR QUÉ: `binance_vision_sync` no tiene libro, así que fabrica el par
/// (bid_qty, ask_qty) a partir del `qty` del aggTrade: un lado recibe
/// `qty + base` y el otro `base`. La SUMA arrastra `2·base` (≈ ×1,5 el qty,
/// con un suelo absoluto), pero la DIFERENCIA cancela la base y devuelve el
/// `qty` exacto, sea cual sea la fórmula de `base`. Ésta es la única
/// cantidad negociada realmente medible en este tape.
fn agg_trade_qty(bq: f64, aq: f64) -> f64 {
    (bq - aq).abs()
}

/// ¿El par (bid_qty, ask_qty) procede de la fabricación de aggTrades?
///
/// Espejo EXACTO del contrato de `src/bin/binance_vision_sync.rs`:
/// `base = max(0,25·qty; 0,1)`, lado agresor = `qty + base`. De ahí la
/// identidad `min = max(0,25·(max − min); 0,1)`, que se cumple tick a tick
/// en ese tape y NO se cumple en el tape sintético de klines
/// (`parquet_to_bin`, donde ambos lados son fracciones del volumen de vela).
/// Si el productor cambiara su fórmula esta comprobación debe romperse
/// RUIDOSAMENTE: etiquetar «nocional» sobre un par que no es de aggTrades
/// sería inventar la magnitud.
fn is_aggtrades_pair(bq: f64, aq: f64) -> bool {
    if !bq.is_finite() || !aq.is_finite() {
        return false;
    }
    let (lo, hi) = if bq <= aq { (bq, aq) } else { (aq, bq) };
    let qty = hi - lo;
    if !(qty.is_finite() && qty > 0.0) {
        return false;
    }
    let expected = (qty * 0.25).max(0.1);
    // Tolerancia NUMÉRICA, no de decisión: 1e-9 relativo queda siete órdenes
    // por encima del épsilon de f64 (2,2e-16) que acumula el round-trip
    // cálculo→fichero→mmap, y varios órdenes por debajo de cualquier
    // desviación estructural (el tape de klines da directamente qty = 0).
    (lo - expected).abs() <= expected * 1e-9
}

/// σ REALIZADA de la ventana, en % de precio: `sqrt(Σ r²) · 100`.
///
/// No se divide por el número de ticks: eso convertiría la volatilidad en
/// «volatilidad por tick» y haría que un tramo con mucha actividad pareciera
/// más tranquilo. Con τ fijo (el horizonte del muestreo) esto ES σ(τ), la
/// magnitud que compara el VOL-BRAKE.
fn realized_vol_pct(sum_sq: f64) -> f64 {
    sum_sq.max(0.0).sqrt() * 100.0
}

/// Recorre `ticks` acumulando retornos² y nocional, anclando el primer
/// retorno en `anchor_mid` (el mid del borde izquierdo de la ventana).
/// `anchor_mid ≤ 0` ⇒ el primer tick sólo fija el ancla y no genera retorno.
fn accumulate_window(ticks: &[BinTick], anchor_mid: f64) -> WindowStats {
    let mut st = WindowStats::default();
    let mut prev = anchor_mid;
    for tk in ticks {
        if tk.bid <= 0.0 || tk.ask <= 0.0 || tk.bid > tk.ask {
            continue;
        }
        let m = (tk.bid + tk.ask) / 2.0;
        if prev > 0.0 {
            let r = (m - prev) / prev;
            st.sum_sq += r * r;
            st.n_ret += 1;
        }
        st.notional += agg_trade_qty(tk.bq, tk.aq) * m;
        prev = m;
    }
    st
}

/// Fin EXCLUSIVO de la ventana futura: primer índice > `i` con `ts > deadline`.
/// La ventana resultante `i+1 .. fin` es estrictamente (t, t+τ]: nunca incluye
/// el propio tick t ni nada posterior al vencimiento.
fn forward_window_end(raw: &[BinTick], i: usize, deadline: u64) -> usize {
    let mut hi = i + 1;
    while hi < raw.len() && raw[hi].ts <= deadline {
        hi += 1;
    }
    hi
}

/// Inicio de la ventana pasada (t−τ, t] y mid de anclaje (último mid válido
/// con `ts ≤ start`). La ventana `inicio ..= i` sólo contiene información
/// disponible en t — es el baseline de persistencia, no puede mirar futuro.
/// Ancla 0.0 ⇒ el tape no cubre τ hacia atrás desde `i`.
fn trailing_window_start(raw: &[BinTick], i: usize, start: u64) -> (usize, f64) {
    let mut lo = i;
    while lo > 0 && raw[lo - 1].ts > start {
        lo -= 1;
    }
    let mut a = lo;
    let mut anchor = 0.0;
    while a > 0 {
        a -= 1;
        let tk = &raw[a];
        if tk.bid > 0.0 && tk.ask > 0.0 && tk.bid <= tk.ask {
            anchor = (tk.bid + tk.ask) / 2.0;
            break;
        }
    }
    (lo, anchor)
}

/// PURGA (López de Prado) del solape entre train y validación.
///
/// POR QUÉ: con τ = 300 s y stride = 50 s las muestras se solapan 6×, así que
/// la etiqueta de las últimas muestras de train se resuelve DENTRO del tramo
/// de validación — el modelo «ya vio» ese futuro y la métrica de validación
/// sale optimista. Devuelve el fin (exclusivo) del train: la última muestra
/// cuya ventana de etiqueta termina ESTRICTAMENTE antes de la primera muestra
/// de validación. El número de muestras descartadas no es una constante: lo
/// fija el propio horizonte τ contra la densidad real del muestreo.
///
/// XLIV-13b (revisión de Codex en el PR #10): la frontera es CERRADA, la
/// misma convención que `purge_training` (`end >= inicio`) y
/// `require_later_holdout` (`inicio <= fin`), y la de la implementación de
/// referencia de López de Prado. La etiqueta del train que acaba justo en
/// `t_val` incluye el tick de `t_val`, que la validación ya conoce como
/// rasgo (información ≤ t): la igualdad cuenta como solape.
fn purge_end(ts: &[u64], split: usize, horizon_ms: u64) -> usize {
    if split == 0 || split >= ts.len() {
        return split.min(ts.len());
    }
    let first_val_ts = ts[split];
    ts[..split]
        .iter()
        .position(|&t| t.saturating_add(horizon_ms) >= first_val_ts)
        .unwrap_or(split)
}

/// Veredicto del gate de REGRESIÓN. Devuelve
/// `(R² contra la media, skill contra la persistencia, pasa)`.
///
/// POR QUÉ: la media constante es un baseline de juguete para magnitudes
/// persistentes (σ, volumen). Un R² de 0,1 contra la media puede convivir con
/// un modelo PEOR que repetir el valor reciente. El gate exige batir la
/// persistencia por `margin` (fracción del MSE de la persistencia); el R²
/// contra la media queda como diagnóstico. Persistencia degenerada (MSE nulo
/// o no finito) ⇒ no hay baseline comparable ⇒ NO pasa.
fn regression_gate(mse_model: f64, mse_mean: f64, mse_persist: f64, margin: f64) -> (f64, f64, bool) {
    let r2_mean = if mse_mean.is_finite() && mse_mean > 0.0 {
        (mse_mean - mse_model) / mse_mean
    } else {
        0.0
    };
    if !(mse_persist.is_finite() && mse_persist > 0.0) || !mse_model.is_finite() {
        return (r2_mean, 0.0, false);
    }
    let skill = (mse_persist - mse_model) / mse_persist;
    // LXXXIX M1: un margen inválido nunca puede relajar el gate (defensa en
    // profundidad — el contrato lo valida al inicio, esto cubre llamadas futuras).
    let pasa = margin.is_finite() && margin >= 0.0 && skill >= margin;
    (r2_mean, skill, pasa)
}

// HAR-RV secundario y QLIKE: diagnosticos de volatilidad, sin intervenir el gate.
// Los intervalos se anclan en tiempo fisico, independientemente del stride.
#[derive(Default)]
struct HarHistory {
    // Prefijo causal de retornos al cuadrado, agrupado por timestamp.
    points: Vec<(u64, f64)>,
    last_mid: Option<f64>,
    ignored_overlap: usize,
    invalid_prices: usize,
    seams: Vec<(u64, u64)>,
}

impl HarHistory {
    fn append_ticks(&mut self, raw: &[BinTick]) -> Result<(), String> {
        validate_tick_order(raw)?;
        let cutoff = self.points.last().map(|p| p.0);
        let mut seam_pending = cutoff.is_some();
        let mut sum = self.points.last().map_or(0.0, |p| p.1);
        for tk in raw {
            // En solapes, el tape ya incorporado es la fuente del pasado.
            if cutoff.is_some_and(|t| tk.ts <= t) {
                self.ignored_overlap += 1;
                continue;
            }
            if !(tk.bid.is_finite() && tk.ask.is_finite() && tk.bid > 0.0 && tk.ask >= tk.bid) {
                self.invalid_prices += 1;
                continue;
            }
            let mid = (tk.bid + tk.ask) / 2.0;
            if !mid.is_finite() {
                return Err("HAR: mid no finito".into());
            }
            if seam_pending {
                self.seams.push((cutoff.unwrap(), tk.ts));
                seam_pending = false;
            }
            if let Some(prev) = self.last_mid {
                let r = (mid - prev) / prev;
                sum += r * r;
                if !sum.is_finite() {
                    return Err("HAR: prefijo de varianza no finito".into());
                }
            }
            self.last_mid = Some(mid);
            if self.points.last().is_some_and(|p| p.0 == tk.ts) {
                self.points.last_mut().unwrap().1 = sum;
            } else {
                self.points.push((tk.ts, sum));
            }
        }
        if self.points.is_empty() {
            return Err("HAR: tape sin precios utilizables".into());
        }
        Ok(())
    }

    fn append_through(&mut self, raw: &[BinTick], end: u64) -> Result<(), String> {
        // El tape de build ya tiene cronologia validada. Ningun precio posterior
        // al limite participa en prefijos, validaciones numericas ni calibracion.
        let count = raw.partition_point(|t| t.ts <= end);
        self.append_ticks(&raw[..count])
    }

    fn append_admitted(
        &mut self,
        cinta: &quantum_engine::tgb_generation::CintaAdmitida,
        end: u64,
    ) -> Result<(), String> {
        // TGBG-5: HAR consume la misma instantanea admitida que build.
        // El lector comun sirve tambien .bin.tgb; no se reabre la ruta logica.
        let raw = ticks_de_cinta(cinta.bytes())?;
        self.append_through(&raw, end)
    }

    fn rv(&self, start: u64, end: u64) -> Option<f64> {
        if start >= end || start < self.points.first()?.0 || end > self.points.last()?.0 {
            return None;
        }
        let lo = self.points.partition_point(|p| p.0 <= start);
        let hi = self.points.partition_point(|p| p.0 <= end);
        if lo == 0 || hi <= lo {
            return None; // Ningun retorno observado en (start,end].
        }
        let variance = self.points[hi - 1].1 - self.points[lo - 1].1;
        (variance.is_finite() && variance >= 0.0).then(|| realized_vol_pct(variance))
    }

    fn features(&self, t: u64, tau: u64, windows: &[usize]) -> Option<Vec<f64>> {
        if tau == 0 || windows.is_empty() || windows.windows(2).any(|w| w[0] >= w[1]) {
            return None;
        }
        let mut sums = Vec::with_capacity(windows.len());
        let mut sum = 0.0;
        let mut end = t;
        let mut next = 0;
        for k in 1..=*windows.last()? {
            let start = end.checked_sub(tau)?;
            sum += self.rv(start, end)?;
            if windows[next] == k {
                sums.push(sum / k as f64);
                next += 1;
            }
            end = start;
        }
        (sums.len() == windows.len() && sums.iter().all(|x| x.is_finite())).then_some(sums)
    }
}

// Se conserva el tramo contiguo mas largo: nunca se comprime una laguna temporal.
fn har_grid(history: &HarHistory, first: u64, last: u64, tau: u64) -> (Vec<f64>, usize, u64) {
    let (mut best, mut run) = (Vec::new(), Vec::new());
    let (mut missing, mut best_end) = (0, first);
    if tau == 0 {
        return (best, missing, best_end);
    }
    let mut end = first;
    loop {
        if let Some(x) = end
            .checked_sub(tau)
            .and_then(|start| history.rv(start, end))
        {
            run.push(x);
        } else {
            if run.len() > best.len() {
                best = std::mem::take(&mut run);
                best_end = end - tau;
            }
            run.clear();
            missing += 1;
        }
        match end.checked_add(tau) {
            Some(next) if next <= last => end = next,
            _ => break,
        }
    }
    if run.len() > best.len() {
        best = run;
        best_end = end;
    }
    (best, missing, best_end)
}

fn har_windows(grid: &[f64]) -> (Vec<usize>, Vec<String>) {
    let mut windows = vec![1]; // Unidad fisica del problema: un horizonte tau.
    let mut reasons = Vec::new();
    if grid.is_empty() || grid.iter().any(|x| !x.is_finite()) {
        reasons.push("malla_train_ausente_o_no_finita".into());
        return (windows, reasons);
    }
    let n = grid.len();
    let mean = grid.iter().sum::<f64>() / n as f64;
    let centered: Vec<_> = grid.iter().map(|x| x - mean).collect();
    let denom = centered.iter().map(|x| x * x).sum::<f64>();
    if !(denom.is_finite() && denom > 0.0) {
        reasons.push("malla_train_constante; se conserva solo tau".into());
        return (windows, reasons);
    }
    let rho = |lag: usize| {
        centered[lag..]
            .iter()
            .zip(&centered[..n - lag])
            .map(|(a, b)| a * b)
            .sum::<f64>()
            / denom
    };
    // Limite de viabilidad exigido por la especificacion: mitad de la malla.
    let limit = n / 2;
    if let Some(m) = (2..=limit).find(|&k| rho(k).abs() <= 1.0 / (n as f64).sqrt()) {
        windows.push(m);
        if let Some(l) = (m + 1..=limit).find(|&k| rho(k).abs() <= 1.0 / n as f64) {
            windows.push(l);
        } else {
            reasons.push("sin_cruce_l_en_lags_viables; rejilla_reducida".into());
        }
    } else {
        reasons.push("sin_cruce_m_en_lags_viables; rejilla_reducida".into());
    }
    (windows, reasons)
}

#[derive(Debug, Clone, PartialEq)]
struct HarModel {
    windows: Vec<usize>,
    beta: Vec<f64>, // Intercepto seguido de pendientes, en unidades originales.
    ridge_lambda: f64,
}

impl HarModel {
    fn predict(&self, history: &HarHistory, t: u64, tau: u64) -> Option<f64> {
        let x = history.features(t, tau, &self.windows)?;
        let prediction = self.beta[0]
            + self.beta[1..]
                .iter()
                .zip(x)
                .map(|(b, x)| b * x)
                .sum::<f64>();
        prediction.is_finite().then_some(prediction)
    }
}

// OLS centrado y escalado con Ridge numerico. No hay lambda elegida por validacion
// ni un literal de regularizacion: lambda = sqrt(EPSILON) * traza(X'X) / p.
// Para columnas constantes, la escala de respaldo es el numero de filas de train.
fn har_ridge(x: &[Vec<f64>], y: &[f64], windows: Vec<usize>) -> Option<HarModel> {
    let p = windows.len();
    if p == 0
        || x.len() != y.len()
        || y.len() < p + 1
        || y.iter().any(|v| !v.is_finite())
        || x.iter()
            .any(|r| r.len() != p || r.iter().any(|v| !v.is_finite()))
    {
        return None;
    }
    let n = y.len() as f64;
    let mean_y = y.iter().sum::<f64>() / n;
    let means: Vec<_> = (0..p)
        .map(|j| x.iter().map(|r| r[j]).sum::<f64>() / n)
        .collect();
    let scales: Vec<_> = (0..p)
        .map(|j| {
            let sd = (x.iter().map(|r| (r[j] - means[j]).powi(2)).sum::<f64>() / n).sqrt();
            if sd == 0.0 {
                1.0
            } else {
                sd
            }
        })
        .collect();
    let mut a = vec![vec![0.0; p + 1]; p];
    for (row, target) in x.iter().zip(y) {
        let z: Vec<_> = (0..p).map(|j| (row[j] - means[j]) / scales[j]).collect();
        for j in 0..p {
            for k in 0..p {
                a[j][k] += z[j] * z[k];
            }
            a[j][p] += z[j] * (target - mean_y);
        }
    }
    let trace = (0..p).map(|j| a[j][j]).sum::<f64>();
    let scale = if trace > 0.0 { trace / p as f64 } else { n };
    let ridge_lambda = f64::EPSILON.sqrt() * scale;
    if !(ridge_lambda.is_finite() && ridge_lambda > 0.0) {
        return None;
    }
    for (j, row) in a.iter_mut().enumerate() {
        row[j] += ridge_lambda;
    }
    // Eliminacion con pivoteo parcial; el intercepto no se penaliza.
    for col in 0..p {
        let pivot = (col..p).max_by(|&i, &j| a[i][col].abs().total_cmp(&a[j][col].abs()))?;
        a.swap(col, pivot);
        let divisor = a[col][col];
        if !divisor.is_finite() || divisor == 0.0 {
            return None;
        }
        for value in &mut a[col][col..=p] {
            *value /= divisor;
        }
        let pivot_tail: Vec<f64> = a[col][col..=p].to_vec();
        for (j, row) in a.iter_mut().enumerate() {
            if j != col {
                let factor = row[col];
                for (offset, &pivot_value) in pivot_tail.iter().enumerate() {
                    row[col + offset] -= factor * pivot_value;
                }
            }
        }
    }
    let slopes: Vec<_> = (0..p).map(|j| a[j][p] / scales[j]).collect();
    let intercept = mean_y - slopes.iter().zip(means).map(|(b, m)| b * m).sum::<f64>();
    let mut beta = vec![intercept];
    beta.extend(slopes);
    beta.iter().all(|b| b.is_finite()).then_some(HarModel {
        windows,
        beta,
        ridge_lambda,
    })
}

fn har_variance(sigma_pct: f64) -> Option<f64> {
    let v = (sigma_pct / 100.0).powi(2); // Conversion de porcentaje a varianza.
    (sigma_pct.is_finite() && sigma_pct >= 0.0 && v.is_finite()).then_some(v)
}

struct HarCalibration {
    model: Option<HarModel>,
    floor: Option<f64>,
    record: serde_json::Value,
}

fn fit_har(history: &HarHistory, ts: &[u64], y: &[f64], tau: u64) -> HarCalibration {
    let floor = y
        .iter()
        .filter_map(|&v| har_variance(v))
        .filter(|&v| v > 0.0)
        .min_by(f64::total_cmp);
    let (grid, missing_grid, grid_end) = match (ts.first(), ts.last()) {
        (Some(&first), Some(&last)) => har_grid(history, first, last, tau),
        _ => (Vec::new(), 0, 0),
    };
    let (windows, mut reasons) = har_windows(&grid);
    let mut rows = Vec::new();
    let mut targets = Vec::new();
    let (mut missing_history, mut nonfinite) = (0, 0);
    for (&t, &target) in ts.iter().zip(y) {
        if har_variance(target).is_none() {
            nonfinite += 1;
            continue;
        }
        if let Some(x) = history.features(t, tau, &windows) {
            rows.push(x);
            targets.push(target);
        } else {
            missing_history += 1;
        }
    }
    let model = if tau > 0 && ts.len() == y.len() && !grid.is_empty() {
        har_ridge(&rows, &targets, windows.clone())
    } else {
        None
    };
    if tau == 0 || ts.len() != y.len() {
        reasons.push("horizonte_o_alineacion_train_invalida".into());
    } else if targets.len() < windows.len() + 1 {
        reasons.push("menos_filas_train_utiles_que_coeficientes".into());
    } else if model.is_none() {
        reasons.push("sin_malla_train_o_sistema_numerico_no_resoluble".into());
    }
    let n_grid = grid.len() as f64;
    let window_ms: Option<Vec<u64>> = windows
        .iter()
        .map(|&w| u64::try_from(w).ok()?.checked_mul(tau))
        .collect();
    let record = serde_json::json!({
        "schema": "har-rv-qlike-v1", "status": if model.is_some() { "AJUSTADO" } else { "HAR_NO_VIABLE" },
        "horizon_ms": tau, "windows_tau": windows, "windows_ms": window_ms,
        "window_rule": "s=1; m: primer k>1 con |rho(k)|<=1/sqrt(N); l: primer k>m con |rho(k)|<=1/N; k<=N/2",
        "window_reduction_reasons": reasons, "grid_step_ms": tau,
        "grid_origin_ms": ts.first(), "grid_used_end_ms": grid_end,
        "grid_rows": grid.len(), "grid_missing_intervals": missing_grid,
        "grid_policy": "tramo contiguo mas largo; lags en intervalos fisicos tau; ACF centrada con denominador global",
        "threshold_m": if grid.is_empty() { None } else { Some(1.0/n_grid.sqrt()) },
        "threshold_l": if grid.is_empty() { None } else { Some(1.0/n_grid) },
        "train_rows_purged": y.len(), "fit_rows": targets.len(),
        "fit_missing_history": missing_history, "fit_invalid_targets": nonfinite,
        "train_first_ms": ts.first(), "train_last_ms": ts.last(),
        "beta": model.as_ref().map(|m| &m.beta),
        "ridge_lambda": model.as_ref().map(|m| m.ridge_lambda),
        "ridge_rule": "centrado y escala RMS train; intercepto libre; sqrt(f64::EPSILON)*traza(Gram)/p; si traza=0: sqrt(EPSILON)*N_fit",
        "variance_floor": floor, "floor_rule": "min((sigma_train_pct/100)^2 > 0), train purgado",
        "qlike_floor_policy": "max(varianza_pronosticada,floor) para todos; sigma<=0 usa floor",
        "fit_source": "solo etiquetas train purgado; seleccion y test no son argumentos",
        "ignored_overlap_ticks": history.ignored_overlap, "invalid_price_ticks": history.invalid_prices,
        "history_seams_ms": history.seams,
    });
    HarCalibration {
        model,
        floor,
        record,
    }
}

// None distingue el realizado cero y los valores no representables de una perdida.
fn har_qlike(actual_pct: f64, predicted_pct: f64, floor: f64) -> Option<(f64, bool)> {
    let actual = har_variance(actual_pct)?;
    if actual <= 0.0 || !predicted_pct.is_finite() || !floor.is_finite() || floor <= 0.0 {
        return None;
    }
    // Un sigma negativo no se convierte en un pronostico positivo al elevarlo al cuadrado.
    let predicted = if predicted_pct <= 0.0 {
        0.0
    } else {
        har_variance(predicted_pct)?
    };
    let clipped = predicted < floor;
    let ratio = actual / predicted.max(floor);
    if !ratio.is_finite() || ratio <= 0.0 {
        return None;
    }
    let delta = ratio - 1.0;
    let loss = if delta.abs() < 1.0 {
        delta - delta.ln_1p()
    } else {
        ratio - ratio.ln() - 1.0
    };
    (loss.is_finite() && loss >= 0.0).then_some((loss, clipped))
}

// Serie de evaluacion alineada que consume compare_har: etiquetas, las tres
// predicciones baseline y el pronostico medio, junto a los metadatos de la
// particion (marcas temporales y horizonte).
struct HarEvalSeries<'a> {
    ts: &'a [u64],
    y: &'a [f64],
    model: &'a [f64],
    persist: &'a [f64],
    mean: f64,
    tau: u64,
}

fn compare_har(
    calibration: &HarCalibration,
    history: &HarHistory,
    series: HarEvalSeries<'_>,
) -> serde_json::Value {
    let HarEvalSeries {
        ts,
        y,
        model,
        persist,
        mean,
        tau,
    } = series;
    let mut n_mse = 0usize;
    let mut n_qlike = 0usize;
    let (mut zero, mut missing, mut invalid, mut bad_qlike) = (0usize, 0usize, 0usize, 0usize);
    // Orden declarado, compartido por MSE, QLIKE y contadores de recorte.
    let names = ["modelo", "persistencia", "har", "media"];
    let mut mse = [0.0; 4];
    let mut qlike = [0.0; 4];
    let mut clipped = [0usize; 4];
    let aligned = ts.len() == y.len() && y.len() == model.len() && y.len() == persist.len();
    if aligned {
        for i in 0..y.len() {
            let prediction = calibration
                .model
                .as_ref()
                .and_then(|m| m.predict(history, ts[i], tau));
            let Some(har) = prediction else {
                missing += 1;
                continue;
            };
            let forecasts = [model[i], persist[i], har, mean];
            let errors = forecasts.map(|f| (y[i] - f).powi(2));
            if har_variance(y[i]).is_none() || errors.iter().any(|e| !e.is_finite()) {
                invalid += 1;
                continue;
            }
            n_mse += 1;
            for j in 0..names.len() {
                mse[j] += (errors[j] - mse[j]) / n_mse as f64;
            }
            if y[i] == 0.0 {
                zero += 1;
                continue;
            }
            let losses: Option<Vec<_>> = calibration.floor.and_then(|floor| {
                forecasts
                    .iter()
                    .map(|&f| har_qlike(y[i], f, floor))
                    .collect()
            });
            let Some(losses) = losses else {
                bad_qlike += 1;
                continue;
            };
            n_qlike += 1;
            for j in 0..names.len() {
                qlike[j] += (losses[j].0 - qlike[j]) / n_qlike as f64;
                clipped[j] += usize::from(losses[j].1);
            }
        }
    }
    // Regla de calidad declarada por la especificacion, sin tunear con holdout.
    let too_many_clips = clipped[2] > n_qlike.saturating_sub(clipped[2]);
    let viable = aligned
        && calibration.model.is_some()
        && calibration.floor.is_some()
        && n_qlike > 0
        && !too_many_clips
        && invalid == 0
        && bad_qlike == 0;
    let metrics: serde_json::Map<String, serde_json::Value> = (0..names.len())
        .map(|j| {
            (
                names[j].into(),
                serde_json::json!({
                    "mse_sigma_pct_common": if n_mse > 0 { Some(mse[j]) } else { None },
                    "qlike": if n_qlike > 0 { Some(qlike[j]) } else { None },
                    "qlike_clipped": clipped[j],
                }),
            )
        })
        .collect();
    serde_json::json!({
        "status": if viable { "COMPARABLE" } else { "HAR_NO_VIABLE" },
        "qlike_evidence": viable, "gate_uses_qlike": false,
        "rows_total": y.len(), "mse_common_rows": n_mse, "qlike_common_rows": n_qlike,
        "qlike_omitted_zero": zero, "zero_targets_total": y.iter().filter(|&&v| v == 0.0).count(),
        "har_missing_history": missing,
        "invalid_rows": invalid, "qlike_unrepresentable_rows": bad_qlike,
        "aligned_inputs": aligned, "har_more_than_half_clipped": too_many_clips,
        "quality_rule": "HAR_NO_VIABLE si mas de la mitad de filas QLIKE comunes requieren recorte HAR",
        "common_population": "todos los modelos usan las mismas filas; ceros siguen en MSE; gate conserva su poblacion original",
        "variance_floor": calibration.floor, "metrics": metrics,
        "history_seams_ms": history.seams, "ignored_overlap_ticks": history.ignored_overlap,
    })
}

// Los archivos auxiliares no amplian ModelManifest ni cambian la escritura del modelo.
fn write_har_record(directory: &str, name: &str, record: &serde_json::Value) -> Result<(), String> {
    let path = std::path::Path::new(directory).join(name);
    std::fs::create_dir_all(directory).map_err(|e| e.to_string())?;
    let bytes = serde_json::to_vec_pretty(record).map_err(|e| e.to_string())?;
    std::fs::write(&path, bytes).map_err(|e| format!("{}: {e}", path.display()))
}

// ── PARIDAD ENTRENAMIENTO↔SERVICIO ───────────────────────────────────────
//
// Todo lo que sigue existe para que el vector que este binario aprende sea,
// literalmente, el objeto que el motor sirve — y para poder DEMOSTRARLO.

/// Ensamblado del vector ML **tal y como lo hace el servicio**.
///
/// Es una copia literal del contrato de `god-engine-core` (bloque del bosque
/// dentro de `process_tick_dual`): 34 universales ⊕ 10 espectrales ⊕ 4 macro,
/// y saneo de no finitos a 0.0 —el mismo que el servicio aplica para que un
/// feed macro ausente no mate la predicción entera—. Devuelve además cuántas
/// dimensiones hubo que sanear: es una magnitud medible del estado del motor,
/// no un detalle cosmético, y se informa al operador.
///
/// ÉSTA ES LA ÚNICA VÍA por la que este binario construye un vector. El
/// `fe` que recibe debe ser `core.feature_engines[coin]` —el motor mismo—,
/// nunca un `StatefulEngine` alimentado a mano: eso es exactamente el defecto
/// que esta revisión corrige, y `primera_divergencia` lo detecta si vuelve.
fn vector_de_servicio(fe: &StatefulEngine, omni: &[f64; 54]) -> ([f32; DIM_VECTOR], u32) {
    let mut v = [0f32; DIM_VECTOR];
    v[..34].copy_from_slice(&fe.get_universal_features());
    v[34..44].copy_from_slice(&fe.get_spectral_ml_features());
    v[44..].copy_from_slice(&macro_ml_features(omni));
    let mut saneadas = 0u32;
    for x in v.iter_mut() {
        if !x.is_finite() {
            *x = 0.0;
            saneadas += 1;
        }
    }
    (v, saneadas)
}

/// COMPROBACIÓN DE PARIDAD: primera dimensión en la que el vector que el
/// entrenador va a guardar NO coincide con lo que el motor produce en ese
/// mismo instante. `None` ⇒ paridad demostrada en esta muestra.
///
/// CONTRA QUÉ SE COMPARA, Y POR QUÉ ES UNA COMPROBACIÓN CON DIENTES:
///
/// · dims [0..34) — contra `GodEngineCore::build_54d_tensor`, que es un
///   productor del MOTOR (el servicio lo llama en cada evento para el tensor
///   de la red) y cuyas 34 primeras entradas son, por construcción,
///   `self.feature_engines[coin].get_universal_features()`. Si el entrenador
///   volviera a leer un `StatefulEngine` propio —el defecto histórico— o
///   leyera otra moneda, o leyera ANTES de alimentar el evento, estas 34
///   dimensiones dejarían de coincidir y el binario aborta.
/// · dims [44..48) — contra `macro_ml_features` evaluada sobre el MISMO
///   `omni` que se pasó a `process_event`: el bloque macro del servicio no
///   tiene otra fuente.
/// · dims [34..44) — no existe un segundo productor del bloque espectral en
///   el motor: `get_spectral_ml_features()` es la única vía, en servicio y
///   aquí. Pero al quedar certificado por las 34 primeras que el objeto leído
///   ES `core.feature_engines[coin]`, el bloque espectral sale forzosamente
///   de ese mismo objeto: la certificación es transitiva.
///
/// TOLERANCIA: CERO, y no es una elección estética. Entre los dos lados no
/// media ninguna aritmética — sólo un ensanchado f32→f64, que es exacto—, de
/// modo que cualquier diferencia es ESTRUCTURAL (otro objeto, otro instante,
/// otro contrato), nunca numérica. Un épsilon aquí sólo serviría para tapar
/// justo el fallo que se busca.
fn primera_divergencia(
    v: &[f32; DIM_VECTOR],
    tensor_motor: &[f64; 54],
    macro_motor: &[f32; 4],
) -> Option<(usize, f64, f64)> {
    // El saneo del servicio convierte no finitos en 0.0; el tensor de la red
    // no lo hace. Se compara contra el valor SANEADO para no acusar de
    // divergencia a una diferencia que el propio contrato introduce.
    let saneado = |x: f64| if x.is_finite() { x } else { 0.0 };
    for d in 0..34 {
        let mio = v[d] as f64;
        let suyo = saneado(tensor_motor[d]);
        if mio != suyo {
            return Some((d, mio, suyo));
        }
    }
    for k in 0..4 {
        let mio = v[44 + k] as f64;
        let suyo = saneado(macro_motor[k] as f64);
        if mio != suyo {
            return Some((44 + k, mio, suyo));
        }
    }
    None
}

/// Dimensión declarada MUERTA EN SERVICIO que llega distinta de cero.
///
/// `FEATURES_DEAD_IN_SERVE` es la única fuente de verdad del mapa vivo/muerto
/// del contrato 34D y hoy `get_universal_features()` ya la aplica en AMBOS
/// lados. Por eso el entrenador ya NO zerifica esas dims por su cuenta: lo
/// hace el motor. Esta comprobación fija el invariante — si un día el
/// servicio dejara de aplicarlo, el entrenador dejaría de escribir modelos en
/// vez de aprender una distribución que el vivo no sirve.
fn dim_muerta_no_nula(v: &[f32; DIM_VECTOR]) -> Option<usize> {
    FEATURES_DEAD_IN_SERVE
        .iter()
        .copied()
        .find(|&d| d < DIM_VECTOR && v[d] != 0.0)
}

/// REJILLA DE PRECIOS DEL INSTRUMENTO, MEDIDA EN EL PROPIO TAPE.
///
/// El motor necesita un `tick_size` para poner suelo al semi-spread simulado
/// (D-718/D-752), y ese suelo entra en `update_ofi` —dim [2] del vector—, así
/// que no puede ser un literal. La menor diferencia POSITIVA entre precios
/// medios consecutivos ES la rejilla: en un mes de datos, un movimiento de un
/// solo tick ocurre miles de veces. Mismo método que `tape_spread_fix`
/// (D-751b); derivarlo de los datos evita introducir otra constante.
///
/// El umbral `mid · 1e-12` descarta ruido de coma flotante acumulado por el
/// round-trip cálculo→fichero→mmap (cuatro órdenes por encima del épsilon de
/// f64) y queda órdenes por debajo de cualquier rejilla real.
fn tick_del_tape(raw: &[BinTick]) -> f64 {
    let mut menor = f64::INFINITY;
    let mut previo = 0.0f64;
    for t in raw {
        if t.bid <= 0.0 || t.ask <= 0.0 || t.bid > t.ask || t.ts == 0 {
            continue;
        }
        let mid = (t.bid + t.ask) * 0.5;
        if previo > 0.0 {
            let d = (mid - previo).abs();
            if d > mid * 1e-12 && d < menor {
                menor = d;
            }
        }
        previo = mid;
    }
    if menor.is_finite() {
        menor
    } else {
        0.0
    }
}

/// REJILLA DE CANTIDAD DEL INSTRUMENTO, MEDIDA EN EL PROPIO TAPE.
///
/// Mismo argumento que `tick_del_tape` sobre la cantidad negociada
/// `|bq − aq|`: la menor diferencia positiva entre cantidades consecutivas
/// distintas es el `step_size` del exchange. NO entra en ninguna dimensión
/// del vector (sólo se usa para que la ficha del símbolo sea la del
/// instrumento y no un literal), pero registrarla medida es más barato que
/// justificar un número inventado.
fn paso_del_tape(raw: &[BinTick]) -> f64 {
    let mut menor = f64::INFINITY;
    let mut previo = 0.0f64;
    for t in raw {
        if t.bid <= 0.0 || t.ask <= 0.0 || t.bid > t.ask || t.ts == 0 {
            continue;
        }
        let q = agg_trade_qty(t.bq, t.aq);
        if !(q.is_finite() && q > 0.0) {
            continue;
        }
        if previo > 0.0 {
            let d = (q - previo).abs();
            if d > q * 1e-12 && d < menor {
                menor = d;
            }
        }
        previo = q;
    }
    if menor.is_finite() {
        menor
    } else {
        0.0
    }
}

/// Recorrido (mín, máx) de cada dimensión sobre TODAS las muestras.
///
/// Una dimensión con recorrido nulo es una columna CONSTANTE: ningún árbol
/// puede partir por ella, así que el modelo no la usa — y si en servicio esa
/// misma dimensión sí varía, el modelo está sirviendo un vector cuyo soporte
/// no vio. No se aborta por ello (el entrenador no puede medir el servicio),
/// pero se DECLARA: es la lista que hay que mirar cuando un modelo satura.
struct RecorridoDims {
    min: [f32; DIM_VECTOR],
    max: [f32; DIM_VECTOR],
    n: usize,
}

impl RecorridoDims {
    fn nuevo() -> Self {
        Self { min: [f32::INFINITY; DIM_VECTOR], max: [f32::NEG_INFINITY; DIM_VECTOR], n: 0 }
    }
    fn observar(&mut self, v: &[f32; DIM_VECTOR]) {
        for d in 0..DIM_VECTOR {
            if v[d] < self.min[d] {
                self.min[d] = v[d];
            }
            if v[d] > self.max[d] {
                self.max[d] = v[d];
            }
        }
        self.n += 1;
    }
    /// Dimensiones constantes en TODO el entrenamiento.
    fn constantes(&self) -> Vec<usize> {
        (0..DIM_VECTOR).filter(|&d| self.n > 0 && self.min[d] == self.max[d]).collect()
    }
}

// ── GBDT núcleo ──────────────────────────────────────────────────────────

struct TreeNode {
    feature: i32,
    threshold: f32,
    left: i32,
    right: i32,
    value: f32,
}

fn sigmoid(x: f64) -> f64 {
    1.0 / (1.0 + (-x).clamp(-50.0, 50.0).exp())
}

fn regression_residual(prediction: f64, target: f64) -> f64 {
    // build_tree stores +G/(H+lambda), then the booster ADDS the tree.
    // G must therefore be the negative gradient of half squared error.
    target - prediction
}

/// Freeze exactly the additive predictor used during training:
/// F(x) = init + sum(lr * tree(x)). The wire format has global children
/// and no separate learning-rate field, so shrinkage belongs in leaf values.
fn serialize_forest(
    trees: &[Vec<TreeNode>],
    init_score: f32,
    lr: f64,
) -> Result<NanoForestData, String> {
    if trees.is_empty() || !init_score.is_finite() || !lr.is_finite() || lr <= 0.0 {
        return Err(
            "forest export requires trees, finite bias and positive finite learning rate".into(),
        );
    }
    let mut model = NanoForestData {
        children_left: Vec::new(),
        children_right: Vec::new(),
        feature: Vec::new(),
        threshold: Vec::new(),
        value: Vec::new(),
        tree_offsets: Vec::new(),
        init_score,
    };
    for tree in trees {
        if tree.is_empty() {
            return Err("cannot export an empty tree".into());
        }
        let base = i32::try_from(model.value.len()).map_err(|_| "too many tree nodes")?;
        model.tree_offsets.push(base);
        let relocate = |child: i32| -> Result<i32, String> {
            if child == -1 {
                return Ok(-1);
            }
            if child < 0 || child as usize >= tree.len() {
                return Err("invalid local child index".into());
            }
            base.checked_add(child)
                .ok_or_else(|| "child index overflow".into())
        };
        for node in tree {
            model.children_left.push(relocate(node.left)?);
            model.children_right.push(relocate(node.right)?);
            model.feature.push(node.feature);
            model.threshold.push(node.threshold);
            model.value.push((node.value as f64 * lr) as f32);
        }
    }
    model
        .tree_offsets
        .push(i32::try_from(model.value.len()).map_err(|_| "too many tree nodes")?);
    NanoForest::from_data(model.clone())?;
    Ok(model)
}

/// Gate and diagnostics must evaluate the frozen, retained artifact, not
/// f_val from a later early-stopping iteration that has been discarded.
fn serving_predictions(model: &NanoForestData, features: &[Vec<f32>]) -> Result<Vec<f64>, String> {
    let forest = NanoForest::from_data(model.clone())?;
    features
        .iter()
        .map(|x| {
            forest
                .predict_raw_checked(x)
                .map(|(raw, _)| raw as f64)
                .ok_or_else(|| "invalid serving prediction in validation set".into())
        })
        .collect()
}

/// The probe proposes thresholds only. Score every candidate on the same
/// bootstrap multiset as its parent and children (including multiplicities).
/// Sorting once and sweeping prefixes/suffixes avoids O(rows * thresholds).
fn candidate_gains(
    data: &[(Vec<f32>, f64, f64)],
    idx: &[usize],
    feature: usize,
    thresholds: &[f32],
    lambda: f64,
    min_child: usize,
) -> Vec<Option<f64>> {
    let mut gains = vec![None; thresholds.len()];
    if !lambda.is_finite()
        || lambda < 0.0
        || min_child == 0
        || idx.iter().any(|&i| {
            !data[i].0[feature].is_finite()
                || !data[i].1.is_finite()
                || !data[i].2.is_finite()
                || data[i].2 < 0.0
        })
    {
        return gains;
    }
    let mut ordered = idx.to_vec();
    ordered.sort_unstable_by(|&a, &b| data[a].0[feature].total_cmp(&data[b].0[feature]));
    let mut suffix = vec![(0.0, 0.0); ordered.len() + 1];
    for j in (0..ordered.len()).rev() {
        suffix[j] = (
            suffix[j + 1].0 + data[ordered[j]].1,
            suffix[j + 1].1 + data[ordered[j]].2,
        );
    }
    let score = |g: f64, h: f64| g * g / (h + lambda + 1e-12);
    let parent_score = score(suffix[0].0, suffix[0].1);
    let mut queries: Vec<_> = thresholds
        .iter()
        .copied()
        .enumerate()
        .filter(|(_, t)| t.is_finite())
        .collect();
    queries.sort_unstable_by(|a, b| a.1.total_cmp(&b.1));
    let (mut nl, mut gl, mut hl) = (0usize, 0.0, 0.0);
    for (q, thr) in queries {
        while nl < ordered.len() && data[ordered[nl]].0[feature] <= thr {
            gl += data[ordered[nl]].1;
            hl += data[ordered[nl]].2;
            nl += 1;
        }
        if nl >= min_child && ordered.len() - nl >= min_child {
            let (gr, hr) = suffix[nl];
            let gain = score(gl, hl) + score(gr, hr) - parent_score;
            if gain.is_finite() {
                gains[q] = Some(gain);
            }
        }
    }
    gains
}

/// For an inclusive span S and a budget B, floor(S / B) + 1 ensures
/// floor(S / stride) + 1 <= B. Saturation is backed by an explicit counter.
fn effective_stride(span: u64, requested: u64, cap: usize) -> Result<u64, String> {
    if requested == 0 || cap == 0 {
        return Err("stride and sample budget must be positive".into());
    }
    let cap = u64::try_from(cap).map_err(|_| "sample budget does not fit timestamp domain")?;
    Ok(requested.max((span / cap).saturating_add(1)))
}

struct SamplingBudget {
    stride: u64,
    limit: usize,
    attempts: usize,
    next: Option<u64>,
}

impl SamplingBudget {
    fn new(span: u64, stride: u64, limit: usize) -> Result<Self, String> {
        Ok(Self {
            stride: effective_stride(span, stride, limit)?,
            limit,
            attempts: 0,
            next: Some(0),
        })
    }
    /// A reserved attempt consumes budget even if its label is later rejected.
    fn reserve(&mut self, ts: u64) -> bool {
        if self.attempts >= self.limit || self.next.is_none_or(|next| ts < next) {
            return false;
        }
        self.attempts += 1;
        self.next = ts.checked_add(self.stride);
        true
    }
}

#[derive(Clone, Copy, Debug)]
struct LabelInterval {
    start: u64,
    end: u64,
}

#[derive(Default)]
struct TrainingSamples {
    features: Vec<Vec<f32>>,
    labels: Vec<f64>,
    intervals: Vec<LabelInterval>,
}

impl TrainingSamples {
    fn push(&mut self, features: Vec<f32>, label: f64, start: u64, end: u64) {
        self.features.push(features);
        self.labels.push(label);
        self.intervals.push(LabelInterval { start, end });
    }
    fn validate(&self) -> Result<(), String> {
        let n = self.labels.len();
        if n == 0 || self.features.len() != n || self.intervals.len() != n {
            return Err("empty or misaligned sample set".into());
        }
        let dim = self.features[0].len();
        if dim == 0
            || self
                .features
                .iter()
                .any(|x| x.len() != dim || x.iter().any(|v| !v.is_finite()))
            || self.labels.iter().any(|v| !v.is_finite())
            || self
                .intervals
                .iter()
                .any(|v| v.start == 0 || v.end < v.start)
            || self.intervals.windows(2).any(|w| w[1].start <= w[0].start)
        {
            return Err("invalid features, labels or chronological intervals".into());
        }
        Ok(())
    }
    #[allow(dead_code)]
    fn split_at(&mut self, at: usize) -> Self {
        Self {
            features: self.features.split_off(at),
            labels: self.labels.split_off(at),
            intervals: self.intervals.split_off(at),
        }
    }
    #[allow(dead_code)]
    fn max_end(&self) -> u64 {
        self.intervals.iter().map(|v| v.end).max().unwrap_or(0)
    }
}

/// Label windows are closed intervals: equality at the boundary is overlap.
/// End times need not be monotone: filter every row, not only a suffix.
fn purge_training(
    train: TrainingSamples,
    selection: &TrainingSamples,
) -> Result<(TrainingSamples, usize), String> {
    train.validate()?;
    selection.validate()?;
    let boundary = selection.intervals[0].start;
    if train.intervals.last().unwrap().start >= boundary {
        return Err("selection must start after all training feature times".into());
    }
    let mut kept = TrainingSamples::default();
    let mut purged = 0;
    for ((x, y), interval) in train
        .features
        .into_iter()
        .zip(train.labels)
        .zip(train.intervals)
    {
        if interval.end >= boundary {
            purged += 1;
        } else {
            kept.push(x, y, interval.start, interval.end);
        }
    }
    kept.validate()?;
    Ok((kept, purged))
}

/// XLIV-13: adapta las muestras de `build` (marca t, etiqueta en (t, t+τ])
/// al tipo de los contratos FMT, con el intervalo cerrado [t, t+τ] que usan
/// `purge_training` y `require_later_holdout`.
fn como_intervalos(
    feats: &[Vec<f32>],
    labels: &[f64],
    ts: &[u64],
    horizon_ms: u64,
) -> TrainingSamples {
    let mut out = TrainingSamples::default();
    for ((x, &y), &t) in feats.iter().zip(labels).zip(ts) {
        out.push(x.clone(), y, t, t.saturating_add(horizon_ms));
    }
    out
}

fn require_later_holdout(test: &TrainingSamples, evidence_end: u64) -> Result<(), String> {
    test.validate()?;
    if test.intervals[0].start <= evidence_end {
        return Err("test overlaps training/selection information intervals".into());
    }
    Ok(())
}

/// Plan v1.4: el holdout del gate es posterior a TODA etiqueta de early stop.
/// Train se purga contra early stop antes de ajustar el bosque o HAR.
fn prepare_early_stop(
    train: TrainingSamples,
    early: &TrainingSamples,
    selection: &TrainingSamples,
) -> Result<(TrainingSamples, usize), String> {
    early.validate()?;
    require_later_holdout(selection, early.max_end())?;
    purge_training(train, early)
}

/// La tolerancia de mejora y la cadencia legada se conservan.
/// Con --early-stop-in se observa cada arbol y paciencia cuenta arboles.
struct EarlyStopping {
    best_loss: f64,
    best_rounds: usize,
    since_improve: usize,
}

impl EarlyStopping {
    fn new() -> Self {
        Self {
            best_loss: f64::INFINITY,
            best_rounds: 0,
            since_improve: 0,
        }
    }

    fn observe(&mut self, loss: f64, rounds: usize, elapsed: usize, patience: usize) -> bool {
        assert!(loss.is_finite(), "non-finite early-stop loss");
        if loss + 1e-9 < self.best_loss {
            self.best_loss = loss;
            self.best_rounds = rounds;
            self.since_improve = 0;
        } else {
            self.since_improve += elapsed;
        }
        self.since_improve >= patience
    }
}

/// El flag nuevo no puede caer silenciosamente al valor por defecto.
fn early_stop_path(args: &[String], val_in: &str, patience: usize) -> Result<String, String> {
    let positions: Vec<_> = args
        .iter()
        .enumerate()
        .filter(|(_, a)| a.as_str() == "--early-stop-in")
        .map(|(i, _)| i)
        .collect();
    if args.iter().any(|a| a.starts_with("--early-stop-in=")) || positions.len() > 1 {
        return Err("use --early-stop-in FILE una sola vez".into());
    }
    let Some(&i) = positions.first() else {
        return Ok(String::new());
    };
    let path = args
        .get(i + 1)
        .filter(|p| !p.is_empty() && !p.starts_with('-'))
        .ok_or("--early-stop-in requiere una ruta")?;
    if val_in.is_empty() || patience == 0 {
        return Err("--early-stop-in requiere --val-in independiente y --patience > 0".into());
    }
    Ok(path.clone())
}

fn require_promotion_holdout(promote: bool, test_path: &str) -> Result<(), String> {
    if promote && test_path.is_empty() {
        return Err("--promote requires a later, independent --test-in; --val-in is selection evidence, not a final test".into());
    }
    Ok(())
}

fn passes_loss_gate(regression: bool, loss: f64, baseline: f64, margin: f64) -> bool {
    if !loss.is_finite()
        || !baseline.is_finite()
        || !margin.is_finite()
        || loss < 0.0
        || baseline < 0.0
        || margin < 0.0
    {
        return false;
    }
    if regression {
        baseline > 0.0 && (baseline - loss) / baseline >= margin
    } else {
        baseline - loss >= margin
    }
}

/// Quantile candidates may be approximate; their gain uses full-node statistics.
fn build_tree(
    data: &mut Vec<(Vec<f32>, f64, f64)>, // (features, grad, hess) por muestra IN PLACE
    idx: &mut Vec<usize>,                 // índices del nodo (se particiona in situ)
    depth: u32,
    max_depth: u32,
    min_child: usize,
    lambda: f64,
    feat_subset: &[usize],
    n_quantiles: usize,
    rng: &mut StdRng,
    nodes: &mut Vec<TreeNode>,
) {
    let g_sum: f64 = idx.iter().map(|&i| data[i].1).sum();
    let h_sum: f64 = idx.iter().map(|&i| data[i].2).sum();
    let leaf_value = (g_sum / (h_sum + lambda)) as f32;
    if depth >= max_depth || idx.len() < 2 * min_child {
        nodes.push(TreeNode {
            feature: -1,
            threshold: 0.0,
            left: -1,
            right: -1,
            value: leaf_value,
        });
        return;
    }
    // Approximate threshold proposals only; no claim of unbiased gain from probe.
    let probe: Vec<usize> = if idx.len() > 2048 {
        let mut p: Vec<usize> = Vec::with_capacity(2048);
        for _ in 0..2048 {
            p.push(idx[(rng.random::<f64>() * idx.len() as f64) as usize]);
        }
        p
    } else {
        idx.clone()
    };
    let mut best = (f64::MIN, 0usize, 0.0f32); // (ganancia, feature, threshold)
    for &f in feat_subset {
        let mut vals: Vec<f32> = probe.iter().map(|&i| data[i].0[f]).collect();
        if vals.windows(2).all(|w| w[0] == w[1]) {
            continue;
        }
        vals.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
        // Umbrales candidatos por cuantiles (excluye extremos)
        let thresholds: Vec<_> = (1..n_quantiles)
            .map(|q| vals[q * vals.len() / n_quantiles])
            .collect();
        let gains = candidate_gains(data, idx, f, &thresholds, lambda, min_child);
        for (thr, gain) in thresholds.into_iter().zip(gains) {
            let Some(gain) = gain else { continue };
            if gain > best.0 {
                best = (gain, f, thr);
            }
        }
    }
    if best.0 <= 1e-9 {
        nodes.push(TreeNode {
            feature: -1,
            threshold: 0.0,
            left: -1,
            right: -1,
            value: leaf_value,
        });
        return;
    }
    let (f, thr) = (best.1, best.2);
    let mut left_idx: Vec<usize> = Vec::with_capacity(idx.len() / 2);
    let mut right_idx: Vec<usize> = Vec::with_capacity(idx.len() / 2);
    for &i in idx.iter() {
        if data[i].0[f] <= thr {
            left_idx.push(i);
        } else {
            right_idx.push(i);
        }
    }
    let me = nodes.len();
    nodes.push(TreeNode {
        feature: f as i32,
        threshold: thr,
        left: 0,
        right: 0,
        value: 0.0,
    });
    // DFS pre-orden: subárbol izquierdo en [me+1, …]; la raíz del derecho
    // se captura ANTES de construirlo (nodes.len()-1 tras el derecho sería
    // su última hoja, no su raíz).
    build_tree(
        data,
        &mut left_idx,
        depth + 1,
        max_depth,
        min_child,
        lambda,
        feat_subset,
        n_quantiles,
        rng,
        nodes,
    );
    nodes[me].left = (me + 1) as i32;
    let right_root = nodes.len();
    build_tree(
        data,
        &mut right_idx,
        depth + 1,
        max_depth,
        min_child,
        lambda,
        feat_subset,
        n_quantiles,
        rng,
        nodes,
    );
    nodes[me].right = right_root as i32;
}

/// B3.4 — carga las series macro reales de data/macro. VIX/SP500/NASDAQ
/// (Yahoo v8, cierres idénticos a FRED) son OBLIGATORIAS: faltan ⇒ aborto,
/// porque una columna de ceros constante es capacidad fantasma. DXY
/// (DTWEXBGS, sólo FRED) se TOLERA ausente con warning: el CDN de FRED
/// abre y cierra ventanas desde esta red y un re-run del sync lo completa;
/// mientras tanto la dim queda en 0 neutro y NINGÚN árbol parte por ella
/// (columna constante = sin splits — no hay ruptura de paridad).
fn load_macro_series() -> (
    Vec<(u64, f64)>,
    Vec<(u64, f64)>,
    Vec<(u64, f64)>,
    Vec<(u64, f64)>,
) {
    let read = |tag: &str, required: bool| -> Vec<(u64, f64)> {
        let path = format!("data/macro/{tag}.csv");
        let content = match std::fs::read_to_string(&path) {
            Ok(c) => c,
            Err(e) => {
                if required {
                    eprintln!(
                        "❌ macro {tag}: falta data/macro/{tag}.csv ({e}) — ejecuta `macro_history_sync` primero."
                    );
                    std::process::exit(1);
                }
                eprintln!(
                    "⚠️ macro {tag}: sin data/macro/{tag}.csv ({e}) — dim en 0 neutro, sin splits. Re-ejecuta `macro_history_sync` cuando FRED abra."
                );
                return Vec::new();
            }
        };
        let rows: Vec<(u64, f64)> = content
            .lines()
            .skip(1)
            .filter_map(|l| {
                let mut p = l.split(',');
                let ms = p.next()?.trim().parse::<u64>().ok()?;
                let v = p.next()?.trim().parse::<f64>().ok()?;
                (v.is_finite() && v > 0.0).then_some((ms, v))
            })
            .collect();
        if rows.len() < 100 {
            if required {
                eprintln!("❌ macro {tag}: historia irreal ({} filas)", rows.len());
                std::process::exit(1);
            }
            eprintln!(
                "⚠️ macro {tag}: historia irreal ({} filas) — dim en 0 neutro",
                rows.len()
            );
            return Vec::new();
        }
        println!("   [macro] {tag}: {} días", rows.len());
        rows
    };
    (
        read("VIX", true),
        read("SP500", true),
        read("DXY", false),
        read("NASDAQ", true),
    )
}

/// TGB-G / TGBG-5: cinta que puede entrar a la PREPARACION DE MUESTRAS de entrenamiento, por el puente comun
/// (`quantum_engine::tgb_generation`, la misma admision que `dataset_admission`): instantanea privada de la cinta
/// logica, firma RELATIVA recalculada entera y cadena recibo -> evidencia del productor -> indice revisado fijado
/// por `--generation-index-sha256`. Cualquier otra cosa se rechaza con su causa.
fn admitir_cinta_para_muestras(
    path: &str,
    lectores: &dyn quantum_engine::tgb_generation::Lectores,
    indice: Option<&quantum_engine::tgb_generation::IndiceRevisado>,
) -> Result<quantum_engine::tgb_generation::CintaAdmitida, String> {
    quantum_engine::tgb_generation::admitir_fuente(std::path::Path::new(path), lectores, indice).map_err(|e| e.to_string())
}

/// TGBG-5: las fuentes declaradas (train, `--val-in` y `--test-in`), TODAS admitidas antes de preparar una sola
/// muestra. Un rechazo de cualquiera detiene el entrenador antes del primer build.
struct FuentesAdmitidas {
    train: quantum_engine::tgb_generation::CintaAdmitida,
    seleccion: Option<quantum_engine::tgb_generation::CintaAdmitida>,
    test: Option<quantum_engine::tgb_generation::CintaAdmitida>,
}

fn admitir_fuentes(
    in_path: &str,
    val_in: &str,
    test_in: &str,
    admitir: &dyn Fn(&str) -> Result<quantum_engine::tgb_generation::CintaAdmitida, String>,
) -> Result<FuentesAdmitidas, String> {
    let train = admitir(in_path)?;
    let seleccion = if val_in.is_empty() { None } else { Some(admitir(val_in)?) };
    let test = if test_in.is_empty() { None } else { Some(admitir(test_in)?) };
    Ok(FuentesAdmitidas { train, seleccion, test })
}

/// TGBG-5: los registros de la instantanea admitida, con parseo little-endian seguro (sin reinterpretar la memoria
/// ni suponer su alineacion). La reserva falla de forma controlada.
fn ticks_de_cinta(bytes: &[u8]) -> Result<Vec<BinTick>, String> {
    const REGISTRO: usize = 40;
    let cuerpo = bytes.strip_prefix(b"TGMTICK1").ok_or_else(|| "la cinta no empieza por TGMTICK1".to_string())?;
    if cuerpo.len() % REGISTRO != 0 {
        return Err(format!("registro parcial: {} B no son registros de {REGISTRO} B", cuerpo.len()));
    }
    let mut ticks = Vec::new();
    ticks
        .try_reserve_exact(cuerpo.len() / REGISTRO)
        .map_err(|e| format!("sin memoria para {} ticks: {e}", cuerpo.len() / REGISTRO))?;
    let palabra = |r: &[u8], k: usize| {
        let mut w = [0u8; 8];
        w.copy_from_slice(&r[k * 8..(k + 1) * 8]);
        u64::from_le_bytes(w)
    };
    for r in cuerpo.chunks_exact(REGISTRO) {
        ticks.push(BinTick {
            ts: palabra(r, 0),
            bid: f64::from_bits(palabra(r, 1)),
            ask: f64::from_bits(palabra(r, 2)),
            bq: f64::from_bits(palabra(r, 3)),
            aq: f64::from_bits(palabra(r, 4)),
        });
    }
    Ok(ticks)
}

/// CACHEV3-P5 (opt-in, `--cache columnas`): los ticks de la cinta YA ADMITIDA (la admision de TGB-G no se salta),
/// servidos desde la cache de columnas si su manifiesto cubre el original entero (mismo SHA-256, registros
/// `[0, n_total)` sin huecos) y cada hora verifica sus SHA-256. Si no, `None` con el motivo, y el entrenador sigue
/// por la instantanea admitida, sin error. Bits exactos: cada palabra se copia, no se reinterpreta.
fn ticks_desde_cache(cache_dir: &str, sha256_original: &[u8; 32], n_total: usize) -> Option<Vec<BinTick>> {
    let sha = cache_columnas::hex(sha256_original);
    let bytes = match cache_columnas::lector::registros_de_original(std::path::Path::new(cache_dir), &sha, n_total as u64) {
        Ok(Some(b)) => b,
        Ok(None) => {
            println!("   [cache] {cache_dir} no cubre la cinta {sha}: camino de siempre (cinta mapeada)");
            return None;
        }
        Err(e) => {
            println!("   [cache] {cache_dir} rechazada ({e}): camino de siempre (cinta mapeada)");
            return None;
        }
    };
    let palabra = |r: &[u8], k: usize| {
        let mut w = [0u8; 8];
        w.copy_from_slice(&r[k * 8..(k + 1) * 8]);
        u64::from_le_bytes(w)
    };
    let ticks: Vec<BinTick> = bytes
        .chunks_exact(std::mem::size_of::<BinTick>())
        .map(|r| BinTick {
            ts: palabra(r, 0),
            bid: f64::from_bits(palabra(r, 1)),
            ask: f64::from_bits(palabra(r, 2)),
            bq: f64::from_bits(palabra(r, 3)),
            aq: f64::from_bits(palabra(r, 4)),
        })
        .collect();
    println!("   [cache] {} ticks desde {cache_dir} (columnas verificadas)", ticks.len());
    Some(ticks)
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let symbol = if args.len() > 1 && !args[1].starts_with('-') {
        args[1].clone()
    } else {
        eprintln!("Uso: train_forest <SYMBOL> [--in FILE] [--max-samples N] ...");
        std::process::exit(1);
    };
    let arg = |name: &str, dflt: &str| -> String {
        args.iter()
            .position(|a| a == name)
            .and_then(|p| args.get(p + 1))
            .map(|s| s.to_string())
            .unwrap_or_else(|| dflt.to_string())
    };
    let default_in = format!("data/{}_SEP26.bin", symbol);
    let in_path = arg("--in", &default_in);
    let max_samples: usize = arg("--max-samples", "200000").parse().unwrap();
    // Clock horizon, not event count. The inherited 5-minute default is
    // configurable, not a universal scale or a data-derived optimum.
    // Example: 500 events spaced 75ms span about 37.5s, NOT 2s.
    let horizon_ms: u64 = arg("--horizon-ms", "300000").parse().unwrap();
    let stride_ms: u64 = arg("--stride-ms", "50000").parse().unwrap();
    let n_rounds: usize = arg("--trees", "300").parse().unwrap();
    let lr: f64 = arg("--lr", "0.1").parse().unwrap();
    let max_depth: u32 = arg("--depth", "5").parse().unwrap();
    let min_child: usize = arg("--min-child", "40").parse().unwrap();
    let lambda: f64 = arg("--lambda", "1.0").parse().unwrap();
    let patience: usize = arg("--patience", "40").parse().unwrap();
    let promote = args.iter().any(|a| a == "--promote");
    // CACHEV3-P5: lectura por la cache de columnas, SOLO con `--cache columnas` (opt-in del operador).
    let usar_cache_columnas = match arg("--cache", "").as_str() {
        "" => false,
        "columnas" => true,
        otro => {
            eprintln!("❌ --cache inválido: {otro} (columnas)");
            std::process::exit(1);
        }
    };
    let cache_dir = arg("--cache-dir", cache_columnas::RUTA_POR_DEFECTO);
    // XLIV-13 (recompone FMT; el merge 6fdccd64 del PR #5 desconectó este
    // contrato): promover exige un test POSTERIOR e independiente, y se
    // comprueba antes de cualquier E/S. --val-in es evidencia de SELECCIÓN.
    let test_in = arg("--test-in", "");
    require_promotion_holdout(promote, &test_in).expect("promotion contract rejected before I/O");
    // QS-2: libro de usos del holdout final y motivo declarado para reusarlo.
    let holdout_ledger = arg("--holdout-ledger", model_manifest::DEFAULT_HOLDOUT_LEDGER);
    let test_reuse = arg("--test-reuse", "");
    // TGBG-5: --val-in se lee aqui (antes de admitir) y el indice revisado de generacion se carga con su pin.
    let val_in = arg("--val-in", "");
    let early_stop_in = early_stop_path(&args, &val_in, patience)
        .expect("early-stop CLI rejected before I/O");
    let indice_ruta = arg("--generation-index", "");
    let indice_sha = arg("--generation-index-sha256", "");
    let (lectores_generacion, indice_generacion) = match (indice_ruta.is_empty(), indice_sha.is_empty()) {
        (true, true) => (quantum_engine::tgb_generation::LectoresDisco::sin_indice(), None),
        (false, false) => {
            let ruta = std::path::Path::new(&indice_ruta);
            match quantum_engine::tgb_generation::IndiceRevisado::cargar(ruta, &indice_sha) {
                Ok(ix) => (quantum_engine::tgb_generation::LectoresDisco::para_indice(ruta), Some(ix)),
                Err(e) => {
                    eprintln!("❌ --generation-index: {e}");
                    std::process::exit(1);
                }
            }
        }
        _ => {
            eprintln!("❌ --generation-index y --generation-index-sha256 van juntos");
            std::process::exit(1);
        }
    };
    // CALENTAMIENTO — MEDIDO EN RELOJ DEL TAPE, NO EN TICKS.
    //
    // Ningún vector se muestrea hasta que el estado MÁS LENTO que entra en él
    // ha visto su propia memoria. El más lento es `kline_ema_macro`, que
    // alimenta la dim [43] (`dev(kline_ema_macro)` del bloque espectral): una
    // EMA de 720 velas de 1 minuto dentro de `stateful_engine::process_tick`.
    // Su memoria es, por tanto, 720 minutos de RELOJ —no un número de ticks,
    // que depende de la intensidad del feed y cambiaría entre símbolos y
    // entre meses—. El 720 no es una constante de este fichero: es el período
    // que declara el motor, y el calentamiento se deriva de él.
    //
    // El transitorio de la semilla (la EMA arranca en el primer precio) NO se
    // elimina del todo en una memoria: es una propiedad del motor, idéntica
    // en producción tras cada arranque en frío y en el forense, así que
    // reproducirla ES la paridad, no un defecto del entrenador.
    const PERIODO_EMA_KLINE_MAS_LENTA: u64 = 720; // velas de 1 minuto (stateful_engine)
    const MS_POR_VELA: u64 = 60_000;
    let calentamiento_por_defecto = (PERIODO_EMA_KLINE_MAS_LENTA * MS_POR_VELA).to_string();
    let calentamiento_ms: u64 = arg("--calentamiento-ms", &calentamiento_por_defecto)
        .parse()
        .unwrap();
    // El capital NO entra en ninguna dimensión del vector y, con las entradas
    // bloqueadas (ver la reproducción), no se mueve durante toda la corrida:
    // existe sólo porque el arena exige uno. Se toma de INITIAL_CAPITAL —la
    // misma variable que usa el medidor forense, para que la reproducción sea
    // literalmente la suya— y, a falta de ella, del MÍNIMO NOCIONAL PUBLICADO
    // por Binance para USDⓈ-M (5 USDT): el menor capital con el que alguna
    // orden es representable. Al final se verifica que no se movió.
    const MIN_NOTIONAL_PUBLICADO: f64 = 5.0;
    let capital_arena = std::env::var("INITIAL_CAPITAL")
        .ok()
        .and_then(|v| v.parse::<f64>().ok())
        .filter(|c| c.is_finite() && *c > 0.0)
        .unwrap_or(MIN_NOTIONAL_PUBLICADO);
    // P-1/P-2/P-3c — objetivo: dir (barrera triple HOST-010) | vol (σ(τ)
    // REALIZADA del horizonte, % de precio, regresión) | volu (NOCIONAL
    // negociado del horizonte, Σ|qty|·precio en quote, regresión) | oi (ΔOI% a
    // horizonte, regresión con join as-of del histórico horario).
    // D-731/D-732: este comentario decía «σ futura» y «profundidad media»
    // describiendo las etiquetas VIEJAS (ambas divididas por el número de
    // ticks, y la segunda sobre un par de libro que el tape de aggTrades no
    // tiene). Se corrige aquí porque un comentario que miente sobre las
    // unidades es cómo se promueven modelos de escalas incompatibles.
    let label_mode = arg("--label", "dir");
    if !matches!(label_mode.as_str(), "dir" | "vol" | "volu" | "oi") {
        eprintln!("❌ --label inválido: {} (dir|vol|volu|oi)", label_mode);
        std::process::exit(1);
    }

    // P-3c — serie histórica de OI para `--label oi` (join as-of estricto:
    // la última fila con ts ≤ t). El endpoint sólo conserva ~30 días: la
    // validación es DENTRO de la ventana (split temporal del subconjunto
    // con label), documentado en la salida.
    let oi_series: Vec<(u64, f64)> = if label_mode == "oi" {
        let path = format!("data/oihist/{}.csv", symbol);
        let content = std::fs::read_to_string(&path).unwrap_or_else(|e| {
            eprintln!(
                "❌ --label oi sin histórico: {} ({}) — ejecuta oi_history_sync",
                path, e
            );
            std::process::exit(1);
        });
        let mut rows: Vec<(u64, f64)> = content
            .lines()
            .skip(1)
            .filter_map(|ln| {
                let mut it = ln.split(',');
                let ts = it.next()?.trim().parse::<u64>().ok()?;
                let oi = it.next()?.trim().parse::<f64>().ok()?;
                (ts > 0 && oi.is_finite() && oi > 0.0).then_some((ts, oi))
            })
            .collect();
        rows.sort_unstable_by_key(|(ts, _)| *ts);
        if rows.len() < 48 {
            eprintln!("❌ histórico OI demasiado corto: {} filas", rows.len());
            std::process::exit(1);
        }
        println!(
            "   [oi] {} filas · span {:.1} días (validación DENTRO de ventana)",
            rows.len(),
            (rows.last().unwrap().0 - rows[0].0) as f64 / 86_400_000.0
        );
        rows
    } else {
        Vec::new()
    };

    println!("🌲 [TRAIN-FOREST] {} ← {}", symbol, in_path);
    println!("   muestras≤{} horizonte={}ms stride={}ms árboles≤{} lr={} depth={} λ={}",
             max_samples, horizon_ms, stride_ms, n_rounds, lr, max_depth, lambda);
    println!(
        "   rasgos por el camino del MOTOR (GodEngineCore::process_event) · calentamiento \
         {} ms = memoria de la EMA de vela más lenta del vector · paridad verificada dim a dim",
        calentamiento_ms
    );

    let oi_series_ref = &oi_series;
    // TGB-E: cinta lógica (`.bin`, o `.bin.tgb` verificado entero si el `.bin` no existe).
    // TGBG-5: train, selección y test ADMITIDOS por el puente común ANTES de construir una sola muestra; un
    // rechazo de cualquiera sale con su causa sin preparar nada.
    let fuentes = admitir_fuentes(&in_path, &val_in, &test_in, &|p: &str| {
        admitir_cinta_para_muestras(p, &lectores_generacion, indice_generacion.as_ref())
    })
    .unwrap_or_else(|e| {
        eprintln!("❌ {e}");
        std::process::exit(1);
    });
    // La fuente adicional tambien se admite antes del primer build.
    let early_stop_source = if early_stop_in.is_empty() {
        None
    } else {
        Some(
            admitir_cinta_para_muestras(
                &early_stop_in,
                &lectores_generacion,
                indice_generacion.as_ref(),
            )
            .expect("early-stop source rejected before samples"),
        )
    };
    // ── 1+2. Muestras: features+etiquetas desde ticks (reutilizable) ────
    let build = |cinta: &quantum_engine::tgb_generation::CintaAdmitida| -> Samples {
        // TGBG-5: se consume la instantánea ADMITIDA (los bytes analizados), con parseo LE seguro.
        let path_s = cinta.ruta_logica().display().to_string();
        let path: &str = &path_s;
        let ticks = ticks_de_cinta(cinta.bytes()).unwrap_or_else(|e| {
            eprintln!("❌ {path}: {e}");
            std::process::exit(1);
        });
        let n_total = ticks.len();
        if n_total < 50_000 {
            eprintln!("❌ datos insuficientes: {} ticks ({})", n_total, path);
            std::process::exit(1);
        }
        // CACHEV3-P5: sin `--cache columnas` no se consulta nada y `raw` son los ticks de la instantánea admitida.
        let desde_cache = if usar_cache_columnas { ticks_desde_cache(&cache_dir, &cinta.sha256_original(), n_total) } else { None };
        let raw: &[BinTick] = desde_cache.as_deref().unwrap_or(&ticks);
        validate_tick_order(raw).expect("invalid tick chronology");
        println!("   [{}] {} ticks crudos", path, n_total);

        let last_ts = raw[n_total - 1].ts;
        let span_ms = last_ts.saturating_sub(raw[0].ts);
        // XLIV-13 (recompone FMT): el presupuesto lo gobierna `SamplingBudget`,
        // que nunca densifica el stride pedido y lo ensancha lo justo para
        // que floor(span/stride)+1 ≤ presupuesto. La versión inline hacía lo
        // contrario en las dos ramas: en tapes largos usaba el stride pedido
        // y EXCEDÍA el presupuesto (nada más lo acotaba); en cortos lo
        // DENSIFICABA (30 días, 50 s, 200 k ⇒ 13 s: 4× más solape de etiquetas).
        let mut budget = SamplingBudget::new(span_ms, stride_ms.max(1), max_samples)
            .expect("invalid sampling budget");
        let stride_ms_eff = budget.stride;
        println!("   [{}] span {:.1} días · stride efectivo {}ms", path,
                 span_ms as f64 / 86_400_000.0, stride_ms_eff);
        // D-732 — `--label volu` etiqueta NOCIONAL NEGOCIADO, y eso exige que
        // (bid_qty, ask_qty) venga de la fabricación de aggTrades: sólo ahí la
        // diferencia de los dos lados recupera el `qty` del trade. En un tape
        // sintético de klines la diferencia es una fracción del volumen de vela
        // y la etiqueta mediría una invención. Una pasada aritmética sobre los
        // ticks válidos (despreciable frente al O(n·ventana) del bucle
        // principal) decide: si ALGÚN tick incumple la identidad del productor,
        // se aborta en vez de entrenar sobre ruido.
        if label_mode == "volu" {
            let (mut n_valid, mut n_bad) = (0usize, 0usize);
            for tk in raw.iter() {
                if tk.bid <= 0.0 || tk.ask <= 0.0 || tk.bid > tk.ask || tk.ts == 0 {
                    continue;
                }
                n_valid += 1;
                if !is_aggtrades_pair(tk.bq, tk.aq) {
                    n_bad += 1;
                }
            }
            if n_valid == 0 || n_bad > 0 {
                eprintln!(
                    "❌ --label volu sobre un tape que NO es de aggTrades: {}/{} ticks incumplen \
                     la identidad min = max(0,25·qty; 0,1) de binance_vision_sync ({}). \
                     El nocional negociado no es medible en este fichero — usa el tape de \
                     aggTrades o no entrenes este predictor.",
                    n_bad, n_valid, path
                );
                std::process::exit(1);
            }
            println!("   [{}] tape aggTrades verificado: qty recuperable en {} ticks", path, n_valid);
        }
        // ── EL MOTOR, NO UN TROZO DE ÉL (D-753) ─────────────────────────────
        //
        // La rejilla del instrumento se MIDE en el tape (no hay tabla de
        // símbolos cableada aquí): el tick pone suelo al semi-spread simulado
        // y ese suelo entra en `update_ofi` ⇒ dim [2] del vector. El paso de
        // cantidad, el mínimo nocional (5 USDT) y el apalancamiento máximo
        // (125×) son LÍMITES PUBLICADOS del exchange y no entran en ninguna
        // dimensión: existen para que la ficha del símbolo sea la del
        // instrumento y el motor sepa sobre qué moneda corre (de ahí saca la
        // clave `{SÍMBOLO}_MOTOR` y la rama lead/lag).
        let tick_size = tick_del_tape(raw);
        let step_size = paso_del_tape(raw);
        if !(tick_size > 0.0) {
            eprintln!(
                "❌ [{}] no se pudo medir la rejilla de precios del tape: todos los precios \
                 medios consecutivos son iguales o no finitos. Sin tick no hay suelo de \
                 semi-spread y la dim [2] (OFI) del vector no sería la del servicio.",
                path
            );
            std::process::exit(1);
        }
        println!(
            "   [{}] rejilla medida en el tape: tick {:.10} · paso de cantidad {:.10}",
            path, tick_size, step_size
        );
        quantum_arena::symbol_registry::update_registry(vec![SymbolSpec {
            symbol: symbol.clone(),
            step_size: if step_size > 0.0 { step_size } else { tick_size },
            tick_size,
            min_qty: if step_size > 0.0 { step_size } else { tick_size },
            // Límites PUBLICADOS por Binance para USDⓈ-M, no elecciones.
            min_notional: 5.0,
            max_leverage: 125,
            maker_fee: 0.0002,
            taker_fee: 0.0005,
            is_shadow: false,
        }]);
        let arena = quantum_arena::GlobalArena::build_in_own_stack(capital_arena);
        arena.config.live_maker_fee.store(0.0002, Ordering::Relaxed);
        arena.config.live_taker_fee.store(0.0005, Ordering::Relaxed);
        let mut core = GodEngineCore::new(arena.clone());
        // El vector NO depende del genoma —los períodos de las EMAs que lo
        // alimentan están fijados en `stateful_engine` precisamente para no
        // romper esta paridad (nota S-8 del motor)—, pero `refresh_models`
        // relee el almacén cada 1000 ticks y aplicaría a mitad de corrida
        // cualquier generación que otro proceso sancione. Con la generación
        // aplicada al máximo el entrenamiento deja de depender de qué genoma
        // hubiera en disco mientras corría.
        core.applied_generation.store(u64::MAX, Ordering::Relaxed);
        // ENTRADAS BLOQUEADAS DURANTE TODA LA REPRODUCCIÓN.
        //
        // POR QUÉ: si el motor opera, pierde capital; cuando el capital llega
        // a cero `process_event` activa el kill-switch y RETORNA ANTES de
        // actualizar los rasgos (lib.rs, primer guardia). El entrenador
        // seguiría emitiendo muestras con el vector CONGELADO en el instante
        // de la ruina, sin avisar. Es el peor fallo posible aquí: silencioso.
        //
        // POR QUÉ NO ROMPE LA PARIDAD: el propio motor documenta que el
        // bloqueo de entradas actúa DESPUÉS de la analítica —«la gestión de
        // posiciones y la analítica ML/espectral ya corrieron completas»— y
        // el vector es una función pura del tape y del macro: ninguna rama de
        // trading escribe en las 48 dimensiones. Al final se verifica que el
        // capital no se movió y que el kill-switch nunca se activó: la prueba
        // de que ningún camino de operativa se ejecutó.
        quantum_arena::feed_health::stall();
        let mut omni = [0.0f64; 54];
        // B3.4 — bloque MACRO real (dims 44..48): series FRED VIXCLS/SP500/
        // DTWEXBGS/NASDAQCOM de data/macro (macro_history_sync). Join as-of
        // ESTRICTO: último cierre de un día ANTERIOR al tick — el cierre del
        // mismo día aún no existe intradía (lookahead). Sin las cuatro series
        // reales no se entrena: ceros harían columnas muertas, y las columnas
        // muertas son capacidad fantasma (directriz del operador).
        //
        // D-753: ahora devuelve los NIVELES CRUDOS, no el bloque ya
        // transformado. El motor aplica `macro_ml_features` él mismo sobre el
        // `omni` que recibe en `process_event`, así que el entrenador debe
        // pasarle los mismos niveles en las mismas ranuras (21 DXY, 22 SP500,
        // 23 NASDAQ, 24 VIX) y dejar que el contrato lo transforme UNA vez.
        let (macro_vix, macro_spx, macro_dxy, macro_ndx) = load_macro_series();
        let mut c_vix = 0usize;
        let mut c_spx = 0usize;
        let mut c_dxy = 0usize;
        let mut c_ndx = 0usize;
        // (dxy, spx, ndx, vix) — ranuras 21, 22, 23, 24 del omni.
        let mut macro_asof = |ts: u64| -> Option<[f64; 4]> {
            let day_start = ts - (ts % 86_400_000);
            // Serie VACÍA (DXY sin FRED) ⇒ 0 neutro — columna sin splits,
            // no descarte de muestras. Serie con datos pero tick anterior a
            // su cobertura ⇒ muestra descartada (honestidad).
            fn adv(series: &[(u64, f64)], cur: &mut usize, day_start: u64) -> Option<f64> {
                if series.is_empty() {
                    return Some(0.0);
                }
                while *cur + 1 < series.len() && series[*cur + 1].0 < day_start {
                    *cur += 1;
                }
                match series.get(*cur) {
                    Some(&(ms, v)) if ms < day_start => Some(v),
                    _ => None,
                }
            }
            Some([
                adv(&macro_dxy, &mut c_dxy, day_start)?,
                adv(&macro_spx, &mut c_spx, day_start)?,
                adv(&macro_ndx, &mut c_ndx, day_start)?,
                adv(&macro_vix, &mut c_vix, day_start)?,
            ])
        };
        let mut feats: Vec<Vec<f32>> = Vec::new();
        let mut labels: Vec<f64> = Vec::new();
        // D-733 — baseline de PERSISTENCIA: la misma magnitud de la etiqueta
        // medida en la ventana ANTERIOR (t−τ, t]. Se guarda por muestra para
        // poder evaluarla EXACTAMENTE sobre la partición de validación.
        let mut persist: Vec<f64> = Vec::new();
        // Marca temporal de cada muestra: la purga del solape train/val
        // necesita saber cuándo se resuelve cada etiqueta.
        let mut sample_ts: Vec<u64> = Vec::new();
        let mut neutrals = 0usize;
        // Sin contexto temporal COMPLETO (ventana (t−τ, t] dentro del tape, o
        // fila de OI anterior a t−τ) no hay persistencia medible ⇒ la muestra
        // se descarta en los modos de regresión, nunca se rellena con un
        // sustituto inventado.
        let mut sin_contexto = 0usize;
        let tp_pct = 0.0036;
        let sl_pct = 0.0018;
        // Primer tick UTILIZABLE del tape (no `raw[0].ts`: la cabecera puede ir
        // seguida de ticks inválidos con ts = 0). Marca el borde por debajo del
        // cual la ventana (t−τ, t] dejaría de estar cubierta por datos.
        let mut first_ts: u64 = 0;
        // Fin del calentamiento en RELOJ del tape (ver `--calentamiento-ms`).
        let mut fin_calentamiento: u64 = 0;
        // Día UTC cuyo bloque macro está cargado en `omni`, y si ese bloque
        // existe: sin macro as-of t−1 la muestra se descarta, pero el motor
        // sigue alimentándose para no partir su estado.
        let mut dia_macro: i64 = i64::MIN;
        let mut macro_vigente = false;
        let mut sin_macro = 0usize;
        // Telemetría de la comprobación de paridad.
        let mut recorrido = RecorridoDims::nuevo();
        let mut muestras_con_saneo = 0usize;
        let mut dims_saneadas = 0u64;
        let mut prev_ts: u64 = 0;
        for i in 0..n_total {
            let t = &raw[i];
            if !valid_tick(t) {
                continue;
            }
            if first_ts == 0 {
                first_ts = t.ts;
                fin_calentamiento = t.ts.saturating_add(calentamiento_ms);
            }
            let mid = (t.bid + t.ask) / 2.0;
            // D-747 — LA CANTIDAD Y EL LADO DEL AGRESOR SALEN DEL CONVENIO DEL
            // CODIFICADOR, NO DE UNA SUMA NI DE SU INVERSA.
            //
            // `binance_vision_sync` codifica cada aggTrade así: el lado del
            // AGRESOR lleva `qty + base` y el pasivo sólo `base`, con
            // `base = max(0,25·qty; 0,1)`. Por tanto la cantidad real es
            // `|bq − aq|` y `is_buyer_maker` (el comprador era el pasivo) es
            // `aq > bq`. Es el MISMO convenio que lee el forense.
            let vol = (t.bq - t.aq).abs();
            let is_buyer_maker = t.aq > t.bq;

            // ── MACRO DEL DÍA (as-of t−1 estricto) ──────────────────────────
            // El macro es diario: se relee sólo al cambiar de día, como hace
            // el forense. Los NIVELES van a las ranuras del omni y es el
            // motor quien aplica `macro_ml_features`; así el bloque [44..48)
            // se construye UNA sola vez y por el contrato del servicio.
            let dia = (t.ts / 86_400_000) as i64;
            if dia != dia_macro {
                dia_macro = dia;
                match macro_asof(t.ts) {
                    Some([dxy, spx, ndx, vix]) => {
                        omni[21] = dxy;
                        omni[22] = spx;
                        omni[23] = ndx;
                        omni[24] = vix;
                        macro_vigente = true;
                    }
                    None => {
                        omni[21] = 0.0;
                        omni[22] = 0.0;
                        omni[23] = 0.0;
                        omni[24] = 0.0;
                        macro_vigente = false;
                    }
                }
            }

            // ── REPRODUCCIÓN POR EL CAMINO DEL MOTOR (D-753) ────────────────
            //
            // Exactamente el bucle de `audit_forensic_backtest.rs`: libro
            // simulado SIMÉTRICO alrededor del punto medio con suelo de medio
            // tick (D-752: restarlo al bid y sumarlo al ask DUPLICABA la
            // horquilla), OBI real del par de cantidades, cierre de vela por
            // reloj, y DOS eventos por tick —depth y trade—, que es como el
            // motor vivo recibe el mercado. El lado agresor viaja en el dato
            // (D-717), no se deduce de una comparación con el libro simulado.
            let half_spread = ((t.ask - t.bid) / 2.0).max(tick_size * 0.5);
            let sim_bid = mid - half_spread;
            let sim_ask = mid + half_spread;
            let real_obi = if (t.bq + t.aq) > 0.0 {
                (t.bq - t.aq) / (t.bq + t.aq)
            } else {
                0.0
            };
            let is_minute_kline = prev_ts == 0 || (t.ts / 60_000) != (prev_ts / 60_000);
            prev_ts = t.ts;
            core.arena.update_l2_depth(0, t.bq, t.aq);
            let _ = core.process_event(
                0, false, is_minute_kline, true, mid, vol, sim_bid, sim_ask, t.bq, t.aq,
                real_obi, 0.0, t.ts, false, &omni, false,
            );
            let _ = core.process_event(
                0, true, false, false, mid, vol, sim_bid, sim_ask, t.bq, t.aq, real_obi,
                0.0, t.ts, false, &omni, is_buyer_maker,
            );
            // El latch del núcleo sólo se arma aquí con capital ≤ 0 (no hay
            // sistema inmune ni demonio en el entrenador): el replay ya no
            // es el del motor que se quiere reproducir, así que se aborta.
            // Desde CL-43 `process_event` sigue actualizando los rasgos bajo
            // el latch (sólo bloquea entradas), pero el aborto se mantiene.
            if core.arena.kill_switch_active.load(Ordering::Relaxed) {
                eprintln!(
                    "❌ [{}] kill-switch del motor activo en el tick {} (capital ≤ 0): \
                     el replay ya no reproduce el motor. Entrenamiento abortado.",
                    path, i
                );
                std::process::exit(1);
            }

            // `reserve` consume presupuesto aunque la muestra se descarte luego
            // (sin macro, neutra): el presupuesto cuenta intentos, no éxitos.
            if t.ts >= fin_calentamiento && t.ts + horizon_ms <= last_ts && budget.reserve(t.ts) {
                // Sin macro as-of t−1 la muestra se descarta (honestidad),
                // nunca se rellena: el motor ya quedó alimentado arriba.
                if !macro_vigente {
                    sin_macro += 1;
                    continue;
                }
                // EL VECTOR SALE DEL MOTOR, POR EL CONTRATO DEL SERVICIO.
                // `core.feature_engines[0]` es el MISMO objeto del que el
                // servicio lee en `process_tick_dual`; se lee justo después
                // del evento de trade, que es el último del tick y el que
                // fija el vector con el que el motor decide.
                let (full, saneadas) = vector_de_servicio(&core.feature_engines[0], &omni);
                if saneadas > 0 {
                    muestras_con_saneo += 1;
                    dims_saneadas += saneadas as u64;
                }
                // ── COMPROBACIÓN DE PARIDAD (aborta) ────────────────────────
                if let Some(d) = dim_muerta_no_nula(&full) {
                    eprintln!(
                        "❌ [{}] PARIDAD ROTA en el tick {}: la dim [{}] está declarada MUERTA \
                         EN SERVICIO (FEATURES_DEAD_IN_SERVE) y el motor la entrega en {}. \
                         El modelo aprendería una columna que el vivo sirve en 0.",
                        path, i, d, full[d]
                    );
                    std::process::exit(1);
                }
                let tensor_motor = core.build_54d_tensor(0, t.bq, t.aq, mid, &omni);
                let macro_motor = macro_ml_features(&omni);
                if let Some((d, mio, suyo)) =
                    primera_divergencia(&full, &tensor_motor, &macro_motor)
                {
                    eprintln!(
                        "❌ [{}] PARIDAD ROTA en el tick {}: la dim [{}] vale {:e} en el vector \
                         que este entrenador guardaría y {:e} en el que el motor produce en el \
                         MISMO instante. Entre ambos lados no media aritmética alguna, así que \
                         la diferencia es ESTRUCTURAL: el entrenador no está leyendo el motor. \
                         Un entrenador que no puede demostrar su paridad no escribe un modelo.",
                        path, i, d, mio, suyo
                    );
                    std::process::exit(1);
                }
                recorrido.observar(&full);
                // Antes aquí había un `if full.iter().all(is_finite)` que
                // DESCARTABA la muestra. Ya no hace falta y además rompía la
                // paridad: el servicio NO descarta el vector cuando una dim
                // llega no finita — la sanea a 0.0 y sigue opinando (B2.5).
                // `vector_de_servicio` aplica ese mismo saneo, y el número de
                // dimensiones saneadas se informa al operador en vez de
                // desaparecer con la muestra.
                {
                    // P-1/P-2 (MOTOR UNIVERSAL): modos de PREDICCIÓN además de
                    // dirección, todos funciones del MISMO horizonte τ =
                    // horizon_ms (no hay bandas ni pares de valores: τ es el
                    // único parámetro de escala).
                    //   `--label vol`  ⇒ σ(τ) realizada = sqrt(Σ r²)·100  [% de
                    //                    precio sobre (t, t+τ]]  (D-731)
                    //   `--label volu` ⇒ nocional negociado Σ |qty|·precio en
                    //                    (t, t+τ]                        (D-732)
                    //   `--label oi`   ⇒ ΔOI% en (t, t+τ] por join as-of
                    // REGRESIÓN (grad = f−y, hess = 1, init = media; serving por
                    // predict_raw SIN sigmoid). `dir` ⇒ barrera triple HOST-010
                    // (clasificación, como siempre).
                    //
                    // Cada muestra de regresión lleva además su PERSISTENCIA:
                    // la misma magnitud medida en (t−τ, t], información
                    // estrictamente ≤ t. Es el baseline que el gate exige batir.
                    if label_mode != "dir" {
                        if label_mode == "oi" {
                            // P-3c — ΔOI% a horizonte por join as-of de la
                            // serie horaria (última fila con ts ≤ t). Sin
                            // label válido (fuera de ventana o horizonte
                            // incompleto) la muestra se DESCARTA — nunca se
                            // rellena.
                            let oi_asof = |ts: u64| -> Option<f64> {
                                let idx = oi_series_ref
                                    .binary_search_by(|(ots, _)| ots.cmp(&ts))
                                    .unwrap_or_else(|i| i);
                                if idx == 0 {
                                    // ts anterior a la primera fila: sin valor
                                    if oi_series_ref
                                        .first()
                                        .map(|(ots, _)| *ots > ts)
                                        .unwrap_or(true)
                                    {
                                        return None;
                                    }
                                }
                                if idx >= oi_series_ref.len() {
                                    oi_series_ref.last().map(|(_, oi)| *oi)
                                } else if oi_series_ref[idx].0 == ts {
                                    Some(oi_series_ref[idx].1)
                                } else if idx > 0 {
                                    Some(oi_series_ref[idx - 1].1)
                                } else {
                                    None
                                }
                            };
                            let deadline = t.ts + horizon_ms;
                            // D-733 — persistencia del ΔOI: el ΔOI% del
                            // horizonte ANTERIOR (t−τ, t], todo con ts ≤ t.
                            let past = t.ts.saturating_sub(horizon_ms);
                            let (Some(oi_now), Some(oi_fut), Some(oi_past)) =
                                (oi_asof(t.ts), oi_asof(deadline), oi_asof(past))
                            else {
                                sin_contexto += 1;
                                continue;
                            };
                            // honestidad: el futuro debe ser una fila REAL del
                            // histórico (deadline ≤ última fila), no el último
                            // valor colado por el as-of del borde.
                            if deadline > oi_series_ref.last().unwrap().0 {
                                continue;
                            }
                            // …y el pasado debe estar DENTRO del histórico: si
                            // t−τ precede a la primera fila, el as-of no tiene
                            // valor que devolver y la persistencia sería una
                            // invención.
                            if past < oi_series_ref[0].0 || oi_past <= 0.0 {
                                sin_contexto += 1;
                                continue;
                            }
                            let label = (oi_fut - oi_now) / oi_now * 100.0;
                            let pers = (oi_now - oi_past) / oi_past * 100.0;
                            if label.is_finite() && pers.is_finite() {
                                feats.push(full.to_vec());
                                labels.push(label);
                                persist.push(pers);
                                sample_ts.push(t.ts);
                            }
                        } else {
                        let deadline = t.ts + horizon_ms;
                        // Ventana FUTURA (t, t+τ]: arranca en i+1 (nunca en el
                        // propio tick t) y corta en el primer ts > vencimiento.
                        let fwd_end = forward_window_end(raw, i, deadline);
                        let fwd = accumulate_window(&raw[i + 1..fwd_end], mid);
                        if fwd.n_ret == 0 {
                            continue;
                        }
                        // Ventana PASADA (t−τ, t]: mismo ancho τ, información
                        // estrictamente ≤ t. Sin τ completo hacia atrás (arranque
                        // del fichero) no hay persistencia y la muestra se cae.
                        let past = t.ts.saturating_sub(horizon_ms);
                        if past < first_ts {
                            sin_contexto += 1;
                            continue;
                        }
                        let (lo, anchor) = trailing_window_start(raw, i, past);
                        let back = accumulate_window(&raw[lo..=i], anchor);
                        if back.n_ret == 0 {
                            sin_contexto += 1;
                            continue;
                        }
                        // D-731/D-732 — unidades: σ(τ) en % de precio y nocional
                        // negociado del horizonte. Ninguna de las dos se divide
                        // por el número de ticks: esa división convertía la
                        // magnitud en «por tick» y la hacía DECRECER cuando el
                        // mercado se activaba, justo al revés de lo que mide.
                        let (label, pers) = if label_mode == "vol" {
                            (realized_vol_pct(fwd.sum_sq), realized_vol_pct(back.sum_sq))
                        } else {
                            (fwd.notional, back.notional)
                        };
                        if label.is_finite() && pers.is_finite() {
                            feats.push(full.to_vec());
                            labels.push(label);
                            persist.push(pers);
                            sample_ts.push(t.ts);
                        }
                        }
                    } else {
                    // Triple barrera por TIEMPO DE RELOJ dentro de horizon_ms.
                    // HOST-010 (DECIMOCUARTO): la geometría del label debe ser
                    // la del trade REAL — SL −sl_pct vs TP +tp_pct (RR≥2 por
                    // friction_floors). La versión anterior chequeaba el SL del
                    // corto (`>= mid*(1+sl_pct)`, +0.18%) ANTES del TP largo
                    // (+0.36%): el TP era código muerto y todo toque de +0.18%
                    // se contaba como victoria — el modelo aprendía una barrera
                    // simétrica ±0.18% (coin-flip tras fees) en vez del trade
                    // asimétrico RR 2:1 que el vivo ejecuta. Ahora: primer
                    // toque de SL largo ⇒ 0.0, primer toque de TP largo ⇒ 1.0,
                    // tocar ±sl_pct sin llegar al TP ⇒ timeout neutral
                    // (descartado — coincide con la escalera trailing: un
                    // trade que toca +0.18% y vuelve cierra en BE). El corto
                    // consume 1−p en serve (ml_thr_short): con esta definición
                    // es P(SL largo primero) — proxy honesto de la hipótesis
                    // corta.
                    let deadline = t.ts + horizon_ms;
                    let long_tp = mid * (1.0 + tp_pct);
                    let long_sl = mid * (1.0 - sl_pct);
                    let mut label = 0.5f64;
                    'barrier: for f in (i + 1)..n_total {
                        let ft = &raw[f];
                        if ft.ts > deadline {
                            break 'barrier;
                        }
                        if ft.bid <= 0.0 || ft.ask <= 0.0 {
                            continue;
                        }
                        let fut_mid = (ft.bid + ft.ask) / 2.0;
                        if fut_mid <= long_sl {
                            label = 0.0;
                            break 'barrier;
                        }
                        if fut_mid >= long_tp {
                            label = 1.0;
                            break 'barrier;
                        }
                    }
                    if (label - 0.5).abs() < 1e-9 {
                        neutrals += 1;
                    } else {
                        feats.push(full.to_vec());
                        labels.push(label);
                        // Clasificación: el baseline es la tasa base del propio
                        // modelo, no una persistencia — se rellena con la tasa
                        // base más adelante y NUNCA se usa en el gate `dir`.
                        persist.push(0.5);
                        sample_ts.push(t.ts);
                    }
                    } // fin modo dir
                }
            }
        }
        // ── CIERRE DE LA REPRODUCCIÓN: LA PARIDAD, DECLARADA ────────────────
        //
        // El capital intacto y el kill-switch apagado son la PRUEBA de que
        // ninguna rama de operativa corrió: el vector que se acaba de
        // aprender es función únicamente del tape y del macro, como en
        // servicio. Si alguna vez dejaran de cumplirse, el vector podría
        // depender de la simulación de PnL y esto habría que saberlo.
        let capital_final = arena.unified_capital.load(Ordering::Relaxed);
        if (capital_final - capital_arena).abs() > 0.0 {
            eprintln!(
                "❌ [{}] el capital se movió durante la reproducción ({} → {}): el motor operó \
                 pese al bloqueo de entradas. El vector podría depender de la simulación de PnL \
                 y la paridad con el servicio deja de estar demostrada.",
                path, capital_arena, capital_final
            );
            std::process::exit(1);
        }
        println!(
            "   [{}] PARIDAD: {} muestras verificadas dim a dim contra el motor (tolerancia 0; \
             capital intacto, kill-switch apagado)",
            path, recorrido.n
        );
        if muestras_con_saneo > 0 {
            println!(
                "   [{}] ⚠️ {} muestras llegaron con dims no finitas del motor ({} dims en total, \
                 saneadas a 0.0 igual que en servicio)",
                path, muestras_con_saneo, dims_saneadas
            );
        }
        // Columnas CONSTANTES: ningún árbol puede partir por ellas. Si en
        // servicio esa dim sí varía, el modelo sirve un vector cuyo soporte no
        // vio — que es exactamente cómo se satura un modelo. No se aborta (el
        // entrenador no puede medir el servicio), se DECLARA.
        let constantes = recorrido.constantes();
        if !constantes.is_empty() {
            println!(
                "   [{}] dims CONSTANTES en todo el entrenamiento (sin splits posibles): {:?}",
                path, constantes
            );
            println!(
                "   [{}]    de ellas, MUERTAS POR CONTRATO en ambos lados: {:?} — el resto son \
                 columnas que este tape no hace variar; revísalas si el modelo satura en servicio.",
                path, FEATURES_DEAD_IN_SERVE
            );
        }
        let n = labels.len();
        println!(
            "   [{}] stride efectivo {}ms · {} de {} intentos del presupuesto",
            path, stride_ms_eff, budget.attempts, budget.limit
        );
        if n < 5_000 {
            eprintln!(
                "❌ muestras insuficientes: {} (+{} neutros, +{} sin contexto temporal, \
                 +{} sin macro as-of t−1). El calentamiento consume los primeros {} ms del \
                 tape (memoria de la EMA de vela más lenta que entra en el vector): si el tape \
                 es corto, amplíalo o reduce --calentamiento-ms a sabiendas de que las dims \
                 34..44 arrancarán con memoria de la semilla.",
                n, neutrals, sin_contexto, sin_macro, calentamiento_ms
            );
            std::process::exit(1);
        }
        if label_mode == "dir" {
            let pos_rate = labels.iter().filter(|&&y| y > 0.5).count() as f64 / n as f64;
            println!(
                "   [{}] {} muestras decisivas ({} neutros) · largo {:.1}%",
                path,
                n,
                neutrals,
                pos_rate * 100.0
            );
        } else {
            let mean_y = labels.iter().sum::<f64>() / n as f64;
            let mean_p = persist.iter().sum::<f64>() / n as f64;
            println!("   [{}] {} muestras ({} sin contexto temporal) · label[{}] media {:.6} \
                      · persistencia media {:.6}",
                     path, n, sin_contexto, label_mode, mean_y, mean_p);
        }
        Samples { feats, labels, persist, ts: sample_ts }
    };
    let s_train = build(&fuentes.train);
    let (feats, labels, persist_all, sample_ts) =
        (s_train.feats, s_train.labels, s_train.persist, s_train.ts);
    let n = labels.len();
    // El diagnostico HAR consume las fuentes ya admitidas, despues de conocer la purga.
    let mut har_history: Result<HarHistory, String> = Ok(HarHistory::default());

    // ── 3. Split: temporal 80/20 CON PURGA, o CRUZADO por archivo ────────
    // El split cruzado (train AGO → val SEP) es la validación MÁS dura y
    // honesta disponible: el edge debe sobrevivir un mes de mercado nuevo.
    //
    // D-734 — PURGA DEL SOLAPE (López de Prado). Con τ = 300 s y stride
    // efectivo de 50 s las ventanas de etiqueta se solapan 6×: las últimas
    // muestras de train resuelven su barrera/su σ DENTRO del tramo de
    // validación. Eso es fuga: la validación deja de medir mercado nuevo. Se
    // descartan del train las muestras cuya ventana (t, t+τ] alcanza la
    // primera muestra de validación. El recorte no es un número elegido: lo
    // fija τ contra la densidad real del muestreo.
    let split_samples: SplitSamples = if val_in.is_empty() {
        let split = n * 8 / 10;
        let tr_end = purge_end(&sample_ts, split, horizon_ms);
        // Mínimo FÍSICO del entrenador, no un número elegido: un árbol no
        // puede partir con menos de 2·min_child filas, así que por debajo de
        // ese tamaño el GBDT sólo puede devolver la hoja raíz (= la media).
        if tr_end < 2 * min_child {
            eprintln!(
                "❌ tras purgar el solape de horizonte quedan {} muestras de train \
                 (de {}), menos de 2·min_child={}: el horizonte {}ms cubre casi toda \
                 la partición — reduce --horizon-ms o amplía el tape.",
                tr_end, split, 2 * min_child, horizon_ms
            );
            std::process::exit(1);
        }
        println!(
            "   split temporal {}/{} · purga de solape: −{} muestras de train (τ={}ms)",
            tr_end,
            n - split,
            split - tr_end,
            horizon_ms
        );
        (
            feats[..tr_end].to_vec(),
            feats[split..].to_vec(),
            labels[..tr_end].to_vec(),
            labels[split..].to_vec(),
            persist_all[split..].to_vec(),
            sample_ts[split..].to_vec(),
            sample_ts[..tr_end].to_vec(),
        )
    } else {
        // Validación en OTRO fichero. XLIV-13 (recompone FMT): que no haya
        // solape se COMPRUEBA en vez de suponerse. La selección debe empezar
        // después de toda muestra de train, y se purgan las filas de train
        // cuya ventana de etiqueta [t, t+τ] alcanza el inicio de la selección.
        let Some(seleccion) = fuentes.seleccion.as_ref() else {
            eprintln!("❌ --val-in sin admitir");
            std::process::exit(1);
        };
        let s_val = build(seleccion);
        let train = como_intervalos(&feats, &labels, &sample_ts, horizon_ms);
        let seleccion = como_intervalos(&s_val.feats, &s_val.labels, &s_val.ts, horizon_ms);
        let (train, purgadas) = purge_training(train, &seleccion)
            .expect("invalid training/selection separation");
        println!(
            "   validación cruzada por archivo · purga de solape: −{} muestras de train (τ={}ms)",
            purgadas, horizon_ms
        );
        let retained_ts = train.intervals.iter().map(|v| v.start).collect();
        (
            train.features,
            s_val.feats,
            train.labels,
            s_val.labels,
            s_val.persist,
            s_val.ts,
            retained_ts,
        )
    };
    let (tr_feats, va_feats, tr_y, va_y, va_persist, va_ts, tr_ts) = split_samples;
    let early_stop_samples = early_stop_source.as_ref().map(build);
    let (tr_feats, tr_y, tr_ts) = if let Some(early) = early_stop_samples.as_ref() {
        let train = como_intervalos(&tr_feats, &tr_y, &tr_ts, horizon_ms);
        let early_intervals = como_intervalos(&early.feats, &early.labels, &early.ts, horizon_ms);
        let selection = como_intervalos(&va_feats, &va_y, &va_ts, horizon_ms);
        let (train, purged) = prepare_early_stop(train, &early_intervals, &selection)
            .expect("invalid train/early-stop/selection separation");
        let retained_ts = train.intervals.iter().map(|v| v.start).collect();
        println!(
            "   early-stop independiente: {} filas, purga train={} (horizonte={}ms), sha256={}",
            early.labels.len(),
            purged,
            horizon_ms,
            cache_columnas::hex(&early_stop_source.as_ref().unwrap().sha256_original())
        );
        (train.features, train.labels, retained_ts)
    } else {
        (tr_feats, tr_y, tr_ts)
    };
    // Solo train purgado entra en fit_har. El artefacto queda inmutable antes
    // de incorporar el tape de seleccion al historial de diagnostico.
    let har_calibration = if label_mode == "vol" {
        har_history = har_history.and_then(|mut h| {
            h.append_admitted(&fuentes.train, tr_ts.last().copied().unwrap_or(0))?;
            Ok(h)
        });
        let mut calibration = match &har_history {
            Ok(h) => fit_har(h, &tr_ts, &tr_y, horizon_ms),
            Err(reason) => HarCalibration {
                model: None,
                floor: None,
                record: serde_json::json!({"status": "HAR_NO_VIABLE", "reason": reason}),
            },
        };
        calibration.record["train_input"] = serde_json::json!(&in_path);
        calibration.record["history_source"] =
            serde_json::json!("instantanea CintaAdmitida; parseo LE comun; sin reabrir rutas");
        calibration.record["train_admitted_sha256"] =
            serde_json::json!(cache_columnas::hex(&fuentes.train.sha256_original()));
        calibration.record["selection_labels_used_for_fit"] = serde_json::json!(false);
        // El linaje auxiliar identifica los mismos bytes admitidos, aunque la
        // ruta logica cambie despues o solo exista su transporte .bin.tgb.
        calibration.record["train_lineage"] = serde_json::to_value(model_manifest::InputRecord {
            role: "train".into(),
            path: in_path.clone(),
            bytes: fuentes.train.bytes().len() as u64,
            sha256: cache_columnas::hex(&fuentes.train.sha256_original()),
            first_ts_ms: tr_ts.first().copied(),
            last_ts_ms: tr_ts.last().copied(),
            samples: Some(tr_ts.len()),
        })
        .expect("HAR train lineage JSON");
        let bytes = serde_json::to_vec_pretty(&calibration.record).expect("HAR calibration JSON");
        let sha = model_manifest::sha256_hex(&bytes);
        let directory = arg(
            "--har-report-dir",
            &format!("reports/har-rv-qlike/{}/{}", symbol, sha),
        );
        let saved = match write_har_record(&directory, "calibracion.json", &calibration.record) {
            Ok(()) => {
                println!("   HAR calibracion guardada: {directory}/calibracion.json");
                true
            }
            Err(e) => {
                eprintln!("HAR: no se pudo guardar calibracion: {e}");
                false
            }
        };
        println!("   HAR calibracion train congelada (sha256 {sha}; guardada={saved})");
        Some((calibration, directory, sha, saved))
    } else {
        None
    };
    if har_calibration.is_some() {
        har_history = har_history.and_then(|mut h| {
            let end = va_ts.last().copied().unwrap_or(0);
            // El sufijo purgado tambien es pasado legitimo para seleccion.
            h.append_admitted(&fuentes.train, end)?;
            if let Some(cinta) = fuentes.seleccion.as_ref() {
                h.append_admitted(cinta, end)?;
            }
            Ok(h)
        });
    }
    // Fin de la información usada para entrenar y seleccionar: el test
    // posterior debe empezar DESPUÉS (la selección es la parte más tardía).
    let evidence_end = va_ts.last().map_or(0, |&t| t.saturating_add(horizon_ms));
    let split = tr_y.len();
    assert!(
        split >= 2 && split / 10 * 8 >= 2,
        "too few training rows after purging"
    );

    // ── 4. GBDT con early stopping ───────────────────────────────────────
    let is_regression = label_mode != "dir";
    let p_bar = tr_y.iter().sum::<f64>() / tr_y.len() as f64;
    assert!(
        p_bar.is_finite() && (is_regression || (p_bar > 0.0 && p_bar < 1.0)),
        "invalid target mean or single-class training set"
    );
    // Regresión: init = media (predicción cruda); clasificación: logit.
    let init_score = if is_regression {
        p_bar as f32
    } else {
        (p_bar / (1.0 - p_bar)).ln() as f32
    };
    let logloss = |fs: &[Vec<f32>], ys: &[f64], f_pred: &[f64]| -> f64 {
        fs.iter()
            .zip(ys.iter())
            .zip(f_pred.iter())
            .map(|((_, &y), &f)| {
                let p = sigmoid(f).clamp(1e-7, 1.0 - 1e-7);
                -(y * p.ln() + (1.0 - y) * (1.0 - p).ln())
            })
            .sum::<f64>()
            / ys.len() as f64
    };
    // P-1/P-2: MSE sobre la predicción CRUDA (sin sigmoid) — regresión.
    let mse = |ys: &[f64], f_pred: &[f64]| -> f64 {
        ys.iter()
            .zip(f_pred.iter())
            .map(|(&y, &f)| (y - f) * (y - f))
            .sum::<f64>()
            / ys.len().max(1) as f64
    };
    let mut f_train = vec![init_score as f64; split];
    let (stop_feats, stop_y) = early_stop_samples.as_ref()
        .map(|s| (&s.feats, &s.labels)).unwrap_or((&va_feats, &va_y));
    let check_every = if early_stop_samples.is_some() { 1 } else { 5 };
    let mut f_stop = vec![init_score as f64; stop_y.len()];
    let mut trees: Vec<Vec<TreeNode>> = Vec::new();
    let mut early_stop = EarlyStopping::new();
    let mut rng = StdRng::seed_from_u64(42);
    let n_feat = tr_feats[0].len();

    for round in 0..n_rounds {
        // grad/hess por muestra de train (XGBoost-style):
        // clasificación: logloss (y−σ(f), σ(1−σ)); regresión: squared loss
        // (y−f, 1): negative gradient for the additive update.
        let mut data: Vec<(Vec<f32>, f64, f64)> = tr_feats
            .iter()
            .zip(f_train.iter())
            .zip(tr_y.iter())
            .map(|((x, &f), &y)| {
                if is_regression {
                    (x.clone(), regression_residual(f, y), 1.0)
                } else {
                    let p = sigmoid(f);
                    (x.clone(), y - p, p * (1.0 - p))
                }
            })
            .collect();
        // Subconjunto de features (70%) por árbol para diversidad
        let mut all: Vec<usize> = (0..n_feat).collect();
        use rand::seq::SliceRandom;
        all.shuffle(&mut rng);
        let keep = (n_feat * 7 / 10).max(4);
        let feat_subset: Vec<usize> = all[..keep].to_vec();

        let idx: Vec<usize> = (0..split).collect();
        let mut nodes: Vec<TreeNode> = Vec::new();
        // Bootstrap 80% de filas
        let mut boot: Vec<usize> = (0..split / 10 * 8)
            .map(|_| idx[(rng.random::<f64>() * idx.len() as f64) as usize])
            .collect();
        build_tree(
            &mut data,
            &mut boot,
            0,
            max_depth,
            min_child,
            lambda,
            &feat_subset,
            24,
            &mut rng,
            &mut nodes,
        );
        // Aplicar el árbol con shrinkage
        for (i, x) in tr_feats.iter().enumerate() {
            f_train[i] += lr * eval_tree(&nodes, x) as f64;
        }
        for (i, x) in stop_feats.iter().enumerate() {
            f_stop[i] += lr * eval_tree(&nodes, x) as f64;
        }
        trees.push(nodes);
        if round % check_every == 0 || round == n_rounds - 1 {
            let (vl, tl) = if is_regression {
                (mse(stop_y, &f_stop), mse(&tr_y, &f_train))
            } else {
                (
                    logloss(stop_feats, stop_y, &f_stop),
                    logloss(&tr_feats, &tr_y, &f_train),
                )
            };
            let stop = early_stop.observe(vl, trees.len(), check_every, patience);
            println!(
                "   ronda {:3} train {:.6} · early-stop {:.6} {}",
                round,
                tl,
                vl,
                if early_stop.since_improve == 0 { "★" } else { "" }
            );
            if stop {
                println!("   early stopping (paciencia {})", patience);
                break;
            }
        }
    }
    trees.truncate(early_stop.best_rounds.max(1));
    println!("   early-stop best_rounds={} loss={:.6}", early_stop.best_rounds, early_stop.best_loss);
    // El modelo retenido se congela antes de puntuar --val-in por servicio.
    // f_stop puede incluir arboles descartados; nunca se usa para el gate.
    // Con --early-stop-in, va_y no interviene en best_rounds ni paciencia.
    // Sin ese flag se conserva la seleccion legada sobre validacion.
    //
    // XLIV-13 (recompone FMT-190/191): el artefacto que se guardará se
    // serializa AQUÍ y la validación se puntúa con él por la ruta de servicio
    // (`predict_raw_checked`); `best_val` pasa a ser la métrica de ese
    // artefacto. La composición de entrenamiento (`forest_raw`) se conserva
    // como control de paridad.
    let model = serialize_forest(&trees, init_score, lr)
        .expect("invalid forest export; no model has been written");
    // QS-2: los bytes que se escribirán son ÉSTOS; su sha256 identifica al
    // artefacto en el manifiesto y en el libro del holdout.
    let model_bytes = serde_json::to_vec_pretty(&model)
        .expect("invalid forest serialization; no model has been written");
    let model_sha = model_manifest::sha256_hex(&model_bytes);
    let f_val = serving_predictions(&model, &va_feats)
        .expect("invalid validation evidence; no model has been written");
    let paridad = va_feats
        .iter()
        .zip(&f_val)
        .map(|(x, &f)| (forest_raw(init_score, lr, &trees, x) - f).abs())
        .fold(0.0_f64, f64::max);
    println!("   paridad entrenamiento↔servicio en validación: máx |Δ| = {:.3e}", paridad);
    let best_val = if is_regression {
        mse(&va_y, &f_val)
    } else {
        logloss(&va_feats, &va_y, &f_val)
    };

    // Baseline CONSTANTE: tasa base en clasificación, media del train en
    // regresión. En clasificación es el baseline correcto (no hay «clase
    // reciente» que repetir). En regresión es sólo DIAGNÓSTICO — ver abajo.
    let base_pred = vec![init_score as f64; va_y.len()];
    let baseline = if is_regression {
        mse(&va_y, &base_pred)
    } else {
        logloss(&va_feats, &va_y, &base_pred)
    };
    // D-733 — BASELINE DE PERSISTENCIA (sólo regresión): predecir la magnitud
    // futura con la MISMA magnitud medida en la ventana anterior (t−τ, t].
    // σ futura ≈ σ reciente y volumen futuro ≈ volumen reciente son ciertos
    // en cualquier tape, así que la media constante es un rival de paja: un
    // R² de 0,1 contra la media puede esconder un modelo PEOR que repetir el
    // último valor. Se evalúa sobre la MISMA partición de validación y en las
    // MISMAS unidades que la etiqueta.
    let mse_persist = if is_regression { mse(&va_y, &va_persist) } else { f64::NAN };
    let har_selection = har_calibration.as_ref().map(|(calibration, _, _, _)| {
        let report = match &har_history {
            Ok(h) => compare_har(calibration, h, HarEvalSeries {
                ts: &va_ts,
                y: &va_y,
                model: &f_val,
                persist: &va_persist,
                mean: init_score as f64,
                tau: horizon_ms,
            }),
            Err(reason) => serde_json::json!({"status": "HAR_NO_VIABLE", "qlike_evidence": false, "reason": reason}),
        };
        println!("   HAR-RV/QLIKE seleccion: {report}");
        report
    });
    // Diagnóstico: predicciones (sigmoid en clasificación; crudas en regresión)
    let val_preds: Vec<f64> = va_feats
        .iter()
        .zip(f_val.iter())
        .map(|(_, &f)| if is_regression { f } else { sigmoid(f) })
        .collect();
    let mut sorted_p = val_preds.clone();
    sorted_p.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let p_mean = val_preds.iter().sum::<f64>() / val_preds.len() as f64;
    let var = val_preds
        .iter()
        .map(|p| (p - p_mean) * (p - p_mean))
        .sum::<f64>()
        / val_preds.len() as f64;
    let metric_name = if is_regression { "MSE" } else { "logloss" };
    println!("═══ VEREDICTO ═══");
    println!("   val {} modelo: {:.6} · baseline constante: {:.6} (mejora {:+.6})",
             metric_name, best_val, baseline, baseline - best_val);
    println!("   p10={:.6} p50={:.6} p90={:.6} · varianza={:.6}",
             sorted_p[sorted_p.len() / 10], sorted_p[sorted_p.len() / 2],
             sorted_p[9 * sorted_p.len() / 10], var);

    // Margen anti-empate. Clasificación: Δ absoluto de logloss ≥ 0.001.
    // Regresión: el margen se interpreta RELATIVO — fracción del MSE de la
    // PERSISTENCIA que el modelo debe recortar (skill score). Default 0.001 =
    // 0,1% de mejora sobre repetir el valor reciente.
    let gate_margin: f64 = arg("--gate-margin", "0.001").parse().unwrap();
    // LXXXIX (M1/M2 del barrido): margen negativo o no finito relaja el gate
    // de regresión hasta promover modelos PEORES que la persistencia —
    // validado aquí, ANTES de entrenar (antes se tiraba la corrida completa
    // por unwrap y luego dejaba pasar margin<0).
    assert!(
        gate_margin.is_finite() && gate_margin >= 0.0,
        "--gate-margin debe ser finito y >= 0 (recibido: {gate_margin})"
    );
    // D-720 (DÉCIMA OLA · auditoría integral): EL GATE GOBIERNA EL DESTINO.
    //
    // La condición estaba invertida respecto al docstring de este fichero
    // («Sin gate: jamás sobrescribir el modelo vivo con ruido»): al fallar el
    // gate sólo se retornaba SIN `--promote`, es decir, se retornaba cuando el
    // modelo iba al candidato y se CONTINUABA cuando iba al modelo VIVO. Un
    // `train_forest BTCUSDT --promote` sobre un mes sin edge imprimía «el modelo
    // vivo NO se toca» y acto seguido lo sobrescribía; el watcher de god_engine
    // lo hot-swapea en ≤10 s y, desde B3.18, ese modelo decide TODAS las
    // entradas. Ahora un gate no superado nunca escribe el modelo vivo: va al
    // candidato y el proceso termina con código 2 para que cualquier
    // automatización lo detecte.
    //
    // D-733 — para los predictores de regresión (P-1/P-2: _VOL, _VOLU, _OI)
    // el gate YA NO es el R² contra la media: es el SKILL contra la
    // PERSISTENCIA. Batir a la media no demuestra nada en magnitudes que se
    // autocorrelacionan; batir al «mañana será como hoy» sí. El R² contra la
    // media se sigue imprimiendo, etiquetado como informativo.
    //
    // Con --early-stop-in el gate recibe solo el modelo congelado, tras
    // comprobar train < early-stop < seleccion. Sin el flag se conserva
    // el optimismo residual de usar validacion para elegir best_rounds.
    let gate_pass = if is_regression {
        let (r2_mean, skill, pass) = regression_gate(best_val, baseline, mse_persist, gate_margin);
        println!(
            "   MSE persistencia (σ/volumen recientes en (t−τ,t]): {:.6}",
            mse_persist
        );
        println!(
            "   skill vs PERSISTENCIA = {:.4}  ← ESTE es el gate (margen {})",
            skill, gate_margin
        );
        println!(
            "   R² vs media constante = {:.4}  (informativo: NO es el gate)",
            r2_mean
        );
        pass
    } else {
        baseline - best_val >= gate_margin
    };
    let gate_sel = gate_pass;
    if !gate_sel {
        if is_regression {
            println!("🚫 GATE: el modelo no bate a la PERSISTENCIA por el margen {} — \
                      repetir el valor reciente es igual o mejor. El modelo vivo NO se toca.",
                     gate_margin);
        } else {
            println!("🚫 GATE: mejora < margen {} — sin evidencia real. El modelo vivo NO se toca.",
                     gate_margin);
        }
    }
    // XLIV-13 (recompone FMT): el TEST POSTERIOR. Sólo el artefacto congelado
    // llega al test; sus etiquetas no eligen árboles ni ajustan nada. Sin
    // --test-in la evidencia es de selección y NO es apta para promoción
    // (`require_promotion_holdout` ya lo impidió al arrancar).
    let mut har_test = None;
    let (test_ok, test_evidencia): (bool, Option<(f64, f64, model_manifest::InputRecord)>) = if test_in.is_empty() {
        println!("   sin --test-in: evidencia de selección solamente; NO apta para promoción");
        (true, None)
    } else {
        let Some(cinta_test) = fuentes.test.as_ref() else {
            eprintln!("❌ --test-in sin admitir");
            std::process::exit(1);
        };
        let s_test = build(cinta_test);
        let test = como_intervalos(&s_test.feats, &s_test.labels, &s_test.ts, horizon_ms);
        require_later_holdout(&test, evidence_end).expect("invalid final holdout");
        let pred = serving_predictions(&model, &test.features).expect("invalid test predictions");
        let constante = vec![init_score as f64; test.labels.len()];
        if let Some((calibration, _, _, _)) = &har_calibration {
            har_history = har_history.and_then(|mut h| {
                let end = s_test.ts.last().copied().unwrap_or(0);
                // Recupera tambien los sufijos disponibles que ahora son pasado.
                h.append_admitted(&fuentes.train, end)?;
                if let Some(cinta) = fuentes.seleccion.as_ref() { h.append_admitted(cinta, end)?; }
                h.append_admitted(cinta_test, end)?;
                Ok(h)
            });
            let report = match &har_history {
                Ok(h) => compare_har(calibration, h, HarEvalSeries {
                    ts: &s_test.ts,
                    y: &test.labels,
                    model: &pred,
                    persist: &s_test.persist,
                    mean: init_score as f64,
                    tau: horizon_ms,
                }),
                Err(reason) => serde_json::json!({"status": "HAR_NO_VIABLE", "qlike_evidence": false, "reason": reason}),
            };
            println!("   HAR-RV/QLIKE test posterior: {report}");
            har_test = Some(report);
        }
        let (pass, perdida, base) = if is_regression {
            let mse_modelo = mse(&test.labels, &pred);
            let mse_pers = mse(&test.labels, &s_test.persist);
            let (r2_media, skill, pass) = regression_gate(
                mse_modelo,
                mse(&test.labels, &constante),
                mse_pers,
                gate_margin,
            );
            println!(
                "   test posterior (n={}): MSE {:.6} · skill vs PERSISTENCIA {:.4} · R² vs media {:.4} (informativo)",
                test.labels.len(), mse_modelo, skill, r2_media
            );
            (pass, mse_modelo, mse_pers)
        } else {
            let ll = logloss(&test.features, &test.labels, &pred);
            let base = logloss(&test.features, &test.labels, &constante);
            println!(
                "   test posterior (n={}): logloss {:.6} · baseline {:.6} (mejora {:+.6})",
                test.labels.len(), ll, base, base - ll
            );
            (passes_loss_gate(false, ll, base, gate_margin), ll, base)
        };
        if !pass {
            println!("🚫 GATE (test posterior): sin mejora ≥ {} fuera de la selección. El modelo vivo NO se toca.",
                     gate_margin);
        }
        let registro = model_manifest::input_record("test", &test_in, &s_test.ts)
            .expect("cannot hash --test-in for the lineage manifest");
        (pass, Some((perdida, base, registro)))
    };
    let gate_ok = gate_sel && test_ok;
    // QS-2 — UN HOLDOUT FINAL JUZGA UNA SOLA DECISIÓN. Si este test ya juzgó
    // OTRO artefacto del mismo símbolo y objetivo (otro τ, otros
    // hiperparámetros, otro tape de train), lo visto en él ya informó la
    // elección y deja de ser fuera de muestra para promover. Re-juzgar el
    // MISMO artefacto (mismo sha256; el entrenador es determinista) no añade
    // selección. Un libro ilegible no puede certificar frescura: bloquea.
    let mut holdout_prior = 0usize;
    let mut holdout_reason: Option<String> = None;
    let mut holdout_block = false;
    if let Some((_, _, registro)) = &test_evidencia {
        match model_manifest::read_ledger(std::path::Path::new(&holdout_ledger)) {
            Ok(libro) => {
                holdout_prior = model_manifest::prior_other_artifacts(
                    &libro, &symbol, &label_mode, &registro.sha256, &model_sha,
                );
                if holdout_prior > 0 {
                    println!(
                        "   ⚠️ holdout ya usado: este test juzgó {} artefacto(s) distinto(s) de {} [{}]",
                        holdout_prior, symbol, label_mode
                    );
                }
                if promote && gate_ok {
                    match model_manifest::check_holdout_fresh(holdout_prior, &test_reuse) {
                        Ok(motivo) => holdout_reason = motivo,
                        Err(e) => {
                            println!("🚫 HOLDOUT: {e}");
                            holdout_block = true;
                        }
                    }
                }
            }
            Err(e) => {
                println!("🚫 HOLDOUT: libro ilegible ({e}); no se puede certificar que el test esté fresco");
                holdout_block = promote && gate_ok;
            }
        }
    }
    let promote = promote && gate_ok && !holdout_block;

    // ── 5. Serializar al formato NanoForestData ──────────────────────────
    // Already serialized and validated before the gate above.
    // P-1/P-2: el sufijo del modelo declara su objetivo — {SYM}_MOTOR
    // (dirección), {SYM}_VOL (σ(τ) realizada del horizonte, % de precio),
    // {SYM}_VOLU (nocional negociado del horizonte, en quote). El watcher del
    // host auto-carga cualquier models/{KEY}.json bajo esa key; los
    // predictores de regresión se sirven con predict_raw (sin sigmoid).
    //
    // D-731 — CAMBIO DE ESCALA: los {SYM}_VOL anteriores a esta revisión
    // están en la escala vieja (σ dividida por sqrt(n_ticks)) y quedan
    // OBSOLETOS: su `init_score` y sus hojas no son comparables con los
    // nuevos. Hay que reentrenar antes de promover. El VOL-BRAKE del host usa
    // el COCIENTE pronóstico/base, que es invariante de escala, así que el
    // serving no necesita cambios — pero un models/ con modelos de las dos
    // épocas mezcladas compararía magnitudes distintas entre símbolos.
    //
    // D-732 — {SYM}_VOLU no tiene HOY ningún consumidor en el host (sólo
    // _VOL alimenta el freno de sizing). Se conserva porque el nocional del
    // horizonte SÍ es una magnitud física medible en el tape, pero mientras
    // nadie lo lea sigue siendo un predictor sin uso: pasar su gate no
    // cambia ninguna decisión del motor.
    let suffix = match label_mode.as_str() {
        "vol" => "_VOL",
        "volu" => "_VOLU",
        "oi" => "_OI",
        _ => "_MOTOR",
    };
    let out = if promote {
        format!("models/{}{}.json", symbol, suffix)
    } else {
        format!("models/{}{}_CANDIDATE.json", symbol, suffix)
    };
    // FMT-190/191: `model` es el artefacto EXACTO que puntuaron la validación
    // y el test (serializado antes del gate, XLIV-13).
    // LXXXIX (H1 del barrido): escritura ATÓMICA tmp+rename+sync — File::create
    // TRUNCABA el modelo vivo a 0 bytes ANTES de serializar; un crash a mitad
    // dejaba el artefacto destruido y el host arrancaba sin modelo. El rename
    // atómico garantiza que en disco hay siempre el modelo completo viejo o
    // el nuevo, nunca un truncado. `write_model_with_manifest` conserva esa
    // escritura (tmp + sync_all + rename, crea `models/`) y la extiende al
    // manifiesto.
    //
    // QS-2 — LINAJE: junto al modelo, `{KEY}.manifest` con el sha256 de
    // estos bytes, los tapes (sha256 + cobertura), los argumentos, el commit
    // y la evidencia del gate. Ver `quantum_engine::model_manifest`.
    let mut entradas = Vec::new();
    if val_in.is_empty() {
        entradas.push(
            model_manifest::input_record("train+selection", &in_path, &sample_ts)
                .expect("cannot hash --in for the lineage manifest"),
        );
    } else {
        entradas.push(
            model_manifest::input_record("train", &in_path, &sample_ts)
                .expect("cannot hash --in for the lineage manifest"),
        );
        entradas.push(
            model_manifest::input_record("selection", &val_in, &va_ts)
                .expect("cannot hash --val-in for the lineage manifest"),
        );
    }
    let (test_perdida, test_base) = match &test_evidencia {
        Some((l, b, registro)) => {
            entradas.push(registro.clone());
            (Some(*l), Some(*b))
        }
        None => (None, None),
    };
    let version = model_manifest::code_version();
    let manifiesto = model_manifest::ModelManifest {
        schema: model_manifest::MANIFEST_SCHEMA,
        model_file: String::new(),
        model_sha256: String::new(),
        symbol: symbol.clone(),
        label_mode: label_mode.clone(),
        horizon_ms,
        feature_dim: n_feat,
        n_trees: trees.len(),
        trainer: "train_forest".into(),
        trainer_args: args.clone(),
        code_version: version.clone(),
        created_utc: chrono::Utc::now().to_rfc3339(),
        inputs: entradas,
        train_rows: tr_y.len(),
        selection_rows: va_y.len(),
        evidence_end_ms: evidence_end,
        gate: model_manifest::GateRecord {
            metric: metric_name.into(),
            margin: gate_margin,
            selection_loss: best_val,
            selection_baseline: if is_regression { mse_persist } else { baseline },
            selection_pass: gate_sel,
            test_loss: test_perdida,
            test_baseline: test_base,
            test_pass: test_evidencia.as_ref().map(|_| test_ok),
        },
        holdout_prior_other_artifacts: holdout_prior,
        holdout_reuse_reason: holdout_reason,
        promoted: promote,
    };
    let escrito = model_manifest::write_model_with_manifest(
        std::path::Path::new(&out),
        &model_bytes,
        manifiesto,
    )
    .expect("cannot write the model and its lineage manifest");
    println!(
        "🧾 linaje: {} (sha256 {}…, código {})",
        model_manifest::manifest_path_for(std::path::Path::new(&out)).display(),
        &escrito.model_sha256[..12],
        version.as_deref().unwrap_or("desconocido")
    );
    if let Some((_, directory, calibration_sha, calibration_saved)) = &har_calibration {
        let report = serde_json::json!({
            "schema": "har-rv-qlike-v1", "model_sha256": &model_sha,
            "calibration_sha256": calibration_sha, "calibration_written": calibration_saved,
            "code_version": &version,
            "selection": har_selection, "test": har_test,
            "gate_uses_qlike": false, "gate_selection_pass": gate_sel,
            "gate_test_pass": test_evidencia.as_ref().map(|_| test_ok),
        });
        let bytes = serde_json::to_vec_pretty(&report).expect("HAR comparison JSON");
        let sha = model_manifest::sha256_hex(&bytes);
        let name = format!("comparacion.{model_sha}.{sha}.json");
        if let Err(e) = write_har_record(directory, &name, &report) {
            eprintln!("HAR: no se pudo guardar comparacion: {e}");
        } else {
            println!("   HAR-RV/QLIKE: {directory}/{name}");
        }
    }
    // QS-2: toda consulta del test queda anotada, pase o no el gate y se
    // promueva o no: un intento fallido también es información usada.
    if let Some((_, _, registro)) = &test_evidencia {
        let uso = model_manifest::HoldoutUse {
            utc: escrito.created_utc.clone(),
            symbol: symbol.clone(),
            label_mode: label_mode.clone(),
            test_sha256: registro.sha256.clone(),
            model_sha256: escrito.model_sha256.clone(),
            test_pass: test_ok,
            promoted: promote,
            code_version: version.clone(),
        };
        model_manifest::append_ledger(std::path::Path::new(&holdout_ledger), &uso)
            .expect("cannot append to the holdout ledger");
        println!("   consulta del holdout anotada en {}", holdout_ledger);
    }
    println!(
        "💾 {} ({} árboles, init {:.4}){}",
        out,
        trees.len(),
        init_score,
        if !gate_ok {
            " — [gate NO superado, revisar antes de promover]"
        } else {
            ""
        }
    );
    if !promote && gate_ok {
        println!("   candidato solamente; promover exige --promote --test-in <archivo posterior independiente>. No reutilizar el test para ajustar hiperparámetros.");
    }
    // D-720: sin evidencia, salida distinta de cero — el fichero escrito es el
    // candidato, no el vivo.
    if !gate_ok {
        std::process::exit(2);
    }
    // QS-2: gate superado pero holdout ya usado sin motivo ⇒ candidato y
    // código 3 (distinto de «sin evidencia») para la automatización.
    if holdout_block {
        std::process::exit(3);
    }
}

/// Puntuación CRUDA del bosque sobre un vector: `init_score + Σ lr·hoja(árbol)`.
///
/// Es la MISMA composición que sirve `NanoForest::predict_raw` en vivo, y toma
/// los árboles por rodaja: evaluarla sobre el bosque ya TRUNCADO devuelve
/// exactamente lo que se escribe en el JSON, sin arrastrar los árboles que el
/// early stopping descartó.
fn forest_raw(init_score: f32, lr: f64, trees: &[Vec<TreeNode>], x: &[f32]) -> f64 {
    trees
        .iter()
        .fold(init_score as f64, |acc, t| acc + lr * eval_tree(t, x) as f64)
}

fn eval_tree(nodes: &[TreeNode], x: &[f32]) -> f32 {
    let mut cur = 0usize;
    loop {
        let nd = &nodes[cur];
        if nd.left == -1 && nd.right == -1 {
            return nd.value;
        }
        let v = x.get(nd.feature as usize).copied().unwrap_or(0.0);
        cur = if v <= nd.threshold {
            nd.left as usize
        } else {
            nd.right as usize
        };
    }
}

#[cfg(test)]
mod contract_tests {
    use super::*;
    use god_engine_core::ml_inference::NanoForest;

    #[test]
    fn full_population_split_gain_matches_direct_sum() {
        let data: Vec<_> = (0..4096)
            .map(|i| {
                (
                    vec![(i >= 2048) as u8 as f32],
                    if i < 2048 { 1.0 } else { -1.0 },
                    1.0,
                )
            })
            .collect();
        let idx: Vec<_> = (0..4096).collect();
        let probe: Vec<_> = (0..4096).step_by(2).collect();
        let threshold = data[probe[0]].0[0];
        let gain = candidate_gains(&data, &idx, 0, &[threshold], 1.0, 40)[0].unwrap();
        let expected = 2.0 * 2048.0f64.powi(2) / (2048.0 + 1.0 + 1e-12);
        assert!(
            (gain - expected).abs() < 1e-8,
            "gain={gain} expected={expected}"
        );
    }

    #[test]
    fn child_eligibility_uses_full_population_not_probe() {
        let data: Vec<_> = (0..4096)
            .map(|i| {
                (
                    vec![(i >= 80) as u8 as f32],
                    if i < 80 { 1.0 } else { -1.0 },
                    1.0,
                )
            })
            .collect();
        let idx: Vec<_> = (0..4096).collect();
        let probe: Vec<_> = (0..4096).step_by(4).collect();
        let threshold = data[probe[0]].0[0];
        assert!(candidate_gains(&data, &idx, 0, &[threshold], 1.0, 40)[0].is_some());
    }

    #[test]
    fn sampling_stride_respects_budget_on_long_span() {
        assert_eq!(effective_stride(10_000_000, 1000, 100).unwrap(), 100_001);
    }

    #[test]
    fn sampling_stride_never_densifies_requested_grid() {
        assert_eq!(effective_stride(1_000_000, 50_000, 1000).unwrap(), 50_000);
    }

    #[test]
    fn split_gains_match_bruteforce_with_bootstrap_duplicates() {
        let data = vec![
            (vec![0.0], 2.0, 0.5),
            (vec![1.0], -3.0, 1.0),
            (vec![2.0], 0.25, 0.125),
        ];
        let idx = vec![0, 0, 1, 2, 2, 2];
        let thresholds = [1.5, 0.5, 1.5, -1.0, 2.0];
        let gains = candidate_gains(&data, &idx, 0, &thresholds, 0.3, 1);
        let sum = |rows: &[usize]| -> (f64, f64) {
            (
                rows.iter().map(|&i| data[i].1).sum(),
                rows.iter().map(|&i| data[i].2).sum(),
            )
        };
        let score = |(g, h): (f64, f64)| g * g / (h + 0.3 + 1e-12);
        for (&threshold, gain) in thresholds.iter().zip(gains) {
            let (left, right): (Vec<_>, Vec<_>) = idx
                .iter()
                .copied()
                .partition(|&i| data[i].0[0] <= threshold);
            if left.is_empty() || right.is_empty() {
                assert!(gain.is_none());
            } else {
                assert!(
                    (gain.unwrap() - (score(sum(&left)) + score(sum(&right)) - score(sum(&idx))))
                        .abs()
                        < 1e-12
                );
            }
        }
    }

    #[test]
    fn invalid_split_statistics_cannot_be_selected() {
        for (gradient, hessian) in [(f64::NAN, 1.0), (1.0, -1.0), (f64::INFINITY, 1.0)] {
            let data = vec![(vec![0.0], gradient, hessian), (vec![1.0], -1.0, 1.0)];
            assert_eq!(
                candidate_gains(&data, &[0, 1], 0, &[0.5], 1.0, 1),
                vec![None]
            );
        }
    }

    #[test]
    fn large_node_tree_uses_valid_full_population_children() {
        let mut data: Vec<_> = (0..4096)
            .map(|i| {
                (
                    vec![(i >= 2048) as u8 as f32],
                    if i < 2048 { 1.0 } else { -1.0 },
                    1.0,
                )
            })
            .collect();
        let mut nodes = Vec::new();
        build_tree(
            &mut data,
            &mut (0..4096).collect(),
            0,
            1,
            40,
            1.0,
            &[0],
            24,
            &mut StdRng::seed_from_u64(42),
            &mut nodes,
        );
        assert_eq!(nodes.len(), 3);
        assert!(eval_tree(&nodes, &[0.0]) > 0.99);
        assert!(eval_tree(&nodes, &[1.0]) < -0.99);
    }

    #[test]
    fn sampling_budget_caps_attempts_inclusive_end_and_irregular_ticks() {
        for span in [0, 1, 9, 100, 1000] {
            for cap in [1, 2, 7, 100] {
                for stride in [1, 3, 50] {
                    let mut plan = SamplingBudget::new(span, stride, cap).unwrap();
                    let selected: Vec<_> = (0..=span).filter(|&t| plan.reserve(t)).collect();
                    assert!(selected.len() <= cap);
                    assert!(selected.windows(2).all(|w| w[1] - w[0] >= stride));
                }
            }
        }
        let mut plan = SamplingBudget::new(1000, 50, 3).unwrap();
        for t in [10, 10, 11, 500, 999, 5000, 10000] {
            plan.reserve(t);
        }
        assert_eq!(plan.attempts, 3); // Counter applies even beyond the planned span.
    }

    #[test]
    fn sampling_budget_rejects_zero_and_handles_timestamp_overflow() {
        assert!(SamplingBudget::new(1000, 0, 10).is_err());
        assert!(SamplingBudget::new(1000, 10, 0).is_err());
        let mut plan = SamplingBudget::new(u64::MAX, 1, 1).unwrap();
        assert!(plan.reserve(u64::MAX));
        assert!(!plan.reserve(u64::MAX));
        let mut plan = SamplingBudget::new(100, 10, 10).unwrap();
        assert!(plan.reserve(u64::MAX - 1));
        assert!(!plan.reserve(u64::MAX));
    }

    fn sample_set(windows: &[(u64, u64)]) -> TrainingSamples {
        let mut samples = TrainingSamples::default();
        for &(start, end) in windows {
            samples.push(vec![start as f32], start as f64, start, end);
        }
        samples
    }

    #[test]
    fn xliv_como_intervalos_compone_los_contratos_de_seleccion_y_test() {
        // Train t = 1000, 2000, 3000 con τ = 1500: la ventana [3000, 4500]
        // alcanza una selección que empieza en 4000 ⇒ se purga esa fila.
        let tr = como_intervalos(
            &[vec![0.1], vec![0.2], vec![0.3]],
            &[1.0, 0.0, 1.0],
            &[1000, 2000, 3000],
            1500,
        );
        let sel = como_intervalos(&[vec![0.4], vec![0.5]], &[0.0, 1.0], &[4000, 5000], 1500);
        let (kept, purged) = purge_training(tr, &sel).unwrap();
        assert_eq!((kept.labels.len(), purged), (2, 1));
        // La selección termina en 5000 + 1500: un test que empiece en 6500
        // (frontera cerrada) solapa; en 6501 es posterior.
        let fin = 5000 + 1500;
        let solapa = como_intervalos(&[vec![0.6]], &[1.0], &[6500], 1500);
        let posterior = como_intervalos(&[vec![0.6]], &[1.0], &[6501], 1500);
        assert!(require_later_holdout(&solapa, fin).is_err());
        assert!(require_later_holdout(&posterior, fin).is_ok());
        // Una selección que empieza antes que el último train es inválida.
        let tr2 = como_intervalos(&[vec![0.1], vec![0.2]], &[1.0, 0.0], &[1000, 5000], 10);
        let sel2 = como_intervalos(&[vec![0.3]], &[1.0], &[4000], 10);
        assert!(purge_training(tr2, &sel2).is_err());
    }

    #[test]
    fn temporal_purge_uses_closed_information_intervals() {
        let train = sample_set(&[(1, 3), (4, 10), (7, 11)]);
        let validation = sample_set(&[(10, 12), (13, 14)]);
        let (kept, purged) = purge_training(train, &validation).unwrap();
        assert_eq!(purged, 2); // equality at 10 also overlaps
        assert_eq!(kept.labels, vec![1.0]);
        assert_eq!(kept.features, vec![vec![1.0]]);
    }

    #[test]
    fn purge_handles_nonmonotone_label_ends_without_losing_alignment() {
        let train = sample_set(&[(1, 100), (2, 3), (4, 50), (5, 6)]);
        let validation = sample_set(&[(10, 12)]);
        let (kept, purged) = purge_training(train, &validation).unwrap();
        assert_eq!(purged, 2);
        assert_eq!(kept.labels, vec![2.0, 5.0]);
        assert_eq!(kept.features, vec![vec![2.0], vec![5.0]]);
        assert_eq!(
            kept.intervals.iter().map(|v| v.end).collect::<Vec<_>>(),
            vec![3, 6]
        );
    }

    #[test]
    fn temporal_split_purges_only_training_and_keeps_selection_intact() {
        let mut training = sample_set(&[(1, 2), (3, 6), (5, 8), (7, 9)]);
        let selection = training.split_at(2);
        let (kept, purged) = purge_training(training, &selection).unwrap();
        assert_eq!(kept.labels, vec![1.0]);
        assert_eq!(purged, 1);
        assert_eq!(selection.labels, vec![5.0, 7.0]);
    }

    #[test]
    fn reversed_identical_and_exhausted_training_windows_fail_closed() {
        assert!(purge_training(sample_set(&[(10, 11)]), &sample_set(&[(10, 11)])).is_err());
        assert!(purge_training(sample_set(&[(10, 11)]), &sample_set(&[(1, 3)])).is_err());
        assert!(purge_training(sample_set(&[(1, 11)]), &sample_set(&[(10, 12)])).is_err());
    }

    #[test]
    fn sample_contract_rejects_misalignment_nonfinite_and_bad_intervals() {
        for windows in [
            vec![],
            vec![(0, 1)],
            vec![(2, 1)],
            vec![(1, 2), (1, 3)],
            vec![(2, 3), (1, 2)],
        ] {
            assert!(sample_set(&windows).validate().is_err());
        }
        let mut bad = sample_set(&[(1, 2)]);
        bad.labels.clear();
        assert!(bad.validate().is_err());
        let mut bad = sample_set(&[(1, 2)]);
        bad.features[0][0] = f32::NAN;
        assert!(bad.validate().is_err());
        let mut bad = sample_set(&[(1, 2)]);
        bad.labels[0] = f64::INFINITY;
        assert!(bad.validate().is_err());
        let mut bad = sample_set(&[(1, 2), (3, 4)]);
        bad.features[1].push(0.0);
        assert!(bad.validate().is_err());
    }

    #[test]
    fn holdout_must_follow_the_latest_label_end_not_the_latest_start() {
        let evidence = sample_set(&[(1, 100), (5, 6)]);
        assert!(require_later_holdout(&sample_set(&[(99, 101)]), evidence.max_end()).is_err());
        assert!(require_later_holdout(&sample_set(&[(100, 101)]), evidence.max_end()).is_err());
        assert!(require_later_holdout(&sample_set(&[(101, 102)]), evidence.max_end()).is_ok());
    }

    #[test]
    fn promotion_requires_a_separate_test_but_research_candidate_does_not() {
        assert!(require_promotion_holdout(true, "").is_err());
        assert!(require_promotion_holdout(false, "").is_ok());
        assert!(require_promotion_holdout(true, "later.bin").is_ok());
    }

    #[test]
    fn loss_gate_rejects_invalid_and_zero_baseline_evidence() {
        for regression in [false, true] {
            for (loss, base, margin) in [
                (f64::NAN, 1.0, 0.1),
                (0.0, f64::INFINITY, 0.1),
                (-1.0, 1.0, 0.1),
                (0.1, 1.0, f64::NAN),
                (0.1, 1.0, -0.1),
            ] {
                assert!(!passes_loss_gate(regression, loss, base, margin));
            }
        }
        assert!(!passes_loss_gate(true, 0.0, 0.0, 0.0));
        assert!(passes_loss_gate(true, 0.8e-20, 1e-20, 0.1));
        assert!(!passes_loss_gate(true, 1.2e-20, 1e-20, 0.1));
        assert!(passes_loss_gate(false, 0.4, 0.5, 0.01));
        assert!(!passes_loss_gate(false, 0.51, 0.5, 0.01));
    }

    #[test]
    fn tick_contract_rejects_bad_chronology_and_nonfinite_market_values() {
        let good = BinTick {
            ts: 10,
            bid: 100.0,
            ask: 101.0,
            bq: 1.0,
            aq: 2.0,
        };
        assert!(valid_tick(&good));
        assert!(validate_tick_order(&[good, good]).is_ok());
        assert!(validate_tick_order(&[good, BinTick { ts: 9, ..good }]).is_err());
        assert!(validate_tick_order(&[BinTick { ts: 0, ..good }]).is_err());
        for bad in [
            BinTick {
                bid: f64::NAN,
                ..good
            },
            BinTick { ask: 99.0, ..good },
            BinTick { bq: -1.0, ..good },
            BinTick {
                aq: f64::INFINITY,
                ..good
            },
        ] {
            assert!(!valid_tick(&bad));
        }
    }

    #[test]
    fn serialized_json_and_binary_preserve_additive_predictions() {
        let trees = vec![stump(1.0, -1.0), stump(4.0, -5.0)];
        for lr in [0.05, 0.1, 0.5, 1.0] {
            let model = serialize_forest(&trees, 0.25, lr).unwrap();
            let json: NanoForestData =
                serde_json::from_slice(&serde_json::to_vec(&model).unwrap()).unwrap();
            let binary: NanoForestData =
                bincode::deserialize(&bincode::serialize(&model).unwrap()).unwrap();
            for data in [json, binary] {
                let forest = NanoForest::from_data(data).unwrap();
                for x in [[0.1], [0.9]] {
                    let expected =
                        0.25 + lr * trees.iter().map(|t| eval_tree(t, &x) as f64).sum::<f64>();
                    let (raw, p) = forest.predict_raw_checked(&x).unwrap();
                    assert!((raw as f64 - expected).abs() < 2e-6);
                    assert!((p as f64 - sigmoid(expected)).abs() < 2e-6);
                }
            }
        }
    }
    #[test]
    fn invalid_exports_are_rejected_without_writing_a_model() {
        for lr in [0.0, -1.0, f64::NAN, f64::INFINITY] {
            assert!(serialize_forest(&[stump(1.0, -1.0)], 0.0, lr).is_err());
        }
        assert!(serialize_forest(&[stump(f32::MAX, 0.0)], 0.0, 2.0).is_err());
        assert!(serialize_forest(&[], 0.0, 1.0).is_err());
        let mut invalid = stump(1.0, -1.0);
        invalid[0].left = 3;
        assert!(serialize_forest(&[invalid], 0.0, 1.0).is_err());
    }
    #[test]
    fn retained_ensemble_predictions_exclude_discarded_rounds() {
        let mut trees = vec![stump(1.0, -1.0), stump(4.0, -5.0)];
        trees.truncate(1);
        let model = serialize_forest(&trees, 0.25, 0.1).unwrap();
        let p = serving_predictions(&model, &[vec![0.1], vec![0.9]]).unwrap();
        assert!((p[0] - 0.35).abs() < 1e-6);
        assert!((p[1] - 0.15).abs() < 1e-6);
        assert!(serving_predictions(&model, &[vec![f32::NAN]]).is_err());
    }
    #[test]
    fn regression_residual_is_the_negative_gradient_in_both_directions() {
        for (prediction, target) in [(0.0, 2.0), (2.0, -3.0)] {
            let residual = regression_residual(prediction, target);
            let epsilon = 1e-5;
            let loss = |p: f64| 0.5 * (p - target).powi(2);
            let gradient =
                (loss(prediction + epsilon) - loss(prediction - epsilon)) / (2.0 * epsilon);
            assert!((residual + gradient).abs() < 1e-8);
        }
    }

    fn stump(left: f32, right: f32) -> Vec<TreeNode> {
        vec![
            TreeNode {
                feature: 0,
                threshold: 0.5,
                left: 1,
                right: 2,
                value: 0.0,
            },
            TreeNode {
                feature: -1,
                threshold: 0.0,
                left: -1,
                right: -1,
                value: left,
            },
            TreeNode {
                feature: -1,
                threshold: 0.0,
                left: -1,
                right: -1,
                value: right,
            },
        ]
    }
    #[test]
    fn exported_forest_keeps_learning_rate() {
        let trees = vec![stump(2.0, -3.0)];
        let forest = NanoForest::from_data(serialize_forest(&trees, 0.25, 0.1).unwrap()).unwrap();
        for x in [[0.1], [0.9]] {
            let expected = 0.25 + 0.1 * eval_tree(&trees[0], &x) as f64;
            assert!((forest.predict_raw(&x).0 as f64 - expected).abs() < 1e-6);
        }
    }
    #[test]
    fn exported_forest_keeps_each_tree_children() {
        let trees = vec![stump(1.0, -1.0), stump(4.0, -5.0)];
        let forest = NanoForest::from_data(serialize_forest(&trees, 0.0, 1.0).unwrap()).unwrap();
        for x in [[0.1], [0.9]] {
            let expected: f32 = trees.iter().map(|t| eval_tree(t, &x)).sum();
            assert_eq!(forest.predict_raw(&x).0, expected);
        }
    }
    #[test]
    fn regression_newton_step_reduces_squared_error() {
        let prediction = 0.0;
        let target = 2.0;
        let mut data = vec![(vec![0.0], regression_residual(prediction, target), 1.0)];
        let mut nodes = Vec::new();
        build_tree(
            &mut data,
            &mut vec![0],
            0,
            0,
            1,
            1.0,
            &[0],
            2,
            &mut StdRng::seed_from_u64(1),
            &mut nodes,
        );
        let updated = prediction + 0.1 * eval_tree(&nodes, &[0.0]) as f64;
        assert!((target - updated).powi(2) < (target - prediction).powi(2));
    }
}

// ── PRUEBAS ──────────────────────────────────────────────────────────────
//
// Cada prueba fija un invariante que el código ANTERIOR violaba: la etiqueta
// de volatilidad dividida por la intensidad de ticks, la etiqueta `volu`
// construida sobre un par fabricado, el gate contra la media constante y el
// solape de horizonte entre train y validación.
#[cfg(test)]
mod tests {
    use super::*;

    /// Reproduce la fabricación de `binance_vision_sync`: un aggTrade de
    /// `qty` al precio `price` se escribe como (bid_qty, ask_qty) con la
    /// base `max(0,25·qty; 0,1)` en el lado pasivo.
    fn agg_tick(ts: u64, price: f64, qty: f64, is_maker: bool) -> BinTick {
        let base = (qty * 0.25).max(0.1);
        let (bq, aq) = if is_maker { (base, qty + base) } else { (qty + base, base) };
        let half = price * 0.00005;
        BinTick { ts, bid: price - half, ask: price + half, bq, aq }
    }

    /// D-731 — la σ del horizonte NO puede depender de cuántos ticks lo
    /// partan. Dos caminos con la misma varianza realizada acumulada sobre la
    /// MISMA ventana deben etiquetarse igual. La fórmula vieja
    /// `sqrt(Σr²/n_ticks)·100` daba a la ruta de 8 pasos la MITAD de σ que a
    /// la de 2 pasos: un mercado más activo parecía más tranquilo.
    #[test]
    fn vol_es_sigma_del_horizonte_no_media_por_tick() {
        // Camino A: 2 retornos de +r. Camino B: 8 retornos de +r/2.
        // Sigma r^2 (A) = 2r^2;  (B) = 8*(r/2)^2 = 2r^2. Misma varianza realizada.
        let p0 = 100.0f64;
        let r = 0.002f64;
        let mut a = Vec::new();
        let mut p = p0;
        for k in 0..2u64 {
            p *= 1.0 + r;
            a.push(agg_tick(1_000 + k * 1_000, p, 1.0, false));
        }
        let mut b = Vec::new();
        let mut q = p0;
        for k in 0..8u64 {
            q *= 1.0 + r / 2.0;
            b.push(agg_tick(1_000 + k * 250, q, 1.0, false));
        }
        let sa = accumulate_window(&a, p0);
        let sb = accumulate_window(&b, p0);
        assert_eq!(sa.n_ret, 2);
        assert_eq!(sb.n_ret, 8);

        let vol_a = realized_vol_pct(sa.sum_sq);
        let vol_b = realized_vol_pct(sb.sum_sq);
        // Tolerancia: los retornos compuestos no son exactamente r, así que
        // se comparan en relativo con holgura de capitalización (2e-3).
        assert!(
            ((vol_a - vol_b) / vol_a).abs() < 2e-3,
            "sigma(tau) debe ser independiente de la intensidad de ticks: A={vol_a} B={vol_b}"
        );

        // Y la fórmula VIEJA (media por tick) las separaba un factor ~2.
        let viejo_a = (sa.sum_sq / sa.n_ret as f64).sqrt() * 100.0;
        let viejo_b = (sb.sum_sq / sb.n_ret as f64).sqrt() * 100.0;
        assert!(
            viejo_a / viejo_b > 1.9,
            "la etiqueta vieja hacía parecer MENOS volátil al tramo más activo: {viejo_a} vs {viejo_b}"
        );
    }

    /// D-732 — `|bid_qty − ask_qty|` recupera el `qty` EXACTO del aggTrade,
    /// también cuando el suelo absoluto de la base (0,1) domina.
    #[test]
    fn agg_trade_qty_recupera_la_cantidad_exacta() {
        for &q in &[0.05f64, 0.4, 1.0, 37.25, 1_234.5] {
            let tk = agg_tick(0, 100.0, q, false);
            assert!(
                (agg_trade_qty(tk.bq, tk.aq) - q).abs() <= q * 1e-12,
                "qty={q} no recuperado: {}",
                agg_trade_qty(tk.bq, tk.aq)
            );
            let tk2 = agg_tick(0, 100.0, q, true);
            assert!((agg_trade_qty(tk2.bq, tk2.aq) - q).abs() <= q * 1e-12);
            assert!(is_aggtrades_pair(tk.bq, tk.aq));
        }
        // Tape sintético de klines (parquet_to_bin): los dos lados son
        // fracciones del volumen de vela, no un aggTrade.
        assert!(!is_aggtrades_pair(2.5, 2.5), "bq==aq no es un aggTrade");
        assert!(
            !is_aggtrades_pair(1.15 * 3.0, 0.85 * 3.0),
            "el par de klines no cumple la identidad del productor"
        );
    }

    /// D-732 — `volu` es el NOCIONAL negociado de la ventana, no la media
    /// inflada de `(bq + aq)`. La etiqueta vieja valía ≈1,5× la cantidad
    /// MEDIA por tick (sin precio y con suelo artificial): ni volumen, ni
    /// profundidad, ni nocional.
    #[test]
    fn volu_es_nocional_negociado_no_media_inflada() {
        let qtys = [2.0f64, 5.0, 0.05, 11.0];
        let precios = [100.0f64, 100.5, 99.5, 101.0];
        let ticks: Vec<BinTick> = qtys
            .iter()
            .zip(precios.iter())
            .enumerate()
            .map(|(k, (&q, &p))| agg_tick(1_000 + k as u64 * 100, p, q, k % 2 == 0))
            .collect();
        let st = accumulate_window(&ticks, precios[0]);

        let esperado: f64 = qtys
            .iter()
            .zip(ticks.iter())
            .map(|(&q, tk)| q * ((tk.bid + tk.ask) / 2.0))
            .sum();
        assert!(
            (st.notional - esperado).abs() <= esperado * 1e-12,
            "nocional {} != suma qty*precio {}",
            st.notional,
            esperado
        );

        // La etiqueta vieja: media de (bq+aq) por tick — sin precio, con la
        // base fabricada dentro. Debe diferir en ÓRDENES de magnitud.
        let viejo: f64 = ticks.iter().map(|tk| tk.bq + tk.aq).sum::<f64>() / ticks.len() as f64;
        assert!(
            st.notional > viejo * 100.0,
            "la etiqueta vieja ({viejo}) no medía nocional ({})",
            st.notional
        );
    }

    /// D-733 — un modelo puede batir holgadamente a la MEDIA y aun así ser
    /// peor que repetir el valor reciente. Con el gate viejo (R² vs media)
    /// ese modelo se promovía al vivo.
    #[test]
    fn el_gate_exige_batir_la_persistencia_no_la_media() {
        let margen = 0.001;
        // MSE media 1.0, modelo 0.85 ⇒ R² = 0.15 (el gate viejo pasaba),
        // pero la persistencia da 0.50: el modelo es PEOR que no hacer nada.
        let (r2, skill, pasa) = regression_gate(0.85, 1.0, 0.50, margen);
        assert!((r2 - 0.15).abs() < 1e-12, "R2 vs media informativo: {r2}");
        assert!(skill < 0.0, "skill vs persistencia debe ser negativo: {skill}");
        assert!(!pasa, "un modelo peor que la persistencia NO puede pasar el gate");

        // Modelo que sí bate a la persistencia.
        let (_, skill2, pasa2) = regression_gate(0.40, 1.0, 0.50, margen);
        assert!(skill2 > margen && pasa2, "skill {skill2} debería pasar");

        // Persistencia degenerada ⇒ sin baseline comparable ⇒ no pasa.
        assert!(!regression_gate(0.1, 1.0, 0.0, margen).2);
        assert!(!regression_gate(0.1, 1.0, f64::NAN, margen).2);
    }

    /// Las ventanas son CAUSALES: la de etiqueta vive en (t, t+τ] y la de
    /// persistencia en (t−τ, t]. Ninguna puede tocar el otro lado de t.
    #[test]
    fn las_ventanas_no_cruzan_el_instante_t() {
        // Ticks cada 100 ms durante 10 s.
        let raw: Vec<BinTick> = (0..100u64)
            .map(|k| agg_tick(k * 100, 100.0 + k as f64 * 0.01, 1.0, false))
            .collect();
        let i = 50usize; // t = 5000 ms
        let tau = 2_000u64;
        let t = raw[i].ts;

        let end = forward_window_end(&raw, i, t + tau);
        for tk in &raw[i + 1..end] {
            assert!(
                tk.ts > t && tk.ts <= t + tau,
                "ventana futura fuera de (t, t+tau]: {}",
                tk.ts
            );
        }
        assert!(end < raw.len() && raw[end].ts > t + tau, "el corte es el primer ts > t+tau");

        let (lo, ancla) = trailing_window_start(&raw, i, t - tau);
        for tk in &raw[lo..=i] {
            assert!(
                tk.ts > t - tau && tk.ts <= t,
                "ventana pasada fuera de (t-tau, t]: {}",
                tk.ts
            );
        }
        assert!(lo > 0 && raw[lo - 1].ts <= t - tau, "el inicio es el primer ts > t-tau");
        assert!(ancla > 0.0, "el ancla existe cuando el tape cubre tau hacia atrás");
        // El ancla es un mid ANTERIOR a la ventana, nunca uno futuro.
        let mid_ancla = (raw[lo - 1].bid + raw[lo - 1].ask) / 2.0;
        assert!((ancla - mid_ancla).abs() < 1e-12);
    }

    /// El diagnóstico (p10/p50/p90, varianza) debe salir del bosque TRUNCADO,
    /// que es el que se serializa. Acumular la puntuación a lo largo del bucle
    /// deja dentro los `patience` árboles que el early stopping descarta, y
    /// entonces las percentiles impresas describen un modelo que no existe en
    /// disco. Aquí se fija la diferencia: dos rodajas distintas del mismo
    /// bosque dan puntuaciones distintas, luego la rodaja importa.
    #[test]
    fn la_puntuacion_solo_cuenta_los_arboles_que_se_guardan() {
        let hoja = |v: f32| {
            vec![TreeNode { feature: -1, threshold: 0.0, left: -1, right: -1, value: v }]
        };
        // Bosque completo: 3 árboles. Truncado por early stopping: los 2
        // primeros. El tercero es el que la paciencia descartó.
        let bosque = vec![hoja(1.0), hoja(2.0), hoja(-10.0)];
        let x = [0.0f32; 4];
        let init = 0.25f32;
        let lr = 0.1f64;

        let guardado = forest_raw(init, lr, &bosque[..2], &x);
        let acumulado = forest_raw(init, lr, &bosque, &x);
        assert!(
            (guardado - (0.25 + 0.1 * 3.0)).abs() < 1e-12,
            "el bosque guardado vale init + lr*(1+2): {guardado}"
        );
        assert!(
            (acumulado - (0.25 + 0.1 * (-7.0))).abs() < 1e-12,
            "el acumulado arrastra el árbol descartado: {acumulado}"
        );
        assert!(
            (guardado - acumulado).abs() > 0.5,
            "los árboles descartados mueven la puntuación, así que el \
             diagnóstico NO puede salir del acumulado: {guardado} vs {acumulado}"
        );
        // Bosque vacío ⇒ la puntuación es exactamente el init_score.
        assert!((forest_raw(init, lr, &[], &x) - 0.25).abs() < 1e-12);
    }

    /// D-753 — LA PRUEBA DE QUE EL ENTRENADOR VIEJO NO TENÍA PARIDAD.
    ///
    /// Se construyen dos `StatefulEngine` sobre el MISMO tape:
    ///  · `motor`: alimentado como lo hace `GodEngineCore::process_event` —
    ///    DOS eventos por tick (depth y trade), `process_tick` con el volumen
    ///    del LIBRO (`clamp((bq+aq)·0,005; 0,01; 10)`, D-119) y `update_ofi`
    ///    en cada uno, más un `update_trade_flow` con la cantidad del trade.
    ///  · `viejo`: alimentado como lo hacía este binario — UN `process_tick`
    ///    con la cantidad del trade, un `update_trade_flow` y un `update_ofi`.
    ///
    /// Con el código viejo el vector guardado era el de `viejo` y el servido
    /// era el de `motor`, y aquí se fija que NO son el mismo vector: el
    /// comparador de paridad los separa. Si algún día alguien «simplificara»
    /// el entrenador volviendo a alimentar un motor suelto, esta prueba y la
    /// comprobación en línea lo dirían.
    #[test]
    fn el_camino_viejo_y_el_del_motor_dan_vectores_distintos() {
        let precios: Vec<f64> = (0..600u64).map(|k| 100.0 + (k as f64 * 0.37).sin() * 0.5).collect();
        let mut motor = StatefulEngine::new();
        let mut viejo = StatefulEngine::new();
        for (k, &p) in precios.iter().enumerate() {
            let tk = agg_tick(1_000 + k as u64 * 250, p, 1.0 + (k % 7) as f64, k % 3 == 0);
            let mid = (tk.bid + tk.ask) / 2.0;
            let qty = agg_trade_qty(tk.bq, tk.aq);
            let is_maker = tk.aq > tk.bq;

            // Camino del MOTOR: dos eventos por tick, volumen del libro.
            let vol_libro = ((tk.bq + tk.aq) * 0.005).clamp(0.01, 10.0);
            for evento in 0..2 {
                motor.process_tick(mid, vol_libro, tk.ts);
                let _ = motor.update_ofi(tk.bid, tk.ask, tk.bq, tk.aq);
                if evento == 1 {
                    motor.update_trade_flow(qty, is_maker);
                }
            }
            // Camino VIEJO del entrenador: un solo paso, volumen del trade.
            viejo.process_tick(mid, qty, tk.ts);
            viejo.update_trade_flow(qty, is_maker);
            let _ = viejo.update_ofi(tk.bid, tk.ask, tk.bq, tk.aq);
        }

        let omni = [0.0f64; 54];
        let (v_motor, _) = vector_de_servicio(&motor, &omni);
        let (v_viejo, _) = vector_de_servicio(&viejo, &omni);

        // El tensor del motor es el productor independiente contra el que la
        // comprobación en línea compara: aquí se emula con las 34 del motor.
        let mut tensor_motor = [0.0f64; 54];
        for d in 0..34 {
            tensor_motor[d] = motor.get_universal_features()[d] as f64;
        }
        let macro_motor = macro_ml_features(&omni);

        // El vector del motor pasa la comprobación contra sí mismo…
        assert!(
            primera_divergencia(&v_motor, &tensor_motor, &macro_motor).is_none(),
            "el vector tomado del motor debe ser idéntico al que el motor produce"
        );
        // …y el que construía el entrenador viejo NO la pasa.
        let divergencia = primera_divergencia(&v_viejo, &tensor_motor, &macro_motor);
        assert!(
            divergencia.is_some(),
            "el vector del camino viejo tendría que diferir del del motor y no difiere: \
             la prueba dejó de medir lo que dice medir"
        );
        // Y la divergencia no es una curiosidad de una sola dim.
        let distintas = (0..34).filter(|&d| v_motor[d] != v_viejo[d]).count();
        assert!(
            distintas >= 3,
            "sólo {} de las 34 dims micro/omni difieren entre el camino viejo y el del motor; \
             se esperaba una divergencia estructural amplia",
            distintas
        );
    }

    /// D-753 — la tolerancia de la comprobación es CERO y eso tiene que
    /// doler: una diferencia de un solo ulp de f32 en una sola dimensión ya
    /// es estructural, porque entre los dos lados no media aritmética.
    #[test]
    fn la_comprobacion_de_paridad_no_tiene_epsilon() {
        let mut v = [0f32; DIM_VECTOR];
        let mut tensor = [0.0f64; 54];
        for d in 0..34 {
            v[d] = 0.125 * (d as f32 + 1.0);
            tensor[d] = v[d] as f64;
        }
        let macro_ok = [v[44], v[45], v[46], v[47]];
        assert!(primera_divergencia(&v, &tensor, &macro_ok).is_none());

        // Un ulp de diferencia en la dim 17 ⇒ divergencia señalada EN LA 17.
        let mut tensor_ulp = tensor;
        tensor_ulp[17] = f32::from_bits(v[17].to_bits() + 1) as f64;
        let (d, _, _) = primera_divergencia(&v, &tensor_ulp, &macro_ok)
            .expect("un ulp de diferencia debe romper la paridad");
        assert_eq!(d, 17, "la dimensión señalada debe ser la que difiere");

        // El bloque macro también se compara: dim 46 desalineada.
        let mut macro_mal = macro_ok;
        macro_mal[2] = 1.5;
        let (d2, _, _) = primera_divergencia(&v, &tensor, &macro_mal)
            .expect("el bloque macro también entra en la comprobación");
        assert_eq!(d2, 46);

        // Un no finito del lado del motor se compara SANEADO, como en
        // servicio: el contrato lo convierte en 0.0 y eso no es divergencia.
        let mut tensor_nan = tensor;
        tensor_nan[3] = f64::NAN;
        v[3] = 0.0;
        assert!(
            primera_divergencia(&v, &tensor_nan, &macro_ok).is_none(),
            "el saneo NaN⇒0.0 es parte del contrato de servicio, no una divergencia"
        );
    }

    /// D-753 — las dims declaradas MUERTAS EN SERVICIO deben salir del motor
    /// en cero. Hoy `get_universal_features` las zerifica en ambos lados; el
    /// entrenador ya no las borra por su cuenta, así que necesita detectar el
    /// día en que el contrato deje de cumplirse.
    #[test]
    fn una_dim_muerta_distinta_de_cero_rompe_el_entrenamiento() {
        let v = [0f32; DIM_VECTOR];
        assert!(dim_muerta_no_nula(&v).is_none(), "todo a cero cumple el contrato");
        for &d in FEATURES_DEAD_IN_SERVE {
            let mut sucio = [0f32; DIM_VECTOR];
            sucio[d] = 1e-30; // cualquier valor distinto de cero, por pequeño que sea
            assert_eq!(
                dim_muerta_no_nula(&sucio),
                Some(d),
                "la dim muerta [{}] con valor no nulo debe detenerse",
                d
            );
        }
        // Una dim VIVA con valor no nulo no es un fallo: es lo normal.
        let viva = (0..DIM_VECTOR)
            .find(|d| !FEATURES_DEAD_IN_SERVE.contains(d))
            .expect("el contrato no puede declarar muertas todas las dims");
        let mut v2 = [0f32; DIM_VECTOR];
        v2[viva] = 0.7;
        assert!(dim_muerta_no_nula(&v2).is_none());
    }

    /// D-753 — EL INVENTARIO DE CANALES SIN PRODUCTOR EN EL TAPE, FIJADO.
    ///
    /// Reproducir un fichero de aggTrades deja sin productor el feed de
    /// liquidaciones (`dark_alpha`, dim [9]) y el libro L2 que alimenta
    /// `obi_accel` (dims [4] [5] [10]). El contrato las declara MUERTAS y el
    /// motor las zerifica en AMBOS lados, que es justo lo que hace que su
    /// ausencia no rompa la paridad. Dos invariantes se fijan aquí:
    ///
    ///  1. esas cuatro siguen declaradas muertas — si alguien reviviera [9]
    ///     en servicio, el vivo serviría severidad real mientras este
    ///     entrenador (que no tiene feed de liquidaciones) habría aprendido
    ///     una columna de ceros: exactamente el fallo que costó dos modelos
    ///     saturados;
    ///  2. todas viven por debajo de 34 — `get_universal_features()` sólo
    ///     zerifica `d < 34`, así que una dim muerta declarada en [34..48)
    ///     jamás llegaría en cero y `dim_muerta_no_nula` abortaría CADA
    ///     muestra. El contrato sería incumplible y hay que enterarse aquí,
    ///     no tras una corrida de un mes de tape.
    #[test]
    fn el_inventario_de_canales_sin_productor_en_el_tape_esta_declarado_muerto() {
        for &d in &[4usize, 5, 9, 10] {
            assert!(
                FEATURES_DEAD_IN_SERVE.contains(&d),
                "la dim [{}] no tiene productor al reproducir un tape (liquidaciones / libro \
                 L2) y ha dejado de estar declarada muerta: el entrenador aprendería ceros \
                 donde el vivo sirve dato real",
                d
            );
        }
        for &d in FEATURES_DEAD_IN_SERVE {
            assert!(
                d < 34,
                "la dim muerta [{}] cae fuera del bloque que el motor zerifica (d < 34): \
                 llegaría distinta de cero y `dim_muerta_no_nula` abortaría toda muestra",
                d
            );
        }
    }

    /// D-753 — la rejilla del instrumento se MIDE, no se cablea: la menor
    /// diferencia positiva entre precios medios consecutivos ES el tick, y el
    /// tick pone suelo al semi-spread que entra en la dim [2] (OFI).
    #[test]
    fn la_rejilla_sale_del_tape() {
        let tick = 0.001f64;
        // Precios en la rejilla: saltos de 1, 3 y 2 ticks. El mínimo es 1.
        let saltos = [1i64, 3, -2, 1, 5, -1, 2];
        let mut nivel = 100_000i64; // en ticks
        let mut ticks = Vec::new();
        for (k, s) in saltos.iter().enumerate() {
            nivel += s;
            // Libro simétrico de medio tick: el mid cae exactamente en la rejilla.
            let mid = nivel as f64 * tick;
            ticks.push(BinTick {
                ts: 1_000 + k as u64 * 100,
                bid: mid - tick * 0.5,
                ask: mid + tick * 0.5,
                bq: 3.0,
                aq: 0.75,
            });
        }
        let medido = tick_del_tape(&ticks);
        assert!(
            (medido - tick).abs() <= tick * 1e-9,
            "la rejilla medida ({medido}) debe ser el tick real ({tick})"
        );
        // Un tape con un único precio no tiene rejilla observable: 0.0, y el
        // binario aborta en vez de inventarse un suelo de fricción.
        let plano = vec![ticks[0], ticks[0], ticks[0]];
        assert_eq!(tick_del_tape(&plano), 0.0);

        // El paso de cantidad se mide igual sobre |bq − aq|.
        let paso = 0.01f64;
        // Saltos consecutivos de 3, 1 y 5 pasos: el mínimo observable es 1.
        let cantidades = [1.0f64, 1.03, 1.02, 1.07];
        let por_cantidad: Vec<BinTick> = cantidades
            .iter()
            .enumerate()
            .map(|(k, &q)| agg_tick(1_000 + k as u64 * 100, 100.0, q, false))
            .collect();
        let paso_medido = paso_del_tape(&por_cantidad);
        assert!(
            (paso_medido - paso).abs() <= paso * 1e-6,
            "paso medido {paso_medido} != {paso}"
        );
    }

    /// D-753 — una columna constante no admite splits: el modelo no puede
    /// usarla. El recorrido por dimensión es cómo se detecta.
    #[test]
    fn el_recorrido_declara_las_columnas_sin_splits() {
        let mut r = RecorridoDims::nuevo();
        assert!(r.constantes().is_empty(), "sin muestras no se declara nada");
        let mut a = [0f32; DIM_VECTOR];
        let mut b = [0f32; DIM_VECTOR];
        a[0] = 1.0;
        b[0] = 2.0; // dim 0 varía
        a[1] = 7.0;
        b[1] = 7.0; // dim 1 constante (no nula)
        r.observar(&a);
        r.observar(&b);
        let c = r.constantes();
        assert!(!c.contains(&0), "la dim 0 varía: no es constante");
        assert!(c.contains(&1), "la dim 1 no varía: es una columna sin splits");
        assert!(c.contains(&2), "una dim idénticamente 0 también es constante");
    }

    /// D-734 — las muestras solapan τ/stride veces: sin purga, la etiqueta de
    /// las últimas muestras de train se resuelve DENTRO de la validación.
    #[test]
    fn la_purga_elimina_el_solape_de_horizonte() {
        let stride = 50_000u64;
        let tau = 300_000u64;
        let ts: Vec<u64> = (0..100u64).map(|k| k * stride).collect();
        let split = 80usize;
        let tr_end = purge_end(&ts, split, tau);

        // Con tau/stride = 6, la muestra situada EXACTAMENTE tau antes del
        // borde resuelve su etiqueta en el instante de la primera muestra de
        // validación. XLIV-13b: con intervalos CERRADOS (la convención de
        // `purge_training`, `require_later_holdout` y López de Prado) esa
        // frontera es solape — su etiqueta usa el tick que la validación ya
        // tiene como rasgo —, así que se purgan 6, no 5. El número lo fija la
        // geometría de las ventanas, no una constante elegida.
        assert_eq!(split - tr_end, 6, "purgadas {} muestras", split - tr_end);
        for &t in &ts[..tr_end] {
            assert!(t + tau < ts[split], "muestra de train que resuelve en validación: {t}");
        }
        assert!(ts[tr_end] + tau >= ts[split], "la primera purgada debe tocar la validación");
        // Misma frontera que el contrato FMT sobre intervalos cerrados. Los
        // contratos FMT rechazan el instante 0 (los tapes usan ms de época):
        // se desplaza la rejilla 1 ms, y purge_end es invariante a traslación.
        let ts1: Vec<u64> = ts.iter().map(|t| t + 1).collect();
        assert_eq!(purge_end(&ts1, split, tau), tr_end);
        let tr = como_intervalos(
            &vec![vec![0.0f32]; split],
            &vec![0.0; split],
            &ts1[..split],
            tau,
        );
        let sel = como_intervalos(&[vec![0.0]], &[0.0], &ts1[split..split + 1], tau);
        let (kept, _) = purge_training(tr, &sel).unwrap();
        assert_eq!(kept.labels.len(), tr_end, "purge_end y purge_training deben coincidir");

        // Sin solape (tau < stride) no se purga nada.
        assert_eq!(purge_end(&ts, split, 10_000), split);
    }

    /// QS-2 — guardia del CABLEADO. El merge 6fdccd64 ya demostró que los
    /// helpers pueden sobrevivir con sus tests mientras `main()` deja de
    /// llamarlos. Se exige que `main()` escriba el modelo con su manifiesto,
    /// consulte y aplique el libro del holdout y anote cada consulta, y que
    /// la escritura directa del JSON del modelo (sin linaje) no vuelva.
    #[test]
    fn qs2_main_escribe_linaje_y_aplica_el_libro_del_holdout() {
        let src = include_str!("train_forest.rs");
        let inicio = src.find("\nfn main() {").expect("fn main");
        let fin = inicio + 1 + src[inicio + 1..].find("\n}\n").expect("fin de main");
        let cuerpo = &src[inicio..fin];
        for llamada in [
            "model_manifest::write_model_with_manifest(",
            "model_manifest::read_ledger(",
            "model_manifest::prior_other_artifacts(",
            "model_manifest::check_holdout_fresh(",
            "model_manifest::append_ledger(",
            "model_manifest::input_record(\"test\"",
        ] {
            assert!(cuerpo.contains(llamada), "main() ya no llama a {llamada}");
        }
        assert!(
            !cuerpo.contains("to_writer_pretty(&mut f, &model)"),
            "el modelo vuelve a escribirse sin manifiesto"
        );
        assert!(cuerpo.contains("std::process::exit(3)"), "falta el código de salida del holdout reusado");
    }

    /// TGB-G: a la preparación de muestras sólo llega la cinta de firma RELATIVA con recibo de procedencia
    /// acreditada ligado a su SHA-256; la antigua, la que no tiene recibo y la de recibo ajeno se rechazan con su
    /// causa, también si llegan como `.bin.tgb`.
    #[test]
    fn tgb_solo_la_cinta_relativa_acreditada_llega_a_las_muestras() {
        use model_manifest::sha256_hex;
        // Recetas auditadas (binance_vision_sync.rs:222 y tape_spread_fix.rs:4) como datos de prueba.
        let rel = |p: f64| (p - p * 0.00005, p + p * 0.00005);
        let ant = |p: f64| {
            let h = (p * 0.00005f64).max(0.01);
            (p - h, p + h)
        };
        let cinta = |f: &dyn Fn(f64) -> (f64, f64)| {
            let mut v = b"TGMTICK1".to_vec();
            for k in 0..300u64 {
                let (b, a) = f(0.45 + 0.013 * k as f64);
                for x in [1_780_272_000_000 + k * 1_000, b.to_bits(), a.to_bits(), 1f64.to_bits(), 2f64.to_bits()] {
                    v.extend_from_slice(&x.to_le_bytes());
                }
            }
            v
        };
        let recibo = |bytes: &[u8]| {
            format!(
                "formato=tg-procedencia-1\nsha256_original={}\nbytes_original={}\nprocedencia_acreditada=relativa\nevidencia_productor=prueba\ncobertura=completa\n",
                sha256_hex(bytes),
                bytes.len()
            )
        };
        let dir = std::env::temp_dir().join(format!("train_forest_tgb_{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();
        let relativa = cinta(&rel);
        let antigua = cinta(&ant);
        let poner = |nombre: &str, bytes: &[u8], rec: Option<String>| -> String {
            let p = dir.join(nombre);
            std::fs::write(&p, bytes).unwrap();
            if let Some(r) = rec {
                std::fs::write(dir.join(format!("{nombre}.procedencia")), r).unwrap();
            }
            p.to_string_lossy().into_owned()
        };
        // TGBG-2: el recibo autodeclarado de H (`evidencia_productor=prueba`) ya NO acredita la generacion, ni como
        // `.bin` ni como `.bin.tgb`: sin una cadena reconocida en el indice revisado no hay procedencia.
        let ok = poner("REL_REAL.bin", &relativa, Some(recibo(&relativa)));
        let e = admitir_cinta_para_muestras(&ok, &quantum_engine::tgb_generation::LectoresDisco::sin_indice(), None).err().unwrap_or_default();
        assert!(e.contains("procedencia_no_acreditada"), "recibo autodeclarado admitido o sin causa: {e:?}");
        let tgb = dir.join("RELT_REAL.bin");
        std::fs::write(dir.join("RELT_REAL.bin.tgb"), tick_tape::tgb::empaquetar(&relativa).unwrap()).unwrap();
        std::fs::write(dir.join("RELT_REAL.bin.procedencia"), recibo(&relativa)).unwrap();
        let e = admitir_cinta_para_muestras(&tgb.to_string_lossy(), &quantum_engine::tgb_generation::LectoresDisco::sin_indice(), None).err().unwrap_or_default();
        assert!(e.contains("procedencia_no_acreditada"), "recibo autodeclarado admitido desde .tgb: {e:?}");
        for (nombre, path) in [
            ("firma antigua", poner("ANT_REAL.bin", &antigua, Some(recibo(&antigua)))),
            ("sin recibo", poner("SINREC_REAL.bin", &relativa, None)),
            ("recibo de otro original", poner("OTRO_REAL.bin", &relativa, Some(recibo(&antigua)))),
        ] {
            match admitir_cinta_para_muestras(&path, &quantum_engine::tgb_generation::LectoresDisco::sin_indice(), None) {
                Ok(_) => panic!("{nombre}: la cinta llegó a la preparación de muestras sin evidencia suficiente"),
                Err(e) => assert!(!e.is_empty(), "{nombre}: sin causa"),
            }
        }
        let _ = std::fs::remove_dir_all(&dir);
    }

    /// CACHEV3-P5: (a) con una cache que cubre la cinta, el adaptador entrega los MISMOS ticks, bit a bit, que el
    /// camino actual extrae de la cinta mapeada (también si la cinta llega como `.bin.tgb`); (b) sin cache, con una
    /// cache que no la cubre o con una hora alterada, cae al camino de siempre (`None`) sin error.
    #[test]
    fn train_forest_cache_paridad_y_caida() {
        let dir = std::env::temp_dir().join(format!("train_forest_cache_{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();
        // Tres horas con bits incómodos (-0.0, NaN con carga, infinito) en las cantidades.
        let mut cinta = b"TGMTICK1".to_vec();
        let raros = [(-0.0f64).to_bits(), 0x7ff8_0000_dead_beef, f64::INFINITY.to_bits()];
        for k in 0..540u64 {
            let p = 0.45 + 0.013 * k as f64;
            let bq = if k % 5 == 0 { raros[(k as usize / 5) % 3] } else { (1.0 + k as f64).to_bits() };
            for x in [1_789_351_200_000 + k * 20_000, (p * 0.99995).to_bits(), (p * 1.00005).to_bits(), bq, 2f64.to_bits()] {
                cinta.extend_from_slice(&x.to_le_bytes());
            }
        }
        let bin = dir.join("PARUSDT_REAL.bin");
        std::fs::write(&bin, &cinta).unwrap();
        let solo_tgb = dir.join("PARTUSDT_REAL.bin");
        std::fs::write(dir.join("PARTUSDT_REAL.bin.tgb"), tick_tape::tgb::empaquetar(&cinta).unwrap()).unwrap();
        let cache = dir.join("cache");
        let op = |s: &str| cache_columnas::construir::Opciones {
            fuente: "tgb-reales".into(),
            simbolo: s.into(),
            ..Default::default()
        };
        let i = cache_columnas::construir::construir_desde_cinta(&cache, &bin, &op("PARUSDT")).unwrap();
        assert_eq!(i.horas_nuevas.len(), 3);
        let cache_s = cache.to_string_lossy().into_owned();
        // El camino actual: la cinta lógica mapeada, leída como `BinTick` (igual que `build`).
        let actual = |p: &std::path::Path| {
            let t = tick_tape::logica::open_logical_tape(p, tick_tape::tgb::Presupuesto::sin_tope()).unwrap();
            let n = (t.len() - 8) / std::mem::size_of::<BinTick>();
            let v = unsafe { std::slice::from_raw_parts(t.as_bytes().as_ptr().add(8) as *const BinTick, n) }.to_vec();
            (t, v)
        };
        let bits = |v: &[BinTick]| -> Vec<[u64; 5]> {
            v.iter().map(|t| [t.ts, t.bid.to_bits(), t.ask.to_bits(), t.bq.to_bits(), t.aq.to_bits()]).collect()
        };
        for ruta in [&bin, &solo_tgb] {
            let (tape, esperado) = actual(ruta);
            assert_eq!(esperado.len(), 540);
            // TGBG-5: el parseo LE seguro da los mismos bits que el cast de memoria de antes (NaN con carga, -0.0, inf).
            assert_eq!(bits(&ticks_de_cinta(tape.as_bytes()).expect("parseo LE")), bits(&esperado), "parseo LE ({})", ruta.display());
            let servido = ticks_desde_cache(&cache_s, &tape.sha256_original(), esperado.len()).expect("la cache cubre la cinta");
            assert_eq!(bits(&servido), bits(&esperado), "paridad valor a valor ({})", ruta.display());
        }
        let (tape, esperado) = actual(&bin);
        // (b) Sin cache: cae.
        assert!(ticks_desde_cache(&dir.join("no_existe").to_string_lossy(), &tape.sha256_original(), esperado.len()).is_none());
        // Otro largo pedido (no cubre [0, n)): cae.
        assert!(ticks_desde_cache(&cache_s, &tape.sha256_original(), esperado.len() - 1).is_none());
        // Una hora alterada: cae, sin pánico.
        let man = cache_columnas::leer_manifiesto(&cache).unwrap();
        let ask = cache.join(&man.horas[1].clave).join("ask.f64le");
        let mut b = std::fs::read(&ask).unwrap();
        b[9] ^= 0x10;
        std::fs::write(&ask, &b).unwrap();
        assert!(ticks_desde_cache(&cache_s, &tape.sha256_original(), esperado.len()).is_none());
        // Una hora expulsada: ya no cubre, cae.
        b[9] ^= 0x10;
        std::fs::write(&ask, &b).unwrap();
        assert!(ticks_desde_cache(&cache_s, &tape.sha256_original(), esperado.len()).is_some());
        let uso = cache_columnas::leer_manifiesto(&cache).unwrap().uso_bytes();
        cache_columnas::expulsion::expulsar_hasta(&cache, uso - 1, "prueba").unwrap();
        assert!(ticks_desde_cache(&cache_s, &tape.sha256_original(), esperado.len()).is_none());
        let _ = std::fs::remove_dir_all(&dir);
    }

    // ── TGBG-5 (G10): todas las fuentes admitidas antes de preparar muestras. PURA: lectores en memoria. ──
    struct LectoresMemoria {
        cintas: std::collections::HashMap<String, Vec<u8>>,
        recibos: std::collections::HashMap<String, Vec<u8>>,
        evidencias: Vec<Vec<u8>>,
        registro: std::cell::RefCell<Vec<String>>,
    }

    fn sha32(b: &[u8]) -> [u8; 32] {
        use sha2::Digest;
        sha2::Sha256::digest(b).into()
    }

    fn hx(b: &[u8]) -> String {
        b.iter().map(|x| format!("{x:02x}")).collect()
    }

    impl quantum_engine::tgb_generation::Lectores for LectoresMemoria {
        fn original(&self, logica: &std::path::Path) -> std::io::Result<quantum_engine::tgb_generation::Instantanea> {
            let k = logica.display().to_string();
            self.registro.borrow_mut().push(format!("admitir {k}"));
            let bytes = self.cintas.get(&k).cloned().ok_or_else(|| std::io::Error::new(std::io::ErrorKind::NotFound, k))?;
            Ok(quantum_engine::tgb_generation::Instantanea { bytes, transporte: tick_tape::logica::Transporte::Bin })
        }
        fn recibo(&self, logica: &std::path::Path) -> std::io::Result<Option<Vec<u8>>> {
            Ok(self.recibos.get(&logica.display().to_string()).cloned())
        }
        fn evidencia(&self, sha256: &[u8; 32]) -> std::io::Result<Option<Vec<u8>>> {
            Ok(self.evidencias.iter().find(|e| sha32(e) == *sha256).cloned())
        }
    }

    /// Cinta REAL relativa de precio bajo con `n` registros; `desde` la distingue de las demas.
    fn cinta_g10(n: u64, desde: f64) -> Vec<u8> {
        let mut v = b"TGMTICK1".to_vec();
        for k in 0..n {
            let p = desde + 0.013 * k as f64;
            let h = p * 0.00005;
            for x in [1_780_300_800_000 + k * 1_000, (p - h).to_bits(), (p + h).to_bits(), 1f64.to_bits(), 2f64.to_bits()] {
                v.extend_from_slice(&x.to_le_bytes());
            }
        }
        v
    }

    /// (recibo, evidencia) de `original`, y la linea del indice que reconoce la evidencia.
    fn cadena_g10(original: &[u8]) -> (Vec<u8>, Vec<u8>, String) {
        let commit = "ef".repeat(20);
        let evidencia = format!(
            "formato=tg-evidencia-productor-1\nproductor=binance_vision_sync\ncommit_productor={commit}\nsha256_fuentes_productor={}\nsha256_ejecutable={}\nreceta=relativa\nconstante_r_bits={:016x}\norigen_receta=src/bin/binance_vision_sync.rs:222\ncomando=binance_vision_sync\nentrada_sha256={}\nsha256_salida={}\nbytes_salida={}\nsha256_registro_salida={}\ntransformaciones=ninguna\n",
            hx(&sha32(b"fuentes")),
            hx(&sha32(b"exe")),
            0.00005f64.to_bits(),
            hx(&sha32(original)),
            hx(&sha32(original)),
            original.len(),
            hx(&sha32(b"stdout"))
        );
        let ev = hx(&sha32(evidencia.as_bytes()));
        let n = (original.len() - 8) / 40;
        let recibo = format!(
            "formato=tg-procedencia-2\nversion_analizador=tick-tape-procedencia-2\nruta_logica=g10\nformato_cinta=TGMTICK1+40B_LE\nsha256_original={}\nbytes_original={}\nregistros_esperados={n}\nregistros_analizados={n}\ncobertura=completa\nrecetas_compatibles=relativa\nestado_firma=relativa\nprimer_testigo_relativa=0\nprocedencia_acreditada=relativa\nevidencia_productor_sha256={ev}\nanalisis=intervalos_exhaustivos\n",
            hx(&sha32(original)),
            original.len()
        );
        (recibo.into_bytes(), evidencia.into_bytes(), format!("evidencia={ev};binance_vision_sync;{commit};relativa\n"))
    }

    #[test]
    fn tgb_g_10_todas_las_fuentes_antes_de_muestras() {
        let rutas = ["mem/TRAIN_REAL.bin", "mem/SEL_REAL.bin", "mem/TEST_REAL.bin"];
        let originales = [cinta_g10(300, 0.45), cinta_g10(300, 0.46), cinta_g10(300, 0.47)];
        let mut indice = "formato=tg-indice-procedencia-1\nrevisado_por=contrato G10\n".to_string();
        let mut mem = LectoresMemoria {
            cintas: Default::default(),
            recibos: Default::default(),
            evidencias: Vec::new(),
            registro: Default::default(),
        };
        for (ruta, original) in rutas.iter().zip(&originales) {
            let (recibo, evidencia, linea) = cadena_g10(original);
            mem.cintas.insert(ruta.to_string(), original.clone());
            mem.recibos.insert(ruta.to_string(), recibo);
            mem.evidencias.push(evidencia);
            indice.push_str(&linea);
        }
        let pin = sha32(indice.as_bytes());
        let ix = quantum_engine::tgb_generation::IndiceRevisado::desde_bytes(indice.into_bytes(), pin);
        // Preparador espia, invocado como main: build(&train), build(seleccion), build(test).
        let preparar = |f: &FuentesAdmitidas, mem: &LectoresMemoria| -> Vec<[u8; 32]> {
            let mut vistas = Vec::new();
            for c in std::iter::once(&f.train).chain(f.seleccion.as_ref()).chain(f.test.as_ref()) {
                mem.registro.borrow_mut().push(format!("preparar {}", c.ruta_logica().display()));
                assert_eq!(ticks_de_cinta(c.bytes()).expect("ticks").len(), 300);
                vistas.push(sha32(c.bytes()));
            }
            vistas
        };
        let orquestar = |mem: &LectoresMemoria| -> Result<Vec<[u8; 32]>, String> {
            let f = admitir_fuentes(rutas[0], rutas[1], rutas[2], &|p: &str| admitir_cinta_para_muestras(p, mem, Some(&ix)))?;
            Ok(preparar(&f, mem))
        };

        // Train, seleccion o test rechazados por turno (recibo retirado): cero preparaciones.
        for k in 0..3 {
            let recibo = mem.recibos.remove(rutas[k]).expect("recibo");
            mem.registro.borrow_mut().clear();
            let r = orquestar(&mem);
            assert!(r.is_err(), "{}: rechazada y aun asi admitida", rutas[k]);
            let reg = mem.registro.borrow().clone();
            assert!(reg.iter().all(|l| !l.starts_with("preparar")), "{}: se prepararon muestras: {reg:?}", rutas[k]);
            mem.recibos.insert(rutas[k].to_string(), recibo);
        }

        // Todas admitidas: el preparador recibe exactamente sus tres fuentes verificadas, despues de admitirlas todas.
        mem.registro.borrow_mut().clear();
        let vistas = orquestar(&mem).expect("las tres admitidas");
        assert_eq!(vistas, originales.iter().map(|o| sha32(o)).collect::<Vec<_>>());
        let reg = mem.registro.borrow().clone();
        let ultima_admision = reg.iter().rposition(|l| l.starts_with("admitir")).expect("admisiones");
        let primera_preparacion = reg.iter().position(|l| l.starts_with("preparar")).expect("preparaciones");
        assert!(ultima_admision < primera_preparacion, "se preparo antes de admitir todo: {reg:?}");

        // main: la admision de todas las fuentes precede al primer build, y ningun build recibe una ruta sin admitir.
        let src = include_str!("train_forest.rs").replace("\r\n", "\n");
        let inicio = src.find("\nfn main() {").expect("fn main");
        let fin = inicio + 1 + src[inicio + 1..].find("\n}\n").expect("fin de main");
        let cuerpo = &src[inicio..fin];
        let admision = cuerpo.find("let fuentes = admitir_fuentes(&in_path, &val_in, &test_in,").expect("main admite las fuentes");
        let primer_build = cuerpo.find("let s_train = build(&fuentes.train);").expect("build de train");
        assert!(admision < primer_build, "main construye muestras antes de admitir todas las fuentes");
        for viejo in ["build(&in_path)", "build(&val_in)", "build(&test_in)"] {
            assert!(!cuerpo.contains(viejo), "main vuelve a construir desde una ruta sin admitir: {viejo}");
        }
    }
}
#[cfg(test)]
mod har_qlike_tests {
    use super::*;

    // Cada incremento de prefijo equivale a UNA volatilidad de horizonte conocida.
    fn history(vols: &[f64]) -> HarHistory {
        let mut h = HarHistory::default();
        let mut sum = 0.0;
        h.points.push((1, sum));
        for (i, &v) in vols.iter().enumerate() {
            sum += (v / 100.0).powi(2);
            h.points.push((1 + (i as u64 + 1) * 10, sum));
        }
        h
    }

    fn tick(ts: u64, mid: f64) -> BinTick {
        BinTick {
            ts,
            bid: mid,
            ask: mid,
            bq: 1.0,
            aq: 1.0,
        }
    }

    fn calibration(beta: Vec<f64>, windows: Vec<usize>, floor: Option<f64>) -> HarCalibration {
        HarCalibration {
            model: Some(HarModel {
                beta,
                windows,
                ridge_lambda: 0.0,
            }),
            floor,
            record: serde_json::json!({}),
        }
    }

    fn close(actual: f64, expected: f64) {
        assert!(
            (actual - expected).abs() < 1e-10,
            "actual={actual}, expected={expected}"
        );
    }

    #[test]
    fn har_ventanas_y_prediccion_basica() {
        let h = history(&[1.0, 2.0, 3.0, 4.0]);
        let x = h.features(41, 10, &[1, 2, 3]).unwrap();
        for (a, b) in x.into_iter().zip([4.0, 3.5, 3.0]) {
            close(a, b);
        }
        let m = HarModel {
            windows: vec![1, 2, 3],
            beta: vec![0.0, 0.5, 0.25, 0.25],
            ridge_lambda: 0.0,
        };
        close(m.predict(&h, 41, 10).unwrap(), 3.625);
    }

    #[test]
    fn har_intervalos_fisicos_no_dependen_del_stride() {
        let h = history(&[1.0, 2.0, 3.0, 4.0]);
        let fine: Vec<_> = (11..=41)
            .step_by(5)
            .filter_map(|t| h.features(t, 10, &[1]))
            .collect();
        let coarse: Vec<_> = (11..=41)
            .step_by(10)
            .filter_map(|t| h.features(t, 10, &[1]))
            .collect();
        close(fine.last().unwrap()[0], coarse.last().unwrap()[0]);
        close(h.features(41, 10, &[1, 2, 3]).unwrap()[2], 3.0);
        let wrong_index_mean = fine[fine.len() - 3..].iter().map(|x| x[0]).sum::<f64>() / 3.0;
        assert!((wrong_index_mean - 3.0).abs() > 0.1);
    }

    #[test]
    fn har_autocorrelacion_usa_malla_no_solapada_tau() {
        let h = history(&[1.0, 2.0, 3.0, 4.0]);
        let (grid, missing, end) = har_grid(&h, 11, 41, 10);
        assert_eq!((grid.len(), missing, end), (4, 0, 41));
        for (a, b) in grid.iter().zip([1.0, 2.0, 3.0, 4.0]) {
            close(*a, b);
        }
        // Un stride de cinco produciria siete muestras: no se usa como N de ACF.
        assert_eq!((11..=41).step_by(5).count(), 7);
    }

    #[test]
    fn har_autocorrelacion_negativa_se_compara_en_valor_absoluto() {
        // Media=2, denominador=8, rho(2)=-6/8, rho(3)=-1/8.
        // N=8, umbral_m=1/sqrt(8): k=2 no cruza; k=3 si cruza.
        let (windows, reasons) = har_windows(&[1.0, 1.0, 3.0, 3.0, 1.0, 1.0, 3.0, 3.0]);
        assert_eq!(windows, vec![1, 3]);
        assert!(!reasons.is_empty());
    }

    #[test]
    fn har_larga_es_posterior_a_media_y_usa_umbral_mas_estricto() {
        // N=12, un impulso inicial: rho(k)=-k/(12*11).
        // |rho(2)|=1/66 <= 1/sqrt(12); |rho(3)|=1/44 <= 1/12.
        let mut grid = vec![1.0; 12];
        grid[0] = 2.0;
        let (windows, reasons) = har_windows(&grid);
        assert_eq!(windows, vec![1, 2, 3]);
        assert!(reasons.is_empty());
    }

    #[test]
    fn har_lagunas_no_se_comprimen_en_la_malla() {
        let mut h = history(&[1.0, 2.0, 3.0, 4.0, 5.0]);
        for p in &mut h.points[3..] {
            p.0 += 10;
        }
        let (grid, missing, end) = har_grid(&h, 11, 61, 10);
        assert_eq!((grid.len(), missing, end), (3, 1, 61));
        for (a, b) in grid.iter().zip([3.0, 4.0, 5.0]) {
            close(*a, b);
        }
    }

    #[test]
    fn har_ventanas_faltantes_se_registran_sin_defaults() {
        let (windows, reasons) = har_windows(&[1.0, 2.0]);
        assert_eq!(windows, vec![1]);
        assert!(reasons.iter().any(|r| r.contains("sin_cruce_m")));
    }

    #[test]
    fn har_ridge_serie_plana_tiene_intercepto_media_y_pendientes_cero() {
        let x = vec![vec![2.0, 2.0, 2.0]; 4];
        let m = har_ridge(&x, &[1.0, 2.0, 3.0, 4.0], vec![1, 2, 3]).unwrap();
        close(m.beta[0], 2.5);
        for b in &m.beta[1..] {
            close(*b, 0.0);
        }
        close(m.ridge_lambda, 4.0 * f64::EPSILON.sqrt());
    }

    #[test]
    fn har_ridge_colinealidad_exacta_tiene_solucion_finita() {
        let x: Vec<_> = (1..=4).map(|i| vec![i as f64; 3]).collect();
        let m = har_ridge(&x, &[2.0, 4.0, 6.0, 8.0], vec![1, 2, 3]).unwrap();
        let expected = 2.0 / (3.0 + f64::EPSILON.sqrt());
        // Las columnas identicas comparten la pendiente; el intercepto es libre.
        for b in &m.beta[1..] {
            assert!((*b - expected).abs() < 1e-7);
        }
        assert!((m.beta[0] - (5.0 - 7.5 * expected)).abs() < 1e-7);
        assert!(m.beta.iter().all(|b| b.is_finite()));
    }

    #[test]
    fn har_ridge_caso_dos_filas_calculable_a_mano() {
        // x=[0,2], y=[1,5]: media_x=1, media_y=3, Gram=2, rhs=4.
        let m = har_ridge(&[vec![0.0], vec![2.0]], &[1.0, 5.0], vec![1]).unwrap();
        let slope = 2.0 / (1.0 + f64::EPSILON.sqrt());
        close(m.beta[1], slope);
        close(m.beta[0], 3.0 - slope);
    }

    #[test]
    fn har_ridge_rechaza_nan_e_infinito() {
        assert!(har_ridge(&[vec![f64::NAN], vec![1.0]], &[1.0, 2.0], vec![1]).is_none());
        assert!(har_ridge(&[vec![0.0], vec![1.0]], &[1.0, f64::INFINITY], vec![1]).is_none());
    }

    #[test]
    fn har_ajuste_cuenta_y_descarta_etiquetas_no_finitas() {
        let h = history(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]);
        let ts: Vec<_> = (11..=81).step_by(10).collect();
        let c = fit_har(&h, &ts, &[1.0, 2.0, 3.0, 4.0, 5.0, f64::NAN, 7.0, 8.0], 10);
        assert_eq!(c.record["fit_invalid_targets"], 1);
        assert!(c.model.is_some());
        assert!(c.model.unwrap().beta.iter().all(|v| v.is_finite()));
    }

    #[test]
    fn qlike_floor_procede_solo_de_train_positivo() {
        let h = history(&[1.0, 2.0, 3.0, 4.0]);
        let c = fit_har(&h, &[11, 21, 31, 41], &[4.0, 2.0, 0.0, f64::NAN], 10);
        close(c.floor.unwrap(), 0.0004);
        assert!(fit_har(&h, &[11, 21], &[0.0, 0.0], 10).floor.is_none());
    }

    #[test]
    fn qlike_cero_cuando_acierto() {
        // sigma=200% equivale a varianza=4.
        assert_eq!(har_qlike(200.0, 200.0, 1.0), Some((0.0, false)));
    }

    #[test]
    fn qlike_penaliza_mas_subestimar() {
        let under = har_qlike(200.0, 100.0 * 2.0_f64.sqrt(), 1.0).unwrap().0;
        let over = har_qlike(200.0, 100.0 * 8.0_f64.sqrt(), 1.0).unwrap().0;
        close(under, 1.0 - 2.0_f64.ln());
        close(over, 2.0_f64.ln() - 0.5);
        assert!(under > over);
    }

    #[test]
    fn qlike_cero_realizado_se_omite_pero_sigue_en_mse() {
        assert!(har_qlike(0.0, 2.0, 0.0001).is_none());
        let c = calibration(vec![0.0, 1.0], vec![1], Some(0.0001));
        let r = compare_har(
            &c,
            &history(&[1.0, 2.0]),
            HarEvalSeries {
                ts: &[11, 21],
                y: &[0.0, 2.0],
                model: &[1.0, 2.0],
                persist: &[1.0, 2.0],
                mean: 2.0,
                tau: 10,
            },
        );
        assert_eq!(r["mse_common_rows"], 2);
        assert_eq!(r["qlike_common_rows"], 1);
        assert_eq!(r["qlike_omitted_zero"], 1);
        close(
            r["metrics"]["har"]["mse_sigma_pct_common"]
                .as_f64()
                .unwrap(),
            0.5,
        );
    }

    #[test]
    fn qlike_recorte_ignora_el_signo_al_cuadrar_y_aplica_suelo_positivo() {
        for prediction in [-200.0, -50.0, 0.0, 5.0] {
            // actual=100% -> varianza exactamente 1.0; floor=1.0. Las cuatro
            // predicciones quedan por debajo del suelo (0.0 o ~0.0025), asi
            // que predicted.max(floor)=1.0 y ratio=1.0/1.0=1.0 exacto: la rama
            // delta - delta.ln_1p() de har_qlike devuelve 0.0 - 0.0 = 0.0.
            // -200.0 vigila el signo: si se elevara al cuadrado daria 4.0,
            // por encima del suelo, y la perdida dejaria de ser 0.0.
            assert_eq!(har_qlike(100.0, prediction, 1.0), Some((0.0, true)));
        }
    }

    #[test]
    fn qlike_suelo_simetrico_en_los_cuatro_modelos() {
        let c = calibration(vec![0.0, 0.0], vec![1], Some(0.01));
        let r = compare_har(
            &c,
            &history(&[1.0]),
            HarEvalSeries {
                ts: &[11],
                y: &[10.0],
                model: &[0.0],
                persist: &[0.0],
                mean: 0.0,
                tau: 10,
            },
        );
        for name in ["modelo", "persistencia", "har", "media"] {
            assert_eq!(r["metrics"][name]["qlike_clipped"], 1);
            close(r["metrics"][name]["qlike"].as_f64().unwrap(), 0.0);
        }
        assert_eq!(r["qlike_evidence"], false);
    }

    #[test]
    fn qlike_sin_floor_train_no_fabrica_epsilon() {
        let c = calibration(vec![0.0, 1.0], vec![1], None);
        let r = compare_har(
            &c,
            &history(&[1.0]),
            HarEvalSeries {
                ts: &[11],
                y: &[1.0],
                model: &[1.0],
                persist: &[1.0],
                mean: 1.0,
                tau: 10,
            },
        );
        assert_eq!(r["qlike_evidence"], false);
        assert!(r["metrics"]["har"]["qlike"].is_null());
    }

    #[test]
    fn qlike_mas_de_mitad_recortada_invalida_evidencia() {
        let c = calibration(vec![-1.0, 1.0], vec![1], Some(0.0001));
        let h = history(&[1.0, 1.0, 3.0]);
        let half = compare_har(
            &c,
            &h,
            HarEvalSeries {
                ts: &[11, 31],
                y: &[1.0, 1.0],
                model: &[1.0, 1.0],
                persist: &[1.0, 1.0],
                mean: 1.0,
                tau: 10,
            },
        );
        assert_eq!(half["har_more_than_half_clipped"], false);
        assert_eq!(half["qlike_evidence"], true);
        let majority = compare_har(
            &c,
            &h,
            HarEvalSeries {
                ts: &[11, 21, 31],
                y: &[1.0; 3],
                model: &[1.0; 3],
                persist: &[1.0; 3],
                mean: 1.0,
                tau: 10,
            },
        );
        assert_eq!(majority["har_more_than_half_clipped"], true);
        assert_eq!(majority["qlike_evidence"], false);
    }

    #[test]
    fn har_causalidad_estricta_ante_ticks_futuros_modificados() {
        let mut ticks: Vec<_> = (0..=6)
            .map(|i| tick(1 + i * 10, 100.0 * 1.01_f64.powi(i as i32)))
            .collect();
        let mut truncated = HarHistory::default();
        truncated.append_ticks(&ticks[..=4]).unwrap();
        let expected = truncated.features(41, 10, &[1, 2, 3]).unwrap();
        ticks[5].bid = 500.0;
        ticks[5].ask = 500.0;
        ticks[6].bid = 900.0;
        ticks[6].ask = 900.0;
        let mut full = HarHistory::default();
        full.append_ticks(&ticks).unwrap();
        assert_eq!(expected, full.features(41, 10, &[1, 2, 3]).unwrap());
    }

    #[test]
    fn har_calibracion_no_inspecciona_precios_posteriores_a_train() {
        let mut ticks: Vec<_> = (0..=8)
            .map(|i| tick(1 + i * 10, 100.0 * 1.01_f64.powi(i as i32)))
            .collect();
        let mut before = HarHistory::default();
        before.append_through(&ticks, 61).unwrap();
        let ts = [11, 21, 31, 41, 51, 61];
        let y = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        let first = fit_har(&before, &ts, &y, 10);
        // Un mid que desbordaria la suma no debe invalidar el train anterior.
        ticks[7].bid = f64::MAX;
        ticks[7].ask = f64::MAX;
        let mut after = HarHistory::default();
        after.append_through(&ticks, 61).unwrap();
        let second = fit_har(&after, &ts, &y, 10);
        assert_eq!(before.points, after.points);
        assert_eq!(first.model, second.model);
        assert!(first.model.is_some());
    }

    #[test]
    fn har_sin_fuga_de_etiquetas_de_seleccion_y_test() {
        let h = history(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]);
        let ts: Vec<_> = (11..=81).step_by(10).collect();
        let mut y = vec![2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0];
        let end = purge_end(&ts, 6, 10);
        let before = fit_har(&h, &ts[..end], &y[..end], 10);
        for target in &mut y[6..] {
            *target = 1_000_000.0;
        }
        let after = fit_har(&h, &ts[..end], &y[..end], 10);
        assert_eq!(before.model, after.model);
        assert_eq!(before.floor, after.floor);
        assert_eq!(
            before.model.as_ref().unwrap().predict(&h, 81, 10),
            after.model.as_ref().unwrap().predict(&h, 81, 10)
        );
    }

    #[test]
    fn har_purga_excluye_etiqueta_que_alcanza_la_frontera() {
        let h = history(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]);
        let ts: Vec<_> = (11..=81).step_by(10).collect();
        let mut y = vec![2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0];
        // Seleccion empieza en 71; la etiqueta de t=61 termina en 71 y se purga.
        let end = purge_end(&ts, 6, 10);
        assert_eq!(end, 5);
        let before = fit_har(&h, &ts[..end], &y[..end], 10);
        y[5] = 1_000_000.0;
        let after = fit_har(&h, &ts[..end], &y[..end], 10);
        assert_eq!(before.model, after.model);
        assert_eq!(after.record["train_rows_purged"], 5);
        assert!(ts[end - 1] + 10 < ts[6]);
    }

    #[test]
    fn har_reutiliza_solo_pasado_de_la_particion_anterior() {
        let mut h = HarHistory::default();
        h.append_ticks(&[tick(1, 100.0), tick(11, 101.0), tick(21, 102.01)])
            .unwrap();
        let selection = [tick(21, 102.01), tick(31, 103.0301), tick(41, 104.060401)];
        let mut isolated = HarHistory::default();
        isolated.append_ticks(&selection).unwrap();
        assert!(isolated.features(41, 10, &[1, 2, 3]).is_none());
        h.append_ticks(&selection).unwrap();
        for x in h.features(41, 10, &[1, 2, 3]).unwrap() {
            close(x, 1.0);
        }
        assert_eq!(h.ignored_overlap, 1);
        assert_eq!(h.seams, vec![(21, 31)]);
    }

    #[test]
    fn qlike_no_finitud_elimina_la_misma_fila_en_todos() {
        let c = calibration(vec![0.0, 1.0], vec![1], Some(0.0001));
        let r = compare_har(
            &c,
            &history(&[1.0, 2.0]),
            HarEvalSeries {
                ts: &[11, 21],
                y: &[1.0, 2.0],
                model: &[f64::NAN, 2.0],
                persist: &[1.0, 2.0],
                mean: 2.0,
                tau: 10,
            },
        );
        assert_eq!(r["invalid_rows"], 1);
        assert_eq!(r["qlike_common_rows"], 1);
        assert_eq!(r["qlike_evidence"], false);
        for name in ["modelo", "persistencia", "har", "media"] {
            close(r["metrics"][name]["qlike"].as_f64().unwrap(), 0.0);
        }
        assert!(har_qlike(f64::INFINITY, 1.0, 0.0001).is_none());
    }

    #[test]
    fn har_timestamp_duplicado_respeta_borde_izquierdo_abierto() {
        let mut h = HarHistory::default();
        h.append_ticks(&[
            tick(1, 100.0),
            tick(11, 101.0),
            tick(11, 102.01),
            tick(21, 103.0301),
        ])
        .unwrap();
        close(h.rv(1, 11).unwrap(), 2.0_f64.sqrt());
        close(h.rv(11, 21).unwrap(), 1.0);
        assert_eq!(h.points.len(), 3);
    }
}

#[cfg(test)]
mod early_stop_v1_tests {
    use super::*;

    fn sample(times: &[u64], horizon: u64) -> TrainingSamples {
        let mut samples = TrainingSamples::default();
        for &t in times {
            samples.push(vec![t as f32], t as f64, t, t + horizon);
        }
        samples
    }

    #[test]
    fn early_stop_cli_rejects_missing_duplicate_or_ambiguous_paths() {
        let args = |values: &[&str]| values.iter().map(|s| s.to_string()).collect::<Vec<_>>();
        assert_eq!(early_stop_path(&args(&["bin", "SYM"]), "", 0).unwrap(), "");
        assert_eq!(
            early_stop_path(&args(&["--early-stop-in", "early.bin"]), "val.bin", 2).unwrap(),
            "early.bin"
        );
        for values in [
            vec!["--early-stop-in"],
            vec!["--early-stop-in", "--val-in"],
            vec!["--early-stop-in", ""],
            vec!["--early-stop-in", "a", "--early-stop-in", "b"],
            vec!["--early-stop-in=a"],
        ] {
            assert!(early_stop_path(&args(&values), "val.bin", 2).is_err());
        }
        assert!(early_stop_path(&args(&["--early-stop-in", "a"]), "", 2).is_err());
        assert!(early_stop_path(&args(&["--early-stop-in", "a"]), "val.bin", 0).is_err());
    }

    #[test]
    fn early_stop_purges_train_at_equality_and_rejects_gate_overlap() {
        let early = sample(&[40, 50], 10);
        let (train, purged) =
            prepare_early_stop(sample(&[10, 20, 30], 10), &early, &sample(&[61, 71], 10)).unwrap();
        assert_eq!(purged, 1);
        assert_eq!(train.labels, vec![10.0, 20.0]);
        assert!(prepare_early_stop(sample(&[10, 20], 10), &early, &sample(&[60], 10)).is_err());
        assert!(prepare_early_stop(sample(&[40, 50], 10), &early, &sample(&[61], 10)).is_err());
        assert!(prepare_early_stop(
            sample(&[10], 10),
            &TrainingSamples::default(),
            &sample(&[61], 10)
        )
        .is_err());
    }

    #[test]
    fn early_stop_checks_each_tree_counts_patience_and_retains_best() {
        let mut stop = EarlyStopping::new();
        assert!(!stop.observe(4.0, 1, 1, 2));
        assert!(!stop.observe(1.0, 2, 1, 2));
        assert!(!stop.observe(1.0, 3, 1, 2));
        assert!(stop.observe(2.0, 4, 1, 2));
        assert_eq!(stop.best_rounds, 2);
        assert_eq!(stop.best_loss, 1.0);
        assert_eq!(stop.since_improve, 2);
        let mut reset = EarlyStopping::new();
        assert!(!reset.observe(4.0, 1, 1, 2));
        assert!(!reset.observe(5.0, 2, 1, 2));
        assert!(!reset.observe(3.0, 3, 1, 2));
        assert_eq!(reset.since_improve, 0);
        assert_eq!(reset.best_rounds, 3);
    }

    #[test]
    fn early_stop_gate_labels_do_not_choose_trees_and_frozen_predictions_are_retained() {
        // Fixture: el mejor arbol para early es el primero; para gate seria el segundo.
        let leaf = || {
            vec![TreeNode {
                feature: -1,
                threshold: 0.0,
                left: -1,
                right: -1,
                value: 1.0,
            }]
        };
        let x = vec![vec![0.0; DIM_VECTOR]];
        let early_y = 1.0;
        let mut retained = Vec::new();
        for gate_y in [2.0, -3.0] {
            let mut trees = Vec::new();
            let mut stop = EarlyStopping::new();
            for round in 1..=2 {
                trees.push(leaf());
                let prediction = forest_raw(0.0, 1.0, &trees, &x[0]);
                if stop.observe((early_y - prediction).powi(2), round, 1, 1) {
                    break;
                }
            }
            trees.truncate(stop.best_rounds);
            let model = serialize_forest(&trees, 0.0, 1.0).unwrap();
            let prediction = serving_predictions(&model, &x).unwrap()[0];
            assert_eq!(prediction, 1.0);
            retained.push((stop.best_rounds, (gate_y - prediction).powi(2)));
        }
        assert_eq!(retained, vec![(1, 1.0), (1, 16.0)]);
    }
}
