//! D-751 — EL DEFECTO DEL CODIFICADOR SIGUE CONGELADO EN LOS FICHEROS.
//!
//! `binance_vision_sync` construía el libro sintético de cada aggTrade con un
//! semi-spread de `max(price · 0,00005; 0,01)`: un suelo de UN CENTAVO en
//! términos ABSOLUTOS. Para BTCUSDT a 60 000 $ eso es despreciable (1,7e-7),
//! pero para ATOMUSDT a 1,95 $ son 0,01/1,955 = **1,02 % de spread**, y para un
//! activo de 0,45 $, un 4,4 %. D-723 corrigió el codificador —el suelo pasó a
//! ser relativo— pero los ficheros ya escritos NO cambiaron: el histórico de
//! todas las monedas baratas del roster sigue declarando un spread fabricado
//! de entre el 1 % y el 4 %.
//!
//! Consecuencia medida: el motor, correctamente, NO OPERA con esos spreads —su
//! filtro exige `spread ≤ (ATR·0,25) acotado a [6, 25] pb—, así que un forense
//! sobre esos tapes devuelve CERO operaciones en un mes entero de datos y
//! cualquier conclusión sacada de ahí («el genoma no encuentra entradas», «el
//! motor está paralizado») es un artefacto del fichero, no del motor.
//!
//! Esta herramienta reescribe un tape conservando lo que es DATO —instante,
//! precio medio, cantidad y lado del agresor— y recalculando sólo lo que era
//! MODELO: el semi-spread. El nuevo es `max(price · 0,00005; tick/2)`, es
//! decir un punto básico o medio tick del instrumento, lo que sea mayor:
//! ningún libro real cotiza más fino que su propia rejilla de precios.
//!
//! Uso: tape_spread_fix <entrada.bin> <salida.bin> [tick_size]
//!      (sin tick_size se usa sólo el suelo relativo de 1 pb)

use std::fs::File;
use std::io::{BufWriter, Read, Write};

#[repr(C)]
#[derive(Clone, Copy)]
struct BinTick {
    timestamp: u64,
    bid_price: f64,
    ask_price: f64,
    bid_qty: f64,
    ask_qty: f64,
}

const MAGIC: &[u8; 8] = b"TGMTICK1";
/// Semi-spread relativo mínimo: un punto básico. Es el que ya usa el
/// codificador corregido (D-723).
const HALF_SPREAD_REL: f64 = 0.00005;

fn main() {
    let args: Vec<String> = std::env::args().collect();
    if args.len() < 3 {
        eprintln!("uso: tape_spread_fix <entrada.bin> <salida.bin> [tick_size]");
        std::process::exit(1);
    }
    let tick_size: f64 = args.get(3).and_then(|s| s.parse().ok()).unwrap_or(0.0);

    let mut entrada = File::open(&args[1]).expect("no se pudo abrir la entrada");
    let mut cabecera = [0u8; 8];
    entrada.read_exact(&mut cabecera).expect("cabecera");
    if &cabecera != MAGIC {
        eprintln!(
            "❌ {} no lleva la cabecera TGMTICK1 (aggTrades reales): no se reescribe",
            args[1]
        );
        std::process::exit(2);
    }
    let mut datos = Vec::new();
    entrada.read_to_end(&mut datos).expect("lectura");
    let rec = std::mem::size_of::<BinTick>();
    if datos.len() % rec != 0 {
        eprintln!("❌ tamaño no múltiplo del registro ({} bytes)", datos.len());
        std::process::exit(3);
    }
    let n = datos.len() / rec;
    let ticks: &[BinTick] =
        unsafe { std::slice::from_raw_parts(datos.as_ptr() as *const BinTick, n) };

    let mut salida = BufWriter::new(File::create(&args[2]).expect("no se pudo crear la salida"));
    salida.write_all(MAGIC).expect("cabecera de salida");

    let (mut peor_antes, mut peor_despues) = (0.0f64, 0.0f64);
    let (mut suma_antes, mut suma_despues) = (0.0f64, 0.0f64);
    let mut escritos = 0u64;
    for t in ticks {
        let mid = (t.bid_price + t.ask_price) * 0.5;
        if !(mid > 0.0) || !t.bid_qty.is_finite() || !t.ask_qty.is_finite() {
            continue;
        }
        // El lado del agresor lleva `qty + base`; el pasivo, sólo `base`.
        let antes = (t.ask_price - t.bid_price) / mid;
        let half = (mid * HALF_SPREAD_REL).max(tick_size * 0.5);
        let despues = 2.0 * half / mid;
        peor_antes = peor_antes.max(antes);
        peor_despues = peor_despues.max(despues);
        suma_antes += antes;
        suma_despues += despues;
        let nuevo = BinTick {
            timestamp: t.timestamp,
            bid_price: mid - half,
            ask_price: mid + half,
            bid_qty: t.bid_qty,
            ask_qty: t.ask_qty,
        };
        let bytes: [u8; std::mem::size_of::<BinTick>()] =
            unsafe { std::mem::transmute_copy(&nuevo) };
        salida.write_all(&bytes).expect("escritura");
        escritos += 1;
    }
    salida.flush().expect("flush");
    let den = escritos.max(1) as f64;
    println!(
        "📼 {} → {} · {} ticks · spread medio {:.4} % → {:.4} % · peor {:.4} % → {:.4} %{}",
        args[1],
        args[2],
        escritos,
        100.0 * suma_antes / den,
        100.0 * suma_despues / den,
        100.0 * peor_antes,
        100.0 * peor_despues,
        if tick_size > 0.0 {
            format!(" · suelo de medio tick {}", tick_size * 0.5)
        } else {
            String::new()
        }
    );
}
