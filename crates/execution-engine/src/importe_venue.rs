//! CICLOS-1 (pedido 20261118-0535-cicloscompletos, espec. `ciclos_completos_v2.md` §2.3, opcion C):
//! importes de caja del venue (income `REALIZED_PNL`, `COMMISSION`, `FUNDING_FEE`, la comision y
//! el `rp` de un fill) en punto fijo, con la MISMA escala que las cantidades del stream
//! (`user_data_stream::decimal_e8`: unidades de 1e-8, hasta ocho cifras decimales).
//!
//! QUE: `importe_de_texto` / `importe_venue_e8` devuelven el entero en unidades de 1e-8 y la
//! cadena original tal como llego. A diferencia de `decimal_e8`, el importe lleva signo (un
//! income negativo es un pago; una comision positiva, un rebate).
//! POR QUE: hoy esos importes se leen como `f64` (`order_types::string_or_f64`), que pierde la
//! representacion decimal del venue; la cuadratura del certificado de ciclos es por igualdad
//! exacta, sin tolerancia (decision del coordinador; la tolerancia de paridad interna de
//! `audit-engine/src/state_validator.rs` NO se reutiliza).
//! FAIL-CLOSED: ausente, vacio, no numerico, notacion exponencial, mas cifras decimales distintas
//! de cero que la escala, no finito o fuera de rango => N/D con su causa. Nunca un cero inventado.

/// Unidades de 1e-8 por unidad del activo: la escala de `user_data_stream::decimal_e8`.
pub const UNIDADES_POR_ACTIVO: i64 = 100_000_000;
/// Cifras decimales que admite la escala.
pub const CIFRAS_DECIMALES: usize = 8;

/// Un importe del venue en punto fijo, con la cadena original.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct ImporteVenue {
    /// Importe firmado en unidades de 1e-8.
    pub e8: i64,
    /// Texto tal como llego del venue (para un numero JSON, su forma textual de serde_json).
    pub original: String,
}

/// Por que un importe es N/D.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum CausaImporteNd {
    /// El campo no vino.
    Ausente,
    /// Vino `null`, booleano, arreglo u objeto.
    TipoNoNumerico,
    /// Cadena vacia o solo espacios.
    Vacio,
    /// No es un decimal plano (letras, exponente, varios puntos, signo suelto...).
    Formato,
    /// `NaN`, `inf`, `Infinity` y sus variantes.
    NoFinito,
    /// Mas cifras decimales distintas de cero de las que admite la escala de 1e-8.
    MasDecimalesQueLaEscala,
    /// No cabe en `i64` a la escala de 1e-8.
    FueraDeRango,
}

impl ImporteVenue {
    /// El importe como `f64` (solo para presentacion y para los consumidores que ya leen `f64`).
    pub fn como_f64(&self) -> f64 {
        self.e8 as f64 / UNIDADES_POR_ACTIVO as f64
    }
}

/// Importe de un campo del evento o de la respuesta REST (`"−0.015"` o `-0.015`).
pub fn importe_venue_e8(v: Option<&serde_json::Value>) -> Result<ImporteVenue, CausaImporteNd> {
    match v {
        None => Err(CausaImporteNd::Ausente),
        Some(serde_json::Value::String(s)) => importe_de_texto(s),
        Some(serde_json::Value::Number(n)) => importe_de_texto(&n.to_string()),
        Some(_) => Err(CausaImporteNd::TipoNoNumerico),
    }
}

/// Importe desde su texto decimal, exacto a 1e-8.
pub fn importe_de_texto(texto: &str) -> Result<ImporteVenue, CausaImporteNd> {
    let original = texto.to_string();
    let t = texto.trim();
    if t.is_empty() {
        return Err(CausaImporteNd::Vacio);
    }
    let (negativo, cuerpo) = match t.as_bytes()[0] {
        b'-' => (true, &t[1..]),
        b'+' => (false, &t[1..]),
        _ => (false, t),
    };
    let minus = cuerpo.to_ascii_lowercase();
    if matches!(minus.as_str(), "nan" | "inf" | "infinity") {
        return Err(CausaImporteNd::NoFinito);
    }
    let (entero, fraccion) = cuerpo.split_once('.').unwrap_or((cuerpo, ""));
    if entero.is_empty()
        || !entero.bytes().all(|b| b.is_ascii_digit())
        || !fraccion.bytes().all(|b| b.is_ascii_digit())
        || (cuerpo.contains('.') && fraccion.is_empty())
    {
        return Err(CausaImporteNd::Formato);
    }
    let (cifras, resto) = fraccion.split_at(fraccion.len().min(CIFRAS_DECIMALES));
    if resto.bytes().any(|b| b != b'0') {
        return Err(CausaImporteNd::MasDecimalesQueLaEscala);
    }
    let mut frac = cifras.to_string();
    while frac.len() < CIFRAS_DECIMALES {
        frac.push('0');
    }
    let ent: i64 = entero.parse().map_err(|_| CausaImporteNd::FueraDeRango)?;
    let fr: i64 = frac.parse().map_err(|_| CausaImporteNd::FueraDeRango)?;
    let magnitud = ent
        .checked_mul(UNIDADES_POR_ACTIVO)
        .and_then(|x| x.checked_add(fr))
        .ok_or(CausaImporteNd::FueraDeRango)?;
    let e8 = if negativo { -magnitud } else { magnitud };
    Ok(ImporteVenue { e8, original })
}

/// Fila de `/fapi/v1/income` con el importe exacto (el `IncomeEntry` legado lo lee como `f64`).
#[derive(Debug, Clone, PartialEq)]
pub struct FilaIncomeExacta {
    pub symbol: String,
    pub income_type: String,
    pub importe: Result<ImporteVenue, CausaImporteNd>,
    pub asset: String,
    pub time: u64,
    pub tran_id: u64,
    pub trade_id: String,
}

impl FilaIncomeExacta {
    /// Lee una fila cruda de la respuesta. Sin `income` legible la fila sigue, con su causa N/D.
    pub fn desde_valor(v: &serde_json::Value) -> Self {
        let texto = |k: &str| match v.get(k) {
            Some(serde_json::Value::String(s)) => s.clone(),
            Some(serde_json::Value::Number(n)) => n.to_string(),
            _ => String::new(),
        };
        let entero = |k: &str| match v.get(k) {
            Some(serde_json::Value::Number(n)) => n.as_u64().unwrap_or(0),
            Some(serde_json::Value::String(s)) => s.trim().parse().unwrap_or(0),
            _ => 0,
        };
        Self {
            symbol: texto("symbol"),
            income_type: texto("incomeType"),
            importe: importe_venue_e8(v.get("income")),
            asset: texto("asset"),
            time: entero("time"),
            tran_id: entero("tranId"),
            trade_id: texto("tradeId"),
        }
    }
}
