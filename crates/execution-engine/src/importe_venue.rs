//! CICLOS-1 (pedido 20261118-0535-cicloscompletos, espec. `ciclos_completos_v2.md` §2.3, opcion C),
//! corregido por CICLOS-3 (adenda 20261118-0547 del coordinador, punto 4, y cambio C6 de la N1 de
//! QODER-APP `qoder_app_n1_ciclos_v2.md`): importes de caja del venue (income `REALIZED_PNL`,
//! `COMMISSION`, `FUNDING_FEE`, la comision y el `rp` de un fill, el `walletBalance`) como DECIMAL
//! EXACTO CON SIGNO sobre el texto que entrega el venue.
//!
//! QUE: `importe_de_texto` / `importe_venue` devuelven `DecimalExacto` (mantisa entera con signo y
//! numero de cifras decimales QUE TRAE EL PROPIO TEXTO) y la cadena original tal como llego.
//! POR QUE: hoy esos importes se leen como `f64` (`order_types::string_or_f64`), que pierde la
//! representacion decimal del venue, y la cuadratura del certificado de ciclos es por igualdad
//! exacta, sin tolerancia. NO se elige ninguna escala: la de `user_data_stream::decimal_e8`
//! (1e-8) es de CANTIDAD, es SIN SIGNO y corta en 8 cifras; aplicada a dinero descartaria en
//! silencio comisiones y fundings pagados (QODER-APP 2.2). La escala monetaria canonica queda
//! pendiente de declaracion explicita y no se deduce de la de cantidades.
//! FAIL-CLOSED: ausente, vacio, no numerico, notacion exponencial, no finito, un numero JSON con
//! decimales (serde_json, sin `arbitrary_precision`, ya lo paso por `f64`: el texto se perdio) o
//! un importe que no se puede representar exacto => N/D con su causa. Nunca un cero inventado,
//! nunca un redondeo.

use std::cmp::Ordering;

/// Maximo de cifras decimales representables (10^38 cabe en `i128`).
pub const MAX_CIFRAS: u32 = 38;

/// Decimal exacto con signo: `mantisa * 10^-cifras`, normalizado (sin ceros de cola en la
/// mantisa; el cero es `(0, 0)`). Dos valores iguales tienen la misma representacion, asi que la
/// igualdad derivada es igualdad de VALOR. Sin division: solo suma, resta y comparacion exactas.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
pub struct DecimalExacto {
    mantisa: i128,
    cifras: u32,
}

impl DecimalExacto {
    pub const CERO: DecimalExacto = DecimalExacto { mantisa: 0, cifras: 0 };

    /// `None` si `cifras` excede lo representable.
    pub fn nuevo(mantisa: i128, cifras: u32) -> Option<Self> {
        if cifras > MAX_CIFRAS {
            return None;
        }
        let (mut m, mut c) = (mantisa, cifras);
        while c > 0 && m % 10 == 0 {
            m /= 10;
            c -= 1;
        }
        if m == 0 {
            c = 0;
        }
        Some(Self { mantisa: m, cifras: c })
    }

    pub fn mantisa(&self) -> i128 {
        self.mantisa
    }

    pub fn cifras(&self) -> u32 {
        self.cifras
    }

    /// Las dos mantisas a la misma cantidad de cifras (la mayor), sin perder nada; `None` si no cabe.
    fn alinear(a: Self, b: Self) -> Option<(i128, i128, u32)> {
        let c = a.cifras.max(b.cifras);
        let ma = a.mantisa.checked_mul(10i128.checked_pow(c - a.cifras)?)?;
        let mb = b.mantisa.checked_mul(10i128.checked_pow(c - b.cifras)?)?;
        Some((ma, mb, c))
    }

    pub fn checked_add(self, o: Self) -> Option<Self> {
        let (a, b, c) = Self::alinear(self, o)?;
        Self::nuevo(a.checked_add(b)?, c)
    }

    pub fn checked_sub(self, o: Self) -> Option<Self> {
        let (a, b, c) = Self::alinear(self, o)?;
        Self::nuevo(a.checked_sub(b)?, c)
    }

    /// Comparacion exacta; `None` solo si alinear desborda.
    pub fn cmp_exacto(&self, o: &Self) -> Option<Ordering> {
        let (a, b, _) = Self::alinear(*self, *o)?;
        Some(a.cmp(&b))
    }

    pub fn signo(&self) -> Ordering {
        self.mantisa.cmp(&0)
    }

    /// El valor en unidades de `10^-cifras` si es EXACTO a esa escala (sin redondeo); si no, `None`.
    pub fn en_unidades(&self, cifras: u32) -> Option<i128> {
        if cifras < self.cifras {
            return None;
        }
        self.mantisa.checked_mul(10i128.checked_pow(cifras - self.cifras)?)
    }

    /// Texto canonico exacto (`-0.015`, `3`, `0.123456789`).
    pub fn texto(&self) -> String {
        let neg = self.mantisa < 0;
        let digitos = self.mantisa.unsigned_abs().to_string();
        let c = self.cifras as usize;
        let cuerpo = if c == 0 {
            digitos
        } else if digitos.len() > c {
            format!("{}.{}", &digitos[..digitos.len() - c], &digitos[digitos.len() - c..])
        } else {
            format!("0.{}{}", "0".repeat(c - digitos.len()), digitos)
        };
        if neg {
            format!("-{cuerpo}")
        } else {
            cuerpo
        }
    }

    /// Solo para presentacion y para los consumidores que ya leen `f64` (el EV de la puerta).
    pub fn como_f64(&self) -> f64 {
        self.texto().parse::<f64>().unwrap_or(f64::NAN)
    }
}

impl serde::Serialize for DecimalExacto {
    fn serialize<S: serde::Serializer>(&self, s: S) -> Result<S::Ok, S::Error> {
        s.serialize_str(&self.texto())
    }
}

impl<'de> serde::Deserialize<'de> for DecimalExacto {
    fn deserialize<D: serde::Deserializer<'de>>(d: D) -> Result<Self, D::Error> {
        let t = String::deserialize(d)?;
        match importe_de_texto(&t) {
            Ok(i) if i.valor.texto() == t => Ok(i.valor),
            Ok(_) => Err(serde::de::Error::custom("decimal no canonico")),
            Err(c) => Err(serde::de::Error::custom(format!("decimal N/D: {c:?}"))),
        }
    }
}

/// Un importe del venue, exacto, con la cadena original.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct ImporteVenue {
    pub valor: DecimalExacto,
    /// Texto tal como llego del venue.
    pub original: String,
}

/// Por que un importe es N/D.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum CausaImporteNd {
    /// El campo no vino.
    Ausente,
    /// Vino `null`, booleano, arreglo u objeto.
    TipoNoNumerico,
    /// Un numero JSON con decimales: serde_json ya lo convirtio a `f64` y el texto del venue se perdio.
    NumeroJsonSinTexto,
    /// Cadena vacia o solo espacios.
    Vacio,
    /// No es un decimal plano (letras, exponente, varios puntos, signo suelto...).
    Formato,
    /// `NaN`, `inf`, `Infinity` y sus variantes.
    NoFinito,
    /// No se puede representar exacto (mas de `MAX_CIFRAS` decimales o mantisa fuera de `i128`).
    FueraDeRango,
}

impl ImporteVenue {
    /// Solo para presentacion.
    pub fn como_f64(&self) -> f64 {
        self.valor.como_f64()
    }
}

/// Importe de un campo del evento o de la respuesta REST. El venue entrega los importes como
/// cadena (`"-0.015"`); un entero JSON es exacto; un numero JSON con decimales es N/D.
pub fn importe_venue(v: Option<&serde_json::Value>) -> Result<ImporteVenue, CausaImporteNd> {
    match v {
        None => Err(CausaImporteNd::Ausente),
        Some(serde_json::Value::String(s)) => importe_de_texto(s),
        Some(serde_json::Value::Number(n)) if n.is_i64() || n.is_u64() => importe_de_texto(&n.to_string()),
        Some(serde_json::Value::Number(_)) => Err(CausaImporteNd::NumeroJsonSinTexto),
        Some(_) => Err(CausaImporteNd::TipoNoNumerico),
    }
}

/// Importe desde su texto decimal, exacto, con las cifras que trae el texto.
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
    // Los ceros de cola no cambian el valor; se quitan antes de medir si cabe.
    let fraccion = fraccion.trim_end_matches('0');
    let cifras = u32::try_from(fraccion.len()).map_err(|_| CausaImporteNd::FueraDeRango)?;
    if cifras > MAX_CIFRAS {
        return Err(CausaImporteNd::FueraDeRango);
    }
    let digitos = format!("{entero}{fraccion}");
    let digitos = digitos.trim_start_matches('0');
    let magnitud: i128 = if digitos.is_empty() {
        0
    } else {
        digitos.parse().map_err(|_| CausaImporteNd::FueraDeRango)?
    };
    let mantisa = if negativo { -magnitud } else { magnitud };
    let valor = DecimalExacto::nuevo(mantisa, cifras).ok_or(CausaImporteNd::FueraDeRango)?;
    Ok(ImporteVenue { valor, original })
}

/// Fila de `/fapi/v1/income` con el importe exacto (el `IncomeEntry` legado lo lee como `f64`).
/// `asset` ausente queda vacio: NUNCA se supone USDT (QODER-APP C7; el lector legado
/// `parse_income_jsonl` lo supone y no se usa para el certificado).
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
            importe: importe_venue(v.get("income")),
            asset: texto("asset"),
            time: entero("time"),
            tran_id: entero("tranId"),
            trade_id: texto("tradeId"),
        }
    }
}
