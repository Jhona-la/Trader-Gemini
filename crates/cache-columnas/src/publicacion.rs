//! Momento de PUBLICACION (`pub_ms`) por fuente: la UNICA tabla de mapeo del crate (pedido v3, P4).
//!
//! Cada fila cita su fuente en el inventario de variables de VIBE (`20261116-1725-vibe-variables\informe.md`, columna
//! «Disponibilidad»). Editarla exige citar el inventario. Solo dos reglas tienen formula cerrada (el dia D se publica
//! al empezar D+1); el resto exige que el productor entregue el `pub_ms` de cada fila (no se inventa un retraso).

use crate::MS_POR_DIA;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Regla {
    /// `pub_ms` = `E` del sobre del mensaje (hora de transmision del exchange); la da el productor por fila.
    EnvioDelExchange,
    /// El dia D no es visible hasta el inicio del dia D+1 (UTC).
    DiaSiguiente,
    /// Cierre del dia t-1: el valor del dia D se conoce al empezar D+1 (UTC).
    CierreDiaAnterior,
    /// Cada 15 m, publicado al minuto siguiente; la da el productor por fila.
    QuinceMinutosAlMinutoSiguiente,
    /// Diario; la da el productor por fila.
    Diaria,
    /// Al instante (250 ms); la da el productor por fila.
    AlInstante,
}

impl Regla {
    pub fn nombre(self) -> &'static str {
        match self {
            Regla::EnvioDelExchange => "envio_del_exchange",
            Regla::DiaSiguiente => "dia_siguiente",
            Regla::CierreDiaAnterior => "cierre_dia_anterior",
            Regla::QuinceMinutosAlMinutoSiguiente => "15m_al_minuto_siguiente",
            Regla::Diaria => "diaria",
            Regla::AlInstante => "al_instante",
        }
    }
}

#[derive(Debug, Clone, Copy)]
pub struct Fuente {
    pub nombre: &'static str,
    pub regla: Regla,
    pub cita: &'static str,
}

/// Tabla de publicacion (pedido v3, P4).
pub const TABLA: &[Fuente] = &[
    Fuente {
        nombre: "tgl4-captura",
        regla: Regla::EnvioDelExchange,
        cita: "LEEME_columnas:42-45 + inventario vibe (Instantaneo / Al instante): pub_ms = E",
    },
    Fuente {
        nombre: "tgl4-vision",
        regla: Regla::DiaSiguiente,
        cita: "inventario vibe, fila Ticks historicos .zip: Al dia siguiente",
    },
    Fuente {
        nombre: "tgb-reales",
        regla: Regla::DiaSiguiente,
        cita: "inventario vibe, fila Ticks .bin: Al final del dia = disponible al dia siguiente (CAMBIO v2)",
    },
    Fuente {
        nombre: "macro-yahoo",
        regla: Regla::CierreDiaAnterior,
        cita: "inventario vibe, filas Yahoo: t-1 (dia anterior)",
    },
    Fuente {
        nombre: "dvol-deribit",
        regla: Regla::QuinceMinutosAlMinutoSiguiente,
        cita: "inventario vibe, fila DVOL",
    },
    Fuente { nombre: "funding-okx-fng", regla: Regla::Diaria, cita: "inventario vibe, filas 172" },
    Fuente { nombre: "omni-telemetria", regla: Regla::AlInstante, cita: "inventario vibe, fila Omni" },
];

pub fn fuente(nombre: &str) -> Option<&'static Fuente> {
    TABLA.iter().find(|f| f.nombre == nombre)
}

/// `pub_ms` por formula de la regla, o `None` si la regla exige el valor del productor.
pub fn por_regla(regla: Regla, ts_ms: u64) -> Result<Option<i64>, String> {
    match regla {
        Regla::DiaSiguiente | Regla::CierreDiaAnterior => {
            let dia_siguiente = (ts_ms / MS_POR_DIA)
                .checked_add(1)
                .and_then(|d| d.checked_mul(MS_POR_DIA))
                .ok_or_else(|| format!("ts {ts_ms} fuera de rango"))?;
            i64::try_from(dia_siguiente).map(Some).map_err(|_| format!("ts {ts_ms} fuera de rango"))
        }
        Regla::EnvioDelExchange | Regla::QuinceMinutosAlMinutoSiguiente | Regla::Diaria | Regla::AlInstante => Ok(None),
    }
}
