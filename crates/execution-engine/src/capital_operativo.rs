//! CAPITAL-UNICO (b) — tope de capital operativo `TG_CAPITAL_OPERATIVO`
//! (decisión del coordinador del 2026-11-11, 16:09).
//!
//! Para que la Demo se comporte como una cuenta pequeña (por ejemplo 50 USDT)
//! sobre una Testnet con 1 655,81 USDT, el capital que ven el arena y el freno
//! se limita en UN solo punto: [`PuenteCapital::capital`], por el que pasan la
//! lectura REST (`fetch_account_balance`) y el user-data stream
//! (`ACCOUNT_UPDATE` → `on_capital`).
//!
//! - Apagado (variable ausente): el capital es el saldo real, idéntico a antes.
//! - Encendido: `capital = min(saldo, tope + (saldo − ancla))`, recortado a 0,
//!   con `ancla` = el primer saldo observado (el del arranque). Es el tope como
//!   asignación: al arrancar vale `min(saldo, tope)`, nunca más que el saldo
//!   real, y una pérdida real lo baja en la misma cantidad, así que el freno la
//!   ve sobre la cuenta pequeña. Con un `min(saldo, tope)` puro, una pérdida de
//!   100 USDT sobre 1 655,81 dejaba el capital en 50: el freno no la vería
//!   hasta perder más de 1 605 USDT.
//! - Valor no numérico, no finito, cero o negativo: falla cerrado. El host no
//!   arranca, REST devuelve error y el stream no publica capital.
//!
//! Pico y tope: el pico se guarda en la unidad del capital operativo. Junto al
//! archivo del pico se anota el tope y el ancla ([`Anotacion`]); si al arrancar
//! el tope cambió, el pico guardado no vale ([`decidir_pico`]) y se reinicia con
//! una línea de aviso, para que un pico de otra corrida no dispare el freno en
//! falso. Con el mismo tope se restaura el ancla y la caída acumulada sobrevive
//! al reinicio (CL-99).
//!
//! No toca d*, `capital_del_freno` ni ningún umbral.

use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::OnceLock;

/// Variable de entorno del tope.
pub const VARIABLE_TOPE: &str = "TG_CAPITAL_OPERATIVO";

/// Interpreta el valor de [`VARIABLE_TOPE`]. `None` = apagado.
pub fn interpretar_tope(valor: Option<&str>) -> Result<Option<f64>, String> {
    let Some(crudo) = valor else {
        return Ok(None);
    };
    match crudo.trim().parse::<f64>() {
        Ok(t) if t.is_finite() && t > 0.0 => Ok(Some(t)),
        _ => Err(format!(
            "{VARIABLE_TOPE}={crudo:?} no es válido: debe ser un número de USDT mayor que 0 \
             (o no estar definida para operar con el saldo real). Falla cerrado: sin capital no se opera."
        )),
    }
}

/// Lee [`VARIABLE_TOPE`] del entorno del proceso.
pub fn tope_de_entorno() -> Result<Option<f64>, String> {
    match std::env::var(VARIABLE_TOPE) {
        Ok(v) => interpretar_tope(Some(&v)),
        Err(std::env::VarError::NotPresent) => Ok(None),
        Err(std::env::VarError::NotUnicode(_)) => interpretar_tope(Some("<no unicode>")),
    }
}

/// El puente único entre el saldo real de USDT y el capital que ven el arena
/// y el freno.
#[derive(Debug)]
pub struct PuenteCapital {
    tope: Option<f64>,
    /// `f64` en bits; NaN = todavía sin ancla.
    ancla: AtomicU64,
}

impl PuenteCapital {
    /// Puente con `tope` (`None` = apagado).
    pub fn new(tope: Option<f64>) -> Self {
        Self {
            tope,
            ancla: AtomicU64::new(f64::NAN.to_bits()),
        }
    }

    pub fn tope(&self) -> Option<f64> {
        self.tope
    }

    /// Saldo real de USDT con que se ancló el tope (`None` si no hay).
    pub fn ancla(&self) -> Option<f64> {
        let a = f64::from_bits(self.ancla.load(Ordering::Acquire));
        a.is_finite().then_some(a)
    }

    /// Restaura el ancla anotada junto al pico (mismo tope que la corrida
    /// anterior). Sólo antes de la primera observación; un valor no finito o
    /// ≤ 0 se ignora.
    pub fn restaurar_ancla(&self, ancla: f64) -> bool {
        if self.tope.is_none() || !(ancla.is_finite() && ancla > 0.0) {
            return false;
        }
        self.ancla
            .compare_exchange(
                f64::NAN.to_bits(),
                ancla.to_bits(),
                Ordering::AcqRel,
                Ordering::Acquire,
            )
            .is_ok()
    }

    /// El capital operativo para un saldo real observado (REST o stream).
    /// Nunca mayor que `saldo`.
    pub fn capital(&self, saldo: f64) -> f64 {
        let Some(tope) = self.tope else {
            return saldo;
        };
        if !saldo.is_finite() {
            return saldo;
        }
        let ancla = match self.ancla.compare_exchange(
            f64::NAN.to_bits(),
            saldo.to_bits(),
            Ordering::AcqRel,
            Ordering::Acquire,
        ) {
            Ok(_) => saldo,
            Err(previa) => f64::from_bits(previa),
        };
        (tope + (saldo - ancla)).min(saldo).max(0.0)
    }
}

static PUENTE: OnceLock<Result<PuenteCapital, String>> = OnceLock::new();

/// El puente del proceso, con el tope de [`VARIABLE_TOPE`] leído una vez.
/// `Err` (valor inválido) = falla cerrado en todos los que lo usan.
pub fn puente_global() -> Result<&'static PuenteCapital, &'static str> {
    PUENTE
        .get_or_init(|| tope_de_entorno().map(PuenteCapital::new))
        .as_ref()
        .map_err(String::as_str)
}

/// Lo que se anota junto al pico: con qué tope (y ancla) se midió.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Anotacion {
    pub tope: Option<f64>,
    pub ancla: Option<f64>,
}

impl Anotacion {
    pub fn a_texto(&self) -> String {
        match (self.tope, self.ancla) {
            (None, _) => "tope=apagado\n".to_string(),
            (Some(t), Some(a)) => format!("tope={t} ancla={a}\n"),
            (Some(t), None) => format!("tope={t}\n"),
        }
    }

    pub fn desde_texto(texto: &str) -> Result<Self, String> {
        let mut tope = None;
        let mut ancla = None;
        let mut visto_tope = false;
        for parte in texto.split_whitespace() {
            let (clave, valor) = parte
                .split_once('=')
                .ok_or_else(|| format!("campo sin '=': {parte:?}"))?;
            let numero = |v: &str| -> Result<f64, String> {
                v.parse::<f64>()
                    .ok()
                    .filter(|x| x.is_finite() && *x > 0.0)
                    .ok_or_else(|| format!("{clave}={v:?} no es un número > 0"))
            };
            match clave {
                "tope" => {
                    visto_tope = true;
                    tope = if valor == "apagado" { None } else { Some(numero(valor)?) };
                }
                "ancla" => ancla = Some(numero(valor)?),
                otra => return Err(format!("campo desconocido {otra:?}")),
            }
        }
        if !visto_tope {
            return Err("falta tope=".into());
        }
        Ok(Self { tope, ancla })
    }
}

/// `pico_capital_prueba_usdt.txt` → `pico_capital_prueba_usdt.tope.txt`.
pub fn ruta_anotacion(ruta_pico: &Path) -> PathBuf {
    ruta_pico.with_extension("tope.txt")
}

/// `Ok(None)` si no hay anotación; `Err` si existe y no se puede leer.
pub fn leer_anotacion(ruta: &Path) -> Result<Option<Anotacion>, String> {
    match std::fs::read_to_string(ruta) {
        Ok(t) => Anotacion::desde_texto(&t)
            .map(Some)
            .map_err(|e| format!("{}: {e}", ruta.display())),
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => Ok(None),
        Err(e) => Err(format!("{}: no se puede leer: {e}", ruta.display())),
    }
}

/// Escribe la anotación (temporal y renombrado, como el pico).
pub fn guardar_anotacion(ruta: &Path, anotacion: &Anotacion) -> std::io::Result<()> {
    let tmp = ruta.with_extension("tope.txt.tmp");
    std::fs::write(&tmp, anotacion.a_texto())?;
    std::fs::rename(&tmp, ruta)
}

/// Qué hacer con el pico guardado según el tope con que se midió.
#[derive(Debug, Clone, PartialEq)]
pub enum DecisionPico {
    /// Mismo tope: el pico vale; si hay ancla, se restaura en el puente.
    /// `anotada == false`: no había anotación (primer arranque con la
    /// definición `_usdt`). Si además el pico falta, es una MIGRACIÓN, no un
    /// retiro del operador: el freno en stops (en R, no depende de la unidad
    /// del capital) NO se rearma (revisión independiente, 7a).
    Conservar { ancla: Option<f64>, anotada: bool },
    /// Otro tope: el pico guardado está en otra escala; se reinicia.
    Reiniciar { tope_anotado: Option<f64> },
    /// La anotación existe pero no se puede leer: como un pico ilegible.
    Ilegible(String),
}

/// Sin anotación = el pico se midió con el tope apagado (archivos de antes
/// de este cambio).
pub fn decidir_pico(anotacion: Result<Option<Anotacion>, String>, tope_actual: Option<f64>) -> DecisionPico {
    let (anotada, hay_anotacion) = match anotacion {
        Err(e) => return DecisionPico::Ilegible(e),
        Ok(a) => (a.unwrap_or(Anotacion { tope: None, ancla: None }), a.is_some()),
    };
    let mismo = match (anotada.tope, tope_actual) {
        (None, None) => true,
        (Some(a), Some(b)) => a.to_bits() == b.to_bits(),
        _ => false,
    };
    if mismo {
        DecisionPico::Conservar {
            ancla: anotada.ancla,
            anotada: hay_anotacion,
        }
    } else {
        DecisionPico::Reiniciar { tope_anotado: anotada.tope }
    }
}

impl DecisionPico {
    /// Si el pico falta, ¿lo retiró el operador (rearme de los stops, F1-011)?
    /// Sólo con una anotación del mismo tope: sin anotación es la migración a
    /// la definición `_usdt`, y con otro tope es un reinicio por escala; en los
    /// dos casos el freno en stops se conserva.
    pub fn ausente_es_retiro_del_operador(&self) -> bool {
        matches!(self, DecisionPico::Conservar { anotada: true, .. })
    }
}

/// Texto del tope para los avisos.
pub fn describir_tope(tope: Option<f64>) -> String {
    match tope {
        Some(t) => format!("{t} USDT"),
        None => "apagado".into(),
    }
}
