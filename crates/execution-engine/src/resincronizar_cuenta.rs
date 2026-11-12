//! CAPITAL-UNICO V3 (dictamen de Codex QRA, P1.2 y P1.3) — resincronización
//! REST del puente de la cuenta FUERA del lazo del sistema inmune.
//!
//! En 1757709c4 el inmune esperaba `/fapi/v2/account` dentro de su ciclo:
//! un GET lento retrasaba la evaluación del capital y de la latencia, y con
//! ella los cierres. Ahora el host corre esta pasada en su propia tarea; el
//! inmune sólo lee el puente (un mutex, sin E/S).
//!
//! Una pasada: ticket de la era vigente → GET con tope de tiempo → lectura al
//! puente (frescura por componente) → `publicar` bajo el mutex del puente si
//! la era sigue vigente. Un fallo (GET, tope, rechazo) no cambia el puente:
//! sigue vetando entradas y el host reintenta con [`Pausa`].

use crate::puente_cuenta::{LecturaCuenta, Observacion, PuenteCuenta};
use std::future::Future;
use std::time::Duration;

/// Una pasada de resincronización. `Err` con el motivo; el puente sólo cambia
/// si la lectura se acepta.
pub async fn resincronizar_una_vez<F>(
    puente: &PuenteCuenta,
    lectura: F,
    tope: Duration,
    publicar: impl FnOnce(&Observacion),
) -> Result<Observacion, String>
where
    F: Future<Output = Result<LecturaCuenta, String>>,
{
    let ticket = puente.ticket();
    let l = match tokio::time::timeout(tope, lectura).await {
        Ok(Ok(l)) => l,
        Ok(Err(e)) => return Err(format!("GET fallido: {e}")),
        Err(_) => return Err(format!("GET sin respuesta en {} ms", tope.as_millis())),
    };
    // El cero explícito de 3a también entra: es el plano de seguridad.
    let obs = puente.observar_rest(ticket, &l).map_err(|e| e.to_string())?;
    // Se publica lo VIGENTE bajo el mutex (un evento pudo llegar en medio).
    puente
        .con_era_vigente(ticket.era(), |vigente| publicar(&vigente.unwrap_or(obs)))
        .ok_or_else(|| "la era cambió durante la lectura".to_string())?;
    Ok(obs)
}

/// Espera entre pasadas: 1 s en reposo; tras un fallo 5 s, doblando hasta 30 s.
#[derive(Debug, Clone)]
pub struct Pausa {
    actual: Duration,
}

impl Pausa {
    pub const REPOSO: Duration = Duration::from_secs(1);
    pub const PRIMER_FALLO: Duration = Duration::from_secs(5);
    pub const MAXIMA: Duration = Duration::from_secs(30);

    pub fn nueva() -> Self {
        Self { actual: Self::REPOSO }
    }

    pub fn actual(&self) -> Duration {
        self.actual
    }

    pub fn exito(&mut self) {
        self.actual = Self::REPOSO;
    }

    pub fn fallo(&mut self) {
        self.actual = if self.actual < Self::PRIMER_FALLO {
            Self::PRIMER_FALLO
        } else {
            (self.actual * 2).min(Self::MAXIMA)
        };
    }
}
