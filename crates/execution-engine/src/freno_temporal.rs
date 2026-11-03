//! I-8 (Línea I, fila N18 del censo C-F7b): el rechazo por freno TEMPORAL.
//!
//! Con un 429/418 del exchange, o con su cuota agotada, el ejecutor rechaza
//! en local, sin enviarla, toda petición que pase por su freno (N5/CL-178,
//! N20/CL-188): brackets, cierres, purgas y la lectura de posiciones
//! incluidas. Ese freno caduca solo (M4-C02). Antes, tras una entrada
//! llenada, los tres intentos de bracket, el cierre de emergencia y su
//! reintento se rechazaban así en milisegundos, y la escalada X-009 armaba el
//! kill-switch PERMANENTE, que sólo levanta un rearme humano, por un freno
//! que el exchange levanta solo.
//!
//! Aquí se decide sólo con el texto del rechazo. Esos textos los forman
//! `client.rs` y `executor.rs`, y las pruebas de este módulo los fijan
//! recorriendo el ejecutor real.

/// Prefijos de los rechazos por freno temporal. Todos caducan solos:
/// - `RATE_BRAKE`: el freno escalado (tercer 429) o el ban 418 (M4-C02).
/// - `RATE_LIMIT_COOLDOWN`: el Retry-After de un 429 (F1.8).
/// - `HTTP_429_RATE_LIMITED`, `HTTP_418_IP_BANNED`: la respuesta que arma el freno.
/// - `FRENO_ARMADO`: el mismo 429/418 en una ruta sin ejecutor (CL-189, CL-194).
/// - `RATE_LIMIT_EXHAUSTED_` y `RATE_LIMIT_HEADROOM_`: la cuota del exchange
///   (CL-179), que se libera al rodar su ventana de 10 s o de 1 min.
const PREFIJOS_FRENO: [&str; 7] = [
    "RATE_BRAKE",
    "RATE_LIMIT_COOLDOWN",
    "HTTP_429_RATE_LIMITED",
    "HTTP_418_IP_BANNED",
    "FRENO_ARMADO",
    "RATE_LIMIT_EXHAUSTED_",
    "RATE_LIMIT_HEADROOM_",
];

/// Las dos formas compuestas del error de `execute_oco_order` (OCO-F1, F1.9).
const AMBAS_PIERNAS: &str = "Ambas órdenes OCO fallaron. SL_err: ";
const SEPARADOR_TP: &str = " | TP_err: ";
const PIERNA_PARCIAL: &str = "OCO PARCIAL: pierna fallida tras retry (";

#[inline]
fn empieza_por_freno(texto: &str) -> bool {
    PREFIJOS_FRENO.iter().any(|p| texto.starts_with(p))
}

/// Si `err` es un rechazo por freno temporal.
///
/// Reconoce el rechazo directo y las dos formas compuestas del bracket:
/// «Ambas órdenes OCO fallaron. SL_err: … | TP_err: …», cuando las dos
/// piernas están frenadas, y «OCO PARCIAL: pierna fallida tras retry (…)»,
/// cuando la pierna caída lo está. Un código del exchange, un AMBIGUOUS, el
/// kill-switch o «Volumen 0» no son freno.
pub fn es_freno_temporal(err: &str) -> bool {
    if empieza_por_freno(err) {
        return true;
    }
    if let Some(piernas) = err.strip_prefix(AMBAS_PIERNAS) {
        let Some(i) = piernas.rfind(SEPARADOR_TP) else {
            return false;
        };
        return empieza_por_freno(&piernas[..i])
            && empieza_por_freno(&piernas[i + SEPARADOR_TP.len()..]);
    }
    err.strip_prefix(PIERNA_PARCIAL)
        .is_some_and(empieza_por_freno)
}

/// Qué hace el host cuando una entrada llenada agota sus intentos de bracket.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TrasBracketFallido {
    /// Todos los intentos los rechazó el freno temporal. Eso no prueba que el
    /// bracket sea inaplicable; sólo que el exchange pidió esperar. No hay
    /// cierre de emergencia, que el mismo freno rechazaría, ni kill-switch: la
    /// protección queda marcada como sucia y el vigilante (CL-44) la repone
    /// cuando acaba el freno. Mientras dura, la lectura de posiciones del
    /// vigilante también frena y la auditoría espera.
    DiferirAlVigilante,
    /// Algún intento lo rechazó el exchange de verdad, o no hubo ninguno: el
    /// protocolo X-009 de siempre.
    CierreDeEmergencia,
}

/// La decisión de I-8 con los errores de los intentos de bracket, en orden.
pub fn tras_bracket_fallido<S: AsRef<str>>(errores: &[S]) -> TrasBracketFallido {
    if !errores.is_empty() && errores.iter().all(|e| es_freno_temporal(e.as_ref())) {
        TrasBracketFallido::DiferirAlVigilante
    } else {
        TrasBracketFallido::CierreDeEmergencia
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::executor::{ExecutionProvider, OrderExecutor};

    const EXECUTOR: &str = include_str!("executor.rs");
    const CLIENT: &str = include_str!("client.rs");

    #[test]
    fn i8_los_rechazos_directos_del_freno_son_temporales() {
        for e in [
            "RATE_BRAKE: freno temporal activo hasta 2 (ahora 1)",
            "RATE_LIMIT_COOLDOWN: activo hasta 2 (ahora 1)",
            "HTTP_429_RATE_LIMITED retry_after=30",
            "HTTP_418_IP_BANNED retry_after=300",
            "FRENO_ARMADO: HTTP_429_RATE_LIMITED retry_after=30",
            "RATE_LIMIT_EXHAUSTED_WEIGHT: 2400 usado + 0 en vuelo + 5 > 2400",
            "RATE_LIMIT_HEADROOM_ORDERS_10S: 250/300",
        ] {
            assert!(es_freno_temporal(e), "«{e}»");
        }
    }

    #[test]
    fn i8_un_rechazo_del_exchange_no_es_freno() {
        for e in [
            r#"Binance API Error: {"code":-2021,"msg":"Order would immediately trigger."}"#,
            r#"REJECTED code=-2022 msg=ReduceOnly Order is rejected."#,
            "AMBIGUOUS: HTTP 503 unreadable response body",
            "KILL SWITCH ACTIVE. Execution blocked.",
            "LOCAL RATE LIMIT EXCEEDED. Throttling execution.",
            "EXCHANGE_INFO_EN_ESPERA",
            "Volumen 0",
            "",
            "rate_brake en minúsculas no es el prefijo",
        ] {
            assert!(!es_freno_temporal(e), "«{e}»");
        }
    }

    #[test]
    fn i8_el_oco_compuesto_es_freno_solo_si_lo_son_sus_piernas_caidas() {
        let freno = "RATE_LIMIT_COOLDOWN: activo hasta 2 (ahora 1)";
        let exchange =
            r#"Binance API Error: {"code":-2021,"msg":"Order would immediately trigger."}"#;
        let ambas = |sl: &str, tp: &str| format!("{AMBAS_PIERNAS}{sl}{SEPARADOR_TP}{tp}");
        assert!(es_freno_temporal(&ambas(
            freno,
            "RATE_BRAKE: freno temporal activo hasta 2 (ahora 1)"
        )));
        assert!(!es_freno_temporal(&ambas(freno, exchange)));
        assert!(!es_freno_temporal(&ambas(exchange, freno)));
        assert!(!es_freno_temporal(&format!(
            "{AMBAS_PIERNAS}{freno} sin separador"
        )));

        let parcial = |e: &str| {
            format!(
                "{PIERNA_PARCIAL}{e}); pierna(s) confirmada(s) [x_TP] cancelada(s); cancelaciones pendientes []. Posición SIN protección — aplanar o alertar."
            )
        };
        assert!(es_freno_temporal(&parcial(freno)));
        assert!(!es_freno_temporal(&parcial(exchange)));
    }

    #[test]
    fn i8_solo_se_difiere_si_todos_los_intentos_fueron_freno() {
        let freno = "RATE_LIMIT_COOLDOWN: activo hasta 2 (ahora 1)";
        let exchange = r#"Binance API Error: {"code":-2021,"msg":"x"}"#;
        assert_eq!(
            tras_bracket_fallido(&[freno, freno, freno]),
            TrasBracketFallido::DiferirAlVigilante
        );
        assert_eq!(
            tras_bracket_fallido(&[freno, exchange, freno]),
            TrasBracketFallido::CierreDeEmergencia,
            "un rechazo genuino del exchange basta para X-009"
        );
        assert_eq!(
            tras_bracket_fallido::<&str>(&[]),
            TrasBracketFallido::CierreDeEmergencia,
            "sin intentos no hay prueba de freno"
        );
        let propios: Vec<String> = vec![freno.to_string(); 3];
        assert_eq!(
            tras_bracket_fallido(&propios),
            TrasBracketFallido::DiferirAlVigilante
        );
    }

    /// Los textos que este módulo reconoce son los que forman el ejecutor y
    /// el cliente. Si cambia uno, esta prueba lo dice antes de que el host
    /// vuelva a armar el kill-switch por un freno.
    #[test]
    fn i8_los_textos_reconocidos_son_los_del_ejecutor_y_el_cliente() {
        assert!(
            EXECUTOR.contains("\"Ambas órdenes OCO fallaron. SL_err: {} | TP_err: {}\""),
            "forma compuesta de las dos piernas"
        );
        assert!(
            EXECUTOR.contains("\"OCO PARCIAL: pierna fallida tras retry ({}); "),
            "forma compuesta de la pierna caída"
        );
        assert!(
            EXECUTOR.contains("\"RATE_LIMIT_EXHAUSTED_{}: "),
            "cuota del tramo crítico"
        );
        assert!(
            EXECUTOR.contains("\"RATE_LIMIT_HEADROOM_{}: "),
            "cuota de lecturas"
        );
        assert!(CLIENT.contains("\"RATE_BRAKE: freno temporal activo hasta {} (ahora {})\""));
        assert!(CLIENT.contains("\"RATE_LIMIT_COOLDOWN: activo hasta {} (ahora {})\""));
        assert!(CLIENT.contains("format!(\"FRENO_ARMADO: {}\""));
    }

    /// La cadena entera de X-009 sobre el ejecutor REAL con el freno armado:
    /// el bracket, el cierre reduce-only, la purga de piernas y la lectura de
    /// posiciones vuelven como freno temporal sin salir a la red, y el host
    /// difiere. Antes de I-8 los mismos rechazos armaban el kill-switch.
    #[tokio::test]
    async fn i8_el_ejecutor_frenado_rechaza_la_cadena_x009_como_freno_temporal() {
        for armado in [
            "HTTP_429_RATE_LIMITED retry_after=30",
            "HTTP_418_IP_BANNED retry_after=300",
        ] {
            let ex = OrderExecutor::new("k".into(), "s".into(), true);
            ex.client().freno().registrar(armado);

            let mut errores = Vec::new();
            for intento in 1..=3 {
                let r = ex
                    .execute_oco_order(
                        "BTCUSDT",
                        true,
                        0.01,
                        70_000.0,
                        60_000.0,
                        0.001,
                        0.1,
                        &format!("I8_oco_{intento}"),
                    )
                    .await;
                assert!(r.is_err(), "con {armado} el bracket no debía salir");
                errores.extend(r.err());
            }
            for e in &errores {
                assert!(es_freno_temporal(e), "{armado}: bracket «{e}»");
            }
            assert_eq!(
                tras_bracket_fallido(&errores),
                TrasBracketFallido::DiferirAlVigilante,
                "{armado}"
            );

            let cierre = ex
                .execute_reduce_only_market("BTCUSDT", true, 0.01, 0.001)
                .await;
            assert!(
                matches!(&cierre, Err(e) if es_freno_temporal(e)),
                "{armado}: cierre {cierre:?}"
            );
            let purga = ex.cancel_position_oco_orders("BTCUSDT", true).await;
            assert!(
                matches!(&purga, Err(e) if es_freno_temporal(e)),
                "{armado}: purga {purga:?}"
            );
            let posiciones = ex.fetch_position_risk().await;
            assert!(
                matches!(&posiciones, Err(e) if es_freno_temporal(e)),
                "{armado}: el vigilante tampoco lee posiciones durante el freno"
            );
        }
    }
}
