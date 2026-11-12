//! CAPITAL-UNICO V4C: pruebas de aceptacion de Codex QRA para V4C, TAL COMO
//! ESTAN ESCRITAS en `.agent_tx/diagnostics/auditoria_perdidas/qra_capital_v4c_socket.rs`
//! (11:35) y `qra_capital_v4c_frescura.rs` (13:04). Solo se anade esta cabecera,
//! el `#[path]` al modulo (su arnes lo compilaba suelto), un modulo por archivo
//! y `super::` en el `use` del puente (y `use super::puente_cuenta;` para la
//! ruta `puente_cuenta::PosicionRest` de la segunda).

#[path = "../src/puente_cuenta.rs"]
#[allow(dead_code)]
mod puente_cuenta;
mod qra_capital_v4c_socket {
    use super::puente_cuenta::{DatoPosicion, EventoStream, LecturaCuenta, Observacion, PosicionRest, PuenteCuenta, Rechazo};

    fn row(symbol: &str, upnl: f64) -> ((String, String), DatoPosicion) {
        ((symbol.into(), "BOTH".into()), DatoPosicion::Abierta { cantidad: 1.0, upnl })
    }

    fn rest(wallet: f64, t: u64, btc: f64, eth: f64) -> LecturaCuenta {
        let posiciones = [row("BTCUSDT", btc), row("ETHUSDT", eth)]
            .into_iter()
            .map(|(key, dato)| (key, PosicionRest { dato, tiempo_ms: Some(t) }))
            .collect();
        LecturaCuenta {
            saldo: wallet,
            tiempo_saldo_ms: Some(t),
            posiciones: Some(posiciones),
            rechazo_arranque: None,
            upnl_del_activo: Some(btc + eth),
        }
    }

    fn ws(p: &PuenteCuenta, socket: u64, t: u64, e: u64, wallet: f64, btc: f64, eth: f64)
        -> Result<Option<Observacion>, Rechazo>
    {
        let update = EventoStream {
            tiempo_ms: Some(t),
            saldo: Some(wallet),
            posiciones: vec![row("BTCUSDT", btc), row("ETHUSDT", eth)],
        };
        p.observar_stream_del_socket(p.era(), socket, Some(e), &update, |_| {})
    }

    #[test]
    fn ambiguous_open_pnl_takes_the_minimum_per_position_and_recovers_on_new_order() -> Result<(), Rechazo> {
        let p = PuenteCuenta::new();
        p.observar_rest(p.ticket(), &rest(100.0, 100, -10.0, -40.0))?;
        ws(&p, p.socket(), 300, 301, 100.0, -10.0, -40.0)?;
        p.reconexion(p.era())?;
        let reconciled = p.observar_rest(p.ticket(), &rest(100.0, 299, -50.0, -5.0))?;
        assert_eq!(reconciled, Observacion { saldo: 100.0, equity: 10.0 });
        assert!(!p.veta_entradas());
        let later = ws(&p, p.socket(), 400, 401, 100.0, -5.0, -5.0)?;
        assert_eq!(later, Some(Observacion { saldo: 100.0, equity: 90.0 }));
        Ok(())
    }

    #[test]
    fn another_applied_event_invalidates_a_rest_ticket_even_at_equal_t() -> Result<(), Rechazo> {
        let p = PuenteCuenta::new();
        p.observar_rest(p.ticket(), &rest(100.0, 100, 0.0, 0.0))?;
        let socket = p.socket();
        ws(&p, socket, 200, 201, 95.0, -1.0, 0.0)?;
        let ticket_before_event = p.ticket();
        ws(&p, socket, 200, 202, 94.0, -2.0, 0.0)?;
        let _ = p.observar_rest(ticket_before_event, &rest(90.0, 200, -2.0, 0.0));
        assert_eq!(p.observacion().map(|obs| obs.saldo), Some(94.0));
        let current = p.observar_rest(p.ticket(), &rest(94.0, 200, -2.0, 0.0))?;
        assert_eq!(current, Observacion { saldo: 94.0, equity: 92.0 });
        assert!(!p.veta_entradas());
        Ok(())
    }

    #[test]
    fn a_packet_from_the_previous_socket_does_not_change_a_reconciled_account() -> Result<(), Rechazo> {
        let p = PuenteCuenta::new();
        p.observar_rest(p.ticket(), &rest(100.0, 100, 0.0, 0.0))?;
        let old = p.socket();
        p.reconexion(p.era())?;
        let reconciled = p.observar_rest(p.ticket(), &rest(90.0, 150, -1.0, 0.0))?;
        assert!(ws(&p, old, 999, 999, 100.0, 0.0, 0.0).is_err());
        assert_eq!(p.observacion(), Some(reconciled));
        assert!(!p.veta_entradas());
        Ok(())
    }
}

mod qra_capital_v4c_frescura {
    use super::puente_cuenta;

    use super::puente_cuenta::{DatoPosicion, EventoStream, LecturaCuenta, PuenteCuenta, Rechazo};
    use std::collections::HashMap;

    fn key() -> (String, String) {
        ("BTCUSDT".to_owned(), "BOTH".to_owned())
    }

    fn abierta(upnl: f64) -> DatoPosicion {
        DatoPosicion::Abierta { cantidad: 1.0, upnl }
    }

    fn rest(saldo: f64, t: u64, upnl: f64) -> LecturaCuenta {
        LecturaCuenta {
            saldo,
            tiempo_saldo_ms: Some(t),
            posiciones: Some(HashMap::from([(key(), puente_cuenta::PosicionRest {
                dato: abierta(upnl),
                tiempo_ms: Some(t),
            })])),
            rechazo_arranque: None,
            upnl_del_activo: None,
        }
    }

    fn ws(p: &PuenteCuenta, socket: u64, t: u64, e: Option<u64>, saldo: f64, upnl: f64) {
        let ev = EventoStream { tiempo_ms: Some(t), saldo: Some(saldo), posiciones: vec![(key(), abierta(upnl))] };
        let _ = p.observar_stream_del_socket(p.era(), socket, e, &ev, |_| {});
    }

    fn nueva() -> Result<PuenteCuenta, Rechazo> {
        let p = PuenteCuenta::new();
        p.observar_rest(p.ticket(), &rest(100.0, 100, 0.0))?;
        Ok(p)
    }

    #[test]
    fn f1_foto_que_confirma_no_hace_perder_orden_del_proximo_fill() -> Result<(), Rechazo> {
        let p = nueva()?;
        let s = p.socket();
        ws(&p, s, 200, Some(201), 99.0, -1.0);
        p.observar_rest(p.ticket(), &rest(99.0, 200, -1.0))?;
        ws(&p, s, 200, Some(202), 98.0, -2.0);
        assert_eq!(p.observacion().map(|o| (o.saldo, o.equity)), Some((98.0, 96.0)));
        assert!(!p.veta_entradas());
        Ok(())
    }

    #[test]
    fn f2_ticket_anterior_al_ws_no_levanta_veto_aunque_la_foto_coincida() -> Result<(), Rechazo> {
        let p = nueva()?;
        let s = p.abrir_socket(p.era())?;
        let a = p.ticket();
        ws(&p, s, 200, Some(201), 99.0, -1.0);
        assert_eq!(p.observacion().map(|o| o.saldo), Some(99.0));
        assert!(p.veta_entradas());
        let _ = p.observar_rest(a, &rest(99.0, 200, -1.0));
        assert!(p.veta_entradas(), "la foto se pidió antes del último WS aplicado");
        p.observar_rest(p.ticket(), &rest(99.0, 200, -1.0))?;
        assert!(!p.veta_entradas());
        Ok(())
    }

    #[test]
    fn f2_evento_sin_e_no_borra_la_e_mayor_ya_observada() -> Result<(), Rechazo> {
        let p = nueva()?;
        let s = p.socket();
        ws(&p, s, 200, Some(202), 99.0, -1.0);
        ws(&p, s, 200, None, 98.0, -2.0);
        let saldo_antes = p.observacion().map(|o| o.saldo);
        ws(&p, s, 200, Some(201), 97.0, -3.0);
        assert_eq!(p.observacion().map(|o| o.saldo), saldo_antes);
        Ok(())
    }

    #[test]
    fn f3_saldo_en_conflicto_no_oculta_la_perdida_abierta_del_mismo_lote() -> Result<(), Rechazo> {
        let p = nueva()?;
        ws(&p, p.socket(), 200, Some(202), 100.0, -10.0);
        let _ = p.observar_rest(p.ticket(), &rest(101.0, 200, -60.0));
        assert_eq!(p.observacion().map(|o| o.saldo), Some(100.0));
        assert!(p.veta_entradas());
        assert_eq!(p.equity_del_freno(), Some(40.0), "el freno ve la pérdida, el saldo conserva su autoridad");
        Ok(())
    }
}
