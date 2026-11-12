//! CAPITAL-UNICO V4B: las dos RED de Codex QRA, TAL COMO ESTAN ESCRITAS en
//! `.agent_tx/diagnostics/auditoria_perdidas/qra_n1_propuesta_reconexion.rs`
//! (09:58) y `qra_capital_rest_en_vuelo.rs` (10:18). Solo se anade esta
//! cabecera, el `#[path]` al modulo (su arnes lo compilaba suelto, con
//! `mod puente_cuenta;` al lado) y un modulo por archivo, con `super::` en
//! el `use` del puente.

#[path = "../src/puente_cuenta.rs"]
#[allow(dead_code)]
mod puente_cuenta;

mod qra_n1_propuesta_reconexion {
    use super::puente_cuenta::{DatoPosicion, EventoStream, PuenteCuenta, Rechazo};

    fn evento(t: u64, saldo: f64, posicion: DatoPosicion) -> EventoStream {
        EventoStream {
            tiempo_ms: Some(t),
            saldo: Some(saldo),
            posiciones: vec![(("VETUSDT".into(), "BOTH".into()), posicion)],
        }
    }

    #[test]
    fn un_evento_repetido_con_t_empatada_no_rebobina_al_reconectar() -> Result<(), Rechazo> {
        let puente = PuenteCuenta::new();
        let era = puente.era();
        let viejo = evento(100, 100.0, DatoPosicion::Abierta { cantidad: -1.0, upnl: -20.0 });
        let nuevo = evento(100, 0.0, DatoPosicion::Cerrada);
        puente.observar_stream(era, &viejo)?;
        assert_eq!(puente.observar_stream(era, &nuevo)?.map(|v| v.saldo), Some(0.0));
        puente.reconexion(era)?;
        assert!(puente.veta_entradas());
        let rebobinado = puente.observar_stream(era, &viejo);
        assert!(rebobinado.is_err(), "evento de conexión anterior aceptado: {rebobinado:?}");
        Ok(())
    }
}

mod qra_capital_rest_en_vuelo {
    use super::puente_cuenta::{LecturaCuenta, PuenteCuenta, Rechazo};
    use std::collections::HashMap;

    fn cuenta_quieta() -> LecturaCuenta {
        LecturaCuenta {
            saldo: 100.0,
            tiempo_saldo_ms: Some(100),
            posiciones: Some(HashMap::new()),
            rechazo_arranque: None,
            upnl_del_activo: Some(0.0),
        }
    }

    #[test]
    fn foto_iniciada_antes_de_reconectar_no_rearma_entradas() -> Result<(), Rechazo> {
        let puente = PuenteCuenta::new();
        let cuenta = cuenta_quieta();
        puente.observar_rest(puente.ticket(), &cuenta)?;
        assert!(!puente.veta_entradas());

        let ticket_previo = puente.ticket();
        puente.reconexion(puente.era())?;
        assert!(puente.veta_entradas());
        let respuesta_tardia = puente.observar_rest(ticket_previo, &cuenta);
        assert!(puente.veta_entradas(), "una lectura iniciada antes del corte rearmó: {respuesta_tardia:?}");
        Ok(())
    }

    #[test]
    fn foto_nueva_confirma_cuenta_quieta_con_update_time_inalterado() -> Result<(), Rechazo> {
        let puente = PuenteCuenta::new();
        let cuenta = cuenta_quieta();
        puente.observar_rest(puente.ticket(), &cuenta)?;
        puente.reconexion(puente.era())?;
        assert!(puente.veta_entradas());
        puente.observar_rest(puente.ticket(), &cuenta)?;
        assert!(!puente.veta_entradas());
        Ok(())
    }
}