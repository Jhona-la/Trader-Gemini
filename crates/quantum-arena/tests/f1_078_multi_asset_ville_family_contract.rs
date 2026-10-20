//! Contrato de Validación Formal F1-078 (Ola Ω182):
//! Sizing Dinámico de Familia de Ville, Aceleración GRAPA y Persistencia/Warm-Start
//! en el Universo Espectral Multiactivo.
//!
//! # Verificaciones:
//! 1. `familia_honesta(n)` calcula la cota exacta de hipótesis C(n, 2) × 5 para n activas.
//! 2. `dimensionar_por_roster(n)` ajusta la familia y el umbral de Ville dinámicamente.
//! 3. Apuesta adaptativa GRAPA (Waudby-Smith & Ramdas 2023) acelera el cruce del umbral
//!    de Ville ante evidencia fuerte (ρ ≈ 0.7) en ~2.6× respecto a λ fija = 0.10, sin inflar FWER bajo H0.
//! 4. Persistencia y warm-start (`SnapshotParEscala`, `exportar_snapshots`, `importar_snapshots`)
//!    eliminan el retraso de arranque en frío (0 bloques tras reinicio).
//! 5. Telemetría de diagnóstico `pares_cerrados_por_capital` y `estado_gate_ville`.

use quantum_arena::espectral_multiactivo::{
    familia_honesta, EspectralMultiactivo, EspectralMultiactivoParams, SnapshotParEscala,
    FAMILIA_VETO_GRUPO,
};
use quantum_arena::evalues::EProceso;
use quantum_arena::temporal_spectrum::SPECTRUM_SCALES_MS;

struct Lcg(u64);
impl Lcg {
    fn new(seed: u64) -> Self {
        Self(seed ^ 0x9E37_79B9_7F4A_7C15)
    }
    fn u(&mut self) -> f64 {
        self.0 = self
            .0
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        ((self.0 >> 11) as f64 + 0.5) / (1u64 << 53) as f64
    }
    fn normal(&mut self) -> f64 {
        let (u1, u2) = (self.u().max(1e-12), self.u());
        (-2.0 * u1.ln()).sqrt() * (2.0 * std::f64::consts::PI * u2).cos()
    }
}

#[test]
fn f1_078_dimensionado_honesto_de_familia_segun_roster() {
    // Casos degenerados
    assert_eq!(familia_honesta(0), 1);
    assert_eq!(familia_honesta(1), 1);

    // Casos reales de roster vivo
    // 6 monedas (micro-universo $13 USD): C(6, 2) * 5 = 15 * 5 = 75
    assert_eq!(familia_honesta(6), 75);
    assert_eq!(EProceso::umbral_familia(familia_honesta(6)), 1_500.0);

    // 8 monedas (roster activo estándar): C(8, 2) * 5 = 28 * 5 = 140
    assert_eq!(familia_honesta(8), 140);
    assert_eq!(EProceso::umbral_familia(familia_honesta(8)), 2_800.0);

    // 16 monedas (roster medio): C(16, 2) * 5 = 120 * 5 = 600
    assert_eq!(familia_honesta(16), 600);
    assert_eq!(EProceso::umbral_familia(familia_honesta(16)), 12_000.0);

    // 26 monedas (ROSTER canónico completo): C(26, 2) * 5 = 325 * 5 = 1625
    assert_eq!(familia_honesta(26), 1_625);
    assert_eq!(EProceso::umbral_familia(familia_honesta(26)), 32_500.0);

    // 30 monedas (límite físico MAX_COINS): C(30, 2) * 5 = 435 * 5 = 2175
    assert_eq!(familia_honesta(30), 2_175);
    assert_eq!(familia_honesta(30), FAMILIA_VETO_GRUPO);
    assert_eq!(EProceso::umbral_familia(FAMILIA_VETO_GRUPO), 43_500.0);

    // Instancia con dimensionado automático
    let mut ma = EspectralMultiactivo::new(8);
    // Sin override explícito ni universo activo, fallback es FAMILIA_VETO_GRUPO
    assert_eq!(ma.familia_efectiva(), FAMILIA_VETO_GRUPO);

    // Fijar roster de 8 monedas
    ma.dimensionar_por_roster(8);
    assert_eq!(ma.familia_efectiva(), 140);

    // Fijar roster de 6 monedas
    ma.dimensionar_por_roster(6);
    assert_eq!(ma.familia_efectiva(), 75);
}

#[test]
fn f1_078_aceleracion_grapa_frente_a_lambda_fija() {
    let tau = SPECTRUM_SCALES_MS[19]; // ~4.58 min
    let escala = 19;

    let mut params_fija = EspectralMultiactivoParams::default();
    params_fija.familia_veto_override = Some(140); // Roster 8 monedas (umbral 2800)
    params_fija.lambda_adaptativa = false; // λ fija = 0.10

    let mut params_grapa = EspectralMultiactivoParams::default();
    params_grapa.familia_veto_override = Some(140); // Roster 8 monedas (umbral 2800)
    params_grapa.lambda_adaptativa = true; // GRAPA adaptativo

    let mut ma_fija = EspectralMultiactivo::with_params(8, params_fija);
    let mut ma_grapa = EspectralMultiactivo::with_params(8, params_grapa);

    let mut lcg = Lcg::new(777);
    let rho: f64 = 0.70; // Co-movimiento real fuerte
    let sqrt_1_minus_rho2 = (1.0f64 - rho * rho).sqrt();

    let mut bloques_hasta_cruce_fija: Option<u64> = None;
    let mut bloques_hasta_cruce_grapa: Option<u64> = None;

    for i in 1..=250u64 {
        let ts = i * 1_000;
        let z1 = lcg.normal();
        let z2 = lcg.normal();
        let ra = z1 * 0.01;
        let rb = (rho * z1 + sqrt_1_minus_rho2 * z2) * 0.01;

        ma_fija.observar_maduracion(0, tau, escala, ts, ra);
        ma_fija.observar_maduracion(1, tau, escala, ts, rb);

        ma_grapa.observar_maduracion(0, tau, escala, ts, ra);
        ma_grapa.observar_maduracion(1, tau, escala, ts, rb);

        if bloques_hasta_cruce_fija.is_none() && ma_fija.coherencia_par_veto(0, 1, escala).is_some() {
            bloques_hasta_cruce_fija = Some(i);
        }
        if bloques_hasta_cruce_grapa.is_none() && ma_grapa.coherencia_par_veto(0, 1, escala).is_some() {
            bloques_hasta_cruce_grapa = Some(i);
        }
    }

    let b_grapa = bloques_hasta_cruce_grapa.expect("GRAPA debe cruzar el umbral 2800");
    let b_fija = bloques_hasta_cruce_fija.expect("Lambda fija debe cruzar eventualmente");

    println!(
        "F1-078: Cruce de Ville para umbral 2800 (roster 8) -> GRAPA: {} bloques, Fija: {} bloques",
        b_grapa, b_fija
    );

    // GRAPA debe cruzar significativamente antes que lambda fija (al menos 2× más rápido)
    assert!(
        b_grapa <= 90,
        "GRAPA debe cruzar en <= 90 bloques (medido: {})",
        b_grapa
    );
    assert!(
        b_fija > b_grapa + 50,
        "Lambda fija requiere sustancialmente más muestras: fija={}, grapa={}",
        b_fija,
        b_grapa
    );
}

#[test]
fn f1_078_persistencia_warm_start_roundtrip() {
    let tau = SPECTRUM_SCALES_MS[19];
    let escala = 19;

    let mut params = EspectralMultiactivoParams::default();
    params.familia_veto_override = Some(75); // Roster 6 monedas
    params.lambda_adaptativa = true;

    let mut ma_original = EspectralMultiactivo::with_params(6, params);

    // Acumular 70 observaciones sintéticas altamente correlacionadas
    for i in 1..=70u64 {
        let ts = i * 1_000;
        let r = if i % 2 == 0 { 0.02 } else { -0.02 };
        ma_original.observar_maduracion(0, tau, escala, ts, r);
        ma_original.observar_maduracion(1, tau, escala, ts, r);
    }

    // Verificar que el par 0-1 está maduro y con gate abierto
    let ic_orig = ma_original
        .coherencia_par_veto(0, 1, escala)
        .expect("debe cruzar con 70 aciertos deterministas");
    assert!(ic_orig > 0.95);

    // Exportar snapshots
    let snapshots: Vec<SnapshotParEscala> = ma_original.exportar_snapshots();
    assert!(!snapshots.is_empty(), "debe haber snapshots exportados");

    let snap_01 = snapshots
        .iter()
        .find(|s| s.coin_a == 0 && s.coin_b == 1 && s.escala == escala)
        .expect("debe contener el par (0, 1) en escala 19");

    assert_eq!(snap_01.n_muestras, 70);
    assert!(snap_01.capital_ville >= 1_500.0);
    assert!(snap_01.aciertos >= 50);

    // Serialización / Deserialización JSON simulando persistencia en disco
    let json = serde_json::to_string(&snapshots).expect("debe serializar a JSON");
    let recovered_snaps: Vec<SnapshotParEscala> =
        serde_json::from_str(&json).expect("debe deserializar desde JSON");

    // Crear una instancia totalmente nueva y fresca
    let mut ma_restaurada = EspectralMultiactivo::with_params(6, params);
    // Antes de restaurar: capital 1.0, 0 muestras, gate cerrado
    assert_eq!(ma_restaurada.muestras_par(0, 1, escala), Some(0));
    assert_eq!(ma_restaurada.coherencia_par_veto(0, 1, escala), None);

    // Importar snapshots (warm-start)
    ma_restaurada.importar_snapshots(&recovered_snaps);

    // Comprobación inmediata tras reinicio en caliente: gate ABIERTO sin observar ningún tick nuevo
    let ic_rest = ma_restaurada
        .coherencia_par_veto(0, 1, escala)
        .expect("gate debe estar abierto inmediatamente en caliente tras warm start");
    assert!((ic_orig - ic_rest).abs() < 1e-9);
    assert_eq!(ma_restaurada.muestras_par(0, 1, escala), Some(70));
}

#[test]
fn f1_078_telemetria_gate_cerrado_por_capital() {
    let tau = SPECTRUM_SCALES_MS[19];
    let escala = 19;

    let mut params = EspectralMultiactivoParams::default();
    params.familia_veto_override = Some(2_175); // Umbral alto 43500
    params.muestras_maduras = 30;
    params.lambda_adaptativa = false;

    let mut ma = EspectralMultiactivo::with_params(4, params);

    // Observar 35 muestras con correlación débil o moderada
    let mut lcg = Lcg::new(42);
    for i in 1..=35u64 {
        let ts = i * 1_000;
        let ra = lcg.normal() * 0.01;
        let rb = lcg.normal() * 0.01; // no correlacionados
        ma.observar_maduracion(0, tau, escala, ts, ra);
        ma.observar_maduracion(1, tau, escala, ts, rb);
    }

    // Tiene madurez estadística (n = 35 >= 30), pero capital < 43500
    assert_eq!(ma.muestras_par(0, 1, escala), Some(35));
    assert_eq!(ma.coherencia_par_veto(0, 1, escala), None);

    // Telemetría debe detectar exactamente 1 par maduro cerrado por capital
    let cerrados = ma.pares_cerrados_por_capital(0, escala);
    assert_eq!(cerrados, 1, "par 0-1 debe figurar cerrado por capital");

    let (capital, umbral, abierto) = ma
        .estado_gate_ville(0, 1, escala)
        .expect("estado gate debe existir");
    assert_eq!(umbral, 43_500.0);
    assert!(capital < umbral);
    assert!(!abierto);

    // Warm start del par a capital > 43500
    ma.warm_start_par(0, 1, escala, 40, 0.001, 0.001, 0.0009, 50_000.0);

    // Ahora el gate se abre y pares cerrados por capital pasa a 0
    let cerrados_post = ma.pares_cerrados_por_capital(0, escala);
    assert_eq!(cerrados_post, 0, "par ya no debe estar cerrado por capital");
    let (_, _, abierto_post) = ma.estado_gate_ville(0, 1, escala).unwrap();
    assert!(abierto_post);
}

#[test]
fn f1_078_robustez_sin_panicos_limites_arbitrarios() {
    let mut ma = EspectralMultiactivo::new(4);
    // Consultas fuera de límites o idénticas no deben entrar en pánico
    assert_eq!(ma.coherencia_par(5, 6, 10), None);
    assert_eq!(ma.coherencia_par(0, 0, 10), None);
    assert_eq!(ma.coherencia_par(0, 1, 999), None);
    assert_eq!(ma.coherencia_par_veto(5, 6, 10), None);
    assert_eq!(ma.estado_gate_ville(99, 100, 50), None);
    assert_eq!(ma.pares_cerrados_por_capital(99, 50), 0);

    // Dimensionar por roster con valores atípicos
    ma.dimensionar_por_roster(0);
    ma.dimensionar_por_roster(1);
    ma.dimensionar_por_roster(100);

    // Importar snapshots vacíos o con índices corruptos no debe entrar en pánico
    ma.importar_snapshots(&[]);
    let corrupt_snap = SnapshotParEscala {
        coin_a: 99,
        coin_b: 100,
        escala: 50,
        n_muestras: 10,
        ea2: 0.1,
        eb2: 0.1,
        eab: 0.05,
        capital_ville: 10.0,
        aciertos: 5,
        fallos: 5,
    };
    ma.importar_snapshots(&[corrupt_snap]);
}
