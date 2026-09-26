# MEMORIA DEL PROYECTO — Trader Gemini (estado vivo)

> Memoria compartida entre sesiones de agente (Claude, Gemini, freebuff…).
> Las REGLAS viven en `.agents/AGENTS.md`; aquí va el ESTADO: qué está
> integrado, qué decisiones siguen vigentes y qué queda abierto.
> Actualízala al cerrar cada ola (un bloque por fecha, lo nuevo arriba).
> Detalle forense completo: `INFORME_FORENSE_MAESTRO.md` (ADENDA 22 = lo último).

---

## 2026-09-26 — `main` integra el trabajo local (Olas 11-26 + FMT XXX-XXXIX) con el PR #5

### Estado de ramas
- `main` = 6fdccd64: merge LOCAL (≈90 hunks, 17 archivos) del PR #5 con 47
  commits que sólo vivían en el PC (Olas 11-26, FMT XXX-XXXIX, CES, 39
  informes forenses). PR #4 y #5 cerrados. No hay PRs abiertos.
- Los arreglos de la auditoría del PR #5 (D-744b/c, D-754b/c, D-750b,
  F-009/D-411 en `puertas_del_continuo`) están VIVOS en `main` (verificado).

### Decisiones de esa fusión (vigentes)
- **3 slots espectrales por moneda** (`scalp`/`swing`/`position`) RESTAURADOS:
  las Olas 22-26 les dieron productores reales (`find_resonant_slot`,
  despacho multi-banda). El binario de horizonte SIGUE erradicado: los tres
  viven en `Continuous` y se distinguen por τ, no por etiqueta.
- **D-751b — arranque frío total**: con 0 operaciones del símbolo y sin p
  calibrada, la orden sale como SONDA D-750 en vez de rechazarse por falta de
  evidencia (rompía el oráculo T-1: 0/144). Con ≥1 operación rige D-751.
  Nota: esto NO cubre el hallazgo #1 (p calibrada que cae tras operar).
- Slots REJ del risk-engine 15→17; techo micro de apalancamiento 5×→4×.
- `feature_exporter` = research-v2 de main; el exportador de paridad del PR
  se conserva como `feature_parity_exporter` (pendiente de checkout, Ola XL).
- T-1 (`backtest-engine/tests/t1_cobertura_genetica.rs`) IGNORADO: la física
  de viabilidad es inviable sobre el fixture sintético (spread 4 pb vs
  fricción 7 pb). Recalibrar sobre tape real = primer ítem de la Ola XL.

### Verificación cloud de `main` (windows-gnu + wine) — 4 tests ROJOS
Compila limpio (`--workspace --all-targets`). Fallan en `quantum-arena`:
- **Introducidos por la fusión** (pasan en los dos padres, 3ee49b05 y
  ac136633): los contratos CES de `temporal_spectrum` esperan la semántica
  anterior a D-742, y la fusión adoptó D-742:
  - `spectral_contract_learning_refreshes_fusion_without_new_tick`:
    `fused_score` ≠ `spectral_coherence(true)` (Δ ≈ 0,039). `refresh_fusion`
    pondera por masa×resolución (D-742) y `spectral_coherence` no: DOS
    definiciones de la fusión. Remedio: una sola función de pesos para ambas.
  - `spectral_contract_long_scale_retains_small_elapsed_mass`: espera la
    semilla 1e-7 (0,0000001); D-742 da la vol OBSERVADA (0,0100). El test
    codifica el defecto que D-742 corrigió → actualizar el test.
  - `spectral_contract_timestamp_zero_is_a_valid_origin`: `assert_eq!` exacto
    entre origen 0 y origen desplazado; difieren en 6e-14 → comparar con
    tolerancia o hacer el cálculo invariante al desplazamiento.
- **Preexistente** (ya fallaba en ac136633, antes de la fusión; intermitente
  porque la mutación es aleatoria): `genome_store::test_r11_evolution_
  pipeline_never_blocked_by_gate`. `mutate_cmaes`/`from_vector` producen
  genomas cuya curva de SL no alcanza el piso de fricción (15,4 pb a fee
  0,0010) que exige `GenomeEnvelope::validate` → el gate de promoción
  rechaza mutantes. Los bounds de mutación y la política de fricción no
  coinciden (regresión de R1.1).

---

## 2026-09-25 — Integración ola «espectro predictivo» (PR #4) + F-009/WS de main

### Ramas
- `main` (1fa9a914) llevaba 3 commits que la rama de auditoría no tenía:
  F-009 / D-411 (desasfixia ML), M2-C02 / M1-M02 / WS (Hawkes por símbolo,
  OI en USD, timeout WS adaptativo) y streams WS en minúsculas.
- `claude/decima-ola-auditoria-2` (PR #4) llevaba 13 commits
  (D-742 … D-758). Se unificaron las dos en `claude/elegant-euler-mmtht4`
  (PR #5, auditado y fusionado a `main`; el #4 se cerró como sustituido).
  Una sesión que siga sobre `claude/decima-ola-auditoria-2` debe traer
  `main` antes de continuar.
- Ramas muertas (no integrar): `claude/decima-ola-auditoria-forense`
  (155 detrás), `subagent-*` (121 detrás, de junio).

### Qué quedó integrado (vigente)
- **Espectro predictivo** (`quantum_arena::spectral_tape`,
  `SpectralForecastBank`): puntuación prequencial; volatilidad (R² OOS
  +0,05…+0,18), volumen (+0,25…+0,50) e intensidad (+0,36…+0,64) SÍ son
  predecibles y ganan a la persistencia; flujo de dinero y concentración de
  entes NO (sólo observación).
- **D-742**: el espectro temporal no opina con escalas τ que no ha observado.
- **D-743 `puertas_del_continuo`**: UNA sola función con las cuatro puertas
  (viabilidad, invariante bayesiano D-472, ponderación ML F-009, vetos
  CVD/muro L2) aplicada a `fast_intent` Y `slow_intent`. **Cualquier cambio
  a esas puertas va AHÍ**, nunca inline en una de las dos bandas.
- **F-009 / D-411 (desde main, portado a la puerta)**: dirección ML medida
  contra `ml_model_base` (≈0,30 con etiquetado honesto), veto sólo si
  contradice < −0,80 sin acción de precio extrema, penalización blanda
  `(1 + d·0,5).clamp(0,20, 1)`. Scaler de DarkAlpha: z-scores en [−3, 3].
- **D-744**: cortacircuitos de drawdown con una sola semántica medida.
- **D-745 / U-ERR-5**: una posición, un horizonte. `PositionHorizon` sólo
  tiene `Continuous`; el horizonte REAL es `Position::entry_tau_ms` (τ con la
  que se dimensionó, `ValidatedOrder::tau_ms`). **No reintroducir las
  ETIQUETAS Swing/Scalping** (F-009 de main lo hacía; se revirtió). Desde el
  2026-09-26 hay 3 slots por moneda, distinguidos por τ (ver bloque de arriba).
- **D-746**: la volatilidad no es convicción (freno ATR vs distancia al stop).
- **D-747**: flujo de entrenamiento des-espejado (`qty = |bq − aq|`,
  `is_buyer_maker = aq > bq`) en entrenador, exportador, sonda y forense.
- **D-749**: veto de muro compara en el espacio del gen (razón entre muros).
- **D-751/751b/752**: spread de los tapes corregido (`tape_spread_fix`), tick
  inferido del propio tape, el medidor cobra el spread UNA vez.
- **D-752/D-756**: convicción por rama = cota de Wilson de su propio
  historial (`TasaAcierto`, `conviccion_de_rama`), sin literales 0,72/0,68;
  `exigencia_tras_racha` monótona y saturada en 1.
- **D-753 + paridad**: el entrenador (`train_forest`) reproduce el tape por
  `GodEngineCore::process_event` y verifica paridad 48D con tolerancia CERO
  contra `build_54d_tensor`; aborta si difiere. Slippage por latencia
  unificado a la función de difusión pura.
- **D-754**: `compute_tp_sl` usa σ pronosticada (`CoinArena::sigma_forecast_at`)
  sólo si el pronóstico tiene habilidad positiva; si no, idéntico a antes.
- **Riesgo**: Kelly micro sin piso ficticio (se RECHAZA lo que no cabe en el
  tope de ruina), `min_notional` publicado del símbolo, guard de correlación
  = Pearson real, profit factor con cota de Jeffreys.
- **M5-H01**: `fitness_compute(initial, final, dd, trades)` — <30 cierres ⇒
  INVIABLE; dd clamp [0,1]; Darwin arranca `best_all_time` en −∞.
- **CERT-M2-C02**: Hawkes por símbolo (`hawkes_by_coin`), λ/μ real al registry.
- **CERT-M1-M02**: OI normalizado en notional USD (escala log $100M…$10B).
- **WS**: streams en minúsculas; watchdog `BINANCE_WS_TIMEOUT_SECS`
  (defecto 15 s mainnet / 30 s testnet).

### Auditoría del PR #5 (2026-09-25) — corregido antes de fusionar
- **D-744b**: sin `riesgo_por_operacion` medido (arranque del proceso) los
  DOS cortacircuitos de drawdown quedaban desarmados. Ahora
  `drawdown::drawdown_maximo` cae al gen como fracción de caída.
- **D-744c**: la EWMA del riesgo tomado se registra tras el último rechazo.
- **D-754b**: el σ pronosticado vuelve a ceros si la habilidad cae, y sólo
  se publican anclas puntuadas (`MUESTRAS_MADURAS` = 30).
- **D-754c**: el pronóstico σ entra al stop ×√(8/π) (`RANGO_PARKINSON`):
  el respaldo ATR es un rango y el gen del SL está en esa escala.
- **D-750b**: el guard de correlación compara r CON signo (misma dirección
  + anticorrelación = cobertura, no la misma apuesta).

### Hallazgos de la auditoría NO corregidos (diseño — siguiente ola)
1. **EV gate con p calibrada (D-751) puede ser un estado absorbente**: el
   calibrador sólo aprende de lo ejecutado; si baja p, el EV veta todo y p
   no se recupera (lo que D-690 evitó en el gate de confianza). Vigilar en
   el forense si una moneda deja de operar tras ~20 cierres. Remedio
   candidato: calibrar con resultados contrafactuales de intenciones
   vetadas (auditor de trayectorias) o una cota superior de p como piso.
2. **Convicción por rama (D-752) se re-infla aguas abajo** (boost ML ×1,5,
   fusión `max`, persistencia ×1,3): una rama perdedora demostrada puede
   superar `min_confidence_btc`. Y las ramas 11–14 (tensor, impulso,
   tendencia lenta, consenso) no pasan por `conviccion_de_rama`.
3. **VPIN**: el lado lo decide la regla del tick sobre el mid (no el
   `is_buyer_maker` real); y los llamadores legacy de `process_tick`
   (DarwinDaemon opt-in, bin `evolver`) nunca alimentan volumen ⇒ VPIN
   congelado ahí.
4. **Deslizamiento por latencia (D-753)** unificado sólo en gate y física;
   brackets del host (`genome_protection_prices`, fallback de entrada) y el
   pre-screen del daemon siguen con la ley lineal (~3× distinta).
5. **Freno de apalancamiento (D-746)** mide contra el τ del gen y la curva
   de SL genómica, no contra la τ medida ni el stop real (D-745/745b).
6. **Anti-whiplash (D-757)**: sus ventanas de tiempo son código muerto
   (el enfriamiento de `viable_para_entrar` ya las cubre).
7. **Bosque (freno)**: la exactitud direccional se puntúa con PnL NETO
   (`is_win`); un acierto menor que las comisiones cuenta como fallo.
8. Guard de correlación: dirección OPUESTA + anticorrelación también es la
   misma apuesta y el llamador la descarta (preexistente).
9. Proceso: los commits 6c06ee3d y 8cb5504b mezclan varios bloques
   (incumplen «commits atómicos»); no se reescribe historia.

### Abierto / pendiente
- **Nada autoriza a operar**: el motor sigue perdiendo en las tres
  configuraciones medidas (ADENDA 22.7). Falta el forense sobre los tapes
  regenerados con D-751/D-752 y el modelo reentrenado con paridad D-753.
- Modelos se llaman `{SÍMBOLO}_MOTOR` (+ `_VOL` para el predictor P-1).
  BTCUSDT NO tiene `_MOTOR` ⇒ la puerta de roster veta todas sus entradas;
  el forense acepta `FORENSIC_SYMBOL=ATOMUSDT` (u otro con modelo).
- Erradicación U-ERR incompleta: examen walk-forward del daemon, etiquetas y
  puertas del entrenador (ver commit 8afcc677).
- Predictores P-1/P-2: su gate compara R² contra la media, no contra la
  persistencia ⇒ un R² de 0,106 no demuestra habilidad.

### Cómo compilar / probar desde Linux (sesiones cloud)
- `os-guardian` es sólo-Windows (`windows::Win32`) y casi todo depende de él:
  `cargo check` nativo en Linux FALLA en `main` también. No es regresión.
- Chequeo completo: `rustup target add x86_64-pc-windows-gnu`, instalar
  `gcc-mingw-w64-x86-64`, luego
  `cargo check --target x86_64-pc-windows-gnu --workspace --all-targets`.
- Tests: instalar `wine64` y exportar
  `CARGO_TARGET_X86_64_PC_WINDOWS_GNU_RUNNER=/usr/lib/wine/wine64 WINEDEBUG=-all`,
  luego `cargo test --target x86_64-pc-windows-gnu -p <crate> --lib`.
- DEMO desde la nube: la política de red del entorno bloquea (403) todos los
  hosts de Binance (`testnet.binancefuture.com`, `stream.binancefuture.com`,
  `fapi.binance.com`, `data.binance.vision`) y no hay claves en el entorno;
  los tapes no están en el repo (viven en el PC del operador). Para operar
  demo en una sesión cloud hay que permitir esos hosts en la configuración
  de red del entorno y definir `USE_TESTNET=true` +
  `BINANCE_TESTNET_API_KEY` / `BINANCE_TESTNET_SECRET_KEY` como variables
  del entorno. Si no, la demo se corre en el PC local.

### Lección de este merge
- Git sólo marcó 4 hunks en `god-engine-core/src/lib.rs`, pero hubo dos
  choques SEMÁNTICOS que auto-mergearon "limpios": `pos_h` Swing/Scalping
  (variantes ya borradas por U-ERR-5) y la ponderación F-009 que main
  editó en el bloque inline que la otra rama había movido a
  `puertas_del_continuo`. Tras cada merge entre sesiones: diff del resultado
  contra CADA padre y compilar con `--all-targets` antes de commitear.
