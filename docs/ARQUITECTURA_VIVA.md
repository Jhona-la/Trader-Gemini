# ARQUITECTURA VIVA — Trader Gemini (mapa para el consejo)

> Mantenedor: Qoder. Últimasync: 2026-10-06 (post Ola 62 / Ω11).
> `.agents/MEMORIA.md`; la bitácora de coordinación en `COORDINACION_CODEX_2026-09-28.md`;
> la doctrina formal en `docs/adr/ADR-0014-doctrina-continuo-espectral.md`.
> REGLA DE ORO: si tu cambio contradice un invariante de §2, o repara el mapa
> de §1, ACTUALIZA este archivo en el mismo commit.

## 1. El pipeline de decisión (punta a punta)

```
WS Binance (data-pipeline/ws_client.rs, parser.rs)
  └─> Arena global (quantum-arena/state.rs) — estado atómico por coin
        │   claves vivas: agg_buy/sell_vol (decay FÍSICO τ=60s, #660,
        │   con timestamp del exchange «T», #660),
        │   spectral_coherence/entropy/intermittency, cuantiles p80
        └─> GodEngineCore::process_tick_dual (god-engine-core/src/lib.rs, ~8k líneas)
              ├─ Espectro temporal (quantum-arena/temporal_spectrum.rs)
              │     bloques por escala τ (malla 4^k, 32 escalas), τ* por
              │     IC prequential (#594), significancia = e-proceso de
              │     Ville capital ≥ 1/α (#661, sustituye al umbral Fisher
              │     N_efectivo), masa SOLO de escalas observadas (D-742/CL-35)
              ├─ 13 motores (signal-engine/src/*.rs) — cada uno con
              │     evaluate* VIVO + voto_espectral() por escala
              │     moneda de la casa: excitacion_hawkes_norm = λ/μ̂ vs SS
              │     (.max(0.0): calma abstiene, TAMBIÉN en evaluate* vivos #663)
              ├─ Sombras + SkillMotores (signal-engine/skill_motores.rs)
              │     IC por motor×escala con gate Ville (#661, Fisher
              │     retirado), re-arme CAUSAL (#659: snapshot del
              │     último depth ANTERIOR al nacimiento), dedup por ts
              ├─ Consenso espectral (voto_espectral.rs consenso_por_escala)
              │     pesos por habilidad #626, gate observabilidad #648,
              │     TTL del dominante #648, publica dominante=0 si plano
              └─ Consumo (signal-engine/orchestrator.rs #624/#652)
                    dominante≠0 ⇒ dirige con su τ (clamp 30s..12h);
                    sino fallback escalar D-754; H2: τ≥30s para dirigir
        ┌──────────────────────────────────────────────────────────┘
        ├─> puertas_del_continuo (core lib.rs, D-743) — ÚNICA función con
        │     las 4 puertas (viabilidad, bayesiano, ML, vetos flujo/muro)
        │     para fast Y slow intent; sonda de banda #586
        ├─> Geometría TP/SL — H(τ) CONTINUA (#658 smoothstep ln τ),
        │     Hurst honesto por Variance Ratio (F2-C4/Ω8), σ pronosticada
        │     con gate vs persistencia (F1-C1/Ω2), primer toque analítico
        │     BM/OU (R8-A/Ω3) en risk-engine/tp_sl.rs
        ├─> Trailing (core/trailing.rs) — escalera MODULADA por
        │     persistencia espectral (S-2 real, #657), be de fees intacto
        └─> Sizing (risk-engine) — Kelly LCB, Lundberg unidades #651,
              dd-lerp medido #653, guard de correlación D-748/750b
Host vivo (src/bin/god_engine.rs ~5k líneas)
  ├─ reloj NTP seguido por el hot-loop (#657)
  ├─ IOC con evidencia terminal (CL-39/39b/39c), margen_de_envio (CL-41)
  └─ PPO slots 0-4 con PARIDAD evaluate/update (#625/#657:
        ema_ofi/prev_obi contra umbrales dinámicos de FUENTE ÚNICA;
        slot 2 λ/μ̂; slot 3 lead-lag REAL con lags medidos #658)
Evolución (evolution-engine) — demonio + walk-forward sobre barras,
  examen juzga con curvas continuas a τ de barra (AGY P26), genomes por
  from_vector con TODOS los invariantes (GENOME-GATE, AGY Ω6)
```

## 2. Invariantes duros (romperlos = bug, no decisión de diseño)

1. **Exceso sobre estado estacionario** — la excitación SIEMPRE es
   `excitacion_hawkes_norm(λ/μ̂)` (0 en SS ⇒ abstención); nunca tanh(λ/μ̂)
   crudo ni umbrales absolutos < SS. [#649/#657]
2. **Paridad voto/aprendizaje** — todo slot del PPO usa la MISMA variable y
   escala en la entrada y en el cierre; tocar un lado exige tocar el otro
   EN EL MISMO COMMIT. [#625/#657] Ídem sombra↔vivo: la física corregida
   va a AMBOS caminos o a ninguno. [#656-658]
3. **Fuente única** — umbrales dinámicos, fricción (roundtrip_friction),
   banda operable: UNA función pura; el consumidor la llama, no la copia.
   [#657/#586/XLIV-8]
4. **Relojes físicos, no de eventos** — decay/CVD por `exp(−dt/τ)` en ms
   (τ=60 s, con timestamp del exchange «T», #660); Hawkes con kernel
   α·β·dt integrado en el tiempo (#660); W₁/lead-lag con timestamp;
   retrógrados no excitan; el host sigue al NTP (#657). La memoria del
   sistema no puede depender de la tasa del feed.
5. **Causalidad prequential** — cada score usa el voto existente AL
   NACIMIENTO del bloque (re-arme con snapshot previo, #659); la maduración
   deduplica por ts; el voto vivo se muestrea al ARMARSE, nunca post-hoc.
6. **Anti-staleness del registro** — el valor publicado es SIEMPRE el del
   tick en curso (dominante=0 explícito si plano, #624); toda clave nueva
   exige escritor Y lector en el mismo commit (lección #613).
7. **Dedup de observación** — EWMA por generación de cache (D₀, #659);
   una observación cuenta UNA vez sin importar la cadencia de consulta.
8. **Continuidad C¹** — sin escalones/signum/umbrales duros en señales,
   trailing, H(τ), confianzas; las fronteras son centros de transición
   suave (smoothstep/tanh), no cortes. [ADR-0014/#658]
9. **Significancia honesta** — un IC cuenta sólo si su e-proceso de Ville
   cruza el umbral de FAMILIA M/α donde hay selección múltiple (#663:
   640 para el banco de τ*, 8320 para los 416 pares motor×escala;
   #661 retiró Fisher); gates contra el MEJOR nulo (persistencia),
   no contra climatology. [F1-A1/C1/#661/#663]
10. **El espectro sólo opina con escalas OBSERVADAS** — D-742 en fusión,
    masa, composición (#648) y funciones de estructura (XLIV-6).

## 3. Zonas por agente (no-choque)

- **Qoder (línea A: entender)** — signal-engine, quantum-arena/espectro,
  feature-engine física, core pipeline. Cola: F2-B8 + F3 MED/LOWs.
- **GLM (línea B: aprender)** — trainer/datasets/L2, barrido F5-F6
  (tubería de promoción + datos/storage). REGLA: regenerar datasets tras
  cada cambio de física de señales (avisos #649/#656/#658).
- **Antigravity (Ω)** — ejecuta el plan S1-S12 contra la misma cola del
  barrido; ha cerrado F1 (matemática), F4 (GENOME-GATE), S8/F2-C4, R8-A y
  S5 (Ω9: DSR/Gumbel + compuerta OOS 50/50 en Darwin). S1-S12 resueltos.
- **Codex** — contratos raíz (parser/OOS/modelos), PR #28 en vuelo.
- **Claude (línea C: ejecución)** — IOC/evidencia/margen (CL-36..42),
  GENOME-GATE de carga, ejecución.

## 4. Telemetría/registro — claves del contrato escritor→lector

`hawkes_intensity` = λ/μ̂ real clamp [0.1,10] (core→motores) ·
`consenso_espectral_dominante/_tau` (core→orchestrator, dominante=0 si
plano) · `spectral_persistence` (core→trailing) · `multifractal_d0_gen`
(dedup D₀) · `sombra_*` (13 motores). Toda clave nueva: par escritor/lector
verificado por grep AMBOS lados antes de pushear.

## 5. Protocolo de ola (obligatorio para cambios de conducta)

worktree aislado `.olaNN` → commits atómicos por bloque → tests del crate
+ `cargo check --workspace --all-targets` → **ORÁCULO T-1** (release,
`--test-threads=1 --nocapture`, trinquete 11.0%):
`cargo test --release -p backtest-engine --test t1_cobertura_genetica --
t1_cobertura_genetica_del_oraculo_de_aptitud --test-threads=1 --nocapture`
→ PASA ⇒ forense (FORENSIC_INTELLIGENCE_AUDIT.md) + buzón (UNIÓN en
conflicto) + MEMORIA → **push por refspec** `git push origin rama:main`
(el checkout compartido puede estar ocupado por otra sesión) → cleanup
worktree+rama. Si tocó un VETO ⇒ tocó su entrada del registro de vetos
(en el mismo commit — regla GLM LXXXVI).

## 6. Deuda viva (verificada por grep contra el árbol 09186b81, 2026-10-05 post Ω9/LXXXXVIII)

**Qoder (cola propia, verificada por grep en este corte):**
- [RONDA 2 §G: G1-1/G2-1/G2-2 CERRADOS por #663/Ola 62 (Ville familia
  M/α 640/8320, calma abstiene en confluence, exceso-SS en flow_impulse
  vivo); G1-2/G1-3/G0-4 por AGY Ω10; G0-2/G2-10/G1-5 por AGY Ω11.
  5/5 HIGH + 6/15 MED resueltos.]
- G2-3..G2-9 ola mecánica: signums/gates duros residuales (confluence
  gate OBI, perceptron signum+piso, coaxial sombra, conformal vecino,
  trend_runner escala 1e-3, shockwave sub-dólar, renyi doble gate).
- G1-4 ζ(2) mide (E|dev|)² no E[dev²] — χ sesgado alto. temporal_spectrum.
- F2-B8 IC cruzado ρ(τ) del veto de grupo sin significancia — la
  maquinaria de familia Ville (#663) ya existe para cablearlo.
  espectral_multiactivo.rs.
- F2-B8 el IC cruzado ρ(τ) del veto de grupo NO aplica significancia
  (#599) — umbral autoajustado 2/√(n−3). espectral_multiactivo.rs.
- F3-A4 sombras con knobs muertos: `quantum_k_spring`/
  `soliton_amplitude`/`nash_equilibrium_drift`/`conformal_epsilon` sin
  escritor productivo (verificado por grep en 09186b81: sólo tests en
  signal-engine escriben) — publicar del genoma o retirar la lectura.
  lib.rs:~1922/1951/2064/2088. [F3-A3 SR CERRADO en Ola 60: lee
  `microstructure_noise_variance`, escrita por core lib.rs:4210]
- F3-A5 sombra trend_runner stale-by-one (lee hurst/cvpin/atr_pct del
  tick previo). F3-A6 atr_5s ≈ ATR 1s (no √5). F3-A7 BTC/ETH
  auto-referenciales en lead-lag (parcial: la 57 añadió historia por
  coin, la auto-referencia del líder sigue).
- F3-B2 `qo_624_fraccion_espectral` = todas las monedas / coin 0
  (inflada ~N×) — orchestrator.rs:452. F3-B3 hawkes per-tick con
  ema_ofi como delta. F3-B4 familia legacy can_open_at_muerto.
- F3-C2 `execute_reduce_only_market` sin intención registrada ni
  tipado (la SALIDA de dinero con la evidencia más débil). F3-C3
  contabilidad bracket por TIPO de orden (fills ajenos contaminan
  Kelly). F3-C5 dedup core↔bracket sin lado. F3-C4/C6 fee core
  fabricado 2·(maker+taker) / fetch_open_positions tragado.
- LOWs agrupados: F2-B7/B9-B13, F3-A8-A13, F3-B5-B10, F3-C7-C13 —
  ola de limpieza mecánica.

**Consejo (coordinar antes de tocar):**
- S5 CERRADO por AGY Ω9 (1a540356): PSR/DSR con control de multiplicidad
  de Gumbel centralizado en `selection_stats` + compuerta OOS cronológica
  50/50 en Darwin. RESIDUO: `anti_bias_governor.rs` sigue SIN CONSUMIDOR
  (sólo `pub mod` en evolution-engine/lib.rs:18) — decidir cablear o
  eliminar en el consejo.
- Ω4 CERRADA (auditoría micro-vetos VERDE: ρ<0 = hedge admitido, Ω9).
- e-values anytime-valid CERRADO (#661: EProceso — λ=0.10, α=0.05,
  capital ≥ 20, n ≥ 20 — en banco de τ* y SkillMotores; ataca la
  selección de τ* y la multiplicidad de escalas). Milenio 4/4
  IMPLEMENTADO: primer toque BM/OU (Ω3), Ville (#661), Fokker-Planck/OU
  (Ω7), DSR/Gumbel (Ω9). Resto condicional: W₁-L2 (a la IOC).
- GLM: colas F5/F6 (tubería de promoción + datos) — ver su buzón.
