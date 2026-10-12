# Plan de la Línea H (Claude, prefijo LH-n) — la cadena de la meta bajo una ventaja conocida

> Fecha de apertura: 2026-10-12. Rama `claude/linea-h-ventaja-conocida`
> (desde `main` `6e212b8c`), worktree `.claude/worktrees/linea-h`. Datos en
> `docs/audit/linea_h/`. Se acopla al plan maestro de sincronización
> (`docs/PLAN_MAESTRO_SINCRONIZACION.md`), a las fases C-F0…C-F11 de la Línea C,
> al plan de la Línea G (`docs/PLAN_LINEA_G_CLAUDE_2026-10-12.md`) y al plan
> integral Ω90 de AGY. No abre otro censo: usa los que ya existen.

## 0. Por qué otra línea y por qué ésta

Al abrir esta sesión había otra sesión Claude viva en la Línea G (ola 5, rama
`claude/linea-g-ola5`, LG-21 commiteado un minuto antes). La regla del
proyecto es no pisar la rama ni el worktree de otra sesión, así que esta línea
toma el hueco que la ola 4 de la G dejó escrito en su §12.5:

> «T-2: el fixture no tiene ventaja y el sistema pierde lo que cuesta
> operarlo: correcto, pero no dice nada de la meta. Ventaja fuera de muestra:
> sin medir en ningún sitio con un procedimiento válido.»

Todo lo que mide hoy el repositorio mide **ausencia** de ventaja (T-2) o
**expresividad** de los genes (T-1). Nada mide la pregunta de la meta en su
forma falsable: **si la serie tiene una ventaja conocida, ¿la cadena de
decisión la deja pasar y la compone, o la mata? Y si la mata, ¿qué puerta y
con qué regla?** Un sistema que no compone una ventaja que sabemos que existe
no va a componer una que todavía no hemos medido. Esta línea construye el
instrumento y audita, desde la meta hacia el código, la cadena que la
convierte en órdenes.

## 1. La meta en números (lo ya derivado; no se vuelve a derivar)

- ×2 cada 72 h ⇔ tasa continua g = ln 2 / 3 ≈ 0,231 por día (Ω81, Ω90).
- Con RR 2,25 el equilibrio de acierto es p* = (SL + f)/(TP + SL) ≈ 0,31–0,37
  según la fricción (Línea F; contrato de asignación de crédito).
- Con el tamaño actual (orden mínima, ≈ 0,2 % del capital al stop) hacen
  falta miles de operaciones independientes al día (Línea F §2, Línea G §2):
  la palanca es el riesgo por operación, sólo segura con ventaja medida.
- El veto de drawdown de falsación es d* = 0,632 (ADR-0016): con 13 USD, el
  suelo es 4,79 USD.

Lo que esta línea añade: una **escala** entre «sin ventaja» (T-2) y
«tendencia determinista» (T-1), con la ventaja de cada peldaño medida sobre
la propia serie antes de que el motor la vea.

## 2. LH-1 — el T-3, escalera de ventaja conocida

Archivo nuevo `crates/backtest-engine/tests/t3_ventaja_conocida.rs`. No toca
ningún archivo existente.

**Fixture.** Cadena de Markov en el signo del retorno por vela de 1 min: el
estado oculto s_t ∈ {+1, −1} se conserva con probabilidad a y se invierte con
1 − a; el retorno logarítmico es s_t·d + ruido (uniforme centrado, σ). Con
a = 0,5 no hay estructura; con a → 1 los regímenes duran 1/(1 − a) velas de
media y la deriva acumulada domina al ruido. Máximos, mínimos y volúmenes se
sintetizan como en el T-2; 13 USD; lote fino (CL-98b); el mismo predictor
sintético del T-2 (tocón sobre la separación EMA, base 0,30), que en una serie
persistente lee el régimen con el retardo de las EMAs: un lector imperfecto,
como lo sería un bosque entrenado.

**Techo medido sin el motor (la «ventaja conocida»).** Para cada peldaño:
persistencia medida â; P(signo de la vela = estado oculto) (cuánto deja ver el
ruido); retorno medio neto de fricción a 1, 5 y 20 velas siguiendo al oráculo
(que lee el estado oculto) y siguiendo al proxy (el signo de la vela anterior,
que es lo que ven las features de flujo); y la P(TP antes que SL) de un bracket
RR 2,25 con stop 0,5 % resuelto con máximos y mínimos (si ambos se tocan en la
misma vela cuenta el SL), con su esperanza en R: p·RR − (1 − p) − f/SL.

**Lo que mide del motor.** Operaciones, acierto neto, PnL neto, capital final,
drawdown máximo y el reparto de los rechazos del riesgo como diferencia de
`risk_engine::REJECT_COUNTERS` antes y después (los peldaños corren en serie).

**Peldaños.** El nulo (a = 0,5, d = 0); a ∈ {0,6; 0,7; 0,8; 0,9; 0,95} con
d = 0,10 %; y d ∈ {0,05 %; 0,20 %} con a = 0,9. σ = 0,25 % (la del T-2).

**Pruebas.** `t3_la_serie_tiene_la_ventaja_que_declara` (propiedad del
generador, sin motor); `t3_diag_un_peldano_persistente_micro` (un peldaño
fuerte, sin `--ignored`, exige actividad como el T-2);
`t3_escalera_de_ventaja_conocida` (`#[ignore]`, la tabla completa).

**Qué NO es.** No es un trinquete ni una prueba de rentabilidad: las series son
sintéticas y la estructura es conocida. Es un instrumento de **eficiencia de
la cadena**: realizado / techo, y la lista de puertas que separan ambos.

## 3. LH-2 — revisión desde la base de la cadena meta → orden

Tres lectores de sólo lectura (tandas de 3, Plan agents, sin cargo ni git) y,
después, un escéptico por lente con la medición del T-3 en la mano. Cada
lector entrega hallazgos citados `archivo:línea` con escenario numérico y
**predicciones falsables** de lo que verá la escalera; el escéptico contrasta
predicción y medida.

| Lente | Pregunta | Zona leída |
|---|---|---|
| H1 tamaño y capital | Con ventaja conocida y 13 USD, ¿crece el tamaño con la evidencia o queda clavado en la orden mínima? ¿Qué fracción del capital se arriesga de verdad y quién la decide? ¿Sigue valiendo con 100 y 1 000 USD? | `risk-engine` (lib, kelly_envelope, leverage_matrix, ruin, drawdown, sonda), `booktick_replay::live_envelope_gate`, contabilidad de `run_backtest_native`, genes de tamaño |
| H2 evidencia y EV | ¿Sube p con los cierres y se abre la puerta EV, o p se mide contra la etiqueta equivocada (LG-5: P(neto > 0) como P(TP primero); lazos que puntúan contra ½ y no contra p*) y la puerta es absorbente? | `evidence.rs`, `calibration.rs`, puerta EV y fase de sonda, `TasaAcierto`/`conviccion_de_rama`, resultado de orden (LG-9), puerta ML B3.18, ensamble, ledger §7.5/7.6 |
| H3 dirección, τ y ritmo | En regímenes de ~20 velas, ¿qué ramas emiten, con qué τ, qué puertas del núcleo vetan, y cuántos cierres por hora dejan el enfriamiento y la sonda única? | `puertas_del_continuo`, `tau_de_la_orden`, `viable_para_entrar`, ramas 1–15, consenso espectral (#624), `tau_operativa_unificada`, consejo, sonda |

## 4. Fases de la Línea H (de la meta al último archivo)

| Fase | Qué | Puerta de salida |
|---|---|---|
| H0 | Git limpio y reserva en el buzón y el tablero | `origin/main` = `main`; ramas integradas sin worktree borradas; esta línea anunciada |
| H1 | El T-3 compila, su generador prueba lo que declara y el peldaño fuerte opera | 3 pruebas en verde; tabla de la escalera en `docs/audit/linea_h/t3_escalera.tsv` |
| H2 | Lectura de la cadena con predicciones; escépticos con la medida | `docs/audit/linea_h/lectores.tsv` y `verificacion.tsv` |
| H3 | Arreglos en zona propia (archivos nuevos y documentación) con prueba RED→GREEN; entregas por dueño en el buzón | cada arreglo con su prueba; `cargo check --workspace --all-targets` verde; T-2 y golden sin moverse salvo que se diga |
| H4 | Cierre: plan, `MEMORIA.md`, tablero y buzón; merge a `main` resolviendo conflictos con diff contra cada padre; push; borrar la rama | `main` contiene la rama; rama borrada en local y remoto |

El recorrido archivo por archivo del repositorio ya existe (Línea G, olas 1 y
2: los 627 `.rs` con veredicto; Línea C, C-F10). Esta línea no lo repite: lo
usa, y su aporte es la **ordenación por la meta** (qué puerta mata la ventaja
primero) y el instrumento que la mide.

## 5. Zonas y reglas

- **No toco**: núcleo (`god-engine-core/src/lib.rs`), riesgo, host, orquestador
  y espectro (Qoder), registro y consejo (AGY), `spectral_tape` (Codex SI),
  `booktick_replay`/`metrics` (Sol), `online_daemon` (Línea C, C-F7),
  `genome.rs` (LG-21 en vuelo). Sólo archivos nuevos en
  `crates/backtest-engine/tests/` y documentación.
- Nada de constantes nuevas a mano: cada umbral del instrumento se declara
  con su origen (la fricción del spec, la geometría RR 2,25 / 0,5 % de la
  doctrina, el σ del T-2) y se imprime en la tabla.
- Commits atómicos, un bloque por commit, prefijo LH-n. `git add` sólo de mis
  archivos.

## 6. Resultados (se rellena al cerrar cada fase)

### 6.1 Estado de Git al abrir (H0)

- `origin/main = 6e212b8c`; `main` local estaba 5 commits detrás (ola 4 y
  reserva de la ola 5 de la G); avanzado por fast-forward.
- En remoto sólo `main`. Ramas locales con commits no integrados y worktree de
  otro agente: `claude/tanda-cf4` (18, Línea C: CL-100…CL-108b, ADR-0017),
  `claude/linea-g-ola5` (1), `qoder/r7r6-aprender-medir` (9),
  `codex/integration-recovery-2026-10-07` (3). Locales sin worktree y no
  equivalentes por parche a `main`: `cf4-atr-local` (2),
  `claude/cl84-f3702-local` (2), `claude/cl99-pico` (1): de la Línea C, no se
  borran sin su acuse. Integradas pero con worktree ajeno montado:
  `codex/root-spectral-zero-replay-2026-10-11`, `sol/replay-accounting-2026-10-09`.
- Nada que borrar sin acuse de su dueño en este corte.

### 6.2 La escalera (H1)

Pendiente.

### 6.3 Lectores y escépticos (H2)

Pendiente.

### 6.4 Arreglos y entregas (H3)

Pendiente.
