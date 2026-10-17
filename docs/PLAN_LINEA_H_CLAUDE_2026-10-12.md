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

### 6.2 La escalera (H1) — medida el 2026-10-12 sobre `e0cd88b7` (release, 3 000 velas de 1 min ≈ 50 h por peldaño)

Tabla completa: [`t3_escalera_13usd.tsv`](audit/linea_h/t3_escalera_13usd.tsv),
[`t3_escalera_capital.tsv`](audit/linea_h/t3_escalera_capital.tsv),
[`t3_capital100_max_fee_035.tsv`](audit/linea_h/t3_capital100_max_fee_035.tsv);
trazas del runner en `t3_trazas_*.txt`; lado de cada apertura frente al estado
oculto en [`t3_lado_vs_estado.txt`](audit/linea_h/t3_lado_vs_estado.txt).

| a | d | oráculo 5 v / 20 v (pb) | p_TP oráculo | EV_R oráculo | ops | acierto | PnL (USD) | % capital | rechazos del riesgo | aperturas−cierres | lado del estado |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 0,50 | 0 | −11 / −11 | 0,275 | −0,31 | 36 | 0,361 | −0,126 | −0,97 | EV 929 | 0 | 0,42 |
| 0,60 | 0,10 % | −7 / −10 | 0,306 | −0,20 | 63 | 0,460 | +0,057 | +0,44 | — | 0 | 0,70 |
| 0,70 | 0,10 % | −5 / −8 | 0,313 | −0,18 | 107 | 0,477 | +0,104 | +0,80 | sonda 20 | **124** | 0,58 |
| 0,80 | 0,10 % | +4 / +4 | 0,363 | −0,02 | 90 | 0,433 | +0,048 | +0,37 | EV 1, sonda 9 | 0 | 0,58 |
| 0,90 | 0,10 % | +16 / +26 | 0,477 | **+0,35** | 44 | 0,409 | **−0,118** | **−0,91** | — | 0 | 0,66 |
| 0,95 | 0,10 % | +29 / +75 | 0,573 | +0,66 | 111 | 0,640 | +1,251 | +9,62 | — | 0 | 0,75 |
| 0,90 | 0,05 % | +1 / +7 | 0,352 | −0,06 | 81 | 0,432 | −0,143 | −1,10 | EV 502 | 1 | 0,74 |
| 0,90 | 0,20 % | +48 / +88 | 0,657 | +0,94 | 89 | 0,652 | +1,408 | +10,83 | — | 0 | 0,82 |

Los dos peldaños fuertes con la MISMA serie y más capital:

| a / d | capital | ops | acierto | PnL (USD) | % capital | rechazos | aperturas−cierres |
|---|---|---|---|---|---|---|---|
| 0,95 / 0,10 % | 100 | 7 | 0,43 | −0,002 | −0,00 | **fee_impact 3 429** | 0 |
| 0,90 / 0,20 % | 100 | 1 | 0,00 | −0,027 | −0,03 | **fee_impact 3 523** | 0 |
| 0,95 / 0,10 % | 1 000 | 30 | 0,60 | −0,498 | −0,05 | — | **5 574** |
| 0,90 / 0,20 % | 1 000 | 89 | 0,65 | +20,47 | +2,05 | — | 0 |
| 0,95 / 0,10 % | 100, `max_fee_pct` = 3,5 % | 111 | 0,64 | +1,971 | +1,97 | — | 0 |
| 0,90 / 0,20 % | 100, `max_fee_pct` = 3,5 % | 89 | 0,65 | +3,860 | +3,86 | — | 0 |

**El valle de capital** (a = 0,9 / d = 0,20 %, la misma serie, LH-1b;
[`t3_valle_de_capital.tsv`](audit/linea_h/t3_valle_de_capital.tsv), con el
pago realizado por cierre leído de `out_pnl`):

| capital | ops | acierto | PnL (USD) | % capital | gana / pierde por cierre (USD) | b real | max \|PnL\| por cierre | rechazos |
|---|---|---|---|---|---|---|---|---|
| 13 | 89 | 0,652 | +1,41 | +10,83 | 0,039 / 0,028 | 1,40 | 0,10 | — |
| 20 | 89 | 0,652 | +1,53 | +7,67 | 0,044 / 0,034 | 1,32 | 0,17 | — |
| 26 | 89 | 0,652 | +1,57 | +6,03 | 0,047 / 0,038 | 1,24 | 0,19 | — |
| **35** | **0** | — | 0 | 0 | — | — | — | **fee_impact 6 914** |
| **50** | **0** | — | 0 | 0 | — | — | — | **fee_impact 6 914** |
| **65** | **0** | — | 0 | 0 | — | — | — | **fee_impact 6 914** |
| **100** | **1** | 0 | −0,03 | −0,03 | — | — | 0,03 | **fee_impact 3 523** |
| 215 | 89 | 0,652 | +8,36 | +3,89 | 0,263 / 0,222 | 1,18 | 1,46 | — |
| 500 | 89 | 0,652 | +14,87 | +2,97 | 0,489 / 0,435 | 1,12 | 2,59 | — |
| 1 000 | 89 | 0,652 | +20,47 | +2,05 | 0,724 / 0,694 | 1,04 | 5,02 | — |

Y el aislamiento del término que dispara `fee_impact` con 100 USD
([`t3_capital100_sin_latencia_ni_suelo.tsv`](audit/linea_h/t3_capital100_sin_latencia_ni_suelo.tsv)):
con latencia 0 y suelo de deslizamiento 0 (la fricción queda en 2 × taker =
0,10 %) y el gen `max_fee_pct` en su 1 %, los dos peldaños operan (195 y 176
cierres, +2,4 % y +10,6 %): el veto lo decide `roundtrip_fee · L` con el L
del rescate hasta la orden mínima (≈ 8–10×), y bastan ≈ 3 pb de deslizamiento
por pierna para cruzar el 1 %.

**Lo que dice la medición (antes de los escépticos):**

1. **Las puertas del riesgo no matan la ventaja; la ausencia de tamaño sí.**
   En los peldaños con ventaja (EV_R del oráculo > 0) el riesgo no rechaza
   nada; el EV sólo cierra el peldaño nulo y el de deriva 0,05 % (donde el
   oráculo tampoco gana). Pero TODAS las aperturas trazadas llevan nocional
   5,2–5,9 USD (la orden mínima), a 13, a 100 y a 1 000 USD. El mejor
   peldaño (+10,8 % en 50 h ≈ 0,049/día) es la quinta parte del ritmo de la
   meta (0,231/día) con una serie en la que el oráculo gana +0,94 R por
   operación: con ese EV y esa cadencia, un tamaño en espacio de riesgo
   compondría; la orden mínima no. Confirma H1-01/H1-02 (R-20, ADR-0016 D1).
2. **Doblar la cuenta apaga el sistema.** Con 100 USD (régimen estándar,
   `micro_w = 0`) el veto `fee_impact` rechaza 3 429 y 3 523 evaluaciones y
   quedan 7 y 1 operaciones: el techo de fricción apalancada vuelve al gen
   `max_fee_pct` (taker × 20 = 1 %) y el rescate de apalancamiento hasta la
   orden mínima lo supera. Intervención sobre ese único gen (3,5 %, el valor
   micro): 111 y 89 operaciones, las mismas que con 13 USD. **Medido con el
   valle: la cuenta opera con 13, 20 y 26 USD, no opera NADA con 35, 50 y
   65 USD (0 cierres, 6 914 rechazos), hace 1 operación con 100 y vuelve a
   operar con 215.** El apagón empieza donde `micro_weight` deja de valer 1
   (≥ 3 órdenes mínimas ≈ 15 USD con la rampa hacia 0 en 50 USD) y dura
   hasta que el margen de Kelly cubre la orden mínima con L ≤ 7
   (≈ 150–215 USD). Es exactamente el tramo de la primera, la segunda y la
   tercera duplicación de la cuenta del dueño. Nuevo: **LH-02**, HIGH,
   riesgo (Línea C).
3. **Con 1 000 USD la envolvente del replay es absorbente.** Tras los 30
   cierres de arranque, `live_envelope_gate` declara no operable (Jeffreys,
   z = 1,64 fuera de micro), anula y revierte 5 574 aperturas del núcleo, que
   las repite tick a tick porque la reversión no cuenta como cierre: 30
   operaciones y después nada, sobre una serie con EV_R +0,66. Es LF-12 /
   `hallazgos.tsv host:255` medido: con la regla actual, la cuenta que llega
   a 1 000 USD se apaga sola si su acierto medido no supera la cota en 30
   cierres. Además esas reversiones no salen por `REJECT_COUNTERS`: la
   escalera sólo las ve como `opened − closes` (H1-07/H2-6).
4. **La ventaja moderada se pierde en la dirección y el ritmo.** En a = 0,9
   (regímenes de 10 velas, EV_R del oráculo +0,35) el motor pierde −0,91 %:
   la rama 15 produce ≥ 90 % de las aperturas, las ramas tendenciales 1/2/4/5
   y 13 no abren nunca (H3-03), la τ mediana es de 64–128 s sea cual sea la
   longitud del régimen (H3-09, la τ no sigue a la deriva) y el lado coincide
   con el estado oculto el 58–75 % de las veces (H3-02: la dirección es la
   vela previa ∧ la EMA). Sólo cuando los regímenes duran 20 velas o la
   deriva dobla al ruido el acierto neto pasa de 0,64.
5. El predictor sintético (tocón sobre la EMA) no veta nada (`ml_vetoes = 0`
   en todos los peldaños) y el consejo casi nada (≤ 6): en este fixture
   ninguna de las dos puertas decide.

### 6.3 Lectores y escépticos (H2)

Tres lectores de sólo lectura (1,09 M tokens, 28 min; 27 + 23 + 22 archivos
leídos) entregaron **30 hallazgos** (7 HIGH, 8 MED, 7 LOW, 8 INFO) y **27
predicciones falsables** ([`lectores.tsv`](audit/linea_h/lectores.tsv),
[`predicciones.tsv`](audit/linea_h/predicciones.tsv),
[`lectores_resumen.md`](audit/linea_h/lectores_resumen.md)). Un escéptico por
lente (0,60 M tokens, 15 min) los contrastó con la escalera medida
([`verificacion.tsv`](audit/linea_h/verificacion.tsv),
[`predicciones_veredicto.tsv`](audit/linea_h/predicciones_veredicto.tsv),
[`nuevos_escepticos.tsv`](audit/linea_h/nuevos_escepticos.tsv),
[`escepticos_sintesis.md`](audit/linea_h/escepticos_sintesis.md)):
**11 confirmados, 10 plausibles, 8 ya registrados con el mismo mecanismo, 1
refutado** (H3-08: la columna `sonda_ocupada` sí se dispara con una moneda), y
**17 hallazgos nuevos que sólo la medición reveló** (5 HIGH).

**Lo que la medición decidió sobre las tesis de los lectores:**

| Tesis | Veredicto con los números |
|---|---|
| H1: «el dinero no muere en una puerta: muere en que el tamaño no existe como variable» | Cierto con 13 USD (todas las órdenes al mínimo, PnL por cierre ≤ +0,016 USD, 0 rechazos de tamaño). Falso en cuanto sube el capital: con 100 USD muere en `fee_impact` y con 1 000 USD en la envolvente, dos puertas que los lectores no predijeron y que la tabla de rechazos no mostraba. |
| H2: «la puerta EV no es absorbente, es un trinquete hacia la reapertura; p sube con la ventaja» | Cierto: REJ_EV = 929 (nulo) y 502 (d = 0,05 %) y 0–1 en los seis peldaños con ventaja; pero REJ_EV **no es función del acierto** (0 con acierto 0,409; 502 con 0,432): lo fijan el prior de Platt a s = 0,98, los transitorios y el p_gate de cada τ (H2-N5). Fuera de micro la cadena se rompe por el **pago realizado** b, no por p: 30 cierres con acierto 0,60 y PnL −0,50 USD ⇒ b < 0,67 ⇒ Kelly ≤ 0 ⇒ envolvente cerrada (H2-N3). |
| H3: «el dinero muere en el núcleo: dirección de una vela (D-472), ventana del cruce de EMA21 y τ de minutos» | Cierto a 13 USD en lo esencial (τ 64–128 s en TODOS los peldaños, rama 15 ≥ 90 % de las aperturas, tendenciales nunca, a = 0,9 pierde con oráculo +0,35 R). La rama 15 **no** pasa por la ley anti-persecución (H3-04 queda plausible, no confirmado) y la τ de la rama 15 colapsa al nodo del minuto (H3-N4): las dos bandas operan a la misma τ y el co-despacho armónico casi nunca se cumple. |

**Predicciones:** 27 contrastadas (`predicciones_veredicto.tsv`). Acertaron
la cota de capital (≤ 15 USD con 13), el tamaño binario, los contadores de
tamaño a cero, la fracción del lado del estado (0,65–0,78 en a = 0,9; medido
0,66), la τ insensible al régimen y las ramas tendenciales muertas. Fallaron
en que la envolvente «sea inerte» en micro (124 anulaciones en a = 0,7:
ventana no monótona 0 < f < f_min de `kelly_envelope.rs:408`, H1-N3), en que
REJ_EV baje gradualmente con a (cae a 0 desde a = 0,6) y en los números por
capital (la cadena a 100 USD ni llega a dimensionar: `fee_impact` la corta
antes).

**Nuevos de los escépticos que van a su dueño** (con cita en
`nuevos_escepticos.tsv`): H1-N1 / H2-N1 / H3-N2 (el valle de `fee_impact`,
HIGH, Línea C), H1-N2 / H3-N1 (envolvente absorbente a 1 000 USD y reversión
que no enfría ni cuenta, HIGH, Línea C), H1-N3 (anulaciones en micro, MED),
H2-N3 (la envolvente mata por b realizado, MED), H2-N4 (la puerta de confianza
es inerte a 0,98: `confianza_modulada` y `min_required_confidence` comparten
el techo y la comparación es estricta, LOW, Línea E), H2-N5 (REJ_EV depende
del camino, no del acierto, MED), H3-N4 (τ de la rama 15 colapsa al minuto,
MED). Para la propia Línea H: H1-N4 / H2-N2 / H2-N6 / H3-N3 (la escalera no
ve la envolvente, no exporta tamaño ni motivo de cierre y su techo no es
comparable con el acierto neto): la primera parte se cierra en LH-1b (pago
realizado por cierre desde `out_pnl`, valle de capital); el resto exige
exportar `envelope_vetoes` y `diag_*` del runner, que no es zona H (pedido a
Sol/Línea C en el buzón).

### 6.4 Arreglos y entregas (H3)

**En zona propia (archivos nuevos y documentación):**

| ID | Qué | Prueba |
|---|---|---|
| **LH-1** | El T-3: generador con estado oculto, techo sin el motor, medición del motor con diferencias de `REJECT_COUNTERS`, trazas del runner (`TG_TRACE_NATIVO`) | `t3_la_serie_tiene_la_ventaja_que_declara`, `t3_diag_un_peldano_persistente_micro` (sin `--ignored`); `t3_escalera_de_ventaja_conocida` |
| **LH-1b** | Pago realizado por cierre desde `out_pnl` (gana/pierde medios, b real, p* real, max \|PnL\|: lo que la envolvente consume fuera de micro, H2-N3, y la cota del tamaño, H1-N4); escalera por capital e intervenciones controladas sobre un gen (`T3_CAPITALES`, `T3_MAX_FEE_PCT`, `T3_LATENCY_MS`, `T3_SLIP_FLOOR`); el valle de capital | `t3_escalera_por_capital`, `t3_valle_de_capital` |

Verificación: `cargo check --workspace --all-targets` verde sobre el árbol
fusionado con `main` (`65631504`); T-2 de control idéntico a `main`
(`a7c2da7647f26f8e`, 68 operaciones, −0,164); el golden no se toca (ningún
archivo del motor cambia).

**Entregas por dueño (en el buzón, con la prueba del T-3 que lo reproduce):**

| ID | Dueño | Sev. | Qué | Cómo se reproduce |
|---|---|---|---|---|
| **LH-02** (= H1-N1 / H2-N1 / H3-N2) | Línea C (riesgo) | HIGH | El techo de fricción apalancada vuelve al gen `max_fee_pct` (1 %) al salir de micro y el rescate de apalancamiento hasta la orden mínima lo supera: la cuenta no opera entre ≈ 30 y ≈ 150 USD | `t3_valle_de_capital`; con `T3_MAX_FEE_PCT=0.035` o con `T3_LATENCY_MS=0 T3_SLIP_FLOOR=0` vuelve a operar |
| **LH-03** (= H1-N2 / H3-N1 / H2-5) | Línea C (riesgo, replay) | HIGH | Fuera de micro la envolvente es absorbente al cierre 30 (z = 1,64, b realizado < 0,67) y su reversión no enfría ni cuenta: 5 574 aperturas revertidas en silencio | `t3_escalera_por_capital` con `T3_CAPITALES=1000` (a = 0,95), `opened − closes` en la traza |
| **LH-04** (= H1-01 / H1-02, R-20 medido) | Línea C (riesgo) | HIGH | El tamaño no es variable: todas las órdenes al mínimo con 13–26 USD; de 215 a 1 000 USD el nocional crece ×3,4 cuando el capital crece ×4,65 y la fracción cae del 42 % al 8 % | `t3_valle_de_capital` (columna `max_abs_pnl_usd`) |
| **LH-05** (= H3-03) | Qoder / Línea C (núcleo) | HIGH | Las ramas tendenciales 1/2/4/5/13 nunca abren (0 de 604 aperturas): el piso de tendencia exige un diferencial EMA9−EMA21 ≈ 2,15 σ que la deriva no alcanza en regímenes más cortos que la EMA | `t3_lado_vs_estado.txt` (histograma de ramas) |
| **LH-06** (= H3-N4 / H3-09 / H3-02) | Qoder / Línea C (núcleo) | MED | τ mediana 64–128 s en todos los peldaños, también en la rama 15: las dos bandas operan a la misma τ; con regímenes de 10 velas y EV_R +0,35 el motor pierde | fila a = 0,9 / d = 0,1 % y τ de las trazas |
| **LH-07** (= H3-01) | Qoder (núcleo, D-754) | MED | El retroceso binario del enfriamiento se amortiza a la misma velocidad con la que crece: espera ≈ k·base, no 2^k·base | aritmética sobre `stateful_engine.rs:567`, contrastada |
| **LH-08** (= H2-N5 / H2-1 / H2-2) | Línea E / Codex | MED | REJ_EV depende del camino, no del acierto (0 con 0,409; 502 con 0,432): prior de Platt a s = 0,98 con olvido hacia la puntuación cruda y p puntual sin cota | columna `rechazos` frente a `wr_neto` |
| **LH-09** (= H2-N4) | Línea E | LOW | La puerta de confianza es inerte a 0,98 (techo compartido y comparación estricta): REJ_CONFIDENCE = 0 siempre | columna `rechazos` |
| **LG-5** (= H2-3, confirmado) | Codex / Línea E | HIGH | Los cuatro estimadores de p aprenden neto > 0 con cualquier salida; la puerta EV los usa como P(TP primero) | motivos de cierre TP/SL/TRAIL_HIT/ALPHA_DECAY en las trazas |
| **Pedido a Sol / Línea C** | runner | MED | Exportar `envelope_vetoes` y `diag_*` del runner por `out_stats` o por la traza: sin eso la escalera publica «sin rechazos» donde la envolvente anuló 5 574 entradas (H1-07 / H2-6) | — |

**Lo que no hago y por qué:** ningún arreglo de riesgo, núcleo o runner,
aunque la medición los señale: son zonas de otras líneas con tandas en vuelo
(CL-101…CL-108 en el riesgo; LG-21 en el genoma). El instrumento y la
medición son el entregable; cada dueño tiene la prueba que reproduce su
hallazgo en un comando.

### 6.5 La meta con lo medido hoy (ola 1)

Con la serie más favorable de la escalera (regímenes de 10 velas con deriva
0,2 %, oráculo +0,94 R por operación, 1,8 operaciones por hora) el motor
realiza +10,8 % en 50 h: 0,049/día frente a los 0,231/día de la meta, con un
acierto de 0,65 y un pago realizado b = 1,40. Con ese acierto y ese pago el
Kelly pleno es f* = p − (1 − p)/b = 0,40 del capital por operación; el motor
arriesga 0,2 %. **No hace falta más acierto para la meta: hace falta que el
tamaño exista, que no se apague al doblar la cuenta, y que la τ de las órdenes
siga a la deriva.** En ese orden: LH-04 / ADR-0016 D1 (tamaño en espacio de
riesgo), LH-02 y LH-03 (los apagones por capital), LH-05 y LH-06 (dirección y
τ). Y sigue sin medirse la ventaja real fuera de muestra (tapes): la escalera
mide la cadena, no el mercado.

## 7. Ola 2 (2026-10-12, base `57abc601`): ¿se adapta el sistema? La ventaja conocida puesta en el tiempo

### 7.1 Método

El mandato del dueño es que el sistema sea autoadaptable y autoevolutivo. La
ola 1 midió la cadena en estado estacionario; la ola 2 la mide en el tiempo con
el mismo instrumento. El motor es determinista y una serie compuesta de dos
tramos tiene el mismo prefijo que el primer tramo solo, así que los cierres
del primer tramo coinciden en los dos casos y **la diferencia del segundo
tramo frente al mismo tramo corrido en frío es exactamente lo que el estado
aprendido en el primero hace con el segundo** (LH-10, pruebas
`t3_adaptacion_en_el_tiempo` y helpers `serie_concatenada` / `cuartiles`
añadidos al T-3 sin tocar el generador ni el techo, que la Línea G revisa en
su ola 6).

| Exp. | Serie | Pregunta |
|---|---|---|
| E1 | ventaja larga (a = 0,9, d = 0,2 %, 6 000 velas) | ¿Mejora la R por operación con la experiencia (cuartiles del índice de cierre)? |
| E2 | ventaja (3 000) → nulo (3 000) | ¿El estado ganador agrava o reduce las pérdidas del nulo frente al nulo en frío? |
| E3 | nulo (3 000) → ventaja (3 000) | ¿Una primera mitad perdedora deja al sistema fuera de la ventaja que llega después (estado absorbente)? |
| E4 | ventaja (3 000) → antipersistencia (a = 0,1, 3 000) | Cuando el signo de la vela anterior pasa a ser el contrario del siguiente, ¿invierte el motor su conducta? |

Antes de medir, tres lectores de sólo lectura inventariaron los lazos que
aprenden en cada cierre (etiqueta, memoria, consumidor, escala) y predijeron
E1–E4 con números (LH-11); después, un escéptico por lente con la medición.

### 7.2 Resultado (release, 13 USD; [`ola2_adaptacion.tsv`](audit/linea_h/ola2_adaptacion.tsv), [`ola2_adaptacion_lineas.txt`](audit/linea_h/ola2_adaptacion_lineas.txt))

Tramos en frío (referencia): ventaja (a = 0,9, d = 0,2 %) 89 cierres, acierto
0,652, +1,408 USD; nulo 36 cierres, 0,361, −0,126 USD, 929 rechazos EV;
antipersistencia (a = 0,1, d = 0,2 %) 12 cierres, 0,083, −0,241 USD, 27 EV.
En los tres compuestos los cierres del primer tramo coinciden uno a uno con
los del tramo en frío (89/89, 36/36, 89/89): el método mide lo que dice.

| Exp. | Segundo tramo tras el primero | Segundo tramo en frío | Lectura |
|---|---|---|---|
| **E1** ventaja larga, 135 cierres, +3,84 USD (+29,6 % en 100 h) | cuartiles por índice de cierre (cierres, acierto, PnL medio): Q1 (33, 0,79, +0,031) · Q2 (34, 0,62, +0,021) · Q3 (34, 0,59, +0,009) · Q4 (34, 0,74, +0,053) | — | **No hay curva de aprendizaje**: el primer cuartil ya es el mejor en acierto, el tercero el peor, y el PnL por cierre no crece con la experiencia sino con el tramo de la serie. Nada de lo que el motor aprende en 100 cierres cambia la operación 101. |
| **E2** ventaja → nulo | 40 cierres, −0,063 USD, acierto 0,40, **0 rechazos** | 36 cierres, −0,126 USD, 0,36, **929 rechazos EV** | Tras un tramo ganador la puerta EV **no se cierra en 50 h de nulo** (0 frente a 929): la pérdida es pequeña porque la orden es mínima, no porque algo frene. |
| **E3** nulo → ventaja | 65 cierres, +0,711 USD, acierto 0,615, +9 rechazos EV sobre los 929 del nulo | 89 cierres, +1,408 USD, 0,652 | Tras 50 h perdedoras el sistema **captura la mitad** de la ventaja que llega (+0,71 frente a +1,41; 65 frente a 89 cierres). La puerta reabre pronto (sólo 9 rechazos más), pero el estado heredado (ritmo, racha, calibración) cuesta 24 operaciones y 0,7 USD. |
| **E4** ventaja → antipersistencia | 45 cierres, −0,244 USD, acierto 0,333, **0 rechazos** | 12 cierres, −0,241 USD, 0,083, 27 EV | Cuando la estructura se invierte, el motor **no invierte ni se detiene**: opera ×3,75 más que en frío, con la estructura vieja, y la puerta EV no se entera en 50 h. Pierde lo mismo en dólares porque el fresco se para a los 12 cierres y el compuesto no se para nunca. |

**E1b — la misma serie dos veces** (el control que faltaba, señalado por los
tres lectores: la serie «larga» de E1 es otra realización del ruido porque la
sal del generador incluye n; aquí el segundo tramo es la MISMA realización que
el primero, así que la única diferencia entre las dos pasadas es el estado que
el motor lleva aprendido, más el nivel de precio;
[`ola2_misma_serie_dos_veces.tsv`](audit/linea_h/ola2_misma_serie_dos_veces.tsv)):

| Serie | 1.ª pasada (= en frío) | 2.ª pasada | Cierres idénticos entre pasadas | Lectura |
|---|---|---|---|---|
| ventaja a = 0,9, d = 0,2 % | 89 cierres, +1,408 USD, acierto 0,652 | 86 cierres, **+1,737 USD**, 0,640 | **0 / 86** | La segunda pasada gana un 23 % más con menos cierres y menos acierto: lo que cambia es el tamaño/la vida de cada operación, no la selección. |
| régimen medio a = 0,9, d = 0,1 % | 44 cierres, −0,118 USD, 0,409 | 72 cierres, **+0,414 USD**, 0,514, 29 EV | 0 / 44 | Donde en frío perdía, con estado caliente gana: no por aprender el régimen sino porque los estimadores ya están calientes (el tramo en frío gasta parte de sus 50 h en calentarse). |
| nulo a = 0,5 | 36 cierres, −0,126 USD, 0,361, 929 EV | **89 cierres, −0,292 USD**, 0,427, 638 EV | 0 / 36 | Con estado caliente el nulo se opera ×2,5 más y se pierde ×2,3 más: la puerta EV rechaza menos (638 frente a 929) justo donde no hay nada que ganar. |

Ni un solo cierre de la segunda pasada coincide con el de la primera: el
motor es MUY dependiente de su estado. Pero el efecto del estado es **más
actividad**, no **más discriminación**: ayuda en el régimen favorable (+23 %),
duplica la pérdida en el nulo y no frena en el adverso (E4). La adaptación que
existe es de actividad; la que hace falta es de selección.

**Por qué (con la cita):** la p que alimenta la puerta EV es un ajuste de Platt
sobre las últimas `WINDOW = 512` operaciones (`calibration.rs:45`) con un
prior de 3,84 pseudo-observaciones a la puntuación cruda (0,98 en todas las
órdenes); el olvido por tiempo (`calibrate_at`, semivida 12 h) sólo actúa
entre cierres, y con 1–2 cierres por hora no actúa nunca. Con 89 cierres al
65 % en la ventana, 45 cierres al 33 % en antipersistencia no bajan p por
debajo del umbral de la puerta: **la memoria efectiva del permiso es de
~500 operaciones (≈ 10 días a este ritmo), y no hay ningún lazo que reaccione
en la escala de un régimen (10–20 velas) ni de un tramo (50 h)**. El único
estado que cambia de verdad entre cierres es el enfriamiento por racha, y
LH-07 mostró que crece linealmente, no en 2^k.

**Lo que esto dice de la meta.** El sistema no es autoadaptable en la escala en
la que cambia la estructura que explota: aprende despacio (ventana de 512
cierres), no desaprende (E4) y arrastra el estado perdedor a la ventaja que
sigue (E3). Y aunque aprendiera, no tendría dónde ponerlo: el tamaño no
cambia (E1: el PnL por cierre del cuarto cuartil es el del tramo, no el de la
experiencia). La adaptación que la meta necesita es de dos escalas: una
rápida, en la que la estructura cambia (regímenes de minutos, medida con
e-procesos por rama/escala como los de #661 pero sobre la R neta frente a p*),
y una lenta para el tamaño (ADR-0016 D1).

### 7.3 Lectores y escépticos

Tres lectores (0,93 M tokens, 20 min; 30 + 32 + 27 archivos) inventariaron
**46 lazos que aprenden en cada cierre** con etiqueta, memoria, consumidor y
escala ([`ola2_lazos.tsv`](audit/linea_h/ola2_lazos.tsv)), entregaron 25
hallazgos y 32 predicciones falsables de E1–E4 ([`ola2_lectores.tsv`](audit/linea_h/ola2_lectores.tsv),
[`ola2_predicciones.tsv`](audit/linea_h/ola2_predicciones.tsv)). Tres
escépticos (0,59 M tokens, 15 min) los contrastaron con la medición
([`ola2_verificacion.tsv`](audit/linea_h/ola2_verificacion.tsv),
[`ola2_predicciones_veredicto.tsv`](audit/linea_h/ola2_predicciones_veredicto.tsv),
[`ola2_nuevos_escepticos.tsv`](audit/linea_h/ola2_nuevos_escepticos.tsv),
[`ola2_escepticos_sintesis.md`](audit/linea_h/ola2_escepticos_sintesis.md)):
**13 confirmados, 3 plausibles, 7 ya registrados, 2 refutados** (A1-03: el
umbral epigenético SÍ tiene lector, `risk-engine/src/lib.rs:1109-1120`, y es
inerte por el techo 0,98 de H2-N4; A2-02: la ventana ×6 de la envolvente es
inalcanzable porque sólo evalúa con la cartera vacía). Predicciones: 6
cumplidas, 11 parciales, 9 falladas, 6 no medibles; **16 nuevos** de la
medición. Correcciones al inventario en
[`ola2_correcciones.md`](audit/linea_h/ola2_correcciones.md).

**Lo que la medición decidió:**

| Tesis | Veredicto con los números |
|---|---|
| A1: «tras 89 cierres la operación 90 es igual a la 1; la p de Platt heredada no cierra la puerta» | Mecánica confirmada (0 rechazos EV en E2₂ y E4₂; 9 en E3₂). Consecuencia equivocada: la puerta abierta no agrava (E2₂ −0,063 frente a −0,126), y **lo que sí cambia con la experiencia es el pago por cierre y la cadencia**, que la lente dio por inertes a 13 USD. |
| A2: «ningún freno del riesgo actúa en micro; el estado ganador agrava el nulo; E3 recupera ≥ 85 %» | Frenos inertes: confirmado (dd máximo 2,6 %; 0 reversiones; 0 rechazos salvo EV). El signo de las consecuencias, refutado: E2 no agrava, E3 recupera el 50 %, el antipersistente frío lo frena el núcleo (43 intenciones en 50 h), no Platt. |
| A3: «ningún lazo puede invertir el lado; la τ es una ley cerrada» | **Confirmado (HIGH)**: E4₂ acierta 0,333 ≈ seguir la vela previa (0,32), y lo hace ×3,75 veces más que en frío con cero frenos. |

**Las tres lecturas convergen:** el sistema adapta la EXPOSICIÓN (cadencia,
pago por cierre: E1 ×1,6 por cierre con acierto plano; E4 ×3,75 aperturas),
no la SELECCIÓN (ni dirección ni τ ni permiso útil). Y la puerta EV, el único
lazo del riesgo que decide a 13 USD, vale ≈ 4 cierres y ≈ 0,01–0,06 USD en
50 h de nulo (A2-N2): cerrada o abierta, el resultado es el mismo.

### 7.4 Arreglos y entregas

**En zona propia:** LH-10 (`serie_concatenada`, `cuartiles`,
`t3_adaptacion_en_el_tiempo`), LH-10b (`t3_adaptacion_misma_serie_dos_veces`,
el control que pedían los tres lectores). Verificación: check del workspace
verde sobre el árbol fusionado; T-2 de control idéntico a `main`.

| ID | Dueño | Sev. | Qué | Reproducir |
|---|---|---|---|---|
| **LH-12** (= A1-01 / A2-01 / LH-A3-N6) | Línea E / Codex | HIGH | La memoria del permiso es la ventana de 512 cierres del Platt; la semivida de 12 h sólo corre en el hueco entre cierres: operando, la evidencia no envejece. Tras 89 cierres de ventaja hacen falta ≈ 146 de antipersistencia para cerrar la puerta (en frío, 12). | E2/E4: 0 rechazos EV en 50 h |
| **LH-13** (= LH-A3-1 / LH-A3-N1) | Qoder / Línea C (núcleo) | HIGH | Ningún lazo puede invertir el lado (D-472, hebbiano y PPO positivos, Platt con pendiente ≥ 0); el estado ganador heredado multiplica ×3,75 la exposición al régimen adverso sin que ningún freno actúe. | E4₂: 45 cierres al 33 %, 0 rechazos |
| **LH-14** (= LH-A3-2 / LH-A3-N3) | Qoder (espectro) | HIGH | La τ de la orden es una ley cerrada (media geométrica entre la τ dominante modulada por Hurst y el centroide, clavado en 30 s por la persistencia negativa de las escalas sub-minuto del puente intravela); la deuda del e-proceso del nodo de 68,7 s tarda ≈ 10 h en pagarse: la mitad de la ventaja de E3 se pierde ahí. | E3₂: 65 cierres / +0,71 frente a 89 / +1,41 |
| **LH-15** (= A2-N2 / A2-N3) | Línea C (riesgo) / Línea E | MED | La puerta EV no protege: en el mismo camino, cerrada 929 veces (nulo frío) o abierta todo el tramo (E2₂) el resultado es el mismo; lo que limita la pérdida en antipersistencia es la tasa de propuesta del núcleo, no el riesgo. | E2, E4 frente a frescos |
| **LH-16** (= LH-A3-4 / LH-A3-6) | Qoder (núcleo) | MED | El peso hebbiano satura en 2,0 a los ≈ 23 cierres y vuelve a −0,008 por cierre; la rama 11 se apaga con Hurst < 0,45, justo donde un motor de reversión ganaría peso. | aritmética contrastada |
| **LH-17** (= A2-04) | Sol / Línea C (runner) | LOW | El runner suaviza b dos veces (EWMA α 0,05 en el runner y otra en `record_trade`) y clasifica pnl ≥ 0 como ganancia donde la envolvente usa > 0. | `backtest lib.rs:389-409` |
| **Propio (ola 3)** | Línea H | — | Publicar el pago del segundo tramo y el fresco en la fila ADAPT; E4' anti → anti y E5 antipersistencia larga por cuartiles (prueba directa de inversión); guardar las trazas completas y extender `lado_vs_estado.py` a series compuestas; pedir fuera de zona que el tope de 100 trazas del núcleo sea configurable. | — |

### 7.5 Re-medición con el PR #62 de la Línea C (`fc2ba2ec`, árbol fusionado)

La Línea C avisó de que la tanda C-F4 cambia puertas que el T-3 mide (nuevo
veto `estrategia_falsada`, slot 18: proceso e de Ville sobre la R de cada
cierre con H0 = +0,30 R; freno duro K·s̄/pico; R-15 en sombra; D4). Y la
Línea G revisó el T-3 (31 hallazgos, 3 HIGH; `docs/audit/linea_g/ola6_t3_linea_h.tsv`).
Medido sobre el árbol con el #62
([`ola2_post_pr62_escalera_13usd.tsv`](audit/linea_h/ola2_post_pr62_escalera_13usd.tsv),
[`ola2_post_pr62_valle_de_capital.tsv`](audit/linea_h/ola2_post_pr62_valle_de_capital.tsv),
[`ola2_post_pr62_adaptacion.tsv`](audit/linea_h/ola2_post_pr62_adaptacion.tsv)):

- **La escalera de 13 USD, el valle de capital y E1–E3 son idénticos al
  céntimo** a los de antes del #62: ninguna de las puertas nuevas actúa en
  ellos (el freno de la estrategia se alimenta en el cierre del núcleo,
  `lib.rs:4725`, pero no se falsa en 50 h de ningún peldaño; el valle de
  `fee_impact` sigue intacto: 35–65 USD → 0 cierres). El T-2 de `main` sí
  se movió con el #62 (de `a7c2da76`, 68 operaciones, −0,164 a
  `cb690a15`, 65, −0,093): lo que lo mueve no está en los peldaños del T-3.
- **E4 cambia, poco**: el freno nuevo se falsa al final del tramo
  antipersistente (31 evaluaciones ≈ una vela, tras ≈ 40 cierres perdedores):
  41 cierres en vez de 45, −0,265 USD en vez de −0,244, acierto 0,317. Es la
  primera puerta adaptativa que reacciona a un régimen adverso heredado, y
  reacciona cuando el tramo ya ha pasado: con H0 = +0,30 R y la cadencia de
  un cierre por hora, el proceso e necesita decenas de cierres para
  rechazar. Lo que la meta pide es que reaccione en la escala del régimen
  (10–20 velas), por rama y escala, sobre la R neta frente a p*: la misma
  idea, otra escala y otra granularidad.
- **De la revisión de la Línea G, aceptado para la ola 3 (T-3 v2)**: H1-01
  (HIGH): el ruido uniforme acotado y las mechas y el volumen construidos con
  el mismo |u| hacen el estado oculto decodificable del OHLCV; el «oráculo»
  de las olas 1 y 2 es un lector de lo que el motor también ve, no una cota,
  y lo realizado en a = 0,6–0,8 no se atribuye a la cadena. La v2 llevará
  ruido gaussiano, sorteos independientes para mechas y volumen y el lector
  causal óptimo (filtro de Bayes sobre el OHLCV) como techo, conservando la
  v1 para reproducir las tablas publicadas. Aceptados también H1-07 (el
  apagón empieza hacia 27,5 USD), H1-04 (el −0,91 % de a = 0,9 tiene
  |t| < 0,6: indicio, no prueba), H1-10 (fricción del techo 10 pb frente a
  13–14 pb del motor), H3V-04 (constante de la EMA21 = 10,5 velas), H3V-07
  (LH-07: el retroceso del enfriamiento es geométrico de razón √2, no
  k·base ni 2^k) y H2-08 (fricción en las dos ramas del dimensionado,
  `r* = p/(1 + f_R) − q/(RR − f_R)`).

## 8. Ola 3 — revisión del instrumento y cierre del 15 de octubre

Base histórica: `aecc33d4`; commits LH-18/LH-18b hasta `2359ba19`.
La revisión y los datos `ola3_*` recuperados pertenecen a ese trabajo previo.
Las tablas no incluyen una huella verificable del binario y del código exacto
que las produjo. Se preservan como evidencia histórica; no certifican Ω128 ni
el motor actual. El manifiesto `ola3_historical_manifest.json` fija sus bytes.

### 8.1 Cambios del instrumento

V2 usa ruido gaussiano con Box–Muller y sorteos separados para mechas y
volumen. El filtro HMM usa los parámetros conocidos del generador y el calendario
de tramos: es una referencia privilegiada para cierres observados. No constituye
un techo universal de PnL por operación, ni comparte la información intravela
del runner OHLC, cuyo puente conserva el canal H1-01(iii).

La revisión corrige la dirección del lector latente en antipersistencia;
reinicia el posterior en cada tramo; omite retornos y Brier que crucen la unión;
usa log-odds para evitar perder evidencia por subdesbordamiento gaussiano;
publica abstención por fricción junto a su cobertura; y representa ausencia de
operaciones con NaN. El retorno esperado que decide la abstención es logarítmico
y asume las comisiones de referencia; no es un modelo completo de ejecución.

El bracket informa tanto el EV condicional legado como el promedio en R de
entradas con seguimiento completo, con vencimientos al precio final. Excluye
la censura al final de serie y las uniones de tramo; supone fills exactos en
barreras y cuenta SL primero si ambas se tocan en una vela. No mide impacto,
gaps ni fills reales. La correlación ruido-volumen detecta la fuga conocida con
V1 como control positivo; no demuestra independencia frente a toda alternativa.

### 8.2 Qué permiten concluir los datos históricos

En la corrida guardada E3 (nulo→ventaja), el segundo tramo tuvo cero cierres
y el contador total `estrategia_falsada` fue 2075; el tramo equivalente en frío
registró 169 cierres y +2,33107 USD. Es un testigo de bloqueo en aquella
composición, pendiente de repetirse con el motor actualizado. E4 guardó 119
cierres y −0,20895 USD frente a 63 y −0,12671 en frío.

Estas diferencias no aíslan aprendizaje: también cambian capital, precio,
posiciones heredadas, geometría y estado de filtros. El corte por índice de
cierre sólo atribuye temporalmente si se verifica el prefijo; una posición
abierta al cambiar el tramo mezcla las dos poblaciones. Los cuartiles de una
realización no demuestran ausencia de aprendizaje ni inversión. V1 y V2
cambian colas, rango, mechas y volumen, por lo que su comparación no identifica
causalmente el efecto de eliminar una fuga. El crecimiento observado en un
fixture tampoco demuestra duplicación sostenida cada 72 horas.

### 8.3 Validación y trabajo restante

El modelo puro se comparte desde `tests/support/t3_synthetic.rs` y admite
pruebas con `rustc --test` sin compilar Polars. Los contratos exactos cubren
abstención, dirección, censura, vencimientos, evidencia extrema y causalidad
de prefijo. La compilación del workspace, pruebas de integración y nuevas
mediciones se registran en `docs/audit/LINEA_H_T3_V2_CIERRE_2026-10-15.md`.
Sólo ese registro determina qué se verificó contra la composición actual.

Pendientes fuera de esta entrega: replay causal de eventos reales, atribución
de fills por tiempo y nocional, múltiples semillas con intervalos que respeten
dependencia, y evaluación fuera de muestra de los cambios de política.

### 8.4 Cierre de la ola 3 (2026-10-17) y alcance de la ola 4

El PR #64 (rama `codex/t3-v2-2026-10-15`, LH-18/18b míos y LH-18c de Codex
por failover de AGENT_SYNC §3) entró en `main` en `69ccd80e` (2026-10-17, 11:40 local) tras fusionar
`origin/main` (e0d8e501) con el buzón y la MEMORIA por unión. Verificado en local sobre el árbol fusionado: `cargo check --workspace --all-targets` 0 errores / 0 avisos (3 min 02 s de compilación tras 25 min de cola en el semáforo LG-25), 12/12 pruebas rápidas del T-3 en release (los 10 contratos std-only, la cabecera TSV y el diagnóstico); CI del PR 6/6 verde en la cabecera anterior y relanzada tras el merge.

Diagnóstico sobre este `main` (a = 0,95, d = 0,1 %, 2 000 velas, 13 USD): v1 62 cierres, acierto 0,661, **+1,72 USD (+13,2 %)** frente a 64 / +1,02 / +7,8 % en la ola 2 sobre `aecc33d4`: **LH-04 ya mueve el tamaño** (max |PnL| por cierre 0,164 frente a 0,068); v2 45 cierres, 0,556, +0,32 USD y **2 593 rechazos `correlacion` en un fixture de UNA moneda** (LH-21, abierto; se mide en la ola 4).

**Ola 4** (rama `claude/linea-h-ola4`): (1) re-medir escalera, valle y E1–E5
con la v2 sobre `main` con LH-04 (GLM 117/118) y comparar con las tablas
históricas `ola3_*` (manifiesto en `ola3_historical_manifest.json`); (2) la
verificación barata de LH-06 (`docs/DISENO_LH06_TAU.md` §5: τ mediana por
peldaño y |Δln τ| fast–slow desde `[OPEN TRACE]`, con el pedido a Qoder/AGY de
`tau_hab=` en la traza y del tope de 100 trazas configurable); (3) LH-21: los
rechazos `correlacion` en un fixture monoactivo; (4) LH-20 (freno absorbente
CL-107) al dueño y a la Línea C.
