# Plan de Línea F — revisión independiente de composición, lazos adaptativos y campo multiactivo (2026-10-11)

> Dueño: Claude (sesión cloud «laughing-fermi»), prefijo **LF-n**, rama
> `claude/laughing-fermi-xoqnb7`. Base de este plan: `origin/main`
> `977ba2f8` (Ω84, #722). Las citas `ruta:línea` son de esa revisión:
> re-grep del símbolo antes de editar.
>
> Este documento **no crea otro censo ni otro recorrido**. Se acopla a los
> que ya existen y les añade un frente que nadie tenía asignado:
>
> - recorrido R0–R9 (Qoder/AGY y Codex/Sol):
>   [PLAN_REVISION_ARCHIVO_POR_ARCHIVO_2026-10-07.md](PLAN_REVISION_ARCHIVO_POR_ARCHIVO_2026-10-07.md);
> - fases C-F0…C-F11 y lista de recibos por archivo (Línea C, Claude):
>   [PLAN_LINEA_C_DESDE_LA_BASE_2026-10-11.md](PLAN_LINEA_C_DESDE_LA_BASE_2026-10-11.md),
>   [audit/REINICIO_BASE_FASES_C_2026-10-11.tsv](audit/REINICIO_BASE_FASES_C_2026-10-11.tsv);
> - reinicio coordinado y protocolo de recibo (Codex SI):
>   [audit/REINICIO_COORDINADO_BASE_2026-10-11.md](audit/REINICIO_COORDINADO_BASE_2026-10-11.md);
> - plan rector (AGY, Ω81):
>   [PLAN_MAESTRO_REVISION_BASE_UNIVERSO_ESPECTRAL_2026-10-11.md](PLAN_MAESTRO_REVISION_BASE_UNIVERSO_ESPECTRAL_2026-10-11.md);
> - contratos entre agentes: [PLAN_MAESTRO_SINCRONIZACION.md](PLAN_MAESTRO_SINCRONIZACION.md) §45.

## 0. Qué hace la Línea F y por qué hacía falta

Desde el 9 de octubre, AGY empuja olas directamente a `main` (Ω54…Ω84) sin
PR ni revisión independiente, y cada una cambia el camino vivo. La Línea C
midió el coste de #720 (+15,2 µs por evento) y por eso nació #722; pero
**nadie revisa la semántica de lo que entra sin PR**. Los barridos de
Qoder (R6, R7) y de Línea C (C-F1) encontraron, cada vez, que los arreglos
de una ola cargaban un defecto nuevo («todo fix carga bug», cuatro rondas
seguidas). La Línea F toma tres frentes sin dueño:

| Frente | Pregunta | Salida |
|---|---|---|
| F-A. Revisión de composición | ¿Lo que entró a `main` sin PR cambia conducta que su mensaje no declara? | Hallazgos LF-n con test que falla antes del arreglo; arreglo sólo si el archivo no está reservado, si no, pedido al dueño. |
| F-B. Lazos adaptativos | ¿Qué «aprendizaje» está realmente cerrado en vivo, cuál está abierto (sólo telemetría), cuál es absorbente y cuál es un literal disfrazado? | Mapa §4 con escritor, lector, reloj y clasificación por lazo. |
| F-C. Campo multiactivo y continuo en τ | ¿Dónde el código vivo colapsa (moneda, τ) a un escalar por moneda o global y se cruzan señales entre activos o escalas? | Hallazgos §5 con escenario numérico. |

La Línea F **no posee archivos de producción**: revisa los de todos. Edita
sólo (a) sus documentos, (b) tests nuevos propios y (c) arreglos mínimos de
regresiones introducidas por commits sin PR cuando el archivo no tiene
reserva activa en el buzón; en todo otro caso entrega reproducción al dueño.

## 1. Estado medido de Git (2026-10-11, al abrir la línea)

| Ref | SHA | Estado |
|---|---|---|
| `origin/main` | `977ba2f8` | Ω84 (#722) de AGY, empujado sin PR. |
| `claude/auditoria-deslizamiento-apalancamiento-sqtc08` | `ab2f4671` (avanza) | Línea C, PR #40 abierto (C-F2). **Activa: no tocar.** |
| `claude/elegant-euler-mmtht4` | `f1a70192` | Línea E. Ancestro de `main` (0 commits propios). **Se intentó borrar del remoto: el proxy git de la nube corta el `push --delete`** (mismo bloqueo que registró Línea C). Queda para el dueño o una sesión local: `git push origin --delete claude/elegant-euler-mmtht4`. |
| `claude/laughing-fermi-xoqnb7` | igual a `main` al abrir | Esta línea. Alguien la había borrado del remoto; se re-publica con este plan. |

PR abiertos: sólo #40 (Línea C). Lo que AGY empuja sin PR no tiene CI
previa al merge: la CI de push a `main` es la única red.

## 2. La meta, en números que pueden refutarla

Se adopta la definición de Codex SI (REINICIO §2): sobre patrimonio neto
MTM, \(G_{72}(t)=\log V(t+72h)/V(t)\) y la referencia \(\log 2\) por
ventana, es decir **0,231049 de crecimiento log por día (≡ 25,99 %
diario compuesto)**. La pregunta útil no es «¿se puede?» sino «¿qué
ventaja y cuántas operaciones independientes por día exige?».

Para una apuesta binaria con relación R:R = b = 2,25 (el piso de RR que
usan los planes), probabilidad de acierto p y fracción del capital
arriesgada al stop f, el crecimiento log por operación es
\(g(f)=p\ln(1+fb)+(1-p)\ln(1-f)\), máximo en \(f^*=p-(1-p)/b\) (Kelly).
Operaciones independientes por día necesarias = 0,231049 / g:

| Ventaja neta por operación (en R) | p que implica | f* (Kelly pleno) | Kelly pleno | ½ Kelly | ¼ Kelly | Riesgo actual ≈ 0,196 % (1 orden mínima, SL 0,5 %) |
|---|---|---|---|---|---|---|
| +0,02 R | 0,314 | 0,89 % | 2 609 /día | 3 474 | 5 949 | 6 632 |
| +0,05 R | 0,323 | 2,22 % | 420 | 558 | 954 | 2 469 |
| +0,10 R | 0,338 | 4,44 % | 106 | 140 | 239 | 1 207 |
| +0,20 R | 0,369 | 8,89 % | 27 | 36 | 60 | 597 |
| +0,4625 R (p = 0,45, el supuesto de los planes Ω) | 0,450 | 20,6 % | 5,2 | 6,8 | 11,4 | 256 |

Lectura (calculado, no medido; independencia supuesta, y con correlación
entre operaciones el número crece):

1. Con el tamaño actual (riesgo ≈ 0,2 % por operación: orden mínima de
   5,10 USD y stop de 0,5 % sobre 13 USD) la meta exige cientos o miles de
   operaciones independientes al día **aun con ventajas que nadie ha
   medido**. Coincide con QS-R0 de Línea E: la palanca es el riesgo por
   operación, y sólo es segura con ventaja OOS medida.
2. El supuesto p = 0,45 con RR 2,25 de los planes Ω equivale a +0,46 R
   por operación: una ventaja excepcional que **ningún fixture ni tape ha
   mostrado** (el T-2 mide acierto real 0,20 y p calibrada mediana 0,377;
   ledger §7.5). No debe usarse como dato.
3. Por tanto el orden de trabajo para la meta es: (i) medir ventaja OOS
   neta con etiquetas honestas y nulos; (ii) dimensionar en espacio de
   riesgo sobre la cota inferior de esa ventaja (D1 de ADR-0016); (iii)
   sólo entonces subir frecuencia/activos. Ninguna teoría nueva sustituye
   (i).

Falsadores: G₇₂ por ventana con su distribución (no la media), fracción de
ventanas con G₇₂ ≥ log 2, drawdown e insolvencia; ventaja OOS con
intervalo en las dos mitades del examen; independencia de operaciones
(correlación de resultados por moneda y por τ).

## 3. Fases de la Línea F y su acople con R / C-F

Cada fase termina con: hallazgos LF-n en §6, pruebas ejecutadas con su SHA
y comando, entrada en el buzón y en MEMORIA, y PR propio.

| Fase LF | Contenido | Acople | Estado |
|---|---|---|---|
| LF-F0 | Git, ramas, PRs, recibo de qué llegó a `main` | R9 / C-F10 | Hecha (§1). |
| LF-F1 | Revisión de composición de Ω82–Ω84 (registro, reloj, getters estrictos, `sym`) | R1 / C-F1 (contratos) | Esta ola (§6). |
| LF-F2 | Mapa de lazos adaptativos (F-B) | R6 / C-F7 y §5 de Línea C | Esta ola (§4). |
| LF-F3 | Campo multiactivo y τ (F-C) | R1/R5 / C-F3, C-F4 | Esta ola (§5). |
| LF-F4 | Vetos: alcanzabilidad, frontera, recuperación desde cero posiciones, precedencia | R5 / C-F3–F5, protocolo REINICIO §6 | Siguiente ola. Por cada veto del registro y cada `return`/`clamp` del gate: test de admisión y de rechazo que llegue al gate, ambos lados de la frontera, y prueba de que no es absorbente. |
| LF-F5 | Revisión de cada ola nueva que entre sin PR, en las 24 h siguientes | transversal | Permanente. |

Recorrido archivo por archivo: la Línea F usa la lista
[REINICIO_BASE_FASES_C_2026-10-11.tsv](audit/REINICIO_BASE_FASES_C_2026-10-11.tsv)
(1 584 rutas, una fase dueña por ruta). No crea otra. Su cola propia son
los archivos tocados por commits sin PR y sus consumidores directos; cada
revisión deja recibo con el formato de REINICIO §3 (ruta, blob, rangos
leídos, contrato, productores, consumidores, hallazgos, pruebas, límites).
Revisar una función no cierra un archivo.

## 4. Lazos adaptativos (F-B)

«Autoadaptativo» se define operativamente: **observación → actualización →
consumo en una decisión viva**, con reloj causal, y con salida de los
estados absorbentes sin intervención humana. Cada lazo se clasifica:

- **CERRADO-VIVO**: escritor y lector vivo verificados con `ruta:línea`;
- **ABIERTO**: se actualiza pero ninguna decisión viva lo consume (o el
  lector existe y el escritor está muerto);
- **ABSORBENTE**: hay un estado del que el lazo no sale solo (p. ej. un
  gate que bloquea las operaciones que darían la evidencia para reabrirlo);
- **LITERAL**: lo «adaptativo» lo decide una constante en el camino vivo.

El mapa medido está en §4.1 (se completa con la revisión de esta ola).

### 4.1 Mapa

(ver §6, hallazgos LF-F2)

## 5. Campo multiactivo continuo en τ (F-C)

Doctrina (ADR-0014): no hay scalping/swing; cada posición vive en su τ
continuo (`Position::entry_tau_ms`), hasta 3 ranuras por moneda separadas
en ln τ; el mercado es un campo \(\mathcal S(\text{instrumento}, \tau, t)\)
con incertidumbre. Las pruebas transversales que la Línea F exige a todo
cambio del camino vivo:

1. **Permutación de activos**: permutar monedas junto con sus specs,
   modelos y estado deja las decisiones permutadas igual.
2. **Aislamiento**: escribir el estado de la moneda A no cambia la
   decisión de B salvo dependencias declaradas (correlación, margen común).
3. **Aislamiento entre τ**: un cierre a τ = 4 h no cambia la admisión de
   un candidato a τ = 30 s de la misma moneda salvo por margen/riesgo
   común declarado.
4. **Espejo largo/corto** con costes simétricos.
5. **Invariancia a la cadencia** de consulta y a eventos tardíos,
   duplicados o con igual timestamp.

Hallazgos de esta ola: §6, LF-F3.

## 6. Hallazgos de esta ola

(se completa al cerrar la ola)

## 7. Teoría: estado, siguiente paso y prueba que la falsa

Se aplica ADR-0005 (existe / parcial / candidata / especulativa; la
medición en datos reales es el portón). La tabla completa por dominio está
en §3 del plan de Línea C y en la Parte IV del plan rector de AGY; aquí
sólo se fija **el orden** y se corrige lo que esos planes presentan como
hecho sin serlo.

| Pieza viva | Siguiente paso | Prueba que debe pasar antes de cablear | Dueño propuesto |
|---|---|---|---|
| Bosque aleatorio por moneda (`NanoForest`) | Gradient boosting por moneda con etiqueta de **primer toque** TP/SL del lado y log-loss OOS; después calibración (isotónica/Platt con edad por dato). | Log-loss y Brier OOS mejores que el bosque y que la base, en las dos mitades, con el mismo presupuesto de datos; paridad 48D train/serve. | GLM (B) |
| Calibración por moneda | Calibración condicional a (moneda, τ) con olvido por tiempo y corrección del sesgo de selección (sólo aprende de lo ejecutado). | Contrafactual del libro sombra: la p de las intenciones vetadas no difiere de la de las ejecutadas a igual score. | Línea E (QS-R4) |
| Espectro temporal y τ* por habilidad (#594, #661 Ville) | e-valor de familia también para la τ de la **orden** (no sólo telemetría); incertidumbre de τ* propagada al tamaño. | En ruido barajado, la τ elegida no tiene habilidad y el tamaño cae a la sonda. | Qoder (A) |
| Primer toque analítico (BM con deriva, `tp_sl.rs`) | OU / difusión con saltos (Hawkes en la intensidad) para el primer toque; volatilidad rugosa sólo si H < 0,5 se mide. | Frecuencia empírica de TP antes de SL en tapes vs la fórmula, por decil de score. | Codex (R8-A) |
| Kelly / ruina (Cramér-Lundberg, LCB) | Dimensionar en espacio de riesgo: riesgo = fracción de Kelly sobre la cota inferior de la ventaja, nocional = riesgo / SL, apalancamiento = consecuencia (D1). | Con ventaja 0 el tamaño es la sonda mínima; con ventaja medida, crecimiento log OOS > 0 en cada mitad. | Línea C + E |
| Dependencia entre activos (correlación, RMT, Hodge del contagio) | Riesgo de cartera monetario ponderado por posición, τ y stop, con reserva atómica de margen. | Dos admisiones concurrentes nunca gastan el mismo margen; correlación desconocida = adversa. | Codex (D) |
| Evolución (Darwin, demonio, Ville) | Multiplicidad acumulada entre reinicios; promoción sólo con DSR/e-valor OOS; rollback al linaje correcto. | Registro de consultas al holdout; un genoma promovido sobrevive al reinicio y el rollback restaura el padre exacto. | GLM / Codex |
| Conducta de la multitud (L/S ratio, liquidaciones, Prospect Theory) | Tratarlo como **observable** con contribución marginal OOS, no como «lectura de pensamientos». | Ablation: quitar el asiento no empeora el log-loss OOS ⇒ se retira. | Qoder |

**Teorías «del milenio» y analogías físicas — triage honesto:**

- *Navier–Stokes*: el «número de Reynolds del libro» es una razón de
  magnitudes de microestructura con nombre físico; no resuelve ninguna
  ecuación. Se conserva sólo si su contribución marginal OOS es positiva.
- *Yang–Mills*: con todos los pares cotizados contra USDT, la holonomía de
  log-precios con β = 1 es **idénticamente cero** (los ciclos telescopian;
  R6-C2). Con β estimado es un residuo de cointegración, no un campo gauge.
  Útil sólo como estadístico de cointegración con su nulo.
- *Hodge*: la descomposición de Helmholtz–Hodge de un flujo dirigido entre
  activos sí es matemática aplicable (gradiente vs rotacional en un grafo);
  su valor depende de que el flujo de entrada no sea un gradiente por
  construcción (R6-A1).
- *Riemann, P vs NP, Birch–Swinnerton-Dyer, Poincaré*: sin observable de
  mercado identificable. Se registran como especulativos y no se cablean.
- *Cuántica*: todo lo que hay es cálculo clásico con amplitudes complejas
  («inspiración cuántica»). No acredita ventaja cuántica; se evalúa como
  cualquier otro estimador.

## 8. Recursos y compilación en la nube (lo que funcionó hoy)

- `apt-get install gcc-mingw-w64-x86-64 wine64` funciona a través del proxy
  de la sesión (≈ 3 min). Después `rustup target add x86_64-pc-windows-gnu`.
- Variables: `RUSTFLAGS="-C target-cpu=x86-64-v2 -C opt-level=3"` (el
  contenedor puede reanudarse en otra CPU; `native` dejó binarios con
  AVX-512 FP16), `CARGO_TARGET_DIR` propio por worktree,
  `CARGO_TARGET_X86_64_PC_WINDOWS_GNU_RUNNER=wine`, `WINEDEBUG=-all`.
- `os-guardian` es sólo Windows: `cargo check` nativo en Linux falla en
  `main` también; no es regresión.
- Una sola cadena Cargo pesada a la vez (4 CPU, 15 GB).

## 9. Protocolo con los demás agentes

- Antes de editar: `git fetch`, buzón (`COORDINACION_CODEX_2026-09-28.md`)
  y TABLERO; re-grep de cada ancla.
- Stage sólo de archivos propios (nunca `git add -A`); diff contra cada
  padre tras cada merge; `cargo check --workspace --all-targets`.
- Cambios de conducta viva entran por PR con CI y, si tocan el camino del
  fixture, con T-1 (≥ 11,0 % gen a gen) adjunto. La Línea F pide lo mismo a
  las olas Ω: **toda ola que cambie el camino vivo entra por PR**, para que
  la CI y una revisión independiente corran antes del merge.
