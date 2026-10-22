# Plan maestro por fases — estado medido y átomos priorizados

**Fecha:** 2026-10-22 (Sol, consejo Quant Sr.)
**Base:** `main = 14806e19e`, **CI verde en ambos workflows** (unit-suites 6/6 y
replay-contracts). Este documento no propone teoría: fija qué está medido, qué
falta y en qué orden, para que cualquier agente que entre o vuelva de una
ausencia larga sepa dónde está el suelo firme.

---

## 0. Regla de admisión de todo átomo

Ningún cambio entra a `main` sin las cuatro piezas:

1. **Hipótesis** causal (qué variable, por qué, en qué instante `t` está
   disponible — nada de look-ahead).
2. **Insumo** real y medible (no un placeholder ni un mock).
3. **Costo** explícito (nanosegundos, bytes, comisiones).
4. **Beneficio OOS** medido o, si no se puede medir aún, **deuda declarada** con
   el experimento que la cerraría.

Y una prueba que **ejecute el camino real**. Un contrato que no corre en ningún
workflow no certifica nada: en esta sesión apareció uno
(`wf_numerario_invariancia_contract`, desde GLM 145) que ningún CI ejecutaba;
ya está conectado a `replay-contracts.yml`.

---

## 1. Estado medido (no estimado)

| Área | Medida | Fuente |
|---|---|---|
| `main` | `14806e19e`, unit-suites ✔ + replay ✔ | runs 39728438032 / 39728438084 |
| Ramas | 2 remotas (`main`, `codex/base-audit`), 3 locales | `git ls-remote`, `git branch` |
| Ramas archivadas | 14 tags `archivo/*` (locales: el remoto rechaza push de tags por hook) | `git tag -l "archivo/*"` |
| `.gitignore` | cubre `target/`, `data/`, `.env`, `*.log`, `*.bin` | `git check-ignore -v` |
| Crates duplicados | **68 versiones** (syn ×4, hashbrown ×4, rand ×3, tungstenite/tokio-tungstenite ×2, zstd ×2, windows* ×2-3) | `cargo tree --workspace --duplicates` |
| Capital en vivo | extraído por API con reintento fail-closed | `god_engine.rs:1425` |
| Numerario del pre-screen | 1.0 (unidad), invariancia certificada | `online_daemon.rs` + contrato en CI |
| Clientes HTTP | 1 por poller (no por llamada); 3 en `omni_multiplexer` sin tuning explícito | `git grep reqwest::Client` |

---

## 2. FASE 0 — Higiene de recursos (cerrada, con vigilancia)

**Hecho:** `target/` compartido (37 GB) es el 58 % de la carpeta (63,9 GB de
300); `data/` 23,9 GB; `.git` 3,2 GB. 7 worktrees huérfanos eliminados.

**Reglas vigentes para todos:**

- Un solo `CARGO_TARGET_DIR`. Worktree propio = **prohibido**.
- Si otro agente compila, `cargo` falla con `Access is denied` sobre
  `.cargo-build-lock`: **se espera el turno o se certifica por CI**, nunca se
  crea un `target/` privado.
- `CARGO_INCREMENTAL=0` en máquinas con poca RAM: con incremental hubo
  `STATUS_ACCESS_VIOLATION` en `libsqlite3-sys` por contención.
- Antes de fusionar: anotar en `AGENT_SYNC.md` qué archivo se toca y la huella
  de RAM estimada.

**Pendiente F0:** podar los 68 duplicados. Empezar por la capa de red
(`tokio-tungstenite` ×2) porque es la que el operador percibe como latencia.

---

## 3. FASE 1 — Ramas y CI (cerrada)

**Hecho:** 27 refs → 5. 14 ramas rescatadas/archivadas, 9 remotas borradas,
worktrees podados. Tres rojos de otros agentes diagnosticados y cerrados con
prueba local antes de fusionar (ver §7).

**Regla:** rama con el nombre del agente, merge a `main`, borrar rama local y
remota. Nunca `git add -A` en un checkout compartido.

---

## 4. FASE 2 — Hardcodes, red, OMS y punto fijo

### 4.1 Hecho en esta sesión

- **Capital ficticio erradicado**: el arranque live ya pide `fetch_account_balance()`
  por API con reintento; el numerario del pre-screen pasó de `13.0` a `1.0`
  (unidad multiplicativa, invariancia certificada) y el backtest forense
  **anuncia** el origen de su capital en vez de inyectarlo en silencio.
- **ARL del detector LG-22**: se eliminó el suelo arbitrario `.max(10.0)`; sólo
  los valores degenerados caen al default 200. La política la dicta el genoma.

### 4.2 Átomos siguientes, en orden

| # | Átomo | Hipótesis | Aceptación |
|---|---|---|---|
| 2.1 | Unificar `reqwest`/`tokio-tungstenite`/`hyper` a una sola versión | Cada versión duplicada mantiene su propio pool TLS y su propio estado de conexión: dos stacks = dos handshakes y dos colas | `cargo tree --duplicates` sin `reqwest`/`tungstenite`; latencia p50/p99 medida antes/después con el benchmark de `latency_flow_audit` |
| 2.2 | Fábrica única de cliente HTTP (`OnceLock<Client>`) con `tcp_nodelay`, pool y keep-alive explícitos | Hoy cada módulo construye el suyo con defaults distintos; el pool no se comparte entre pollers | Contrato: `Arc::ptr_eq` entre dos llamadas + petición real contra un listener local |
| 2.3 | Envío de órdenes por WebSocket API (REST sólo con pool persistente) | REST por orden añade un RTT completo; la WS API lo elimina | Contrato con servidor WS local que verifica el frame; comparación de RTT |
| 2.4 | Punto fijo + `exchangeInfo` (`tickSize`, `stepSize`, `MIN_NOTIONAL`) | `f64` redondea distinto que el exchange y produce rechazos `PRICE_FILTER`/`LOT_SIZE` | Contrato con los filtros reales descargados; cero rechazos por redondeo en 10 000 órdenes sintéticas |
| 2.5 | Kalman de reloj contra el servidor | El reloj de Windows deriva y `recvWindow` rechaza firmas (`-1021`) | Contrato: deriva simulada de ±500 ms corregida a < 5 ms |
| 2.6 | Bucle del reconciliador OMS en vivo | `reconciliation.rs` existe y está probado, pero su ciclo no está certificado contra el estado real | Contrato E2E: posición huérfana adoptada y orden fantasma cancelada |

---

## 5. FASE 3 — Evolución online, ADN y fees

- **Persistencia del ADN**: `estado_aprendido` ya guarda calibrador y evidencia
  con reloj; falta extenderlo a cromosomas y pesos tensoriales con `mmap` +
  checksum. Criterio: reinicio en < 1 ms sin perder una generación.
- **Cola FIFO y selección adversa**: el backtest debe modelar nuestra posición
  en la cola L2/L3; sin eso, los llenados son ilusorios y el edge desaparece en
  vivo.
- **Esperanza neta ergódica**: dimensionar por `g_time = E[log(1 + f·r)]` tras
  fees, no por `E[r]`. Criterio: `P(ruina) = 0` bajo la distribución empírica de
  colas.
- **Top 10 dinámico**: rotación por estadísticos completos por símbolo, no por
  una lista fija.

---

## 6. FASE 4 — Matemática de frontera (sólo con insumo causal)

Ordenado por **relación beneficio/costo** para este repositorio, no por fama:

1. **DMD / operador de Koopman en streaming** — bajo costo (SVD incremental
   sobre ventanas), detecta bifurcaciones antes de que el precio las muestre.
2. **TDA ligera (β₀/β₁ en streaming)** — complejos simpliciales sobre la nube de
   activos; detecta huecos de liquidez y ciclos de arbitraje.
3. **Wasserstein W₂ + métrica de Fisher-Rao** — ya hay anillo de Wasserstein en
   el espectro; extenderlo a geodésicas entre distribuciones de retorno.
4. **Rough paths / firmas truncadas** — sólo si el análisis de rugosidad
   (H < 0.5) muestra señal OOS; antes, medir H por escala.
5. **MPS/DMRG** — compresión de un espacio de Hilbert de activos × escalas; sólo
   si el número de activos × escalas hace inviable el tensor denso.
6. **MFG (Fokker-Planck + HJB)** — el más caro; exige calibrar la densidad de
   stops y liquidaciones. Entra como **modelo de riesgo**, no como señal.
7. **Tsallis S_q** — reemplaza el supuesto gaussiano en las colas; medir `q` por
   régimen antes de tocar el sizing.
8. **Red-team pre-trade (< 500 ns)** — el de mayor retorno esperado por unidad
   de riesgo: rechaza trampas de liquidez antes de pagar el spread.

**Regla:** cada teoría entra en **paralelo** (A/B), nunca sustituyendo a ciegas;
se queda sólo si mejora las métricas netas OOS.

---

## 7. Lecciones de esta sesión (para no repetirlas)

1. **`f64::clamp` panica si `min > max`** y con `panic = "abort"` mata el
   proceso. Un piso variable (el τ del trade) contra un techo fijo (1 día)
   revienta con horizontes largos. Usar `acotar_continuo`.
2. **`f64::min` devuelve el otro operando con NaN**: un Kelly NaN se colaba como
   el tope del genoma y la orden se admitía. Guarda de finitud fail-closed.
3. **Un solver de arranque en caliente no converge a un punto fijo**: comparar
   dos trayectorias con tolerancia de 1e-8 es comparar ruido. La tolerancia debe
   ser la del solver (4e-5), con aserciones exactas donde sí las hay (reloj,
   número de observaciones).
4. **Los fixtures inconsistentes fabrican bugs**: arena con 1000 y motor con
   2000 se lee como una caída del 50 %. El capital debe ser el mismo objeto.
5. **Un límite arbitrario pisa una política**: `.max(10.0)` sobre el ARL del
   genoma rompió el caso ARL = 1. Los límites sólo existen para valores
   degenerados, y se documentan.
6. **Un contrato que ningún workflow ejecuta no certifica**: verificar siempre
   que el test nuevo corre en CI, no sólo en la máquina propia.

---

## 8. Cómo usa este plan un agente que llega

1. Leer `AGENT_SYNC.md` (últimas 10 filas) y este documento.
2. Comprobar `gh run list --branch main --limit 2`: si hay rojo, **eso** es la
   prioridad absoluta, por encima de cualquier átomo nuevo.
3. Elegir **un** átomo de §4/§5/§6, registrarlo en `AGENT_SYNC.md` con archivos y
   huella, implementarlo, certificarlo en local y en CI, fusionar y borrar rama.
4. Dejar el estado escrito aunque la cuota se agote a mitad: el siguiente agente
   debe poder retomarlo sin adivinar.
