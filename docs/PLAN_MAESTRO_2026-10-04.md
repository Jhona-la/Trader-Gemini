# PLAN MAESTRO — Sincronización del consejo hacia la meta

> Creado por GLM (LXXXIV, 2026-10-04) a pedido del operador: *"Sincronízate
> con los demás agentes a un documento de planeación para lograr nuestras
> metas de crecimiento exponencial e interés compuesto de 100% cada 3 días."*
> **Documento VIVO de todo el consejo**: cada agente edita SU sección (como
> TABLERO). No es asignación de trabajo — es sincronización de lo que ya
> hacemos, la ruta crítica compartida y las reglas que nos coordinan.

## 1. La meta, encuadrada sin decoración (doctrina ADR-0003)

**100% cada 3 días = crecimiento geométrico sostenido** (≈26%/día
compuesto). El encuadre honesto que el consejo ya adoptó: maximizar el
crecimiento geométrico sujeto a riesgo/DD/costos/capacidad. La meta
sirve de DIRECCIÓN; los multiplicadores medibles del camino:

1. **Cobertura con edge honesto**: hoy 7/26 símbolos del roster con
   evidencia `selección∧test`. Cada símbolo nuevo con edge ≈ +3.5% de
   cobertura; el volumen contra la brecha (hoy ~310× al estándar de
   ~10 trades/día) sube con cada sonda desbloqueada.
2. **Calidad de la agregación**: el hallazgo LXXXIII (sombra_osc 53%
   sola vs 48.7% del agregado — la modulación fija AHOGA su mejor
   componente) marca el techo disponible en el L2 sin tocar el riesgo.
3. **Sizing multivariante**: el veto ya aprende colas (λ̂ cópula t) y
   coherencia espectral (IC(τ*)); el paso siguiente (cópula dinámica
   por ventana) está medido y espera diseño.
4. **Vigencia**: ADR-0008/0009 — un edge caducado no es edge; la
   revalidación mensual es parte del plan, no burocracia.

**No negociable** (decisión del operador, arquitectura del sistema):
riesgo-duro (kill-switch, veto estructural, tope Lundberg) NUNCA entra
en el gradiente. Es código, no peso aprendido.

## 2. Estado del sistema (una página, 2026-10-04)

- **Modelos**: familia honesta de 7 (BTC/ADA/ATOM/BNB/SOL/XRP/XLM),
  mediana OOS +0.008, récord SOL +0.0203; 2 bloqueos honestos (NEAR,
  ICP) prueban que el gate discrimina. Manifest 20/17/10-inertes.
- **Certificación**: cadena oráculo (genes, trinquete 11.0%) + paridad
  bt↔vivo (conducta) al día hasta `f9fbcbd5` — el estado actual está
  certificado de punta a punta (PR#27 incluido).
- **Veto same-bet**: 3 etapas hacia 1 componiendo sobre el complemento
  ((1−ρ)=(1−base)(1−curl²)(1−λ̂)) + apriete espectral IC(τ*) vivo (Ola
  51) + cota Lundberg con unidades honestas.
- **Sustrato espectral**: 32 filtros EWMA base-4 → 13 motores resueltos
  por escala → consenso integrado al orquestador con pesos por skill
  IC-prequential; símplex continuo de régimen p∈Δ³; ejecución IOC con
  techo de slippage adaptativo.
- **Proceso**: regla de casa operativa — oráculo T-1 verde commiteado
  ANTES del push para todo cambio de conducta viva; paridad como segundo
  gate para ejecución; plantilla honesta para todo modelo.

## 3. Mapa de frentes por agente (lo que ya está en vuelo)

### GLM
- **Frente modelos**: escalar la plantilla honesta al roster (siguientes
  con tapes: DOGE/LINK/LTC/DOT/AVAX/UNI/FIL/VET/BCH cuando octubre dé
  el tercer período). Revalidaciones ADR-0008 (test_hasta ~2026-11-01).
- **Frente cópulas**: regeneración mensual del manifest (ADR-0009);
  diseño de ventana dinámica medido-que-espera.
- **Frente L2 (DL del operador)**: ADR-0010 — v1 CERRADO PARCIAL
  DEFINITIVO (adendas 3-4): dirección robusta (+5 pts OOS constante,
  fija anti-predictiva en meses hostiles) pero la calibración NO
  transfiere entre regímenes (Platt en selección arregla selección, no
  test). Cableado BLOQUEADO. Fase 4 opcional (convicción por rango o
  no-lineal) — requiere petición del consejo, no se promete.
- **Frente certificación**: oráculo+paridad como servicio del consejo.

### Qoder
- Refactor espectral por olas certificadas (#624..#653): física de
  motores corregida (solitón, choque firmado, SR canónica, oscilador),
  coaxial refutado, pesos por skill, integridad del consenso (H1-H7),
  dd-lerp, vetos dormidos despertados. En vuelo: Ola 54 (D₀ —
  distribución antes de cablear). Propuesta activa: kernel Hawkes.
- **Deuda declarada por ellos**: D₀ (#597) medición de distribución.

### Claude (cloud)
- Ciclos CL de ejecución/sizing: ciclo 8 (CL-36..42) aterrizado vía PR
  #27 (IOC/AMBIGUOUS, random_forest, guardia Lundberg); su CI verde +
  la paridad GLM lo certificó. Revisión cruzada de qo-653 documentada.

### Codex
- Serie "MW:" (model-reload-contract): tracker de recargas con scope
  honesto, 4 commits en vuelo SIN merge — review GLM: dirección
  aprobada; al merge exigirá oráculo+paridad (toca el watcher vivo).

### Antigravity
- Símplex continuo Δ³ + colchón direccional en el orquestador + IOC
  routing (P29-P32) — todo certificado por la cadena tras su aterrizaje.

## 4. Ruta crítica compartida (lo que bloquea la meta, en orden)

1. **Tapes de octubre** (externo, bloquea): revalidaciones de los 7
   modelos + regeneración de cópulas + tercer período para 9 símbolos
   más. TODO lo demás está desbloqueado.
2. **Veredicto L2 v1** (inminente): pasa ⇒ ola de cableado con
   oráculo+paridad (el multiplicador de calidad de agregación); no pasa
   ⇒ negativo documentado y el frente DL pasa a modelo no lineal.
3. **Cobertura 7/26**: cada promoción honesta mueve el volumen contra
   la brecha 310×.
4. **Decisión FDUSD** (consejo): 10 modelos inertes marcados — remoción
   o archivo.
5. **Serie MW de Codex**: merge con su oráculo — el linaje del watcher
   es prerrequisito de confianza para el hot-reload del L2.

## 5. Compromisos del próximo ciclo (cada agente append aquí)

- **GLM**: cerrar el gate L2 v1 (veredicto binario documentado);
  mantener la cadena de certificación; al llegar tapes de octubre,
  revalidaciones + cópulas en el mismo ciclo.
- *(Qoder, Claude, Codex, AGY: añadan su compromiso — una línea)*

## 5b. Mapa de sincronización de planes (LXXXV)

Respondemos al mismo mandato en paralelo y los documentos COEXISTEN con
roles distintos — este mapa evita duplicación:
- **PLAN_MAESTRO_2026-10-04.md (GLM, este)**: estado del sistema, ruta
  crítica compartida, compromisos por sección.
- **PLAN_MAESTRO_SINCRONIZACION.md (Qoder, qo-655)**: contratos de
  trabajo en tres líneas (A-entender / B-aprender / C-ejecutar) y §4 de
  decisiones del dueño.
- **Extensión G0-G8 (Codex, PR #28)**: cobertura de contratos raíz por
  ruta con contraejemplos.
Los tres se referencian mutuamente; ninguno sustituye a los demás.

## 6. El marco que nos sincroniza (ya operativo)

- **Buzón** (append-only, dedupe de copias exactas) + **TABLERO** (fila
  por agente) + este plan (sección por agente).
- **Reglas**: oráculo T-1 pre-push sin excepciones para conducta viva;
  merge inspecciona su primer padre; paridad para cambios de ejecución;
  plantilla honesta para todo modelo; riesgo-duro fuera del gradiente.
- **Meta de proceso**: que NINGUNA ola llegue a main sin certificación —
  la auditoría de LXXIX/LXXXIV demostró que la cadena cierra lagunas.

---
*Este documento se actualiza por ciclo. Fuente de verdad del estado:
buzón + TABLERO + ADRs 0001-0013. La meta es la dirección; los gates
son el camino.*

---

## ESTADO AL CIERRE DE LA SESIÓN GLM (LXV–XCVII, 2026-10-06)

> Consolidado para el operador y el consejo: qué quedó construido, qué
> quedó verificado, y qué desbloquea qué. Cualquier agente que retome
> empieza aquí.

### La familia honesta de modelos (el activo principal)

7 símbolos con evidencia `selección ∧ test posterior` (la plantilla
honesta completa): **BTC (+0.0168), ADA (+0.0081), ATOM (+0.0114),
BNB (+0.0066), SOL (+0.0203 — récord), XRP (+0.0080), XLM (+0.0053)**.
2 bloqueos honestos (NEAR, ICP — sin edge medible). Cobertura: 7/26 del
roster. La tubería de promoción fue barrida y saneada (LXXXIX:
atomicidad tmp+rename+sync del gate, margen fail-closed).

### El barrido total (directriz del operador: F0-F8)

**Completado**: 338 archivos src + 964 tests, **217 hallazgos**. La
escalera pedida (metas → conceptos → matemática → física → código) se
cumplió: F0 metas (Qoder), F1 matemática, F2 física, F3 núcleo, F4
dinero (Ω), F5 tubería (GLM), F6 datos, F7 observabilidad, F8 tests.
Los ~15 HIGH: **todos drenados, resueltos o diseñados**. El inventario
vivo es `docs/BARRIDO_EXHAUSTIVO_FASES.md`.

### La cadena de certificación (las reglas de la casa)

1. **Muralla CI**: 431 tests de 5 crates + auto-EOF + blindaje
   whitespace (XC–XCII) — ioc_fill fue el rojo invisible que la motivó.
2. **Oráculo T-1 pre-push** sin excepciones para conducta viva
   (trinquete 11.0%).
3. **Paridad bt↔vivo** para cambios de ejecución.
4. **Plantilla honesta** para todo modelo (split cronológico + test
   posterior + purga de frontera — la frontera es CERRADA, verificada).

### Investigación cerrada con evidencia

- **Saga L2 v1** (ADR-0010, adendas 1-6): dirección robusta (+5 pts
  OOS constante) pero ni probabilidad, ni rango, ni condicionado por
  régimen transfieren el quiebre de septiembre. Señal concentrada en
  muestras no-range (+8.7 pts). La información direccional existe;
  capturarla como probabilidad estable es el problema abierto.
- **Cópulas t** (LXXI–LXXV): medidas, consumidas por el veto, y la
  deriva mensual documentada (mediana 0.106 — ADR-0009).
- **Vigilancia**: el watchdog sigue a la generación activa (LXXXXI) —
  cosecha, manual y futuro cubiertos.

### La cola restante — QUÉ DESBLOQUEA QUÉ

| frente | condición que lo desbloquea |
|---|---|
| Revalidaciones 7 modelos + regenerar cópulas + 9 símbolos nuevos | **Tapes de octubre** (el exchange no los ha publicado) |
| FDUSD (10 modelos inertes) | **Palabra del dueño** (A remoción / B archivo — DECISION_FDUSD_BORRADOR.md) |
| Cablear veto drift (la señal ya se mide: drift_real_vs_control_pct) | **Distribución medida en vivo** (sesiones de trading) |
| DSR cosecha completo | **API de retornos por trade** (diseño en BARRIDO §DISEÑO-DSR) |
| ~79 rojos perpetuos | Triaje por ciclo (mapa de deuda viva ya catalogado) |

### El proceso multiagente (lo que hizo posible todo esto)

Rama por agente → merge → push → eliminar. Buzón append-only (con
dedupe de copias exactas). TABLERO con fila por agente. Planes
maestros reconciliados (GLM estado/ruta, Qoder contratos, Codex
cobertura, Antigravity Ω). Pre-aprobaciones de diffs en vuelo. El
patrón F de barrido por zonas. La regla de mismo-commit para vetos.
