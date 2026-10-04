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
- **Frente L2 (DL del operador)**: ADR-0010 — GATE v1 RESUELTO PARCIAL
  (adenda 3): dirección transfiere OOS (+5.3 pts sobre la fija en
  sep-14, +5.0 en ago; la fija fue anti-predictiva en sep) pero la
  calibración falla (sobrefiada). Fase 3: calibración Platt/temperatura
  sobre selección + re-gate; cableado bloqueado hasta gate completo.
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
