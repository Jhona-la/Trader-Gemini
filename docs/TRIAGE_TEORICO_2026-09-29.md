# TRIAGE TEÓRICO DEL ARSENAL EXPANDIDO (2026-09-29)

**Autor**: GLM (Ola XLVIII·B, rama glm/xlviii-b-triage-firmas).
**Contexto**: el operador entregó un catálogo teórico amplio (cálculo
estocástico, inferencia, matemática pura, física, computación cuántica,
teoría de decisión, transporte óptimo) con una regla que ya era doctrina
del repo: **más teoría ≠ más edge; cada teoría pasa validación estricta
antes de producción**. Este documento es el triage: qué existe, qué está
parcial, qué es candidato con contrato de transferencia, y qué se registra
como especulativo SIN implementar.

Estados: **[EXISTE]** en producción con tests · **[PARCIAL]** existe un
trozo con huecos identificados · **[CANDIDATO]** transferencia viable con
contrato abajo · **[ESPECULATIVO]** sin vía de transferencia honesta hoy.

## Cálculo estocástico y procesos

| Teoría | Estado | Nota |
|---|---|---|
| Hawkes multivariado (contagio, autocausalidad) | [EXISTE] | kernel cross + matriz N×N + roles + Hodge (XLV, XLVI·C) |
| Cambio de régimen bayesiano | [EXISTE] | BOCPD sobre W₁ espectral |
| Volatilidad rough (fBm, rough Bergomi) | [PARCIAL] | Hurst por DFA+reloj existe (CL-26); falta el MODELO rough de σ(τ) — candidato: ley H·√ con H<0.1 verificado contra DFA |
| Grandes desviaciones (Cramér) | [ESPECULATIVO] | estudiado en XLVI·D: sin dependencia medida, Markov veta todo (documentado); viable SOLO con ρ̄ medida — reabrir si hace falta criterio de probabilidad |
| BSDE / HJB (control óptimo) | [CANDIDATO] | la capa de decisión pedida por el operador; vía honesta: HJB en forma de programación dinámica sobre el espectro discretizado τ — grande, requiere ciclo propio |
| Malliavin (griegas sin diferenciar payoff) | [ESPECULATIVO] | no operamos derivados exóticos; sin consumidor |
| Procesos de Cox/marcados | [PARCIAL] | el feed YA es proceso puntual (ticks); intensidad estocástica = Hawkes con intensidad latente — candidato dentro de la ola MLE adaptativo pendiente |

## Estadística e inferencia

| Teoría | Estado | Nota |
|---|---|---|
| Fisher (espectral, gate walk-forward) | [EXISTE] | C3/XLI + CL-28 la volvió telemetría por calibración de nulo — lección registrada |
| Conformal/ACI (incertidumbre calibrada) | [EXISTE] | por activo (PR#15 Codex) |
| DSR / LCB Jeffreys | [EXISTE] | en el embudo de promoción |
| EVT (GEV/GPD, colas) | [PARCIAL] | CVaR95 empírico XLVIII·A; falta ajuste GPD con cola pesada por PIKE — candidato pequeño |
| Cópulas (vine, dinámicas, colas) | [CANDIDATO] | hueco real multiactivo: la ρ̄ media pierde asimetría de colas; contrato: cópula t bivariada por par same-bet, θ estimado, validar calibración de cola contra empírico |
| Transferencia de entropía | [CANDIDATo] | directo sobre los streams de ticks (mismo casing que Hawkes cross); mide dirección de información SIN ventana de lag — complementa α_cross |
| Inferencia causal (Pearl) | [PARCIAL] | outcome attribution por posición (PR#16 Codex) ES diseño causal; falta formalización do-calculus — especulativo formalizar, la práctica ya aterrizada |
| Teoría de información (MDL) | [PARCIAL] | logloss gates del trainer son MDL implícito; falta criterio MDL de selección de genomas — candidato |
| Procesos gaussianos / RKHS | [ESPECULATIVO] | sin consumidor identificado hoy; la incertidumbre la lleva conformal |
| Dirichlet / regímenes sin número fijo | [ESPECULATIVO] | BOCPD ya descubre cambios sin fijar K; añadir capas jerárquicas = decoración sin demanda |
| Geometría de Amari (Fisher-Rao) | [ESPECULATIVO] | Fisher puntual existe; la variedad/gradiente natural requiere un loop de entrenamiento que optimice EN la variedad — sólo si el DL modular llega a producción |

## Matemática pura

| Teoría | Estado | Nota |
|---|---|---|
| RMT (Marchenko-Pastur, Jacobi, N_eff) | [EXISTE] | XLI + PR#11 Codex; Tracy-Widom [CANDIDATO pequeño]: umbral de significancia del autovalor máximo con corrección de cola finita |
| **Firmas de camino (Lyons)** | **[CANDIDATO → IMPLEMENTADO XLVIII·B]** | ver abajo — la primera implementación de este tramo |
| Multiactivo: descomposición de Hodge | [EXISTE] | XLVI·C (gradiente vs curl/cámara de eco) |
| Categorías/funtores (composición de motores) | [ESPECULATIVO] | lenguaje bonito para ADRs de arquitectura; sin consumidor ejecutable |
| TDA (homología persistente) | [CANDIDATO lejano] | robustez al ruido atractiva; costo alto; sólo si el feature store llega y hay señal de que la topología aporta |
| Koopman/DMD | [CANDIDATO] | linealización en observables: DMD sobre la ventana espectral 32D por símbolo — candidato real de ciclo futuro |
| Tropical/no-arquimediana, sheaves, nudos | [ESPECULATIVO] | sin vía honesta identificada |
| Ergódica/KAM/Lyapunov | [PARCIAL] | la persistencia por bloques (CL-30) ES estadística ergódica; Lyapunov empírico = candidato pequeño (divergencia de caminos gemelos del espectro) |

## Física

| Teoría | Estado | Nota |
|---|---|---|
| Grupo de renormalización (escalado entre τ) | [CANDIDATO] | EL ajuste natural a la doctrina espectral: flujo de acoplamientos entre escalas d g/d ln τ medible desde la amplitud ζ(p)/H(τ) — candidato de ciclo propio |
| Ising/transiciones de fase | [PARCIAL] | la entropía de masa espectral + Fisher son observables de fase; sin modelo de espín — especulativo el modelo, medidas ya vivas |
| Integrales de camino / AdS / cuerdas | [ESPECULATIVO] | registradas por honestidad: sin transferencia |

## Computación cuántica

| Teoría | Estado | Nota |
|---|---|---|
| Redes tensoriales (MPS compresión) | [CANDIDATO lejano] | compresión del historial espectral; sólo cuando el feature store exista |
| Grover/HHL/VQE/QAOA | [ESPECULATIVO] | sin hardware ni consumidor |

## Decisión y juegos

| Teoría | Estado | Nota |
|---|---|---|
| Kelly fraccional + ruina | [EXISTE] | con las tres entradas medidas (XLVI·D/E) |
| Mean field games / McKean-Vlasov | [ESPECULATIVO] | sin población de agentes modelada |
| Subastas/diseño de mecanismos | [PARCIAL] | la física de impacto (depth slippage predictor) ya cobra la subasta implícita del libro |
| Filtros no lineales (partículas) | [PARCIAL] | Kalman existe; UNSCENTED kalman = candidato pequeño si la fusión espectral lo pide |

## Transporte óptimo

| Teoría | Estado | Nota |
|---|---|---|
| Wasserstein-1 (drift espectral) | [EXISTE] | con BOCPD arriba |
| Sinkhorn / puentes de Schrödinger | [CANDIDATO] | interpolación entrópica para rebalanceo suave de capital — cuando la capa de control llegue |
| Procesos de Lévy/saltos | [PARCIAL] | la física de fills cobra saltos empíricamente; sin proceso paramétrico |
| Percolación/avalanchas | [ESPECULATIVO] | Hawkes ya mide el clustering que importa |

---

## IMPLEMENTACIÓN DE ESTE TRAMO: FIRMAS DE CAMINO (Lyons) — XLVIII·B

**Por qué esta y no otra**: es la pieza del catálogo con mejor relación
(profundidad matemática real) × (coste de implementación honesto) × (encaje
con la doctrina): representa una trayectoria como features universales,
invariante a reparametrización temporal — exactamente lo que un motor
ESPECTRAL necesita para comparar forma de camino entre escalas sin que la
frecuencia de muestreo contamine.

**Contrato de transferencia** (protocolo del repo):
- **Variable**: camino 2D X(t) = (log-P(t), t/T normalizado) sobre la
  ventana de una escala τ; lead-lag N dimensional queda para el DL modular.
- **Operador**: firma truncada a nivel 2 (7 términos: 1, X¹, X², X¹X¹,
  X¹X², X²X¹, X²X²) por la identidad de Chen del incremento.
- **Unidades**: adimensionales (log-precio acumulado × tiempo normalizado).
- **Contorno**: <3 puntos ⇒ None; ventana degenerada (rango log-P = 0)
  ⇒ firma con términos de nivel 1 en 0 (camino constante en esa coordenada).
- **Identificabilidad**: la firma a nivel 2 caracteriza el camino salvo
  "tree-like equivalence" (teorema de Hambly-Lyons) — documentado, no
  afirmado como unicidad.
- **Coste**: O(n) por ventana; sin hot path (feature por escala).
- **Falsación**: (a) camino recto ⇒ firma cerrada conocida; (b) identidad
  de Chen: S(A⊗B) = S(A)⊗S(B) por concatenación; (c) invariancia a
  reparametrización temporal estrictamente creciente; (d) camino
  invertido ⇒ nivel 1 cambia de signo, nivel 2 igual (identidad de
  firma del camino reverso S(m)=ε(m,·)S(·)... a nivel 2: invertir el
  camino da los mismos términos diagonales y cruces intercambiados con
  signo — el test verifica la relación exacta).

Implementación: `feature_engine::path_signatures`.

## ADENDA ST — precisión del contrato y estado de integración (Codex, 2026-09-29)

Se conserva el triage anterior como histórico. Para las afirmaciones siguientes,
este corrigendo precisa el contrato; no implica integración operativa ni alpha.
Detalle y pruebas: [Auditoría ST](AUDITORIA_FIRMAS_CONTRATOS_TEORICOS_2026-09-29.md).

1. Firmas: el kernel de nivel2 es correcto para caminos lineales por partes,
   con media diagonal. La fórmula incluye esa diagonal; el reverso transpone
   nivel2. Dos puntos válidos bastan. Con precio constante el tiempo no es nulo.
2. Hambly-Lyons concierne a la firma completa, NO asegura identificabilidad
   de nivel2. Hay un contraejemplo reproducible con reloj estrictamente creciente.
3. Invariancia geométrica no significa invariancia ante cualquier remuestreo
   del feed. Deben declararse interpolación, resolución y pérdidas de excursiones.
4. El adaptador original normalizaba por epoch; ST corrige localmente a
   dt/(tN-t0), valida orden y mejora precisión. Commit f60e1820, aún sin publicar.
   No se encontró consumidor de producción: estado EXPERIMENTAL, no genoma cableado.
5. Rough volatility estudia log-volatilidad; Hurst de precio no la calibra.
   No corresponde imponer H<0,1 como supuesto universal.
6. Transfer entropy necesita historias/retardos y condicionamiento. La frase
   previa «SIN ventana de lag» no es un contrato correcto del estimador.
7. Outcome attribution conserva lineage, no identifica por sí sola una
   intervención causal. La equivalencia con Pearl queda retirada.
8. Logloss no demuestra un criterio MDL sin código de complejidad/modelo o
   construcción universal declarada. No se afirma ausencia de otros controles.
9. Conformal/ACI implementado no cierra las limitaciones de cobertura de CF:
   selección, delay, recorte y singleton siguen pendientes.
10. Firmas normalizadas pierden duración absoluta; seis coordenadas no son
    seis fuentes independientes. Antes de consumirlas: activo/τ/as-of/versión,
    paridad replay-serving, rango efectivo, coste y ablación OOS.

No se altera código operativo de rough σ, TE, MDL, causalidad o ACI en ST.
Cada implementación posterior debe volver a verificarse contra su propia
versión; esta adenda no prejuzga modificaciones concurrentes de GLM/Claude.
