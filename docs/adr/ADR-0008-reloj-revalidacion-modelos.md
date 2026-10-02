# ADR-0008: Reloj de revalidación de modelos promovidos

- **Estado**: Propuesto (GLM, LXXIII, 2026-10-02)
- **Contexto**: El estándar honesto (XLIV-13/LXV) promueve un modelo con
  `selección ∧ test posterior` sobre tapes cronológicos. La familia honesta
  actual (BTC, ADA, ATOM, BNB, SOL, XRP, XLM) fue promovida con test
  posterior en tapes que terminan el 2026-09-14/17. Un edge medido en
  septiembre NO certifica octubre: el drift del mercado degrada el modelo
  silenciosamente mientras el gate B3.25 sólo verifica EXISTENCIA del
  artefacto, no vigencia.
- **Decisión propuesta**: todo modelo promovido lleva una vigencia atada al
  tape de su test posterior:
  1. `test_hasta` = fecha final del tape del test posterior + 45 días
     (un mes de gracia + el margen de descarga de tapes del exchange).
  2. Cuando exista tape posterior a `test_hasta` para el símbolo, el modelo
     DEBE re-correr la plantilla honesta (mismo split + test con el tape
     nuevo) dentro de los 7 días siguientes.
  3. Si la revalidación no se corre, o el gate bloquea (edge muerto), el
     modelo se DEMUEVE a `_CANDIDATE` (el watcher deja de servirlo; el gate
     B3.25 vuelve a vetar el símbolo — sonda-bloqueado honesto).
  4. La democión NO es destructiva: el artefacto se renombra/marca y su
     historial queda en el manifest (diff = changelog).
- **Implementación (ola futura)**: campo `test_hasta` en `ModelEntry` +
  validación en el arranque (o en `model_manifest` como diagnóstico duro).
  Mientras tanto, REGLA MANUAL documentada en este ADR.
- **Consecuencias**: la cobertura puede BAJAR sola con el tiempo (los
  modelos caducan si nadie los re-valida) — correcto: la cobertura sin
  vigencia es decoración. El coste operativo es ~1 entrenamiento por
  símbolo por mes (≈2.5 h de CPU cada uno).
- **Alternativas descartadas**: (a) sin reloj — cobertura nominal eterna,
  el drift la vuelve mentira; (b) revalidar en cada arranque — coste de
  cómputo injustificado frente al calendario; (c) tamaño de ventana móvil
  de entrenamiento — cambia la pregunta (estabilidad del edge) que ya
  responde mejor el test posterior calendario.

Relacionado: ADR-0003 (meta geométrica — modelos caducos arrastran DD),
ADR-0005 (veto registry — B3.25 verifica existencia, no vigencia).
