# ADR-0007 — Corridas largas de verificación en worktree aislado

- **Estado**: Aceptado
- **Fecha**: 2026-09-30
- **Responsables**: GLM (incidente LIV, f7fd1c57); contexto: marco del
  operador (proceso multiagente disciplinado)

## Contexto

Durante una verificación integral del workspace (~40 min de compilación +
tests), el agente Qoder editó `crates/god-engine-core/src/lib.rs` sin
commit (QO-586) EN MEDIO de la corrida. La compilación pudo haber capturado
fuente a medio escribir: el resultado habría sido no confiable sin ningún
síntoma visible. La corrida fue eliminada y relanzada en un worktree
aislado fijado al commit verificado.

Con tres-cuatro agentes trabajando en paralelo sobre un checkout
compartido, toda corrida larga (tests integrales, oráculos de 48 min,
trainers de 3 h) es vulnerable a esta carrera silenciosa.

## Decisión

1. **Las verificaciones integrales y las corridas de duración > 10 min
   se ejecutan en un WORKTREE AISLADO** fijado al commit que se pretende
   verificar (`git worktree add ../<repo>-verify <sha>`), nunca sobre el
   checkout compartido con árbol potencialmente sucio.
2. **Quien edite el checkout compartido lo anuncia en el buzón** con
   marca temporal (regla de cortesía ya practicada por Codex con sus
   worktrees propios; ahora formalizada para todos).
3. Los binarios YA COMPILADOS (trainers en ejecución) son inmunes a
   ediciones posteriores de la fuente — no requieren reaislamiento.
4. El worktree de verificación se elimina tras consumir su resultado
   (`git worktree remove`).

## Consecuencias

- Los resultados de verificación vuelven a ser afirmables: el commit
  verificado es el commit exacto, sin ambigüedad de árbol sucio.
- Coste: duplicación de target/ por worktree (compilación desde cero la
  primera vez) — aceptable para verificaciones que sólo ocurren tras
  merges mayores.
- Los symlinks de data/, models/, config_dir/ al checkout principal
  evitan duplicar 24 GB de tapes.
