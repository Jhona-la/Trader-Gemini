# DECISIÓN PENDIENTE DEL DUEÑO — Modelos FDUSD inertes

> Preparado por GLM (LXXXVIII, 2026-10-04). **La decisión es del dueño
> del sistema** — este documento sólo la deja lista con la evidencia.

## El estado (verificado)

- El roster vivo es **100% USDT** (bootloader.rs:301-328, 26 símbolos).
- Los 10 modelos `{SYM}FDUSD_MOTOR` se cargan al arranque pero **jamás
  se consultan**: la clave `{SYM}FDUSD_MOTOR` no se genera en
  inferencia. Ya están marcados `inerte_en_roster` en el manifest
  (LXXIII) — peso muerto VISIBLE, ~2.2 MB en memoria + tiempo de carga.
- Los 10 son EL MISMO archivo (hash idéntico 0b8bef51c26f, era
  candidate-only pre-plantilla honesta: 50 árboles, base 0.2006).

## Opciones

### Opción A — Remoción física
Eliminar los 10 archivos de `models/`. El manifest se regenera sin
ellos. Riesgo: si algún día el roster agrega pares FDUSD, habría que
re-entrenar (los actuales no valen: nunca pasaron plantilla honesta).
Beneficio: cero peso muerto, arranque más limpio.

### Opción B — Archivo con marcado permanente (recomendada)
Moverlos a `models/_archivo/` (fuera del escaneo del watcher) + nota
en el manifest de por qué se archivan. Conserva la trazabilidad del
artefacto histórico sin cargarlo en producción.

## Recomendación de GLM
**Opción B**: el costo de cargarlos es trivial (~2 MB), pero el costo
de CONFIAR en un modelo que jamás pasó el estándar honesto no lo es —
archivar deja eso imposible por construcción y conserva el linaje.
Una palabra del dueño («B» o «A») y se ejecuta en un commit.
