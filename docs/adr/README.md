# Índice de ADRs

| ADR | Título | Estado | Fecha |
|---|---|---|---|
| [0001](ADR-0001-latencia-lognormal-fills.md) | Latencia de fills: RTT lognormal determinista | Aceptado | 2026-09-29 |
| [0002](ADR-0002-veto-tres-entradas-medidas.md) | Veto estructural: tres entradas medidas | Aceptado | 2026-09-29 |
| [0003](ADR-0003-doctrina-meta-geometrica.md) | Doctrina de la meta: crecimiento geométrico medido | Aceptado | 2026-09-29 |
| [0004](ADR-0004-veto-registry-como-codigo.md) | Registro de vetos: el censo es código | Aceptado | 2026-09-29 |
| [0005](ADR-0005-triage-teorico-y-medicion.md) | Teoría nueva: triage 4 estados + medición como portón | Aceptado | 2026-09-29 |
| [0006](ADR-0006-bt-vivo-envoltorio.md) | bt↔vivo: el sesgo vive en el envoltorio | Aceptado | 2026-09-29 |
| [0007](ADR-0007-worktree-aislado-verificaciones.md) | Corridas largas en worktree aislado | Aceptado | 2026-09-30 |

## Convención

- Un ADR por decisión arquitectónica irreversible o doctrinal.
- Estados: Propuesto → Aceptado → (Reemplazado por ADR-NNNN | Obsoleto).
- La actualización de un ADR aceptado NUNCA reescribe la decisión: se
  escribe un ADR nuevo que lo reemplaza, con enlace bidireccional.
- Responsable = agente/ola que la ejecutó; el bucle de revisión cruzada
  se registra en el propio ADR.
