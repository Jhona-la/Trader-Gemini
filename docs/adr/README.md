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
| [0008](ADR-0008-reloj-revalidacion-modelos.md) | Reloj de revalidación de modelos promovidos | Propuesto | 2026-10-02 |
| [0009](ADR-0009-regeneracion-mensual-copulas.md) | Regeneración mensual del manifest de cópulas | Aceptado | 2026-10-02 |
| [0010](ADR-0010-arquitectura-dl-modular.md) | Arquitectura DL-modular espectral | Propuesto | 2026-10-03 |
| [0011](ADR-0011-identidad-canonica-del-simbolo.md) | Un solo nombre por símbolo y slots estables en caliente | Aceptado | 2026-10-03 |
| [0012](ADR-0012-genoma-fijo-en-evaluacion.md) | Sólo el núcleo de ejecución sigue al almacén de genomas | Aceptado | 2026-10-03 |
| [0013](ADR-0013-autoridad-del-riesgo-en-el-envio.md) | El host nunca envía más apalancamiento que el validado | Aceptado | 2026-10-03 |
| [0014](ADR-0014-doctrina-continuo-espectral.md) | Doctrina del continuo espectral: los seis principios del motor | Aceptado | 2026-10-04 |
| [0015](ADR-0015-semantica-del-kill-switch.md) | El kill-switch bloquea lo que aumenta el riesgo, nunca las salidas | Aceptado | 2026-10-10 |

## Convención

- Un ADR por decisión arquitectónica irreversible o doctrinal.
- Estados: Propuesto → Aceptado → (Reemplazado por ADR-NNNN | Obsoleto).
- La actualización de un ADR aceptado NUNCA reescribe la decisión: se
  escribe un ADR nuevo que lo reemplaza, con enlace bidireccional.
- Responsable = agente/ola que la ejecutó; el bucle de revisión cruzada
  se registra en el propio ADR.
