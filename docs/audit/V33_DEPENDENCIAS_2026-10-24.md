# V33 — Inventario de dependencias duplicadas y red

Método: lectura directa de `Cargo.lock` con `tomllib`, sin ejecutar Cargo, compilar ni cambiar versiones. Fecha: 2026-10-24.

## Resultado

Hay **40 paquetes con más de una versión**. Esto sustituye el dato histórico de 68 usado por planes anteriores y no prueba por sí solo desperdicio o posibilidad de unificación.

Red relevante:

- `hyper 0.14.32` y `hyper 1.10.1`.
- `tokio-tungstenite 0.21.0` y `0.24.0`.
- `tungstenite 0.21.0` y `0.24.0`.
- `http 0.2.12` y `1.4.2`.
- `rustls` aparece como dependencia de la capa anterior, pero no se infiere una segunda pila idéntica sin revisar features.

## Padres que explican la separación

- `telemetry-server` depende de `axum 0.7.9`, que lleva `hyper 1.10.1` y `tokio-tungstenite 0.24.0`.
- `warp 0.3.7` y `reqwest 0.11.27` usan `hyper 0.14.32`; `tokio-tungstenite 0.21.0` también aparece en `data-pipeline`, `execution-engine` y el paquete raíz.
- `reqwest 0.11.27` es consumido por `backtest-engine`, `data-pipeline`, `execution-engine`, `god-engine-core`, `quantum-arena`, `telemetry-server` y el paquete raíz.

## Decisión de ingeniería

No unificar por bump automático ni eliminar duplicados por métrica estética. Primero decidir si la convergencia Axum/warp tiene beneficio real, luego medir RAM, binario, pools/TLS, p50/p95/p99 y contratos de comportamiento. Las versiones duplicadas que sean requisitos de frameworks distintos no son prueba de redundancia evitable. `reqwest` ya tiene consumidores reales; no se declara código muerto.

ROI/PnL/WR/Sharpe/EV y latencias antes/después: `N/D`. No hubo build, benchmark ni ejecución financiera.
