# 🏦 TRADER GEMINI v5 — Sistema de Trading Algorítmico en Rust

**Stack**: Rust puro (workspace de 23 crates) · Binance Futures (testnet + mainnet)
**Estado**: FASE 0 del Plan Maestro completada. **NO operar en mainnet hasta Fase 7.4.**

> 📘 Este README describe el sistema REAL (Rust). El histórico del proyecto
> (era Python) está archivado en `../TraderGemini_archivo/`.

## Qué es

Sistema de trading cuantitativo para Binance USDT-M Futures con dos motores
coordinados (scalping + swing), evolución genética de parámetros (genoma),
inferencia ML en el hot-path (forests + MLP) y telemetría de baja latencia.

**Principio de diseño (Plan Maestro)**: paridad total backtest ↔ demo(testnet)
↔ producción; decisiones de riesgo derivadas de matemática (Kelly fraccional
bayesiano con cota de ruina), no de constantes; fees como función objetivo
(expectancy NETO); demo-certificación obligatoria antes de mainnet.

## Arquitectura (resumen)

```
data-pipeline (WS/REST) ──► feature-engine ──► god-engine-core ──► risk-engine
        │                        │                (scalp+swing+ML)       │
        ▼                        ▼                     ▼                 ▼
  storage-engine (lakehouse)  models/           quantum-arena      execution-engine
                                                     ▲                 │
                                              evolution-engine ──► Binance API
                                              (genoma hot-swap)   (orders+fills)
```

Mapa completo: [`docs/INVENTORY.md`](docs/INVENTORY.md) (censo por crate,
binarios, código muerto, linaje de archivos runtime, mapa de hardcode).

## Quickstart (desarrollo)

```bash
cargo check --workspace --all-targets   # debe estar GREEN
cargo test --workspace --lib            # 17/17 tests
```

## Quickstart (demo/testnet — ÚNICO modo permitido por ahora)

```bash
cp template.env .env    # rellenar BINANCE_TESTNET_API_KEY / BINANCE_TESTNET_SECRET_KEY
./LAUNCHER.bat          # elegir modo DEMO
```

⚠️ `--force-live` está deshabilitado por directiva del Plan Maestro hasta que
la certificación demo (Fase 7) se complete con gates estadísticos + gateo humano.

## Métricas de latencia

Las cifras de la era anterior ("2.30 μs tick-to-order") se consideran **sin
evidencia** hasta re-medirlas con el pipeline de telemetría de la Fase 6
(histogramas p50/p99/p999 por etapa, artefactos reproducibles).

## Estructura del repo

- `crates/` — 23 crates del workspace (ver INVENTORY §1)
- `src/bin/` — 20 binarios (god_engine es el principal)
- `config_dir/` — genomas activos (hot-cargados)
- `models/` — forests + MLP entrenados (hot-reload por mtime)
- `data/` — histórico parquet/ticks (no trackeado en git)
- `docs/` — documentación viva (INVENTORY.md es la fuente de verdad)

## Seguridad

- Credenciales SOLO por env (`template.env` de plantilla; `.env` gitignored).
- Nunca se loguean URLs firmadas ni bodies de cuenta (fix F0.2).
- `STOP_TRADING.LOCK` en raíz: apagar el sistema (la lectura por el motor
  se implementa en F5.2).
