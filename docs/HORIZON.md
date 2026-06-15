# 📊 MÓDULO HORIZON — Documentación del Sistema de Horizontes Temporales

> **REGLA FUNDAMENTAL**: Ningún fix, parámetro, configuración o línea de código
> relacionada con trading puede existir sin una etiqueta explícita del horizonte.
> `APLICA A: TODOS` es una etiqueta válida. Lo que no tiene etiqueta no aplica a nada.

## Horizontes Definidos

| ID | Código | TF Análisis | TF Entrada | Hold Objetivo | Hold Max | Trades/Día |
|----|--------|-------------|------------|---------------|----------|------------|
| H1 | `MICROSCALPING` | 1m, 3m | 1m | 30s–5min | 10min | 10–50/activo |
| H2 | `SCALPING` | 5m, 15m, 1h | 5m | 5min–4h | 8h | 2–15/activo |
| H3 | `SWING` | 4h, 1d | 4h | 4h–7d | 14d | 1–5/semana |

---

## Tablas TP/SL Per-Asset × Per-Horizonte

### MICROSCALPING (ATR ref: 1m)
| Activo | TP Default | SL Default |
|--------|-----------|-----------|
| BTC/USDT | 0.14% | 0.09% |
| ETH/USDT | 0.17% | 0.11% |
| BNB/USDT | 0.21% | 0.13% |
| SOL/USDT | 0.27% | 0.16% |
| XRP/USDT | 0.22% | 0.13% |
| DOGE/USDT | 0.38% | 0.23% |

### SCALPING (ATR ref: 15m)
| Activo | TP Default | SL Default |
|--------|-----------|-----------|
| BTC/USDT | 0.55% | 0.32% |
| ETH/USDT | 0.68% | 0.40% |
| BNB/USDT | 0.75% | 0.45% |
| SOL/USDT | 0.90% | 0.55% |
| XRP/USDT | 0.75% | 0.45% |
| DOGE/USDT | 1.20% | 0.70% |

### SWING (ATR ref: 4h)
| Activo | TP Default | SL Default |
|--------|-----------|-----------|
| BTC/USDT | 2.50% | 1.30% |
| ETH/USDT | 3.00% | 1.60% |
| BNB/USDT | 3.50% | 1.80% |
| SOL/USDT | 4.50% | 2.20% |
| XRP/USDT | 3.50% | 1.80% |
| DOGE/USDT | 6.00% | 3.00% |

---

## Capital Allocation 3-Way

| Régimen | MICRO | SCALP | SWING |
|---------|-------|-------|-------|
| NEUTRAL | 25% | 45% | 30% |
| TRENDING | 20% | 40% | 40% |
| RANGING | 35% | 50% | 15% |
| HIGH_VOL | 15% | 50% | 35% |

### Bounds (inviolables)
- MICRO: min 10%, max 40%
- SCALP: min 25%, max 60%
- SWING: min 10%, max 50%
- Suma siempre = 100%

---

## Consensus Gates Per-Horizonte

| Horizonte | Fee Drag Gate | Descripción |
|-----------|---------------|-------------|
| MICRO | 1.5x round-trip | Loose — más señales, SL ajustado protege |
| SCALP | 2.0x round-trip | Balance calidad/frecuencia |
| SWING | 2.8x round-trip | Strict — pocas señales de alta calidad |

---

## APE Floor Logic Per-Horizonte

| Horizonte | TP Floor | SL Range (vs Config) |
|-----------|----------|---------------------|
| MICRO | max(APE, Config) | APE puede ampliar hasta 1.5x Config, piso 80% |
| SCALP | max(APE, Config) | APE ±30% de Config |
| SWING | max(APE, Config) | APE ±20% de Config (Config domina) |

---

## APE Bounds Per-Horizonte

| Horizonte | SL Min | SL Max | TP Min | TP Max | ATR Mult |
|-----------|--------|--------|--------|--------|----------|
| MICRO | 0.05% | 0.30% | 0.08% | 0.60% | 0.40 |
| SCALP | 0.15% | 1.50% | 0.20% | 3.00% | 1.00 |
| SWING | 0.80% | 5.00% | 1.50% | 10.0% | 1.50 |

---

## Archivos Modificados (Módulo Horizon v1.0)

| Archivo | Cambio | Fecha |
|---------|--------|-------|
| `config.py` L506-531 | Capital Allocation 3-way + régimen | 2024-06-11 |
| `config.py` L535-579 | MICRO per-asset TP/SL tables | 2024-06-11 |
| `config.py` L585-631 | SCALP per-asset TP/SL tables | 2024-06-11 |
| `config.py` L633-679 | SWING per-asset TP/SL tables | 2024-06-11 |
| `consensus_filter.py` L127-160 | Horizon-aware fee drag gate | 2024-06-11 |
| `risk_manager.py` L350-413 | Per-asset tables + horizon params | 2024-06-11 |
| `risk_manager.py` L417-480 | Horizon-differentiated APE floor | 2024-06-11 |
| `asset_parameter_engine.py` L39-66 | MICRO fields in AssetProfile | 2024-06-11 |
| `asset_parameter_engine.py` L96-133 | MICRO bounds (SL/TP/ATR) | 2024-06-11 |
| `asset_parameter_engine.py` L166-183 | `_get_bounds()` helper | 2024-06-11 |
| `asset_parameter_engine.py` L202-248 | get_tp/get_sl use `_get_bounds` | 2024-06-11 |

---

## Template de Fix Obligatorio

```yaml
fix_id: FIX-YYYYMMDD-NNN
ETIQUETA_HORIZONTE:
  aplica_a: [MICRO | SCALP | SWING | TODOS]
  razon: [por qué solo estos horizontes]
DESCRIPCION:
  archivo: [nombre]
  que_hacia_antes: [comportamiento anterior]
  que_hace_ahora: [comportamiento nuevo]
VALORES_CAMBIADOS:
  parametro: [nombre]
  activo: [BTC | ETH | TODOS]
  valor_anterior: [valor]
  valor_nuevo: [valor]
IMPACTO:
  impacto_en_MICRO: [efecto o N/A]
  impacto_en_SCALP: [efecto o N/A]
  impacto_en_SWING: [efecto o N/A]
```
