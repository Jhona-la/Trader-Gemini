# RUNBOOK — Corrida Demo A3 (primer dato del sistema)

> PLAN MAESTRO DE EVIDENCIA, Fase A, paso A3: encender el bin en Binance
> Testnet 24 h con medición completa. Este runbook es la receta exacta —
> cualquier agente del consejo puede ejecutarla; no requiere decisiones
> sobre la marcha.

## 0. Precondiciones (verificar ANTES de encender)

| # | Chequeo | Cómo |
|---|---------|------|
| 1 | main verde en su punta de código | `gh run list --branch main --limit 4` — ambos workflows success |
| 2 | Claves Testnet presentes | `TESTNET_API_KEY`/`TESTNET_API_KEY_SECRET` en `.env` (GE:1532-1567 las exige) |
| 3 | Genoma demo disponible | `genomes/demo/active.json` existe (prod exige el suyo fail-closed, GE:1649-1688) |
| 4 | `.env` con `USE_TESTNET=1` | Es el discriminador de venue (GE:1586): Testnet vivo, NO papel |
| 5 | Sin `MAINNET_ARMED` en config_dir | Para que `is_demo_mode` no arme producción (GE:1191-1219) |
| 6 | RAM del operador ≥ 4 GiB libres | El bin + telemetría corren en la máquina del operador; si < 4, pedir cierre de apps antes |

## 1. Encendido

```bash
# En el checkout raíz (main), con las claves del .env:
TG_IGNICION_SIN_DARWIN=1 cargo run --release --bin god-engine 2>&1 | tee logs/corrida_a3_$(date +%Y%m%d_%H%M).log
```

**Por qué la puerta**: el training real aún es NO-OP (BLOQUEANTE #1
residual en BOOT:292); el contrato (204) rechazará Darwin con
«training_no_op». `TG_IGNICION_SIN_DARWIN=1` habilita PERO LOGUEA el
rechazo que pisa — es la puerta auditable del consejo, no un default.

## 2. Qué verificar en los primeros 10 minutos del log

| Min | Mensaje esperado (aprox) | Si NO aparece |
|-----|--------------------------|---------------|
| ~0 | `⚠️ [IGNICIÓN] Darwin RECHAZADO (training_no_op) — habilitado SOLO por TG_IGNICION_SIN_DARWIN=1` | El wiring del 204 no está activo — ABORTAR |
| ~0-2 | Balance de Testnet obtenido (retry loop GE:1598-1612 pasa a la primera) | Sin saldo API: revisar claves — ABORTAR |
| ~2-5 | Warmup klines por símbolo (venue Testnet) | Ver red/endpoint |
| ~5-8 | `🚨 [RECONCILIATION]` o snapshot de posiciones OK; `oms_reconciled` TRUE vía apply_snapshot | Sin snapshot ⇒ trading bloqueado (correcto: fail-closed) |
| ~8-10 | Primeros ticks del loop vivo; `on_tick` avanza fase | — |

## 3. Qué medir (la obligación del mandato §28)

La telemetría del profiler de Sol ya publica `TOTAL_NET_PnL` y PnL
neto/gross por tick. Al final de las 24 h, extraer del log/telemetría:

1. **PnL bruto y neto (antes y después de fees)** — con períodos exactos
   (ts de arranque y de corte).
2. **WR antes/después de fees** (trades que eran ganadores brutos y
   quedaron perdedores netos).
3. **EV/trade neto** y número total de trades.
4. **Drawdown máximo** sobre el capital inicial real de la wallet Demo.
5. **Latencia**: conteo de `WARNING: LATENCIA DE RED ELEVADA` en 24 h
   (la línea base del operador son 200+).
6. **Desviación vs. backtest**: MISMO período, mismos símbolos, el
   backtest del tape correspondiente si existe; si no existe, registrarlo
   como pendiente (la paridad bt-con-API es un ítem aparte del plan).

## 4. Criterio de la corrida (pre-definido, del plan maestro §1)

- Ésta es una corrida de 24 h (A3), NO el veredicto de 7 días (A5).
- Éxito de A3 = el informe CON NÚMEROS existe y es publicable, verde o
  rojo. Un EV neto negativo es un hallazgo: dice si el agujero es fees,
  latencia o señal.
- ABORTAR inmediatamente si: drawdown > 50% del capital Demo, o errores
  de reconciliación en loop, o el bin entra en pánico/restart repetido.

## 5. Al terminar

1. Publicar el informe en `docs/audit/CORRIDA_A3_RESULTADO_<fecha>.md`
   con TODAS las métricas del §3 y los períodos exactos.
2. Entrada de buzón con el resumen y el número principal.
3. Los datos alimentan los BLOQUEANTES #2 (EV como puerta) y #3
   (breaker) — ahí ya no será diseño, será calibración con números.

— GLM, Fase A del PLAN MAESTRO DE EVIDENCIA, 2026-11-06.
