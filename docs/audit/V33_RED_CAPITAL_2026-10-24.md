# V33 — RED ejecutado: capital inválido trunca `equity_curve.csv`

## Alcance y aislamiento

- Ejecutable: `target/debug/continuous_evolution_backtest.exe` (mtime 2026-10-24 14:13:15 -05:00).
- Fuente: `crates/backtest-engine/src/bin/continuous_evolution_backtest.rs` (mtime 2026-10-24 12:02:58 -05:00), sin diff contra HEAD.
- Directorio aislado: `.agent_tx/kimi-quant-sr-20261024/red-capital-current/`.
- No hubo API, credenciales, tapes, ticks reales, operaciones ni dinero. No se compiló ni se modificó código.
- Centinela inicial: `SENTINEL-DO-NOT-TRUNCATE` (25 bytes).

## Ejecución

Desde el directorio aislado:

```bash
INITIAL_CAPITAL=abc SIM_DAYS=1 TG_GENOME_ENV=backtest \
  target/debug/continuous_evolution_backtest.exe
```

Salida capturada en `logs/v33_red_capital_invalid.log`.

## Resultado observado

- `PROCESS_EXIT=0`.
- `equity_curve.csv` fue reemplazado por `Day,Capital,DailyPnL,PnLPercent,Trades` (39 bytes), destruyendo el centinela.
- No se creó `config_dir`.
- El log solo informa `Archivo binario no encontrado: "data/BTCUSDT_ticks.bin"`; no rechaza el capital inválido.

## Veredicto RED

La entrada inválida no se rechaza antes de los efectos laterales: el proceso continúa, trunca el CSV y finaliza con éxito. Esto confirma comportamiento defectuoso, no solamente una sospecha estática. La fórmula de crecimiento con literal 13 USD sigue siendo un hallazgo estático separado; este experimento no ejecutó una corrida completa ni midió ROI.

La corrección debe validar capital como primera operación de `main`. El mismo escenario, tras compilar la corrección cuando el semáforo lo admita, debe terminar con código distinto de cero, conservar intactos el centinela y `config_dir`, y mostrar un diagnóstico explícito de `INITIAL_CAPITAL`.

ROI/PnL/WR antes y después de fees, Sharpe y EV/trade: `N/D`. Periodo/datos de mercado: `N/D`. La meta +100 %/72 h no se evalúa en este experimento.
