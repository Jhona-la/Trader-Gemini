# MAPA MAESTRO API BINANCE — USDT-M Futures (F1.0)

> Fuente de verdad para "entendimiento total de la API" (directriz del proyecto).
> Base: `https://fapi.binance.com` · Testnet: `https://testnet.binancefuture.com`
> WS market: `wss://fstream.binance.com` · WS user: `wss://fstream.binance.com/ws/<listenKey>`
> Documentación oficial: binance-docs.github.io/apidocs/futures (siempre verificar contra exchangeInfo en runtime — los límites cambian por cuenta/VIP).

## 1. Endpoints EN USO (mapeados al código)

| Endpoint | Método | Peso | Uso en el sistema | Código |
|---|---|---|---|---|
| `/fapi/v1/order` | POST | 0 | Crear orden (MARKET/LIMIT/GTX/STOP_MARKET/TP_MARKET/TRAILING) | executor.rs |
| `/fapi/v1/order` | GET | 1 | Query por origClientOrderId (idempotencia F1.2) | executor.rs query_order |
| `/fapi/v1/order` | DELETE | 1 | Cancelar orden | executor.rs cancel_order |
| `/fapi/v1/batchOrders` | POST | 5 | OCO simulado (2 POST), ver §4 | executor.rs (a migrar) |
| `/fapi/v1/listenKey` | POST/PUT | 1 | User-data stream (fills en tiempo real); fuera de la cuota del ejecutor (censo C-F7b, N20) | client.rs + user_data_stream.rs |
| `/fapi/v2/balance` | GET | 5 | Balance wallet (paridad de capital) | api_client.rs |
| `/fapi/v2/positionRisk` | GET | 5 | Reconciliación de posiciones | executor.rs fetch_position_risk |
| `/fapi/v2/account` | GET | 5 | Estado completo de cuenta | fetch_account_balance |
| `/fapi/v1/commissionRate` | GET | 1 | Fees reales de LA cuenta (paridad fees) | fetch_commission_rate |
| `/fapi/v1/exchangeInfo` | GET | 1 | tickSize/stepSize/filtros (cacheado exchange_info.json) | fetch_exchange_info |
| `/fapi/v1/leverage` | POST | 1 | Leverage por símbolo | set_leverage |
| `/fapi/v1/marginType` | POST | 1 | Cross/Isolated | binance_api.rs |
| `/fapi/v1/time` | GET | 1 | Sync de reloj (NTP offset) | fetch_server_time |
| `/fapi/v1/klines` | GET | 1-10 | Warmup swing (1h) | data-pipeline |
| `/fapi/v1/aggTrades` | GET | 20 | Histórico real para backtest (F2/F3) | historical.rs |
| `/fapi/v1/algoOrder` | POST | 1 (SUPUESTO) | Piernas TP/SL y trailing (servicio Algo, OCO-F5) | executor.rs place_algo_leg, execute_oco_order |
| `/fapi/v1/algoOrder` | DELETE | 1 (SUPUESTO) | Cancelar una pierna algo | cancel_algo_order_ids |
| `/fapi/v1/algoOpenOrders` | DELETE | 1 (SUPUESTO) | Purga algo de un símbolo | cancel_all_algo_open_orders |
| `/fapi/v1/openAlgoOrders` | GET | 1 con símbolo (SUPUESTO, como openOrders) | Protección viva de una posición; sin símbolo, huérfanas | fetch_open_algo_orders_inner |
| `/fapi/v1/allOpenOrders` | DELETE | 1 (SUPUESTO) | Purga de órdenes normales de un símbolo | cancel_all_symbol_orders, aplanado |
| `/fapi/v1/positionSide/dual` | GET/POST | 1 (ver nota) | Modo hedge (arranque, OmsSession::refresh cada 60 s, aplanado) | ensure_hedge_mode, fetch_hedge_mode |

> N5 (CL-179): sólo los pesos de las peticiones críticas son costes del
> ejecutor (`CosteBinance` en binance_api.rs): order POST/GET/DELETE,
> positionRisk, algoOrder POST/DELETE, algoOpenOrders y allOpenOrders DELETE,
> y openAlgoOrders con símbolo. Lo marcado SUPUESTO no está documentado y toma
> el valor del análogo más caro, porque sobrestimar sólo frena antes. Las
> lecturas (commissionRate, account, income, openAlgoOrders sin símbolo, el
> GET de modo del aplanado) no reservan coste: frenan con el uso por encima
> del 80 % del límite. ensure_hedge_mode y fetch_hedge_mode
> (positionSide/dual), exchangeInfo, time y listenKey quedan fuera de la cuota
> (censo C-F7b, fila N20). Tres pesos del mapa se recuerdan distintos en la
> documentación de Binance y están por comprobar desde el PC (fila N22):
> commissionRate 1 frente a 20, positionSide/dual 1 frente a 30, income 20
> frente a 30. Ninguno de los tres es un coste del ejecutor, así que
> corregirlos aquí no cambia su cuota; sí cambia cuánto gasta el motor.

## 2. Endpoints DISPONIBLES NO explotados (oportunidades F1.10+/F4)

| Endpoint | Peso | Valor estratégico |
|---|---|---|
| `/fapi/v1/openOrders` | 1×símbolo / 40 todos | Cancel-masivo de seguridad; arranque limpio |
| `/fapi/v1/allOrders` | 5 | Historial completo de órdenes (auditoría fills) |
| `/fapi/v1/userTrades` | 5 | Historial de fills CON comisión exacta → Fee Slayer métricas reales |
| `/fapi/v1/income` | 20 | PnL, comisiones, funding pagado/cobrado — la VERDAD de "cuánto pagamos" (directiva de informes pre/post fees) |
| `/fapi/v1/fundingRate` + `/fapi/v1/premiumIndex` | 1 | Funding actual y previsto → feature de decisión + costo de carry |
| `/fapi/v1/leverageBracket` | 1 | Caps de leverage/notional por símbolo (input de Kelly bayesiano F4.4) |
| `/fapi/v1/openInterest` + `/futures/data/openInterestHist` | 1/1 | Sentimiento de posicionamiento (feature) |
| `/futures/data/topLongShortAccountRatio` etc. | 1 | Flujos retail vs whales (dark alpha) |
| `/fapi/v1/depth` | 2-20 | Libro completo (ya consumido por WS) |
| `/fapi/v1/forceOrders` | 1 | Cancel-ALL con un solo request — kill-switch real (F5.2) |
| `/fapi/v1/multiAssetsMargin` | 1 | Multi-asset margin (optimización capital) |
| `/fapi/v1/positionSide/dual` | 1 | Modo hedge — en uso desde F1.11, ver §1 |

## 3. Parámetros avanzados de orden (los que hoy IGNORAMOS)

| Parámetro | Aplica a | Qué hace | Estado |
|---|---|---|---|
| `newClientOrderId` | Todas | Idempotencia + trazabilidad | ✅ F1.2 (UUIDv7) |
| `reduceOnly` | Cierres | La orden SOLO cierra — imposible invertir posición accidentalmente | ⚠️ solo OCO; faltan cierres parciales |
| `workingType` | STOP/TP | MARK_PRICE vs CONTRACT_PRICE para disparo | ❌ sin usar (default CONTRACT_PRICE) |
| `priceProtect` | STOP/TP | No disparar si precio devía >1% de mark (protección wick-hunt) | ❌ sin usar — RECOMENDADO |
| `closePosition` | STOP_MARKET/TP_MARKET | Cierra TODA la posición al disparar (sin qty) | ❌ ideal para SL de emergencia |
| `stopPrice` | STOP* | Precio de disparo | ✅ OCO |
| `activationPrice`+`callbackRate` | TRAILING_STOP_MARKET | Trailing nativo del exchange (cero latencia nuestra) | ✅ execute_exchange_trailing_stop |
| `priceMatch` | LIMIT | Anclar al best bid/ask/opposite (maker perfecto sin race) | ❌ sin usar — CLAVE para Fee Slayer |
| `selfTradePreventionMode` | Todas | Evita self-match (netting scalp↔swing, directiva "no pisarnos") | ❌ sin usar — RECOMENDADO F4.9 |
| `timeInForce=GTX` | LIMIT | Post-only garantizado (maker o nada) | ✅ F1.4 |
| `goodTillDate` | GTD | Expira solo (ordens de paciencia) | ❌ sin usar |

## 4. Gaps de implementación detectados (a cerrar)

1. **OCO es 2 POST separados** → `/fapi/v1/order/cancelReplace` o el par TP/SL con `closePosition=true` reduce el riesgo de pierna huérfana (F1.9a ya mitiga con retry+cancel).
2. **`workingType=MARK_PRICE` + `priceProtect=true`** en SL: los stops deben disparar por mark price (las manipulaciones de last-price no nos cazan). Cambio de 2 parámetros, efecto grande.
3. **`/fapi/v1/income` no se consulta**: sin él, "ROI pre/post fees" es una estimación. Con él, es un hecho (F3.5/F6.3).
4. **Tick/step size hardcodeados 0.001/0.0001** en varias llamadas (maker_chase, OCO) → deben venir de exchangeInfo cacheado (la infra existe: fetch_exchange_info).
5. **`forceOrders` (cancel-all)**: pieza del kill-switch real (F5.2) — aplanar todo con 1 request.

## 5. Rate limits (por defecto; verificar con exchangeInfo de la cuenta)

- Peso IP: 2400/min (headers `X-MBX-USED-WEIGHT-1M`). Órdenes: 300/10s, 1200/min (`X-MBX-ORDER-COUNT-*`).
- 429 = freno (respetar `Retry-After`) → política F1.8: cooldown temporal.
- 418 = IP baneada 2min–3días → freno temporal `RATE_BRAKE` por su ventana (`Retry-After` o 300 s),
  M4-C02; no es un kill-switch y se rearma si el ban persiste.
- WS: 10 mensajes/s por conexión; conexiones: 300/5min por IP.
- **Presupuesto del sistema** (F1.8, M4-C02, N5): cooldown post-429 (`Retry-After` o 60 s) y freno
  temporal (`RATE_BRAKE`) al tercer 429 consecutivo (cooldown + 60 s) o con un 418 (su ventana o
  300 s); ya no es un kill a 3×429. Mientras duran no sale ninguna petición del ejecutor que pase por
  `check_backoff`, salidas incluidas; quedan fuera las de la fila N20 del censo C-F7b (modo hedge,
  que `OmsSession::refresh` pide cada 60 s, filtros, hora, NTP y listenKey). Fuera de los frenos, dos
  tramos de cuota (CL-179): entradas y lecturas frenan con el uso por encima del 80 % del límite; las
  críticas (cierres, protección, positionRisk, cancelaciones, consulta de una orden) pasan mientras
  uso + críticas en vuelo + coste ≤ límite en cada contador en que gastan (CL-181). El uso es el
  máximo de las cabeceras vigentes de cada contador, cada una con su sello (CL-177, CL-180), y una
  crítica registra su cabecera antes de salir del libro en vuelo (CL-182). Una respuesta de error no
  trae cabecera (fila N21).

## 6. WebSocket (streams)

- Market: `<symbol>@aggTrade`, `@bookTicker`, `@depth20@100ms`, `@markPrice@1s`, `@kline_1m`.
- User-data (F1.6): `ORDER_TRADE_UPDATE` (fills), `ACCOUNT_UPDATE` (balance/posiciones), `MARGIN_CALL`, `listenKeyExpired` (regenerar clave — handled por reconexión del streamer).
