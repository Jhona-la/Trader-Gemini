# 🔰 TRADER GEMINI: MANUAL DEL OPERADOR (PULSO-NOUVEAUCRAFT)

> *"El código ejecuta, pero la mente del operador gobierna. El conocimiento the la 'Psicología Algorítmica' es la primera y última línea de defensa contra la ruina del capital."* — Protocolo NouveauCraft.

Este manual es la guía definitiva para interpretar, gestionar y supervisar al ecosistema **Trader Gemini** en vivo. Describe detalladamente cómo leer los dashboards the tiempo real, vigilar a la inteligencia artificial, y cuándo jalar el gatillo the contingencia.

---

## 🖥️ I. LA CONSOLA DE MANDO (GRAFANA: TIEMPO REAL)

El dashboard de Grafana es el "Cardiógrafo" del Metal-Core. Se nutre de `Prometheus` recopilando la métrica cada 500ms.

### 1. Lectura del 'Order Flow Delta' (Momentum vs Absorción)

El gráfico de "Limit Order Book (LOB) Imbalance" es el indicador adelantado de la máquina.

- **Escenario de Momentum (Verde Oscuro y Precio Subiendo):** El Bid-Delta es agresivamente positivo, hay compras masivas a mercado (Takers), y el precio sigue la fuerza. *Acción: El bot entra en `LONG` apoyado por el Machine Learning.*
- **Escenario de ABSORCIÓN (Verde Dinámico pero Precio Estancado o Bajando):** ⚠️ **ALERTA ROJA ICEBERG**. Se ven inyecciones gigantes the Bids (Verde Profundo), pero el precio de la moneda colisiona con una pared invisible y no sube. Instituciones están vendiendo pasivamente (Limit Asks) absorbiendo las compras.
  - *Interpretación NouveauCraft:* El mercado es engañoso (Spoofing). La liquidez que *parece* que va a subir el precio, en realidad es carnada. El Módulo de Riesgo entrará en `Cooldown`. **No entres manualmente ni fuerces compras.**

### 2. Monitoreo de la Trinidad Omega (Consenso Múltiple)

El Panel *"Swarm Intelligence Consensus"* muestra la alineación the los modelos de Machine Learning sobre una moneda.

- **Lectura BFT (Byzantine Fault Tolerance):** Verás las 20 barras de los activos Elite. Si para `BTC/USDT` Random Forest vota "Buy", pero XGBoost vota "Neutro" y Gradient Boosting "Short", el Consenso final es **NEUTRAL (Bloqueado)**.
- *Meta:* La paciencia paga. Trader Gemini es un francotirador. Solo cuando los 3 ejes probabilísticos colisionan hacia el mismo vector, la herramienta The Riesgo arma la orden the $13.00.

### 3. Métricas the Hardware Ryzen 7 5700U (Latencia vs Throttling)

Monitorea la gráfica "Metal-Core Health":

- **Latencia End-To-End (Verde):** Debe permanecer por debajo de `30ms`. Picos esporádicos a `100ms` son tolerables (Binance lag). **Si se mantiene > 100ms:** Los servidores de intercambio están congestionados, *el Slippage devorará el capital*; pausa el trading usando Kill-Switch Nivel 1.
- **CPU Throttling (Rojo):** Los 4 núcleos *Core Pinned* (0, 2, 4, 6) no deben superar el `85%` sostenido o perderán caché L3. Si superan los `85°C`, el OS limitará la frecuencia de 4.3GHz a 1.8GHz, arruinando los WebSockets. Verifica el enfriamiento del hardware local.

---

## 🧠 II. EL MICROSCOPIO EVOLUTIVO (WANDB: IA Y GENÉTICA)

A través the Weights & Biases observamos dentro the las "sinapsis" de la Trinidad Omega.

### 4. Interpretación de Curvas de Aprendizaje (PPO/RL)

La pestaña "Policy Loss / Reward" determina si el bot de auto-mejora se está adaptando, o está "perdiendo la cabeza":

- **Curva Sólida (Convergencia):** La función de *Reward Promedio* sube progresivamente en escalada, y el *Loss* decae suavemente y se estabiliza. El bot está aprendiendo thel mercado actual.
- **Caída Libre (Alucinación Estadística):** El *Reward* oscila violentamente o cae en picado después de una racha ganadora. El mercado ha cambiado drásticamente (Cambio the Régimen HMM) y el modelo está sobreajustado (Overfitted) a datos the ayer. *El bot requiere reentrenamiento urgente o se quemará protegiendo una tesis muerta.*

### 5. Seguimiento the Genes (El Darwinismo de Parámetros)

El Algoritmo Genético optimiza parámetros base durante los cierres o "Domingos". Verás un Dashboard con "Mapeo de Hiperparámetros":

- **Generación Dominante:** Presta atención a los puntos más oscuros del gráfico *Parallel Coordinates*. Si el algoritmo empieza a seleccionar mutaciones donde `Max_Leverage` o `Risk_Per_Trade` son excesivamente altos, está apostando al azar para sobrevivir. Tú debes dictaminar qué genes pasan a producción y vetar comportamientos de Ludopatía de la Máquina.

### 6. Análisis SHAP de Importancia (SHAP Values)

El gráfico the barras SHAP responde por qué la máquina toma decisiones *hoy*.

- **Ejemplo Táctico:** Habitualmente el SHAP más alto es el *Order Flow Delta*. Pero si the repente el *Z-Score the Volatilidad* desplaza al resto, la IA te está diciendo: *"La estructura técnica the soportes (RSI/MACD) se rompió, solo estoy transando para evitar la hecatombe the precios."*

---

## 🩺 III. EL ESPEJO DE SOPHIA (METACOGNICIÓN Y SALUD MENTAL)

Sophia es el alter-ego del bot. Observa *cómo* opera el bot y lo regula.

### 7. El Brier Score (La Honestidad de la IA)

El Brier Score (0.0 a 1.0) mide qué tan arrogante o modesta es la predicción del Machine Learning. Define el error cuadrático entre la predicción y el fallo de realidad.

- **Score < 0.25 (Honesto/Calibrado):** Cuando la IA dice "Soy 80% confidente the ganar", tiene la razón 8 de cada 10 veces.
- **Score > 0.40 (Arrogancia Fatal):** El bot está dando predicciones súper sesgadas the 99% the confianza, y el trade rebota para el lado contrario y choca StopLoss. **Alerta**: Baja inmediatamente la agresividad de las órdenes.

### 8. Análisis de Error Temporal (Volatilidad Súbita)

El *Time-To-Resolve (TTR)* del trade.

- **Fenómeno:** Si Trader Gemini planea capturar Scalps de 1-minuto (*TTR proyectado: 5 min*) pero el capital se queda enfrascado por 50 minutos flotando en números rojos (*Stuck-In-Trade*)...
- **Acción:** La volatilidad local se ha secado o el régimen de *mean-reverting* ha fallado. Re-calibra en `config.py` los multiplicadores de GARCH obligando a buscar objetivos de TakeProfit más cortos (Ej. pasar de `0.3%` a `0.15%`).

### 9. Dashboard de Sophia-View (Mapas de Calor Flotilla)

Muestra los 20 activos en un mapa the 2-ejes (Retorno vs Riesgo-HMM).

- **Zonas "Verde Bosque" (Líderes de Flota):** Monedas con tendencia predecible. Deja al bot absorberlas.
- **Zonas "Rojo Magma" (Enfermas/Cuarentena):** Monedas con colapsos estadísticos. Si una moneda entra a Magma, el bot le restará su "Lealtad (`loyalty_score`)". Déjalo que trabaje: en varios ticks, el MarketScanner expulsará la moneda tóxica y pescará un nuevo activo the la lista de las 100 top the Binance automáticamente.

---

## 🚨 IV. PROTOCOLOS DE INTERVENCIÓN (SUPERVIVENCIA)

La supervivencia del saldo Mínimo Base ($13.00) está por encima the cualquier ambición de escalada a los $1000.

### 10. Uso thel Kill-Switch the Emergencia

El Botón de **"Pánico Nuclear"** en el Streamlit Dashboard.

- **Cuando usarlo:** Al detectar "Fat Fingers" Institucionales, Flash Crashes, Tweets regulatorios disruptivos, o si Uvloop lanza más the 5 excepciones seguidas en los WebSockets (Desfase the memoria).
- **Cómo usarlo:** Pulsa "EMERGENCY LIQUIDATE".
- **Qué ocurre bajo el capó:**
  1. Se bloquea el archivo `STOP_TRADING.LOCK`.
  2. Todas las órdenes `LIMIT` in-flight son canceladas en Binance (Cero exposición the contraparte).
  3. Todo el cripto residual se vacía a `USDT_MARKET`.
  4. Los archivos `Parquet` y Log SQLite se guardan de forma atómica. No hay corrupción the Datos. Para reiniciar, borras manualmente el archivo `.LOCK` de la carpeta raíz.

### 11. Mantenimiento del Ledger Pre-Cisión Axioma ($13.00 USDT)

En el dashboard, revisa diariamente la gráfica **`Omega Precision Drift`**.

- El bot cuenta en `Decimal` a 28 digitos de precisión la resta the cobros the fee `(Binance_Start - Fees_Pagados + PnL_Teórico)`.
- **Descuadre:** Si notas que tu balance the Binance difiere del "Axioma Balance" por más the 1 céntimo (>$0.01), Binance te está aplicando Slippage invisible o "Ghost Fees". Apaga el bot y revisa tus cálculos de comisión `Maker/Taker` the los 30 días de tu cuenta nivel-VIP the Binance.

### 12. La Rutina del Domingo (La Evolución Consciente)

La Máquina aprende sóla, pero requiere permisos Semanales. Cierra todas las conexiones el Domingo en la mañana.

1. **Auditoría de Equity:** Revisa la curva The rendimiento general (`walk_forward_metrics`).
2. **Revisión Genética:** Inicia el script de entrenamiento Genético `python run_genetic_algorithm.py` para buscar mejores Hyperparámetros en los datos cerrados the la semana anterior.
3. **El Escudo del Arquitecto:** Revisa el Output in Weights and Biases. Si los nuevos hiperparámetros sugieren arriesgar menos thel 3% thel wallet, apruébalos. Evita aprobar pesos donde la máquina insinúe sobre-apalancarse.
4. **Data Purge:** Limpia la carpeta `data/cache_parquet` the archivos the hace una semana para obligar al Market Scanner a redescubrir "sangre fresca" en las Altcoins top para el Lunes. Empezar the cero the forma adaptativa.

---

## 🚨 V. ANEXO DE TÁCTICA: GLOSARIO DE ALERTAS (SIGNALS & NOISE)

El Metal-Core interactuará contigo a través de códigos visuales the Logger y Dashboard. Aprende a descifrarlos:

### Codificación del 'Logger CLI'

- 🟢 **[GREEN] `[OK] / [SUCCESS]`**: Flujo Asincrónico operando perfectamente (Sub-20ms).
- 🟡 **[YELLOW] `⚠️ [WARNING]`**: Desfase Menor. El sistema entrará en degradación elegante (Cooldown the activo, o recálculo the fees). Ignorable pero Auditable.
- 🔴 **[RED] `❌ [ERROR]`**: Falla the API (Key Inválida, Orden Rechazada por Margin insuficiente). La posición *no* se tomó, el capital está a salvo. Requiere intervención manual the Permisos.
- 🟣 **[MAGENTA] `[CRITICAL] / [KILL-SWITCH]`**: EL EVENTO FATAL. Drawdown Inaceptable o Falla del Motor Cuántico. El Bot se ha autodestruido para proteger el Ledger the U$D 13.00. (El Archivo `.LOCK` existe).

### Codificación de 'Sophia-View' (Grafana/Streamlit)

- 🟦 **Burbujas Azules (Z-Score Bajo):** El mercado está muerto (Fin de Semana). El bot no operará porque no hay volatilidad The la cual sacar provecho.
- 🟧 **Burbujas Naranjas (Z-Score Alto):** Explosión the volatilidad. Si se combina con un *Hurst Exponent* alto, el bot preparará la "Phalanx" y lanzará ráfagas M1. Preparaos para la contabilidad PnL de alta velocidad.

> **Mensaje NouveauCraft:** *El código ejecuta a la velocidad the la luz, pero la mente del operador lo gobierna todo. La arquitectura está diseñada para sobrevivir, tu conocimiento es la primera línea the defensa de tu capital.*
