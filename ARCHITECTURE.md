# 🏛️ SISTEMA AUTÓNOMO DE FUTUROS BINANCE — TRADER GEMINI
**DOCUMENTO MAESTRO DE ARQUITECTURA Y REGLAS INAMOVIBLES**

**Capital Inicial:** ~$13 USD  
**Objetivo Maestro:** +100% neto cada 3 días (Crecimiento Exponencial Compuesto)  
**Modalidades Simultáneas:** Microscalping, Scalping, Swing  
**Dirección:** Long & Short (Hedging Controlado)

---

## 1. AXIOMAS FUNDACIONALES

### 1.1 PRINCIPIO DE RESPONSABILIDAD ÚNICA
Cada estrategia y cada feature cumple una sola función atómica y bien definida. No existe lógica compartida ni responsabilidades mezcladas. Si una pieza ejecuta más de una cosa, se divide hasta que cada fragmento sea indivisible. Este principio rige el diseño de todo el sistema sin excepción.

### 1.2 RENTABILIDAD INDIVIDUAL DEMOSTRABLE
Cada estrategia y feature debe ser rentable de forma autónoma y estadísticamente demostrable, sin depender del desempeño de ninguna otra pieza. Toda pieza nueva pasa por validación aislada antes de integrarse. Una feature que no demuestra rentabilidad propia no entra. Se corrige, se rediseña o se elimina permanentemente.

### 1.3 OBJETIVO MAESTRO — CRECIMIENTO EXPONENCIAL COMPUESTO
El sistema opera bajo una única métrica de éxito: **+100% de retorno neto sobre el capital total cada ciclo de 3 días**, de forma sostenida y compuesta. 
- Ciclo 1: $13 → $26
- Ciclo 2: $26 → $52
- Ciclo 10: ~$13,312
Toda decisión de diseño o configuración se evalúa exclusivamente contra esta curva exponencial. El capital de cada ciclo completado se convierte en la nueva base del siguiente. El compounding es automático, total e inmediato.

### 1.4 RESTRICCIÓN CLAVE — CRITERIO SUPREMO
> **¿Esta pieza es rentable de forma autónoma, no colisiona con ninguna otra, está correctamente registrada, opera en el régimen/sesión correctos, respeta los límites de riesgo, gestiona su dirección con evidencia y contribuye al +100% cada 3 días?**
*Si la respuesta a cualquier parte de esa pregunta es no, la pieza no se integra.*

---

## 2. ARQUITECTURA DE GESTIÓN — 5 CAPAS OBLIGATORIAS

### Capa 1 — Gestión Individual de Estrategia
Cada estrategia tiene su propio sistema de configuración, control de riesgo, métricas y estado **completamente aislados**. Ninguna estrategia puede leer ni modificar el estado de otra directamente.

### Capa 2 — Registro Omnisciente y Sistema de No-Colisión (`omniscient_registry.py`)
Núcleo de integridad absoluta. Bloquea conflictos antes de la ejecución. Posee dos categorías:
- **Valores fijos:** Límites inamovibles de identidad y seguridad. (Prioridad Absoluta).
- **Valores adaptativos:** Parámetros ajustables dinámicamente dentro de rangos fijos.

### Capa 3 — Consciencia Grupal y Orquestación
Inteligencia de portafolio en tiempo real. Detecta sinergias, elimina redundancias, redistribuye capital y asegura que el conjunto avance hacia el +100%. No reemplaza la Capa 1, la coordina.

### Capa 4 — Motor de Dirección (Long / Short Intelligence)
El sistema nunca asume una dirección por defecto. Cada operación justifica su dirección con evidencia cuantitativa. Long y Short tienen lógicas optimizadas separadas. Puede operar hedging controlado.

### Capa 5 — Motor de Compounding y Escala
Al completar cada ciclo de 3 días con +100%, recalcula tamaños, apalancamiento y riesgos. Gestiona la fase de micro-capital (bajo $50) priorizando supervivencia y compounding sobre diversificación.

---

## 3. GESTIÓN DE FUTUROS PERPETUOS (BINANCE)

1. **Apalancamiento Inteligente:** Variable dinámica basada en régimen, señal, capital y distancia al stop. En fase micro, compensa el tamaño sin exceder los límites seguros.
2. **Liquidación (Línea Roja Absoluta):** Monitorea precio de liquidación virtual y mantiene un buffer. Si se acerca, reduce tamaño, añade margen o cierra posición. Margen aislado obligatorio en modo micro-capital.
3. **Funding Rate:** Descuenta costo acumulado del profit esperado (si dura > 4h). Puede rechazar trades.
4. **Fees y Costos:** Opera maker sobre taker preferiblemente. Calcula breakeven exacto (incluyendo fees) en microscalping antes de entrar.
5. **Liquidez y Selección de Pares:** Monitorea order books. Rota pares dinámicamente priorizando volumen, spread, y volatilidad. 

---

## 4. GESTIÓN DE RIESGO ESTRUCTURAL

- **Riesgo por operación:** Porcentaje fijo (más conservador en micro-capital).
- **Riesgo por sesión:** Drawdown máximo diario (congela el bot si se toca).
- **Riesgo por estrategia:** Drawdown acumulado por módulo (desactiva la estrategia específica).
- **Riesgo sistémico:** Drawdown crítico del portafolio (protocolo de emergencia).
- **Stop loss obligatorio:** Ninguna operación nace sin Stop Loss definido en Binance.
- **Ratio R:R:** Mínimos obligatorios no intercambiables por modalidad.
- **Trailing Stop Dinámico:** Para proteger ganancias al cruzar umbrales.

---

## 5. REGLAS COMPLEMENTARIAS

### 5.1 Detección de Régimen
El sistema reconoce: *tendencial alcista/bajista, lateral comprimido/volátil, ruptura, reversión, volatilidad extrema*. En "régimen incierto", reduce tamaño y prohíbe alta volatilidad. 

### 5.2 Calidad de Señal
Score de 0 a 100 basado en confluencia, timeframes, régimen y liquidez. Aprende estadísticamente: penaliza puntajes en regímenes donde históricamente fallan.

### 5.3 Gestión de Tiempo y Sesiones
- **Microscalping:** Overlaps (Londres-NY, Tokio-Londres).
- **Scalping:** Intradía, post-apertura de 30 minutos.
- **Swing:** Ignora ruido intradía, alta liquidez para evitar slippage.
- *Calendario de restricciones:* Evita noticias de alto impacto.

### 5.4 Fatiga y Correlación
- **Correlación:** Bloquea posiciones nuevas correlacionadas (ej. BTC y altcoins direccionales simultáneas) para evitar riesgo invisible.
- **Sobreoperación:** Límite máximo de trades, cooldowns entre señales.

### 5.5 Auditoría y Evolución
- Todo es loggeado, rastreable y reconstruible.
- Ciclos de validación estrictos: Backtest -> Paper Trading -> Promoción.
- Si una estrategia sufre degradación (Overfitting), se retira automáticamente.

### 5.6 FASE MICRO-CAPITAL ($13 a $100)
Modo supervivencia-compounding:
- Estrategias de mayor winrate y menor drawdown.
- Posiciones simultáneas limitadas para controlar impacto de fees.
- Prioridad absoluta: Llegar a $100 para desbloquear diversificación total.
