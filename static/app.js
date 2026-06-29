// God Engine L3 Terminal - Vanilla JS (O(1) updates)

const DOM = {
    wsStatus: document.getElementById('ws-status'),
    wsLed: document.getElementById('ws-led'),
    capital: document.getElementById('capital-val'),
    scalpPnl: document.getElementById('scalp-pnl'),
    swingPnl: document.getElementById('swing-pnl'),
    alphaBar: document.getElementById('alpha-bar'),
    alphaVal: document.getElementById('alpha-val'),
    alphaAlert: document.getElementById('alpha-alert'),
    latency: document.getElementById('latency-val'),
    latencyPanic: document.getElementById('latency-panic'),
    tensorGrid: document.getElementById('tensor-grid'),
    logContainer: document.getElementById('log-container')
};

// Canvas for latency radar
const canvas = document.getElementById('latency-canvas');
const ctx = canvas.getContext('2d');
const latencyHistory = new Array(400).fill(0);
let maxLatency = 100000; // 100 microseconds max scale baseline

// Initialize Tensors
const tensors = [];
for (let i = 0; i < 10; i++) {
    const container = document.createElement('div');
    container.className = 'tensor-bar-container';
    const fill = document.createElement('div');
    fill.className = 'tensor-bar-fill';
    container.appendChild(fill);
    DOM.tensorGrid.appendChild(container);
    tensors.push(fill);
}

// Format numbers
function formatMoney(val) {
    return val.toFixed(6);
}

function updateColor(element, val) {
    if (val > 0) {
        element.className = 'val green';
    } else if (val < 0) {
        element.className = 'val red';
    } else {
        element.className = 'val';
        element.style.color = '#fff';
    }
}

function addLog(msg, type = "info") {
    const line = document.createElement('div');
    line.className = 'log-line';
    const now = new Date();
    const ts = `[${now.getHours().toString().padStart(2, '0')}:${now.getMinutes().toString().padStart(2, '0')}:${now.getSeconds().toString().padStart(2, '0')}]`;
    line.innerHTML = `<span class="timestamp">${ts}</span> <span class="${type}">${msg}</span>`;
    DOM.logContainer.appendChild(line);
    if (DOM.logContainer.children.length > 50) {
        DOM.logContainer.removeChild(DOM.logContainer.firstChild);
    }
    DOM.logContainer.scrollTop = DOM.logContainer.scrollHeight;
}

// Latency Radar Drawing
function drawRadar() {
    ctx.clearRect(0, 0, canvas.width, canvas.height);
    
    // Draw Grid
    ctx.strokeStyle = '#222';
    ctx.lineWidth = 1;
    ctx.beginPath();
    ctx.moveTo(0, canvas.height / 2);
    ctx.lineTo(canvas.width, canvas.height / 2);
    ctx.stroke();

    // Draw Line
    ctx.strokeStyle = '#00e5ff';
    ctx.lineWidth = 2;
    ctx.beginPath();
    for (let i = 0; i < latencyHistory.length; i++) {
        const val = latencyHistory[i];
        let normalized = val / maxLatency;
        if (normalized > 1) normalized = 1;
        
        const x = i;
        const y = canvas.height - (normalized * canvas.height);
        
        if (i === 0) ctx.moveTo(x, y);
        else ctx.lineTo(x, y);
    }
    ctx.stroke();
}

// Connect SSE
function connect() {
    addLog("Connecting to Quantum Telemetry...", "info");
    const source = new EventSource('/events');

    source.onopen = () => {
        DOM.wsLed.className = 'led pulse-green';
        DOM.wsStatus.innerText = 'TELEMETRY LINK: ONLINE';
        addLog("Telemetry Link Established.", "info");
    };

    source.onerror = () => {
        DOM.wsLed.className = 'led pulse-red';
        DOM.wsStatus.innerText = 'TELEMETRY LINK: OFFLINE';
    };

    source.onmessage = (event) => {
        try {
            const data = JSON.parse(event.data);
            
            if (data.OmniUpdate) {
                // Latency
                const lat = data.OmniUpdate.latency_ms; // actually ns
                DOM.latency.innerText = lat.toString();
                
                latencyHistory.shift();
                latencyHistory.push(lat);
                
                if (lat > maxLatency) maxLatency = lat * 1.2; // auto-scale
                
                requestAnimationFrame(drawRadar);
                
                if (data.OmniUpdate.latency_panic) {
                    DOM.latencyPanic.innerText = "PANIC: >50ms";
                    DOM.latencyPanic.style.color = "var(--neon-red)";
                } else {
                    DOM.latencyPanic.innerText = "STABLE";
                    DOM.latencyPanic.style.color = "var(--neon-green)";
                }

                // Dark Alpha (0.0 to 1.0)
                const alpha = data.OmniUpdate.dark_alpha;
                DOM.alphaBar.style.width = `${(alpha * 100).toFixed(1)}%`;
                DOM.alphaVal.innerText = `${(alpha * 100).toFixed(2)}%`;
                
                if (alpha > 0.8) {
                    DOM.alphaAlert.innerText = "CASCADE WARNING";
                    DOM.alphaAlert.style.color = "var(--neon-red)";
                    DOM.alphaBar.style.background = "var(--neon-red)";
                } else {
                    DOM.alphaAlert.innerText = "SAFE";
                    DOM.alphaAlert.style.color = "var(--neon-green)";
                    DOM.alphaBar.style.background = "linear-gradient(90deg, var(--neon-blue), var(--neon-red))";
                }

                // PnL
                DOM.scalpPnl.innerText = data.OmniUpdate.scalp_pnl.toFixed(4);
                DOM.swingPnl.innerText = data.OmniUpdate.swing_pnl.toFixed(4);
                updateColor(DOM.scalpPnl, data.OmniUpdate.scalp_pnl);
                updateColor(DOM.swingPnl, data.OmniUpdate.swing_pnl);
            }
            
            if (data.CapitalUpdate) {
                DOM.capital.innerText = formatMoney(data.CapitalUpdate);
            }
            
            if (data.TensorUpdate) {
                const vec = data.TensorUpdate;
                for (let i = 0; i < 10; i++) {
                    // Assume inputs are normalized mostly -1 to 1 or 0 to 1
                    let val = Math.abs(vec[i]);
                    if (val > 1) val = 1;
                    tensors[i].style.height = `${(val * 100).toFixed(1)}%`;
                    tensors[i].style.background = vec[i] > 0 ? 'var(--neon-green)' : 'var(--neon-red)';
                }
            }
            
            if (data.LogUpdate) {
                addLog(data.LogUpdate[1], data.LogUpdate[0]);
            }
            
        } catch (e) {
            console.error("Parse error", e);
        }
    };
}

// Start
connect();
