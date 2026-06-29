const eventSource = new EventSource('/events');

// DOM Elements
const capitalEl = document.getElementById('capital');
const latencyEl = document.getElementById('latency');
const logsEl = document.getElementById('logs');

// Tensor Elements
const tensorBars = [];
const tensorVals = [];
for (let i = 0; i < 10; i++) {
    tensorBars.push(document.getElementById(`t${i}`));
    tensorVals.push(document.getElementById(`tv${i}`));
}

// Configuration for tensor normalization (min, max for visual scaling 0-100%)
const tensorRanges = [
    [-0.01, 0.01],    // 0: Price Δ
    [-0.05, 0.05],    // 1: Fast EMA Dist
    [-0.1, 0.1],      // 2: Slow EMA Dist
    [0, 50],          // 3: Velocity
    [-5, 5],          // 4: OBI Velocity
    [-1, 1],          // 5: OBI Accel
    [-0.005, 0.005],  // 6: Fund. Elasticity
    [-100, 100],      // 7: C-VPIN Net
    [0, 5],           // 8: Shannon Entropy
    [0, 100],         // 9: Dark Alpha Panic
];

eventSource.onmessage = function(event) {
    try {
        const data = JSON.parse(event.data);
        
        if (data.CapitalUpdate) {
            const cap = data.CapitalUpdate;
            capitalEl.innerText = `$${cap.toFixed(4)}`;
            
            // Add a subtle flash effect
            capitalEl.style.textShadow = '0 0 20px #00ff88';
            setTimeout(() => {
                capitalEl.style.textShadow = '0 0 10px rgba(0,255,136,0.4)';
            }, 300);
        } 
        else if (data.LatencyUpdate) {
            const ns = data.LatencyUpdate;
            if (ns < 1000) {
                latencyEl.innerText = `${ns} ns`;
            } else if (ns < 1000000) {
                latencyEl.innerText = `${(ns / 1000).toFixed(2)} µs`;
            } else {
                latencyEl.innerText = `${(ns / 1000000).toFixed(2)} ms`;
            }
        }
        else if (data.LogUpdate) {
            const type = data.LogUpdate[0];
            const msg = data.LogUpdate[1];
            
            const entry = document.createElement('div');
            entry.className = `log-entry log-${type}`;
            const time = new Date().toISOString().split('T')[1].slice(0, -1);
            entry.innerText = `[${time}] ${msg}`;
            
            logsEl.appendChild(entry);
            
            // Keep only last 50 logs
            if (logsEl.children.length > 50) {
                logsEl.removeChild(logsEl.firstChild);
            }
            
            // Auto scroll to bottom
            logsEl.scrollTop = logsEl.scrollHeight;
        }
        else if (data.TensorUpdate) {
            const tensors = data.TensorUpdate;
            
            for (let i = 0; i < 10; i++) {
                const val = tensors[i];
                const [min, max] = tensorRanges[i];
                
                // Normalize to 0-100%
                let pct = ((val - min) / (max - min)) * 100;
                pct = Math.max(0, Math.min(100, pct)); // clamp
                
                tensorBars[i].style.width = `${pct}%`;
                
                // Set color intensity based on value magnitude
                const intensity = Math.abs(val) > (max / 2) ? 'var(--neon-magenta)' : 'var(--neon-cyan)';
                tensorBars[i].style.boxShadow = `0 0 10px ${intensity}`;
                
                tensorVals[i].innerText = val.toFixed(4);
            }
        }
    } catch (e) {
        console.error("Error parsing event:", e, event.data);
    }
};

eventSource.onerror = function(err) {
    console.error("EventSource failed:", err);
};

// Initial welcome log
setTimeout(() => {
    const entry = document.createElement('div');
    entry.className = `log-entry log-success`;
    entry.innerText = `[SYSTEM] Omni-Dashboard UI Initialized. Waiting for Quantum Tensors...`;
    logsEl.appendChild(entry);
}, 500);
