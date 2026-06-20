#include "physics_engine.h"

PhysicsState* physics_init() {
    PhysicsState* state = (PhysicsState*)malloc(sizeof(PhysicsState));
    if (state == NULL) return NULL;
    
    memset(state->prices, 0, sizeof(double) * ZSCORE_WINDOW);
    state->count = 0;
    state->index = 0;
    state->sum = 0.0;
    state->sum_sq = 0.0;

    state->rsi_count = 0;
    state->prev_close = -1.0;
    state->smooth_up = 0.0;
    state->smooth_down = 0.0;

    state->rsi_out = 50.0; // Neutral start
    state->zscore_out = 0.0;
    state->log_return_out = 0.0;

    return state;
}

void physics_update(PhysicsState* state, double current_price) {
    if (state->prev_close < 0.0) {
        state->prev_close = current_price;
    }

    // 1. Log Return
    state->log_return_out = log(current_price / state->prev_close);

    // 2. RSI 14 (Wilder's Smoothing)
    double change = current_price - state->prev_close;
    double gain = (change > 0.0) ? change : 0.0;
    double loss = (change < 0.0) ? -change : 0.0;

    if (state->rsi_count < RSI_WINDOW) {
        // SMA initialization phase
        state->smooth_up += gain;
        state->smooth_down += loss;
        state->rsi_count++;
        
        if (state->rsi_count == RSI_WINDOW) {
            state->smooth_up /= RSI_WINDOW;
            state->smooth_down /= RSI_WINDOW;
        }
    } else {
        // Wilder's Exponential Smoothing
        state->smooth_up = (state->smooth_up * (RSI_WINDOW - 1) + gain) / RSI_WINDOW;
        state->smooth_down = (state->smooth_down * (RSI_WINDOW - 1) + loss) / RSI_WINDOW;
    }

    if (state->rsi_count >= RSI_WINDOW) {
        if (state->smooth_down == 0.0) {
            state->rsi_out = 100.0;
        } else {
            double rs = state->smooth_up / state->smooth_down;
            state->rsi_out = 100.0 - (100.0 / (1.0 + rs));
        }
    }

    // 3. Z-Score 20 (Ring Buffer for moving average & std)
    // Remove oldest element if buffer is full
    if (state->count == ZSCORE_WINDOW) {
        double oldest = state->prices[state->index];
        state->sum -= oldest;
        state->sum_sq -= (oldest * oldest);
    } else {
        state->count++;
    }

    // Add new element
    state->prices[state->index] = current_price;
    state->sum += current_price;
    state->sum_sq += (current_price * current_price);
    
    // Advance ring buffer index
    state->index = (state->index + 1) % ZSCORE_WINDOW;

    // Calculate moving Z-Score (ddof=0 matching Python's np.std(..., ddof=0))
    if (state->count > 0) {
        double mean = state->sum / state->count;
        double variance = (state->sum_sq / state->count) - (mean * mean);
        double std_dev = (variance > 0.0) ? sqrt(variance) : 0.0;

        if (std_dev > 0.0) {
            state->zscore_out = (current_price - mean) / std_dev;
        } else {
            state->zscore_out = 0.0;
        }
    }

    // Prepare for next tick
    state->prev_close = current_price;
}

void physics_free(PhysicsState* state) {
    if (state != NULL) {
        free(state);
    }
}
