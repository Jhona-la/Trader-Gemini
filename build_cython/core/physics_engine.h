#ifndef PHYSICS_ENGINE_H
#define PHYSICS_ENGINE_H

#include <math.h>
#include <stdlib.h>
#include <string.h>

#define ZSCORE_WINDOW 20
#define RSI_WINDOW 14

typedef struct {
    // Ring buffer for Z-Score (requires last N prices to drop oldest for Welford/Moving variance)
    // Actually, Welford doesn't need ring buffer if we do it recursively, but standard moving std 
    // requires dropping the oldest value to compute the sliding window accurately.
    // For 20 periods, a simple ring buffer of size 20 is perfectly fine and O(1).
    double prices[ZSCORE_WINDOW];
    int count;
    int index;
    double sum;
    double sum_sq;

    // RSI Wilder's Smoothing State
    int rsi_count;
    double prev_close;
    double smooth_up;
    double smooth_down;

    // Output features
    double rsi_out;
    double zscore_out;
    double log_return_out;

} PhysicsState;

// Initialize state
PhysicsState* physics_init();

// Update state with new price and compute features in O(1)
void physics_update(PhysicsState* state, double current_price);

// Free state
void physics_free(PhysicsState* state);

#endif
