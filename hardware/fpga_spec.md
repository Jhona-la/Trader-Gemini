# OMEGA Protocol: FPGA Hardware Acceleration Spec

## Overview
This document outlines the architectural requirements for offloading the Trader Gemini Event Loop to Xilinx Alveo U50 FPGA cards.

## Verilog Module Specifications

### 1. `ome_engine_v1` (Order Matching Engine)
- **Input**: FIX 4.4 / SBE Stream via UDP (10GbE Kernel Bypass)
- **Logic**: 
  - Parsing incoming Market Data packets (No CPU interrupt).
  - Maintaining Limit Order Book (LOB) in BRAM (Block RAM).
- **Output**: Order Triggers (GPIO/PCIe)

### 2. `risk_check_core`
- **Function**: Pre-Trade Risk Validation (< 200ns)
- **Checks**:
  - Max Position Size
  - Max Notional Value
  - Kill Switch Signal (Hardware Pin)

### 3. `strategy_matrix_mul`
- **Function**: Acceleration for `ml_strategy.py` inference.
- **Math**: INT8 Matrix Multiplication (DSP Slices).
- **Latency Target**: < 5μs per inference.

## Memory Architecture
- **HBM2 (High Bandwidth Memory)**: Store historical tick data for `market_regime`.
- **DDR4**: Logging buffer (DMA to Host CPU).

## Deploy Strategy
1. **Simulation**: Verilator / Vivado Suite.
2. **Interface**: Integration via `pynq` (Python overlay).
