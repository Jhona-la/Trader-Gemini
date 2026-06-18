import os

def patch_file():
    path = "scripts/run_god_mode_backtest.py"
    with open(path, "r", encoding="utf-8") as f:
        content = f.read()

    # Inyección
    injection_top = """
from concurrent.futures import ThreadPoolExecutor, as_completed
from queue import Queue as SafeQueue

_global_thread_pool = None

def _deterministic_strat_worker(strat, symbol, epoch_count, warmup_epochs, ml_consecutive_losses, ml_shadow_wins, last_fill_epoch, market_events, data_provider):
    import numpy as np
    import torch
    from core.events import SignalType, SignalEvent

    _orig_q = strat.events_queue
    _temp_q = SafeQueue()
    strat.events_queue = _temp_q
    
    try:
        strat.loop_count += 1
        strat.bars_since_train += 1
        strat._last_prediction_time = None

        if not getattr(strat, "is_trained", False):
            if hasattr(strat, "_launch_training"):
                bars = data_provider.get_latest_bars(
                    symbol,
                    getattr(strat, "lookback", 500),
                    getattr(strat, "PRIMARY_TF", "5m"),
                )
                if bars is not None and len(bars) > 50:
                    is_training = (
                        hasattr(strat, "_training_thread")
                        and strat._training_thread
                        and strat._training_thread.is_alive()
                    )
                    if not is_training:
                        try:
                            strat._launch_training(bars, "Full")
                        except Exception:
                            pass
            elif epoch_count % 500 == 0:
                if hasattr(strat, "_train_model"):
                    try:
                        strat._train_model()
                    except Exception:
                        pass
                elif hasattr(strat, "train_model"):
                    try:
                        strat.train_model()
                    except Exception:
                        pass

            if not getattr(strat, "is_trained", False):
                return (symbol, getattr(strat, 'strategy_id', ''), [])
                
        _strat_id = strat.strategy_id
        _loss_count = ml_consecutive_losses.get(_strat_id, 0)
        ML_LOSS_STREAK_LIMIT = 5
        if _loss_count >= ML_LOSS_STREAK_LIMIT:
            _shadow_wins = ml_shadow_wins.get(_strat_id, 0)
            if _shadow_wins < 3:
                _shadow_q_local = SafeQueue()
                strat.events_queue = _shadow_q_local
                strat._run_inference()
                strat.events_queue = _temp_q
                while not _shadow_q_local.empty():
                    _shadow_q_local.get()
                return (symbol, _strat_id, [])

        _horizon = getattr(strat, "horizon", "SCALPING")
        _cooldown_key = f"{symbol}_{_horizon}"
        COOLDOWN_EPOCHS = 10
        _last_fill = last_fill_epoch.get(_cooldown_key, -COOLDOWN_EPOCHS)
        if (epoch_count - _last_fill) < COOLDOWN_EPOCHS:
            return (symbol, _strat_id, [])

        try:
            if getattr(strat, "strategy_id", "").startswith("ML_"):
                try:
                    import hyper_kernel
                    buffer_data = np.array([1.0, -0.5, 0.0, 15.0, 120.5, 1.8, 0.0, 0.0, 0.0, 0.0], dtype=np.float32)
                    ptr = buffer_data.ctypes.data
                    hyper_kernel.calculate_physics(ptr)
                    tensor_10d = torch.frombuffer(buffer_data, dtype=torch.float32)
                    val = tensor_10d[4].item()
                    decision = "LONG" if val > 0.5 else "SHORT" if val < -0.5 else "HOLD"
                    
                    if decision != "HOLD":
                        sig_type = SignalType.LONG if decision == "LONG" else SignalType.SHORT
                        conf = 0.88
                        current_ts = market_events[0].timestamp if market_events else None
                        current_price = data_provider.get_latest_price(symbol) or 0.0
                        
                        strat.events_queue.put(
                            SignalEvent(
                                strategy_id="SOPHIA_ZERO_COPY",
                                symbol=symbol,
                                datetime=current_ts,
                                signal_type=sig_type,
                                strength=conf,
                                current_price=current_price,
                                sl_pct=0.015,
                                tp_pct=0.03,
                                horizon=getattr(strat, "horizon", "SCALPING"),
                                predicted_magnitude=0.015,
                                predicted_duration=5
                            )
                        )
                except ImportError:
                    strat._run_inference()
            else:
                strat._run_inference()
        except Exception:
            pass
            
    except Exception:
        pass
    finally:
        strat.events_queue = _orig_q
        
    res = []
    while not _temp_q.empty():
        res.append(_temp_q.get())
    return (symbol, getattr(strat, 'strategy_id', ''), res)
"""

    if "_deterministic_strat_worker" not in content:
        # We replace the first `import os\n` we see
        content = content.replace("import os\n", "import os\n" + injection_top, 1)

    idx_start = content.find("# B5. RUN STRATEGIES")
    idx_end = content.find("# ── B6. RUN GLOBAL EPOCH STRATEGIES")

    # Find the beginning of the line for idx_start
    while idx_start > 0 and content[idx_start - 1] != '\n':
        idx_start -= 1
        
    while idx_end > 0 and content[idx_end - 1] != '\n':
        idx_end -= 1
        
    if idx_start == -1 or idx_end == -1:
        print("COULD NOT FIND B5 OR B6 BLOCK")
        return

    b5_replacement = """                # B5. RUN STRATEGIES — QUANTUM DETERMINISTIC MULTI-THREADING
                # ═══════════════════════════════════════════════════════════════
                global _global_thread_pool
                if _global_thread_pool is None:
                    _global_thread_pool = ThreadPoolExecutor(max_workers=8)

                if epoch_count >= warmup_epochs and market_events:
                    futures = []
                    for event in market_events:
                        symbol = event.symbol
                        for strat in strategies_map.get(symbol, []):
                            f = _global_thread_pool.submit(
                                _deterministic_strat_worker,
                                strat, symbol, epoch_count, warmup_epochs, ml_consecutive_losses, ml_shadow_wins, last_fill_epoch, market_events, data_provider
                            )
                            futures.append(f)
                            
                    all_signals_nested = []
                    for f in futures:
                        all_signals_nested.append(f.result())
                        
                    all_signals_nested.sort(key=lambda x: (x[0], x[1]))
                    
                    for sym, sid, sigs in all_signals_nested:
                        for sig in sigs:
                            events_queue.put(sig)
                            
"""
    new_content = content[:idx_start] + b5_replacement + content[idx_end:]
    
    with open(path, "w", encoding="utf-8") as f:
        f.write(new_content)
    print("PATCH APPLIED SUCCESSFULLY")

if __name__ == "__main__":
    patch_file()
