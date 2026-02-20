import numpy as np
from typing import List, Tuple, Optional, Deque
import logging
from collections import deque
from utils.wandb_tracker import wandb_tracker # Phase 17: Telemetry

logger = logging.getLogger("OnlineLearner")

class DriftDetector:
    """
    [DF-D10] Detects Concept Drift using Online Error Statistics.
    Triggers when error moving average deviates significantly from baseline.
    """
    def __init__(self, window_size=1000, threshold=2.0):
        self.window = deque(maxlen=window_size)
        self.threshold = threshold
        self.baseline_mean = 0.0
        self.baseline_std = 0.0
        self.initialized = False
        
    def update(self, error: float) -> bool:
        """Returns True if drift detected."""
        self.window.append(error)
        
        if len(self.window) < 100:
            return False
            
        current_mean = np.mean(self.window)
        current_std = np.std(self.window)
        
        if not self.initialized:
            if len(self.window) >= 100:
                self.baseline_mean = current_mean
                self.baseline_std = current_std
                self.initialized = True
            return False
            
        # Check for Drift (Significant rise in error)
        # We only care if error INCREASES (model getting worse)
        z_score = (current_mean - self.baseline_mean) / (self.baseline_std + 1e-6)
        
        if z_score > self.threshold:
            return True
            
        # Adaptive Baseline (Slow update)
        self.baseline_mean = 0.99 * self.baseline_mean + 0.01 * current_mean
        self.baseline_std = 0.99 * self.baseline_std + 0.01 * current_std
        return False

class OnlineLearner:
    """
    Aprendiz Online (Trinidad Omega - Phase 46).
    Implementa Stochastic Gradient Descent (SGD) para ajustar pesos en tiempo real.
    """
    def __init__(self, learning_rate: float = 0.001, clip_value: float = 0.01, weight_decay: float = 1e-4, target_clip: float = 10.0):
        self.learning_rate = learning_rate
        self.clip_value = clip_value
        self.weight_decay = weight_decay  # [SS-015 FIX] L2 regularization
        self.target_clip = target_clip    # [DF-B5 FIX] Target Clipping
        
        # [DF-D10] Drift Detector
        self.drift_detector = DriftDetector()
        self.drift_detected_count = 0

        # PHASE 16: CI/CD Intelligence (Drift Detection)
        self.error_history = deque(maxlen=100)
        self.drift_threshold = 2.0 # Standard Deviations
        self.last_retrain_time = 0
        
    def detect_drift(self, error: float) -> bool:
        """
        Public interface for Concept Drift Detection.
        """
        return self.drift_detector.update(error)
        
    def update_weights(self, 
                       weights: np.ndarray, 
                       inputs: np.ndarray, 
                       target: float, 
                       prediction: float) -> np.ndarray:
        """
        Ajusta los pesos usando la regla delta (LMS - Least Mean Squares).
        
        Error = Target - Prediction
        Delta = Learning_Rate * Error * Input
        New_Weights = Old_Weights + Delta
        
        Args:
            weights: Vector/Matriz de pesos actuales.
            inputs: Tensor de entrada que generó la predicción.
            target: Valor real esperado (e.g., retorno siguiente barra).
            prediction: Valor predicho por la red.
            
        Returns:
            Nuevos pesos ajustados.
        """
        # [DF-B5] Target Clipping
        target = max(min(target, self.target_clip), -self.target_clip)
        
        # Calcular Error
        error = target - prediction
        
        # Calcular Gradiente (negativo del error * input)
        # Delta = LearningRate * Error * Input
        
        # Manejo de dimensión para asegurar broadcast correcto
        # Si weights es Matrix (Input x Output), y inputs es Vector (Input)
        # Necesitamos saber a qué salida corresponde la predicción o si es una actualización vectorizada.
        
        # REGRESSION MODE (Single Output assumed for simplicity in Phase 46)
        # We start simpler: The 'brain' predicts ONE value (Score/Return)
        # But our brain is 25x4. 
        # Strategy: Update ONLY the weights contributing to the active action?
        # OR: Train a separate 'Predictor' head?
        
        # For Phase 46, we implement the math for the generic Linear Layer update.
        # Assuming weights is 1D or logic handles the shape.
        
        # Simple Linear Case:
        # y = w . x
        # dE/dw = -error * x
        
        if weights.shape != inputs.shape and weights.ndim > 1:
             # Matrix Case (Input x Output) - NOT HANDLED SIMPLE YET
             # Requires passing the 'Action Index' that was taken
             logger.warning("Matrix update request without Action Index. Skipping.")
             return weights
             
        # Vector Case (e.g. Single Neuron or Specific Column)
        delta = self.learning_rate * error * inputs
        
        # AEGIS-ULTRA Phase 17: Telemetry (Gradient Logging)
        try:
            grad_norm = np.linalg.norm(delta)
            wandb_tracker.log_metric("ml/gradient_norm", float(grad_norm))
            
            # Exploding Gradient Detection
            if grad_norm > 10.0: # Threshold for warning
                logger.warning(f"⚠️ [EXPLODING GRADIENT] Norm={grad_norm:.2f}. Clipping applied.")
                wandb_tracker.log_metric("ml/gradient_exploded", 1.0)
        except Exception:
            pass # Non-critical path

        # Gradient Clipping (Safety)
        delta = np.clip(delta, -self.clip_value, self.clip_value)
        
        # [SS-015 FIX] L2 Weight Decay: prevents unbounded growth during Black Swan outliers
        new_weights = weights * (1.0 - self.weight_decay) + delta
        return new_weights

    def update_matrix(self,
                     weights_matrix: np.ndarray,
                     inputs: np.ndarray,
                     target: float,
                     prediction: float,
                     output_index: int) -> np.ndarray:
        """
        Actualiza una columna específica de una matriz de pesos.
        Usado cuando la red tiene múltiples salidas (acciones) y solo entrenamos la elegida.
        """
        # Validar dimensiones
        # weights: (Input, Output)
        # inputs: (Input,)
        
        input_dim, output_dim = weights_matrix.shape
        if inputs.shape[0] != input_dim:
            logger.error(f"Input shape mismatch. Exp {input_dim}, got {inputs.shape[0]}")
            return weights_matrix
            
        # Extraer columna de pesos activa
        active_weights = weights_matrix[:, output_index]
        
        # Calcular Delta para esa columna
        # Error * Input
        error = target - prediction
        delta = self.learning_rate * error * inputs
        
        # Clip
        delta = np.clip(delta, -self.clip_value, self.clip_value)
        
        # Actualizar Matriz (Copia para inmutabilidad/seguridad)
        new_matrix = weights_matrix.copy()
        # [SS-015 FIX] Apply L2 decay to active column before adding delta
        new_matrix[:, output_index] = active_weights * (1.0 - self.weight_decay) + delta
        return new_matrix  # F15: Was missing — returned None!
        
    # ------------------------------------------------------------------
    # PHASE 32: BATCH LEARNING (JIT)
    # ------------------------------------------------------------------
    
    def train_on_batch(self, buffer, batch_size: int = 32) -> float:
        """
        Entrena el modelo usando un lote de experiencia del buffer compartido.
        """
        # 1. Sample Batch (Zero-Copy)
        states, actions, rewards, next_states = buffer.sample(batch_size)
        
        # If empty or not enough data
        if len(states) == 0:
            return 0.0
            
        # 2. Compute Targets (Bellman: r + gamma * max(Q(s')))
        # For Phase 32, we assume a simpler target for now: 
        # The 'reward' IS the target return? Or are we doing Q-Learning?
        # NeuralBridge architecture suggests we are predicting "Score" or "Action Value"
        # If we are doing simple Online Learning (Phase 46 style), we might just supervised train:
        # Prediction = Model(s)
        # Target = r (Actual Return)
        # Error = r - Prediction
        
        # But we need the MODEL (Weights) to update.
        # OnlineLearner in Phase 46 was "Stateless" helper or assumed caller holds weights.
        # GENOTYPE holds the weights.
        # This brings up a detailed architecture question:
        # Who calls train_on_batch? Usage:
        # strategy.learner.train_on_batch(buffer) -> updates WHOSE weights?
        # It needs the weights passed in.
        pass # Placeholder to clarify architecture
        
        # REFACTOR: We need to pass weights IN or store them.
        # Since Genotypes hold weights, and we might be training the "Alpha" genotype...
        # Let's adjust signature to take weights.
        
    def train_weights_on_batch(self, 
                             weights: np.ndarray, 
                             buffer, 
                             batch_size: int = 32,
                             gamma: float = 0.95) -> Tuple[np.ndarray, float]:
        """
        Executes SGD on a batch from the buffer.
        Returns (updated_weights, avg_error).
        """
        # 1. Sample
        s, a, r, sn = buffer.sample(batch_size)
        if len(s) == 0: return weights, 0.0
            
        # 2. JIT Update
        from core.online_learning_kernels import jit_sgd_batch
        
        # We need a kernel. Let's create it inline or import.
        # Creating a dedicated kernel file is cleaner.
        # But for now, let's assume we import `jit_sgd_batch`.
        
        new_weights, avg_err = jit_sgd_batch(
            weights, s, a, r, sn, 
            self.learning_rate, self.clip_value, gamma,
            self.target_clip  # [DF-B5]
        )
        return new_weights, avg_err  # F15: Was missing — returned None!
        
    def learn_single(self,
                     weights: np.ndarray,
                     state: np.ndarray,
                     action: float,
                     reward: float,
                     next_state: np.ndarray,
                     gamma: float = 0.95) -> float:
        """
        Executes Single-Step SGD (<500ns target).
        MODIFIES WEIGHTS IN-PLACE.
        Returns: Absolute Error.
        """
        from core.online_learning_kernels import jit_sgd_single
        
        # Ensure Types for Numba (float32 preferred for speed/simd)
        # Assuming caller handles typing or we cast here (casting adds latency!)
        # Fast path: assume inputs are correct dtypes.
        
        error = jit_sgd_single(weights, state, action, reward, next_state, 
                              self.learning_rate, self.clip_value, gamma,
                              self.target_clip) # [DF-B5]

        # [DF-D10] Concept Drift Detection
        if self.drift_detector.update(error):
            self.drift_detected_count += 1
            if self.drift_detected_count % 50 == 0:  # Avoid spam
                logger.warning(
                    f"⚠️ [CONCEPT DRIFT] Error mean deviated (Z > {self.drift_detector.threshold}). "
                    f"Model may be obsolete or regime changed."
                )
                
        return error

    def learn_batch(self,
                    weights_batch: np.ndarray,
                    states_batch: np.ndarray,
                    actions_batch: np.ndarray,
                    rewards_batch: np.ndarray,
                    next_states_batch: np.ndarray,
                    gamma: float = 0.95) -> np.ndarray:
        """
        Batch update for multiple symbols simultaneously (Parallel Dispatch).
        [PHASE 63] Enforces C-contiguity for cache-locality.
        """
        from core.online_learning_kernels import jit_sgd_parallel
        
        # Ensure memory locality for SIMD/Multi-threading
        w_cont = np.ascontiguousarray(weights_batch)
        s_cont = np.ascontiguousarray(states_batch)
        ns_cont = np.ascontiguousarray(next_states_batch)
        
        errors = jit_sgd_parallel(
            w_cont, s_cont, actions_batch,
            rewards_batch, ns_cont,
            self.learning_rate, self.clip_value, gamma,
            self.target_clip # [DF-B5]
        )
        
        # If weights were modified (they are modified in-place in kernel), 
        # but w_cont might be a copy. We must copy back if not same.
        if w_cont is not weights_batch:
            weights_batch[:] = w_cont
            
        return errors

    # ------------------------------------------------------------------
    # PHASE 9 (NEURAL-FORTRESS): PROXIMAL POLICY OPTIMIZATION (PPO)
    # ------------------------------------------------------------------

    def update_ppo_batch(self, 
                         weights: np.ndarray,
                         states: np.ndarray,
                         actions: np.ndarray,
                         old_log_probs: np.ndarray,
                         rewards: np.ndarray,
                         advantages: Optional[np.ndarray] = None,
                         epsilon: float = 0.2) -> Tuple[np.ndarray, np.ndarray]:
        """
        Executes a PPO update step on a batch.
        Assumes a Linear Policy: Action ~ Normal(Weights @ State, Fixed_Std).
        
        Args:
            weights: Current policy weights (Input x Output).
            states: Batch of states (B x Input).
            actions: Batch of actions taken (B x Output).
            old_log_probs: Log probs of actions under OLD policy (B).
            rewards: Raw rewards for the batch (B).
            advantages: Optional pre-calculated advantage estimates (B).
            epsilon: PPO Clip parameter (default 0.2).
            
        Returns:
            Tuple[updated_weights, absolute_advantages]
        """
        # 0. Advantage Calculation (if not provided)
        if advantages is None:
            # Simple baseline: Normalize rewards to get relative advantage
            mean_r = np.mean(rewards)
            std_r = np.std(rewards) + 1e-8
            advantages = (rewards - mean_r) / std_r
            
        # 1. Forward Pass (New Policy)
        if weights.ndim == 1:
            # Case: Linear Model, Single Output (B x Input) @ (Input) -> (B)
            mu = states @ weights
            
            # --- [PRECISION-AXIOMA] DEEP MATRIX AUDIT (Shadow Calculation) ---
            try:
                # Check precision using numpy's longdouble (np.float128/np.float96) if available, else float64
                high_prec_type = np.longdouble if hasattr(np, 'longdouble') else np.float64
                states_hp = states.astype(high_prec_type)
                weights_hp = weights.astype(high_prec_type)
                mu_hp = states_hp @ weights_hp
                
                # Check drift (Max deviation)
                max_drift = np.max(np.abs(mu - mu_hp.astype(np.float64)))
                if max_drift > 1e-12:
                    logger.warning(f"⚠️ [AXIOMA-LOG] Precision Drift Detected in PPO Tensor Audit! Max Deviation: {max_drift:.4e}")
            except Exception as e:
                pass # Fail silently if high precision is not supported
            # -----------------------------------------------------------------
            
        else:
             # Case: Matrix
             logger.warning("PPO not fully implemented for Matrix weights yet. Using column 0.")
             mu = states @ weights[:, 0]

        # 2. Calculate New Log Probs
        # Assuming Fixed Std Dev = 1.0 for simplicity in this Phase.
        sigma = 1.0
        new_log_probs = -0.5 * ((actions - mu) / sigma)**2
        
        # 3. Calculate Ratio with Underflow/Overflow Shield (Axioma Protocol)
        # Prevent np.exp from exploding (NaN) if discrepancy is gigantic
        log_diff = new_log_probs - old_log_probs
        # Safe clipping: max exp(700) approx, keep it well within safe bounds
        log_diff_safe = np.clip(log_diff, a_min=-100.0, a_max=80.0)
        ratio = np.exp(log_diff_safe)
        
        # 4. PPO Loss Logic
        # Gradient of LogProb w.r.t Mu
        d_log_d_mu = (actions - mu) # Assuming sigma=1
        
        # Check clipping condition
        grad_mask = np.ones_like(ratio)
        
        # If A > 0 and r > 1+eps: Gradients cut
        grad_mask[(advantages > 0) & (ratio > 1.0 + epsilon)] = 0.0
        
        # If A < 0 and r < 1-eps: Gradients cut
        grad_mask[(advantages < 0) & (ratio < 1.0 - epsilon)] = 0.0
        
        # Total Gradient of Objective J
        # Grad = A * r * d_log_d_mu * mask
        delta_mu = advantages * ratio * d_log_d_mu * grad_mask
        
        # Backprop to weights
        if weights.ndim == 1:
            grad_weights = states.T @ delta_mu / len(states)
        else:
            grad_weights = np.zeros_like(weights)
        
        # 5. Apply Update (Gradient Ascent on Objective)
        update = self.learning_rate * grad_weights
        
        # 6. Telemetry (Phase 17)
        try:
             grad_norm = np.linalg.norm(update)
             wandb_tracker.log_metric("ml/ppo_grad_norm", float(grad_norm))
             wandb_tracker.log_metric("ml/ppo_ratio_mean", float(np.mean(ratio)))
             wandb_tracker.log_metric("ml/ppo_clip_fraction", float(1.0 - np.mean(grad_mask)))
        except: 
            pass
            
        new_weights = weights + update
            
        # L2 Decay
        new_weights = new_weights * (1.0 - self.weight_decay)
        
        # Return weights AND absolute advantages (for PER priority update)
        return new_weights, np.abs(advantages)


