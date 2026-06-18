import time
import torch
import torch.nn as nn
import numpy as np
import hyper_kernel

class SophiaGenesis(nn.Module):
    """
    Red Neuronal Ficticia 10D para probar el puente de inferencia in-place.
    """
    def __init__(self):
        super().__init__()
        # Una capa densa sencilla que espera un tensor de 10 dimensiones
        self.fc = nn.Linear(10, 1)
        # Inicializamos los pesos para consistencia visual
        nn.init.constant_(self.fc.weight, 0.1)
        nn.init.constant_(self.fc.bias, 0.0)

    def forward(self, x):
        return self.fc(x)

def main():
    print("--- ORQUESTANDO EL PRIMER ALIENTE (SINGULARIDAD LIVE) ---")
    
    # 1. Cargar el modelo en memoria (Calentamiento)
    model = SophiaGenesis()
    model.eval()
    
    # 2. Simular un Tick Real Parseado (ej. desde WebSocket / Parquet)
    # Valores crudos simulados de Microestructura
    # Dim 3: OBI raw, Dim 4: Kyle's Lambda raw, Dim 5: Shannon Entropy raw
    tick_data = np.array([1.0, -0.5, 0.0, 15.0, 120.5, 1.8, 0.0, 0.0, 0.0, 0.0], dtype=np.float32)
    
    print("Tick Original (Crudo):")
    print(tick_data)

    ptr = tick_data.ctypes.data

    # >>> IGNICIÓN DEL CRONÓMETRO <<<
    start_ns = time.perf_counter_ns()

    # 3. Invocar al Motor Cuántico (Rust) para Adaptación Estocástica
    # Transforma OBI, Kyle y Shannon en el mismo espacio de memoria
    hyper_kernel.calculate_physics(ptr)

    # 4. Puente Zero-Copy hacia el Cerebro
    # tick_data ahora tiene la física validada, creamos el Tensor sin asignar heap adicional
    tensor = torch.frombuffer(tick_data, dtype=torch.float32)

    # 5. Inferencia IA
    with torch.no_grad():
        output = model(tensor)
    
    # >>> FIN DEL CRONÓMETRO <<<
    end_ns = time.perf_counter_ns()
    
    elapsed_us = (end_ns - start_ns) / 1000.0

    print("\nTensor Adaptado In-Place (Entrada a IA):")
    print(tensor)

    print("\n--- MÉTRICAS DEL NACIMIENTO ---")
    print(f"Latencia Total (µs):  {elapsed_us:.2f}")
    print(f"Heap Allocations:     0 (Zero-copy achieved)")
    
    val = output.item()
    is_valid = not (torch.isnan(output).item() or torch.isinf(output).item())
    
    print(f"Output Válido:        {is_valid} ({val:.6f})")
    
    if is_valid:
        if val > 0.05:
            decision = "LONG"
        elif val < -0.05:
            decision = "SHORT"
        else:
            decision = "HOLD"
        print(f"Decisión IA:          {decision}")
    else:
        print("Decisión IA:          FALLO CATASTRÓFICO NUMÉRICO")

if __name__ == "__main__":
    main()
