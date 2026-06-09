import os
import torch
import sys

# Append root so we can import the models
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from models.omniscient_predictor import OmniscientTransformer, TORCH_AVAILABLE

def export_omniscient_to_onnx():
    print("🚀 Iniciando Traductor Cuántico de PyTorch a ONNX (GPU AMD)...")
    if not TORCH_AVAILABLE:
        print("❌ PyTorch no instalado. Abortando.")
        return

    # 1. Definir Rutas
    base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    pt_weights_path = os.path.join(base_dir, "models", "omniscience", "omni_weights_1m.pt")
    onnx_weights_path = os.path.join(base_dir, "models", "omniscience", "omni_weights_1m.onnx")

    if not os.path.exists(pt_weights_path):
        print(f"❌ No se encontró el cerebro PyTorch en: {pt_weights_path}")
        return

    # 2. Instanciar la Arquitectura Base del Oráculo
    print("🧠 Instanciando OmniscientTransformer (143 features, seq 60, horizon 1000)...")
    input_dim = 143
    seq_len = 60
    horizon = 1000
    
    model = OmniscientTransformer(input_dim=input_dim, seq_len=seq_len, horizon=horizon)
    
    # 3. Cargar Pesos Pre-Entrenados
    print("📂 Cargando pesos neuronales desde disco...")
    state_dict = torch.load(pt_weights_path, map_location=torch.device("cpu"), weights_only=True)
    
    # Tolerancia a cambios de dimensión (Feature Dim mismatch fix)
    saved_input_dim = state_dict.get("input_proj.weight", torch.empty(0)).shape[-1] if "input_proj.weight" in state_dict else input_dim
    if saved_input_dim != input_dim:
        print(f"⚠️ Alerta: Feature dim guardado es {saved_input_dim}, la red espera {input_dim}. Limpiando capa de entrada.")
        incompatible_keys = [k for k in state_dict if "input_proj" in k]
        for k in incompatible_keys:
            del state_dict[k]
            
    model.load_state_dict(state_dict, strict=False)
    model.eval()  # Freeze Dropout layers for deterministic export

    # 4. Crear un Tensor Fantasma (Dummy Input) para trazar el grafo matemático
    # Shape: [batch_size, seq_len, num_features]
    print(f"📐 Creando matriz tensorial fantasma [1, {seq_len}, {input_dim}]...")
    dummy_input = torch.randn(1, seq_len, input_dim, dtype=torch.float32)

    # 5. Exportar a ONNX
    print("⚡ Compilando Grafo de Inteligencia a formato Universal ONNX...")
    torch.onnx.export(
        model, 
        dummy_input, 
        onnx_weights_path,
        export_params=True,
        opset_version=14,          # Opset maduro para soporte MultiHeadAttention y DirectML
        do_constant_folding=True,  # Optimización de nodos constantes (Velocidad Extra)
        input_names=['sequence_features'], 
        output_names=['trajectory_prediction'],
        dynamic_axes={
            'sequence_features': {0: 'batch_size'},      # Batch variable para Inferencia Multi-moneda
            'trajectory_prediction': {0: 'batch_size'}
        }
    )

    print(f"✅ ¡ÉXITO! Oráculo compilado en: {onnx_weights_path}")
    print("La gráfica integrada AMD Radeon ahora puede procesar el futuro.")

if __name__ == "__main__":
    export_omniscient_to_onnx()
