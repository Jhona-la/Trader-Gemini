import numpy as np
import hyper_kernel

def main():
    print("--- INICIANDO SONDA DE CANARIO ---")
    
    # Creamos la topología inicial en Python
    # Alineación de C en el array float32
    original_array = np.array([1.1, 2.2, 3.3, 4.4, 5.5, 6.6, 7.7, 8.8, 9.9, 10.0], dtype=np.float32)
    
    # Precalculamos lo que esperamos que Rust devuelva (multiplicación por 2)
    expected_array = original_array.copy() * 2.0

    print(f"Estado Original Python: {original_array}")

    # Extraemos el puntero de memoria pura
    ptr = original_array.ctypes.data

    # Inyectamos a Rust: Mutación In-Place Zero Copy
    hyper_kernel.mutate_canary(ptr)

    print(f"Estado Mutado Rust: {original_array}")

    # TEST DE FUEGO BITWISE
    # tobytes() garantiza que estamos comparando los datos crudos hexadecimales de la memoria
    if original_array.tobytes() == expected_array.tobytes():
        print("\n[✔] SUCCESS: BRIDGE MEMORY CORRUPTION TEST PASSED.")
        print("[✔] El fotón cruzó el puente intacto. Zero-copy garantizado.")
    else:
        print("\n[X] FATAL ERROR: BRIDGE MEMORY CORRUPTION.")
        exit(1)

if __name__ == "__main__":
    main()
