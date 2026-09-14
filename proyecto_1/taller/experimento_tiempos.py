import time
import numpy as np
from implementacion_transformadas import dft_bruteforce, fft_recursivo
import os
import matplotlib.pyplot as plt
from matplotlib.ticker import ScalarFormatter

def medir_tiempo(func,x,repeticiones=3):
    tiempos = []
    for _ in range(repeticiones):
        inicio = time.perf_counter()
        func(x)
        tiempos.append(time.perf_counter()-inicio)
    return min(tiempos)
if __name__ == "__main__":
    Ns = [64, 128, 256, 512, 1024, 2048, 4096]
    tiempos_dft = []
    tiempos_fft = []
    tiempos_np = []

    print(f"{'N':>6} {'DFT (s)':>12} {'FFT propia (s)':>16} {'numpy (s)':>12}")
    for N in Ns:
        x = np.random.rand(N)
        t_dft = medir_tiempo(dft_bruteforce, x)
        t_fft = medir_tiempo(lambda s: fft_recursivo(list(s)), x)
        t_np = medir_tiempo(np.fft.fft,x)
        tiempos_dft.append(t_dft)
        tiempos_fft.append(t_fft)
        tiempos_np.append(t_np)
        print(f"{N:6d} {t_dft:12.6f} {t_fft:16.6f} {t_np:12.8f}")

    fig, ax = plt.subplots(figsize=(7, 5))

    ax.plot(Ns, tiempos_dft, marker='o', markersize=8, linewidth=2,
            color='#2a78d6', label='DFT (fuerza bruta)')
    ax.plot(Ns, tiempos_fft, marker='s', markersize=8, linewidth=2,
            color='#eb6834', label='FFT propia (recursiva)')
    ax.plot(Ns, tiempos_np, marker='^', markersize=8, linewidth=2,
            color='#1baf7a', label='numpy.fft.fft')

    ax.set_xscale('log', base=2)
    ax.set_yscale('log')
    ax.set_xticks(Ns)
    ax.get_xaxis().set_major_formatter(ScalarFormatter())
    ax.set_xlabel('N (número de muestras)')
    ax.set_ylabel('Tiempo de ejecución (s)')
    ax.set_title('Comparación de tiempos: DFT vs FFT (escala log-log)')
    ax.grid(True, which='both', linestyle='-', linewidth=0.5, color='#e1e0d9', alpha=0.7)
    ax.legend(frameon=False)
    fig.tight_layout()
    os.makedirs('proyecto_1/taller/figuras', exist_ok=True)
    fig.savefig('proyecto_1/taller/figuras/comparacion_tiempos.png', dpi=150)
    plt.show()