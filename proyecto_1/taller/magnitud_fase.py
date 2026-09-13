import numpy as np
import matplotlib.pyplot as plt

def graficar_magnitud_fase(x, fs, titulo, archivo=None, umbral_mag=1e-6):
    N = len(x)
    X = np.fft.fft(x)
    frecuencias = np.fft.fftfreq(N, d=1/fs)

    mitad = N // 2
    frecuencias = frecuencias[:mitad]
    magnitud = np.abs(X[:mitad])
    fase = np.angle(X[:mitad])
    fase_mostrar = np.where(magnitud > umbral_mag, fase, np.nan)

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(7, 6), sharex=True)

    ax1.plot(frecuencias, magnitud, color='#2a78d6', linewidth=2)
    ax1.set_ylabel('Magnitud |X(f)|')
    ax1.set_title(titulo)
    ax1.grid(True, linewidth=0.5, color='#e1e0d9', alpha=0.7)

    ax2.plot(frecuencias, fase_mostrar, color='#eb6834', linewidth=1.5,
              marker='.', markersize=4, linestyle='none')
    ax2.set_ylabel('Fase (radianes)')
    ax2.set_xlabel('Frecuencia (Hz)')
    ax2.grid(True, linewidth=0.5, color='#e1e0d9', alpha=0.7)

    fig.tight_layout()
    if archivo:
        fig.savefig(archivo, dpi=150)
    plt.show()


fs = 256
N = 256
n = np.arange(N)
t = n / fs

#Seno puro
x1 = np.sin(2*np.pi*50*t)
graficar_magnitud_fase(x1, fs, 'Seno puro (50 Hz)', 'proyecto_1/taller/figuras/mag_fase_seno_puro.png')

#Suma de dos senos
x2 = np.sin(2*np.pi*50*t) + 0.5*np.sin(2*np.pi*120*t)
graficar_magnitud_fase(x2, fs, 'Suma de dos senos (50 Hz y 120 Hz)', 'proyecto_1/taller/figuras/mag_fase_dos_senos.png')

#Seno con ruido
np.random.seed(0)
x3 = np.sin(2*np.pi*50*t) + 0.3*np.random.randn(N)
graficar_magnitud_fase(x3, fs, 'Seno con ruido (50 Hz)', 'proyecto_1/taller/figuras/mag_fase_con_ruido.png')

#Frecuencia no alineada a un bin
x4 = np.sin(2*np.pi*53.7*t)
graficar_magnitud_fase(x4, fs, 'Frecuencia no alineada a un bin (fuga espectral)', 'proyecto_1/taller/figuras/mag_fase_fuga_espectral.png')