import numpy as np
import sounddevice as sd
import matplotlib.pyplot as plt

FS = 48000                # frecuencia de muestreo de la tarjeta de sonido
V_SONIDO = 343.0
DUR_CHIRP = 0.010         # 10 ms
F_INICIO, F_FIN = 2000, 8000
DUR_GRABACION = 0.5       # margen para la latencia
VOLUMEN = 0.5             # 0 a 1;
D_PARLANTE_MIC = 0.35
D_MIN, D_MAX = 0.15, 2.0  # rango de búsqueda del eco (m)


def generar_chirp(fs, duracion, f0, f1):
    """Barrido lineal de f0 a f1, con ventana Hanning."""
    t = np.arange(int(fs * duracion)) / fs
    fase = 2 * np.pi * (f0 * t + (f1 - f0) / (2 * duracion) * t**2)
    return np.sin(fase) * np.hanning(len(t))


def correlacion_fft(recibida, pulso):
    n_recibida, n_pulso = len(recibida), len(pulso)
    N = n_recibida + n_pulso - 1
    X = np.fft.fft(recibida, N)
    H = np.fft.fft(pulso[::-1], N)
    conv = np.fft.ifft(X * H).real
    inicio = n_pulso - 1
    return conv[inicio:inicio + (n_recibida - n_pulso + 1)]


def envolvente(x):
    """anular frecuencias negativas."""
    N = len(x)
    X = np.fft.fft(x)
    h = np.zeros(N)
    h[0] = 1
    if N % 2 == 0:
        h[N // 2] = 1
        h[1:N // 2] = 2
    else:
        h[1:(N + 1) // 2] = 2
    return np.abs(np.fft.ifft(X * h))


def medir(chirp):
    """Reproduce y graba al mismo tiempoe en base al mismo reloj de la tarjeta de sonido."""
    salida = np.zeros(int(FS * DUR_GRABACION), dtype=np.float32)
    salida[:len(chirp)] = VOLUMEN * chirp
    grabacion = sd.playrec(salida, samplerate=FS, channels=1, dtype='float32')
    sd.wait()
    return grabacion[:, 0]


def muestras_por_distancia(d):
    return int(round(2 * d / V_SONIDO * FS))


# --- Medición ---
chirp = generar_chirp(FS, DUR_CHIRP, F_INICIO, F_FIN)
grabacion = medir(chirp)

if np.max(np.abs(grabacion)) > 0.99:
    print("Advertencia: la grabación se saturó, bajar VOLUMEN")

env = envolvente(correlacion_fft(grabacion, chirp))
env = env / np.max(env)

k_directo = int(np.argmax(env))
inicio = k_directo + muestras_por_distancia(D_MIN)
fin = min(k_directo + muestras_por_distancia(D_MAX), len(env))
k_eco = inicio + int(np.argmax(env[inicio:fin]))

delta_t = (k_eco - k_directo) / FS
distancia = V_SONIDO * delta_t / 2 + D_PARLANTE_MIC / 2

print(f"Camino directo en la muestra {k_directo}, eco en la muestra {k_eco}")
print(f"Δt = {delta_t * 1000:.3f} ms  ->  distancia estimada = {distancia:.3f} m")

# --- Gráficas ---
t_ms = np.arange(len(grabacion)) / FS * 1000
d_eje = (np.arange(len(env)) - k_directo) / FS * V_SONIDO / 2 + D_PARLANTE_MIC / 2

fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(9, 6))

ax1.plot(t_ms, grabacion, color="#2a78d6", linewidth=0.8)
ax1.set_xlabel("Tiempo (ms)")
ax1.set_ylabel("Amplitud")
ax1.set_title("Grabación del micrófono")

ax2.plot(d_eje, env, color="#2a78d6", linewidth=1.2)
ax2.axvspan(d_eje[inicio], d_eje[fin - 1], color="#e1e0d9", alpha=0.5,
            label="Ventana de búsqueda del eco")
ax2.axvline(d_eje[k_directo], color="#1baf7a", linestyle="--", label="Camino directo")
ax2.axvline(distancia, color="#eb6834", linestyle="--",
            label=f"Eco detectado ({distancia:.2f} m)")
ax2.set_xlim(-0.1, D_MAX + 0.3)
ax2.set_xlabel("Distancia equivalente (m)")
ax2.set_ylabel("Envolvente de la correlación")
ax2.legend(frameon=False)

for ax in (ax1, ax2):
    ax.grid(True, linewidth=0.5, color="#e1e0d9", alpha=0.7)

fig.tight_layout()
plt.show()