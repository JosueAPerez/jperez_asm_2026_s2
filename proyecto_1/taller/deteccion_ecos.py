import numpy as np
import matplotlib.pyplot as plt
from experimento_tiempos import medir_tiempo


def generar_pulso(fs, f0, duracion):
    n = int(fs * duracion)
    t = np.arange(n) / fs
    señal = np.sin(2 * np.pi * f0 * t)
    ventana = np.hanning(n)
    return señal * ventana


def generar_señal_con_ecos(pulso, fs, duracion_total, retardos, atenuaciones, ruido_std=0.05):
    n_total = int(fs * duracion_total)
    recibida = np.zeros(n_total)

    for retardo, atenuacion in zip(retardos, atenuaciones):
        inicio = int(retardo * fs)
        fin = inicio + len(pulso)
        if fin > n_total:
            raise ValueError("El retardo es demasiado grande para la duración total")
        recibida[inicio:fin] += atenuacion * pulso

    ruido = np.random.normal(0, ruido_std, n_total)
    return recibida + ruido

def correlacion_directa(recibida, pulso):
    n_recibida = len(recibida)
    n_pulso = len(pulso)
    n_salida = n_recibida - n_pulso + 1
    correlacion = np.zeros(n_salida)
    for k in range(n_salida):
        correlacion[k] = np.dot(recibida[k:k + n_pulso], pulso)
    return correlacion

def correlacion_fft(recibida, pulso):
    n_recibida = len(recibida)
    n_pulso = len(pulso)

    pulso_invertido = pulso[::-1]
    N = n_recibida + n_pulso - 1

    X = np.fft.fft(recibida, N)
    H = np.fft.fft(pulso_invertido, N)
    conv_completa = np.fft.ifft(X * H).real

    inicio = n_pulso - 1
    fin = inicio + (n_recibida - n_pulso + 1)
    return conv_completa[inicio:fin]

fs = 44100 # frecuencia de muestreo (Hz)
f0 = 2000 # frecuencia del pulso (Hz)
duracion_pulso = 0.005
duracion_total = 0.05
v_sonido = 343.0
distancia_real = 1.0
retardo_real = 2 * distancia_real / v_sonido

pulso = generar_pulso(fs, f0, duracion_pulso)
recibida = generar_señal_con_ecos(pulso, fs, duracion_total,retardos=[retardo_real],atenuaciones=[0.6])
recibida_ruidosa = generar_señal_con_ecos(pulso, fs, duracion_total,retardos=[retardo_real],atenuaciones=[0.6],ruido_std=0.8)

t_total = np.arange(len(recibida)) / fs

plt.figure(figsize=(8, 4))
plt.plot(t_total * 1000, recibida, color='#2a78d6', linewidth=1)
plt.xlabel('Tiempo (ms)')
plt.ylabel('Amplitud')
plt.title(f'Señal recibida (eco simulado a {distancia_real} m, retardo real = {retardo_real*1000:.2f} ms)')
plt.grid(True, linewidth=0.5, color='#e1e0d9', alpha=0.7)
plt.tight_layout()
plt.savefig('proyecto_1/taller/figuras/senal_con_eco.png', dpi=150)
plt.show()

plt.figure(figsize=(8, 4))
plt.plot(t_total * 1000, recibida_ruidosa, color='#2a78d6', linewidth=1)
plt.xlabel('Tiempo (ms)')
plt.ylabel('Amplitud')
plt.title('Señal recibida con ruido fuerte')
plt.grid(True, linewidth=0.5, color='#e1e0d9', alpha=0.7)
plt.tight_layout()
plt.savefig('proyecto_1/taller/figuras/senal_con_eco_ruidosa.png', dpi=150)
plt.show()

correlacion = correlacion_directa(recibida_ruidosa, pulso)
correlacion_via_fft = correlacion_fft(recibida_ruidosa, pulso)

print("Coincide con la correlacion directa?", np.allclose(correlacion, correlacion_via_fft))

t_directa = medir_tiempo(lambda s: correlacion_directa(s, pulso), recibida_ruidosa)
t_fft = medir_tiempo(lambda s: correlacion_fft(s, pulso), recibida_ruidosa)

print(f"Tiempo correlacion directa: {t_directa:.6f} s")
print(f"Tiempo correlacion via FFT: {t_fft:.6f} s")

t_correlacion = np.arange(len(correlacion)) / fs

plt.figure(figsize=(8, 4))
plt.plot(t_correlacion * 1000, correlacion, color='#eb6834', linewidth=2, label='Directa')
plt.plot(t_correlacion * 1000, correlacion_via_fft, color='#1baf7a', linewidth=1, linestyle='--', label='Via FFT')
plt.xlabel('Retardo (ms)')
plt.ylabel('Correlación')
plt.title('Correlación: implementación directa vs. vía FFT')
plt.legend(frameon=False)
plt.grid(True, linewidth=0.5, color='#e1e0d9', alpha=0.7)
plt.tight_layout()
plt.savefig('proyecto_1/taller/figuras/correlacion_directa_vs_fft.png', dpi=150)
plt.show()

k_max = np.argmax(correlacion)
retardo_estimado = k_max / fs

print(f"Retardo real: {retardo_real*1000:.3f} ms ({int(retardo_real*fs)} muestras)")
print(f"Retardo estimado: {retardo_estimado*1000:.3f} ms ({k_max} muestras)")

correlacion_np = np.correlate(recibida_ruidosa, pulso, mode='valid')
print("Coincide con numpy?", np.allclose(correlacion, correlacion_np))

plt.figure(figsize=(8, 4))
plt.plot(t_correlacion * 1000, correlacion, color='#eb6834', linewidth=1.5)
plt.axvline(retardo_estimado * 1000, color='#1baf7a', linestyle='--', label=f'Pico detectado ({retardo_estimado*1000:.3f} ms)')
plt.xlabel('Retardo (ms)')
plt.ylabel('Correlación')
plt.title('Correlación directa: pulso conocido vs. señal recibida')
plt.legend(frameon=False)
plt.grid(True, linewidth=0.5, color='#e1e0d9', alpha=0.7)
plt.tight_layout()
plt.savefig('proyecto_1/taller/figuras/correlacion_directa.png', dpi=150)
plt.show()