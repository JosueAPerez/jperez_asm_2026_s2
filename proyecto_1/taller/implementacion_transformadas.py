import cmath

def dft_bruteforce(x):
    N = len(x)
    X = []
    for k in range(N):
        suma = 0j
        for n in range(N):
            suma += x[n]*cmath.exp(-2j*cmath.pi*k*n/N)
        X.append(suma)
    return X

def fft_recursivo(x):
    N = len(x)
    if N <=1:
        return x
    if N%2!=0:
        raise ValueError("N debe ser potencia de 2")

    pares = fft_recursivo(x[0::2])
    impares = fft_recursivo(x[1::2])

    X = [0] * N
    for k in range(N // 2):
        factor = cmath.exp(-2j * cmath.pi * k / N) * impares[k]
        X[k] = pares[k] + factor
        X[k + N//2] = pares[k] - factor
    return X