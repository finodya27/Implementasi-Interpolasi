import numpy as np
import matplotlib.pyplot as plt

# Data dari gambar
x_data = np.array([5, 10, 15, 20, 25, 30, 35, 40])
y_data = np.array([40, 30, 25, 40, 18, 20, 22, 15])

# Interpolasi Polinom Lagrange
def interpolasi_lagrange(x, x_titik, y_titik):
    def L(k, x):
        term = [(x - x_titik[j]) / (x_titik[k] - x_titik[j]) for j in range(len(x_titik)) if j != k]
        return np.prod(term, axis=0)
    return np.sum([y_titik[k] * L(k, x) for k in range(len(x_titik))], axis=0)

# Interpolasi Polinom Newton
def interpolasi_newton(x, x_titik, y_titik):
    def beda_terbagi(x_titik, y_titik):
        n = len(y_titik)
        coef = np.zeros([n, n])
        coef[:,0] = y_titik
        for j in range(1,n):
            for i in range(n-j):
                coef[i][j] = (coef[i+1][j-1] - coef[i][j-1]) / (x_titik[i+j] - x_titik[i])
        return coef[0,:]

    koef = beda_terbagi(x_titik, y_titik)
    n = len(koef)
    hasil = koef[0]
    for i in range(1, n):
        term = koef[i]
        for j in range(i):
            term *= (x - x_titik[j])
        hasil += term
    return hasil

# Menguji fungsi dan plot hasilnya
x_range = np.linspace(5, 40, 100)
y_lagrange = interpolasi_lagrange(x_range, x_data, y_data)
y_newton = interpolasi_newton(x_range, x_data, y_data)

# Membuat grafik hasil interpolasi
plt.figure(figsize=(10, 6))
plt.plot(x_data, y_data, 'o', label='Titik Data')
plt.plot(x_range, y_lagrange, '-', label='Interpolasi Lagrange')
plt.plot(x_range, y_newton, '--', label='Interpolasi Newton')
plt.xlabel('Tegangan, x (kg/mm²)')
plt.ylabel('Waktu patah, y (jam)')
plt.title('Interpolasi Waktu Patah vs. Tegangan')
plt.legend()
plt.grid(True)
plt.show()
