import numpy as np
import matplotlib.pyplot as plt
from accurate_solver import AccurateSolver
plt.style.use('sciart.mplstyle')
# Входные параметры
Lx = 1.0
Nv = 200
t_k = 0.2
x_p = 0.0
Co = 0.1
k = 1.4
d = 0.5
v_L = 0.0
v_R = 0.0

# Начальные состояния газа
rho_L = 1.0
rho_R = 0.125
p_L = 1.0
p_R = 0.1
u_L = 0.0
u_R = 0.0

x = np.linspace(-0.5, 0.5, Nv + 1) # Координаты границ ячеек
dx0 = x[1] - x[0]
x_centr = np.linspace(-0.5 + 0.5*dx0, 0.5 - 0.5*dx0, Nv) # Координаты центров ячеек

# Инициализация параметров газа
p    = np.zeros(Nv)     # Давление газа
rho  = np.zeros(Nv)     # Плотность газа
u    = np.zeros(Nv + 1) # Скорость интерфейсов
E    = np.zeros(Nv)     # Полная удельная энергия газа
eps  = np.zeros(Nv)     # Удельная внутренняя энергия газа
ksi   = np.zeros(Nv)     # Массовая лагранжева координата
ksi_p = np.zeros(Nv + 1) # Масса перегородок
u_cell  = np.zeros(Nv)     # Скорость газа в узле
c  = np.zeros(Nv)     # Скорость звука в газе

A = np.pi*(d**2)/4  # Площадь поперечного сечения трубы

# Запись начальных параметров газа в массивы
for i in range(Nv):
    if x_centr[i] <= x_p:
        p[i] = p_L
        rho[i] = rho_L
        u_cell[i] = u_L
        u[i] = u_L
        u[i+1] = u_L
    else:
        p[i] = p_R
        rho[i] = rho_R
        u_cell[i] = u_R
        u[i] = u_R
        u[i+1] = u_R

c = np.sqrt(k*p/rho)
eps = p/ ((k - 1) * rho)
E = eps + 0.5 * u_cell**2
ksi = rho * A * dx0

step = 0
t = 0.0

# Цикл интегрирования
while t < t_k:
    step += 1
    x_old       = x.copy()
    u_old       = u.copy()
    p_old       = p.copy()
    rho_old     = rho.copy()
    eps_old     = eps.copy()

    c = np.sqrt((k * (p_old / rho_old)))
    u_cell = (u_old[1:] + u_old[:-1]) / 2
    dx = x_old[1:] - x_old[:-1]
    dt = Co * np.min(dx / (c + np.abs(u_cell)))

    for i in range(1, Nv):
        u_new = u_old[i] - dt * A * (p_old[i] - p_old[i-1]) / (
            0.5 * (ksi[i] + ksi[i-1]) + ksi_p[i])
        u[i] = u_new

    u[0] = u[1]
    u[Nv] = u[Nv-1]

    for i in range(Nv+1):
        x_new = x_old[i] + dt*u[i]
        x[i] = x_new

    for i in range(Nv):
        dx = x[i+1] - x[i]
        rho_new = (1 / A) * (ksi[i] / dx)
        rho[i] = rho_new

    for i in range(Nv):
        du = u[i+1] - u[i]
        Z = 0.5 * dt * A * du / ksi[i]

        eps_new = (eps_old[i] - Z * p_old[i]) / (1 + Z * (k - 1) * rho[i])
        eps[i] = eps_new

        p_new = (k - 1) * eps[i] * rho[i]
        p[i] = p_new

    t += dt

x_centers = (x[:-1] + x[1:]) / 2

accurate_solve = AccurateSolver(
    x = x_centers,
    t = t_k,
    rho = (rho_L, rho_R),
    p = (p_L, p_R),
    gamma = k
)

rho_rel = rho/rho_L
p_rel   = p/p_L



exact = accurate_solve.solve(relative=False)
p_exact   = exact["pressure"]
rho_exact = exact["density"]
u_exact   = exact["speed"]
eps_exact = p_exact / ((k - 1)*rho_exact)

exact = accurate_solve.solve(relative=True)
p_exact = exact["pressure"]
rho_exact = exact["density"]


eps_ex_max = np.max((eps_exact))
eps_exact_rel = eps_exact/eps_ex_max
eps_max = np.max(eps)
u_max   = np.max(np.abs(u_cell))
u_rel   = u_cell/u_max
eps_rel = eps/eps_max

# Создание четырех графиков
fig, axes = plt.subplots(2, 2, figsize=(12, 10))

# 1. Давление
axes[0, 0].plot(x_centers, p, 'r-', lw=1.5, label='Схема фон Неймана')
axes[0, 0].plot(x_centers, p_exact, 'k--', lw=1, label='Точное решение')
axes[0, 0].set_ylabel('$p$')
axes[0, 0].set_title('Давление')
axes[0, 0].legend()
axes[0, 0].grid(True)

# 2. Плотность
axes[0, 1].plot(x_centers, rho, 'b-', lw=1.5, label='Схема фон Неймана')
axes[0, 1].plot(x_centers, rho_exact, 'k--', lw=1, label='Точное решение')
axes[0, 1].set_ylabel(r'ρ, $кг/м^{3}$')
axes[0, 1].set_title('Плотность')
axes[0, 1].legend()
axes[0, 1].grid(True)

# 3. Скорость
axes[1, 0].plot(x_centers, u_cell, 'g-', lw=1.5, label='Схема фон Неймана')
axes[1, 0].plot(x_centers, u_exact, 'k--', lw=1, label='Точное решение')
axes[1, 0].set_xlabel('x')
axes[1, 0].set_ylabel('$u$, $м\с$')
axes[1, 0].set_title('Скорость')
axes[1, 0].legend()
axes[1, 0].grid(True)

# 4. Скорость звука
axes[1, 1].plot(x_centers, eps, 'm-', lw=1.5, label='Схема фон Неймана')
axes[1, 1].plot(x_centers, eps_exact, 'k--', lw=1, label='Точное решение')
axes[1, 1].set_xlabel('x')
axes[1, 1].set_ylabel('$e$, Дж/кг')
axes[1, 1].set_title('Внутренняя энергия')
axes[1, 1].legend()
axes[1, 1].grid(True)

plt.suptitle(f'Распад разрыва, $N_V={Nv}$, $t={t_k}$, $Co={Co}$', fontsize=14)
plt.tight_layout()
plt.savefig('sod_results.png', dpi=150)
plt.show()