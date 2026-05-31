import numpy as np
import matplotlib.pyplot as plt
from accurate_solver import AccurateSolver
from errors import l1_error, l2_error, inf_error
plt.style.use('sciart.mplstyle')
# Входные параметры
Lx = 1.0
Nv = 1000
t_k = 0.15
x_p = 0.0
Co = 0.9
k = 1.4
d = 0.5
v_L = 0.0
v_R = 0.0
W = 1.5
h = 15
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

    if t > 0:
        dt_old = dt
        dt = Co * np.min(dx / (c + np.abs(u_cell)))  # Обновляем шаг по времени
        if dt > W*dt_old:
            dt = W*dt_old
    else:
        dt = Co * np.min(dx / (c + np.abs(u_cell)))

    for i in range(1, Nv):
        u_new = u_old[i] - dt * A * (p_old[i] - p_old[i-1]) / (
            0.5 * (ksi[i] + ksi[i-1]) + ksi_p[i])
        u[i] = u_new

    u[0] = u[1]
    u[Nv] = u[Nv-1]

    for i in range(1,Nv-1):
        u_new = (1/(h + 2))*(u[i-1] + h*u[i] + u[i+1])
        u[i] = u_new

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

exact = accurate_solve.solve(relative=True)
p_exact   = exact["pressure"]
rho_exact = exact["density"]

rho_rel = rho/rho_L
p_rel   = p/p_L

fig, ax = plt.subplots()
ax.plot(x_centers, p, linestyle = '--', label='Давление')
ax.plot(x_centers, rho, linestyle = '--', label='Плотность')
ax.plot(x_centers, p_exact, label='Точное давление' )
ax.plot(x_centers, rho_exact, label='Точная плотность')
ax.set_title(f'$h={h}$, $t$={t_k}')
ax.set_ylabel('$Параметры$ $газа$')
ax.set_xlabel('$x$')
fig.tight_layout()
plt.legend()
plt.grid(True)
fig.savefig('s_filtrom.svg')
plt.show()

dx = np.max(dx)

# Выводим все ошибки
err_rho_l1 = l1_error(rho, rho_exact, dx0)
err_p_l1 = l1_error(p, p_exact, dx0)
err_rho_l2 = l2_error(rho, rho_exact, dx0)
err_p_l2 = l2_error(p, p_exact, dx0)
err_rho_max = inf_error(rho, rho_exact)
err_p_max = inf_error(p, p_exact)

print("Тип ошибки\tПлотность\tДавление")
print(f"L_max\t{err_rho_max:.4e}\t{err_p_max:.4e}")
print(f"L1\t{err_rho_l1:.4e}\t{err_p_l1:.4e}")
print(f"L2\t{err_rho_l2:.4e}\t{err_p_l2:.4e}")