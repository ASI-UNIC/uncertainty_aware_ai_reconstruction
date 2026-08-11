# We will generate three separate plots:
# 1) Fixed point attractor
# 2) Limit cycle attractor
# 3) Strange attractor (Lorenz system projection)

import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp

# -----------------------------
# 1️⃣ Fixed Point Attractor
# dx/dt = -x  → decays to 0
# -----------------------------
def fixed_point(t, x):
    return -x

t_span = (0, 10)
t_eval = np.linspace(0, 10, 1000)
sol_fixed = solve_ivp(fixed_point, t_span, [5], t_eval=t_eval)

plt.figure()
plt.plot(sol_fixed.y[0], np.zeros_like(sol_fixed.y[0]))
plt.title("Fixed Point Attractor (All trajectories go to 0)")
plt.xlabel("x(t)")
plt.ylabel("")
plt.show()


# -----------------------------
# 2️⃣ Limit Cycle Attractor
# Van der Pol Oscillator
# -----------------------------
def vdp(t, z, mu=1.0):
    x, y = z
    dxdt = y
    dydt = mu * (1 - x**2) * y - x
    return [dxdt, dydt]

sol_vdp = solve_ivp(vdp, (0, 40), [2, 0], t_eval=np.linspace(0, 40, 5000))

plt.figure()
plt.plot(sol_vdp.y[0], sol_vdp.y[1])
plt.title("Limit Cycle Attractor (Van der Pol Oscillator)")
plt.xlabel("x")
plt.ylabel("y")
plt.show()


# -----------------------------
# 3️⃣ Strange Attractor
# Lorenz System (x vs y projection)
# -----------------------------
def lorenz(t, state, sigma=10, beta=8/3, rho=28):
    x, y, z = state
    dx = sigma * (y - x)
    dy = x * (rho - z) - y
    dz = x * y - beta * z
    return [dx, dy, dz]

sol_lorenz = solve_ivp(lorenz, (0, 40), [1, 1, 1], t_eval=np.linspace(0, 40, 10000))

plt.figure()
plt.plot(sol_lorenz.y[0], sol_lorenz.y[1])
plt.title("Strange Attractor (Lorenz System Projection x vs y)")
plt.xlabel("x")
plt.ylabel("y")
plt.show()
