"""Does the feedback control (sec_theory eq. 2.6) remove the laser quiver term from the weights?

One electron (q=-1, c=omega_p=1) at rest, hit by a linearly polarized laser pulse
a_x = a0 exp(-x^2/w0^2) exp(-(psi-psi_c)^2/tau^2) cos(w0l psi), psi = t - z.
E_x = -d_t a_x, B_y = d_z a_x (paraxial; E_z neglected). Force F = -(E + v x B).
theta = ln E_L: delta_a = F/2 at fixed (x,u).
Controls: pure Girsanov h = delta_a; feedback (2.6) h = (dF/dx + wc^2 M^-1) xi_x + (dF/du + 2 wc) xi_u;
damping-only h = wc^2 M^-1 xi_x + 2 wc xi_u (keeps the true tangent dynamics); pathwise h = 0.
Reports int |h|^2 dt (weight variance times sigma^2) and the net momentum tangent after the pulse.
Usage: python -I laser_feedback.py a0 w0l wc
"""
import sys
import numpy as np
import os
NPC = int(os.environ.get('NPC', '80')); MODES = os.environ.get('MODES', 'path,lr,fb,damp').split(',')

a0 = float(sys.argv[1]); WL = float(sys.argv[2]); WC = float(sys.argv[3])
TAU = 2.55; PSIC = 3 * TAU; W0 = 4.0; X0 = 1.0


def fields(r, t):
    x, y, z = r
    psi = t - z
    env = np.exp(-x ** 2 / W0 ** 2) * np.exp(-((psi - PSIC) / TAU) ** 2)
    denv = -2 * (psi - PSIC) / TAU ** 2 * env      # d env / d psi
    c, s = np.cos(WL * psi), np.sin(WL * psi)
    da_dpsi = a0 * (denv * c - WL * env * s)
    Ex = -da_dpsi            # -d_t a, d_t = d_psi
    By = -da_dpsi            # d_z a = -d_psi a
    return Ex, By


def force(r, u, t):
    g = np.sqrt(1 + u @ u); v = u / g
    Ex, By = fields(r, t)
    return np.array([-Ex + v[2] * By, 0.0, -v[0] * By])


def jac(f, x0, eps=1e-6):
    J = np.zeros((3, 3))
    for k in range(3):
        e = np.zeros(3); e[k] = eps
        J[:, k] = (f(x0 + e) - f(x0 - e)) / (2 * eps)
    return J


def rhs(t, s, mode):
    r, u, xr, xu, I = s[0:3], s[3:6], s[6:9], s[9:12], s[12]
    g = np.sqrt(1 + u @ u); v = u / g
    M = (np.eye(3) - np.outer(v, v)) / g
    F = force(r, u, t)
    da = 0.5 * F
    Fx = jac(lambda rr: force(rr, u, t), r); Fu = jac(lambda uu: force(r, uu, t), u)
    Minv = np.linalg.inv(M)
    if mode == "lr":
        h = da + Fx @ xr + Fu @ xu        # keeps xi = 0
    elif mode == "fb":
        h = (Fx + WC ** 2 * Minv) @ xr + (Fu + 2 * WC * np.eye(3)) @ xu
    elif mode == "damp":
        h = WC ** 2 * Minv @ xr + 2 * WC * xu
    else:
        h = np.zeros(3)
    dxr = M @ xu
    dxu = Fx @ xr + Fu @ xu + da - h
    return np.concatenate([v, F, dxr, dxu, [h @ h]])



def run(mode):
    s = np.zeros(13); s[0] = X0
    dt = 2 * np.pi / WL / NPC; t = 0.0
    while True:
        psi = t - s[2]
        if psi > 2 * PSIC: break
        k1 = rhs(t, s, mode); k2 = rhs(t + dt / 2, s + dt / 2 * k1, mode)
        k3 = rhs(t + dt / 2, s + dt / 2 * k2, mode); k4 = rhs(t + dt, s + dt * k3, mode)
        s = s + dt / 6 * (k1 + 2 * k2 + 2 * k3 + k4); t += dt
    return s, t


for mode in MODES:
    s, t = run(mode)
    print(f"a0={a0} w0={WL} wc={WC} {mode:5s} T={t:6.2f} int|h|^2={s[12]:10.4g} xi_x={np.round(s[6:9],4)} xi_u={np.round(s[9:12],4)} u_end={np.round(s[3:6],4)}")
