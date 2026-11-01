"""R1 pilot: 1D-1V electrostatic Debye-shielding PIC of Chung et al. (2020), Sec. 4.

Normalized units: x in lambda_D, t in 1/omega_p, v in v0 = lambda_D*omega_p,
density in n_e0, phi in k_B T0/e. Electron q/m = -1. Quadratic B-spline (b_2)
deposit/gather (paper: S_x = b_1, interpolation b_{l+1} = b_2), 3-point Poisson
via FFT, centered-difference E, velocity-Verlet leapfrog.
Modes: base run (J_D), twin-run divergence (Delta theta), CRN secant.
"""
import sys, time, json
import numpy as np

L, NM, DT, TF = 20.0, 64, 0.05, 150.0
DX = L / NM
XC, WQ = L / 2, 0.1 * L
xg = np.arange(NM) * DX
rho_ext = -2.0 / L + 2.0 / (np.sqrt(2 * np.pi) * WQ) * np.exp(-(xg - XC) ** 2 / (2 * WQ ** 2))
k = 2 * np.pi * np.fft.rfftfreq(NM, d=DX)
lap = -(2 - 2 * np.cos(k * DX)) / DX ** 2  # 3-point Laplacian symbol
lap[0] = 1.0


def weights(x):
    s = x / DX
    i = np.floor(s + 0.5).astype(np.int64)  # nearest node
    d = s - i
    w = np.stack([0.5 * (0.5 - d) ** 2, 0.75 - d ** 2, 0.5 * (0.5 + d) ** 2])
    idx = np.stack([i - 1, i, i + 1]) % NM
    return idx, w


def efield(x, W):
    idx, w = weights(x)
    ne = np.bincount(idx.ravel(), (w * W).ravel(), NM) / DX
    rho = 1.0 - ne + rho_ext
    rk = np.fft.rfft(rho)
    phik = -rk / lap
    phik[0] = 0
    phi = np.fft.irfft(phik, NM)
    E = -(np.roll(phi, -1) - np.roll(phi, 1)) / (2 * DX)
    return idx, w, E, phi


def run(theta, N, seed, tf=TF, xs=None, record=False):
    rng = np.random.default_rng(seed)
    x = rng.uniform(0, L, N)
    g = rng.standard_normal(N)
    v = theta * g
    W = L / N
    nt = int(round(tf / DT))
    idx, w, E, phi = efield(x, W)
    a = -(E[idx] * w).sum(0)
    J = 0.0
    hist = []
    for n in range(nt):
        v += 0.5 * DT * a
        x = (x + DT * v) % L
        idx, w, E, phi = efield(x, W)
        a = -(E[idx] * w).sum(0)
        v += 0.5 * DT * a
        Jk = W * np.sum((x - XC) ** 2)
        J += Jk
        if record and (n % 20 == 0):
            hist.append((n * DT + DT, Jk, x.copy()))
    return J / nt, phi, hist


if __name__ == "__main__":
    mode = sys.argv[1]
    if mode == "base":
        N, seed = int(sys.argv[2]), int(sys.argv[3])
        t0 = time.time()
        J, phi, _ = run(1.5, N, seed)
        print(json.dumps({"J": J, "phi_max": float(phi.max() - phi.min()), "time": time.time() - t0}))
    elif mode == "phiext":
        rk = np.fft.rfft(rho_ext); pk = -rk / lap; pk[0] = 0
        p = np.fft.irfft(pk, NM); print("phi_ext peak-to-min", p.max() - p.min())
        print("Jmax", L ** 3 / 12, "Jmin", L ** 3 / 12 - L ** 2 / 6 + 2 * WQ ** 2)
    elif mode == "twin":
        N, seed, dth = int(sys.argv[2]), int(sys.argv[3]), float(sys.argv[4])
        _, _, h0 = run(1.5, N, seed, tf=float(sys.argv[5]), record=True)
        _, _, h1 = run(1.5 + dth, N, seed, tf=float(sys.argv[5]), record=True)
        out = []
        for (t, J0, x0), (_, J1, x1) in zip(h0, h1):
            dx = (x1 - x0 + L / 2) % L - L / 2
            out.append((t, float(np.mean(np.abs(dx))), float((J1 - J0) / dth)))
        print(json.dumps(out))
    elif mode == "secant":
        N, seed, h = int(sys.argv[2]), int(sys.argv[3]), float(sys.argv[4])
        Jp, _, _ = run(1.5 + h, N, seed); Jm, _, _ = run(1.5 - h, N, seed)
        print(json.dumps({"seed": seed, "h": h, "sec": (Jp - Jm) / (2 * h), "Jp": Jp, "Jm": Jm}))
