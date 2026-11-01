"""Check of the mean-field Girsanov weights on a 1D periodic cold electrostatic plasma.

Units: omega_p = k_p = 1, nonrelativistic electrons, fixed ion background.
Noisy leapfrog: u^{n+1/2} = u^{n-1/2} + dt a(x^n) + sigma sqrt(dt) zeta, x^{n+1} = x^n + dt u^{n+1/2}.
Parameter theta = amplitude of an external driver E_ext = theta sin(x) exp(-(t-t0)^2/tau^2).
Weights: omega += delta_a . zeta sqrt(dt) / sigma, delta_a = -(delta_E + dE_ext/dtheta).
Variant 'center': subtract the shape-weighted local mean of the increments (charge-neutral increments).
Usage: python -I verify_mf_girsanov.py N sigma theta variant nseeds
"""
import sys, json
import numpy as np

L = 2 * np.pi; NG = 64; DX = L / NG; DT = 0.05; T = 30.0; T0, TAU = 5.0, 2.0
K = 2 * np.pi * np.fft.rfftfreq(NG, DX)


def shape(x):
    s = x / DX; i = np.floor(s).astype(np.int64); d = s - i
    return i % NG, (i + 1) % NG, d


def deposit(i0, i1, d, q):
    return np.bincount(i0, q * (1 - d), NG) + np.bincount(i1, q * d, NG)


def efield(rho):
    rk = np.fft.rfft(rho); ek = np.zeros_like(rk); ek[1:] = rk[1:] / (1j * K[1:])
    return np.fft.irfft(ek, NG)


def gather(f, i0, i1, d):
    return f[i0] * (1 - d) + f[i1] * d


def run(N, sigma, theta, seed, tangent=True, variant="plain"):
    rng = np.random.default_rng(seed)
    x = (np.arange(N) + 0.5) * L / N; u = np.zeros(N); om = np.zeros(N)
    w = L / N  # n0 = 1 per unit length
    nt = int(round(T / DT)); varw = []
    for n in range(nt):
        t = n * DT; env = np.exp(-((t - T0) / TAU) ** 2)
        i0, i1, d = shape(x)
        ne = deposit(i0, i1, d, np.full(N, w)) / DX
        E = efield(1.0 - ne)
        a = -(gather(E, i0, i1, d) + theta * np.sin(x) * env)
        z = rng.standard_normal(N)
        if tangent:
            dne = deposit(i0, i1, d, w * om) / DX
            dE = efield(-dne)
            da = -(gather(dE, i0, i1, d) + np.sin(x) * env)
            inc = da * z * np.sqrt(DT) / sigma
            if variant == "center":
                num = deposit(i0, i1, d, w * inc); den = deposit(i0, i1, d, np.full(N, w))
                inc = inc - gather(num / np.maximum(den, 1e-30), i0, i1, d)
            om = om + inc
            if (n + 1) % 60 == 0:
                varw.append(float(om.var()))
        u = u + DT * a + sigma * np.sqrt(DT) * z
        x = (x + DT * u) % L
    g = np.stack([u ** 2, u * np.sin(x), np.cos(x)])  # observables per particle
    J = g.mean(1)
    grad = (om[None, :] * (g - J[:, None])).mean(1) if tangent else None
    return J, grad, varw


if __name__ == "__main__":
    N = int(sys.argv[1]); sigma = float(sys.argv[2]); theta = float(sys.argv[3]); variant = sys.argv[4]; M = int(sys.argv[5])
    h = 0.02 * max(theta, 0.1)
    G = []; FD = []; VW = []
    for s in range(M):
        J, gr, vw = run(N, sigma, theta, 1000 + s, True, variant)
        Jp, _, _ = run(N, sigma, theta + h, 1000 + s, False)
        Jm, _, _ = run(N, sigma, theta - h, 1000 + s, False)
        G.append(gr); FD.append((Jp - Jm) / (2 * h)); VW.append(vw)
    G = np.array(G); FD = np.array(FD); VW = np.array(VW)
    out = dict(N=N, sigma=sigma, theta=theta, variant=variant, M=M,
               lr_mean=G.mean(0).tolist(), lr_se=(G.std(0, ddof=1) / np.sqrt(M)).tolist(),
               fd_mean=FD.mean(0).tolist(), fd_se=(FD.std(0, ddof=1) / np.sqrt(M)).tolist(),
               var_omega_t=np.exp(np.log(VW).mean(0)).tolist())
    print(json.dumps(out))
