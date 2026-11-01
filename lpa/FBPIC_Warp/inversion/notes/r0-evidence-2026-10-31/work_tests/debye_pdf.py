"""R1 pilot part 2: particle-pdf (non-collocated, Chung et al. Sec. 3.2.5, eqs. 51-54)
and a semi-Lagrangian Vlasov continuum reference for the Debye case.

Same normalization as debye_pilot.py. theta = v_T/v0 = 1.5.
"""
import sys, os, time, json
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
from debye_pilot import L, NM, DT, TF, DX, XC, rho_ext, lap, weights, efield

VMAX, NV = 9.0, 64
DV = 2 * VMAX / NV
vg = -VMAX + (np.arange(NV) + 0.5) * DV  # v-cell centers


def b3p(z):
    """Derivative of the cubic B-spline b3 (support |z|<2) w.r.t. z."""
    a = np.abs(z); s = np.sign(z)
    out = np.where(a < 1, -2 * a + 1.5 * a * a, np.where(a < 2, -0.5 * (2 - a) ** 2, 0.0))
    return s * out


def vweights2(v):
    s = (v + VMAX) / DV - 0.5
    j = np.floor(s + 0.5).astype(np.int64); d = s - j
    w = np.stack([0.5 * (0.5 - d) ** 2, 0.75 - d ** 2, 0.5 * (0.5 + d) ** 2])
    jj = np.stack([j - 1, j, j + 1])
    ok = (jj >= 0) & (jj < NV)
    return np.clip(jj, 0, NV - 1), w * ok


def vgrad3(v):
    """Weights c_j such that d/dv [sum_p S_v(v-v_p)] at v_j = sum_p c_j/DV, S_v = cubic."""
    s = (v + VMAX) / DV - 0.5
    j0 = np.floor(s).astype(np.int64)
    jj = np.stack([j0 - 1, j0, j0 + 1, j0 + 2])
    z = jj - s  # (v_j - v_p)/DV
    w = b3p(z) / DV
    ok = (jj >= 0) & (jj < NV)
    return np.clip(jj, 0, NV - 1), w * ok


def ps_deposit(ix, wx, jv, wv, val):
    """Deposit val*wx*wv onto the (NM, NV) phase-space mesh."""
    flat = (ix[:, None, :] * NV + jv[None, :, :])
    ww = wx[:, None, :] * wv[None, :, :] * val
    return np.bincount(flat.ravel(), ww.ravel(), NM * NV).reshape(NM, NV)


def ps_gather(G, ix, wx, jv, wv):
    return (G[ix[:, None, :], jv[None, :, :]] * wx[:, None, :] * wv[None, :, :]).sum((0, 1))


def dfield(xs, Ws):
    idx, w = weights(xs)
    dne = np.bincount(idx.ravel(), (w * Ws).ravel(), NM) / DX
    drho = -dne
    pk = -np.fft.rfft(drho) / lap; pk[0] = 0
    dphi = np.fft.irfft(pk, NM)
    return -(np.roll(dphi, -1) - np.roll(dphi, 1)) / (2 * DX)


def run_pdf(theta, N, M, seed, tf=TF, collocated=False):
    rng = np.random.default_rng(seed)
    x = rng.uniform(0, L, N); g = rng.standard_normal(N); v = theta * g
    W = L / N
    if collocated:
        xs, vs = x, v  # same arrays move together
        Ws = W * (g * g - 1) / theta  # score-function weights: df0/dtheta / f0 * W
        M = N
    else:
        xs = rng.uniform(0, L, M); vs = rng.uniform(-VMAX, VMAX, M)
        f0 = np.exp(-vs ** 2 / (2 * theta ** 2)) / (np.sqrt(2 * np.pi) * theta)
        Ws = (vs ** 2 / theta ** 2 - 1) / theta * f0 * (L * 2 * VMAX / M)
    nt = int(round(tf / DT))
    idx, w, E, _ = efield(x, W)
    a = -(E[idx] * w).sum(0)
    if not collocated:
        ids, wsx, _, _ = efield(xs, 0.0)[0], weights(xs)[1], None, None
        ids, wsx = weights(xs)
        as_ = -(E[ids] * wsx).sum(0)
    J = 0.0; dJ = 0.0
    t0 = time.time()
    for n in range(nt):
        # --- plasma push (velocity Verlet) ---
        v += 0.5 * DT * a
        x[:] = (x + DT * v) % L
        if not collocated:
            vs += 0.5 * DT * as_
            xs[:] = (xs + DT * vs) % L
        idx, w, E, _ = efield(x, W)
        a = -(E[idx] * w).sum(0)
        v += 0.5 * DT * a
        if not collocated:
            ids, wsx = weights(xs)
            as_ = -(E[ids] * wsx).sum(0)
            vs += 0.5 * DT * as_
        else:
            ids, wsx = idx, w
        # --- sensitivity field from weights ---
        dE = dfield(xs, Ws)
        # --- source H = (q/m) dE * df/dv on phase-space mesh (eq. 52) ---
        jg, wg = vgrad3(v)
        dfdv = ps_deposit(idx, w, jg, wg, W / (DX * DV))
        H = -dE[:, None] * dfdv
        # --- n-hat of sensitivity particles (53a) and h_s (53b) ---
        jv2, wv2 = vweights2(vs)
        nhat = ps_deposit(ids, wsx, jv2, wv2, 1.0 / (DX * DV))
        hm = np.where(nhat > 0, H / np.maximum(nhat, 1e-300), 0.0)
        hs = ps_gather(hm, ids, wsx, jv2, wv2)
        Ws = Ws - DT * hs
        J += W * np.sum((x - XC) ** 2)
        dJ += np.sum(Ws * (xs - XC) ** 2)
    return J / nt, dJ / nt, time.time() - t0


# ---------------- semi-Lagrangian Vlasov reference -----------------
def run_vlasov(theta, NX=256, NVv=512, VM=12.0, tf=TF, dt=DT):
    dx = L / NX; xv = np.arange(NX) * dx
    dv = 2 * VM / NVv; vv = -VM + (np.arange(NVv) + 0.5) * dv
    f = np.exp(-vv ** 2 / (2 * theta ** 2))[None, :] / (np.sqrt(2 * np.pi) * theta) * np.ones((NX, 1))
    kx = 2 * np.pi * np.fft.fftfreq(NX, d=dx)
    kv = 2 * np.pi * np.fft.fftfreq(NVv, d=dv)
    rex = -2.0 / L + 2.0 / (np.sqrt(2 * np.pi) * WQ_) * np.exp(-(xv - XC) ** 2 / (2 * WQ_ ** 2))
    def Ef(f):
        rho = 1.0 - f.sum(1) * dv + rex
        rk = np.fft.fft(rho); ek = np.zeros_like(rk)
        ek[1:] = rk[1:] / (1j * kx[1:])
        return np.real(np.fft.ifft(ek))
    def xshift(f, tau):
        return np.real(np.fft.ifft(np.fft.fft(f, axis=0) * np.exp(-1j * kx[:, None] * vv[None, :] * tau), axis=0))
    def vshift(f, acc, tau):
        return np.real(np.fft.ifft(np.fft.fft(f, axis=1) * np.exp(-1j * kv[None, :] * acc[:, None] * tau), axis=1))
    nt = int(round(tf / dt)); J = 0.0
    w2 = (xv - XC) ** 2
    f = xshift(f, 0.5 * dt)
    for n in range(nt):
        E = Ef(f)
        f = vshift(f, -E, dt)
        f = xshift(f, dt)
        # J at x-half-step-shifted f is O(dt^2) offset; fine for a slope reference
        J += np.sum(w2 * f.sum(1) * dv) * dx
    return J / nt

WQ_ = 0.1 * L

if __name__ == "__main__":
    mode = sys.argv[1]
    if mode == "pdf":
        N, M, seed, col = int(sys.argv[2]), int(sys.argv[3]), int(sys.argv[4]), sys.argv[5] == "col"
        J, dJ, t = run_pdf(1.5, N, M, seed, collocated=col)
        print(json.dumps({"N": N, "M": M, "seed": seed, "col": col, "J": J, "dJ": dJ, "time": t}))
    elif mode == "vlasov":
        h = float(sys.argv[2]); NX = int(sys.argv[3]); NVv = int(sys.argv[4])
        t0 = time.time()
        Jp = run_vlasov(1.5 + h, NX, NVv); Jm = run_vlasov(1.5 - h, NX, NVv); J0 = run_vlasov(1.5, NX, NVv)
        print(json.dumps({"h": h, "NX": NX, "NV": NVv, "J0": J0, "dJ": (Jp - Jm) / (2 * h), "time": time.time() - t0}))
