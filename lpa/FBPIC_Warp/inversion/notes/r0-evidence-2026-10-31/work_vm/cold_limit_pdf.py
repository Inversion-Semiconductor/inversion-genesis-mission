"""Cold-limit test of the Chung et al. (2020) particle-pdf source, 1D-1V.

One step of the sensitivity source for a uniform field perturbation dE = 1
(q = m = dt = 1). Exact linear response: dJ = q * integral(v * df) with
df = -dt (q/m) dE d_v f, so dJ_exact = n (number density).
Discretization: Chung Eqs. (36), (41), (42), (44) with l = 1:
S_x = b1, S_v = b2 for f (cell averages b2 in x, b3 in v);
S_x = S_v = b1 for the marker density n_hat (cell averages b2, b2).
h_s = sum (H / n_hat) b2 b2; cells with n_hat = 0 are dropped (Eq. 43 fails).
"""
import numpy as np

rng = np.random.default_rng(1)


def b2(s):
    a = np.abs(s)
    return np.where(a < 0.5, 0.75 - a**2, np.where(a < 1.5, 0.5 * (1.5 - a) ** 2, 0.0))


def b3p(s):  # derivative of the cubic B-spline b3
    return b2(s + 0.5) - b2(s - 0.5)


def run(v_pic, v_sens, Lx=64, Nx=64, dv=0.1, nv=400, v_lo=-20.0):
    dx = Lx / Nx
    Np = v_pic.size
    x_pic = rng.uniform(0, Lx, Np)
    W = np.full(Np, 1.0 * Lx / Np)  # density n = 1
    collocated = v_sens is None
    if collocated:
        x_s, v_s = x_pic, v_pic
    else:
        x_s = rng.uniform(0, Lx, v_sens.size)
        v_s = v_sens
    vc = v_lo + dv * (np.arange(nv) + 0.5)
    xc = dx * (np.arange(Nx) + 0.5)

    def xw(x):  # 3-point periodic b2 stencil in x
        i0 = np.floor(x / dx - 0.5).astype(int)
        idx = [(i0 + k) % Nx for k in (-1, 0, 1, 2)]
        wts = [b2((x - xc[i % Nx] - dx * np.round((x - xc[i % Nx]) / Lx) * Lx / dx) / dx) for i in idx]
        return idx, wts

    def vw(v, fn, width):
        j0 = np.floor((v - v_lo) / dv - 0.5).astype(int)
        idx = [j0 + k for k in range(-width, width + 2)]
        wts = [fn((vc[np.clip(j, 0, nv - 1)] - v) / dv) for j in idx]
        return idx, wts

    H = np.zeros((Nx, nv))
    nh = np.zeros((Nx, nv))
    ix, wx = xw(x_pic)
    iv, wv = vw(v_pic, b3p, 2)
    for a, ga in zip(ix, wx):
        for b, gb in zip(iv, wv):
            np.add.at(H, (a, np.clip(b, 0, nv - 1)), W * ga * gb / (dx * dv * dv))
    jx, ux = xw(x_s)
    jv, uv = vw(v_s, b2, 1)
    for a, ga in zip(jx, ux):
        for b, gb in zip(jv, uv):
            np.add.at(nh, (a, np.clip(b, 0, nv - 1)), ga * gb / (dx * dv))
    ok = nh > 0
    hmesh = np.where(ok, H / np.where(ok, nh, 1.0), 0.0)
    lost = np.abs(H[~ok]).sum() / np.abs(H).sum()
    hs = np.zeros(x_s.size)
    for a, ga in zip(jx, ux):
        for b, gb in zip(jv, uv):
            hs += hmesh[a, np.clip(b, 0, nv - 1)] * ga * gb
    dW = -hs  # one step, Eq. (44) with dt = 1
    dJ = np.sum(dW * v_s) / Lx  # x-averaged current change
    dn = np.sum(dW) / Lx
    return dJ, dn, lost, np.sqrt(np.mean(dW**2)) * x_s.size / Lx


if __name__ == "__main__":
    Np, dv = 200_000, 0.1
    print("exact: dJ = 1, dn = 0")
    print("case                         dJ/dJ_exact    dn     H lost   rms|dW|*M/L")
    for v0 in (0.0, 0.03, 0.05):
        r = run(np.full(Np, v0), None, dv=dv)
        print(f"cold collocated v0={v0:<5}        {r[0]:+.4f}   {r[1]:+.1e}  {r[2]:.2f}    {r[3]:.2e}")
    for k in (0.25, 0.5, 1.0, 2.0, 4.0):
        v = rng.normal(0, k * dv, Np)
        r = run(v, None, dv=dv)
        print(f"warm collocated sigma={k}dv      {r[0]:+.4f}   {r[1]:+.1e}  {r[2]:.2f}    {r[3]:.2e}")
    for k in (1.0, 2.0, 4.0):
        vs = rng.uniform(-3 * k * dv, 3 * k * dv, Np)
        r = run(np.full(Np, 0.03), vs, dv=dv)
        print(f"cold PIC, sens. spread +-{3*k}dv  {r[0]:+.4f}   {r[1]:+.1e}  {r[2]:.2f}    {r[3]:.2e}")
