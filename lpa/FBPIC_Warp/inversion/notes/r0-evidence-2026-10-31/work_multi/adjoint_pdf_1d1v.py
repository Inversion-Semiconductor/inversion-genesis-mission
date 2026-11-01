"""1D-1V driven Vlasov-Poisson PIC: particle-exact tangent, collocated particle-pdf
forward sensitivity, and its exact discrete transpose (adjoint), against CRN central FD.

Parameter theta: amplitude of an external driver E_ext = theta*sin(k x - w t).
QoI: J(t) = sum_p w_p v_p^2 / 2 (kinetic energy) at several t.
Units: omega_p = 1, v_th = 1, electrons q = -1, m = 1, uniform ion background.
"""
import sys
import time
import numpy as np

k0, omega, theta0 = 0.5, 1.3, 0.08
L = 2 * np.pi / k0
Ng, Nv, vmax = 64, 48, 6.0
dx, dv = L / Ng, 2 * vmax / (Nv - 1)
vnodes = -vmax + dv * np.arange(Nv)
dt, Nt = 0.1, int(sys.argv[2]) if len(sys.argv) > 2 else 1500
N = int(sys.argv[1]) if len(sys.argv) > 1 else 50000
rec_every = 50
xg = dx * np.arange(Ng)

# Poisson: dE/dx = rho, E_k = rho_k/(i k), mean-free. Explicit matrix P (Ng x Ng).
kk = 2 * np.pi * np.fft.fftfreq(Ng, d=dx)
def poisson(rho):
    rk = np.fft.fft(rho)
    Ek = np.zeros_like(rk)
    Ek[1:] = rk[1:] / (1j * kk[1:])
    return np.real(np.fft.ifft(Ek))
P = np.stack([poisson(e) for e in np.eye(Ng)], axis=1)

def cic(x):
    g = x / dx
    j = np.floor(g).astype(np.int64)
    fr = g - j
    return j % Ng, (j + 1) % Ng, fr

def deposit(j0, j1, fr, q):
    return (np.bincount(j0, (1 - fr) * q, Ng) + np.bincount(j1, fr * q, Ng)) / dx

def gather(j0, j1, fr, F):
    return (1 - fr) * F[j0] + fr * F[j1]

def b2(t):  # quadratic B-spline, support |t|<1.5, and derivative
    a = np.abs(t)
    val = np.where(a < 0.5, 0.75 - a**2, np.where(a < 1.5, 0.5 * (1.5 - a) ** 2, 0.0))
    der = np.where(a < 0.5, -2 * t, np.where(a < 1.5, -np.sign(t) * (1.5 - a), 0.0))
    return val, der

def vstencil(v):
    """3 v-node indices, values, and d/dv_node derivative (per Delta v) for quadratic shape."""
    g = (v + vmax) / dv
    c = np.rint(g).astype(np.int64)
    idx = np.stack([c - 1, c, c + 1])
    t = idx - g  # (v_node - v)/dv
    val, der = b2(t)
    ok = (idx >= 0) & (idx < Nv)
    idx = np.clip(idx, 0, Nv - 1)
    return idx, val * ok, der * ok / dv

def init(seed):
    r = np.random.default_rng(seed)
    x = r.uniform(0, L, N)
    v = r.normal(0, 1, N)
    return x, v

w = L / N  # electron weight (density 1)

def run(theta, seed, tangent=False, record=False):
    x, v = init(seed)  # v at -1/2 (first push from t=0)
    xi_x = np.zeros(N); xi_v = np.zeros(N)
    Js, dJs, xinorm = [], [], []
    recs = []
    for n in range(Nt):
        t = n * dt
        j0, j1, fr = cic(x)
        rho = 1.0 - deposit(j0, j1, fr, np.full(N, w))
        E = poisson(rho)
        if record:
            recs.append((x.copy(), v.copy()))
        Ep = gather(j0, j1, fr, E)
        ph = k0 * x - omega * t
        s = np.sin(ph)
        if tangent:
            dne = (np.bincount(j0, -w * xi_x / dx, Ng) + np.bincount(j1, w * xi_x / dx, Ng)) / dx
            dE = P @ (-dne)
            dEp = gather(j0, j1, fr, dE) + (E[j1] - E[j0]) / dx * xi_x
            xi_v = xi_v - dt * (dEp + s + theta * k0 * np.cos(ph) * xi_x)
        v = v - dt * (Ep + theta * s)
        if tangent:
            xi_x = xi_x + dt * xi_v
        x = (x + dt * v) % L
        if (n + 1) % rec_every == 0:
            Js.append(w * np.sum(0.5 * v**2))
            if tangent:
                dJs.append(w * np.sum(v * xi_v))
                xinorm.append((np.sqrt(np.mean(xi_x**2)), np.sqrt(np.mean(xi_v**2))))
    return dict(J=np.array(Js), dJ=np.array(dJs), xinorm=np.array(xinorm), recs=recs, vfinal=v)

def pdf_coeffs(x, v):
    """Per-step coefficients of the collocated particle-pdf scheme (depend on the primal only)."""
    j0, j1, fr = cic(x)
    iv, bv, dbv = vstencil(v)
    # mesh f_v and n_hat on (Ng, Nv): Sx = CIC, Sv = quadratic
    fv = np.zeros((Ng, Nv)); nh = np.zeros((Ng, Nv))
    for jx, wx in ((j0, 1 - fr), (j1, fr)):
        for a in range(3):
            np.add.at(fv, (jx, iv[a]), w * wx * dbv[a])
            np.add.at(nh, (jx, iv[a]), wx * bv[a])
    fv /= dx * dv; nh /= dx * dv
    D = np.where(nh > 0, -1.0 * fv / np.where(nh > 0, nh, 1.0), 0.0)  # (q/m) d_v f / n_hat
    # d_ip for the two x nodes of each marker: sum_j D[i, j] Sv_jp
    d0 = sum(D[j0, iv[a]] * bv[a] for a in range(3))
    d1 = sum(D[j1, iv[a]] * bv[a] for a in range(3))
    return j0, j1, fr, d0, d1

def pdf_forward(recs, theta, psi_from_v=True):
    What = np.zeros(N)
    out, wnorm = [], []
    for n, (x, v) in enumerate(recs):
        t = n * dt
        j0, j1, fr, d0, d1 = pdf_coeffs(x, v)
        drho = -deposit(j0, j1, fr, What)
        Etot = P @ drho + np.sin(k0 * xg - omega * t)
        h = (1 - fr) * d0 * Etot[j0] + fr * d1 * Etot[j1]
        What = What - dt * h
        wnorm.append(np.sqrt(np.mean(What**2)) / w)
        if (n + 1) % rec_every == 0:
            vnext = recs[n + 1][1] if n + 1 < len(recs) else None
            out.append(What.copy())
    return What, out, np.array(wnorm)

def pdf_adjoint(recs, lam_T):
    """Exact transpose of pdf_forward: returns dJ/dtheta and the history of |lambda|."""
    lam = lam_T.copy()
    grad = 0.0
    lnorm = []
    for n in range(len(recs) - 1, -1, -1):
        x, v = recs[n]
        t = n * dt
        j0, j1, fr, d0, d1 = pdf_coeffs(x, v)
        # forward: W' = W - dt * G (P A W + s); G[p, i] = Sx_ip d_ip; A = -deposit
        gl = -dt * (np.bincount(j0, (1 - fr) * d0 * lam, Ng) + np.bincount(j1, fr * d1 * lam, Ng))  # G^T lam * (-dt)
        grad += gl @ np.sin(k0 * xg - omega * t)
        a = P.T @ gl  # adjoint field
        lam = lam - gather(j0, j1, fr, a) / dx  # A^T a, A = -deposit/dx-shape
        lnorm.append((np.sqrt(np.mean(lam**2)), np.max(np.abs(lam))))
    return grad, np.array(lnorm[::-1])

if __name__ == "__main__":
    t0 = time.time()
    base = run(theta0, 1, tangent=True, record=True)
    print(f"N={N} Nt={Nt} T={Nt*dt:.0f}; base run {time.time()-t0:.1f}s")
    recs = base["recs"]
    # final-time QoI uses v after the last push; record final v for the adjoint terminal condition
    vT = base["vfinal"]
    t0 = time.time()
    WT, Wrec, wnorm = pdf_forward(recs, theta0)
    # forward pdf sensitivity at record times: sum W * psi(v at that time). Use the v after the push
    # at that step = recs[n+1][1] (or vT at the end).
    dJ_pdf = []
    for m, Wm in enumerate(Wrec):
        n = (m + 1) * rec_every - 1
        vn = recs[n + 1][1] if n + 1 < Nt else vT
        dJ_pdf.append(np.sum(Wm * 0.5 * vn**2))
    dJ_pdf = np.array(dJ_pdf)
    print(f"pdf forward {time.time()-t0:.1f}s")
    t0 = time.time()
    g_adj, lnorm = pdf_adjoint(recs, 0.5 * vT**2)
    print(f"pdf adjoint {time.time()-t0:.1f}s; adjoint grad {g_adj:.10e} forward {dJ_pdf[-1]:.10e} "
          f"rel diff {abs(g_adj-dJ_pdf[-1])/abs(dJ_pdf[-1]):.2e}")
    # CRN central FD over seeds, several h
    seeds = [1, 2, 3, 4]
    fd = {}
    for h in (0.02, 0.01, 0.005):
        rows = []
        for sd in seeds:
            Jp = run(theta0 + h, sd)["J"]; Jm = run(theta0 - h, sd)["J"]
            rows.append((Jp - Jm) / (2 * h))
        fd[h] = np.array(rows)
    # pdf and exact on other seeds for the spread
    pdf_seeds, ex_seeds = [dJ_pdf], [base["dJ"]]
    for sd in seeds[1:]:
        b = run(theta0, sd, tangent=True, record=True)
        _, Wr, _ = pdf_forward(b["recs"], theta0)
        vals = []
        for m, Wm in enumerate(Wr):
            n = (m + 1) * rec_every - 1
            vn = b["recs"][n + 1][1] if n + 1 < Nt else b["vfinal"]
            vals.append(np.sum(Wm * 0.5 * vn**2))
        pdf_seeds.append(np.array(vals)); ex_seeds.append(b["dJ"])
        del b
    pdf_seeds, ex_seeds = np.array(pdf_seeds), np.array(ex_seeds)
    times = rec_every * dt * np.arange(1, len(dJ_pdf) + 1)
    print("\n t | xi_v rms | exact dJ (seed mean / sd) | pdf dJ (mean / sd) | FD h=.02 (mean/sd) | FD h=.01 | FD h=.005")
    for i in range(0, len(times), 2):
        print(f"{times[i]:5.0f} | {base['xinorm'][i,1]:9.2e} | {ex_seeds[:,i].mean():10.3e} / {ex_seeds[:,i].std():9.2e} | "
              f"{pdf_seeds[:,i].mean():9.4f} / {pdf_seeds[:,i].std():7.4f} | "
              f"{fd[0.02][:,i].mean():8.4f} / {fd[0.02][:,i].std():7.4f} | {fd[0.01][:,i].mean():8.4f} / {fd[0.01][:,i].std():7.4f} | "
              f"{fd[0.005][:,i].mean():8.4f} / {fd[0.005][:,i].std():7.4f}")
    print("\nforward weight rms (per w) at steps:", {s: f"{wnorm[s]:.3e}" for s in (100, 500, 1000, Nt-1) if s < Nt})
    print("adjoint lambda rms / max at steps:", {s: f"{lnorm[s,0]:.3e}/{lnorm[s,1]:.3e}" for s in (0, 500, 1000, Nt-1) if s < Nt})
    print("terminal lambda rms / max:", f"{np.sqrt(np.mean((0.5*vT**2)**2)):.3e}/{np.max(0.5*vT**2):.3e}")
