"""Check of the mixed pathwise / Girsanov estimator with tangent feedback on a chaotic noisy Duffing oscillator.

dx = p dt, dp = [-g p + x - x^3 + theta (c cos(w t) + B cos(W t))] dt + sigma dW.
The B term mimics a fast quiver force that averages out. For any adapted control h:
  dxi_p = (A xi_x - g xi_p + delta_a - h) dt, dxi_x = xi_p dt, domega = h dW / sigma,
  dE[G]/dtheta = E[G'(x) xi_x + omega (G - mean G)].
h = 0: pathwise. h = delta_a (xi = 0): pure Girsanov (LR). h = (A + wc^2) xi_x + (2 wc - g) xi_p: feedback.
Usage: python -I verify_mixed_feedback.py M B sigma wc
"""
import sys, json
import numpy as np

G_, C_, W_, WF = 0.25, 0.3, 1.0, 15.0
DT, T = 0.01, 30.0


def run(theta, M, B, sigma, seed, mode="none", wc=0.5):
    rng = np.random.default_rng(seed)
    x = rng.normal(1.0, 0.1, M); p = rng.normal(0.0, 0.1, M)
    xx = np.zeros(M); xp = np.zeros(M); om = np.zeros(M)
    nt = int(round(T / DT))
    for n in range(nt):
        t = n * DT
        da = C_ * np.cos(W_ * t) + B * np.cos(WF * t)
        z = rng.standard_normal(M)
        if mode != "none":
            A = 1.0 - 3.0 * x ** 2
            if mode == "path":
                h = 0.0
            elif mode == "lr":
                h = da + A * xx - G_ * xp + xp / DT  # sets xi_p' = 0 (xi stays 0)
            else:
                h = (A + wc ** 2) * xx + (2 * wc - G_) * xp
            om = om + h * z * np.sqrt(DT) / sigma
            xp = xp + DT * (A * xx - G_ * xp + da - h)
            xx = xx + DT * xp
        p = p + DT * (-G_ * p + x - x ** 3 + theta * da) + sigma * np.sqrt(DT) * z
        x = x + DT * p
    g = np.stack([x ** 2, np.cos(x)]); gp = np.stack([2 * x, -np.sin(x)])
    if mode == "none":
        return g.mean(1), None
    est = gp * xx[None] + om[None] * (g - g.mean(1, keepdims=True))
    return g.mean(1), est


if __name__ == "__main__":
    M = int(sys.argv[1]); B = float(sys.argv[2]); sigma = float(sys.argv[3]); wc = float(sys.argv[4])
    out = dict(M=M, B=B, sigma=sigma, wc=wc)
    for mode in ("path", "lr", "fb"):
        _, est = run(1.0, M, B, sigma, 7, mode, wc)
        out[mode] = dict(mean=est.mean(1).tolist(), se=(est.std(1) / np.sqrt(M)).tolist())
    h = 0.05
    Jp, _ = run(1.0 + h, M, B, sigma, 7); Jm, _ = run(1.0 - h, M, B, sigma, 7)
    gp_ = run(1.0 + h, M, B, sigma, 7, "none")
    out["fd_crn_h0.05"] = ((Jp - Jm) / (2 * h)).tolist()
    Jp2, _ = run(1.0 + h, M, B, sigma, 11); Jm2, _ = run(1.0 - h, M, B, sigma, 12)
    out["fd_indep_h0.05"] = ((Jp2 - Jm2) / (2 * h)).tolist()
    out["fd_indep_se_est"] = float(np.sqrt(2 * 0.5 / M) / (2 * h))
    print(json.dumps(out))
