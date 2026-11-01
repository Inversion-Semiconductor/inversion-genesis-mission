"""Smoothed brute-force reference: J(theta) slice over 4 seeds, local quadratic fit at theta0."""
import sys
sys.argv = [sys.argv[0], sys.argv[1], sys.argv[2]]
import importlib.util
spec = importlib.util.spec_from_file_location("a", sys.path[0] + "/adjoint_pdf_1d1v.py" if False else __file__.replace("slice_ref.py", "adjoint_pdf_1d1v.py"))
a = importlib.util.module_from_spec(spec); spec.loader.exec_module(a)
import numpy as np
thetas = np.linspace(0.05, 0.11, 13)
Js = np.array([[a.run(th, sd)["J"] for th in thetas] for sd in (1, 2, 3, 4)])  # (seed, theta, time)
Jm = Js.mean(axis=0)
times = a.rec_every * a.dt * np.arange(1, Jm.shape[1] + 1)
print("t | slope of quadratic fit at theta0 (seed-mean J) | per-seed slope sd | linear-fit slope")
for i in range(0, len(times), 2):
    c2 = np.polyfit(thetas - a.theta0, Jm[:, i], 2)
    c1 = np.polyfit(thetas - a.theta0, Jm[:, i], 1)
    per = [np.polyfit(thetas - a.theta0, Js[s, :, i], 2)[1] for s in range(4)]
    print(f"{times[i]:5.0f} | {c2[1]:9.3f} | {np.std(per):7.3f} | {c1[0]:9.3f}")
np.save(__file__.replace(".py", f"_N{a.N}_Nt{a.Nt}.npy"), Js)
