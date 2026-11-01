"""Particle-pdf forward sensitivity vs velocity-mesh resolution and seed (bias check)."""
import sys
N, Nt = sys.argv[1], sys.argv[2]
sys.argv = [sys.argv[0], N, Nt]
import importlib.util
spec = importlib.util.spec_from_file_location("a", __file__.replace("pdf_mesh_scan.py", "adjoint_pdf_1d1v.py"))
a = importlib.util.module_from_spec(spec); spec.loader.exec_module(a)
import numpy as np
res = {}
for Nv in (24, 96):
    a.Nv = Nv; a.dv = 2 * a.vmax / (Nv - 1)
    for sd in (1, 2):
        b = a.run(a.theta0, sd, record=True)
        _, Wr, _ = a.pdf_forward(b["recs"], a.theta0)
        vals = []
        for m, Wm in enumerate(Wr):
            n = (m + 1) * a.rec_every - 1
            vn = b["recs"][n + 1][1] if n + 1 < a.Nt else b["vfinal"]
            vals.append(np.sum(Wm * 0.5 * vn**2))
        res[(Nv, sd)] = np.array(vals)
        del b
        print("done", Nv, sd, flush=True)
times = a.rec_every * a.dt * np.arange(1, len(vals) + 1)
print("t | Nv=24 s1, s2 | Nv=96 s1, s2")
for i in range(0, len(times), 2):
    print(f"{times[i]:5.0f} | {res[(24,1)][i]:8.3f} {res[(24,2)][i]:8.3f} | {res[(96,1)][i]:8.3f} {res[(96,2)][i]:8.3f}")
