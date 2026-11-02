"""Wall time of one es1d_lr.run with and without the LR weight and the exact tangent (single process)."""
import os, sys, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import es1d_lr as M
for rep in range(3):
    for mf, tan, tag in ((False, False, "plain"), (True, False, "+mf LR"), (False, True, "+tangent")):
        t0 = time.perf_counter(); M.run(1.0, 0.03, 131072, 1, do_mf=mf, do_tan=tan, tmax=5); dt = time.perf_counter() - t0
        print(f"rep {rep} {tag:9s} {dt:.2f} s", flush=True)
