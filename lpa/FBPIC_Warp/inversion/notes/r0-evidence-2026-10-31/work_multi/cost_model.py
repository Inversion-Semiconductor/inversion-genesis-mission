"""Cost and memory model for forward-batched vs adjoint particle-pdf (R0, sec_multi)."""
from math import comb

def revolve_forward_steps(l, c):
    """Griewank-Walther minimal number of forward step evaluations to reverse l steps with c checkpoints."""
    r = 0
    while comb(c + r, c) < l:
        r += 1
    # p(l,c) = r*l - beta(c+1, r-1), beta(c,r)=C(c+r,c)
    return r * l - comb(c + 1 + r - 1, c + 1) if r > 0 else 0, r

print("== Revolve: extra forward step evaluations / N_t ==")
for name, Nt_seg, cs in [("Twin-S, per-step GPU storage of 1 step", 2000, [10, 20, 45, 100]),
                         ("Twin-S, segments of 40 steps on GPU (50 segments)", 50, [3, 5, 7, 10]),
                         ("Prod, segments of 8 steps on GPU (250 segments)", 250, [5, 10, 20, 40])]:
    for c in cs:
        p, r = revolve_forward_steps(Nt_seg, c)
        print(f"{name}: c={c:3d} r={r} forward segment-evals={p} -> recompute factor {p/Nt_seg:.2f}")

print()
print("== Cost per Jacobian (units: primal runs P) ==")
def fwd(K, c0, c1, Kb=None):
    passes = 1 if Kb is None else -(-K // Kb)
    return passes * (1 + c0) + K * c1, passes
def adj(nres, c0, c1, R, Nb=None):
    passes = 1 if Nb is None else -(-nres // Nb)
    # one recording forward pass (1), then per backward sweep: replay R + mesh c0 + nres_b*c1
    return 1 + passes * (R + c0) + nres * c1, passes

for label, c0, c1, R, Kb, Nb in [
    ("Twin-S optimistic", 1.0, 0.4, 1.0, None, None),
    ("Twin-S central", 2.0, 0.7, 1.2, None, None),
    ("Twin-S pessimistic", 5.0, 1.0, 2.0, None, None),
    ("Prod central (mem-batched)", 2.0, 0.8, 2.0, 8, 6),
]:
    row = []
    for K in (1, 2, 3, 5, 10, 15, 25):
        f, fp = fwd(K, c0, c1, Kb)
        a, ap = adj(11, c0, c1, R, Nb)
        row.append(f"K={K}: F={f:.1f} A={a:.1f}")
    # crossover
    Kx = next((K for K in range(1, 200) if fwd(K, c0, c1, Kb)[0] > adj(11, c0, c1, R, Nb)[0]), None)
    print(f"{label} (c0={c0}, c1={c1}, R={R}, Kb={Kb}, Nb={Nb}): crossover K*={Kx}")
    print("   " + "; ".join(row))

print()
print("== Lagrangian tangent (measured) for comparison ==")
for c1L in (2.47, 1.2):
    print(f"c1_L={c1L}: " + "; ".join(f"K={K}: {1+K*c1L:.1f}" for K in (1, 2, 3, 5, 10, 25)))

print()
print("== Memory ==")
GB = 1e9
# Twin-S
Np = 10e6; Nion = 1.2e6
field_param_twins = 14 * 0.9e6  # 7 spectral arrays x 2 modes x ~0.9 MB
print(f"Twin-S per-param fwd particle-pdf: weights {Np*8/GB:.2f} GB + fields {field_param_twins/GB:.3f} GB")
print(f"Twin-S per-step adjoint replay record (64 B/marker + 32 B/ion): {(Np*64+Nion*32)/GB:.2f} GB at peak")
print(f"Twin-S full checkpoint (72 B/marker + 32 B/ion + fields): {(Np*72+Nion*32+field_param_twins)/GB:.2f} GB at peak")
# Production
Nz, Nr, Nm = 2592, 1080, 5
arr = 16 * Nz * Nr
print(f"Prod field array (one mode, one comp): {arr/1e6:.1f} MB")
spect = 7 * Nm * arr; interp = 6 * Nm * arr; src = 4 * Nm * arr
print(f"Prod per-param field tangent: spectral {spect/GB:.2f} GB + interp {interp/GB:.2f} GB + J/rho {src/GB:.2f} GB = {(spect+interp+src)/GB:.2f} GB")
Npp = 60e6
print(f"Prod per-param weights: {Npp*8/GB:.2f} GB; Lagrangian tangent per param: {Npp*112/GB:.2f} GB")
print(f"Prod per-step adjoint record: {Npp*64/GB:.2f} GB")
dht = 4 * Nr**2 * Nz * Nm  # real flops of complex-by-real DHT, one array
print(f"Prod DHT flops per array transform: {dht:.2e}; 10 transforms at 1.4 TF fp64: {10*dht/1.4e12:.2f} s; at fp32 ~80 TF eff 40 TF: {10*dht/40e12*1e3:.1f} ms")
