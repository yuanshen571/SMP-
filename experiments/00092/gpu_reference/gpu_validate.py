"""End-to-end validation: GpuPhysicalFilter vs PhysicalFilter, verdicts must match."""
import sys, time, json
from pathlib import Path
import numpy as np, torch
ROOT = Path("/root/gpufree-data/smp-g1"); REPO = ROOT/"repo"; EXP = REPO/"experiments"/"00091"
sys.path[:0] = [str(EXP), str(EXP/"vendor"), str(REPO/"src"), "/tmp"]
from smp.rl.utils import load_denoiser
from gsi_physics import FilterConfig, PhysicalFilter
from gpu_filter import GpuPhysicalFilter
from noise_rescue import decode
from mjlab.tasks.registry import load_env_cfg
from mjlab.envs import ManagerBasedRlEnv
import smp.rl.tasks  # noqa
DEV = "cuda:0"
ARCH = Path("/root/gpufree-share/smp_gsi_paired_0009/formal/"
            "0009_location_seed2027_0009_execution_20261120_165921_vwjhPg")
cfg = load_env_cfg("Smp-Location-G1", play=False); cfg.scene.num_envs = 1
for k in [k for k in cfg.events if "smp" in k or "gsi" in k]: cfg.events.pop(k)
env = ManagerBasedRlEnv(cfg=cfg, device=DEV, render_mode=None)
bundle = load_denoiser(str(REPO/"datasets/pretrain_ckpt/pretrained_lafan_run.pt"), DEV)
cfgre = FilterConfig(repair=False, control_dt=0.02)
ref = PhysicalFilter(env.sim.mj_model, cfgre)
gpu = GpuPhysicalFilter(env.sim.mj_model, cfgre, device=DEV)

def windows(batch, rows):
    p = torch.load(ARCH/batch, map_location="cpu")
    w = decode(bundle, p["initial"][rows].to(DEV), p["posterior"][:, rows].to(DEV))
    return w.detach().cpu().numpy()

def cmp(name, w, repair=False):
    t0 = time.perf_counter(); o1, m1, d1 = ref.process(w, repair=repair); t1 = time.perf_counter()-t0
    t0 = time.perf_counter(); o2, m2, d2 = gpu.process(w, repair=repair); t2 = time.perf_counter()-t0
    same_mask = bool(np.array_equal(m1, m2))
    same_diag = {k: (d1[k] == d2[k]) for k in d1}
    bad = {k: (d1[k], d2[k]) for k, v in same_diag.items() if not v}
    same_out = bool(np.array_equal(o1, o2))
    print(f"{name}: n={len(w)} mask_equal={same_mask} out_equal={same_out} diag_equal={not bad} "
          f"| numpy {t1*1000:.0f} ms  gpu {t2*1000:.0f} ms  {t1/max(t2,1e-9):.1f}x")
    if bad: print("   diag diffs:", bad)
    if not same_mask:
        idx = np.flatnonzero(m1 != m2)
        print("   mask diff rows:", idx[:10], "count", len(idx))
    return same_mask and not bad and same_out

ok = True
ok &= cmp("calib600", windows("batch_0002.pt", torch.arange(1024))[:300].copy(), False)
ok &= cmp("calib600b", windows("batch_0003.pt", torch.arange(1024))[:300].copy(), False)
ok &= cmp("random1k", np.concatenate([windows(f, torch.arange(1024)) for f in
                                     (f"batch_{i:04d}.pt" for i in (0, 1, 2, 3))]), False)
ok &= cmp("repair256", windows("batch_0004.pt", torch.arange(256)), True)
print("ALL_EQUAL" if ok else "MISMATCH")
