import sys, time
from pathlib import Path
import numpy as np, torch
ROOT=Path("/root/gpufree-data/smp-g1"); REPO=ROOT/"repo"; EXP=REPO/"experiments"/"00091"
sys.path[:0]=[str(EXP), str(EXP/"vendor"), str(REPO/"src"), "/tmp"]
from smp.rl.utils import load_denoiser
from gsi_physics import FilterConfig, PhysicalFilter
from gpu_filter import GpuPhysicalFilter
from gpu_scorer import GpuScorer
from noise_rescue import PhysicalScorer, decode
from mjlab.tasks.registry import load_env_cfg
from mjlab.envs import ManagerBasedRlEnv
import smp.rl.tasks  # noqa
DEV="cuda:0"
ARCH=Path("/root/gpufree-share/smp_gsi_paired_0009/formal/0009_location_seed2027_0009_execution_20261120_165921_vwjhPg")
cfg=load_env_cfg("Smp-Location-G1", play=False); cfg.scene.num_envs=1
for k in [k for k in cfg.events if "smp" in k or "gsi" in k]: cfg.events.pop(k)
env=ManagerBasedRlEnv(cfg=cfg, device=DEV, render_mode=None)
bundle=load_denoiser(str(REPO/"datasets/pretrain_ckpt/pretrained_lafan_run.pt"), DEV)
c=FilterConfig(repair=False, control_dt=0.02)
ref_sc=PhysicalScorer(PhysicalFilter(env.sim.mj_model, c))
gpu_sc=GpuScorer(GpuPhysicalFilter(env.sim.mj_model, c, device=DEV))

def wins(batch, rows):
    p=torch.load(ARCH/batch, map_location="cpu")
    return decode(bundle, p["initial"][rows].to(DEV), p["posterior"][:, rows].to(DEV)).detach().cpu().numpy()

def cmp(name, w):
    t0=time.perf_counter(); a=ref_sc.evaluate(w); t1=time.perf_counter()-t0
    t0=time.perf_counter(); b=gpu_sc.evaluate(w); t2=time.perf_counter()-t0
    def eq(x, y):
        if x.dtype.kind in "fc": return bool(np.array_equal(x, y, equal_nan=True))
        return bool(np.array_equal(x, y))
    res={k: eq(a[k], b[k]) for k in ("passed","numeric","joint","ground","overlap","fitness")}
    bad=[k for k,v in res.items() if not v]
    print(f"{name}: n={len(w)} all_equal={not bad} {bad if bad else ''} | ref {t1*1000:.0f} ms  gpu {t2*1000:.0f} ms  {t1/max(t2,1e-9):.1f}x")
    if bad:
        k=bad[0]; x, y = a[k], b[k]
        d=np.abs(np.asarray(x,float)-np.asarray(y,float))
        print(f"   {k}: max|d|={np.nanmax(d):.6g}  n_diff={(d>0).sum()}  rows={np.flatnonzero(d>0)[:5]}")
    return not bad

ok=True
ok&=cmp("calib600a", wins("batch_0002.pt", torch.arange(300)))
ok&=cmp("calib600b", wins("batch_0003.pt", torch.arange(300)))
ok&=cmp("random4096", np.concatenate([wins(f"batch_{i:04d}.pt", torch.arange(1024)) for i in (0,1,2,3)]))
print("ALL_EQUAL" if ok else "MISMATCH")
