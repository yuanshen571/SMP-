import sys, time
from pathlib import Path
import numpy as np, torch
ROOT=Path("/root/gpufree-data/smp-g1"); REPO=ROOT/"repo"; EXP=REPO/"experiments"/"00091"
sys.path[:0]=[str(EXP), str(EXP/"vendor"), str(REPO/"src"), "/tmp"]
from smp.rl.utils import load_denoiser
from gsi_physics import FilterConfig, PhysicalFilter
from gpu_filter import GpuPhysicalFilter
from gpu_scorer import GpuScorer
from noise_rescue import PhysicalScorer, SearchConfig, search_batch, decode
from mjlab.tasks.registry import load_env_cfg
from mjlab.envs import ManagerBasedRlEnv
import smp.rl.tasks  # noqa
DEV="cuda:0"
cfg=load_env_cfg("Smp-Location-G1", play=False); cfg.scene.num_envs=1
for k in [k for k in cfg.events if "smp" in k or "gsi" in k]: cfg.events.pop(k)
env=ManagerBasedRlEnv(cfg=cfg, device=DEV, render_mode=None)
bundle=load_denoiser(str(REPO/"datasets/pretrain_ckpt/pretrained_lafan_run.pt"), DEV)
c=FilterConfig(repair=False, control_dt=0.02)
B=1024
init=(torch.randn(B,10,59,device=DEV)*0.5); post=torch.randn(49,B,10,59,device=DEV)
raw=decode(bundle,init,post)
for name, sc in (("cpu-reference", PhysicalScorer(PhysicalFilter(env.sim.mj_model, c))),
                 ("gpu-filter+scorer", GpuScorer(GpuPhysicalFilter(env.sim.mj_model, c, device=DEV)))):
    t0=time.perf_counter()
    out,passed,st=search_batch(bundle, raw, init, post, np.zeros(B,bool), sc, seed=7, global_start=0,
                               config=SearchConfig(), log=lambda *a, **kw: None)
    dt=time.perf_counter()-t0
    per=dt/(st["generations"]*32)
    print(f"{name:18s} wall {dt:7.1f}s  gens {st['generations']:5d}  per-candidate {per*1000:6.3f} ms  "
          f"succ {st['succeeded']} exh {st['exhausted']}")
    np.save(f"/tmp/out_{name}.npy", np.asarray(passed))
