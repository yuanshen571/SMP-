import sys, time
from pathlib import Path
import numpy as np, torch
ROOT=Path("/root/gpufree-data/smp-g1"); REPO=ROOT/"repo"; EXP=REPO/"experiments"/"00091"
sys.path[:0]=[str(EXP), str(EXP/"vendor"), str(REPO/"src"), "/tmp"]
from smp.rl.utils import load_denoiser
from gsi_physics import FilterConfig, PhysicalFilter
from gpu_filter import GpuPhysicalFilter
from noise_rescue import decode
from mjlab.tasks.registry import load_env_cfg
from mjlab.envs import ManagerBasedRlEnv
import smp.rl.tasks  # noqa
DEV="cuda:0"
cfg=load_env_cfg("Smp-Location-G1", play=False); cfg.scene.num_envs=1
for k in [k for k in cfg.events if "smp" in k or "gsi" in k]: cfg.events.pop(k)
env=ManagerBasedRlEnv(cfg=cfg, device=DEV, render_mode=None)
bundle=load_denoiser(str(REPO/"datasets/pretrain_ckpt/pretrained_lafan_run.pt"), DEV)
c=FilterConfig(repair=False, control_dt=0.02)
ref=PhysicalFilter(env.sim.mj_model, c); gpu=GpuPhysicalFilter(env.sim.mj_model, c, device=DEV)
big=decode(bundle, torch.randn(8192,10,59,device=DEV), torch.randn(49,8192,10,59,device=DEV)).detach().cpu().numpy()
print(" batch |  numpy ms/win |  gpu ms/win | speedup")
for B in (32, 128, 512, 2048, 8192):
    w=big[:B].copy()
    t0=time.perf_counter(); ref.process(w); t_cpu=(time.perf_counter()-t0)*1000/B
    # warm-up once, then time
    gpu.process(w); t0=time.perf_counter(); gpu.process(w); t_gpu=(time.perf_counter()-t0)*1000/B
    print(f"{B:6d} | {t_cpu:12.3f} | {t_gpu:11.3f} | {t_cpu/t_gpu:6.1f}x")
