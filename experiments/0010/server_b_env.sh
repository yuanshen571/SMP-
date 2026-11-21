# SMP-G1 server B shared environment, version 1. Source this file for all runs.
export SMP_SHARED_ROOT='/home/yang/SMP+'
export SMP_SHARED_REPO="${SMP_SHARED_ROOT}/repo"
export UV_PROJECT_ENVIRONMENT="${SMP_SHARED_ROOT}/venv"
export UV_CACHE_DIR="${SMP_SHARED_ROOT}/cache/uv"
export UV_PYTHON_INSTALL_DIR="${SMP_SHARED_ROOT}/cache/python"
export UV_PYTHON_BIN_DIR="${SMP_SHARED_ROOT}/tools/python"
export UV_PYTHON_PREFERENCE=only-managed
export HF_HOME="${SMP_SHARED_ROOT}/cache/hf"
export TORCH_HOME="${SMP_SHARED_ROOT}/cache/torch"
export XDG_CACHE_HOME="${SMP_SHARED_ROOT}/cache/xdg"
export TORCHINDUCTOR_CACHE_DIR="${SMP_SHARED_ROOT}/cache/torch_inductor"
export WARP_CACHE_PATH="${SMP_SHARED_ROOT}/cache/warp"
export TRITON_CACHE_DIR="${SMP_SHARED_ROOT}/cache/triton"
export PIP_CACHE_DIR="${SMP_SHARED_ROOT}/cache/pip"
export UV_CONCURRENT_BUILDS=2 TORCHINDUCTOR_COMPILE_THREADS=4 MAX_JOBS=4
export PYTHONUTF8=1 PYTHONIOENCODING=utf-8 WANDB_MODE=disabled MUJOCO_GL=egl
export PATH="${UV_PROJECT_ENVIRONMENT}/bin:${SMP_SHARED_ROOT}/tools:${PATH}"
# CUDA libraries supplied by the frozen wheel set; no system CUDA installation.
for SMP_CUDA_LIB in "${UV_PROJECT_ENVIRONMENT}"/lib/python3.13/site-packages/nvidia/*/lib "${UV_PROJECT_ENVIRONMENT}"/lib/python3.13/site-packages/torch/lib; do
  if [[ -d "${SMP_CUDA_LIB}" && ":${LD_LIBRARY_PATH:-}:" != *":${SMP_CUDA_LIB}:"* ]]; then
    export LD_LIBRARY_PATH="${SMP_CUDA_LIB}${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"
  fi
done
unset SMP_CUDA_LIB
