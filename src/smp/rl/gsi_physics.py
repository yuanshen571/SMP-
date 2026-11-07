"""Offline kinematic checks for G1 GSI windows; never advances the simulator.

All lengths in metres internally. A private MjData uses the actual CPU model;
the environment's live state, contact filters, and dynamics parameters are untouched.
"""

from __future__ import annotations

import hashlib
import itertools
import json
from dataclasses import dataclass

import mujoco
import numpy as np

JOINT_NAMES = (
  tuple(
    f"{side}_{name}_joint"
    for side in ("left", "right")
    for name in (
      "hip_pitch",
      "hip_roll",
      "hip_yaw",
      "knee",
      "ankle_pitch",
      "ankle_roll",
    )
  )
  + ("waist_yaw_joint", "waist_roll_joint", "waist_pitch_joint")
  + tuple(
    f"{side}_{name}_joint"
    for side in ("left", "right")
    for name in (
      "shoulder_pitch",
      "shoulder_roll",
      "shoulder_yaw",
      "elbow",
      "wrist_roll",
      "wrist_pitch",
      "wrist_yaw",
    )
  )
)
EE_NAMES = (
  "left_ankle_roll_link",
  "right_ankle_roll_link",
  "torso_link",
  "left_wrist_yaw_link",
  "right_wrist_yaw_link",
)
EXCLUDED_BODIES = (
  ("pelvis", "left_hip_roll_link"),
  ("pelvis", "right_hip_roll_link"),
  ("left_elbow_link", "left_wrist_pitch_link"),
  ("right_elbow_link", "right_wrist_pitch_link"),
)


def local(name: str) -> str:
  return name.removeprefix("robot/")


def rotations(x):
  """Same columns 0/2 and Gram--Schmidt convention as feature_to_state.py."""
  a, b = x[..., 3:6], x[..., 6:9]
  with np.errstate(invalid="ignore", divide="ignore", over="ignore"):
    n0 = np.linalg.norm(a, axis=-1, keepdims=True)
    c0 = a / np.maximum(n0, 1e-12)
    c2 = b - np.sum(c0 * b, axis=-1, keepdims=True) * c0
    n2 = np.linalg.norm(c2, axis=-1, keepdims=True)
    c2 /= np.maximum(n2, 1e-12)
    matrix = np.stack((c0, np.cross(c2, c0), c2), axis=-1)
  valid = (
    np.isfinite(matrix).all(axis=(-2, -1)) & (n0[..., 0] > 1e-12) & (n2[..., 0] > 1e-12)
  )
  return matrix, valid


def segment_distance(p, q, r, s):
  """Exact segment minimum: feasible interior point and four boundaries."""
  u, v, w = q - p, s - r, p - r

  def dot(a, b):
    return np.sum(a * b, axis=-1)

  def div(a, b):
    return np.divide(a, b, out=np.zeros_like(a), where=b > 1e-25)

  aa, bb, cc, dd, ee = dot(u, u), dot(u, v), dot(v, v), dot(u, w), dot(v, w)
  t0, t1 = np.clip(div(ee, cc), 0, 1), np.clip(div(ee + bb, cc), 0, 1)
  s0, s1 = np.clip(div(-dd, aa), 0, 1), np.clip(div(bb - dd, aa), 0, 1)
  best = np.minimum(
    dot(w - t0[..., None] * v, w - t0[..., None] * v),
    dot(w + u - t1[..., None] * v, w + u - t1[..., None] * v),
  )
  z = w + s0[..., None] * u
  best = np.minimum(best, dot(z, z))
  z = w + s1[..., None] * u - v
  best = np.minimum(best, dot(z, z))
  den = aa * cc - bb * bb
  si, ti = div(bb * ee - cc * dd, den), div(aa * ee - bb * dd, den)
  z = w + si[..., None] * u - ti[..., None] * v
  interior = (den > 1e-25) & (si >= 0) & (si <= 1) & (ti >= 0) & (ti <= 1)
  return np.sqrt(np.maximum(0, np.where(interior, np.minimum(best, dot(z, z)), best)))


@dataclass(frozen=True)
class FilterConfig:
  repair: bool = False
  joint_tolerance_rad: float = 1e-5
  repair_limit_deg: float = 1.0
  ground_limit_mm: float = 19.72
  control_dt: float = 0.02


class PhysicalFilter:
  def __init__(self, model: mujoco.MjModel, config: FilterConfig):
    self.model, self.config = model, config
    self.data = mujoco.MjData(model)
    joint_map = {local(model.joint(j).name): j for j in range(model.njnt)}
    self.joints = np.array([joint_map[n] for n in JOINT_NAMES])
    if not np.all(model.jnt_type[self.joints] == mujoco.mjtJoint.mjJNT_HINGE):
      raise ValueError("G1 filter requires 29 named scalar hinge joints")
    if not np.all(model.jnt_limited[self.joints]):
      raise ValueError("all 29 G1 joints must have model limits")
    self.qadr = model.jnt_qposadr[self.joints]
    self.lo, self.hi = model.jnt_range[self.joints].T.copy()
    frees = np.flatnonzero(model.jnt_type == mujoco.mjtJoint.mjJNT_FREE)
    if len(frees) != 1:
      raise ValueError("expected exactly one free root")
    self.root_adr = int(model.jnt_qposadr[frees[0]])
    body_map = {local(model.body(b).name): b for b in range(model.nbody)}
    self.ee = np.array([body_map[n] for n in EE_NAMES])
    self.geoms = np.array(
      [
        g
        for g in range(model.ngeom)
        if model.geom_type[g] != mujoco.mjtGeom.mjGEOM_PLANE
        and (model.geom_contype[g] or model.geom_conaffinity[g])
      ]
    )
    types = model.geom_type[self.geoms]
    if len(self.geoms) != 33 or set(types) != {
      int(mujoco.mjtGeom.mjGEOM_SPHERE),
      int(mujoco.mjtGeom.mjGEOM_CAPSULE),
    }:
      raise ValueError("expected original G1's 31 capsule and 2 sphere colliders")
    if int((types == mujoco.mjtGeom.mjGEOM_CAPSULE).sum()) != 31:
      raise ValueError("G1 collider topology changed")
    self.radius = model.geom_size[self.geoms, 0].copy()
    self.half = model.geom_size[self.geoms, 1].copy()
    self.half[types == mujoco.mjtGeom.mjGEOM_SPHERE] = 0
    excluded = {tuple(sorted((body_map[a], body_map[b]))) for a, b in EXCLUDED_BODIES}
    bodies = model.geom_bodyid[self.geoms]
    self.pairs = []
    for i, j in itertools.combinations(range(len(self.geoms)), 2):
      a, b = int(bodies[i]), int(bodies[j])
      if a == b or model.body_parentid[a] == b or model.body_parentid[b] == a:
        continue
      if tuple(sorted((a, b))) not in excluded:
        self.pairs.append((i, j))
    if len(self.pairs) != 469:
      raise ValueError(f"expected 469 checked collider pairs, got {len(self.pairs)}")
    planes = np.flatnonzero(model.geom_type == mujoco.mjtGeom.mjGEOM_PLANE)
    if len(planes) != 1:
      raise ValueError("expected one flat ground plane")
    mujoco.mj_kinematics(model, self.data)
    normal = self.data.geom_xmat[planes[0]].reshape(3, 3)[:, 2]
    if not np.allclose(normal, (0, 0, 1), atol=1e-10):
      raise ValueError("ground must be a horizontal upward plane")
    self.ground_z = float(self.data.geom_xpos[planes[0], 2])
    self.manifest = dict(
      joints=list(JOINT_NAMES),
      limits_rad=np.stack((self.lo, self.hi), axis=1).tolist(),
      excluded_bodies=EXCLUDED_BODIES,
      ground_z=self.ground_z,
      joint_axes=model.jnt_axis[self.joints].tolist(),
      joint_positions=model.jnt_pos[self.joints].tolist(),
      colliders=[
        dict(
          name=local(model.geom(int(g)).name),
          body=local(model.body(int(model.geom_bodyid[g])).name),
          type=int(model.geom_type[g]),
          size=model.geom_size[g].tolist(),
          pos=model.geom_pos[g].tolist(),
          quat=model.geom_quat[g].tolist(),
        )
        for g in self.geoms
      ],
      bodies=[
        dict(
          name=local(model.body(b).name),
          parent=local(model.body(int(model.body_parentid[b])).name),
          pos=model.body_pos[b].tolist(),
          quat=model.body_quat[b].tolist(),
        )
        for b in range(model.nbody)
      ],
      checked_pairs=[
        (
          local(model.geom(int(self.geoms[i])).name),
          local(model.geom(int(self.geoms[j])).name),
        )
        for i, j in self.pairs
      ],
    )
    self.manifest["sha256"] = hashlib.sha256(
      json.dumps(self.manifest, sort_keys=True).encode()
    ).hexdigest()

  def _poses(self, window):
    """Canonical yaw/origin; self distances and horizontal ground depth invariant."""
    flat = window.reshape(-1, 59).astype(np.float64)
    matrix, valid = rotations(flat)
    if not valid.all():
      raise ValueError("invalid rotations reached geometry")
    q = np.tile(self.model.qpos0, (len(flat), 1))
    adr = self.root_adr
    q[:, adr : adr + 3] = flat[:, :3]
    for k in range(len(flat)):
      mujoco.mju_mat2Quat(q[k, adr + 3 : adr + 7], matrix[k].ravel())
    q[:, self.qadr] = flat[:, 9:38]
    return q

  def _kinematics(self, windows):
    qpos = self._poses(windows)
    p = np.empty((len(qpos), 33, 3))
    q = np.empty_like(p)
    ee = np.empty((len(qpos), 5, 3))
    for k, pose in enumerate(qpos):
      self.data.qpos[:] = pose
      mujoco.mj_kinematics(self.model, self.data)
      center = self.data.geom_xpos[self.geoms]
      axis = self.data.geom_xmat[self.geoms].reshape(-1, 3, 3)[:, :, 2]
      p[k] = center - self.half[:, None] * axis
      q[k] = center + self.half[:, None] * axis
      ee[k] = self.data.xpos[self.ee]
    return p, q, ee

  def process(self, windows, *, repair: bool | None = None):
    """Return corrected copy, per-candidate pass mask, and aggregate diagnostics.

    EE changes use FK(corrected)-FK(original), preserving the original generated
    residual instead of adding an unrelated full-feature reconstruction change.
    No velocity/acceleration magnitude cutoff or extra dynamic quality gate.
    """
    x = np.asarray(windows)
    if x.ndim != 3 or x.shape[1:] != (10, 59) or x.dtype != np.float32:
      raise ValueError("expected raw float32 [N,10,59] GSI windows")
    out = x.copy()
    n = len(out)
    _, rot_ok = rotations(out.astype(np.float64))
    numeric = np.isfinite(out).all(axis=(1, 2)) & rot_ok.all(axis=1)
    with np.errstate(invalid="ignore", over="ignore"):
      canonical_ee = (
        out[:, :, 38:53].reshape(n, 10, 5, 3) + out[:, :, :3][:, :, None, :]
      )
      raw_velocity = np.diff(out[:, :, 9:38], axis=1) / np.float32(
        self.config.control_dt
      )
      numeric &= np.isfinite(canonical_ee).all(axis=(1, 2, 3)) & np.isfinite(
        raw_velocity
      ).all(axis=(1, 2))
    angles = out[:, :, 9:38].astype(np.float64)
    with np.errstate(invalid="ignore", over="ignore"):
      excess = np.maximum(0, np.maximum(self.lo - angles, angles - self.hi)).max(
        axis=(1, 2)
      )
    joint_ok = numeric & (excess <= self.config.joint_tolerance_rad)
    do_repair = self.config.repair if repair is None else repair
    repaired = (
      numeric & ~joint_ok & (excess <= np.deg2rad(self.config.repair_limit_deg))
      if do_repair
      else np.zeros(n, dtype=bool)
    )
    diagnostic = dict(
      generated=n,
      numerical_failed=int((~numeric).sum()),
      joint_failed=int((numeric & ~joint_ok & ~repaired).sum()),
      repair_attempted=int(repaired.sum()),
      repair_velocity_change_max_rad_s=0.0,
      repair_acceleration_change_max_rad_s2=0.0,
      repair_ee_delta_max_m=0.0,
    )
    if repaired.any():
      original = out[repaired].copy()
      qnew = np.clip(original[:, :, 9:38].astype(np.float64), self.lo, self.hi).astype(
        np.float32
      )
      out[repaired, :, 9:38] = qnew
      _, _, ee_before = self._kinematics(original)
      _, _, ee_after = self._kinematics(out[repaired])
      delta = (ee_after - ee_before).reshape(-1, 10, 15)
      corrected_ee = (original[:, :, 38:53].astype(np.float64) + delta).astype(
        np.float32
      )
      out[repaired, :, 38:53] = corrected_ee
      before_v = (
        np.diff(original[:, :, 9:38].astype(np.float64), axis=1)
        / self.config.control_dt
      )
      after_v = np.diff(qnew.astype(np.float64), axis=1) / self.config.control_dt
      diagnostic["repair_velocity_change_max_rad_s"] = float(
        np.abs(after_v - before_v).max()
      )
      diagnostic["repair_acceleration_change_max_rad_s2"] = float(
        np.abs(np.diff(after_v - before_v, axis=1) / self.config.control_dt).max()
      )
      diagnostic["repair_ee_delta_max_m"] = float(np.abs(delta).max())
      if not np.array_equal(out[repaired, :, :9], original[:, :, :9]):
        raise AssertionError("root pose must not change during joint correction")
      if not np.array_equal(out[repaired, :, 53:59], original[:, :, 53:59]):
        raise AssertionError("root velocity must not change during joint correction")
    # Re-check strict model limits after any repair, retaining the original tiny tolerance.
    qfinal = out[:, :, 9:38].astype(np.float64)
    joint_final = (
      (qfinal >= self.lo - self.config.joint_tolerance_rad)
      & (qfinal <= self.hi + self.config.joint_tolerance_rad)
    ).all(axis=(1, 2))
    with np.errstate(invalid="ignore", over="ignore"):
      qvelocity = np.diff(qfinal, axis=1) / self.config.control_dt
      numeric &= np.isfinite(out).all(axis=(1, 2)) & np.isfinite(qvelocity).all(
        axis=(1, 2)
      )
      canonical_ee = (
        out[:, :, 38:53].reshape(n, 10, 5, 3) + out[:, :, :3][:, :, None, :]
      )
      numeric &= np.isfinite(canonical_ee).all(axis=(1, 2, 3))
    eligible = np.flatnonzero(numeric & joint_final)
    passed = np.zeros(n, dtype=bool)
    diagnostic.update(
      geometry_checked=len(eligible),
      self_failed=0,
      ground_failed=0,
      qualified=0,
      repair_qualified=0,
      ground_max_mm=0.0,
      self_max_mm=0.0,
    )
    if len(eligible):
      p, q, _ = self._kinematics(out[eligible])
      finite_fk = np.isfinite(p).all(axis=(1, 2)) & np.isfinite(q).all(axis=(1, 2))
      if not finite_fk.all():
        raise FloatingPointError("non-finite reconstructed collision geometry")
      depth = (
        np.maximum(
          0,
          self.ground_z
          - (np.minimum(p[:, :, 2], q[:, :, 2]) - self.radius).min(axis=1),
        )
        * 1000
      )
      ground = (depth.reshape(-1, 10) <= self.config.ground_limit_mm).all(axis=1)
      overlap = np.zeros(len(p), dtype=bool)
      maximum = 0.0
      for i, j in self.pairs:
        gap = (
          segment_distance(p[:, i], q[:, i], p[:, j], q[:, j])
          - self.radius[i]
          - self.radius[j]
        )
        overlap |= gap < 0
        maximum = max(maximum, float(np.maximum(0, -gap).max() * 1000))
      self_ok = ~overlap.reshape(-1, 10).any(axis=1)
      passed[eligible] = ground & self_ok
      diagnostic.update(
        self_failed=int((~self_ok).sum()),
        ground_failed=int((~ground).sum()),
        qualified=int(passed.sum()),
        repair_qualified=int((passed & repaired).sum()),
        ground_max_mm=float(depth.max()),
        self_max_mm=maximum,
      )
    return out, passed, diagnostic
