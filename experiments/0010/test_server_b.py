"""Server-B routing and setup contract checks; no network/install/GPU actions."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import server_b_runtime as sb

E = Path(__file__).resolve().parent


class ServerBTests(unittest.TestCase):
  def test_identity_accepts_only_agreed_non_root_server_and_paths(self):
    with tempfile.TemporaryDirectory() as temp:
      root = Path(temp) / "SMP+"
      repo = root / "repo"
      repo.mkdir(parents=True)
      archive = Path(temp) / "rollout"
      archive.mkdir()
      with (
        patch.object(sb, "PROJECT_ROOT", root),
        patch.object(sb, "ARCHIVE_ROOT", archive),
        patch.object(sb.os, "geteuid", return_value=1000),
        patch.object(sb.pwd, "getpwuid", return_value=SimpleNamespace(pw_name="yang")),
        patch.object(sb.platform, "node", return_value="zxq"),
        patch.object(sb.platform, "system", return_value="Linux"),
        patch.object(sb.platform, "machine", return_value="x86_64"),
      ):
        self.assertEqual(sb.check_identity(repo), repo)
        with (
          patch.object(sb.os, "geteuid", return_value=0),
          self.assertRaises(AssertionError),
        ):
          sb.check_identity(repo)
        with (
          patch.object(sb.platform, "node", return_value="server-A"),
          self.assertRaises(AssertionError),
        ):
          sb.check_identity(repo)
        with self.assertRaises(AssertionError):
          sb.check_identity(root)

  def test_one_shared_env_frozen_install_and_no_sudo(self):
    script = (E / "setup_server_b.sh").read_text()
    env = (E / "server_b_env.sh").read_text()
    self.assertIn("sync --frozen --no-dev --python 3.13.18", script)
    self.assertIn("UV_UNMANAGED_INSTALL", script)
    self.assertIn("shared_environment.json", script)
    self.assertIn('"${SMP_SHARED_ROOT}/venv"', env)
    self.assertIn("UV_PYTHON_INSTALL_DIR", env)
    self.assertIn("WARP_CACHE_PATH", env)
    self.assertNotIn("sudo ", script)
    self.assertNotIn("pip install", script)
    self.assertNotIn("run_0010.py", script)
    self.assertNotIn("venv-0010", script)

  def test_new_identity_and_archive_path_are_recorded(self):
    spec = json.loads((E / "EXPERIMENT.json").read_text())
    self.assertEqual(spec["server"], "B")
    self.assertEqual(spec["seed"], 2028)
    self.assertEqual(spec["repeats_experiment"], "0005")
    self.assertEqual(
      spec["gsi_recording"]["root"], "/home/yang/rollout/smp_gsi_paired_0010"
    )
    self.assertEqual(spec["rng_protocol"]["gsi_seed"], 102028)
    self.assertEqual(
      (E.parent.parent / "0010_EXECUTION.md").read_bytes(),
      (E / "EXECUTION.md").read_bytes(),
    )
    for name in [
      "run_0010.py",
      "execute.sh",
      "activate_0010.sh",
      "setup_server_b.sh",
      "server_b_env.sh",
    ]:
      text = (E / name).read_text()
      self.assertNotIn("/root/", text)
      self.assertNotIn("L40", text)


if __name__ == "__main__":
  unittest.main(verbosity=2)
