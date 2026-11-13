"""Run the CPU startup CLI against a real checkpoint before any GPU work."""

import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

from test_startup import SPEC, fixture


class PreflightCliTests(unittest.TestCase):
  def test_runtime_only_fingerprint_works_and_wrong_source_fails(self):
    with tempfile.TemporaryDirectory() as temp:
      root = Path(temp)
      path, info_path, info = fixture(root)
      out = root / "audit.json"
      command = [
        sys.executable,
        str(Path(__file__).with_name("preflight_control.py")),
        "--initial-checkpoint",
        str(path),
        "--control-info",
        str(info_path),
        "--out",
        str(out),
      ]
      result = subprocess.run(
        command,
        capture_output=True,
        text=True,
        env=dict(os.environ, CUDA_VISIBLE_DEVICES=""),
        timeout=30,
      )
      self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
      audit = json.loads(out.read_text())
      self.assertEqual(audit["seed"], SPEC["seed"])
      self.assertEqual(audit["initial_models_sha256"], info["initial_models_sha256"])
      self.assertFalse(audit["static_models_pin_present"])
      info_path.write_text(json.dumps(dict(info, initial_models_sha256="wrong")))
      out.unlink()
      result = subprocess.run(
        command,
        capture_output=True,
        text=True,
        env=dict(os.environ, CUDA_VISIBLE_DEVICES=""),
        timeout=30,
      )
      self.assertNotEqual(result.returncode, 0)
      self.assertFalse(out.exists())


if __name__ == "__main__":
  unittest.main(verbosity=2)
