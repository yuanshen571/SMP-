"""Prove 0010 differs from the pinned 0008 baseline only in training horizon/audit identity."""
import json
import subprocess
import unittest
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
REFERENCE = '17a5bdad6f4951c6c0363b459cd37a72cbf2fd6d'


def reference(name):
    return subprocess.check_output(
        ['git', 'show', f'{REFERENCE}:experiments/0008/{name}'], cwd=ROOT, text=True
    )


class VariantTests(unittest.TestCase):
    def test_only_training_horizon_changed(self):
        old = json.loads(reference('EXPERIMENT.json'))
        new = json.loads((HERE / 'EXPERIMENT.json').read_text())
        self.assertEqual((new['iterations'], new['samples']), (20000, 491520000))
        self.assertEqual(new['samples'], new['iterations'] * new['num_envs'] * new['num_steps_per_env'])
        self.assertEqual(new['seed'], old['seed'])
        self.assertEqual(new['repeats_experiment'], '0008')
        for item in (old, new):
            for key in ('experiment', 'branch', 'iterations', 'samples', 'runtime_validation',
                        'initialization', 'only_training_change', 'implementation_reference', 'repeats_experiment'):
                item.pop(key)
        self.assertEqual(old, new)

    def test_training_and_statistics_unchanged(self):
        names = ('run_0008.py', 'accept_0008.py', 'activate_0008.sh',
                 'passive_metrics.py', 'inspect_metrics.py', 'verify_completion.py',
                 'contract.py', 'initialization.py', 'test_initialization.py',
                 'test_passive_metrics.py', 'test_reward_formula.py', 'BASELINE_FILES.json')
        for name in names:
            expected = reference(name)
            if name != 'BASELINE_FILES.json':
                expected = expected.replace('0008', '0010').replace('10000', '20000').replace('245760000', '491520000')
            if name == 'run_0008.py':
                expected = expected.replace('"repeats_experiment": "0002"', '"repeats_experiment": "0008"')
                expected = expected.replace('Uses the 0001 training/metrics/checkpoint path. Only the configured reward\ncomposition changes; --no-metrics also retains the additive reward.',
                                            'Extends experiment 0008 from 10000 to 20000 iterations. All simulation,\nreward, PPO, GSI and passive metric behavior is inherited unchanged.')
            if name == 'verify_completion.py':
                expected = expected.replace('info.get("repeats_experiment") == "0002"', 'info.get("repeats_experiment") == "0008"')
                start = expected.index('\n## 运行入口说明')
                end = expected.index('\n"""', start)
                expected = expected[:start] + '\n## 运行入口说明\n\n任务单见 `experiments/0010/EXECUTION.md`，单组入口为 `experiments/0010/execute.sh`。本实验从头训练20000轮，不衔接任何其他组，不依据本完成报告重复启动。\n\nTensorBoard沿用已有统一logs服务。\n' + expected[end:]
            self.assertEqual((HERE / name.replace('0008', '0010')).read_text(), expected, name)

    def test_no_changes_to_pinned_source_tree(self):
        differences = subprocess.check_output(['git', 'diff', REFERENCE, '--', 'src', 'uv.lock', 'pyproject.toml', '.python-version', 'experiments/0008'], cwd=ROOT, text=True)
        self.assertEqual(differences, '')

    def test_standalone_launcher_budget(self):
        script = (HERE / 'execute.sh').read_text()
        commands = [line for line in script.splitlines() if '"${PREFIX}/run_0010.py"' in line]
        self.assertEqual(len(commands), 2)
        self.assertIn('--max-iterations 2 ', commands[0])
        self.assertIn('--max-iterations 20000 ', commands[1])
        self.assertTrue(all('--num-envs 1024' in line and '--seed 2026' in line for line in commands))
        self.assertNotIn('execute_pair.sh', script)
        self.assertNotIn('--allow-incomplete', script)
        self.assertNotIn('--resume', script)


if __name__ == '__main__':
    unittest.main(verbosity=2)
