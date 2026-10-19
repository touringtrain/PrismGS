"""Run with: python -m unittest discover -s tests -v"""
import contextlib
import importlib.util
import io
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]


class ReleaseRegressionTests(unittest.TestCase):
    def test_dp_reports_actual_sizes_within_budget(self):
        spec = importlib.util.spec_from_file_location('quality_model', ROOT / 'core/quality_model.py')
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        with contextlib.redirect_stdout(io.StringIO()):
            optimizer = module.GaussianCompressionOptimizer()
            frontier = optimizer.generate_pareto_frontier(bitrate_range=(200, 400), method='dp')
        self.assertTrue(frontier)
        for config, quality, size in frontier:
            self.assertAlmostEqual(size, float(optimizer.calculate_bitrate(config)))
            self.assertGreaterEqual(size, 200)
            self.assertLessEqual(size, 400)
        for left, right in zip(frontier, frontier[1:]):
            self.assertLess(left[2], right[2])
            self.assertLessEqual(left[1], right[1])

    def test_bundled_single_scene_evaluation(self):
        with tempfile.TemporaryDirectory() as directory:
            result = subprocess.run(
                [sys.executable, str(ROOT / 'evaluation/overall_performance.py')],
                cwd=directory, env={**os.environ, 'MPLBACKEND': 'Agg'},
                capture_output=True, text=True, timeout=90,
            )
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn('PrismGS: FPS =', result.stdout)


if __name__ == '__main__':
    unittest.main()
