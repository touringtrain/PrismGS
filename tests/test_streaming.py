import json
from pathlib import Path
import tempfile
import unittest
from prismgs.streaming import evaluate_trace


class StreamingTests(unittest.TestCase):
    def test_unsorted_rd_selects_quality_and_converts_mbps_to_bytes(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root/'trace.csv').write_text('mobility_mode,Throughput\nwalking,2.4\nwalking,0.24\n')
            # At 2.4 Mbps, 10 KB/frame sustains 30 FPS; 100 KB/frame cannot.
            (root/'rd.json').write_text(json.dumps({'points':[
                {'size_kb':100,'psnr':40}, {'size_kb':5,'psnr':20}, {'size_kb':10,'psnr':30}]}))
            evaluate_trace(root/'trace.csv',root/'rd.json',root/'out.json')
            samples = json.loads((root/'out.json').read_text())['samples']
            self.assertEqual(samples[0]['psnr'],30)
            self.assertAlmostEqual(samples[0]['fps'],30)
            self.assertEqual(samples[1]['psnr'],20)
            self.assertAlmostEqual(samples[1]['fps'],6)

    def test_predicted_only_points_are_rejected(self):
        with tempfile.TemporaryDirectory() as temporary:
            root=Path(temporary)
            (root/'rd.json').write_text(json.dumps({'points':[{'size_kb':10,'predicted_psnr':35}]}))
            with self.assertRaises(ValueError):
                evaluate_trace(root/'absent.csv',root/'rd.json',root/'out.json')
