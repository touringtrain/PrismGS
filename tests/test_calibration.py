"""Calibration protocol and rates, independent of CUDA availability."""
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from prismgs.codec import ATTRIBUTES, CHANNELS, write_json
from prismgs.pipeline import calibrate
from prismgs.quality import collect_profile, predict_quality, validate_qps


class CalibrationTests(unittest.TestCase):
    def test_single_attribute_sweep_and_full_sequence_rates(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            videos, output = root/'videos', root/'calibration'
            # Unequal groups ensure rates divide by frames, not groups.
            write_json(videos/'metadata.json', {'groups': [
                {'index': 0, 'frames': [7, 8]}, {'index': 1, 'frames': [9]}]})
            for qp in [0, 40]:
                for group in [0, 1]:
                    for channel in CHANNELS:
                        path = videos/f'png_all_{qp}/group{group}/{channel}.mp4'
                        path.parent.mkdir(parents=True, exist_ok=True)
                        path.write_bytes(b'x' * (300 if qp == 0 else 150))
            configurations = []

            def decode_stub(videos, checkpoints, config, workers, device='cpu'):
                self.assertEqual(device, 'cuda')
                configurations.append(config.copy())

            def render_stub(checkpoints, dataset, folder, frames, cameras, resolution, background, split):
                self.assertEqual(frames, [7])
                self.assertEqual(cameras, [2, 4])
                quality = 40.0 if len(configurations) == 1 else 36.0
                metrics = {'psnr': quality, 'ssim': 0.95, 'split': split,
                           'resolution': resolution, 'views': [
                               {'frame': 7, 'camera': c, 'psnr': quality, 'ssim': 0.95}
                               for c in cameras]}
                write_json(folder/'metrics.json', metrics)
                return metrics

            with patch('prismgs.pipeline.decode', side_effect=decode_stub), \
                    patch('prismgs.pipeline.evaluate', side_effect=render_stub):
                profile = calibrate(videos, root/'data', output, [0, 40], cameras=[2, 4], decode_device='cuda')
            self.assertEqual(len(configurations), 6)
            self.assertEqual(configurations[0], {a: 0 for a in ATTRIBUTES})
            for attr, config in zip(ATTRIBUTES, configurations[1:]):
                self.assertEqual(config, {a: 40 if a == attr else 0 for a in ATTRIBUTES})
            self.assertEqual(profile['rate_frame_count'], 3)
            self.assertAlmostEqual(profile['fixed_kb'], 0.6)
            for attr, channels in ATTRIBUTES.items():
                zero, lossy = profile['attributes'][attr]
                self.assertEqual(zero['quality_factor'], 1.0)
                self.assertEqual(lossy['quality_factor'], 0.9)
                self.assertAlmostEqual(lossy['size_kb'], len(channels) * 0.1)
            self.assertAlmostEqual(predict_quality(profile, {a: 40 for a in ATTRIBUTES}), 40 * 0.9**5)
            with self.assertRaisesRegex(ValueError, 'Unmeasured QP'):
                predict_quality(profile, {a: 20 for a in ATTRIBUTES})
            with self.assertRaisesRegex(ValueError, 'fresh calibration'):
                calibrate(videos, root/'data', output, [0, 40])
            path = output/'color_qp40/metrics.json'
            bad = json.loads(path.read_text())
            bad['views'][0]['camera'] = 5
            write_json(path, bad)
            with self.assertRaisesRegex(ValueError, 'Inconsistent evaluation'):
                collect_profile(videos, output, [0, 40], root/'invalid.json')

    def test_reject_invalid_qps_before_io(self):
        for qps in [[], [10], [0, 0], [0, -1], [0, 52], [0, 1.5], [False, 40]]:
            with self.subTest(qps=qps), self.assertRaises(ValueError):
                validate_qps(qps)


if __name__ == '__main__':
    unittest.main()
