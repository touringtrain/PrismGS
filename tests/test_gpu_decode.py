"""Actual CUDA round trip, including partial groups and varying point counts."""
from pathlib import Path
import shutil
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
try:
    import torch
except ImportError:
    torch = None

from prismgs.codec import ATTRIBUTES, decode, encode, latest_ply, pack, read_ply, read_video, write_ply


@unittest.skipUnless(torch is not None and torch.cuda.is_available() and shutil.which('ffmpeg'),
                     'CUDA and FFmpeg required')
class GPUDecodeTests(unittest.TestCase):
    def test_cuda_matches_cpu_with_partial_groups_and_mixed_qps(self):
        with tempfile.TemporaryDirectory(prefix='prismgs cuda ') as directory:
            root = Path(directory)
            rng = np.random.default_rng(17)
            for frame, count in [(7, 11), (8, 70), (9, 18)]:
                cloud = rng.normal(size=(count, 17)).astype(np.float32)
                cloud[:, 3:6] = 0
                cloud[:, 9] = -2  # Constant opacity and raw log-scale values.
                cloud[:, 10:13] -= 5
                write_ply(root/f'original/{frame}/point_cloud/iteration_1/point_cloud.ply', cloud)
            pack(root/'original', root/'images', 7, 10, 2)
            encode(root/'images', root/'videos', [0, 40])
            for config in [{a: 0 for a in ATTRIBUTES}, dict(zip(ATTRIBUTES, [40, 0, 40, 0, 40]))]:
                decode(root/'videos', root/'cpu', config, 2)
                stats = decode(root/'videos', root/'gpu', config, 2, device='cuda:0')
                self.assertEqual(stats['device'], 'cuda:0')
                self.assertEqual(stats['groups'], 2)
                self.assertEqual(stats['frames'], 3)
                self.assertGreater(stats['cuda_upload_and_reconstruct_ms'], 0)
                for frame in [7, 8, 9]:
                    cpu = read_ply(latest_ply(root/'cpu', frame))
                    gpu = read_ply(latest_ply(root/'gpu', frame))
                    np.testing.assert_allclose(gpu, cpu, rtol=0, atol=2e-6)
                    np.testing.assert_array_equal(gpu[:, 9], -2)
                    np.testing.assert_array_equal(gpu[:, 3:6], 0)
                decode(root/'videos', root/'gpu_serial', config, 1, device='cuda:0')
                for frame in [7, 8, 9]:
                    self.assertEqual(latest_ply(root/'gpu', frame).read_bytes(),
                                     latest_ply(root/'gpu_serial', frame).read_bytes())
            def truncated(path):
                return read_video(path)[:-1]
            with patch('prismgs.codec.read_video', side_effect=truncated):
                with self.assertRaisesRegex(ValueError, 'frame count or shape mismatch'):
                    decode(root/'videos', root/'truncated', config, 2, device='cuda')
            self.assertFalse((root/'truncated/compression.json').exists())
            self.assertFalse(list((root/'truncated').rglob('*.ply')))


class DeviceValidationTests(unittest.TestCase):
    @unittest.skipIf(torch is None, 'PyTorch required')
    def test_explicit_cuda_never_silently_falls_back(self):
        with patch('torch.cuda.is_available', return_value=False):
            with self.assertRaisesRegex(RuntimeError, 'CUDA is unavailable'):
                decode('unused', 'unused', {a: 0 for a in ATTRIBUTES}, device='cuda')


if __name__ == '__main__':
    unittest.main()
