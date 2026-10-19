import itertools
import json
from pathlib import Path
import shutil
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
from prismgs.codec import ATTRIBUTES, HIGH, pack, encode, decode, read_ply, write_ply, read_video
from prismgs.quality import frontier


class PipelineTests(unittest.TestCase):
    @unittest.skipUnless(shutil.which('ffmpeg'), 'FFmpeg required')
    def test_actual_codec_roundtrip_partial_groups_and_constant_attributes(self):
        with tempfile.TemporaryDirectory(prefix='prismgs space ') as tmp:
            root = Path(tmp)
            original = {}
            rng = np.random.default_rng(7)
            for frame, count in [(7, 10), (8, 70), (9, 15)]:
                data = rng.normal(size=(count, 17)).astype(np.float32)
                data[:,3:6] = 0
                data[:,9] = -2.0  # constant opacity must not divide by zero
                original[frame] = data
                write_ply(root/f'checkpoint/{frame}/point_cloud/iteration_4/point_cloud.ply',data)
            pack(root/'checkpoint',root/'images',7,10,2)
            encode(root/'images',root/'videos',[0,40])
            decode(root/'videos',root/'decoded',{a:0 for a in ATTRIBUTES},2)
            for frame, expected in original.items():
                actual = read_ply(root/f'decoded/{frame}/point_cloud/iteration_1/point_cloud.ply')
                self.assertEqual(actual.shape,expected.shape)
                for i in range(17):
                    limit = 65535 if i < 3 else 255
                    tolerance = float(np.ptp(expected[:,i]))/limit + 2e-6
                    np.testing.assert_allclose(actual[:,i],expected[:,i],atol=tolerance,rtol=0)
            metadata = json.loads((root/'videos/metadata.json').read_text())
            self.assertEqual([len(g['frames']) for g in metadata['groups']],[2,1])
            self.assertEqual(metadata['groups'][0]['side'],16)

            # Mixed QPs must select each attribute's own video channels. Check
            # serial/parallel equality across both full and partial groups.
            config = dict(zip(ATTRIBUTES, [40, 0, 40, 0, 40]))
            paths = []
            def record_video(path):
                paths.append(Path(path))
                return read_video(path)
            with patch('prismgs.codec.read_video', side_effect=record_video):
                decode(root/'videos', root/'parallel', config, 2)
            for attr, channels in ATTRIBUTES.items():
                for group in [0, 1]:
                    for channel in channels:
                        self.assertIn(root/f'videos/png_all_{config[attr]}/group{group}/{channel}.mp4', paths)
            decode(root/'videos', root/'serial', config, 1)
            for frame in original:
                relative = f'{frame}/point_cloud/iteration_1/point_cloud.ply'
                self.assertEqual((root/'parallel'/relative).read_bytes(), (root/'serial'/relative).read_bytes())
            # Position high bytes must remain identical/lossless at every QP.
            for group in [0, 1]:
                for channel in HIGH:
                    relative = f'group{group}/{channel}.mp4'
                    self.assertEqual((root/'videos/png_all_0'/relative).read_bytes(),
                                     (root/'videos/png_all_40'/relative).read_bytes())

            def short_video(path):
                return read_video(path)[:-1]
            with patch('prismgs.codec.read_video', side_effect=short_video):
                with self.assertRaisesRegex(ValueError, 'frame count or shape mismatch'):
                    decode(root/'videos', root/'truncated', config, 2)
            self.assertFalse((root/'truncated/compression.json').exists())
            self.assertFalse(list((root/'truncated').rglob('*.ply')))

    def test_fine_bin_dp_matches_exhaustive_frontier(self):
        profile = {'base_psnr':40,'fixed_kb':2,'attributes':{}}
        for i,a in enumerate(ATTRIBUTES):
            profile['attributes'][a] = [{'qp':0,'psnr':40,'size_kb':2**(i+1)},
                                        {'qp':10,'psnr':40-(i+1)*0.7,'size_kb':1}]
        exhaustive = []
        for choices in itertools.product(*profile['attributes'].values()):
            size = 2+sum(c['size_kb'] for c in choices)
            quality = 40*np.prod([c['psnr']/40 for c in choices])
            exhaustive.append((size,quality))
        best, expected = -1, []
        for size,quality in sorted(exhaustive,key=lambda x:(x[0],-x[1])):
            if quality > best:
                expected.append((size,quality)); best=quality
        actual = frontier(profile,step_kb=0.01)['points']
        np.testing.assert_allclose([(p['size_kb'],p['predicted_psnr']) for p in actual],expected)
        self.assertEqual(frontier(profile,max_kb=1)['points'],[])

    def test_invalid_inputs(self):
        with self.assertRaises(ValueError):
            frontier({},step_kb=0)
        with self.assertRaises(ValueError):
            decode('unused','unused',{'color':20})


if __name__ == '__main__':
    unittest.main()
