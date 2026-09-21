import ast
import json
from pathlib import Path
import unittest

import cv2
import numpy as np


NOTEBOOK = Path(__file__).resolve().parents[1] / 'notebooks' / 'apple_leaf_analysis.ipynb'


class FFTTests(unittest.TestCase):
    def setUp(self):
        notebook = json.loads(NOTEBOOK.read_text())
        cell = next(c for c in notebook['cells'] if c.get('id') == 'fft-funcoes')
        self.ns = dict(np=np, cv2=cv2, JANELA_HANN_ESPECTRO=True)
        exec(compile(''.join(cell['source']), str(NOTEBOOK), 'exec'), self.ns)

    def test_identity_even_odd_and_color(self):
        rng = np.random.default_rng(2)
        for shape in [(64, 96, 3), (63, 95, 3)]:
            image = rng.integers(0, 256, shape, dtype=np.uint8)
            result = self.ns['processar_frequencias'](image)
            np.testing.assert_array_equal(result['saida'], image)
            self.assertLess(result['erro_roundtrip'], 1e-10)
            self.assertLess(result['residuo_imaginario'], 1e-10)

    def test_notch_rejects_target_sinusoid_and_preserves_other_frequency(self):
        x = np.arange(96)
        wave = 127 + 40 * np.cos(2 * np.pi * x / 8) + 20 * np.cos(2 * np.pi * x / 3)
        gray = np.tile(np.rint(wave).astype(np.uint8), (64, 1))
        image = np.repeat(gray[..., None], 3, axis=2)
        result = self.ns['processar_frequencias'](image, [(0.125, 0)], .01)
        before = np.abs(np.fft.fft2(gray.astype(float)))
        after = np.abs(np.fft.fft2(result['saida'][..., 0].astype(float)))
        self.assertLess(after[0, 12], before[0, 12] * .02)
        self.assertGreater(after[0, 32], before[0, 32] * .95)
        self.assertLess(abs(float(result['saida'].mean()) - float(image.mean())), 1)

    def test_symmetry_including_nyquist_odd_shapes(self):
        rng = np.random.default_rng(3)
        for shape in [(64, 96, 3), (63, 95, 3)]:
            image = rng.integers(30, 220, shape, dtype=np.uint8)
            for peaks in [[(.12, -.17)], [(-.5, .2)]]:
                result = self.ns['processar_frequencias'](image, peaks, .02)
                self.assertLess(result['residuo_imaginario'], 1e-10)
                self.assertTrue(np.isfinite(result['mascara']).all())
                self.assertTrue((result['mascara'] >= 0).all())
                self.assertTrue((result['mascara'] <= 1).all())

    def test_gaussian_candidate_preserves_dc_and_smooths(self):
        process = self.ns['processar_frequencias']
        uniform = np.full((63, 95, 3), 127, np.uint8)
        np.testing.assert_array_equal(process(uniform, corte=.2)['saida'], uniform)
        image = np.random.default_rng(4).integers(30, 220, (64, 96, 3), dtype=np.uint8)
        result = process(image, corte=.2)
        self.assertLess(result['saida'].std(), image.std())
        self.assertLess(result['residuo_imaginario'], 1e-10)
        for cutoff in [0, -.1, .6, float('nan')]:
            with self.assertRaises(ValueError):
                process(image, corte=cutoff)

    def test_invalid_notches_rejected(self):
        for peaks, width in [([(0, 0)], .01), ([(.6, 0)], .01),
                             ([(float('nan'), 0)], .01), ([(.1, 0)], 0)]:
            with self.assertRaises(ValueError):
                self.ns['mascara_notch']((64, 64), peaks, width)

    def test_optional_filter_requires_reason_and_peaks(self):
        self.ns.update(CLASSES={'test': 'Test'}, PICOS_NOTCH={'test': []},
                       APLICAR_FILTRO_FREQUENCIAL=True, JUSTIFICATIVA_FREQUENCIAL='')
        with self.assertRaisesRegex(ValueError, 'JUSTIFICATIVA'):
            self.ns['analisar_fft']()
        self.ns['JUSTIFICATIVA_FREQUENCIAL'] = 'Periodic artifact'
        with self.assertRaisesRegex(ValueError, 'picos'):
            self.ns['analisar_fft']()


if __name__ == '__main__':
    unittest.main()
