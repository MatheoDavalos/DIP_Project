import json
from pathlib import Path
import unittest

import cv2
import numpy as np


class BordasTests(unittest.TestCase):
    def setUp(self):
        path = Path(__file__).resolve().parents[1] / 'notebooks' / 'apple_leaf_analysis.ipynb'
        notebook = json.loads(path.read_text())
        self.ns = dict(np=np, cv2=cv2, LIMIAR_SOBEL=100, LIMIARES_CANNY=(50, 100))
        cell = next(c for c in notebook['cells'] if c['id'] == 'bordas-funcoes')
        exec(''.join(cell['source']), self.ns)

    def test_uniform_image_has_no_edges(self):
        image = np.full((32, 32, 3), 120, np.uint8)
        result = self.ns['detectar_bordas'](image)
        for value in result.values():
            self.assertFalse(value.any())

    def test_step_both_detectors_and_input_unchanged(self):
        image = np.zeros((32, 32, 3), np.uint8)
        image[:, 16:] = 255
        before = image.copy()
        result = self.ns['detectar_bordas'](image)
        self.assertTrue(result['sobel_binario'].any())
        self.assertTrue(result['canny'].any())
        self.assertEqual(result['canny'].dtype, bool)
        np.testing.assert_array_equal(before, image)

    def test_identity_shift_empty_and_disjoint(self):
        compare = self.ns['comparar_mapas']
        a = np.zeros((16, 16), bool)
        a[:, 5] = True
        self.assertEqual(compare(a, a, 1), (1, 1))
        self.assertEqual(compare(a, np.roll(a, 1, axis=1), 1), (0, 1))
        self.assertEqual(compare(a, np.roll(a, 5, axis=1), 1), (0, 0))
        empty = np.zeros_like(a)
        self.assertTrue(np.isnan(compare(empty, empty, 1)).all())
        self.assertEqual(compare(a, empty, 1), (0, 0))
        with self.assertRaises(ValueError):
            compare(a, a, -1)


if __name__ == '__main__':
    unittest.main()
