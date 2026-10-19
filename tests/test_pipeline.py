"""Verificações pequenas dos riscos reais: máscaras, descritores e separação."""
import unittest

import cv2
import numpy as np
import pandas as pd

from scripts.image_processing import (color_features, crop_roi, glcm, leaf_features,
                                      masked_glcm, segment_leaf, shape_features, texture_features)
from scripts.compare_texture_descriptors import texture_columns
from scripts.run_experiments import assign_partitions, classification_metrics, duplicate_groups


class ProcessingTests(unittest.TestCase):
    def setUp(self):
        self.image = np.full((64, 64, 3), 180, np.uint8)
        cv2.ellipse(self.image, (32, 32), (17, 24), 0, 0, 360, (30, 130, 40), -1)
        self.mask = np.zeros((64, 64), np.uint8)
        cv2.ellipse(self.mask, (32, 32), (17, 24), 0, 0, 360, 1, -1)
        self.mask = self.mask.astype(bool)

    def test_full_mask_matches_reference_glcm(self):
        gray = np.random.default_rng(42).integers(0, 256, (23, 29), dtype=np.uint8)
        np.testing.assert_allclose(masked_glcm(gray, np.ones_like(gray, bool)), glcm(gray), atol=1e-14)

    def test_background_cannot_change_masked_features(self):
        changed = self.image.copy()
        changed[~self.mask] = (255, 0, 255)
        first = leaf_features(self.image, self.mask)
        second = leaf_features(changed, self.mask)
        np.testing.assert_allclose(list(first.values()), list(second.values()), atol=1e-14)
        self.assertEqual(len(first), 136)

    def test_masks_find_synthetic_leaf_and_preserve_hole(self):
        image = self.image.copy()
        image[30:35, 30:35] = 180
        for method in ("otsu", "kmeans"):
            mask = segment_leaf(image, method)
            self.assertTrue(mask[32, 32])
            self.assertFalse(mask[0, 0])
            self.assertGreater((mask & self.mask).sum() / (mask | self.mask).sum(), .95)
            np.testing.assert_array_equal(mask, segment_leaf(image, method))

    def test_hue_wrap_and_histograms(self):
        hsv = np.zeros((8, 8, 3), np.uint8)
        hsv[:, :, 1:] = 255
        hsv[4:, :, 0] = 179
        features = color_features(cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR), np.ones((8, 8), bool), "hsv")
        self.assertGreater(features["hsv_0_moment_1"], .99)
        self.assertLess(features["hsv_0_moment_2"], .01)
        for channel in range(3):
            self.assertAlmostEqual(sum(features[f"hsv_{channel}_hist_{i}"] for i in range(16)), 1)

    def test_invalid_inputs_and_degenerate_regions(self):
        gray = np.zeros((16, 16), np.uint8)
        with self.assertRaises(ValueError):
            masked_glcm(gray, np.zeros_like(gray, bool))
        with self.assertRaises(ValueError):
            masked_glcm(gray, np.ones_like(gray, bool), distances=(0,))
        with self.assertRaises(ValueError):
            crop_roi(gray, (15, 15, 2, 2))
        with self.assertRaises(ValueError):
            segment_leaf(self.image, "unknown")
        mask = np.zeros_like(gray, bool)
        mask[8, 8] = True
        with self.assertRaises(ValueError):
            shape_features(mask)
        with self.assertRaises(ValueError):
            texture_features(gray, mask)


class ProtocolTests(unittest.TestCase):
    def test_texture_columns_separate_glcm_and_lbp(self):
        frame = pd.DataFrame(columns=["texture_contrast", "texture_energy",
                                     "texture_lbp_0", "texture_lbp_1", "hsv_0_hist_0"])
        self.assertEqual(texture_columns(frame, "glcm"),
                         ["texture_contrast", "texture_energy"])
        self.assertEqual(texture_columns(frame, "lbp"),
                         ["texture_lbp_0", "texture_lbp_1"])

    def test_groups_and_history_never_cross_partitions(self):
        rows = [{"class": label, "path": f"{label}/{i:03d}", "group": f"{label}/{i//2}"}
                for label in "abcd" for i in range(80)]
        frame = assign_partitions(pd.DataFrame(rows))
        self.assertEqual(frame.groupby("group").split.nunique().max(), 1)
        self.assertTrue((frame.groupby("class").head(6).split == "calibration").all())
        dev = frame[frame.split == "development"]
        self.assertEqual(dev.groupby("group").fold.nunique().max(), 1)
        pd.testing.assert_frame_equal(frame, assign_partitions(pd.DataFrame(rows)))
        for _, part in dev.groupby("fold"):
            self.assertEqual(set(part["class"]), set("abcd"))

    def test_duplicates_group_transitively(self):
        frame = pd.DataFrame({"path": ["a", "b", "c"], "class": ["leaf"]*3,
                              "dhash": ["0", "1", "f"], "pixel_sha256": ["x", "y", "z"]})
        groups, pairs = duplicate_groups(frame, [np.zeros((2, 2))]*3)
        self.assertEqual(len(set(groups)), 1)
        self.assertEqual(len(pairs), 3)
        frame.loc[1, ["class", "pixel_sha256"]] = ["other", "x"]
        with self.assertRaises(ValueError):
            duplicate_groups(frame, [np.zeros((2, 2))]*3)

    def test_sensitivity_and_specificity(self):
        score, matrix, rows = classification_metrics(["a", "a", "b", "b"], ["a", "b", "b", "b"], ["a", "b"])
        self.assertAlmostEqual(score, (2/3 + .8)/2)
        np.testing.assert_array_equal(matrix, [[1, 1], [0, 2]])
        self.assertEqual(rows[0]["sensitivity"], .5)
        self.assertEqual(rows[0]["specificity"], 1)


class PersistenceTests(unittest.TestCase):
    def test_hashes_remain_text_and_test_cannot_be_overwritten(self):
        from pathlib import Path
        from tempfile import TemporaryDirectory
        from scripts.run_experiments import read_partitions, evaluate
        with TemporaryDirectory() as temporary:
            output = Path(temporary)
            (output / 'partitions.csv').write_text('path,dhash,sha256,pixel_sha256\na,0000000000000001,001,002\n')
            row = read_partitions(output).iloc[0]
            self.assertEqual(row.dhash, '0000000000000001')
            self.assertEqual(row.sha256, '001')
            (output / 'test_started.json').write_text('{}')
            with self.assertRaises(FileExistsError):
                evaluate(output)


class ClassifierTests(unittest.TestCase):
    def test_both_models_score_multiclass_string_labels(self):
        from sklearn.metrics import get_scorer
        from scripts.run_experiments import model_grid
        x = np.repeat(np.eye(4), 8, axis=0)
        y = np.repeat(["sarna", "podridao", "ferrugem", "saudavel"], 8)
        x[0, 0] = np.nan
        for name in ("svm", "rf"):
            model, _ = model_grid(name)
            model.fit(x, y)
            self.assertGreater(get_scorer("f1_macro")(model, x, y), .9)


if __name__ == "__main__":
    unittest.main()
