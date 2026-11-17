import unittest
import numpy as np
from src.dl.data import CausalDataset, PreparedData, Scale, validate_raw


class TestCausalData(unittest.TestCase):
    def test_windows_reset_and_mask_startup(self):
        x = np.arange(6, dtype=np.float32)[:, None]
        ds = CausalDataset(x, np.zeros((6, 3)), [0, 0, 0, 3, 3, 3], window=4)
        np.testing.assert_array_equal(ds[3][0].numpy(), [[0, 0], [0, 0], [0, 0], [3, 1]])
        np.testing.assert_array_equal(ds[5][0].numpy()[:, 0], [0, 3, 4, 5])

    def test_future_cannot_change_past(self):
        x = np.arange(8, dtype=np.float32)[:, None]
        ds = CausalDataset(x.copy(), np.zeros((8, 3)), np.zeros(8), window=4)
        before = ds[3][0].clone()
        ds.x[4:] = 9999
        np.testing.assert_array_equal(before, ds[3][0])

    def test_scaler_does_not_fit_on_holdout(self):
        scale = Scale.fit(np.array([[1, 4], [3, 4]], dtype=float))
        scale.transform(np.array([[1000, -1000]], dtype=float))
        np.testing.assert_array_equal(scale.mean, [2, 4])
        np.testing.assert_array_equal(scale.std, [1, 1])

    def test_real_schema_alignment_and_gap_resets(self):
        data = PreparedData()
        self.assertEqual(sum(len(p.y) for p in data.parts.values()), 38967)
        self.assertEqual(len(data.segments), 44)
        self.assertEqual(len(data.feature_names), 142)
        for p in data.parts.values():
            self.assertEqual(p.starts[0], 0)
            self.assertTrue(np.isfinite(p.engineered).all())
            reset = np.r_[True, np.diff(p.starts) != 0]
            self.assertTrue((p.dt[reset] == 0).all())
            bad = p.frame.copy()
            bad.loc[1, "timestamp"] = bad.loc[0, "timestamp"]
            with self.assertRaises(ValueError):
                validate_raw(bad)


if __name__ == "__main__":
    unittest.main()
