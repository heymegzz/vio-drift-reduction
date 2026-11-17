import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch

from src.dl.models import CausalConv, build_model
from src.dl.train import load_neural, seed_everything


class TestNeuralModels(unittest.TestCase):
    def setUp(self):
        seed_everything(11)

    def test_shapes_gradients_and_initial_zero(self):
        for name in ["mlp", "raw_mlp", "tcn", "gru"]:
            cfg = {"model": name, "window": 20, "hidden": 8, "dropout": 0.0}
            sequence = name in {"tcn", "gru"}
            x = torch.randn(3, 20, 9) if sequence else torch.randn(3, 8)
            if sequence:
                x[:, :, -1] = 1
            model = build_model(cfg, x.shape[-1])
            output = model(x)
            self.assertEqual(tuple(output.shape), (3, 3))
            self.assertTrue(torch.equal(output, torch.zeros_like(output)))
            (output - 1).square().mean().backward()
            self.assertTrue(torch.isfinite(model.head.weight.grad).all())
            self.assertGreater(model.head.bias.grad.abs().sum().item(), 0)

    def test_convolution_is_causal(self):
        conv = CausalConv(2, 4, 4).eval()
        x = torch.randn(2, 2, 30)
        before = conv(x).detach()
        x[:, :, 15:] = 100
        torch.testing.assert_close(before[:, :, :15], conv(x)[:, :, :15])

    def test_gru_ignores_padding_values(self):
        model = build_model({"model": "gru", "hidden": 8, "dropout": 0.0}, 9).eval()
        torch.nn.init.normal_(model.head.weight)
        x = torch.randn(2, 10, 9)
        x[:, :, -1] = 0
        x[:, -3:, -1] = 1
        before = model(x).detach()
        x[:, :7, :-1] = 10000
        torch.testing.assert_close(before, model(x))

    def test_checkpoint_roundtrip_nonzero_predictions(self):
        cfg = {"model": "tcn", "window": 20, "hidden": 8, "dropout": 0.0}
        model = build_model(cfg, 9).eval()
        torch.nn.init.normal_(model.head.weight)
        x = torch.randn(2, 20, 9)
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "model.pt"
            torch.save({"config": cfg, "inputs": 9, "state_dict": model.state_dict()}, path)
            loaded, _ = load_neural(path)
            torch.testing.assert_close(model(x), loaded(x))

    def test_receptive_field_covers_declared_window(self):
        for window in [1, 100, 200, 400]:
            model = build_model({"model": "tcn", "window": window}, 9)
            self.assertGreaterEqual(model.receptive_field, window)

    def test_positive_control_can_learn_known_signal(self):
        # A negative flight result must not conceal a broken optimization path.
        # Synthetic data is used only in this test, never as reported flight data.
        for family in ["mlp", "tcn", "gru"]:
            seed_everything(17)
            sequence = family in {"tcn", "gru"}
            x = torch.randn(128, 8, 9) if sequence else torch.randn(128, 8)
            if sequence:
                x[:, :, -1] = 1
            current = x[:, -1, :] if sequence else x
            y = torch.stack([.2 * current[:, 0], -.1 * current[:, 1], .15 * current[:, 2]], dim=1)
            model = build_model({"model": family, "window": 8, "hidden": 16, "dropout": 0.0}, x.shape[-1])
            optimizer = torch.optim.Adam(model.parameters(), lr=.005)
            initial = y.square().mean().item()
            for _ in range(100):
                optimizer.zero_grad()
                loss = (model(x) - y).square().mean()
                loss.backward()
                optimizer.step()
            self.assertLess((model(x) - y).square().mean().item(), initial * .2, family)


if __name__ == "__main__":
    unittest.main()
