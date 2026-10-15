import torch

from exp2.config import ModelConfig
from exp2.model import S4D, SectorSSM, gaussian_nll


def test_convolution_matches_recurrence():
    torch.manual_seed(0)
    layer = S4D(4, 8, 0.002, 0.2, 0.0).eval()
    u = torch.randn(2, 60, 4)
    assert torch.allclose(layer(u), layer.step_all(u), atol=1e-4)


def test_layer_is_causal():
    torch.manual_seed(1)
    layer = S4D(3, 6, 0.01, 0.1, 0.0).eval()
    u = torch.randn(1, 40, 3)
    v = u.clone()
    v[:, 30:] += 5
    assert torch.allclose(layer(u)[:, :30], layer(v)[:, :30], atol=1e-5)


def test_prune_keeps_requested_fraction_and_masks_weakest():
    torch.manual_seed(0)
    m = SectorSSM(5, 2, ModelConfig(d_model=8, n_modes=16, n_layers=2, dropout=0.0))
    before = [layer.importance().clone() for layer in m.layers]
    m.prune(0.25)
    active, total = m.modes()
    assert total == 2 * 8 * 16 and abs(active - total * 0.25) <= 2
    for layer, imp in zip(m.layers, before):
        kept, dropped = imp[layer.mask.bool()], imp[~layer.mask.bool()]
        assert kept.min() >= dropped.max()
        assert (layer.importance()[~layer.mask.bool()] == 0).all()
    # pruning again only removes more
    m.prune(0.5)
    assert m.modes()[0] <= active


def test_nll_ignores_unknown_targets_without_nan_gradients():
    mu = torch.zeros(4, 2, requires_grad=True)
    lv = torch.zeros(4, 2, requires_grad=True)
    y = torch.tensor([[1.0, float("nan")], [0.0, 1.0], [float("nan"), float("nan")], [2.0, 0.0]])
    loss = gaussian_nll(mu, lv, y)
    loss.backward()
    assert torch.isfinite(loss) and torch.isfinite(mu.grad).all()
