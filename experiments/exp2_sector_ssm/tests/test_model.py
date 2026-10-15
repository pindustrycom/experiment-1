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


def _set_poles(layer, r, theta, g=1.0):
    """Force the modes of a 1-channel layer to poles r e^{i theta} with gain g (dt = 1, so A = log r + i theta)."""
    with torch.no_grad():
        layer.log_dt.fill_(0.0)
        layer.a_re.copy_(torch.log(-torch.log(r)))
        layer.a_im.copy_(theta)
        a = torch.complex(torch.log(r), theta)
        layer.c.copy_(torch.view_as_real((g * a / (a.exp() - 1)).to(torch.cfloat)))  # C (e^A - 1) / A = g


def test_resonance_index_reduces_to_h2_for_white_input():
    layer = S4D(1, 3, 0.01, 0.1, 0.0)
    _set_poles(layer, torch.tensor([[0.5, 0.8, 0.9]]), torch.tensor([[0.3, 1.0, 2.0]]))
    white = torch.full((1, 512), 2.0)
    res, h2 = layer.scores("resonance", white), layer.scores("h2")
    assert torch.allclose(res, 2.0 * h2, rtol=1e-3)
    assert torch.allclose(layer.scores("flat", white), 2.0 * h2, rtol=1e-6)


def test_resonance_index_prefers_modes_tuned_to_the_input():
    layer = S4D(1, 2, 0.01, 0.1, 0.0)
    _set_poles(layer, torch.tensor([[0.95, 0.95]]), torch.tensor([[0.5, 2.5]]))
    L = 256
    u = torch.sin(0.5 * torch.arange(4 * L)).reshape(1, -1, 1)  # input at the first mode's frequency
    spec = (torch.fft.fft(u[0, :, 0].reshape(4, L), dim=1).abs() ** 2).mean(0)[None] / L
    sc = layer.scores("resonance", spec)[0]
    assert sc[0] > 10 * sc[1]
    assert torch.allclose(layer.scores("h2")[0, 0], layer.scores("h2")[0, 1])  # data-free criterion can't tell them apart


def test_input_spectra_shapes_and_resonance_pruning():
    torch.manual_seed(0)
    m = SectorSSM(5, 2, ModelConfig(d_model=8, n_modes=16, n_layers=2, dropout=0.0))
    spectra = m.input_spectra(torch.randn(1, 600, 5), seg=128)
    assert len(spectra) == 2 and spectra[0].shape == (8, 128) and (spectra[0] >= 0).all()
    m.prune(0.25, "resonance", spectra)
    assert abs(m.modes()[0] - 0.25 * m.modes()[1]) <= 2
