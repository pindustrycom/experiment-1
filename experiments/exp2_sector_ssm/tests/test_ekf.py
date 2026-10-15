import math

import torch

from exp2.ekf import EKFSSM


def kalman_reference(m, z, mask):
    """Plain covariance-form Kalman filter (U = 0, so the model is linear) on one sequence."""
    A, Q, R = m.A(), torch.diag(m.log_q.exp()), torch.diag(m.log_r.exp())
    d = m.d
    x, P, nll, n = torch.zeros(d, dtype=z.dtype), torch.eye(d, dtype=z.dtype), 0.0, 0
    xs = []
    for t in range(z.shape[0]):
        if t > 0:
            x, P = A @ x, A @ P @ A.T + Q
        o = mask[t].bool()
        H, b, Ro = m.H[o], m.b[o], R[o][:, o]
        e = z[t, o] - (H @ x + b)
        S = H @ P @ H.T + Ro
        K = P @ H.T @ torch.linalg.inv(S)
        nll += 0.5 * (torch.logdet(S) + e @ torch.linalg.solve(S, e) + o.sum() * math.log(2 * math.pi))
        n += int(o.sum())
        x, P = x + K @ e, (torch.eye(d, dtype=z.dtype) - K @ H) @ P
        xs.append(x)
    return torch.stack(xs), nll / n


def test_information_form_matches_kalman_filter_with_missing_values():
    torch.manual_seed(0)
    m = EKFSSM(n_obs=6, n_targets=2, d_state=4).double()
    with torch.no_grad():
        m.U.zero_()
        m.log_r.copy_(torch.randn(6, dtype=torch.double) * 0.3)
    z = torch.randn(30, 6, dtype=torch.double)
    mask = torch.ones(30, 6, dtype=torch.double)
    mask[:10, 4:] = 0  # e.g. news not collected yet
    _, _, nll, X = m(z[None], mask[None])
    xs, ref = kalman_reference(m, z, mask)
    assert torch.allclose(X[0], xs, atol=1e-8)
    assert torch.allclose(nll, ref, atol=1e-8)


def test_missing_values_are_ignored_not_zeros():
    torch.manual_seed(1)
    m = EKFSSM(n_obs=5, n_targets=1, d_state=4)
    z = torch.randn(1, 20, 5)
    mask = torch.ones_like(z)
    mask[:, :, -1] = 0
    z2 = z.clone()
    z2[:, :, -1] = 100.0  # garbage in an unobserved column changes nothing
    a, b = m(z, mask), m(z2, mask)
    assert torch.allclose(a[0], b[0]) and torch.allclose(a[2], b[2])


def test_jacobian_matches_autograd():
    torch.manual_seed(2)
    m = EKFSSM(n_obs=3, n_targets=1, d_state=4)
    x = torch.randn(1, 4)
    _, J = m.f(x, m.A())
    Jauto = torch.autograd.functional.jacobian(lambda v: m.f(v[None], m.A())[0][0], x[0])
    assert torch.allclose(J[0], Jauto, atol=1e-5)
