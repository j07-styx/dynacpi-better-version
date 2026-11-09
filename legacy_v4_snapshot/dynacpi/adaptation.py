import numpy as np


def adapt(sq, gq, s, g, y, k, w0=.5, lb=5., lt=.1, B=2., C=1.):
    s, g, y, k = [np.asarray(z, dtype=np.float64) for z in (s, g, y, k)]
    assert s.shape == g.shape == y.shape == k.shape
    assert all(np.isfinite(z).all() for z in (s, g, y, k))
    assert np.isfinite([sq, gq, w0, lb, lt, B, C]).all()
    assert np.all(k >= 0) and 0 <= w0 <= 1 and min(lb, lt, B, C) > 0
    d = g - s
    dq = gq - sq
    base = sq + w0 * dq
    W = float(k.sum())
    if W == 0:
        return dict(mean=base, base=base, offset=0., theta=0., differential=0., V=0., W=0.)
    db = float(k @ d / W)
    x = d - db
    u = y - s - w0 * d
    b = float(np.clip(k @ u / (W + lb), -B, B))
    V = float(k @ (x * x))
    theta = float(np.clip(k @ (x * u) / (V + lt), -w0, 1 - w0))
    delta = float(np.clip(theta * (dq - db), -C, C))
    return dict(mean=base + b + delta, base=base, offset=b, theta=theta,
                differential=delta, V=V, W=W)

