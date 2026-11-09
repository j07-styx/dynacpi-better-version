import json
import numpy as np
from scipy.optimize import minimize
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from dynacpi.adaptation import adapt

def run():
    rng = np.random.default_rng(41104)
    checks = []
    max_gap = 0.
    for _ in range(200):
        n = int(rng.integers(1, 21))
        s = rng.normal(6, 1, n)
        g = s + rng.normal(0, 1, n)
        y = rng.normal(6, 2, n)
        k = rng.uniform(0, 1, n)
        w0 = float(rng.uniform())
        lb, lt = 10 ** rng.uniform(-2, 1, 2)
        a = adapt(6., 7., s, g, y, k, w0, lb, lt)
        d = g - s
        x = d - (k @ d / k.sum())
        u = y - s - w0 * d
        def obj(z):
            b, t = z
            return k @ ((u - b - t * x)**2) + lb*b*b + lt*t*t
        def jac(z):
            b, t = z
            r = u - b - t*x
            return np.array([-2*k@r + 2*lb*b, -2*k@(x*r) + 2*lt*t])
        fit = minimize(obj, [0., 0.], jac=jac, bounds=[(-2.,2.),(-w0,1-w0)],
                       method='L-BFGS-B', options={'ftol':1e-14,'gtol':1e-10,'maxiter':1000})
        gap = abs(obj([a['offset'], a['theta']]) - fit.fun)
        assert gap < 1e-8, gap
        max_gap = max(max_gap, gap)
        p = rng.permutation(n)
        ap = adapt(6., 7., s[p], g[p], y[p], k[p], w0, lb, lt)
        assert np.isclose(a['mean'], ap['mean'], atol=1e-12)
        ashift = adapt(6., 7., s, g, y+4., k, w0, lb, lt)
        assert np.isclose(a['theta'], ashift['theta'], atol=1e-12)
        assert np.isclose(a['differential'], ashift['differential'], atol=1e-12)
        assert abs(a['mean']-a['base']) <= 3. + 1e-12
    checks.extend(['200 constrained optimizer comparisons', 'support permutation',
                   'common label-shift invariance of differential', 'total correction bound'])
    a = adapt(6.,7.,[],[],[],[])
    assert a['mean'] == a['base'] == 6.5
    a = adapt(6.,7.,[6.],[7.],[9.],[1.])
    assert a['theta'] == 0.
    a = adapt(6.,7.,[5.,6.,7.],[5.5,6.5,7.5],[7.,8.,9.],[1.,.4,.2])
    assert abs(a['theta']) < 1e-12
    a = adapt(6.,6.,[5.,6.],[5.,6.],[7.,8.],[1.,1.])
    assert a['differential'] == 0. and a['offset'] > 0.
    a = adapt(6.,7.,[5.,6.],[5.3,7.],[7.,8.],[1e-14,1e-14])
    assert abs(a['mean']-a['base']) < 1e-10
    checks.extend(['empty support', 'single support', 'constant disagreement',
                   'missing geometry retains calibration', 'vanishing evidence'])
    # Numerically demonstrate the audited padding operation, not a full trained model.
    messages = np.array([2.,4.])
    legacy = [np.pad(messages,(0,w-2)).mean() for w in (2,4)]
    fixed = [np.pad(messages,(0,w-2)).sum()/2 for w in (2,4)]
    assert legacy == [3.,1.5] and fixed == [3.,3.]
    checks.append('padded neighbor-average counterexample')
    print(json.dumps({'passed':checks, 'optimizer_trials':200,
                      'max_objective_difference':max_gap,
                      'legacy_means':legacy,'masked_means':fixed,
                      'scope':'NumPy/SciPy adapter algebra and isolated padding arithmetic only; no neural model trained or executed'}, indent=2))


if __name__ == '__main__':
    run()
