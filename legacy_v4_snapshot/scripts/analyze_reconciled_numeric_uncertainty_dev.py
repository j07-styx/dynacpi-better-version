"""Paired target-cluster uncertainty for matched-seed DAVIS development predictions.

Exploratory only: the four validation targets have been repeatedly inspected.
Bootstrap intervals describe resampling variability, not an independent test.
"""
import hashlib
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / 'data/engineering_pilot_primary_reconciled_20261107/rows.json'
SEEDS = (2103, 2104, 2105)
KINDS = ('mixed', 'numeric')


def load():
    source = json.loads(DATA.read_text(encoding='utf-8'))
    expected = [r for r in source if r['split'] == 'val' and r['primary_label_kind'] == 'numeric_Kd']
    key = lambda r: (r['sequence'], r['compound'])
    expected_by_key = {key(r): r for r in expected}
    assert len(expected_by_key) == len(expected) == 71
    result = {}
    for seed in SEEDS:
        for kind in KINDS:
            path = ROOT / f'research/reconciled_{kind}_sequence_seed_{seed}_dev/val_predictions.json'
            rows = json.loads(path.read_text(encoding='utf-8'))
            numeric = {key(r): r for r in rows if r['primary_label_kind'] == 'numeric_Kd'}
            assert numeric.keys() == expected_by_key.keys(), (seed, kind, 'different numeric validation pairs')
            for pair, row in numeric.items():
                assert abs(float(row['y']) - float(expected_by_key[pair]['y'])) < 1e-8
            result[(seed, kind)] = numeric
    return expected_by_key, result


def main():
    expected, predictions = load()
    clusters = defaultdict(list)
    for pair in sorted(expected):
        clusters[pair[0]].append(pair)
    targets = sorted(clusters)
    assert len(targets) == 4
    rng = np.random.default_rng(20261107)
    draws = rng.integers(0, len(targets), size=(20000, len(targets)))
    per_seed = []
    for seed in SEEDS:
        mse = {}
        for kind in KINDS:
            mse[kind] = np.array([np.mean([(float(predictions[(seed, kind)][pair]['pred']) - float(expected[pair]['y']))**2 for pair in clusters[target]]) for target in targets])
        rmse = {kind: float(np.sqrt(mse[kind]).mean()) for kind in KINDS}
        # Equal-target-weighted macro RMSE; bootstrap targets as sampling units.
        macro_diff = np.sqrt(mse['numeric'][draws]).mean(axis=1) - np.sqrt(mse['mixed'][draws]).mean(axis=1)
        per_seed.append({'seed': seed, 'mixed_target_macro_rmse': rmse['mixed'], 'numeric_target_macro_rmse': rmse['numeric'], 'numeric_minus_mixed_macro_rmse': float(rmse['numeric'] - rmse['mixed']), 'bootstrap_95pct_ci_numeric_minus_mixed': np.quantile(macro_diff, [0.025, 0.975]).tolist(), 'target_count': len(targets)})
    report = {'status': 'EXPLORATORY_REPEATEDLY_INSPECTED_DEV_NOT_INDEPENDENT_TEST', 'source_sha256': hashlib.sha256(DATA.read_bytes()).hexdigest(), 'validation_numeric_pairs': len(expected), 'validation_targets': len(targets), 'seeds': per_seed, 'bootstrap_draws': len(draws), 'bootstrap_unit': 'protein target', 'notes': ['Paired predictions on identical numeric-Kd validation pairs; no screening blanks treated as quantitative Kd.', 'Four targets only; intervals are unstable and do not capture training-set or assay-construct uncertainty.', 'This is a post-hoc development comparison, not a statistical confirmation or independent holdout.', 'The mixed-label and numeric-only models are compared at fixed epoch 4.']}
    output = ROOT / 'research/reconciled_numeric_label_objective_uncertainty_dev.json'
    output.write_text(json.dumps(report, indent=2), encoding='utf-8')
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
