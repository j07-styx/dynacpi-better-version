"""Build a read-only metadata registry of completed engineering pilot experiments.
No checkpoint loading, training, dataset modification or private records exported.
"""
import argparse
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RESEARCH = ROOT / 'research'


def build():
    records = []
    for folder in sorted(RESEARCH.glob('pilot_*_dev')):
        for run in sorted(folder.glob('seed_*')):
            manifest_file = run / 'manifest.json'
            history_file = run / 'history.json'
            if not history_file.is_file():
                continue
            history = json.loads(history_file.read_text(encoding='utf-8'))
            if not isinstance(history, list) or not history:
                continue
            if any(not isinstance(epoch, dict) for epoch in history):
                raise ValueError(f'Invalid history in {run.relative_to(ROOT)}')
            manifest = json.loads(manifest_file.read_text(encoding='utf-8')) if manifest_file.exists() else None
            candidates = [e for e in history if isinstance(e.get('validation_geometry_target_macro_rmse'), (int, float))]
            if not candidates:
                continue
            best = min(candidates, key=lambda e:e['validation_geometry_target_macro_rmse'])
            record = {
                'experiment_id': f'{folder.name}/{run.name}',
                'history_path': history_file.relative_to(ROOT).as_posix(),
                'manifest_present': manifest is not None,
                'status': manifest.get('status') if manifest else 'INCOMPLETE_NO_MANIFEST',
                'seed': manifest.get('engineering_seed') if manifest else int(run.name.removeprefix('seed_')),
                'epochs_logged': len(history),
                'best_epoch': best.get('epoch'),
                'best_geometry_macro_rmse': best['validation_geometry_target_macro_rmse'],
                'matched_sequence_macro_rmse': best.get('validation_sequence_target_macro_rmse'),
                'train_pairs': manifest.get('train_pair_count') if manifest else None,
                'validation_pairs': manifest.get('val_pair_count') if manifest else None,
                'independent_holdout': False,
                'development_only': True,
            }
            records.append(record)
    ids = [r['experiment_id'] for r in records]
    if len(ids) != len(set(ids)):
        raise ValueError('Duplicate experiment identifiers')
    return {'schema_version':1, 'status':'ENGINEERING_REGISTRY_NOT_INDEPENDENT_EVALUATION',
            'count':len(records), 'experiments':records}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, help='Optional local JSON output path; refuses overwrite')
    args = parser.parse_args()
    registry = build()
    if args.output:
        target = args.output.resolve()
        if target.exists():
            raise FileExistsError(target)
        target.write_text(json.dumps(registry,indent=2)+'\n',encoding='utf-8')
    print(json.dumps({'status':registry['status'],'experiments':registry['count'],
                      'incomplete':sum(not r['manifest_present'] for r in registry['experiments'])}))


if __name__ == '__main__':
    main()
