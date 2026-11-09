"""Create isolated DAVIS centered-geometry runs for prespecified extra seeds."""
from pathlib import Path
root=Path(__file__).resolve().parents[1]
src=root/'scripts/train_compound_conditioned_geometry_centered_dev.py'
s=src.read_text(encoding='utf-8')
old='if seed not in (1117,): raise ValueError("Only anchored compound-conditioned DEV seed 1114 allowed")'
assert s.count(old)==1
for seed in (1121,1122,1123):
    output=root/f'scripts/train_compound_conditioned_geometry_centered_seed_{seed}_dev.py'
    assert not output.exists(),output
    variant=s.replace(old,f'if seed not in ({seed},): raise ValueError("Only centered geometry DEV seed {seed} allowed")')
    variant=variant.replace('pilot_compound_conditioned_geometry_centered_dev','pilot_compound_conditioned_geometry_centered_multiseed_dev')
    assert variant!=s
    compile(variant,str(output),'exec')
    output.write_text(variant,encoding='utf-8')
    print('CREATED',output)
