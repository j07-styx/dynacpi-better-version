from pathlib import Path
import hashlib
root=Path(__file__).resolve().parents[1];p=root/"chemprop"/"models"/"mpn.py"
s=p.read_text();key="a_message = nei_a_message.mean(dim=1)"
assert s.count(key)==1
new="valid = (a2x != 0).unsqueeze(-1)\n        a_message = (nei_a_message * valid).sum(dim=1) / valid.sum(dim=1).clamp_min(1)"
out=root/"dynacpi"/"mpn.py";out.write_text("# Derived from dmis-lab/PerceiverCPI commit 755f076; see LICENSE.txt.\n"+s.replace(key,new))
print("Created valid-degree MPN port",hashlib.sha256(out.read_bytes()).hexdigest())
