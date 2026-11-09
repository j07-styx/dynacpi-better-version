import ast,json,copy
from pathlib import Path
from types import SimpleNamespace
import torch
import torch.nn as nn
import torch.nn.functional as F
root=Path(r"C:\Users\J0124\dynacpi_v4")
src=(root/"chemprop/models/CAB.py").read_text()
tree=ast.parse(src)
classes=ast.Module(body=[x for x in tree.body if isinstance(x,ast.ClassDef)],type_ignores=[])
ns={"torch":torch,"nn":nn,"F":F,"TrainArgs":SimpleNamespace}
exec(compile(classes,"pinned_CAB.py","exec"),ns)
torch.manual_seed(42)
m=ns["CrossAttentionBlock"](SimpleNamespace(hidden_size=256,dropout=0.)).cuda().eval()
xs=[torch.randn(4,256,device="cuda") for _ in range(3)]
with torch.no_grad():
 allout=m(*xs)
 one=torch.cat([m(*(x[i:i+1] for x in xs)) for i in range(4)])
err=float((allout-one).abs().max())
torch.testing.assert_close(allout,one,rtol=1e-4,atol=1e-5)
result={"scope":"Actual pinned CAB classes on synthetic pooled features; no trained checkpoint or molecular benchmark","device":torch.cuda.get_device_name(),"torch":torch.__version__,"output_shape":list(allout.shape),"batch_single_max_abs_difference":err,"finite":bool(torch.isfinite(allout).all()),"passed":True}
(root/"research"/"gpu_cab_check.json").write_text(json.dumps(result,indent=2))
print(json.dumps(result,indent=2))

