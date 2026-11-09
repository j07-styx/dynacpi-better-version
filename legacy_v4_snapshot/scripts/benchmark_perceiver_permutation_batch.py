"""Synthetic computational benchmark only; no affinity accuracy measurement."""
import sys,json,time,platform
from pathlib import Path
import numpy as np
from threadpoolctl import threadpool_limits
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from dynacpi.perceiver_krr_support_permutation_guard import _permuted_support_mse,_permuted_support_mses
from dynacpi.perceiver_krr_geometry_guard import KRRGeometrySettings
def main():
    rng=np.random.default_rng(2612);results=[]
    with threadpool_limits(limits=1):
        for n in (5,10):
            x=rng.normal(size=(n,8));k=x@x.T;r=rng.normal(size=n)
            d=rng.normal(size=(n,49));kw=dict(chemical_ridge=.5,cap=.5,geometry_settings=KRRGeometrySettings())
            def old():return np.array([_permuted_support_mse(k,r,d[:,i],**kw) for i in range(49)])
            def new():return _permuted_support_mses(k,r,d,**kw)
            reference=old();got=new();error=float(np.max(abs(reference-got)));assert error<1e-12
            times={}
            for name,fn in (("scalar",old),("batched",new)):
                measurements=[]
                for _ in range(20):
                    start=time.perf_counter();fn();measurements.append(time.perf_counter()-start)
                times[name]=float(np.median(measurements))
            results.append({"support":n,"nulls":49,"max_abs_mse_error":error,
                "median_seconds":times,"speedup":times["scalar"]/times["batched"],
                "solve_calls_scalar":3*n*49,"solve_calls_batched":n})
    report={"scope":"Synthetic null-kernel runtime only; not CPI accuracy or end-to-end speedup",
        "seed":2612,"repetitions":20,"blas_threads":1,"python":platform.python_version(),
        "numpy":np.__version__,"results":results}
    path=ROOT/"research/perceiver_permutation_batch_benchmark_20261109.json"
    with path.open("x") as f:json.dump(report,f,indent=2)
    print(json.dumps(report,indent=2))
if __name__=="__main__":main()
