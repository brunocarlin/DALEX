"""Compare numeric pandas outputs with an unmodified upstream source checkout.

Usage: python ci/compare_legacy.py --reference-root PATH_TO_ORIGINAL_PYTHON_PACKAGE
Runs baseline/current packages in separate processes using the same interpreter.
"""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys

import numpy as np


def capture(root):
    sys.path.insert(0, str(Path(root).resolve()))
    import pandas as pd
    from sklearn.linear_model import LinearRegression
    import dalex as dx
    rng = np.random.default_rng(123)
    frame = pd.DataFrame(rng.normal(size=(40,3)), columns=['a','b','c'], index=np.arange(40)*3+7)
    y = frame.a.to_numpy()*2 + frame.b.to_numpy()/2 + np.sin(frame.c.to_numpy())
    model = LinearRegression().fit(frame,y)
    explainer = dx.Explainer(model,frame,y,verbose=False,label='model')
    result = {'prediction':explainer.y_hat.tolist(),'residual':explainer.residuals.tolist()}
    def table(name, value):
        numeric=value.select_dtypes(include=np.number)
        result[name]={'columns':list(numeric.columns),'values':numeric.to_numpy(dtype=float).tolist()}
    table('performance',explainer.model_performance().result)
    table('diagnostics',explainer.model_diagnostics().result)
    for kind in ('variable_importance','ratio','difference'):
        vi=explainer.model_parts(type=kind,N=25,B=4,random_state=42,keep_raw_permutations=True,
                                 variable_groups={'ab':['a','b'],'c':['c']})
        table('importance_'+kind,vi.result)
        table('permutations_'+kind,vi.permutation)
    for kind in ('break_down','break_down_interactions','shap'):
        explanation=explainer.predict_parts(frame.iloc[[2]],type=kind,N=25,B=4,random_state=42,keep_distributions=True)
        table(kind,explanation.result)
    table('cp',explainer.predict_profile(frame.iloc[[2,4]],grid_points=5,verbose=False).result)
    for kind in ('partial','accumulated','conditional'):
        table(kind,explainer.model_profile(type=kind,N=20,grid_points=5,random_state=42,verbose=False).result)
    return result


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--reference-root')
    parser.add_argument('--capture-root')
    parser.add_argument('--output',default='legacy-parity.json')
    args=parser.parse_args()
    if args.capture_root:
        Path(args.output).write_text(json.dumps(capture(args.capture_root)))
        return
    reference=Path(args.reference_root).resolve()
    current=Path(__file__).resolve().parents[1]
    outputs=[]
    env=dict(os.environ,OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1')
    for name,root in [('reference',reference),('current',current)]:
        output=Path(args.output).resolve().with_suffix('.'+name+'.json')
        subprocess.run([sys.executable,__file__,'--capture-root',str(root),'--output',str(output)],env=env,check=True)
        outputs.append(json.loads(output.read_text()))
    a,b=outputs
    errors={}
    for key in a:
        if isinstance(a[key],dict):
            assert a[key]['columns']==b[key]['columns'],key
            first,second=np.asarray(a[key]['values']),np.asarray(b[key]['values'])
        else:
            first,second=np.asarray(a[key]),np.asarray(b[key])
        np.testing.assert_allclose(first,second,rtol=1e-12,atol=1e-12,equal_nan=True,err_msg=key)
        errors[key]=float(np.nanmax(np.abs(first-second)))
    report={'python':sys.version,'reference_root':str(reference),'numeric_outputs_compared':len(errors),
            'tolerance':{'rtol':1e-12,'atol':1e-12},'max_absolute_error_by_output':errors,'passed':True}
    Path(args.output).write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report,indent=2))


if __name__=='__main__':
    main()
