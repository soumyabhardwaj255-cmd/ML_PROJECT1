"""Matched-representation, notebook parity, schema and leakage diagnostics."""
import ast
import importlib.util
import json
from pathlib import Path
import numpy as np
import pandas as pd
import pytest
from scipy.signal import welch
from eeg_seizure import config as C
from eeg_seizure.features.frequency_domain import extract_frequency_domain_features
from eeg_seizure.evaluation import tune_threshold
from eeg_seizure.modeling import FeatureTransform

ROOT=Path(__file__).parents[1]
spec=importlib.util.spec_from_file_location("psd_ablation",ROOT/"research/psd_ablation.py")
ablation=importlib.util.module_from_spec(spec)
spec.loader.exec_module(ablation)


def test_canonical_parity_and_gain_invariance():
    x=np.random.default_rng(71).normal(size=(2,1024))*1e-5
    absolute,relative,notebook=ablation.powers(x)
    reference=extract_frequency_domain_features(x,256,["a","b"])
    np.testing.assert_allclose(absolute,list(reference.values()),rtol=1e-13,atol=0)
    scaled=ablation.powers(3*x)
    np.testing.assert_allclose(scaled[0],9*absolute,rtol=1e-13,atol=0)
    np.testing.assert_allclose(scaled[1],relative,rtol=1e-13,atol=0)
    np.testing.assert_allclose(scaled[2],notebook,rtol=1e-13,atol=0)
    f,p=welch(x,fs=256,nperseg=256,axis=-1)
    np.testing.assert_allclose(relative.reshape(2,5),absolute.reshape(2,5)/np.trapezoid(p,f,axis=-1)[:,None])
    assert not np.allclose(relative,notebook)


def test_actual_notebook_equation_parity_and_zero_power():
    book=json.loads((ROOT/"notebooks/02_preprocess_chb01.ipynb").read_text(encoding="utf-8"))
    cell=next("".join(c["source"]) for c in book["cells"] if "def extract_channel_features" in "".join(c["source"]))
    tree=ast.parse(cell)
    nodes=[n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=="extract_channel_features"]
    bands_node=next(n for n in tree.body if isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id=="FREQUENCY_BANDS" for t in n.targets))
    bands=ast.literal_eval(bands_node.value)
    assert bands==ablation.NOTEBOOK_BANDS
    env=dict(np=np,welch=welch,FREQUENCY_BANDS=bands)
    exec(compile(ast.Module(body=nodes,type_ignores=[]),"<notebook-extraction-parity>","exec"),env)
    for x in (np.zeros(1024),np.random.default_rng(8).normal(size=1024)):
        expected=env["extract_channel_features"](x,256)
        actual=ablation.powers(x[None,:])[2]
        np.testing.assert_allclose(actual,[expected[f"{b}_relative_power"] for b in bands],rtol=1e-13,atol=0)
    with pytest.raises(ValueError):
        ablation.powers(np.zeros((22,2560)))


def test_schema_replaces_exactly_110_and_preserves_220():
    columns=C.feature_names([f"channel{i}" for i in range(22)])
    frame=pd.DataFrame(np.arange(660).reshape(2,330),columns=columns).assign(patient="p",file="f",start_sec=[0.,4.],end_sec=[4.,8.],label=[0,1])
    original=frame.copy(deep=True)
    for arm in ablation.ARMS[1:]:
        changed,schema=ablation.arm_frame(frame,np.ones((2,110))*.2,arm,columns)
        assert len(schema)==330 and len(set(schema))==330
        unchanged=[c for c in columns if not c.startswith("freq__")]
        pd.testing.assert_frame_equal(changed[unchanged+C.META],frame[unchanged+C.META])
        assert sum(a!=b for a,b in zip(columns,schema))==110
    pd.testing.assert_frame_equal(frame,original)


def test_inner_patient_exclusion_and_training_only_scaling():
    rows=[]
    for i,patient in enumerate(C.PATIENTS):
        for j in range(12):
            rows.append(dict(patient=patient,file="f",start_sec=float(j*4),end_sec=float((j+1)*4),label=int(j<2+i),x=float(j+i)))
    frame=pd.DataFrame(rows)
    train,test=ablation.patient_split(frame,"chb08")
    threshold,audit,oof=tune_threshold(train,["x"],"logistic_regression",42)
    assert threshold in C.THRESHOLDS and len(oof)==len(train)
    for fold in audit:
        assert "chb08" not in fold["train_patients"]
        assert fold["validation_patient"] not in fold["train_patients"]
        y=train[train.patient.isin(fold["train_patients"])].label
        assert fold["class_ratio"]==(y==0).sum()/(y==1).sum()
    transform=FeatureTransform().fit(train[["x"]])
    saved=transform.scaler_.mean_.copy()
    transform.transform(test[["x"]]*1e6)
    np.testing.assert_array_equal(saved,transform.scaler_.mean_)
