import importlib.util
import json
from pathlib import Path
import numpy as np
import pandas as pd
import pytest
import torch
from eeg_seizure import config as C
from eeg_seizure import modeling
from eeg_seizure.artifacts import digest
from eeg_seizure.evaluation import metrics,summarize

spec=importlib.util.spec_from_file_location("contribution",Path(__file__).parents[1]/"research/kd_contribution.py")
experiment=importlib.util.module_from_spec(spec)
spec.loader.exec_module(experiment)


def test_random_selection_training_only_seed_and_schema():
    columns=C.feature_names([f"c{i}" for i in range(22)])
    data=pd.DataFrame(np.zeros((5,330)),columns=columns).assign(patient=C.PATIENTS,label=[0,1,0,1,0])
    train=data[data.patient!="chb08"]
    names,seed=experiment.random_features(train,columns,"chb08")
    assert len(names)==len(set(names))==50
    assert names==[c for c in columns if c in names]
    mutated=train.copy(); mutated[columns]=99; mutated.label=1
    assert experiment.random_features(mutated,columns,"chb08")== (names,seed)
    assert experiment.random_features(train,columns,"chb08",43)!=(names,seed)
    with pytest.raises(ValueError):
        experiment.random_features(data,columns,"chb08")
    with pytest.raises(ValueError):
        experiment.random_features(train,columns[:-1],"chb08")


def test_paired_initialization_loader_and_determinism(monkeypatch):
    starts=[]; seeds=[]
    original=modeling.student_network
    def capture(n,hidden=None):
        model=original(n,hidden)
        starts.append(torch.cat([p.detach().flatten() for p in model.parameters()]).clone())
        return model
    monkeypatch.setattr(modeling,"student_network",capture)
    loader=torch.utils.data.DataLoader
    def capture_loader(*args,**kwargs):
        seeds.append(kwargs["generator"].initial_seed())
        return loader(*args,**kwargs)
    monkeypatch.setattr(torch.utils.data,"DataLoader",capture_loader)
    x=np.random.default_rng(4).normal(size=(32,50)); y=np.tile([0,1],16)
    a,_=modeling.train_student(x,y,42,epochs=1)
    b,_=modeling.train_student(x,y,42,np.full(32,.4),epochs=1)
    c,_=modeling.train_student(x,y,42,epochs=1)
    assert all(torch.equal(starts[0],s) for s in starts)
    assert seeds==[42,42,42]
    for k,v in a.state_dict().items():
        assert torch.equal(v,c.state_dict()[k])
    assert experiment.initial_hash(42)==experiment.initial_hash(42)
    assert experiment.initial_hash(42)!=experiment.initial_hash(43)


def test_identical_selected_schema_and_student_capacity():
    selected=[f"s{i}" for i in range(50)]; random_names=[f"r{i}" for i in range(50)]
    arms=experiment.student_features(selected,random_names)
    assert arms["selected_supervised"]==arms["selected_kd"]==selected
    assert arms["random_supervised"]==random_names
    for names in arms.values():
        model=modeling.student_network(len(names))
        assert sum(p.numel() for p in model.parameters())==2177
    with pytest.raises(ValueError):
        experiment.student_features(selected[:-1],random_names)


def test_metrics_paired_effects_and_pooling_are_distinct():
    rows=[]; predictions=[]
    for patient,y,p in (("p1",[1,0],[.9,.1]),("p2",[1,1,0,0],[.2,.2,.9,.9])):
        for model in experiment.ARMS:
            rows.append(dict(metrics(y,p),model=model,seed=42,test_patient=patient))
            predictions.append(pd.DataFrame(dict(label=y,probability_seizure=p,prediction=np.array(p)>=.5,model=model,seed=42)))
    rows=pd.DataFrame(rows)
    macro,pooled=summarize(pd.concat(predictions),rows)
    assert np.allclose(macro.f1_mean,.5)
    assert np.allclose(pooled.f1,1/3)
    assert (experiment.effects(rows)[experiment.METRICS]==0).all().all()
    bad=rows.drop(rows[(rows.model=="selected_kd")&(rows.test_patient=="p2")].index)
    with pytest.raises(ValueError):
        experiment.effects(bad)


def test_manifest_rejects_incomplete_or_changed_artifacts(tmp_path):
    artifact=tmp_path/"features.json"; artifact.write_text('["a","b"]')
    manifest=tmp_path/"manifest.json"
    value=dict(status="complete",artifacts={artifact.name:digest(artifact)})
    manifest.write_text(json.dumps(value))
    assert experiment.verify_complete(tmp_path)==value
    artifact.write_text('["a","changed"]')
    with pytest.raises(ValueError):
        experiment.verify_complete(tmp_path)
    value["status"]="running"; manifest.write_text(json.dumps(value))
    with pytest.raises(ValueError):
        experiment.verify_complete(tmp_path)


def test_seed_repetitions_do_not_inflate_patient_count(monkeypatch):
    monkeypatch.syspath_prepend(str(Path(__file__).parents[1]/"research"))
    import summarize_kd_contribution as summary
    frame=pd.DataFrame([dict(comparison="kd_effect",seed=s,test_patient=p,f1=v,average_precision=v,recall=v)
        for s in (42,43,44) for p,v in (("p1",.1),("p2",-.2))])
    averaged=summary.average_effects_per_patient(frame)
    assert len(averaged)==2
    np.testing.assert_allclose(averaged.f1,[.1,-.2])
    with pytest.raises(ValueError):
        summary.average_effects_per_patient(pd.concat([frame,frame.iloc[:1]]))
