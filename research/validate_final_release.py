"""Check final consolidation identities, costs, selection statistics and tests."""
import json,sys,xml.etree.ElementTree as ET
from pathlib import Path
import numpy as np
import pandas as pd
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from eeg_seizure import config as C
from eeg_seizure.artifacts import digest,atomic_json
from eeg_seizure.inference import load_bundle
from eeg_seizure.modeling import student_network

def main():
    root=C.ROOT;directory=root/'results/final';out=directory/'final-evidence-v1'
    release=json.loads((directory/'final-release-v1.json').read_text())
    assert release['canonical_features']==330
    assert digest(root/'FINAL_PROJECT_REPORT.md')==release['report_sha256']
    for p,h in release['sources'].items():assert digest(root/p)==h,p
    manifest=json.loads((out/'manifest.json').read_text())
    for p,h in manifest['protected_before'].items():assert digest(root/p)==h,p
    for p,h in manifest['artifacts'].items():assert digest(out/p)==h,p
    selections=pd.read_csv(out/'selected_features.csv')
    assert len(selections)==250 and (selections.groupby('patient').size()==50).all()
    for row in pd.read_csv(out/'selection_pairs.csv').itertuples():
        a=set(selections[selections.patient==row.patient_a].feature);b=set(selections[selections.patient==row.patient_b].feature)
        assert row.intersection==len(a&b);np.testing.assert_allclose(row.jaccard,len(a&b)/len(a|b))
    for row in pd.read_csv(out/'sizes.csv').itertuples():
        p=root/f'results/contribution/kd-contribution-v1-seed42/seed42/{row.patient}/{row.arm}.joblib'
        assert p.stat().st_size==row.bundle_bytes
        bundle=load_bundle(p);assert len(bundle['feature_columns'])==330
        if row.arm!='teacher':
            model=student_network(50);model.load_state_dict(bundle['state_dict'])
            assert sum(p.numel() for p in model.parameters())==2177==row.parameters
    timings=pd.read_csv(out/'edf_timing.csv');assert len(timings)==15
    np.testing.assert_allclose(timings.end_to_end_seconds,timings.read_preprocess_seconds+timings.feature_extraction_seconds+timings.inference_seconds)
    tests=ET.parse(directory/'final-tests-v1.xml').getroot()
    suites=list(tests.iter('testsuite'))
    assert sum(int(t.attrib['tests']) for t in suites)==69
    assert all(int(t.attrib.get('failures',0))==0 and int(t.attrib.get('errors',0))==0 for t in suites)
    report=(root/'FINAL_PROJECT_REPORT.md').read_text(encoding='utf-8')
    for i in range(1,25):assert f'## {i}.' in report
    release.update(final_validation=True,validation_source_sha256=digest(__file__),
        documentation={str(p.relative_to(root)):digest(p) for p in [root/'README.md',directory/'README-before-final-release-v1.md',root/'research/consolidate_final_report.py']},
        test_receipt_sha256=digest(directory/'final-tests-v1.xml'),protected_files_verified=len(manifest['protected_before']))
    atomic_json(directory/'final-release-v1.json',release)
    print(json.dumps(dict(validated=True,tests=69,protected_files=len(manifest['protected_before']),selected_features=250,bundles_measured=15,report_sections=24)))

if __name__=='__main__':main()
