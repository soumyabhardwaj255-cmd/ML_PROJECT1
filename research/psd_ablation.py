"""Isolated PSD representation ablation, v1. No canonical writes or KD training."""
import argparse
import json
import sys
from pathlib import Path

import joblib
import mne
import numpy as np
import pandas as pd
from scipy.signal import welch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from eeg_seizure import config as C
from eeg_seizure.artifacts import atomic_csv, atomic_json, digest, environment
from eeg_seizure.dataset import load_dataset, validated_annotations, window_rows
from eeg_seizure.channels import dedupe_channel_names
from eeg_seizure.preprocessing import preprocess_raw
from eeg_seizure.features.frequency_domain import BANDS
from eeg_seizure.evaluation import patient_split, tune_threshold, metrics, summarize
from eeg_seizure.modeling import make_model

ARMS = ("canonical_absolute", "matched_relative", "notebook_relative")
NOTEBOOK_BANDS = {**BANDS, "gamma": (30, 70)}
MEASURES = ["accuracy", "precision", "recall", "specificity", "balanced_accuracy", "f1", "roc_auc", "average_precision"]


def powers(matrix, sfreq=256):
    """Return channel-major/band-minor absolute, matched relative, notebook relative."""
    matrix = np.asarray(matrix, dtype=np.float64)
    if matrix.ndim != 2 or matrix.shape[1] != 1024 or sfreq != 256 or not np.isfinite(matrix).all():
        raise ValueError("Expected finite channels x 1024 canonical 4-second samples")
    outputs = []
    for segment, bands, inclusive in ((256, BANDS, True), (512, NOTEBOOK_BANDS, False)):
        f, p = welch(matrix, fs=sfreq, nperseg=segment, axis=-1)
        total = np.trapezoid(p, f, axis=-1)
        absolute = np.stack([np.trapezoid(p[:, (f>=a)&((f<=b) if inclusive else (f<b))],
                                            f[(f>=a)&((f<=b) if inclusive else (f<b))], axis=-1)
                             for a,b in bands.values()], axis=1)
        relative = np.divide(absolute, total[:,None], out=np.zeros_like(absolute), where=total[:,None]>0)
        if segment == 256:
            outputs.extend([absolute.ravel(), relative.ravel()])
        else:
            outputs.append(relative.ravel())
    return outputs


def arm_frame(canonical, values, arm, columns):
    if arm not in ARMS:
        raise ValueError("Unknown representation")
    freq = [c for c in columns if c.startswith("freq__")]
    frame = canonical[C.META + columns].copy()
    if arm != ARMS[0]:
        values = np.asarray(values)
        if values.shape != (len(frame),len(freq)) or not np.isfinite(values).all():
            raise ValueError("Bad relative feature shape/values")
        frame[freq] = values
        prefix = "freq_relative" if arm == ARMS[1] else "freq02_relative"
        rename = {c:c.replace("freq__",prefix+"__",1) for c in freq}
        frame = frame.rename(columns=rename)
        columns = [rename.get(c,c) for c in columns]
    return frame, columns


def protect():
    roots = [C.ROOT / x for x in ("src","notebooks","data/processed/corrected_v2","results/corrected","results/analysis","results/feasibility")]
    return {str(p.relative_to(C.ROOT)):digest(p) for root in roots for p in root.rglob("*")
            if p.is_file() and "__pycache__" not in p.parts}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-id", default="psd-representation-v1")
    args = parser.parse_args()
    if Path(args.run_id).name != args.run_id or args.run_id in (".",".."):
        raise ValueError("One run directory name required")
    run = C.ROOT / "results/ablations" / args.run_id
    run.mkdir(parents=True,exist_ok=False)
    protected = protect()
    canonical, dm = load_dataset()
    columns, channels = dm["spec"]["feature_columns"], dm["spec"]["channels"]
    frequency = [c for c in columns if c.startswith("freq__")]
    assert frequency == [f"freq__{band}__{ch}" for ch in channels for band in BANDS]
    manifest = dict(status="running", version="psd-representation-v1", environment=environment(),
        arms=ARMS, primary_comparison="matched_relative minus canonical_absolute",
        secondary_comparison="notebook_relative minus canonical_absolute; spectral recipe is confounded with normalization",
        model="logistic_regression", model_parameters=C.LR_PARAMS, seed=C.SEED,
        protocol="outer five-patient LOPO; four inner patient folds choose F1 threshold on 0.05..0.95; refit training-only imputation/scaling every fold",
        threshold_grid=C.THRESHOLDS, feature_selection="none", kd="none", dimensionality=330,
        changed_features=110, unchanged_features=220, dataset_sha256=dm["feature_sha256"], dataset_spec=dm["spec"],
        extraction=dict(sample_rate=256,window_sec=4,channels=channels,
            canonical=dict(nperseg=256,bands=BANDS,upper_edge="inclusive",units="V^2"),
            matched_relative=dict(nperseg=256,bands=BANDS,upper_edge="inclusive",denominator="per-window per-channel integral over all Welch bins 0..128 Hz"),
            notebook_relative=dict(nperseg=512,bands=NOTEBOOK_BANDS,upper_edge="exclusive",denominator="per-window per-channel integral over all Welch bins 0..128 Hz"),
            zero_power="zero relative powers, matching notebook; counted",welch="scipy defaults, Hann, 50% overlap, constant detrend, density, mean"),
        provenance=dict(canonical="Inherited frequency extractor, unchanged", relative="02 notebook per-channel band integral / total PSD integral; matched arm adapts PSD settings to isolate representation",
                        notebook_sha256=digest(C.ROOT/"notebooks/02_preprocess_chb01.ipynb")),
        source_sha256=digest(__file__), protected_before=protected)
    atomic_json(run/"manifest.json",manifest)
    try:
        extracted = {arm:np.empty((len(canonical),110),dtype=np.float64) for arm in ARMS[1:]}
        zero_rows = 0
        for i,record in enumerate(dm["records"]):
            patient,filename = record["patient"],record["file"]
            path=C.RAW_DIR/patient/filename
            summary=path.parent/f"{patient}-summary.txt"
            if digest(path)!=record["raw_sha256"] or digest(summary)!=record["summary_sha256"]:
                raise ValueError("Input hash mismatch")
            subset=canonical[(canonical.patient==patient)&(canonical.file==filename)]
            with mne.io.read_raw_edf(path,preload=True,verbose="ERROR") as raw:
                fs=raw.info["sfreq"]
                if fs!=256:
                    raise ValueError("Unexpected sample rate")
                intervals=validated_annotations(summary,filename,raw.n_times/fs)
                expected=window_rows(raw.n_times,fs,intervals,patient,filename)
                pd.testing.assert_frame_equal(subset[C.META].reset_index(drop=True),expected,check_dtype=False)
                raw=preprocess_raw(dedupe_channel_names(raw),channels,**dm["spec"]["preprocessing"])
                for row in subset.itertuples():
                    matrix=raw.get_data(start=round(row.start_sec*fs),stop=round(row.end_sec*fs))
                    absolute,relative,notebook=powers(matrix,fs)
                    np.testing.assert_allclose(absolute,canonical.loc[row.Index,frequency].to_numpy(dtype=float),rtol=1e-10,atol=1e-25)
                    zero_rows += int(np.any(relative.reshape(22,5).sum(axis=1)==0))
                    extracted[ARMS[1]][row.Index]=relative
                    extracted[ARMS[2]][row.Index]=notebook
            print(f"Spectral extraction {i+1}/{len(dm['records'])}: {patient}/{filename}",flush=True)
        predictions, rows, inner_audits, schemas = [], [], [], {}
        for arm in ARMS:
            frame,arm_columns=arm_frame(canonical,extracted.get(arm),arm,columns)
            schemas[arm]=arm_columns
            if arm!=ARMS[0]:
                atomic_csv(run/f"{arm}_features.csv",frame[C.META+[c for c in arm_columns if c.startswith("freq")]])
            for patient in C.PATIENTS:
                train,test=patient_split(frame,patient)
                threshold,audit,oof=tune_threshold(train,arm_columns,"logistic_regression",C.SEED)
                for a in audit:
                    assert patient not in a["train_patients"] and patient!=a["validation_patient"]
                    inner_audits.append(dict(arm=arm,outer_patient=patient,**a))
                atomic_csv(run/f"inner_{arm}_{patient}.csv",oof)
                model=make_model("logistic_regression",train.label,C.SEED)
                model.fit(train[arm_columns],train.label)
                scores=model.predict_proba(test[arm_columns])[:,1]
                bundle=dict(format_version="psd-ablation-v1",arm=arm,estimator=model,feature_columns=arm_columns,
                    train_patients=sorted(train.patient.unique()),test_patients=[patient],threshold=threshold,
                    canonical_dataset_sha256=dm["feature_sha256"],extraction=manifest["extraction"],seed=C.SEED)
                bundle_path=run/f"bundle_{arm}_{patient}.joblib"
                joblib.dump(bundle,bundle_path)
                np.testing.assert_array_equal(scores,joblib.load(bundle_path)["estimator"].predict_proba(test[arm_columns])[:,1])
                row=metrics(test.label,scores,threshold)
                row.update(specificity=row["tn"]/(row["tn"]+row["fp"]))
                row.update(balanced_accuracy=(row["specificity"]+row["recall"])/2,model=arm,seed=C.SEED,test_patient=patient,threshold=threshold)
                rows.append(row)
                predictions.append(test[C.META].assign(model=arm,seed=C.SEED,threshold=threshold,probability_seizure=scores,prediction=(scores>=threshold).astype(int)))
                print(f"{arm}/{patient}: F1={row['f1']:.4f}, AP={row['average_precision']:.4f}, threshold={threshold}",flush=True)
        predictions=pd.concat(predictions,ignore_index=True)
        per_patient=pd.DataFrame(rows)
        _,pooled=summarize(predictions,per_patient)
        pooled["specificity"]=pooled.tn/(pooled.tn+pooled.fp)
        pooled["balanced_accuracy"]=(pooled.specificity+pooled.recall)/2
        macro=per_patient.groupby("model")[MEASURES].agg(["mean","std","min","max"])
        macro.columns=[f"{m}_{s}" for m,s in macro.columns]
        paired=[]
        baseline=per_patient[per_patient.model==ARMS[0]].set_index("test_patient")
        for arm in ARMS[1:]:
            other=per_patient[per_patient.model==arm].set_index("test_patient")
            delta=other[MEASURES]-baseline[MEASURES]
            paired.append(delta.reset_index().assign(comparison=f"{arm}-canonical_absolute"))
        paired=pd.concat(paired,ignore_index=True)
        atomic_csv(run/"predictions.csv",predictions)
        atomic_csv(run/"per_patient_metrics.csv",per_patient)
        atomic_csv(run/"pooled_window_metrics.csv",pooled)
        atomic_csv(run/"macro_patient_metrics.csv",macro.reset_index())
        atomic_csv(run/"paired_patient_deltas.csv",paired)
        atomic_json(run/"inner_fold_audit.json",inner_audits)
        atomic_json(run/"feature_schemas.json",schemas)
        if protected!=protect():
            raise AssertionError("Protected canonical/analysis artifact changed")
        manifest.update(status="complete",windows=len(canonical),positive_windows=int(canonical.label.sum()),
            absolute_extraction_parity="all 39522 windows / 110 frequency values",zero_band_power_windows=zero_rows,
            protected_files_unchanged=len(protected),fits=75,bundle_reload_parity=True,
            artifacts={p.name:digest(p) for p in run.iterdir() if p.is_file() and p.name!="manifest.json"})
    except Exception as exc:
        manifest.update(status="failed",error=str(exc)); raise
    finally:
        atomic_json(run/"manifest.json",manifest)
    print(run)


if __name__=="__main__":
    main()
