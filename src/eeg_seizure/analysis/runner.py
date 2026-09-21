"""Bounded analysis and frozen-model robustness. No fitting or training imports."""
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
from .. import config as C
from ..artifacts import atomic_csv, atomic_json, digest
from ..inference import load_bundle, predict_features
from ..features.feature_extraction import extract_window_features
from .context import load_windows, new_run, seed_for
from .spectral import psd, add_noise, spectral_snr
from .sr import characteristic_component, bistable, voltage_referenced_transform
from .coherence import fhn, fhn_events, transitions, regularity
from .compression import reconstruct, preservation
from .reporting import summarize, raw_optima, dynamics_plot, psd_plot, robustness_psd_plot


def analyse(windows, signals, spec, args, output):
    ch = spec["channels"].index(args.channel)
    spectra, trials = [], []
    for row, matrix in zip(windows.to_dict("records"), signals):
        original = matrix[ch]
        f, p = psd(original, C.SAMPLE_RATE)
        spectra.extend([{**row, "channel": args.channel, "frequency_hz": a, "psd_v2_hz": b,
                         "nperseg": 256, "units": "V2/Hz"} for a,b in zip(f,p)])
        component, frequency = characteristic_component(original, C.SAMPLE_RATE)
        for model in ("bistable_driven_regularity", "fhn_sde"):
            for drive_name, drive in (("eeg_component",component), ("no_external_drive",np.zeros_like(component))):
                for steps in args.substeps:
                    for level in args.noise_levels:
                        for trial in range(args.trials):
                            # Same innovations across drive/control and noise levels at fixed dt.
                            seed = seed_for(args.seed, row["patient"],row["file"],row["start_sec"],args.channel,model,trial,steps)
                            rng = np.random.default_rng(seed)
                            response, diagnostic = (bistable if model.startswith("bistable") else fhn)(drive,C.SAMPLE_RATE,level,rng,substeps=steps)
                            if model.startswith("bistable"):
                                events=transitions(response,C.SAMPLE_RATE)
                                measure=regularity(events,C.SAMPLE_RATE)
                                detector="bistable hysteresis; CV sample SD; five intervals"
                            else:
                                events=fhn_events(response,C.SAMPLE_RATE)
                                measure=regularity(events,C.SAMPLE_RATE,minimum_intervals=3,ddof=0,minimum_interval_sec=.2)
                                detector="FHN mean+SD peaks; >=0.2s intervals; CV population SD; three intervals"
                            trials.append({**row,"channel":args.channel,"model":model,"drive":drive_name,
                                           "noise_intensity":level,"substeps":steps,"trial":trial,"seed":str(seed),"detector":detector,
                                           "target_frequency":frequency,"snr_db":spectral_snr(response,C.SAMPLE_RATE,frequency),
                                           "input_snr_db":spectral_snr(drive,C.SAMPLE_RATE,frequency),
                                           "snr_interpretation":"local target-band PSD ratio; control is noise-only",
                                           **diagnostic,**measure})
    table, trial_table = pd.DataFrame(spectra), pd.DataFrame(trials)
    baseline_keys=C.META+["channel","model","drive","substeps","trial"]
    baseline=trial_table[trial_table.noise_intensity==0][baseline_keys+["snr_db"]].rename(columns={"snr_db":"zero_noise_snr_db"})
    trial_table=trial_table.merge(baseline,on=baseline_keys,validate="many_to_one")
    trial_table["snr_change_vs_zero_noise_db"]=trial_table.snr_db-trial_table.zero_noise_snr_db
    trial_table["snr_change_vs_input_db"]=trial_table.snr_db-trial_table.input_snr_db
    summary = summarize(trial_table)
    atomic_csv(output / "psd.csv",table)
    atomic_csv(output / "dynamics_trials.csv",trial_table)
    atomic_csv(output / "dynamics_summary.csv",summary)
    atomic_csv(output / "raw_grid_optima.csv",raw_optima(summary))
    psd_plot(table,output); dynamics_plot(summary,output)
    return dict(windows=len(windows), trajectories=len(trials), valid_cr_trials=int(trial_table.valid.sum()),
                invalid_cr_trials=int((~trial_table.valid).sum()), result_role="separate exploratory analysis")


def condition_signals(matrix, condition, args, identity, trial):
    transformed, diagnostics = [], []
    for channel, signal in enumerate(matrix):
        rng = np.random.default_rng(seed_for(args.seed,*identity,channel,trial))
        if condition == "clean":
            result, info = signal.copy(), {}
        elif condition.startswith("noise_"):
            beta = {"noise_white":0,"noise_pink":1,"noise_black":3}[condition]
            result, info = add_noise(signal,C.SAMPLE_RATE,beta,args.noise_ratio,rng), {"noise_ratio":args.noise_ratio,"beta":beta}
        elif condition.startswith("compression_"):
            result, info = reconstruct(signal,args.factor,condition.removeprefix("compression_"))
            info.update(reduced_sample_rate=C.SAMPLE_RATE/args.factor, reduced_nyquist=C.SAMPLE_RATE/(2*args.factor))
        elif condition == "sr_voltage_referenced":
            result, info = voltage_referenced_transform(signal,C.SAMPLE_RATE,args.sr_noise,rng)
        else:
            raise ValueError("Unknown perturbation")
        transformed.append(result); diagnostics.append({"channel_index":channel,**info,**preservation(signal,result)})
    return np.stack(transformed),diagnostics


def robustness(windows, signals, reference, spec, dataset_manifest, args, output):
    bundle = load_bundle(args.bundle)
    if args.patient in bundle["train_patients"]:
        raise ValueError("Robustness patient must be held out from bundle training")
    if bundle["dataset_sha256"] != dataset_manifest["feature_sha256"] or bundle["dataset_spec"] != spec:
        raise ValueError("Bundle differs from canonical dataset/extraction contract")
    conditions = ["clean","noise_white","noise_pink","noise_black","compression_legacy_stride","compression_antialiased"]
    if args.include_sr:
        conditions.append("sr_voltage_referenced")
    features, detail, spectra = [], [], []
    for condition in conditions:
        stochastic = condition.startswith("noise_") or condition.startswith("sr_")
        for trial in range(args.trials if stochastic else 1):
            for row,matrix in zip(windows.to_dict("records"),signals):
                identity = [row[k] for k in C.KEYS]
                changed, diagnostic = condition_signals(matrix,condition,args,identity,trial)
                values = extract_window_features(changed,C.SAMPLE_RATE,spec["channels"])
                f,p=psd(changed[spec["channels"].index(args.channel)],C.SAMPLE_RATE)
                spectra.extend([{**row,"channel":args.channel,"condition":condition,"trial":trial,
                                 "frequency_hz":a,"psd_v2_hz":b} for a,b in zip(f,p)])
                features.append({**row,**values,"condition":condition,"trial":trial})
                detail.extend([{**row,"condition":condition,"trial":trial,**d} for d in diagnostic])
    features = pd.DataFrame(features)
    columns = spec["feature_columns"]
    if not np.isfinite(features[columns].to_numpy()).all():
        raise ValueError("Perturbation generated nonfinite features")
    clean = features[features.condition=="clean"].reset_index(drop=True)
    if not clean[C.META].equals(reference[C.META].reset_index(drop=True)):
        raise AssertionError("Clean window identities differ")
    np.testing.assert_allclose(clean[columns],reference[columns],rtol=1e-5,atol=1e-20)
    ref_scores, ref_decisions = predict_features(bundle,reference[C.META+columns])
    predictions, changes = [], []
    for (condition,trial), group in features.groupby(["condition","trial"],sort=False):
        scores, decisions = predict_features(bundle,group[C.META+columns])
        if condition=="clean":
            np.testing.assert_allclose(scores,ref_scores,rtol=1e-5,atol=1e-7)
            np.testing.assert_array_equal(decisions,ref_decisions)
        predictions.append(group[C.META].assign(condition=condition,trial=trial,probability_seizure=scores,
                           prediction=decisions,clean_probability=ref_scores,clean_prediction=ref_decisions,
                           decision_changed=decisions!=ref_decisions,threshold=bundle["threshold"]))
        delta=group[columns].to_numpy()-clean[columns].to_numpy()
        changes.extend([dict(condition=condition,trial=trial,feature=c,mean_absolute_change=float(np.abs(delta[:,j]).mean()),
                             mean_clean_absolute_value=float(np.abs(clean[c]).mean())) for j,c in enumerate(columns)])
    predictions=pd.concat(predictions,ignore_index=True)
    rows=[]
    for (condition,trial),g in predictions.groupby(["condition","trial"]):
        rows.append(dict(condition=condition,trial=trial,windows=len(g),decision_changes=int(g.decision_changed.sum()),
                         mean_absolute_probability_change=float(np.abs(g.probability_seizure-g.clean_probability).mean()),
                         errors=int((g.prediction!=g.label).sum()),seizure_windows=int(g.label.sum()),
                         interpretation="selected-window robustness; not population performance"))
    atomic_csv(output / "predictions.csv",predictions)
    atomic_csv(output / "robustness_summary.csv",pd.DataFrame(rows))
    atomic_csv(output / "feature_changes.csv",pd.DataFrame(changes))
    atomic_csv(output / "signal_diagnostics.csv",pd.DataFrame(detail))
    atomic_csv(output / "perturbed_psd.csv",pd.DataFrame(spectra))
    robustness_psd_plot(pd.DataFrame(spectra),output)
    return dict(windows=len(windows),conditions=conditions,clean_feature_and_decision_parity=True,
                bundle_sha256=digest(args.bundle),model=bundle["model_name"],threshold=bundle["threshold"],
                sr_mapping="experimental per-window voltage referencing" if args.include_sr else "not requested")


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command",choices=["analyse","robustness"])
    parser.add_argument("--run-id",required=True)
    parser.add_argument("--patient",required=True,choices=C.PATIENTS)
    parser.add_argument("--recording",required=True)
    parser.add_argument("--starts",type=float,nargs="+",required=True)
    parser.add_argument("--trials",type=int,default=3)
    parser.add_argument("--seed",type=int,default=42)
    parser.add_argument("--channel",default="FP1-F7")
    parser.add_argument("--noise-levels",type=float,nargs="+",default=[0.,.1,.5,2.])
    parser.add_argument("--substeps",type=int,nargs="+",default=[1])
    parser.add_argument("--bundle",type=Path)
    parser.add_argument("--noise-ratio",type=float,default=.3)
    parser.add_argument("--factor",type=int,default=4)
    parser.add_argument("--include-sr",action="store_true")
    parser.add_argument("--sr-noise",type=float,default=.5)
    args=parser.parse_args(argv)
    if not 1<=args.trials<=20 or not 1<=len(args.noise_levels)<=10 or any(not np.isfinite(d) or d<0 for d in args.noise_levels):
        parser.error("Bounded batch requires 1-20 trials and 1-10 nonnegative finite noise levels")
    if len(set(args.noise_levels))!=len(args.noise_levels) or len(set(args.substeps))!=len(args.substeps) or not set(args.substeps)<=set([1,2,4]):
        parser.error("Use distinct noise levels and substeps from 1, 2, 4")
    if not np.isfinite(args.noise_ratio) or args.noise_ratio<0 or not np.isfinite(args.sr_noise) or args.sr_noise<0 or not 1<=args.factor<=1024:
        parser.error("Invalid perturbation parameters")
    if args.command=="robustness" and args.bundle is None:
        parser.error("Robustness requires a saved bundle")
    if args.command=="analyse" and 0 not in args.noise_levels:
        parser.error("Include noise intensity 0 for the paired no-noise control")
    windows,signals,reference,dm,record=load_windows(args.patient,args.recording,args.starts)
    if args.channel not in dm["spec"]["channels"]:
        parser.error("Unknown channel")
    configuration={k:str(v) if isinstance(v,Path) else v for k,v in vars(args).items()}
    configuration.update(psd=dict(nperseg=256,overlap=128,window="hann",scaling="density"),
                         component="per-window unlabeled peak 1-30 Hz; bandwidth +/-0.5 Hz; zero mean/unit SD",
                         dynamics="bistable drive .2; FHN drive .35 epsilon .08 a .7 b .8; initial states from notebook",
                         event_detector="bistable: +/-0.5 hysteresis, 0.1s separation, five intervals, ddof1; FHN: mean+SD peaks, >=0.2s intervals, three intervals, ddof0",
                         uncertainty="trial dispersion only; not patient confidence intervals")
    output,manifest=new_run(args.run_id,configuration,dm,record)
    try:
        atomic_csv(output/"windows.csv",windows)
        if args.command=="analyse":
            result=analyse(windows,signals,dm["spec"],args,output)
        else:
            result=robustness(windows,signals,reference,dm["spec"],dm,args,output)
        manifest.update(status="complete",result=result,
                        artifact_hashes={p.name:digest(p) for p in output.iterdir() if p.name!="manifest.json"})
    except Exception as exc:
        manifest.update(status="failed",error=str(exc)); raise
    finally:
        atomic_json(output/"manifest.json",manifest)
    print(output)
    print(result)
