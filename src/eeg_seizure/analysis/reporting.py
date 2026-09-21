"""Report raw measured optima; keep invalid trials and missing grid points visible."""
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def summarize(trials):
    keys = ["patient", "file", "start_sec", "end_sec", "label", "model", "drive", "noise_intensity", "substeps"]
    rows = []
    for key, group in trials.groupby(keys, dropna=False):
        finite = group.loc[group.valid & np.isfinite(group.coherence), "coherence"]
        snr = group.loc[np.isfinite(group.snr_db), "snr_db"]
        rows.append(dict(zip(keys, key)) | dict(trials=len(group), valid_trials=len(finite),
                    invalid_trials=len(group)-len(finite), valid_fraction=len(finite)/len(group),
                    coherence_median=finite.median() if len(finite) else np.nan,
                    coherence_sd=finite.std(ddof=1) if len(finite)>1 else np.nan,
                    snr_mean=snr.mean() if len(snr) else np.nan, snr_valid_trials=len(snr),
                    snr_sd=snr.std(ddof=1) if len(snr)>1 else np.nan,
                    events_mean=group.events.mean(), clipped_fraction_max=group.clipped_fraction.max()))
    return pd.DataFrame(rows)


def raw_optima(summary):
    rows = []
    keys = ["patient", "file", "start_sec", "end_sec", "label", "model", "drive", "substeps"]
    for key, group in summary.groupby(keys):
        group = group.sort_values("noise_intensity")
        for metric in ("coherence_median", "snr_mean"):
            valid = group[np.isfinite(group[metric])]
            if valid.empty:
                rows.append(dict(zip(keys,key)) | dict(metric=metric, status="no_valid_measurement"))
                continue
            best = valid.loc[valid[metric].idxmax()]
            boundary = best.noise_intensity in (group.noise_intensity.min(), group.noise_intensity.max())
            rows.append(dict(zip(keys,key)) | dict(metric=metric, noise_intensity=best.noise_intensity,
                        value=best[metric], status="boundary_maximum" if boundary else "interior_grid_maximum",
                        valid_trials=best.valid_trials if metric=="coherence_median" else best.snr_valid_trials,
                        interpretation="descriptive only; not a resonance claim"))
    return pd.DataFrame(rows)


def dynamics_plot(summary, output):
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    for key, group in summary.groupby(["start_sec", "model", "drive", "substeps"]):
        group = group.sort_values("noise_intensity")
        label = f"{key[0]:g}s / {key[1]} / {key[2]} / steps={key[3]}"
        axes[0].plot(group.noise_intensity, group.coherence_median, marker="o", label=label)
        axes[1].plot(group.noise_intensity, group.valid_fraction, marker="o")
    axes[0].set(xlabel="Noise intensity D", ylabel="Median 1/CV over valid trials", title="Raw measurements; gaps preserved")
    axes[1].set(xlabel="Noise intensity D", ylabel="Valid trial fraction", ylim=(-.05,1.05), title="Reliability of interval estimates")
    axes[0].legend(fontsize=5)
    fig.tight_layout(); fig.savefig(output / "coherence_raw_and_validity.png", dpi=150); plt.close(fig)


def psd_plot(table, output):
    fig, ax = plt.subplots(figsize=(8, 4))
    for start, group in table.groupby("start_sec"):
        ax.semilogy(group.frequency_hz, group.psd_v2_hz, label=f"start {start:g}s")
    ax.set(xlabel="Frequency (Hz)", ylabel="PSD (V²/Hz)", xlim=(0,40), title="Canonical-window PSD; no injected noise")
    ax.legend(); fig.tight_layout(); fig.savefig(output / "canonical_psd.png", dpi=150); plt.close(fig)


def robustness_psd_plot(table, output):
    starts=table.start_sec.unique()
    fig,axes=plt.subplots(len(starts),1,figsize=(10,3*len(starts)),squeeze=False)
    for ax,start in zip(axes[:,0],starts):
        for (condition,trial),g in table[table.start_sec==start].groupby(["condition","trial"]):
            ax.semilogy(g.frequency_hz,g.psd_v2_hz,alpha=.7,label=condition if trial==0 else None)
        ax.set(xlim=(0,60),xlabel="Frequency (Hz)",ylabel="PSD (V²/Hz)",title=f"Window {start:g}s: post-filter perturbations")
        ax.legend(fontsize=6)
    fig.tight_layout();fig.savefig(output/"perturbed_psd.png",dpi=150);plt.close(fig)
