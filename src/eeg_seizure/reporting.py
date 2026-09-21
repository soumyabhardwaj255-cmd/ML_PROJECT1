"""Figures always consume an identified run; aggregation is explicit."""
import json
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
from .artifacts import digest
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, precision_recall_curve, average_precision_score


def report(run):
    run = Path(run)
    manifest = json.loads((run / "manifest.json").read_text(encoding="utf-8"))
    if manifest["status"] != "complete":
        raise ValueError("Cannot report an incomplete run")
    if digest(run / "predictions.csv") != manifest["predictions_sha256"]:
        raise ValueError("Run predictions checksum mismatch")
    predictions = pd.read_csv(run / "predictions.csv")
    output = run / "figures"
    output.mkdir(exist_ok=False)
    qualifier = "validation run" if manifest["validation_only"] else "LOPO experiment"
    means = pd.read_csv(run / "mean_patient_metrics.csv")
    labels = means.model + " / seed " + means.seed.astype(str)
    fig, ax = plt.subplots(figsize=(9, 5))
    has_sd = means.f1_std.notna().all()
    ax.bar(labels, means.f1_mean, yerr=means.f1_std if has_sd else None, capsize=4)
    ylabel = "Mean patient F1 (error bars: patient SD)" if has_sd else "Patient F1 (single patient; SD unavailable)"
    ax.set(ylabel=ylabel, ylim=(0, 1),
           title=f"{qualifier}: patient-level F1")
    ax.tick_params(axis="x", labelrotation=20)
    fig.tight_layout()
    fig.savefig(output / "mean_patient_f1.png", dpi=180)
    plt.close(fig)
    if (run / "feature_importance.csv").exists():
        importance = pd.read_csv(run / "feature_importance.csv")
        for (model, seed), group in importance.groupby(["model", "seed"]):
            ranking = group.groupby("feature").importance.mean().nlargest(20).sort_values()
            fig, ax = plt.subplots(figsize=(10, 8))
            ranking.plot.barh(ax=ax)
            ax.set(xlabel="Mean importance across training folds", title=f"{model}/{seed}: {qualifier}")
            fig.tight_layout()
            fig.savefig(output / f"{model}_seed{seed}_feature_importance.png", dpi=180)
            plt.close(fig)
    for (model, seed), g in predictions.groupby(["model", "seed"]):
        fig, ax = plt.subplots(figsize=(6, 5))
        ConfusionMatrixDisplay(confusion_matrix(g.label, g.prediction, labels=[0, 1]),
                               display_labels=["Non-seizure", "Seizure"]).plot(ax=ax, cmap="Blues", values_format="d")
        ax.set_title(f"{model}, seed {seed}\nPooled windows ({qualifier})")
        fig.tight_layout()
        fig.savefig(output / f"{model}_seed{seed}_pooled_confusion.png", dpi=180)
        plt.close(fig)
    fig, ax = plt.subplots(figsize=(8, 6))
    for (model, seed), g in predictions.groupby(["model", "seed"]):
        precision, recall, _ = precision_recall_curve(g.label, g.probability_seizure)
        ap = average_precision_score(g.label, g.probability_seizure)
        ax.plot(recall, precision, label=f"{model}/{seed}: AP={ap:.3f}")
    ax.set(xlabel="Recall", ylabel="Precision", title=f"Pooled-window PR curves ({qualifier})")
    ax.legend()
    fig.tight_layout()
    fig.savefig(output / "pooled_precision_recall.png", dpi=180)
    plt.close(fig)


def timeline(output, frame, threshold):
    """Recording-relative predictions; known labels are window labels only."""
    fig, ax = plt.subplots(figsize=(12, 4))
    ax.plot(frame.start_sec, frame.probability_seizure, label="Seizure probability", linewidth=1)
    ax.axhline(threshold, color="black", linestyle="--", label=f"Threshold {threshold:.2f}")
    if "label" in frame:
        ax.fill_between(frame.start_sec, 0, 1, where=frame.label.astype(bool),
                        step="post", alpha=0.15, color="red", label="Positive annotated windows")
    ax.set(xlabel="Time from recording start (seconds)", ylabel="Probability", ylim=(0, 1),
           title="Offline recording inference (window-level)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(Path(output) / "prediction_timeline.png", dpi=180)
    plt.close(fig)
