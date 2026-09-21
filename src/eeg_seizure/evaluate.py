"""
Evaluation helpers: precision/recall/F1 and a confusion matrix. Accuracy
alone is meaningless here given the ~1.6% positive class (see Day 2 notes).
"""

from sklearn.metrics import classification_report, confusion_matrix, f1_score, precision_score, recall_score


def evaluate_predictions(y_true, y_pred, label="", verbose=True):
    """
    Compute precision/recall/F1 for a set of predictions.

    Parameters
    ----------
    verbose : bool
        If True (default), prints the full report. If False, just returns
        the metrics dict — useful inside loops (e.g. cross-validation)
        where a full report per fold would be too much output.

    Returns
    -------
    dict with keys: precision, recall, f1
    """
    precision = precision_score(y_true, y_pred, zero_division=0)
    recall = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)

    if verbose:
        header = f"--- {label} ---" if label else "---"
        print(header)
        print(f"Precision: {precision:.3f}")
        print(f"Recall:    {recall:.3f}")
        print(f"F1:        {f1:.3f}")
        print("\nConfusion matrix (rows=actual, cols=predicted):")
        print(confusion_matrix(y_true, y_pred))
        print("\nFull report:")
        print(
            classification_report(
                y_true, y_pred, target_names=["non-seizure", "seizure"], zero_division=0
            )
        )

    return {"precision": precision, "recall": recall, "f1": f1}