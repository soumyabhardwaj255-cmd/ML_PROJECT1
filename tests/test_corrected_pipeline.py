import json
import numpy as np
import pandas as pd
import pytest
from eeg_seizure import config as C
from eeg_seizure.artifacts import atomic_csv, atomic_json, digest
from eeg_seizure.dataset import window_rows, checkpoint_valid, validated_annotations
from eeg_seizure.modeling import FeatureTransform, make_model, select_features, train_student, student_scores
from eeg_seizure.evaluation import patient_split, tune_threshold
from eeg_seizure.inference import save_bundle, load_bundle, predict_features, contiguous_events


def test_small_magnitude_training_only_scaling():
    train = pd.DataFrame({"power": [1e-14, 2e-14, 4e-14, 8e-14], "constant": [0., 0., 0., 0.]})
    transform = FeatureTransform().fit(train)
    z = transform.transform(train)
    np.testing.assert_allclose(z[:, 0].std(), 1, atol=1e-12)
    before = transform.scaler_.mean_.copy()
    transform.transform(train * 100000)
    np.testing.assert_array_equal(before, transform.scaler_.mean_)
    np.testing.assert_array_equal(z, transform.transform(train[train.columns[::-1]]))
    assert np.isfinite(z).all()


def test_imputation_and_schema():
    x = pd.DataFrame({"a": [1., np.nan, 3.], "b": [4., 5., 6.]})
    t = FeatureTransform().fit(x)
    assert t.imputer_.statistics_[0] == 2
    with pytest.raises(ValueError):
        t.transform(x.drop(columns="a"))
    with pytest.raises(ValueError):
        t.transform(x.assign(a=np.inf))


def test_complete_final_window_and_overlap():
    w = window_rows(3600*256, 256., [(2996, 3036)], "p", "f")
    assert len(w) == 900
    assert w.iloc[-1].end_sec == 3600
    assert w.label.sum() == 10
    assert len(window_rows(959*256, 256., [], "p", "f")) == 239


def test_annotations_fail_closed(tmp_path):
    summary = tmp_path / "summary.txt"
    summary.write_text("File Name: f.edf\nNumber of Seizures in File: 1\nSeizure Start Time: 3 seconds\nSeizure End Time: 7 seconds\n")
    assert validated_annotations(summary, "f.edf", 10) == [(3, 7)]
    with pytest.raises(ValueError):
        validated_annotations(summary, "missing.edf", 10)


def test_checkpoint_rejects_partial_or_changed_identity(tmp_path):
    channels = [f"c{i}" for i in range(22)]
    w = window_rows(8*256, 256., [], "p", "f")
    f = pd.DataFrame(np.ones((2, 330)), columns=C.feature_names(channels))
    f = pd.concat([f, w], axis=1)
    path = tmp_path / "record.csv"
    marker = tmp_path / "record.json"
    identity = {"raw": "hash", "configuration": "v1"}
    atomic_csv(path, f)
    atomic_json(marker, {"identity": identity, "sha256": digest(path)})
    assert checkpoint_valid(path, marker, identity, w, channels) is not None
    assert checkpoint_valid(path, marker, {"raw": "different"}, w, channels) is None
    atomic_csv(path, f.iloc[:1])
    atomic_json(marker, {"identity": identity, "sha256": digest(path)})
    assert checkpoint_valid(path, marker, identity, w, channels) is None


def test_inner_weights_and_outer_separation(monkeypatch):
    frame = pd.DataFrame({"patient": np.repeat(["a", "b", "c", "test"], 8),
                          "file": "f", "start_sec": np.tile(np.arange(8)*4., 4),
                          "end_sec": np.tile(np.arange(8)*4.+4, 4),
                          "label": [0]*7+[1]+[0]*6+[1]*2+[0]*5+[1]*3+[0]*4+[1]*4,
                          "x": np.arange(32, dtype=float)})
    train, test = patient_split(frame, "test")
    seen = []
    def factory(name, y, seed):
        model = make_model(name, y, seed)
        seen.append(model.named_steps["model"].get_params()["scale_pos_weight"])
        model.named_steps["model"].set_params(n_estimators=2)
        return model
    monkeypatch.setattr("eeg_seizure.evaluation.make_model", factory)
    threshold, audit, oof = tune_threshold(train, ["x"], "xgboost", 42)
    assert len(oof) == len(train)
    assert set(oof.patient).isdisjoint(test.patient)
    for weight, fold in zip(seen, audit):
        assert "test" not in fold["train_patients"]
        assert fold["validation_patient"] not in fold["train_patients"]
        assert weight == fold["class_ratio"]
    assert len(set(seen)) > 1


def test_teacher_selection_paired_student_and_bundle(tmp_path):
    rng = np.random.default_rng(3)
    x = pd.DataFrame(rng.normal(size=(80, 330)), columns=[f"x{i}" for i in range(330)])
    x.iloc[:, :100] *= 1e-12
    y = np.tile([0, 0, 0, 1], 20)
    teacher = make_model("xgboost", y)
    teacher.named_steps["model"].set_params(n_estimators=3)
    teacher.fit(x, y)
    selected = select_features(teacher, x.columns).feature.tolist()
    assert len(selected) == 50
    t = FeatureTransform().fit(x[selected])
    z = t.transform(x[selected])
    a, _ = train_student(z, y, seed=7, epochs=1)
    b, _ = train_student(z, y, seed=7, epochs=1)
    np.testing.assert_array_equal(student_scores(a, z), student_scores(b, z))
    kd, history = train_student(z, y, seed=7, epochs=1, teacher_probability=teacher.predict_proba(x)[:, 1])
    kd_repeat, _ = train_student(z, y, seed=7, epochs=1, teacher_probability=teacher.predict_proba(x)[:, 1])
    np.testing.assert_array_equal(student_scores(kd, z), student_scores(kd_repeat, z))
    initial_baseline, _ = train_student(z, y, seed=7, epochs=0)
    initial_kd, _ = train_student(z, y, seed=7, epochs=0, teacher_probability=teacher.predict_proba(x)[:, 1])
    for a_param, b_param in zip(initial_baseline.parameters(), initial_kd.parameters()):
        np.testing.assert_array_equal(a_param.detach().numpy(), b_param.detach().numpy())
    assert len(history) == 1 and np.isfinite(history).all()
    bundle = dict(format_version=1, kind="student", selected_features=selected,
                  architecture=[50, 32, 16, 1],
                  feature_columns=list(x.columns), transform=t, state_dict=kd.state_dict(),
                  threshold=.5, train_patients=["a"], test_patients=["b"])
    save_bundle(tmp_path / "model.joblib", bundle)
    p, labels = predict_features(load_bundle(tmp_path / "model.joblib"), x[x.columns[::-1]])
    np.testing.assert_array_equal(p, student_scores(kd, z))
    assert sum(v.numel() for v in kd.parameters()) == 2177


def test_events_do_not_join_recordings():
    frame = pd.DataFrame({"patient": ["a"]*4, "file": ["f", "f", "g", "g"],
                          "start_sec": [0., 4., 8., 12.], "end_sec": [4., 8., 12., 16.],
                          "prediction": [1, 1, 1, 0]})
    events = contiguous_events(frame)
    assert len(events) == 2
    assert events.windows.tolist() == [2, 1]


@pytest.mark.parametrize("fault", ["missing_marker", "checksum", "channels", "feature_count", "duplicate_feature", "label"])
def test_checkpoint_rejects_bad_artifacts(tmp_path, fault):
    channels = [f"c{i}" for i in range(22)]
    w = window_rows(8*256, 256., [], "p", "f")
    f = pd.concat([pd.DataFrame(np.ones((2, 330)), columns=C.feature_names(channels)), w], axis=1)
    path, marker = tmp_path / "r.csv", tmp_path / "r.json"
    if fault == "feature_count":
        f = f.drop(columns=f.columns[0])
    elif fault == "duplicate_feature":
        f = pd.concat([f, f.iloc[:, :1]], axis=1)
    elif fault == "label":
        f.loc[0, "label"] = 1
    atomic_csv(path, f)
    if fault != "missing_marker":
        atomic_json(marker, {"identity": {"spec": "v1"}, "sha256": "wrong" if fault == "checksum" else digest(path)})
    if fault == "channels":
        channels[0] = "different"
    assert checkpoint_valid(path, marker, {"spec": "v1"}, w, channels) is None


@pytest.mark.parametrize("legacy_mismatch", [False, True])
def test_legacy_migration_and_checkpoint_resume(tmp_path, monkeypatch, legacy_mismatch):
    """Exercise both the complete-final-window and no-missing-tail migration branches."""
    import eeg_seizure.dataset as D
    channels = [f"c{i}" for i in range(22)]
    monkeypatch.setattr(C, "ROOT", tmp_path)
    monkeypatch.setattr(C, "DATA_DIR", tmp_path / "data/processed/corrected_v2")
    monkeypatch.setattr(C, "PATIENTS", ("p",))
    directory = tmp_path / "data/raw/p"
    directory.mkdir(parents=True)
    paths = [("p", directory / name) for name in ("complete.edf", "tail.edf")]
    for _, path in paths:
        path.write_bytes(b"synthetic source identity")
    (directory / "p-summary.txt").write_text("".join(
        f"File Name: {p.name}\nNumber of Seizures in File: 0\n" for _, p in paths))

    class Raw:
        def __init__(self, path):
            self.n_times = (8 if path.name == "complete.edf" else 9) * 256
            self.info = {"sfreq": 256.}
        def __enter__(self):
            return self
        def __exit__(self, *args):
            pass

    calls = []
    def extract(raw, windows, channels):
        calls.append(len(windows))
        f = pd.DataFrame(np.full((len(windows), 330), 1.2345678901234567e-14),
                         columns=C.feature_names(channels))
        return pd.concat([f, windows.reset_index(drop=True)], axis=1)
    monkeypatch.setattr(D, "load_channels", lambda: (paths, channels))
    monkeypatch.setattr(D.mne.io, "read_raw_edf", lambda path, **kwargs: Raw(path))
    monkeypatch.setattr(D, "dedupe_channel_names", lambda raw: raw)
    monkeypatch.setattr(D, "preprocess_raw", lambda raw, *args, **kwargs: raw)
    monkeypatch.setattr(D, "feature_rows", extract)
    old_frames = []
    for _, path in paths:
        raw = Raw(path)
        windows = window_rows(raw.n_times - 1, 256., [], "p", path.name)
        old_frames.append(extract(raw, windows, channels))
    old = pd.concat(old_frames, ignore_index=True)
    if legacy_mismatch:
        old.iloc[0, 0] = 99.
    atomic_csv(tmp_path / "data/processed/feature_table.csv", old)
    atomic_csv(tmp_path / "data/processed/window_index.csv", old[C.META])
    calls.clear()
    first = D.prepare(reuse_legacy=True)
    assert first["windows"] == 4 and first["previous_windows"] == 3
    assert sum(r["origin"] == "full_extraction" for r in first["records"]) == int(legacy_mismatch)
    assert len(calls) > 0
    first_hash = first["feature_sha256"]
    calls.clear()
    second = D.prepare(reuse_legacy=True)
    assert calls == []
    assert second["feature_sha256"] == first_hash
    assert all(r["origin"] == "validated_checkpoint" for r in second["records"])
    D.load_dataset()


def test_saved_architecture_independent_of_current_config(tmp_path, monkeypatch):
    from eeg_seizure.modeling import student_network
    x = pd.DataFrame(np.arange(20.).reshape(10, 2), columns=["a", "b"])
    t = FeatureTransform().fit(x)
    net = student_network(2, hidden=(3, 2))
    bundle = dict(kind="student", feature_columns=["a", "b"], selected_features=["a", "b"],
                  architecture=[2, 3, 2, 1], state_dict=net.state_dict(), transform=t, threshold=.5)
    monkeypatch.setitem(C.STUDENT, "hidden", (100, 100))
    scores, _ = predict_features(bundle, x)
    np.testing.assert_array_equal(scores, student_scores(net, t.transform(x)))


@pytest.mark.parametrize("name", ["logistic_regression", "random_forest"])
def test_classical_factories_and_saved_schema(tmp_path, name):
    rng = np.random.default_rng(14)
    x = pd.DataFrame(rng.normal(size=(40, 3)), columns=["a", "b", "c"])
    x.iloc[0, 0] = np.nan
    y = np.tile([0, 0, 0, 1], 10)
    model = make_model(name, y, seed=9)
    if name == "random_forest":
        model.named_steps["model"].set_params(n_estimators=3)
    model.fit(x, y)
    assert model.named_steps["features"].scale == (name == "logistic_regression")
    bundle = dict(format_version=1, kind="classical", estimator=model,
                  train_patients=["train"], test_patients=["test"],
                  feature_columns=list(x.columns), threshold=.73)
    save_bundle(tmp_path / "model.joblib", bundle)
    scores, pred = predict_features(load_bundle(tmp_path / "model.joblib"), x[["c", "a", "b"]])
    np.testing.assert_array_equal(scores, model.predict_proba(x)[:, 1])
    np.testing.assert_array_equal(pred, scores >= .73)
