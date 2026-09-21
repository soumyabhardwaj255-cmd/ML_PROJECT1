"""Shared training transforms, model factories and deterministic students."""
import random
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from . import config as C


class FeatureTransform(BaseEstimator, TransformerMixin):
    """Enforce feature schema, impute and scale on training data only.

    StandardScaler operates in float64 and only treats numerically constant
    columns as constant; there is no absolute voltage-magnitude cutoff.
    """
    def __init__(self, scale=True):
        self.scale = scale

    def _array(self, X):
        if not isinstance(X, pd.DataFrame):
            raise TypeError("Features must be a named DataFrame")
        if X.columns.duplicated().any() or set(X.columns) != set(self.columns_):
            raise ValueError("Feature schema differs from fitted transform")
        a = X.loc[:, self.columns_].to_numpy(dtype=np.float64)
        if np.isinf(a).any():
            raise ValueError("Infinite features are invalid")
        return a

    def fit(self, X, y=None):
        self.columns_ = list(X.columns)
        a = self._array(X)
        if np.isnan(a).all(axis=0).any():
            raise ValueError("Cannot fit an entirely missing feature")
        self.imputer_ = SimpleImputer(strategy="median", keep_empty_features=True)
        a = self.imputer_.fit_transform(a)
        self.scaler_ = StandardScaler().fit(a) if self.scale else None
        if self.scale:
            z = self.scaler_.transform(a)
            active = self.scaler_.var_ > 0
            # Detect regression to the old std < 1e-8 -> 1 bug.
            if not np.allclose(z[:, active].std(axis=0), 1.0, rtol=1e-5, atol=1e-5):
                raise ValueError("Nonconstant training features were not standardized")
        return self

    def transform(self, X):
        a = self.imputer_.transform(self._array(X))
        a = self.scaler_.transform(a) if self.scale else a
        if not np.isfinite(a).all():
            raise ValueError("Nonfinite transformed features")
        return a


def class_ratio(y):
    y = np.asarray(y)
    if set(np.unique(y)) != {0, 1}:
        raise ValueError("Training requires both binary classes")
    return float((y == 0).sum() / (y == 1).sum())


def make_model(name, y_train, seed=C.SEED):
    if name == "logistic_regression":
        estimator = LogisticRegression(**C.LR_PARAMS, random_state=seed)
    elif name == "random_forest":
        estimator = RandomForestClassifier(**C.RF_PARAMS, random_state=seed)
    elif name == "xgboost":
        estimator = XGBClassifier(**C.XGB_PARAMS, random_state=seed,
                                  scale_pos_weight=class_ratio(y_train))
    else:
        raise ValueError(f"Unknown model: {name}")
    return Pipeline([("features", FeatureTransform(scale=name == "logistic_regression")),
                     ("model", estimator)])


def select_features(teacher, columns, k=C.TOP_K):
    importance = teacher.named_steps["model"].feature_importances_
    ranking = pd.DataFrame({"feature": columns, "importance": importance})
    return ranking.sort_values(["importance", "feature"], ascending=[False, True]).head(k)


def train_student(X, y, seed=C.SEED, teacher_probability=None, epochs=None):
    import torch
    from torch import nn
    from torch.utils.data import DataLoader, TensorDataset
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.use_deterministic_algorithms(True)
    torch.set_num_threads(2)
    model = student_network(X.shape[1])
    x = torch.tensor(np.asarray(X), dtype=torch.float32)
    target = torch.tensor(np.asarray(y), dtype=torch.float32)
    if not torch.isfinite(x).all():
        raise ValueError("Student input must be finite")
    soft = np.zeros(len(y), dtype=np.float32)
    if teacher_probability is not None:
        p = np.clip(teacher_probability, 1e-6, 1 - 1e-6)
        soft = 1 / (1 + np.exp(-np.log(p / (1 - p)) / C.STUDENT["temperature"]))
    dataset = TensorDataset(x, target, torch.tensor(soft, dtype=torch.float32))
    generator = torch.Generator().manual_seed(seed)
    loader = DataLoader(dataset, batch_size=C.STUDENT["batch_size"], shuffle=True,
                        generator=generator, num_workers=0)
    optimizer = torch.optim.Adam(model.parameters(), lr=C.STUDENT["learning_rate"],
                                 weight_decay=C.STUDENT["weight_decay"])
    hard_loss = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(class_ratio(y)))
    history = []
    for epoch in range(epochs if epochs is not None else C.STUDENT["epochs"]):
        total = 0.0
        for xb, yb, tb in loader:
            optimizer.zero_grad()
            logits = model(xb).squeeze(1)
            loss = hard_loss(logits, yb)
            if teacher_probability is not None:
                temperature = C.STUDENT["temperature"]
                soft_loss = nn.functional.binary_cross_entropy_with_logits(logits / temperature, tb)
                loss = C.STUDENT["alpha"] * loss + (1 - C.STUDENT["alpha"]) * temperature**2 * soft_loss
            if not torch.isfinite(loss):
                raise ValueError("Nonfinite student loss")
            loss.backward()
            optimizer.step()
            total += loss.item() * len(xb)
        history.append(total / len(dataset))
    return model.eval(), history


def student_network(input_dim, hidden=None):
    from torch import nn
    h1, h2 = C.STUDENT["hidden"] if hidden is None else hidden
    return nn.Sequential(nn.Linear(input_dim, h1), nn.ReLU(),
                         nn.Linear(h1, h2), nn.ReLU(), nn.Linear(h2, 1))


def student_scores(model, X):
    import torch
    model.eval()
    with torch.no_grad():
        return torch.sigmoid(model(torch.tensor(X, dtype=torch.float32)).squeeze(1)).numpy()
