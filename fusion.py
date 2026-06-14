"""Hybrid fusion model: hand-crafted forensic signals + deep features.

This is the learned decision layer of the detector. It concatenates two
complementary feature families into one descriptor and trains a calibrated
classifier on the self-supervised pseudo-labels from :mod:`selfsup`:

  1. **Hand-crafted forensic features** (13-d, fully interpretable) - the seven
     statistical signals from :mod:`detector` plus the four pattern descriptors
     from :mod:`patterns`.
  2. **Deep features** (1024-d) - the frozen dual-stream backbone embedding from
     :mod:`deep_features`, dimensionality-reduced with PCA so it does not swamp
     the hand-crafted block on small corpora.

The fused vector feeds a scaler + classifier pipeline, optionally wrapped in
probability calibration. Everything is persisted with joblib so inference (in
``pipeline.py`` / the web app / CLI) is a single ``load`` + ``predict_proba``.

The same building blocks support the ablation study in ``evaluate.py``: the
``"handcrafted"``, ``"deep"`` and ``"fusion"`` modes select which feature
families are active.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
from PIL import Image

import deep_features
from detector import SIGNAL_NAMES, all_scores
from patterns import all_patterns

# --------------------------------------------------------------------------- #
# feature schema
# --------------------------------------------------------------------------- #

PATTERN_NAMES = (
    "spectral_peak",
    "block_hetero",
    "lbp",
    "lbp_uniform_mass",
    "lbp_entropy",
    "pattern_score",
)
HANDCRAFTED_NAMES: tuple[str, ...] = SIGNAL_NAMES + PATTERN_NAMES  # 13 features
N_HANDCRAFTED = len(HANDCRAFTED_NAMES)

MODE_HANDCRAFTED = "handcrafted"
MODE_DEEP = "deep"
MODE_FUSION = "fusion"
FEATURE_MODES = (MODE_HANDCRAFTED, MODE_DEEP, MODE_FUSION)


def handcrafted_vector(img: Image.Image) -> np.ndarray:
    """13-d interpretable forensic feature vector for one image."""
    s = all_scores(img)
    p = all_patterns(img)
    return np.array(
        [s[n] for n in SIGNAL_NAMES]
        + [
            p["spectral_peaks"]["score"],
            p["block_heterogeneity"]["score"],
            p["lbp"]["score"],
            p["lbp"]["uniform_mass"],
            p["lbp"]["entropy"],
            p["pattern_score"],
        ],
        dtype=np.float64,
    )


def handcrafted_matrix(imgs: list[Image.Image]) -> np.ndarray:
    return np.stack([handcrafted_vector(im) for im in imgs]) if imgs else np.empty((0, N_HANDCRAFTED))


def deep_matrix(imgs: list[Image.Image], spectral: bool = True) -> np.ndarray:
    """(N, 1024) deep embeddings. Requires torch/torchvision."""
    if not imgs:
        return np.empty((0, deep_features.DEEP_DIM if spectral else deep_features.BACKBONE_DIM))
    return deep_features.embeddings(imgs, spectral=spectral)


# --------------------------------------------------------------------------- #
# classifier zoo
# --------------------------------------------------------------------------- #


def _build_classifier(kind: str, seed: int):
    from sklearn.ensemble import (
        HistGradientBoostingClassifier,
        RandomForestClassifier,
    )
    from sklearn.linear_model import LogisticRegression
    from sklearn.neural_network import MLPClassifier
    from sklearn.svm import SVC

    if kind == "logreg":
        return LogisticRegression(C=1.0, max_iter=2000, random_state=seed)
    if kind == "mlp":
        return MLPClassifier(
            hidden_layer_sizes=(64, 32), alpha=1e-3, max_iter=3000,
            early_stopping=False, random_state=seed,
        )
    if kind == "gboost":
        return HistGradientBoostingClassifier(
            max_depth=3, learning_rate=0.1, max_iter=300,
            l2_regularization=1.0, random_state=seed,
        )
    if kind == "rf":
        return RandomForestClassifier(n_estimators=300, max_depth=None, random_state=seed)
    if kind == "svm":
        return SVC(kernel="rbf", C=1.0, gamma="scale", probability=True, random_state=seed)
    raise ValueError(f"unknown classifier kind: {kind!r}")


# --------------------------------------------------------------------------- #
# fusion model
# --------------------------------------------------------------------------- #


@dataclass
class FusionConfig:
    classifier: str = "gboost"
    mode: str = MODE_FUSION          # which feature families are active
    spectral: bool = True            # use spatial+spectral deep streams
    deep_pca_dim: int = 64           # PCA target for the deep block
    calibrate: bool = True           # wrap classifier in probability calibration
    seed: int = 0


class FusionModel:
    """Train / persist / apply the hybrid classifier."""

    def __init__(self, config: FusionConfig | None = None):
        self.config = config or FusionConfig()
        self._pca = None             # fitted sklearn PCA over deep block (or None)
        self._estimator = None       # fitted scaler+classifier pipeline
        self.feature_names_: list[str] = []
        self.train_meta_: dict = {}

    # -- feature assembly ---------------------------------------------------- #

    @property
    def uses_deep(self) -> bool:
        return self.config.mode in (MODE_DEEP, MODE_FUSION)

    @property
    def uses_handcrafted(self) -> bool:
        return self.config.mode in (MODE_HANDCRAFTED, MODE_FUSION)

    def _assemble(self, X_hand: np.ndarray | None, X_deep: np.ndarray | None,
                  fit_pca: bool) -> np.ndarray:
        blocks: list[np.ndarray] = []
        if self.uses_handcrafted:
            if X_hand is None:
                raise ValueError("hand-crafted features required for this mode")
            blocks.append(np.asarray(X_hand, dtype=np.float64))
        if self.uses_deep:
            if X_deep is None:
                raise ValueError("deep features required for this mode")
            deep = np.asarray(X_deep, dtype=np.float64)
            if fit_pca:
                from sklearn.decomposition import PCA

                k = min(self.config.deep_pca_dim, deep.shape[0], deep.shape[1])
                k = max(1, k)
                self._pca = PCA(n_components=k, random_state=self.config.seed).fit(deep)
            if self._pca is not None:
                deep = self._pca.transform(deep)
            blocks.append(deep)
        if not blocks:
            raise ValueError("no active feature blocks")
        return np.concatenate(blocks, axis=1)

    def _build_feature_names(self) -> list[str]:
        names: list[str] = []
        if self.uses_handcrafted:
            names += list(HANDCRAFTED_NAMES)
        if self.uses_deep and self._pca is not None:
            names += [f"deep_pc_{i}" for i in range(self._pca.n_components_)]
        return names

    # -- training ------------------------------------------------------------ #

    def fit(self, X_hand: np.ndarray | None, X_deep: np.ndarray | None,
            y: np.ndarray) -> "FusionModel":
        from sklearn.calibration import CalibratedClassifierCV
        from sklearn.pipeline import Pipeline
        from sklearn.preprocessing import StandardScaler

        y = np.asarray(y).astype(int)
        X = self._assemble(X_hand, X_deep, fit_pca=True)

        clf = _build_classifier(self.config.classifier, self.config.seed)
        pipe = Pipeline([("scaler", StandardScaler()), ("clf", clf)])

        # Calibrate only when each class has enough members for CV folds.
        min_class = int(min(np.bincount(y, minlength=2)))
        if self.config.calibrate and min_class >= 4:
            cv = min(3, min_class)
            estimator = CalibratedClassifierCV(pipe, method="sigmoid", cv=cv)
            self.train_meta_["calibrated"] = f"sigmoid, cv={cv}"
        else:
            estimator = pipe
            self.train_meta_["calibrated"] = "no (insufficient per-class samples)"

        estimator.fit(X, y)
        self._estimator = estimator
        self.feature_names_ = self._build_feature_names()
        self.train_meta_.update(
            n_train=int(len(y)),
            n_real=int((y == 0).sum()),
            n_fake=int((y == 1).sum()),
            n_features=int(X.shape[1]),
            config=asdict(self.config),
        )
        return self

    # -- inference ----------------------------------------------------------- #

    def predict_proba(self, X_hand: np.ndarray | None, X_deep: np.ndarray | None) -> np.ndarray:
        if self._estimator is None:
            raise RuntimeError("model not fitted / loaded")
        X = self._assemble(X_hand, X_deep, fit_pca=False)
        return self._estimator.predict_proba(X)[:, 1]

    def predict_proba_images(self, imgs: list[Image.Image]) -> np.ndarray:
        X_hand = handcrafted_matrix(imgs) if self.uses_handcrafted else None
        X_deep = deep_matrix(imgs, spectral=self.config.spectral) if self.uses_deep else None
        return self.predict_proba(X_hand, X_deep)

    def predict_one(self, img: Image.Image) -> float:
        return float(self.predict_proba_images([img])[0])

    # -- persistence --------------------------------------------------------- #

    def save(self, path: str | Path) -> None:
        import joblib

        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self, path)
        meta_path = path.with_suffix(".meta.json")
        meta_path.write_text(json.dumps(self.train_meta_, indent=2))

    @classmethod
    def load(cls, path: str | Path) -> "FusionModel":
        import joblib

        obj = joblib.load(path)
        if not isinstance(obj, cls):
            raise TypeError(f"{path} did not contain a FusionModel")
        return obj


# --------------------------------------------------------------------------- #
# unsupervised one-class fallback (zero pseudo-labels needed)
# --------------------------------------------------------------------------- #


class OneClassDetector:
    """Pure-anomaly detector: model the *real* feature manifold and flag
    outliers. Useful as a fully unsupervised baseline / cold-start option when
    not even pseudo-fakes are available."""

    def __init__(self, mode: str = MODE_HANDCRAFTED, spectral: bool = True,
                 deep_pca_dim: int = 32, seed: int = 0):
        self.config = FusionConfig(mode=mode, spectral=spectral,
                                   deep_pca_dim=deep_pca_dim, seed=seed)
        self._fm = FusionModel(self.config)
        self._scaler = None
        self._iforest = None

    def fit(self, X_hand, X_deep) -> "OneClassDetector":
        from sklearn.ensemble import IsolationForest
        from sklearn.preprocessing import StandardScaler

        X = self._fm._assemble(X_hand, X_deep, fit_pca=True)
        self._scaler = StandardScaler().fit(X)
        self._iforest = IsolationForest(random_state=self.config.seed, contamination="auto")
        self._iforest.fit(self._scaler.transform(X))
        return self

    def anomaly_score(self, X_hand, X_deep) -> np.ndarray:
        X = self._fm._assemble(X_hand, X_deep, fit_pca=False)
        # IsolationForest: higher score_samples = more normal; invert to [0,1].
        raw = -self._iforest.score_samples(self._scaler.transform(X))
        return raw
