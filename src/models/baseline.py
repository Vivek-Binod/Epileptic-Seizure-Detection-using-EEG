"""Classical machine learning baseline models for EEG seizure detection.

Supports:
- Random Forest with balanced class weights
- LightGBM Gradient Boosted Decision Trees with balanced weights
- Logistic Regression baseline

Integrates leakage-proof feature standardization (fitted strictly on training folds only).
"""

from typing import Dict, List, Optional, Tuple, Union
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
import lightgbm as lgb


class BaselineClassifier:
    """Classical ML baseline classifier with strictly encapsulated feature scaling."""

    SUPPORTED_MODELS = ["rf", "lgbm", "lr"]

    def __init__(
        self,
        model_type: str = "rf",
        random_state: int = 42,
        class_weight: str = "balanced",
        **model_kwargs,
    ):
        model_type_clean = model_type.lower().strip()
        if model_type_clean not in self.SUPPORTED_MODELS:
            raise ValueError(
                f"Unsupported model_type: '{model_type}'. Supported: {self.SUPPORTED_MODELS}"
            )

        self.model_type = model_type_clean
        self.random_state = random_state
        self.class_weight = class_weight
        self.model_kwargs = model_kwargs

        self.scaler = StandardScaler()
        self.model = self._create_model()
        self.is_fitted = False

    def _create_model(self):
        """Instantiate underlying estimator."""
        if self.model_type == "rf":
            kwargs = {
                "n_estimators": 100,
                "max_depth": 15,
                "class_weight": self.class_weight,
                "random_state": self.random_state,
                "n_jobs": -1,
                **self.model_kwargs,
            }
            return RandomForestClassifier(**kwargs)
        elif self.model_type == "lgbm":
            kwargs = {
                "n_estimators": 100,
                "max_depth": 6,
                "learning_rate": 0.05,
                "class_weight": self.class_weight,
                "random_state": self.random_state,
                "n_jobs": -1,
                "verbose": -1,
                **self.model_kwargs,
            }
            return lgb.LGBMClassifier(**kwargs)
        elif self.model_type == "lr":
            kwargs = {
                "max_iter": 1000,
                "class_weight": self.class_weight,
                "random_state": self.random_state,
                **self.model_kwargs,
            }
            return LogisticRegression(**kwargs)

    def fit(self, X: np.ndarray, y: np.ndarray) -> "BaselineClassifier":
        """Fit scaler and estimator strictly on training features.

        Args:
            X: Feature matrix of shape (N, D), e.g. (N, 252)
            y: Target binary labels (0 or 1), shape (N,)
        """
        X = np.asarray(X, dtype=np.float32)
        y = np.asarray(y, dtype=int)

        # Fit scaler ONLY on training data to prevent preprocessing leakage
        X_scaled = self.scaler.fit_transform(X)
        self.model.fit(X_scaled, y)
        self.is_fitted = True
        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Predict positive class probabilities.

        Args:
            X: Feature matrix of shape (N, D)

        Returns:
            1D array of positive class probabilities of shape (N,)
        """
        if not self.is_fitted:
            raise RuntimeError("Model must be fitted before predict_proba.")

        X = np.asarray(X, dtype=np.float32)
        # Transform using training fold scaler
        X_scaled = self.scaler.transform(X)

        proba = self.model.predict_proba(X_scaled)
        # Handle cases where model only learned one class
        if proba.shape[1] == 1:
            return np.zeros(len(X), dtype=float) if self.model.classes_[0] == 0 else np.ones(len(X), dtype=float)
        return proba[:, 1]

    def predict(self, X: np.ndarray, threshold: float = 0.5) -> np.ndarray:
        """Predict binary class labels based on decision threshold."""
        prob = self.predict_proba(X)
        return (prob >= threshold).astype(int)

    def get_feature_importances(
        self, feature_names: Optional[List[str]] = None
    ) -> Optional[pd.DataFrame]:
        """Extract sorted feature importance DataFrame if supported by model."""
        if not self.is_fitted:
            raise RuntimeError("Model must be fitted before getting feature importances.")

        if hasattr(self.model, "feature_importances_"):
            importances = self.model.feature_importances_
        elif hasattr(self.model, "coef_"):
            importances = np.abs(self.model.coef_.ravel())
        else:
            return None

        n_features = len(importances)
        if feature_names is None or len(feature_names) != n_features:
            names = [f"feature_{i}" for i in range(n_features)]
        else:
            names = feature_names

        df = pd.DataFrame({"feature": names, "importance": importances})
        return df.sort_values("importance", ascending=False).reset_index(drop=True)
