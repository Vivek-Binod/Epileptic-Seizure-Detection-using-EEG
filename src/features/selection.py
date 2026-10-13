"""Feature selection module for EEG multimodal classification.

Provides:
- FeatureSelector: Univariate ranking and top-K selection (Mutual Information, RF Gini).
- Feature family accounting (Conventional vs PLV vs PLI).
- Strict leak-free training-only fit and transform.
"""

from typing import Dict, List, Optional, Tuple, Union
import numpy as np
import pandas as pd
from sklearn.feature_selection import mutual_info_classif


class FeatureSelector:
    """Supervised feature selector with feature family accounting."""

    SUPPORTED_METHODS = ["mutual_info", "rf_gini"]

    def __init__(
        self,
        method: str = "mutual_info",
        k: int = 80,
        random_state: int = 42,
    ):
        clean_method = method.lower().strip()
        if clean_method not in self.SUPPORTED_METHODS:
            raise ValueError(
                f"Unsupported method '{method}'. Choose from {self.SUPPORTED_METHODS}"
            )
        if k <= 0:
            raise ValueError(f"k must be positive, got {k}")

        self.method = clean_method
        self.k = k
        self.random_state = random_state

        self.selected_indices_: Optional[np.ndarray] = None
        self.selected_features_: Optional[List[str]] = None
        self.feature_scores_: Optional[np.ndarray] = None
        self.feature_names_: Optional[List[str]] = None
        self.is_fitted: bool = False

    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        feature_names: Optional[List[str]] = None,
    ) -> "FeatureSelector":
        """Fit feature selector strictly on training data."""
        X_arr = np.asarray(X, dtype=float)
        y_arr = np.asarray(y, dtype=int)
        n_features = X_arr.shape[1]

        if feature_names is not None:
            if len(feature_names) != n_features:
                raise ValueError(
                    f"Length of feature_names ({len(feature_names)}) does not match X columns ({n_features})"
                )
            self.feature_names_ = list(feature_names)
        else:
            self.feature_names_ = [f"feat_{i}" for i in range(n_features)]

        actual_k = min(self.k, n_features)

        if self.method == "mutual_info":
            scores = mutual_info_classif(
                X_arr,
                y_arr,
                discrete_features=False,
                n_neighbors=3,
                random_state=self.random_state,
            )
        elif self.method == "rf_gini":
            from sklearn.ensemble import RandomForestClassifier
            rf = RandomForestClassifier(
                n_estimators=100,
                max_depth=15,
                class_weight="balanced",
                random_state=self.random_state,
                n_jobs=-1,
            )
            rf.fit(X_arr, y_arr)
            scores = rf.feature_importances_
        else:
            raise ValueError(f"Unknown method {self.method}")

        self.feature_scores_ = scores
        # Rank features descending by score
        sorted_indices = np.argsort(scores)[::-1]
        self.selected_indices_ = sorted_indices[:actual_k]
        self.selected_features_ = [self.feature_names_[i] for i in self.selected_indices_]
        self.is_fitted = True

        return self

    def set_k(self, new_k: int) -> "FeatureSelector":
        """Update top-K selection without recomputing feature scores."""
        if not self.is_fitted or self.feature_scores_ is None:
            raise RuntimeError("Selector must be fitted before set_k can be called")
        if new_k <= 0:
            raise ValueError(f"new_k must be positive, got {new_k}")
        self.k = new_k
        sorted_indices = np.argsort(self.feature_scores_)[::-1]
        actual_k = min(new_k, len(sorted_indices))
        self.selected_indices_ = sorted_indices[:actual_k]
        self.selected_features_ = [self.feature_names_[i] for i in self.selected_indices_]
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:

        """Slice matrix to selected feature columns."""
        if not self.is_fitted or self.selected_indices_ is None:
            raise RuntimeError("FeatureSelector must be fitted before calling transform")
        X_arr = np.asarray(X, dtype=float)
        return X_arr[:, self.selected_indices_]

    def fit_transform(
        self,
        X: np.ndarray,
        y: np.ndarray,
        feature_names: Optional[List[str]] = None,
    ) -> np.ndarray:
        """Fit selector and transform training data."""
        self.fit(X, y, feature_names=feature_names)
        return self.transform(X)

    def get_family_breakdown(self) -> Dict[str, Union[int, float]]:
        """Compute distribution of selected features across feature families."""
        if not self.is_fitted or self.selected_features_ is None:
            raise RuntimeError("Selector must be fitted first")

        n_conv = 0
        n_plv = 0
        n_pli = 0

        for f in self.selected_features_:
            if "plv" in f:
                n_plv += 1
            elif "pli" in f:
                n_pli += 1
            else:
                n_conv += 1

        total = len(self.selected_features_)
        return {
            "total_selected": total,
            "conventional_count": n_conv,
            "plv_count": n_plv,
            "pli_count": n_pli,
            "connectivity_count": n_plv + n_pli,
            "conventional_pct": float(n_conv / total * 100.0) if total > 0 else 0.0,
            "connectivity_pct": float((n_plv + n_pli) / total * 100.0) if total > 0 else 0.0,
        }

    def get_selected_features_summary(self) -> pd.DataFrame:
        """Return DataFrame of selected features with scores and categories."""
        if not self.is_fitted or self.selected_indices_ is None:
            raise RuntimeError("Selector must be fitted first")

        rows = []
        for rank, idx in enumerate(self.selected_indices_, start=1):
            name = self.feature_names_[idx]
            score = self.feature_scores_[idx]

            if "plv" in name:
                cat = "PLV Connectivity"
            elif "pli" in name:
                cat = "PLI Connectivity"
            else:
                cat = "Conventional Univariate"

            rows.append({
                "rank": rank,
                "feature_index": int(idx),
                "feature_name": name,
                "importance_score": float(score),
                "category": cat,
            })

        return pd.DataFrame(rows)
