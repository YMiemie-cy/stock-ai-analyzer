"""Model training and inference logic."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional

import numpy as np
import joblib
import pandas as pd
from sklearn.base import ClassifierMixin
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from itertools import product

from sklearn.metrics import classification_report, accuracy_score, f1_score, recall_score
from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import StandardScaler
from sklearn.calibration import CalibratedClassifierCV

try:  # pragma: no cover - optional dependency
    import lightgbm as lgb  # type: ignore
except Exception:
    lgb = None

MODEL_DIR = Path(__file__).resolve().parent.parent / "models"
MODEL_DIR.mkdir(exist_ok=True)

LABEL = "label"


@dataclass
class ModelArtifacts:
    scaler: StandardScaler
    model: ClassifierMixin
    class_priors: Dict[str, float] | None = None
    market: str = "global"
    bias_factors: Dict[str, float] | None = None
    cv_metrics: Dict[str, float] | None = None
    model_type: str = "hist_gb"
    model_params: Dict[str, object] | None = None
    ensemble_members: List[Dict[str, object]] | None = None
    meta_scaler: Optional[StandardScaler] = None
    meta_model: Optional[ClassifierMixin] = None
    meta_feature_specs: Optional[List[Dict[str, str]]] = None

    def save(self, name: str) -> Path:
        path = MODEL_DIR / f"{name}.joblib"
        joblib.dump(self, path, compress=3)
        return path

    @classmethod
    def load(cls, path: Path) -> "ModelArtifacts":
        return joblib.load(path)


def _prob_gap_from_matrix(prob_matrix: np.ndarray) -> np.ndarray:
    if prob_matrix.size == 0:
        return np.zeros(prob_matrix.shape[0])
    sorted_probs = np.sort(prob_matrix, axis=1)
    if sorted_probs.shape[1] < 2:
        return sorted_probs[:, -1]
    return (sorted_probs[:, -1] - sorted_probs[:, -2]).astype(float)


def _prob_max_from_matrix(prob_matrix: np.ndarray) -> np.ndarray:
    if prob_matrix.size == 0:
        return np.zeros(prob_matrix.shape[0])
    return np.max(prob_matrix, axis=1).astype(float)


FEATURE_COLUMNS = [
    "Open",
    "High",
    "Low",
    "Close",
    "Adj Close",
    "Volume",
    "sma_10",
    "sma_50",
    "sma_100",
    "sma_200",
    "ema_20",
    "ema_50",
    "ema_100",
    "ema_200",
    "ema_20_slope_5d",
    "ema_50_slope_5d",
    "ema_200_slope_10d",
    "rsi_7",
    "rsi_14",
    "rsi_28",
    "macd",
    "macd_signal",
    "macd_hist",
    "bb_up",
    "bb_mid",
    "bb_low",
    "bb_pct",
    "bollinger_width",
    "keltner_channel_width",
    "return_1d",
    "return_5d",
    "return_3d",
    "return_10d",
    "return_20d",
    "return_60d",
    "return_120d",
    "return_horizon_past",
    "volatility_20d",
    "volatility_60d",
    "volatility_120d",
    "return_zscore_20d",
    "skew_60d",
    "kurtosis_60d",
    "volume_zscore",
    "volume_ratio_20",
    "volume_ratio_5",
    "atr_14",
    "atr_ratio",
    "drawdown",
    "price_sma_50_ratio",
    "price_ema_50_ratio",
    "price_ema_100_ratio",
    "pct_from_52w_high",
    "pct_from_52w_low",
    "bench_return_5d",
    "bench_return_20d",
    "bench_return_60d",
    "relative_return_20d",
    "relative_return_60d",
    "relative_strength_pct",
    "adx",
    "adx_pos",
    "adx_neg",
    "cci",
    "stoch_rsi",
    "stoch_rsi_k",
    "stoch_rsi_d",
    "williams_r",
    "chaikin_mf",
    "obv",
    "obv_pct_change",
    "atr_trend",
    "volatility_ratio",
    "return_momentum_diff",
    "trend_strength_20d",
    "trend_strength_60d",
    "benchmark_spread",
    "macro_vix_level",
    "macro_vix_pct_change_5d",
    "macro_vix_zscore_60d",
    "macro_dxy_level",
    "macro_dxy_pct_change_5d",
    "macro_dxy_zscore_60d",
    "macro_us10y_level",
    "macro_us10y_pct_change_5d",
    "macro_us10y_zscore_60d",
    "macro_vix_ema_ratio_30",
    "macro_dxy_ema_ratio_30",
    "macro_us10y_ema_ratio_30",
    "macro_region_csi300_level",
    "macro_region_csi300_pct_change_5d",
    "macro_region_csi300_zscore_60d",
    "macro_region_csi300_ema_ratio_30",
    "macro_region_shanghai_level",
    "macro_region_shanghai_pct_change_5d",
    "macro_region_shanghai_zscore_60d",
    "macro_region_shanghai_ema_ratio_30",
    "macro_region_cnh_level",
    "macro_region_cnh_pct_change_5d",
    "macro_region_cnh_zscore_60d",
    "macro_region_cnh_ema_ratio_30",
    "macro_region_hang_seng_level",
    "macro_region_hang_seng_pct_change_5d",
    "macro_region_hang_seng_zscore_60d",
    "macro_region_hang_seng_ema_ratio_30",
    "macro_region_nasdaq_level",
    "macro_region_nasdaq_pct_change_5d",
    "macro_region_nasdaq_zscore_60d",
    "macro_region_nasdaq_ema_ratio_30",
    "macro_region_russell_level",
    "macro_region_russell_pct_change_5d",
    "macro_region_russell_zscore_60d",
    "macro_region_russell_ema_ratio_30",
    "macro_region_sp500_level",
    "macro_region_sp500_pct_change_5d",
    "macro_region_sp500_zscore_60d",
    "macro_region_sp500_ema_ratio_30",
    "sector_tech_level",
    "sector_tech_pct_change_5d",
    "sector_tech_zscore_60d",
    "sector_tech_ema_ratio_30",
    "sector_finance_level",
    "sector_finance_pct_change_5d",
    "sector_finance_zscore_60d",
    "sector_finance_ema_ratio_30",
    "sector_energy_level",
    "sector_energy_pct_change_5d",
    "sector_energy_zscore_60d",
    "sector_energy_ema_ratio_30",
    "sector_healthcare_level",
    "sector_healthcare_pct_change_5d",
    "sector_healthcare_zscore_60d",
    "sector_healthcare_ema_ratio_30",
    "sector_industry_level",
    "sector_industry_pct_change_5d",
    "sector_industry_zscore_60d",
    "sector_industry_ema_ratio_30",
    "macro_liquidity_spread",
    "macro_risk_spread",
    "macro_vix_to_us10y",
    "regime_volatility",
    "regime_trend",
    "regime_risk_on",
    "regime_score",
    "trend_alignment_score",
    "fundamental_pe_ratio",
    "fundamental_forward_pe",
    "fundamental_price_to_book",
    "fundamental_beta",
    "fundamental_dividend_yield",
    "fundamental_market_cap_log",
    "range_pct",
    "gap_pct",
    "volume_trend_60d",
    "volume_pct_rank_60d",
    "price_zscore_60d",
    "price_to_ema_20",
    "price_to_ema_100",
    "price_to_bollinger_band",
]

DEFAULT_PARAM_CANDIDATES = (
    {"learning_rate": 0.035, "max_leaf_nodes": 45, "min_samples_leaf": 18, "l2_regularization": 0.02},
    {"learning_rate": 0.04, "max_leaf_nodes": 63, "min_samples_leaf": 18, "l2_regularization": 0.012},
    {"learning_rate": 0.05, "max_leaf_nodes": 63, "min_samples_leaf": 20, "l2_regularization": 0.01},
    {"learning_rate": 0.055, "max_leaf_nodes": 95, "min_samples_leaf": 25, "l2_regularization": 0.0075},
    {"learning_rate": 0.065, "max_leaf_nodes": 127, "min_samples_leaf": 30, "l2_regularization": 0.005},
)

DEFAULT_RF_PARAM_CANDIDATES = (
    {"n_estimators": 400, "max_depth": None, "min_samples_leaf": 2, "max_features": "sqrt"},
    {"n_estimators": 500, "max_depth": 18, "min_samples_leaf": 2, "max_features": "sqrt"},
    {"n_estimators": 600, "max_depth": None, "min_samples_leaf": 3, "max_features": 0.7},
)

DEFAULT_LGBM_PARAM_CANDIDATES = (
    {"learning_rate": 0.045, "num_leaves": 63, "min_child_samples": 25, "subsample": 0.9},
    {"learning_rate": 0.06, "num_leaves": 95, "min_child_samples": 28, "subsample": 0.85},
    {"learning_rate": 0.08, "num_leaves": 127, "min_child_samples": 32, "subsample": 0.8},
)

MARKET_CONFIG: Dict[str, Dict[str, object]] = {
    "global": {
        "param_candidates": DEFAULT_PARAM_CANDIDATES,
        "class_weight_exponent": 0.5,
        "class_weight_clip": 5.0,
        "class_weight_multipliers": {"buy": 1.05, "hold": 0.92, "sell": 1.22},
        "bias_factors": {"buy": 1.06, "hold": 0.96, "sell": 1.15},
        "bias_prior_exponent": 0.4,
        "bias_scale_min": 0.75,
        "bias_scale_max": 1.35,
        "bias_score_weights": (0.35, 0.65),
        "bias_hold_recall_weight": 0.15,
        "bias_multipliers": [0.75, 0.9, 1.0, 1.1, 1.2],
        "rf_param_candidates": DEFAULT_RF_PARAM_CANDIDATES,
        "lgbm_param_candidates": DEFAULT_LGBM_PARAM_CANDIDATES,
    },
    "china_a": {
        "param_candidates": (
            {"learning_rate": 0.03, "max_leaf_nodes": 63, "min_samples_leaf": 22, "l2_regularization": 0.025},
            {"learning_rate": 0.04, "max_leaf_nodes": 95, "min_samples_leaf": 24, "l2_regularization": 0.015},
            {"learning_rate": 0.05, "max_leaf_nodes": 127, "min_samples_leaf": 28, "l2_regularization": 0.01},
            {"learning_rate": 0.06, "max_leaf_nodes": 159, "min_samples_leaf": 30, "l2_regularization": 0.007},
        ),
        "class_weight_exponent": 0.65,
        "class_weight_clip": 6.0,
        "class_weight_multipliers": {"buy": 1.02, "hold": 0.92, "sell": 1.28},
        "bias_factors": {"buy": 1.10, "hold": 0.96, "sell": 1.16},
        "bias_prior_exponent": 0.35,
        "bias_scale_min": 0.75,
        "bias_scale_max": 1.4,
        "bias_score_weights": (0.3, 0.7),
        "bias_hold_recall_weight": 0.1,
        "bias_multipliers": [0.75, 0.9, 1.0, 1.1, 1.25],
        "rf_param_candidates": (
            {"n_estimators": 400, "max_depth": None, "min_samples_leaf": 2, "max_features": "sqrt"},
            {"n_estimators": 500, "max_depth": 20, "min_samples_leaf": 2, "max_features": 0.65},
            {"n_estimators": 650, "max_depth": None, "min_samples_leaf": 3, "max_features": "sqrt"},
        ),
        "lgbm_param_candidates": DEFAULT_LGBM_PARAM_CANDIDATES,
    },
}


def _market_config(market: str) -> Dict[str, object]:
    return MARKET_CONFIG.get(market, MARKET_CONFIG["global"])


def _build_time_series_splits(
    n_samples: int,
    n_splits: int,
    *,
    min_train_size: int = 180,
    test_size: int = 60,
    step: Optional[int] = None,
) -> List[tuple[np.ndarray, np.ndarray]]:
    if n_samples <= 0 or n_splits <= 0:
        return []
    min_train_size = max(60, min_train_size)
    test_size = max(30, test_size)
    if n_samples < (min_train_size + test_size):
        return []
    step = step or test_size
    splits: List[tuple[np.ndarray, np.ndarray]] = []
    start = min_train_size
    total = np.arange(n_samples)
    while start + test_size <= n_samples and len(splits) < n_splits:
        train_idx = total[:start]
        test_idx = total[start : start + test_size]
        splits.append((train_idx, test_idx))
        start += step
    return splits


def _build_estimator(
    model_type: str,
    params: Dict[str, object],
    random_state: int,
    *,
    final: bool = False,
) -> ClassifierMixin:
    if model_type == "hist_gb":
        base_params = {
            "max_iter": 700 if final else 450,
            "max_depth": None,
            "random_state": random_state,
            "validation_fraction": None,
            "early_stopping": False,
        }
        base_params.update(params)
        return HistGradientBoostingClassifier(**base_params)
    if model_type == "random_forest":
        base_params = {
            "random_state": random_state,
            "n_jobs": -1,
        }
        base_params.update(params)
        return RandomForestClassifier(**base_params)
    if model_type == "lightgbm":
        if lgb is None:
            raise ValueError("LightGBM is not installed; please pip install lightgbm.")
        base_params = {
            "random_state": random_state,
            "n_estimators": 400 if final else 250,
            "n_jobs": -1,
        }
        base_params.update(params)
        return lgb.LGBMClassifier(**base_params)
    raise ValueError(f"Unsupported model_type: {model_type}")


class ProbabilityAveragingEnsemble(ClassifierMixin):
    """Simple probability averaging ensemble with optional weights."""

    def __init__(self, models: List[ClassifierMixin], weights: Iterable[float] | None = None) -> None:
        if not models:
            raise ValueError("Ensemble requires at least one model.")
        self.models = models
        weight_array = np.array(list(weights), dtype=float) if weights is not None else np.ones(len(models))
        if weight_array.shape[0] != len(models):
            raise ValueError("Weights length must match number of models.")
        weight_array = np.clip(weight_array, 1e-6, None)
        self.weights = weight_array / np.sum(weight_array)
        self.classes_ = getattr(models[0], "classes_", None)
        if self.classes_ is None:
            raise ValueError("Base models must expose classes_.")

    def predict_proba(self, X):
        probs = None
        for weight, model in zip(self.weights, self.models):
            model_probs = model.predict_proba(X)
            if probs is None:
                probs = weight * model_probs
            else:
                probs += weight * model_probs
        if probs is None:
            raise RuntimeError("Ensemble has no models.")
        row_sums = probs.sum(axis=1, keepdims=True)
        row_sums[row_sums == 0.0] = 1.0
        return probs / row_sums

    def predict(self, X):
        probs = self.predict_proba(X)
        indices = np.argmax(probs, axis=1)
        return np.asarray(self.classes_)[indices]


def get_market_bias_factors(market: str) -> Dict[str, float]:
    base_bias = {"buy": 1.0, "hold": 1.0, "sell": 1.0}
    config = _market_config(market)
    bias_factors_default = config.get("bias_factors")
    if isinstance(bias_factors_default, dict):
        base_bias.update({k: float(v) for k, v in bias_factors_default.items() if isinstance(v, (int, float))})
    return base_bias


def _optimize_bias_factors(
    probs: np.ndarray,
    y_true: pd.Series,
    classes: np.ndarray,
    *,
    base_bias: Dict[str, float],
    scale_min: float,
    scale_max: float,
    score_weights: tuple[float, float] = (0.6, 0.4),
    multipliers: Iterable[float] | None = None,
    hold_label: str = "hold",
    hold_recall_weight: float = 0.0,
) -> Dict[str, float]:
    if probs.shape[1] != len(classes):
        raise ValueError("Probability matrix shape does not match class labels.")

    labels = [str(cls) for cls in classes]
    base = {label: float(base_bias.get(label, 1.0)) for label in labels}
    multiplier_seq = list(multipliers) if multipliers is not None else [0.7, 0.85, 0.95, 1.0, 1.1, 1.25, 1.4]

    candidate_values: Dict[str, List[float]] = {}
    for label in labels:
        base_value = base[label]
        values = []
        for mult in multiplier_seq:
            value = base_value * mult
            value = max(scale_min, min(scale_max, value))
            values.append(round(value, 4))
        candidate_values[label] = sorted(set(values))

    y_true_arr = np.asarray(y_true)
    target_hold = str(hold_label)
    best_score = -np.inf
    best_bias = base.copy()
    weight_acc, weight_f1 = score_weights

    candidate_lists = [candidate_values[label] for label in labels]
    for combo in product(*candidate_lists):
        bias_map = dict(zip(labels, combo))
        scale_vector = np.array([bias_map[label] for label in labels])
        scaled_probs = probs * scale_vector
        preds = classes[np.argmax(scaled_probs, axis=1)]
        acc = accuracy_score(y_true_arr, preds)
        macro_f1 = f1_score(y_true_arr, preds, average="macro", zero_division=0)
        hold_recall = 0.0
        if target_hold in labels:
            hold_recall = recall_score(
                (y_true_arr == target_hold).astype(int),
                (preds == target_hold).astype(int),
                zero_division=0,
            )
        score = weight_acc * acc + weight_f1 * macro_f1 + hold_recall_weight * hold_recall
        if score > best_score:
            best_score = score
            best_bias = bias_map

    return best_bias


def _compute_class_weights(
    labels: pd.Series,
    *,
    exponent: float = 1.0,
    clip: float | None = None,
    multipliers: Dict[str, float] | None = None,
) -> dict:
    counts = labels.value_counts()
    total = len(labels)
    weights = {}
    for label, count in counts.items():
        if count == 0:
            continue
        base = total / (len(counts) * count)
        weight = base ** exponent
        if clip is not None:
            weight = min(weight, clip)
        if multipliers:
            weight *= float(multipliers.get(label, 1.0))
        weights[label] = weight
    return weights


def _evaluate_candidate_model(
    model_type: str,
    param_candidates: Iterable[Dict[str, object]],
    X_all: pd.DataFrame,
    y_all: pd.Series,
    cv_splits: List[tuple[np.ndarray, np.ndarray]],
    class_weight: Dict[str, float],
    random_state: int,
) -> Dict[str, object]:
    param_list = [dict(candidate) for candidate in param_candidates] or [{}]
    best_params = param_list[0]
    best_metrics: list[dict] = []
    best_score = float("-inf")

    if cv_splits:
        for params in param_list:
            candidate_metrics: list[dict] = []
            for fold, (train_idx, valid_idx) in enumerate(cv_splits, start=1):
                X_train, X_valid = X_all.iloc[train_idx], X_all.iloc[valid_idx]
                y_train, y_valid = y_all.iloc[train_idx], y_all.iloc[valid_idx]
                if y_train.nunique() < 2 or y_valid.nunique() < 2:
                    continue

                scaler_cv = StandardScaler()
                X_train_scaled = scaler_cv.fit_transform(X_train)
                X_valid_scaled = scaler_cv.transform(X_valid)

                sample_weight = y_train.map(class_weight).values
                estimator = _build_estimator(model_type, params, random_state, final=False)
                estimator.fit(X_train_scaled, y_train, sample_weight=sample_weight)
                y_pred = estimator.predict(X_valid_scaled)
                report = classification_report(y_valid, y_pred, zero_division=0, output_dict=True)
                candidate_metrics.append(
                    {
                        "fold": fold,
                        "macro_precision": report["macro avg"]["precision"],
                        "macro_recall": report["macro avg"]["recall"],
                        "macro_f1": report["macro avg"]["f1-score"],
                        "accuracy": report["accuracy"],
                        "hold_recall": report.get("hold", {}).get("recall", 0.0),
                    }
                )

            if candidate_metrics:
                avg_macro_f1 = np.mean([m["macro_f1"] for m in candidate_metrics])
                avg_accuracy = np.mean([m["accuracy"] for m in candidate_metrics])
                avg_macro_recall = np.mean([m["macro_recall"] for m in candidate_metrics])
                avg_hold_recall = np.mean([m.get("hold_recall", 0.0) for m in candidate_metrics])
                score = (avg_macro_f1 * 0.7) + (avg_accuracy * 0.2) + (avg_hold_recall * 0.1)
                if score > best_score:
                    best_score = score
                    best_params = params
                    best_metrics = candidate_metrics

    cv_summary = None
    if best_metrics:
        macro_precision = np.mean([m["macro_precision"] for m in best_metrics])
        macro_recall = np.mean([m["macro_recall"] for m in best_metrics])
        macro_f1 = np.mean([m["macro_f1"] for m in best_metrics])
        accuracy = np.mean([m["accuracy"] for m in best_metrics])
        hold_recall = np.mean([m.get("hold_recall", 0.0) for m in best_metrics])
        cv_summary = {
            "macro_precision": float(macro_precision),
            "macro_recall": float(macro_recall),
            "macro_f1": float(macro_f1),
            "accuracy": float(accuracy),
            "hold_recall": float(hold_recall),
        }

    return {
        "model_type": model_type,
        "best_params": best_params,
        "cv_summary": cv_summary,
        "score": best_score,
    }


def train_model(
    dataset: pd.DataFrame,
    test_size: float = 0.2,
    random_state: int = 42,
    *,
    market: str = "global",
    model_type: str = "auto",
) -> ModelArtifacts:
    dataset_sorted = (
        dataset.sort_values(["date", "ticker"])
        .reset_index(drop=True)
    )

    if "label_available" in dataset_sorted.columns:
        dataset_sorted = dataset_sorted[dataset_sorted["label_available"]]

    dataset_sorted = dataset_sorted.dropna(subset=FEATURE_COLUMNS + [LABEL])
    if dataset_sorted.empty:
        raise ValueError("数据量不足，无法训练模型。")

    X_all = dataset_sorted[FEATURE_COLUMNS]
    y_all = dataset_sorted[LABEL]

    config = _market_config(market)
    class_weight = _compute_class_weights(
        y_all,
        exponent=float(config.get("class_weight_exponent", 1.0)),
        clip=config.get("class_weight_clip"),
        multipliers=config.get("class_weight_multipliers"),
    )

    n_samples = len(dataset_sorted)
    max_splits = min(5, max(2, n_samples // 120))
    approx_test = max(45, int(n_samples * 0.15))
    min_train = max(120, approx_test * 2)
    cv_splits: List[tuple[np.ndarray, np.ndarray]] = _build_time_series_splits(
        n_samples,
        max_splits,
        min_train_size=min_train,
        test_size=approx_test,
    )
    if not cv_splits and max_splits >= 2:
        tscv = TimeSeriesSplit(n_splits=max_splits)
        cv_splits = list(tscv.split(dataset_sorted))

    candidate_results: List[Dict[str, object]] = []

    if model_type in {"auto", "hist_gb"}:
        hg_params = config.get("param_candidates", DEFAULT_PARAM_CANDIDATES)
        candidate_results.append(
            _evaluate_candidate_model(
                "hist_gb",
                hg_params,
                X_all,
                y_all,
                cv_splits,
                class_weight,
                random_state,
            )
        )

    if model_type in {"auto", "random_forest"}:
        rf_params = config.get("rf_param_candidates", DEFAULT_RF_PARAM_CANDIDATES)
        if rf_params:
            candidate_results.append(
                _evaluate_candidate_model(
                    "random_forest",
                    rf_params,
                    X_all,
                    y_all,
                    cv_splits,
                    class_weight,
                    random_state,
                )
            )

    if model_type in {"auto", "lightgbm"} and lgb is not None:
        lgb_params = config.get("lgbm_param_candidates", DEFAULT_LGBM_PARAM_CANDIDATES)
        if lgb_params:
            candidate_results.append(
                _evaluate_candidate_model(
                    "lightgbm",
                    lgb_params,
                    X_all,
                    y_all,
                    cv_splits,
                    class_weight,
                    random_state,
                )
            )

    if not candidate_results:
        raise ValueError("模型候选列表为空，请检查配置。")

    for result in candidate_results:
        summary = result.get("cv_summary")
        params = result["best_params"]
        model_label = result["model_type"]
        if summary:
            print(
                f"[{model_label}] TimeSeries CV (macro precision/recall/f1, accuracy): "
                f"{summary['macro_precision']:.3f} / {summary['macro_recall']:.3f} / "
                f"{summary['macro_f1']:.3f}, {summary['accuracy']:.3f}"
            )
            print(
                f"[{model_label}] Selected params: {params} "
                f"(macro_f1 {summary['macro_f1']:.3f}, accuracy {summary['accuracy']:.3f})"
            )
        else:
            print(f"[{model_label}] 无法执行有效的时间序列 CV，采用默认参数: {params}")

    candidate_results_sorted = sorted(
        candidate_results,
        key=lambda r: r.get("score", float("-inf")),
        reverse=True,
    )
    primary = candidate_results_sorted[0]
    secondary = candidate_results_sorted[1] if len(candidate_results_sorted) > 1 else None

    use_ensemble = False
    ensemble_members: List[Dict[str, object]] | None = None

    if model_type == "auto" and secondary:
        primary_score = primary.get("score", float("-inf"))
        secondary_score = secondary.get("score", float("-inf"))
        if (
            np.isfinite(primary_score)
            and np.isfinite(secondary_score)
            and primary.get("cv_summary")
            and secondary.get("cv_summary")
            and (primary_score - secondary_score) <= 0.015
        ):
            use_ensemble = True
            ensemble_members = [
                {
                    "model_type": primary["model_type"],
                    "params": primary["best_params"],
                    "score": float(primary_score),
                },
                {
                    "model_type": secondary["model_type"],
                    "params": secondary["best_params"],
                    "score": float(secondary_score),
                },
            ]

    if use_ensemble and ensemble_members:
        selected_type = "ensemble"
        best_params = {"members": ensemble_members}
        weights = np.array([member["score"] for member in ensemble_members], dtype=float)
        weights = np.clip(weights, 0.0, None)
        if not np.isfinite(weights).any() or weights.sum() == 0.0:
            weights = np.ones(len(ensemble_members), dtype=float)
        weights = weights / weights.sum()

        summaries = [primary.get("cv_summary"), secondary.get("cv_summary")]
        metric_keys = ("macro_precision", "macro_recall", "macro_f1", "accuracy", "hold_recall")
        cv_metrics_summary: Dict[str, float] | None = {}
        for key in metric_keys:
            values = []
            for summary, weight in zip(summaries, weights):
                if summary and summary.get(key) is not None:
                    values.append(weight * float(summary[key]))
            if values:
                cv_metrics_summary[key] = float(np.sum(values))
        if not cv_metrics_summary:
            cv_metrics_summary = None
        print(
            "[model] Using ensemble blend of "
            f"{ensemble_members[0]['model_type']} + {ensemble_members[1]['model_type']} "
            f"with weights {weights.round(3).tolist()}"
        )
        for member, weight in zip(ensemble_members, weights):
            print(
                f"  -> {member['model_type']} params {member['params']} (blend weight {weight:.2f})"
            )
            member["weight"] = float(weight)
        ensemble_weights = weights
    else:
        if model_type == "auto":
            selected = primary if np.isfinite(primary.get("score", float("-inf"))) else candidate_results[0]
        else:
            selected = next((r for r in candidate_results if r["model_type"] == model_type), None)
            if selected is None:
                raise ValueError(f"模型类型 {model_type} 未评估或不可用。")
        selected_type = selected["model_type"]
        best_params = selected["best_params"]
        cv_metrics_summary = selected.get("cv_summary")
        print(f"[model] Using {selected_type} with params {best_params}")
        ensemble_weights = None

    ensemble_members_for_artifacts = ensemble_members if use_ensemble and ensemble_members else None

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_all)
    meta_conf_series = dataset_sorted.get("meta_signal_confidence")
    meta_active_series = dataset_sorted.get("meta_signal_active")
    sample_weight_all = y_all.map(class_weight).astype(float)
    if meta_conf_series is not None:
        sample_weight_all = sample_weight_all * (1.0 + meta_conf_series.clip(0.0, 3.0) * 0.2)
    if meta_active_series is not None:
        sample_weight_all = sample_weight_all * (1.0 + meta_active_series.clip(0, 1) * 0.1)
    sample_weight_all = sample_weight_all.values
    class_priors = (
        y_all.value_counts(normalize=True)
        .reindex(sorted(y_all.unique()), fill_value=0.0)
        .to_dict()
    )
    bias_factors_base = get_market_bias_factors(market)
    priors_nonzero = {k: v for k, v in class_priors.items() if v and v > 0}
    if priors_nonzero:
        mean_prior = sum(priors_nonzero.values()) / len(priors_nonzero)
        prior_exponent = float(config.get("bias_prior_exponent", 0.0))
        bias_scale_min = float(config.get("bias_scale_min", 0.7))
        bias_scale_max = float(config.get("bias_scale_max", 1.5))
        if prior_exponent > 0 and mean_prior > 0:
            for label, prior in priors_nonzero.items():
                base = bias_factors_base.get(label, 1.0)
                scale = (mean_prior / prior) ** prior_exponent
                adjusted = base * scale
                adjusted = max(bias_scale_min, min(bias_scale_max, adjusted))
                bias_factors_base[label] = adjusted

    optimized_bias_oof: Dict[str, float] | None = None
    if selected_type != "ensemble" and cv_splits and cv_metrics_summary:
        oof_probs: List[np.ndarray] = []
        oof_true: List[np.ndarray] = []
        oof_preds: List[np.ndarray] = []
        class_order = sorted(y_all.unique())
        for train_idx, valid_idx in cv_splits:
            X_train, X_valid = X_all.iloc[train_idx], X_all.iloc[valid_idx]
            y_train, y_valid = y_all.iloc[train_idx], y_all.iloc[valid_idx]
            if y_train.nunique() < 2 or y_valid.nunique() < 2:
                continue

            scaler_cv = StandardScaler()
            X_train_scaled = scaler_cv.fit_transform(X_train)
            X_valid_scaled = scaler_cv.transform(X_valid)

            sample_weight_train = y_train.map(class_weight).astype(float)
            if meta_conf_series is not None:
                sample_weight_train = sample_weight_train * (
                    1.0 + meta_conf_series.iloc[train_idx].clip(0.0, 3.0) * 0.2
                )
            if meta_active_series is not None:
                sample_weight_train = sample_weight_train * (
                    1.0 + meta_active_series.iloc[train_idx].clip(0, 1) * 0.1
                )
            sample_weight_train = sample_weight_train.values
            base_estimator = _build_estimator(selected_type, best_params, random_state, final=True)
            calibrated_cv = CalibratedClassifierCV(base_estimator, method="sigmoid", cv=3)
            calibrated_cv.fit(X_train_scaled, y_train, sample_weight=sample_weight_train)

            probs_valid = calibrated_cv.predict_proba(X_valid_scaled)
            classes_fold = list(calibrated_cv.classes_)
            aligned = np.zeros((probs_valid.shape[0], len(class_order)))
            for idx_cls, label in enumerate(classes_fold):
                if label in class_order:
                    aligned[:, class_order.index(label)] = probs_valid[:, idx_cls]
            preds_valid = np.array(class_order)[np.argmax(aligned, axis=1)]

            oof_probs.append(aligned)
            oof_true.append(y_valid.to_numpy())
            oof_preds.append(preds_valid)

        if oof_probs:
            probs_concat = np.vstack(oof_probs)
            y_true_concat = np.concatenate(oof_true)
            y_pred_concat = np.concatenate(oof_preds)
            report_oof = classification_report(
                y_true_concat,
                y_pred_concat,
                zero_division=0,
                output_dict=True,
            )
            cv_metrics_summary = {
                "oof_accuracy": float(report_oof.get("accuracy", 0.0)),
                "oof_macro_precision": float(report_oof.get("macro avg", {}).get("precision", 0.0)),
                "oof_macro_recall": float(report_oof.get("macro avg", {}).get("recall", 0.0)),
                "oof_macro_f1": float(report_oof.get("macro avg", {}).get("f1-score", 0.0)),
            }
            print(
                "OOF metrics (accuracy / macro_f1): "
                f"{cv_metrics_summary['oof_accuracy']:.3f} / {cv_metrics_summary['oof_macro_f1']:.3f}"
            )

            score_weights = config.get("bias_score_weights", (0.6, 0.4))
            if not isinstance(score_weights, (list, tuple)):
                score_weights = (0.6, 0.4)
            multipliers = config.get("bias_multipliers")
            if multipliers is not None and not isinstance(multipliers, (list, tuple)):
                multipliers = None
            optimized_bias_oof = _optimize_bias_factors(
                probs_concat,
                pd.Series(y_true_concat),
                np.array(class_order),
                base_bias=bias_factors_base,
                scale_min=float(config.get("bias_scale_min", 0.7)),
                scale_max=float(config.get("bias_scale_max", 1.5)),
                score_weights=tuple(score_weights),
                multipliers=multipliers,
                hold_label="hold",
                hold_recall_weight=float(config.get("bias_hold_recall_weight", 0.0)),
            )
            print(f"Optimized bias factors (OOF, {market}): {optimized_bias_oof}")

    if selected_type == "ensemble" and ensemble_members_for_artifacts:
        base_models: List[ClassifierMixin] = []
        weights_for_models = (
            np.array(ensemble_weights, dtype=float)
            if ensemble_weights is not None
            else np.ones(len(ensemble_members_for_artifacts), dtype=float)
        )
        if weights_for_models.sum() <= 0:
            weights_for_models = np.ones(len(ensemble_members_for_artifacts), dtype=float)
        for idx, member in enumerate(ensemble_members_for_artifacts):
            member_type = str(member.get("model_type"))
            member_params = dict(member.get("params", {}))
            member_estimator = _build_estimator(
                member_type,
                member_params,
                random_state + (idx * 11),
                final=True,
            )
            member_estimator.fit(X_scaled, y_all, sample_weight=sample_weight_all)
            if hasattr(member_estimator, "feature_importances_"):
                importances = member_estimator.feature_importances_
                top_idx = np.argsort(importances)[::-1][:8]
                top_features = [f"{FEATURE_COLUMNS[i]}: {importances[i]:.1f}" for i in top_idx]
                print(
                    f"Feature importances ({member_type}): " + ", ".join(top_features)
                )
            calibrator_member = CalibratedClassifierCV(member_estimator, method="sigmoid", cv=3)
            calibrator_member.fit(X_scaled, y_all, sample_weight=sample_weight_all)
            base_models.append(calibrator_member)
        calibrated_model: ClassifierMixin = ProbabilityAveragingEnsemble(base_models, weights_for_models)
    else:
        base_model = _build_estimator(selected_type, best_params, random_state, final=True)
        base_model.fit(X_scaled, y_all, sample_weight=sample_weight_all)

        if hasattr(base_model, "feature_importances_"):
            importances = base_model.feature_importances_
            top_idx = np.argsort(importances)[::-1][:10]
            top_features = [f"{FEATURE_COLUMNS[i]}: {importances[i]:.1f}" for i in top_idx]
            print("Top feature importances:", ", ".join(top_features))

        calibrated_model = CalibratedClassifierCV(base_model, method="sigmoid", cv=3)
        calibrated_model.fit(X_scaled, y_all, sample_weight=sample_weight_all)

    bias_factors = bias_factors_base
    if optimized_bias_oof is not None:
        bias_factors = optimized_bias_oof
    else:
        try:
            probs_train = calibrated_model.predict_proba(X_scaled)
            score_weights = config.get("bias_score_weights", (0.6, 0.4))
            if not isinstance(score_weights, (list, tuple)):
                score_weights = (0.6, 0.4)
            multipliers = config.get("bias_multipliers")
            if multipliers is not None and not isinstance(multipliers, (list, tuple)):
                multipliers = None
            hold_recall_weight = float(config.get("bias_hold_recall_weight", 0.0))
            optimized_bias = _optimize_bias_factors(
                probs_train,
                y_all,
                calibrated_model.classes_,
                base_bias=bias_factors_base,
                scale_min=float(config.get("bias_scale_min", 0.7)),
                scale_max=float(config.get("bias_scale_max", 1.5)),
                score_weights=tuple(score_weights),
                multipliers=multipliers,
                hold_label="hold",
                hold_recall_weight=hold_recall_weight,
            )
            bias_factors = optimized_bias
            print(f"Optimized bias factors ({market}): {optimized_bias}")
        except Exception as exc:
            print(f"Bias optimization skipped ({market}): {exc}")

    meta_scaler = None
    meta_model = None
    meta_feature_specs: List[Dict[str, str]] | None = None
    meta_target = dataset_sorted.get("meta_signal_active")
    if meta_target is not None:
        meta_target = meta_target.fillna(0).astype(int)
        positive_count = int(meta_target.sum())
        if positive_count >= max(12, int(len(meta_target) * 0.015)):
            try:
                class_index_map = {str(cls): idx for idx, cls in enumerate(calibrated_model.classes_)}
                base_probs_train = calibrated_model.predict_proba(X_scaled)
                meta_features: Dict[str, np.ndarray] = {}
                meta_feature_specs = []
                for label, idx in class_index_map.items():
                    col_name = f"meta_prob_{label}"
                    meta_features[col_name] = base_probs_train[:, idx]
                    meta_feature_specs.append({"kind": "prob", "label": str(label), "name": col_name})

                candidate_cols = [
                    "meta_signal_confidence",
                    "regime_score",
                    "regime_volatility",
                    "regime_trend",
                    "macro_liquidity_spread",
                    "macro_risk_spread",
                    "macro_vix_to_us10y",
                    "trend_alignment_score",
                    "macro_region_csi300_zscore_60d",
                    "macro_region_csi300_pct_change_5d",
                    "macro_region_shanghai_zscore_60d",
                    "macro_region_cnh_zscore_60d",
                    "macro_region_hang_seng_zscore_60d",
                    "macro_region_nasdaq_zscore_60d",
                    "macro_region_russell_zscore_60d",
                    "macro_region_sp500_zscore_60d",
                    "fundamental_pe_ratio",
                    "fundamental_price_to_book",
                    "fundamental_beta",
                    "return_1d",
                    "return_5d",
                    "return_10d",
                    "return_20d",
                    "return_60d",
                    "volatility_20d",
                    "volatility_60d",
                    "atr_ratio",
                    "trend_strength_20d",
                    "trend_strength_60d",
                    "volume_zscore",
                    "relative_return_20d",
                    "relative_return_60d",
                    "relative_strength_pct",
                    "macro_vix_level",
                    "macro_dxy_level",
                    "macro_us10y_level",
                ]
                for column in candidate_cols:
                    if column in dataset_sorted.columns:
                        col_name = f"meta_col_{column}"
                        meta_features[col_name] = dataset_sorted[column].fillna(0.0).to_numpy()
                        meta_feature_specs.append({"kind": "column", "column": column, "name": col_name})

                meta_features["meta_prob_gap"] = _prob_gap_from_matrix(base_probs_train)
                meta_feature_specs.append({"kind": "prob_gap", "name": "meta_prob_gap"})
                meta_features["meta_prob_max"] = _prob_max_from_matrix(base_probs_train)
                meta_feature_specs.append({"kind": "prob_max", "name": "meta_prob_max"})

                meta_df = pd.DataFrame(meta_features, index=dataset_sorted.index)
                meta_df = meta_df.replace([np.inf, -np.inf], 0.0).fillna(0.0)
                meta_scaler = StandardScaler()
                meta_X = meta_scaler.fit_transform(meta_df)
                meta_y = meta_target.values
                sample_weight_meta = np.ones_like(meta_y, dtype=float)
                if meta_conf_series is not None:
                    sample_weight_meta = sample_weight_meta * (1.0 + meta_conf_series.clip(0.0, 3.0) * 0.1)
                if meta_active_series is not None:
                    sample_weight_meta = sample_weight_meta * (1.0 + meta_active_series.clip(0, 1) * 0.05)

                meta_estimator = HistGradientBoostingClassifier(
                    max_depth=3,
                    learning_rate=0.08,
                    l2_regularization=0.1,
                    max_iter=300,
                    min_samples_leaf=20,
                    random_state=random_state,
                )
                meta_calibrated = CalibratedClassifierCV(meta_estimator, method="sigmoid", cv=3)
                meta_calibrated.fit(meta_X, meta_y, sample_weight=sample_weight_meta)
                meta_model = meta_calibrated
                meta_pred = meta_calibrated.predict(meta_X)
                meta_report = classification_report(
                    meta_y,
                    meta_pred,
                    output_dict=True,
                    zero_division=0,
                )
                print(
                    f"[meta] Trained meta-signal model on {len(meta_y)} samples"
                    f" (positive={positive_count})"
                )
                print(
                    "[meta] accuracy "
                    f"{meta_report.get('accuracy', 0.0):.3f} | macro_f1 "
                    f"{meta_report.get('macro avg', {}).get('f1-score', 0.0):.3f}"
                )
            except Exception as exc:
                meta_scaler = None
                meta_model = None
                meta_feature_specs = None
                print(f"[meta] Meta model training skipped: {exc}")
        else:
            print(f"[meta] Positive meta labels不足（{positive_count}），跳过训练 meta 模型")

    return ModelArtifacts(
        scaler=scaler,
        model=calibrated_model,
        class_priors=class_priors,
        market=market,
        bias_factors=bias_factors,
        cv_metrics=cv_metrics_summary,
        model_type=selected_type,
        model_params=best_params,
        ensemble_members=ensemble_members_for_artifacts,
        meta_scaler=meta_scaler,
        meta_model=meta_model,
        meta_feature_specs=meta_feature_specs,
    )

def predict_signals(dataset: pd.DataFrame, artifacts: ModelArtifacts) -> pd.DataFrame:
    X = dataset[FEATURE_COLUMNS]
    X_scaled = artifacts.scaler.transform(X)
    probs = artifacts.model.predict_proba(X_scaled)
    class_index = {cls: idx for idx, cls in enumerate(artifacts.model.classes_)}

    market = getattr(artifacts, "market", "global") or "global"
    base_bias = get_market_bias_factors(market)
    if isinstance(artifacts.bias_factors, dict):
        base_bias.update({k: float(v) for k, v in artifacts.bias_factors.items() if isinstance(v, (int, float))})
    bias_factors = base_bias
    adjusted = probs.copy()
    for label, factor in bias_factors.items():
        idx = class_index.get(label)
        if idx is not None:
            adjusted[:, idx] = adjusted[:, idx] * factor
    preds = artifacts.model.classes_[np.argmax(adjusted, axis=1)]

    result = dataset.copy()
    result["model_signal"] = preds
    result["prob_buy"] = probs[:, class_index.get("buy", 0)]
    result["prob_sell"] = probs[:, class_index.get("sell", 0)]
    result["prob_hold"] = probs[:, class_index.get("hold", 0)]
    if (
        artifacts.meta_model is not None
        and artifacts.meta_scaler is not None
        and artifacts.meta_feature_specs
    ):
        meta_feature_data: Dict[str, np.ndarray] = {}
        for spec in artifacts.meta_feature_specs:
            name = spec.get("name")
            if not name:
                continue
            kind = spec.get("kind")
            if kind == "prob":
                label = spec.get("label")
                idx = class_index.get(label) if label is not None else None
                meta_feature_data[name] = probs[:, idx] if idx is not None else np.zeros(len(dataset))
            elif kind == "column":
                column = spec.get("column")
                if column and column in dataset.columns:
                    meta_feature_data[name] = dataset[column].to_numpy()
                else:
                    meta_feature_data[name] = np.zeros(len(dataset))
            elif kind == "prob_gap":
                meta_feature_data[name] = _prob_gap_from_matrix(probs)
            elif kind == "prob_max":
                meta_feature_data[name] = _prob_max_from_matrix(probs)
            else:
                meta_feature_data[name] = np.zeros(len(dataset))
        if meta_feature_data:
            meta_df = pd.DataFrame(meta_feature_data, index=dataset.index).fillna(0.0)
            meta_matrix = artifacts.meta_scaler.transform(meta_df)
            try:
                meta_probs = artifacts.meta_model.predict_proba(meta_matrix)[:, 1]
            except AttributeError:
                meta_probs = artifacts.meta_model.predict(meta_matrix)
            result["meta_signal_prob"] = meta_probs
            result["meta_signal_prediction"] = (meta_probs >= 0.45).astype(int)
    for extra_col in (
        "meta_signal_active",
        "meta_signal_confidence",
        "meta_signal_magnitude",
        "meta_signal_direction",
    ):
        if extra_col in dataset.columns:
            result[extra_col] = dataset[extra_col]
    return result


def load_artifacts(name: str) -> ModelArtifacts:
    path = MODEL_DIR / f"{name}.joblib"
    if not path.exists():
        raise FileNotFoundError(f"Model artifacts not found at {path}")
    artifacts = ModelArtifacts.load(path)
    if not hasattr(artifacts, "class_priors"):
        artifacts.class_priors = None
    if not hasattr(artifacts, "market") or not artifacts.market:
        artifacts.market = "global"
    if not isinstance(getattr(artifacts, "bias_factors", None), dict):
        artifacts.bias_factors = get_market_bias_factors(artifacts.market)
    if not hasattr(artifacts, "cv_metrics"):
        artifacts.cv_metrics = None
    if not hasattr(artifacts, "model_type") or not getattr(artifacts, "model_type", None):
        artifacts.model_type = "hist_gb"
    if not hasattr(artifacts, "model_params"):
        artifacts.model_params = None
    if not hasattr(artifacts, "ensemble_members"):
        artifacts.ensemble_members = None
    return artifacts
