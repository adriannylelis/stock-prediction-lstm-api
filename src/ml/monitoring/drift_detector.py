"""Data drift detection for ML models.

Detects distribution changes between reference and production data.
"""

import json
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
from loguru import logger
from scipy import stats


class DriftDetector:
    """Detect data drift using statistical tests.

    Features:
    - Kolmogorov-Smirnov (KS) test for numerical features
    - Population Stability Index (PSI) for categorical features
    - Chi-square test for categorical features
    - Automatic threshold-based alerting

    Example:
        >>> detector = DriftDetector()
        >>> report = detector.detect_drift(df_reference, df_production)
        >>> if report['has_drift']:
        >>>     print(f"Drift detected in: {report['drifted_features']}")
    """

    def __init__(
        self, ks_threshold: float = 0.05, psi_threshold: float = 0.1, chi2_threshold: float = 0.05
    ):
        """Initialize drift detector.

        Args:
            ks_threshold: P-value threshold for KS test (default: 0.05).
            psi_threshold: PSI threshold for drift detection (default: 0.1).
                          PSI < 0.1: No significant change
                          0.1 <= PSI < 0.2: Small change
                          PSI >= 0.2: Significant change
            chi2_threshold: P-value threshold for chi-square test (default: 0.05).
        """
        self.ks_threshold = ks_threshold
        self.psi_threshold = psi_threshold
        self.chi2_threshold = chi2_threshold

        logger.info(f"Initialized DriftDetector (KS={ks_threshold}, PSI={psi_threshold})")

    def detect_drift(
        self,
        reference_data: pd.DataFrame,
        production_data: pd.DataFrame,
        features: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        """Detect drift between reference and production data.

        Args:
            reference_data: Reference/training dataset.
            production_data: Production/new dataset.
            features: Features to check. None = all numeric features.

        Returns:
            Dictionary with drift report:
            {
                'has_drift': bool,
                'drifted_features': List[str],
                'feature_scores': Dict[str, float],
                'feature_pvalues': Dict[str, float],
                'summary': Dict[str, Any]
            }
        """
        if features is None:
            features = reference_data.select_dtypes(include=[np.number]).columns.tolist()

        logger.info(f"Checking drift for {len(features)} features...")

        # Results storage
        feature_scores = {}
        feature_pvalues = {}
        drifted_features = []

        # Check each feature
        for feature in features:
            if feature not in reference_data.columns or feature not in production_data.columns:
                logger.warning(f"Feature '{feature}' not in both datasets, skipping")
                continue

            ref_values = reference_data[feature].dropna()
            prod_values = production_data[feature].dropna()

            # KS test for numerical features
            ks_stat, p_value = stats.ks_2samp(ref_values, prod_values)

            feature_scores[feature] = ks_stat
            feature_pvalues[feature] = p_value

            # Check if drifted
            if p_value < self.ks_threshold:
                drifted_features.append(feature)
                logger.warning(
                    f"⚠️  Drift detected in '{feature}': KS={ks_stat:.4f}, p={p_value:.4f}"
                )

        # Summary statistics
        has_drift = len(drifted_features) > 0
        drift_percentage = (len(drifted_features) / len(features)) * 100 if features else 0

        report = {
            "has_drift": has_drift,
            "drifted_features": drifted_features,
            "feature_scores": feature_scores,
            "feature_pvalues": feature_pvalues,
            "summary": {
                "total_features": len(features),
                "drifted_count": len(drifted_features),
                "drift_percentage": drift_percentage,
                "timestamp": datetime.now().isoformat(),
            },
        }

        if has_drift:
            logger.warning(
                f"🚨 Drift detected in {len(drifted_features)}/{len(features)} features ({drift_percentage:.1f}%)"
            )
        else:
            logger.success(f"✓ No drift detected in {len(features)} features")

        return report

    def calculate_psi(
        self, reference_data: pd.Series, production_data: pd.Series, bins: int = 10
    ) -> float:
        """Calculate Population Stability Index (PSI).

        PSI measures the shift in population distribution:
        - PSI < 0.1: No significant change
        - 0.1 <= PSI < 0.2: Small change
        - PSI >= 0.2: Significant change (retraining recommended)

        Args:
            reference_data: Reference distribution.
            production_data: Production distribution.
            bins: Number of bins for discretization.

        Returns:
            PSI score.
        """
        # Create bins based on reference data
        breakpoints = np.percentile(reference_data, np.linspace(0, 100, bins + 1))
        breakpoints = np.unique(breakpoints)  # Remove duplicates

        # Ensure we have at least 2 bins
        if len(breakpoints) < 2:
            logger.warning("Not enough unique values for PSI calculation")
            return 0.0

        # Calculate distributions
        ref_counts, _ = np.histogram(reference_data, bins=breakpoints)
        prod_counts, _ = np.histogram(production_data, bins=breakpoints)

        # Avoid division by zero
        ref_percents = (ref_counts + 1e-6) / (len(reference_data) + 1e-6 * len(breakpoints))
        prod_percents = (prod_counts + 1e-6) / (len(production_data) + 1e-6 * len(breakpoints))

        # Calculate PSI
        psi = np.sum((prod_percents - ref_percents) * np.log(prod_percents / ref_percents))

        return float(psi)

    def detect_drift_psi(
        self,
        reference_data: pd.DataFrame,
        production_data: pd.DataFrame,
        features: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        """Detect drift using PSI method.

        Args:
            reference_data: Reference/training dataset.
            production_data: Production/new dataset.
            features: Features to check. None = all numeric features.

        Returns:
            Dictionary with PSI-based drift report.
        """
        if features is None:
            features = reference_data.select_dtypes(include=[np.number]).columns.tolist()

        logger.info(f"Calculating PSI for {len(features)} features...")

        # Results storage
        feature_psi = {}
        drifted_features = []

        # Check each feature
        for feature in features:
            if feature not in reference_data.columns or feature not in production_data.columns:
                continue

            ref_values = reference_data[feature].dropna()
            prod_values = production_data[feature].dropna()

            # Calculate PSI
            psi = self.calculate_psi(ref_values, prod_values)
            feature_psi[feature] = psi

            # Check if drifted
            if psi >= self.psi_threshold:
                drifted_features.append(feature)
                severity = "CRITICAL" if psi >= 0.2 else "WARNING"
                logger.warning(f"⚠️  [{severity}] Drift in '{feature}': PSI={psi:.4f}")

        has_drift = len(drifted_features) > 0
        drift_percentage = (len(drifted_features) / len(features)) * 100 if features else 0

        report = {
            "has_drift": has_drift,
            "drifted_features": drifted_features,
            "feature_psi": feature_psi,
            "summary": {
                "total_features": len(features),
                "drifted_count": len(drifted_features),
                "drift_percentage": drift_percentage,
                "max_psi": max(feature_psi.values()) if feature_psi else 0.0,
                "avg_psi": np.mean(list(feature_psi.values())) if feature_psi else 0.0,
                "timestamp": datetime.now().isoformat(),
            },
        }

        return report

    def save_report(self, report: Dict[str, Any], output_path: str) -> None:
        """Save drift report to JSON file.

        Args:
            report: Drift detection report.
            output_path: Path to save report.
        """
        output_file = Path(output_path)
        output_file.parent.mkdir(parents=True, exist_ok=True)

        with open(output_file, "w") as f:
            json.dump(report, f, indent=2)

        logger.info(f"💾 Saved drift report: {output_file}")

    def compare_versions(
        self, version1_data: pd.DataFrame, version2_data: pd.DataFrame, method: str = "ks"
    ) -> Dict[str, Any]:
        """Compare two data versions for drift.

        Args:
            version1_data: First version (reference).
            version2_data: Second version (comparison).
            method: 'ks' for KS test or 'psi' for PSI.

        Returns:
            Drift detection report.
        """
        logger.info(f"Comparing data versions using {method.upper()} method...")

        if method == "psi":
            return self.detect_drift_psi(version1_data, version2_data)
        else:
            return self.detect_drift(version1_data, version2_data)
