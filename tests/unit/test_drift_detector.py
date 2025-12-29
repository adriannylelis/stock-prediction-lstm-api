"""Unit tests for drift detector."""

import json
import shutil
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.ml.monitoring.drift_detector import DriftDetector


@pytest.fixture
def reference_data():
    """Create reference dataset."""
    np.random.seed(42)
    return pd.DataFrame(
        {
            "feature1": np.random.normal(0, 1, 1000),
            "feature2": np.random.normal(10, 2, 1000),
            "feature3": np.random.uniform(0, 100, 1000),
        }
    )


@pytest.fixture
def production_data_no_drift():
    """Create production dataset without drift."""
    np.random.seed(123)
    return pd.DataFrame(
        {
            "feature1": np.random.normal(0, 1, 1000),
            "feature2": np.random.normal(10, 2, 1000),
            "feature3": np.random.uniform(0, 100, 1000),
        }
    )


@pytest.fixture
def production_data_with_drift():
    """Create production dataset with drift."""
    np.random.seed(456)
    return pd.DataFrame(
        {
            "feature1": np.random.normal(2, 1, 1000),  # Mean shifted
            "feature2": np.random.normal(10, 2, 1000),  # No drift
            "feature3": np.random.uniform(50, 150, 1000),  # Range shifted
        }
    )


@pytest.fixture
def temp_report_dir(tmp_path):
    """Create temporary directory for reports."""
    report_dir = tmp_path / "drift_reports"
    report_dir.mkdir()
    yield str(report_dir)
    if report_dir.exists():
        shutil.rmtree(report_dir)


class TestDriftDetector:
    """Test DriftDetector class."""

    def test_init(self):
        """Test initialization."""
        detector = DriftDetector(ks_threshold=0.05, psi_threshold=0.1, chi2_threshold=0.05)
        assert detector.ks_threshold == 0.05
        assert detector.psi_threshold == 0.1
        assert detector.chi2_threshold == 0.05

    def test_init_defaults(self):
        """Test initialization with defaults."""
        detector = DriftDetector()
        assert detector.ks_threshold == 0.05
        assert detector.psi_threshold == 0.1

    def test_detect_no_drift(self, reference_data, production_data_no_drift):
        """Test detection when no drift present."""
        detector = DriftDetector(ks_threshold=0.05)

        report = detector.detect_drift(reference_data, production_data_no_drift)

        # Should not detect drift
        assert report["has_drift"] is False
        assert len(report["drifted_features"]) == 0
        assert "feature_scores" in report
        assert "feature_pvalues" in report

    def test_detect_with_drift(self, reference_data, production_data_with_drift):
        """Test detection when drift present."""
        detector = DriftDetector(ks_threshold=0.05)

        report = detector.detect_drift(reference_data, production_data_with_drift)

        # Should detect drift
        assert report["has_drift"] is True
        assert len(report["drifted_features"]) > 0
        assert "feature1" in report["drifted_features"]
        assert "feature3" in report["drifted_features"]

    def test_detect_drift_specific_features(self, reference_data, production_data_with_drift):
        """Test detection on specific features only."""
        detector = DriftDetector()

        report = detector.detect_drift(
            reference_data, production_data_with_drift, features=["feature1"]
        )

        # Should only check feature1
        assert len(report["feature_scores"]) == 1
        assert "feature1" in report["feature_scores"]

    def test_psi_calculation(self):
        """Test PSI calculation."""
        detector = DriftDetector()

        # Create simple distributions
        ref = pd.Series(np.random.normal(0, 1, 1000))
        prod_no_drift = pd.Series(np.random.normal(0, 1, 1000))
        prod_with_drift = pd.Series(np.random.normal(2, 1, 1000))

        # PSI without drift should be low
        psi_no_drift = detector.calculate_psi(ref, prod_no_drift)
        assert psi_no_drift < 0.1

        # PSI with drift should be high
        psi_with_drift = detector.calculate_psi(ref, prod_with_drift)
        assert psi_with_drift > 0.1

    def test_detect_drift_psi(self, reference_data, production_data_with_drift):
        """Test drift detection using PSI."""
        detector = DriftDetector(psi_threshold=0.1)

        report = detector.detect_drift_psi(reference_data, production_data_with_drift)

        # Verify report structure
        assert "has_drift" in report
        assert "drifted_features" in report
        assert "feature_psi" in report
        assert "summary" in report

        # Should detect drift
        assert report["has_drift"] is True
        assert len(report["drifted_features"]) > 0

    def test_psi_report_summary(self, reference_data, production_data_with_drift):
        """Test PSI report summary statistics."""
        detector = DriftDetector()

        report = detector.detect_drift_psi(reference_data, production_data_with_drift)

        summary = report["summary"]
        assert "total_features" in summary
        assert "drifted_count" in summary
        assert "drift_percentage" in summary
        assert "max_psi" in summary
        assert "avg_psi" in summary
        assert summary["total_features"] == 3

    def test_save_report(self, reference_data, production_data_with_drift, temp_report_dir):
        """Test saving drift report."""
        detector = DriftDetector()

        report = detector.detect_drift(reference_data, production_data_with_drift)

        report_path = Path(temp_report_dir) / "test_report.json"
        detector.save_report(report, str(report_path))

        # Check file was created
        assert report_path.exists()

        # Verify content
        with open(report_path) as f:
            loaded_report = json.load(f)

        assert loaded_report["has_drift"] == report["has_drift"]
        assert loaded_report["drifted_features"] == report["drifted_features"]

    def test_compare_versions_ks(self, reference_data, production_data_with_drift):
        """Test comparing versions using KS method."""
        detector = DriftDetector()

        report = detector.compare_versions(reference_data, production_data_with_drift, method="ks")

        assert report["has_drift"] is True
        assert "feature_pvalues" in report

    def test_compare_versions_psi(self, reference_data, production_data_with_drift):
        """Test comparing versions using PSI method."""
        detector = DriftDetector()

        report = detector.compare_versions(reference_data, production_data_with_drift, method="psi")

        assert report["has_drift"] is True
        assert "feature_psi" in report

    def test_drift_percentage_calculation(self, reference_data, production_data_with_drift):
        """Test drift percentage calculation."""
        detector = DriftDetector()

        report = detector.detect_drift(reference_data, production_data_with_drift)

        total_features = report["summary"]["total_features"]
        drifted_count = report["summary"]["drifted_count"]
        drift_percentage = report["summary"]["drift_percentage"]

        expected_percentage = (drifted_count / total_features) * 100
        assert drift_percentage == pytest.approx(expected_percentage)

    def test_missing_features_warning(self, reference_data):
        """Test handling of missing features."""
        detector = DriftDetector()

        # Production data missing a feature
        prod_missing = pd.DataFrame(
            {
                "feature1": np.random.normal(0, 1, 1000),
                "feature2": np.random.normal(10, 2, 1000),
                # feature3 missing
            }
        )

        # Should not crash, just warn
        report = detector.detect_drift(
            reference_data, prod_missing, features=["feature1", "feature2", "feature3"]
        )

        # Should only have results for existing features
        assert len(report["feature_scores"]) == 2
