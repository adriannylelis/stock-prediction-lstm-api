"""Additional unit tests for monitoring and persistence modules."""

import json

import numpy as np
import pandas as pd
import pytest

from src.ml.monitoring.drift_detector import DriftDetector
from src.ml.utils.persistence import ArtifactManager, DataVersionManager


class TestDriftDetector:
    """Tests for DriftDetector class."""

    @pytest.fixture
    def detector(self):
        """Create DriftDetector instance."""
        return DriftDetector(ks_threshold=0.05, psi_threshold=0.1)

    @pytest.fixture
    def reference_data(self):
        """Create reference dataset."""
        np.random.seed(42)
        return pd.DataFrame(
            {
                "feature1": np.random.normal(0, 1, 1000),
                "feature2": np.random.normal(10, 2, 1000),
                "feature3": np.random.uniform(0, 100, 1000),
            }
        )

    def test_no_drift_detected(self, detector, reference_data):
        """Test when there is no drift."""
        # Production data same as reference
        prod_data = reference_data.copy()

        report = detector.detect_drift(reference_data, prod_data)

        assert not report["has_drift"]
        assert len(report["drifted_features"]) == 0

    def test_drift_detected(self, detector, reference_data):
        """Test when drift is present."""
        # Production data with shifted distribution
        np.random.seed(100)
        prod_data = pd.DataFrame(
            {
                "feature1": np.random.normal(2, 1, 1000),  # Shifted mean
                "feature2": np.random.normal(10, 2, 1000),  # Same
                "feature3": np.random.uniform(0, 100, 1000),  # Same
            }
        )

        report = detector.detect_drift(reference_data, prod_data)

        assert report["has_drift"]
        assert "feature1" in report["drifted_features"]
        assert "feature2" not in report["drifted_features"]

    def test_psi_calculation(self, detector):
        """Test PSI calculation."""
        ref_series = pd.Series(np.random.normal(0, 1, 1000))
        prod_series = pd.Series(np.random.normal(0, 1, 1000))

        psi = detector.calculate_psi(ref_series, prod_series)

        assert isinstance(psi, float)
        assert psi >= 0
        assert psi < 0.1  # No significant change

    def test_psi_with_drift(self, detector):
        """Test PSI with drift."""
        ref_series = pd.Series(np.random.normal(0, 1, 1000))
        prod_series = pd.Series(np.random.normal(2, 1, 1000))  # Shifted

        psi = detector.calculate_psi(ref_series, prod_series)

        assert psi >= 0.2  # Significant change

    def test_save_report(self, detector, reference_data, tmp_path):
        """Test saving drift report."""
        prod_data = reference_data.copy()
        report = detector.detect_drift(reference_data, prod_data)

        output_path = tmp_path / "drift_report.json"
        detector.save_report(report, str(output_path))

        assert output_path.exists()

        with open(output_path) as f:
            loaded_report = json.load(f)

        assert loaded_report["has_drift"] == report["has_drift"]

    def test_compare_versions(self, detector, reference_data):
        """Test version comparison."""
        v1_data = reference_data
        v2_data = reference_data.copy()

        report = detector.compare_versions(v1_data, v2_data, method="ks")

        assert not report["has_drift"]


class TestDataVersionManager:
    """Tests for DataVersionManager class."""

    @pytest.fixture
    def manager(self, tmp_path):
        """Create DataVersionManager instance."""
        return DataVersionManager(base_path=str(tmp_path / "versioned"), auto_cleanup=False)

    @pytest.fixture
    def sample_data(self):
        """Create sample DataFrame."""
        return pd.DataFrame(
            {"Close": np.random.rand(100), "Volume": np.random.rand(100) * 1000},
            index=pd.date_range("2023-01-01", periods=100),
        )

    def test_save_and_load(self, manager, sample_data):
        """Test save and load cycle."""
        version = manager.save(sample_data, "TEST.SA")

        loaded_data = manager.load(version, "TEST.SA")

        # Use check_names=False and check_freq=False as index properties might differ
        pd.testing.assert_frame_equal(sample_data, loaded_data, check_names=False, check_freq=False)

    def test_metadata_tracking(self, manager, sample_data):
        """Test metadata is correctly saved."""
        version = manager.save(sample_data, "TEST.SA", metadata={"source": "test", "version": 1})

        metadata = manager.get_metadata(version, "TEST.SA")

        assert metadata["ticker"] == "TEST.SA"
        assert metadata["n_records"] == 100
        assert metadata["n_features"] == 2
        assert metadata["custom"]["source"] == "test"

    def test_list_versions(self, manager, sample_data):
        """Test listing versions."""
        # Save multiple versions
        versions = []
        for i in range(3):
            v = manager.save(sample_data, "TEST.SA", metadata={"iter": i})
            versions.append(v)

        all_versions = manager.list_versions("TEST.SA")

        assert len(all_versions) == 3

    def test_get_latest_version(self, manager, sample_data):
        """Test getting latest version."""
        manager.save(sample_data, "TEST.SA")
        import time

        time.sleep(0.1)
        v2 = manager.save(sample_data, "TEST.SA")

        latest = manager.get_latest_version("TEST.SA")

        assert latest == v2

    def test_auto_cleanup(self, tmp_path, sample_data):
        """Test auto cleanup."""
        manager = DataVersionManager(
            base_path=str(tmp_path / "versioned"), auto_cleanup=True, max_versions=2
        )

        # Save 4 versions
        for i in range(4):
            manager.save(sample_data, "TEST.SA")
            import time

            time.sleep(0.1)

        versions = manager.list_versions("TEST.SA")

        assert len(versions) <= 2

    def test_manual_cleanup(self, manager, sample_data):
        """Test manual cleanup."""
        # Save 5 versions
        for i in range(5):
            manager.save(sample_data, "TEST.SA")

        manager.cleanup_all("TEST.SA", keep_latest=2)

        versions = manager.list_versions("TEST.SA")
        assert len(versions) == 2


class TestArtifactManager:
    """Tests for ArtifactManager class."""

    @pytest.fixture
    def manager(self, tmp_path):
        """Create ArtifactManager instance."""
        return ArtifactManager(base_path=str(tmp_path / "artifacts"), use_mlflow=False)

    def test_save_and_load_scaler(self, manager):
        """Test scaler save/load."""
        from sklearn.preprocessing import StandardScaler

        scaler = StandardScaler()
        scaler.fit([[1, 2], [3, 4], [5, 6]])

        manager.save_scaler(scaler, "test_scaler")
        loaded_scaler = manager.load_scaler("test_scaler")

        # Test scaler works
        original_transform = scaler.transform([[2, 3]])
        loaded_transform = loaded_scaler.transform([[2, 3]])

        np.testing.assert_array_almost_equal(original_transform, loaded_transform)

    def test_save_and_load_config(self, manager):
        """Test config save/load."""
        config = {
            "model": "lstm",
            "hidden_size": 50,
            "learning_rate": 0.001,
            "nested": {"key": "value"},
        }

        manager.save_config(config, "test_config")
        loaded_config = manager.load_config("test_config")

        assert loaded_config == config

    def test_scaler_with_metadata(self, manager):
        """Test scaler with metadata."""
        from sklearn.preprocessing import MinMaxScaler

        scaler = MinMaxScaler()
        scaler.fit([[0], [1], [2]])

        metadata = {"ticker": "PETR4.SA", "date": "2025-12-28", "version": "v1"}

        manager.save_scaler(scaler, "scaler_with_meta", metadata=metadata)

        # Check metadata file exists
        meta_path = manager.scalers_path / "scaler_with_meta_metadata.json"
        assert meta_path.exists()

        with open(meta_path) as f:
            loaded_meta = json.load(f)

        assert loaded_meta == metadata


@pytest.mark.unit
def test_drift_detector_initialization():
    """Test DriftDetector initialization."""
    detector = DriftDetector(ks_threshold=0.01, psi_threshold=0.2)

    assert detector.ks_threshold == 0.01
    assert detector.psi_threshold == 0.2


@pytest.mark.unit
def test_data_version_manager_initialization(tmp_path):
    """Test DataVersionManager initialization."""
    manager = DataVersionManager(
        base_path=str(tmp_path), auto_cleanup=True, max_versions=5, compress_after_days=7
    )

    assert manager.auto_cleanup
    assert manager.max_versions == 5
    assert manager.compress_after_days == 7
    assert manager.base_path.exists()
