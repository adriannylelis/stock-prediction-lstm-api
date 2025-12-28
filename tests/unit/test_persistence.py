"""Unit tests for persistence module."""

import shutil
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.ml.utils.persistence import ArtifactManager, DataVersionManager


@pytest.fixture
def sample_dataframe():
    """Create sample DataFrame for testing."""
    dates = pd.date_range(start="2023-01-01", periods=100, freq="D")
    return pd.DataFrame(
        {
            "Close": np.random.random(100),
            "Volume": np.random.randint(1000, 10000, 100),
            "RSI": np.random.random(100) * 100,
        },
        index=dates,
    )


@pytest.fixture
def temp_version_dir(tmp_path):
    """Create temporary versioning directory."""
    version_dir = tmp_path / "versioned_test"
    version_dir.mkdir()
    yield str(version_dir)
    if version_dir.exists():
        shutil.rmtree(version_dir)


@pytest.fixture
def temp_artifact_dir(tmp_path):
    """Create temporary artifact directory."""
    artifact_dir = tmp_path / "artifacts_test"
    artifact_dir.mkdir()
    yield str(artifact_dir)
    if artifact_dir.exists():
        shutil.rmtree(artifact_dir)


class TestDataVersionManager:
    """Test DataVersionManager functionality."""

    def test_init(self, temp_version_dir):
        """Test initialization."""
        manager = DataVersionManager(base_path=temp_version_dir)
        assert manager.base_path == Path(temp_version_dir)
        assert manager.auto_cleanup is False
        assert manager.max_versions == 10

    def test_init_with_cleanup(self, temp_version_dir):
        """Test initialization with cleanup enabled."""
        manager = DataVersionManager(base_path=temp_version_dir, auto_cleanup=True, max_versions=5)
        assert manager.auto_cleanup is True
        assert manager.max_versions == 5

    def test_save_version(self, temp_version_dir, sample_dataframe):
        """Test saving a data version."""
        manager = DataVersionManager(base_path=temp_version_dir)

        version = manager.save(sample_dataframe, ticker="TEST.SA", metadata={"test": "data"})

        # Check version format (timestamp with milliseconds)
        assert len(version) == 19  # YYYYMMDD_HHMMSS_mmm
        assert version[8] == "_"
        assert version[15] == "_"  # Second underscore for milliseconds

        # Check files were created
        ticker_dir = Path(temp_version_dir) / "TEST" / version
        assert (ticker_dir / "data.parquet").exists()
        assert (ticker_dir / "metadata.json").exists()

    def test_load_version(self, temp_version_dir, sample_dataframe):
        """Test loading a specific version."""
        manager = DataVersionManager(base_path=temp_version_dir)

        # Save
        version = manager.save(sample_dataframe, "LOAD.SA")

        # Load
        loaded_df = manager.load(version, "LOAD.SA")

        # Verify
        assert len(loaded_df) == len(sample_dataframe)
        assert list(loaded_df.columns) == list(sample_dataframe.columns)
        # Use check_names=False and check_freq=False as index properties might differ
        pd.testing.assert_frame_equal(
            loaded_df, sample_dataframe, check_names=False, check_freq=False
        )

    def test_list_versions(self, temp_version_dir, sample_dataframe):
        """Test listing all versions."""
        manager = DataVersionManager(base_path=temp_version_dir)

        # Save multiple versions
        for i in range(3):
            manager.save(sample_dataframe, "LIST.SA", metadata={"iteration": i})

        # List
        versions = manager.list_versions("LIST.SA")

        # Verify
        assert len(versions) == 3
        assert all("version" in v for v in versions)
        assert all("timestamp" in v for v in versions)

    def test_get_latest_version(self, temp_version_dir, sample_dataframe):
        """Test getting latest version."""
        manager = DataVersionManager(base_path=temp_version_dir)

        # Save versions
        manager.save(sample_dataframe, "LATEST.SA")
        v2 = manager.save(sample_dataframe, "LATEST.SA")

        # Get latest
        latest = manager.get_latest_version("LATEST.SA")

        # Should be v2
        assert latest == v2

    def test_load_latest(self, temp_version_dir, sample_dataframe):
        """Test loading latest version."""
        manager = DataVersionManager(base_path=temp_version_dir)

        # Save versions
        manager.save(sample_dataframe, "LOAD_LATEST.SA")
        df_modified = sample_dataframe.copy()
        df_modified["Close"] = df_modified["Close"] * 2
        manager.save(df_modified, "LOAD_LATEST.SA")

        # Load latest
        loaded = manager.load_latest("LOAD_LATEST.SA")

        # Should match modified version (check_names and check_freq=False for index properties)
        pd.testing.assert_frame_equal(loaded, df_modified, check_names=False, check_freq=False)

    def test_auto_cleanup(self, temp_version_dir, sample_dataframe):
        """Test automatic cleanup of old versions."""
        manager = DataVersionManager(base_path=temp_version_dir, auto_cleanup=True, max_versions=3)

        # Save 5 versions
        for i in range(5):
            manager.save(sample_dataframe, "CLEANUP.SA", metadata={"i": i})

        # Should keep only 3
        versions = manager.list_versions("CLEANUP.SA")
        assert len(versions) <= 3

    def test_manual_cleanup(self, temp_version_dir, sample_dataframe):
        """Test manual cleanup."""
        manager = DataVersionManager(base_path=temp_version_dir)

        # Save 5 versions
        for i in range(5):
            manager.save(sample_dataframe, "MANUAL.SA")

        # Manual cleanup
        manager.cleanup_all("MANUAL.SA", keep_latest=2)

        # Should have 2 left
        versions = manager.list_versions("MANUAL.SA")
        assert len(versions) == 2

    def test_get_metadata(self, temp_version_dir, sample_dataframe):
        """Test getting version metadata."""
        manager = DataVersionManager(base_path=temp_version_dir)

        custom_meta = {"source": "test", "quality": "high"}
        version = manager.save(sample_dataframe, "META.SA", metadata=custom_meta)

        # Get metadata
        metadata = manager.get_metadata(version, "META.SA")

        # Verify
        assert metadata["custom"] == custom_meta
        assert metadata["ticker"] == "META.SA"
        assert metadata["n_records"] == 100


class TestArtifactManager:
    """Test ArtifactManager functionality."""

    def test_init(self, temp_artifact_dir):
        """Test initialization."""
        manager = ArtifactManager(base_path=temp_artifact_dir, use_mlflow=False)
        assert manager.base_path == Path(temp_artifact_dir)
        assert manager.use_mlflow is False

    def test_save_load_config(self, temp_artifact_dir):
        """Test config save and load."""
        manager = ArtifactManager(base_path=temp_artifact_dir, use_mlflow=False)

        config = {"model": "LSTM", "params": {"hidden": 50, "lr": 0.001}}

        # Save
        path = manager.save_config(config, "test_config")
        assert path.exists()

        # Load
        loaded = manager.load_config("test_config")
        assert loaded == config

    def test_save_load_scaler(self, temp_artifact_dir):
        """Test scaler save and load."""
        from sklearn.preprocessing import MinMaxScaler

        manager = ArtifactManager(base_path=temp_artifact_dir, use_mlflow=False)

        # Create and fit scaler
        scaler = MinMaxScaler()
        data = np.array([[1], [2], [3], [4], [5]])
        scaler.fit(data)

        # Save
        path = manager.save_scaler(scaler, "test_scaler")
        assert path.exists()

        # Load
        loaded_scaler = manager.load_scaler("test_scaler")

        # Test transform
        result = loaded_scaler.transform([[3]])
        expected = scaler.transform([[3]])
        np.testing.assert_array_almost_equal(result, expected)

    def test_save_scaler_with_metadata(self, temp_artifact_dir):
        """Test scaler save with metadata."""
        from sklearn.preprocessing import StandardScaler

        manager = ArtifactManager(base_path=temp_artifact_dir, use_mlflow=False)

        scaler = StandardScaler()
        scaler.fit([[1], [2], [3]])

        metadata = {"ticker": "TEST.SA", "date": "2024-01-01"}
        manager.save_scaler(scaler, "scaler_with_meta", metadata=metadata)

        # Check metadata file exists
        meta_path = Path(temp_artifact_dir) / "scalers" / "scaler_with_meta_metadata.json"
        assert meta_path.exists()
