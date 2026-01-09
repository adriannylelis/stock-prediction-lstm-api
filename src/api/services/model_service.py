import json
import warnings
from pathlib import Path
from typing import Any, Dict

import joblib
import mlflow
import torch
import yaml
from loguru import logger

from src.ml.models.lstm import StockLSTM

# Suppress MLflow filesystem deprecation warnings
warnings.filterwarnings("ignore", category=FutureWarning, module="mlflow")


class ModelService:
    """Singleton para gerenciar modelo LSTM e scaler.
    
    Suporta:
    - MLflow models (production_model.yaml)
    - Artifacts locais (fallback)
    """

    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialized = False
        return cls._instance

    def __init__(self):
        if self._initialized:
            return

        self.project_root = Path(__file__).parent.parent.parent.parent
        self.artifacts_path = self.project_root / 'artifacts' / 'models'  # ✅ Updated
        self.production_config_path = self.project_root / 'configs' / 'production_model.yaml'

        self.model = None
        self.scaler = None  # X features scaler (19 cols)
        self.y_scaler = None  # y target scaler (1 col)
        self.config = None
        self.model_uri = None
        self.ticker_to_id = {}  # Mapping ticker → ID para multi-ticker
        self.num_tickers = 1  # Default: single-ticker

        self._load_artifacts()
        self._initialized = True

    def _load_from_mlflow(self, model_uri: str, skip_tracking_uri: bool = False) -> bool:
        """Load model from MLflow.
        
        Args:
            model_uri: MLflow model URI
            skip_tracking_uri: If True, assumes tracking URI is already set
            
        Returns:
            True if successful
        """
        try:
            logger.info(f"Loading model from MLflow: {model_uri}")

            # Set tracking URI if not already set
            if not skip_tracking_uri:
                tracking_uri = "file:data/mlflow/tracking"  # Default
                if self.production_config_path.exists():
                    try:
                        with open(self.production_config_path) as f:
                            prod_config = yaml.safe_load(f)
                        tracking_uri = prod_config.get('tracking_uri', tracking_uri)
                    except Exception as e:
                        logger.warning(f"Could not read tracking_uri from config: {e}")

                mlflow.set_tracking_uri(tracking_uri)
                logger.info(f"📍 MLflow tracking URI: {tracking_uri}")

            # Load model
            self.model = mlflow.pytorch.load_model(model_uri)
            self.model.eval()
            logger.info("✅ Model loaded from MLflow")

            # Try to load scaler from MLflow artifacts
            scaler_loaded = False
            try:
                client = mlflow.tracking.MlflowClient()

                # Extract run_id from model_uri
                run_id = None
                if "runs:/" in model_uri:
                    # Format: runs:/<run_id>/model
                    run_id = model_uri.split("/")[1]
                elif "models:/" in model_uri:
                    # Format: models:/<name>/<version_or_stage>
                    parts = model_uri.split("/")
                    model_name = parts[1]
                    version_or_stage = parts[2]

                    # Get the model version
                    if version_or_stage.isdigit():
                        version = version_or_stage
                    else:
                        # It's a stage (Production, Staging, etc.)
                        versions = client.get_latest_versions(model_name, stages=[version_or_stage])
                        if versions:
                            version = versions[0].version
                        else:
                            raise ValueError(f"No model found in stage '{version_or_stage}'")

                    # Get the run_id from the model version
                    model_version = client.get_model_version(model_name, version)
                    run_id = model_version.run_id
                    logger.info(f"📌 Extracted run_id: {run_id} from {model_uri}")

                if run_id:
                    # Download scaler artifact directly
                    try:
                        scaler_path = client.download_artifacts(run_id, "scaler.pkl")
                        self.scaler = joblib.load(scaler_path)
                        logger.info(f"✅ X features scaler loaded from MLflow: {scaler_path}")
                        scaler_loaded = True

                        # Try to load y_scaler as well
                        try:
                            y_scaler_path = client.download_artifacts(run_id, "y_scaler.pkl")
                            self.y_scaler = joblib.load(y_scaler_path)
                            logger.info(f"✅ y target scaler loaded from MLflow: {y_scaler_path}")
                        except Exception:
                            # Use first column of X scaler for backward compatibility
                            self.y_scaler = self.scaler
                            logger.info("ℹ️ Using X scaler for y (backward compatibility)")

                    except Exception as e:
                        logger.warning(f"⚠️ Scaler artifact not found: {e}")
                        # Try alternative path (preprocessing artifacts might be in subdirectory)
                        try:
                            artifacts_path = client.download_artifacts(run_id, "")
                            scaler_path = Path(artifacts_path) / "scaler.pkl"
                            if scaler_path.exists():
                                self.scaler = joblib.load(scaler_path)
                                logger.info(f"✅ Scaler loaded from alternative path: {scaler_path}")
                                scaler_loaded = True
                        except Exception as e2:
                            logger.warning(f"⚠️ Could not load scaler from alternative path: {e2}")
                else:
                    logger.warning("⚠️ Could not extract run_id from model_uri")

            except Exception as e:
                logger.warning(f"⚠️ Could not load scaler from MLflow: {e}")
                import traceback
                logger.debug(traceback.format_exc())

            # Try to load ticker mapping for multi-ticker support
            if run_id:
                try:
                    # Try to load preprocessing config which may contain ticker mapping
                    preprocessing_config_path = client.download_artifacts(run_id, "preprocessing_config.json")
                    if Path(preprocessing_config_path).exists():
                        import json
                        with open(preprocessing_config_path) as f:
                            prep_config = json.load(f)

                        # Check if it's a multi-ticker model
                        if 'ticker_to_id' in prep_config:
                            self.ticker_to_id = prep_config['ticker_to_id']
                            self.num_tickers = len(self.ticker_to_id)
                            logger.info(f"✅ Loaded multi-ticker mapping: {self.num_tickers} tickers")
                        elif 'ticker_list' in prep_config:
                            # Create mapping from list
                            ticker_list = prep_config['ticker_list']
                            self.ticker_to_id = {ticker: idx for idx, ticker in enumerate(ticker_list)}
                            self.num_tickers = len(ticker_list)
                            logger.info(f"✅ Created multi-ticker mapping: {self.num_tickers} tickers")
                        else:
                            # Single-ticker model
                            self.num_tickers = 1
                            logger.info("ℹ️ Single-ticker model detected")
                except Exception as e:
                    logger.debug(f"No ticker mapping found in MLflow: {e}")
                    self.num_tickers = 1

            # If scaler not found in MLflow, try local artifacts as fallback
            if not scaler_loaded:
                logger.warning("⚠️ Scaler not found in MLflow, trying local artifacts...")

                # Try common local paths (based on project structure)
                local_scaler_paths = [
                    Path("artifacts/models/scaler.pkl"),  # Latest trained model scaler
                    Path("artifacts/scaler.pkl"),
                    Path("models/scaler.pkl"),
                    Path("data/artifacts/scaler.pkl"),
                ]

                for local_path in local_scaler_paths:
                    if local_path.exists():
                        try:
                            self.scaler = joblib.load(local_path)
                            logger.success(f"✅ Scaler loaded from local path: {local_path}")
                            scaler_loaded = True
                            break
                        except Exception as e:
                            logger.warning(f"Could not load scaler from {local_path}: {e}")

                # If still not loaded, create unfitted scaler (will fail on prediction)
                if not scaler_loaded:
                    logger.error("❌ Scaler not found in MLflow or local artifacts!")
                    logger.error("🔴 CRITICAL: Model cannot make predictions without fitted scaler")
                    logger.info("💡 Solution: Retrain model with: python -m cli.main train --ticker PETR4.SA --epochs 20")

                    from sklearn.preprocessing import StandardScaler
                    self.scaler = StandardScaler()
                    logger.warning("⚠️ Using unfitted scaler - predictions WILL FAIL")

            # Set config (basic info for is_ready check)
            self.config = {
                "model_uri": model_uri,
                "loaded_from": "mlflow",
                "has_scaler": scaler_loaded
            }

            self.model_uri = model_uri
            return True

        except Exception as e:
            logger.error(f"❌ Failed to load from MLflow: {e}")
            return False

    def _load_from_local_artifacts(self) -> bool:
        """Load model from local artifacts (fallback).
        
        Returns:
            True if successful
        """
        try:
            logger.info("Loading model from local artifacts (fallback)...")

            config_path = self.artifacts_path / 'model_config.json'
            with open(config_path) as f:
                self.config = json.load(f)
            logger.info(f"Configuração carregada: {self.config['architecture']}")

            self.model = StockLSTM(
                input_size=self.config['input_size'],
                hidden_size=self.config['hidden_size'],
                num_layers=self.config['num_layers'],
                dropout=self.config['dropout']
            )

            model_path = self.artifacts_path / 'model_lstm_1x16.pt'
            state_dict = torch.load(model_path, map_location=torch.device('cpu'), weights_only=False)
            self.model.load_state_dict(state_dict)
            self.model.eval()
            logger.info(f"✅ Modelo carregado: {model_path}")

            scaler_path = self.artifacts_path / 'scaler_corrected.pkl'
            self.scaler = joblib.load(scaler_path)
            logger.info(f"✅ Scaler carregado: {scaler_path}")

            return True

        except Exception as e:
            logger.error(f"❌ Failed to load local artifacts: {e}")
            return False

    def _load_artifacts(self):
        """Load model artifacts from MLflow using stage-based approach.
        
        Priority:
        1. Production stage from MLflow
        2. Staging stage from MLflow
        3. Latest version from MLflow
        4. Local artifacts (fallback)
        """
        try:
            # Configure MLflow tracking URI
            tracking_uri = "file:data/mlflow/tracking"  # Default
            if self.production_config_path.exists():
                try:
                    with open(self.production_config_path) as f:
                        prod_config = yaml.safe_load(f)
                    tracking_uri = prod_config.get('tracking_uri', tracking_uri)
                except Exception as e:
                    logger.warning(f"Could not read tracking_uri from config: {e}")

            mlflow.set_tracking_uri(tracking_uri)
            logger.info(f"📍 MLflow tracking URI: {tracking_uri}")

            # Try to load from MLflow using stage-based approach
            model_name = "stock-lstm-model"
            loaded = False

            # Priority 1: Production stage
            try:
                model_uri = f"models:/{model_name}/Production"
                logger.info(f"🎯 Attempting to load Production model: {model_uri}")
                if self._load_from_mlflow(model_uri, skip_tracking_uri=True):
                    logger.success("✅ Loaded model from Production stage")
                    self._update_production_config(model_uri, "Production")
                    loaded = True
            except Exception as e:
                logger.info(f"No model in Production stage: {e}")

            # Priority 2: Staging stage
            if not loaded:
                try:
                    model_uri = f"models:/{model_name}/Staging"
                    logger.info(f"🎯 Attempting to load Staging model: {model_uri}")
                    if self._load_from_mlflow(model_uri, skip_tracking_uri=True):
                        logger.warning("⚠️ Using Staging model (no Production model available)")
                        self._update_production_config(model_uri, "Staging")
                        loaded = True
                except Exception as e:
                    logger.info(f"No model in Staging stage: {e}")

            # Priority 3: Latest version (any stage)
            if not loaded:
                try:
                    client = mlflow.tracking.MlflowClient()
                    versions = client.search_model_versions(f'name="{model_name}"')
                    if versions:
                        latest_version = versions[0].version
                        model_uri = f"models:/{model_name}/{latest_version}"
                        logger.info(f"🎯 Attempting to load latest version: {model_uri}")
                        if self._load_from_mlflow(model_uri, skip_tracking_uri=True):
                            logger.warning(f"⚠️ Using latest version v{latest_version} (no staged model available)")
                            self._update_production_config(model_uri, f"None (v{latest_version})")
                            loaded = True
                except Exception as e:
                    logger.error(f"Failed to load latest version: {e}")

            if loaded:
                return

            # Priority 4: Fallback to local artifacts
            logger.warning("⚠️ MLflow models not available, trying local artifacts...")
            if self._load_from_local_artifacts():
                logger.info("✅ Loaded from local artifacts")
                return

            # If nothing worked, raise clear error
            raise RuntimeError(
                f"Modelo '{model_name}' não encontrado no MLflow. "
                f"Por favor, treine um modelo usando: python -m cli.main train --ticker PETR4.SA --epochs 20"
            )

        except FileNotFoundError as e:
            logger.error(f"Arquivo não encontrado: {str(e)}")
            raise RuntimeError(f"Artefato necessário não encontrado: {str(e)}")
        except Exception as e:
            logger.error(f"Erro ao carregar artefatos: {str(e)}")
            raise RuntimeError(f"Falha ao inicializar modelo: {str(e)}")

    def _update_production_config(self, model_uri: str, stage: str):
        """Update production_model.yaml automatically after loading model.
        
        Args:
            model_uri: MLflow model URI that was loaded
            stage: Stage of the loaded model
        """
        try:
            from datetime import datetime

            config = {
                'deployed_at': datetime.now().isoformat(),
                'deployed_by': 'api_auto_load',
                'model_uri': model_uri,
                'stage': stage,
                'reason': f'Auto-loaded from MLflow {stage} stage',
                'rollback': False,
                'rollback_from': None,
                'tracking_uri': str(mlflow.get_tracking_uri()),
                'version': model_uri.split('/')[-1] if '/' in model_uri else 'unknown'
            }

            self.production_config_path.parent.mkdir(parents=True, exist_ok=True)
            with open(self.production_config_path, 'w') as f:
                yaml.dump(config, f, default_flow_style=False)

            logger.info(f"📝 Updated production_model.yaml: {model_uri} ({stage})")

        except Exception as e:
            logger.warning(f"Could not update production_model.yaml: {e}")

    def reload(self) -> bool:
        """Reload model from production config.
        
        Returns:
            True if successful
        """
        try:
            logger.info("🔄 Reloading model...")
            self._initialized = False
            self._load_artifacts()
            self._initialized = True
            logger.success("✅ Model reloaded successfully")
            return True
        except Exception as e:
            logger.error(f"❌ Reload failed: {e}")
            return False

    def get_model(self) -> torch.nn.Module:
        if self.model is None:
            raise RuntimeError("Modelo não foi carregado corretamente")
        return self.model

    def get_scaler(self):
        """Get X features scaler (19 columns)."""
        if self.scaler is None:
            raise RuntimeError("Scaler não foi carregado corretamente")
        return self.scaler

    def get_y_scaler(self):
        """Get y target scaler (1 column) for denormalization."""
        if self.y_scaler is None:
            # Fallback to X scaler for backward compatibility
            logger.warning("y_scaler not found, using X scaler")
            return self.scaler
        return self.y_scaler

    def get_config(self) -> Dict[str, Any]:
        if self.config is None:
            raise RuntimeError("Configuração não foi carregada corretamente")
        return self.config

    def is_ready(self) -> bool:
        return (self.model is not None and
                self.scaler is not None and
                self.config is not None)

    def get_ticker_id(self, ticker: str) -> int:
        """Obtém o ID do ticker para modelos multi-ticker.
        
        Args:
            ticker: Símbolo do ticker (e.g., 'PETR4.SA')
            
        Returns:
            ID do ticker (0 se modelo single-ticker ou ticker não encontrado)
        """
        if not self.ticker_to_id:
            return 0  # Single-ticker fallback

        # Busca exata
        if ticker in self.ticker_to_id:
            return self.ticker_to_id[ticker]

        # Tenta sem .SA
        ticker_base = ticker.replace('.SA', '')
        for key in self.ticker_to_id:
            if key.replace('.SA', '') == ticker_base:
                return self.ticker_to_id[key]

        logger.warning(f"⚠️ Ticker '{ticker}' não encontrado no mapeamento. Usando ID=0")
        return 0

    def is_multi_ticker(self) -> bool:
        """Verifica se o modelo suporta múltiplos tickers."""
        return len(self.ticker_to_id) > 1

    def predict(self, input_data, ticker_id: int = 0):
        """Make prediction using loaded model.
        
        Args:
            input_data: Input tensor or numpy array (batch, seq_len, features)
            ticker_id: Ticker ID for embedding-based models (default: 0)
            
        Returns:
            numpy array of predictions
        """
        if not self.is_ready():
            raise RuntimeError("ModelService not ready - call load first")

        import numpy as np
        import torch

        # Convert to tensor if needed
        if isinstance(input_data, np.ndarray):
            input_tensor = torch.FloatTensor(input_data)
        else:
            input_tensor = input_data

        # Ensure batch dimension
        if input_tensor.dim() == 2:
            input_tensor = input_tensor.unsqueeze(0)

        # Move to same device as model
        device = next(self.model.parameters()).device
        input_tensor = input_tensor.to(device)

        # Make prediction
        self.model.eval()
        with torch.no_grad():
            # Check if model uses embeddings
            if hasattr(self.model, 'ticker_embedding'):
                ticker_ids = torch.tensor([ticker_id] * input_tensor.shape[0]).to(device)
                output = self.model(input_tensor, ticker_ids)
            else:
                output = self.model(input_tensor)

        # Convert to numpy
        predictions = output.cpu().numpy().flatten()

        return predictions
