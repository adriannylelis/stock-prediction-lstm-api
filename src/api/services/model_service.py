"""
Serviço para carregar e gerenciar o modelo LSTM.

Responsável por carregar o modelo treinado e o scaler do disco.
"""

import torch
import joblib
import json
from pathlib import Path
from typing import Dict, Any
import logging

from src.api.models.lstm_model import StockLSTM

logger = logging.getLogger(__name__)


class ModelService:
    """
    Serviço singleton para gerenciar modelo e scaler.
    
    Carrega modelo PyTorch e scaler sklearn na inicialização
    e mantém em memória para reutilização.
    """
    
    _instance = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialized = False
        return cls._instance
    
    def __init__(self):
        """Inicializa serviço (apenas uma vez)."""
        if self._initialized:
            return
            
        self.artifacts_path = Path(__file__).parent.parent.parent.parent / 'artifacts'
        self.model = None
        self.scaler = None
        self.config = None
        
        self._load_artifacts()
        self._initialized = True
    
    def _load_artifacts(self):
        """Carrega modelo, scaler e configuração do disco."""
        try:
            # Carregar configuração
            config_path = self.artifacts_path / 'model_config.json'
            with open(config_path, 'r') as f:
                self.config = json.load(f)
            logger.info(f"Configuração carregada: {self.config['architecture']}")
            
            # Criar modelo LSTM com arquitetura do config
            self.model = StockLSTM(
                input_size=self.config['input_size'],
                hidden_size=self.config['hidden_size'],
                num_layers=self.config['num_layers'],
                dropout=self.config['dropout']
            )
            
            # Carregar pesos do modelo
            model_path = self.artifacts_path / 'model_lstm_1x16.pt'
            state_dict = torch.load(model_path, map_location=torch.device('cpu'), weights_only=False)
            self.model.load_state_dict(state_dict)
            self.model.eval()  # Modo de inferência
            logger.info(f"Modelo carregado: {model_path}")
            
            # Carregar scaler
            scaler_path = self.artifacts_path / 'scaler_corrected.pkl'
            self.scaler = joblib.load(scaler_path)
            logger.info(f"Scaler carregado: {scaler_path}")
            
            # Carregar scaler
            scaler_path = self.artifacts_path / 'scaler_corrected.pkl'
            self.scaler = joblib.load(scaler_path)
            logger.info(f"Scaler carregado: {scaler_path}")
            
        except FileNotFoundError as e:
            logger.error(f"Arquivo não encontrado: {str(e)}")
            raise RuntimeError(f"Artefato necessário não encontrado: {str(e)}")
        except Exception as e:
            logger.error(f"Erro ao carregar artefatos: {str(e)}")
            raise RuntimeError(f"Falha ao inicializar modelo: {str(e)}")
    
    def get_model(self) -> torch.nn.Module:
        """
        Retorna modelo carregado.
        
        Returns:
            torch.nn.Module: Modelo LSTM em modo eval.
        """
        if self.model is None:
            raise RuntimeError("Modelo não foi carregado corretamente")
        return self.model
    
    def get_scaler(self):
        """
        Retorna scaler carregado.
        
        Returns:
            MinMaxScaler: Scaler sklearn para normalização.
        """
        if self.scaler is None:
            raise RuntimeError("Scaler não foi carregado corretamente")
        return self.scaler
    
    def get_config(self) -> Dict[str, Any]:
        """
        Retorna configuração do modelo.
        
        Returns:
            dict: Configuração e metadados do modelo.
        """
        if self.config is None:
            raise RuntimeError("Configuração não foi carregada corretamente")
        return self.config
    
    def is_ready(self) -> bool:
        """
        Verifica se o serviço está pronto para uso.
        
        Returns:
            bool: True se modelo e scaler estão carregados.
        """
        return (self.model is not None and 
                self.scaler is not None and 
                self.config is not None)
