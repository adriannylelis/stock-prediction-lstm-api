export interface PredictionResponse {
  success: boolean;
  data: PredictionData;
}

export interface PredictionData {
  ticker: string;
  predicted_price: number;
  current_price: number;
  change_percent: number;
  change_direction: 'up' | 'down' | 'neutral';
  prediction_date: string;
  confidence: 'high' | 'medium' | 'low';
  timestamp: string;
}

export interface Stock {
  symbol: string;
  name: string;
}

export interface ApiError {
  error: string;
  message: string;
  status: number;
}

export interface HealthResponse {
  status: string;
  timestamp: string;
  service: string;
}

export interface ModelInfo {
  model_type: string;
  architecture: string;
  input_size: number;
  hidden_size: number;
  num_layers: number;
  dropout: number;
  lookback_days: number;
  training: {
    epochs: number;
    batch_size: number;
    learning_rate: number;
    optimizer: string;
    loss_function: string;
  };
  performance: {
    test_mae: number;
    test_rmse: number;
    test_mape: number;
    test_r2: number;
    directional_accuracy: number;
  };
}
