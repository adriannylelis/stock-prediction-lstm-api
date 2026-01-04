export interface HealthResponse {
  status: string;
  timestamp: string;
  service: string;
}

export interface ModelInfo {
  architecture: string;
  input_size: number;
  hidden_size: number;
  num_layers: number;
  dropout: number;
  lookback: number;
  features: string[];
  metrics: {
    mape: number;
    mae: number;
    rmse: number;
    r2: number;
  };
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

export interface PredictionResponse {
  success: boolean;
  data: PredictionData;
}

export interface ApiError {
  error: string;
  message: string;
  status: number;
  details?: Record<string, any>;
}
